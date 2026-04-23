from __future__ import annotations

import asyncio
from collections import OrderedDict
from io import BytesIO
from uuid import uuid4

from app.schemas.admin import (
    RAGLabDocument,
    RAGLabQuestionResult,
    RAGLabRequest,
    RAGLabSessionResponse,
    RAGLabSourcePreview,
    RAGLabVariant,
    RAGLabVariantResult,
)
from app.schemas.chat import ChatMessage, IntentDecision, SourceChunk
from app.services.embedding_service import CrossEncoderProvider, SentenceTransformerProvider
from app.services.generator_service import GenerationRequest, QueuedGenerationService


class RAGLabService:
    def __init__(
        self,
        embedder_provider: SentenceTransformerProvider | None,
        reranker_provider: CrossEncoderProvider | None,
        generation_service: QueuedGenerationService,
        max_sessions: int = 12,
    ) -> None:
        self._embedder_provider = embedder_provider
        self._reranker_provider = reranker_provider
        self._generation_service = generation_service
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, RAGLabSessionResponse] = OrderedDict()
        self._lock = asyncio.Lock()

    async def run(self, request: RAGLabRequest) -> RAGLabSessionResponse:
        if self._embedder_provider is None:
            raise RuntimeError("Embedding model is not available for RAG lab evaluation.")

        session_id = uuid4().hex
        variant_results: list[RAGLabVariantResult] = []
        for variant in request.variants:
            variant_results.append(
                await self._evaluate_variant(
                    variant=variant,
                    questions=request.questions,
                    documents=request.documents,
                )
            )

        session = RAGLabSessionResponse(
            session_id=session_id,
            document_count=len(request.documents),
            question_count=len(request.questions),
            variants=variant_results,
        )
        async with self._lock:
            self._sessions[session_id] = session
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        return session

    async def get(self, session_id: str) -> RAGLabSessionResponse | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_variant(self, session_id: str, variant_id: str) -> RAGLabVariantResult | None:
        session = await self.get(session_id)
        if session is None:
            return None
        for variant in session.variants:
            if variant.variant_id == variant_id:
                return variant
        return None

    async def export_excel(self, session_id: str) -> bytes:
        session = await self.get(session_id)
        if session is None:
            raise KeyError(session_id)

        from openpyxl import Workbook

        workbook = Workbook()
        summary_sheet = workbook.active
        summary_sheet.title = "Summary"
        summary_sheet.append(
            ["Session ID", session.session_id, "Documents", session.document_count, "Questions", session.question_count]
        )
        summary_sheet.append([])
        summary_sheet.append(
            ["Variant", "Question", "Answer Preview", "Source Titles", "Chunk Size", "Overlap Ratio", "Retrieval K", "Rerank K", "Temperature"]
        )
        for variant in session.variants:
            for question in variant.questions:
                summary_sheet.append(
                    [
                        variant.name,
                        question.question,
                        question.answer_preview,
                        " | ".join(source.title for source in question.sources),
                        variant.chunk_size,
                        variant.chunk_overlap_ratio,
                        variant.retrieval_k,
                        variant.rerank_k,
                        variant.temperature,
                    ]
                )

        for variant in session.variants:
            sheet = workbook.create_sheet(title=_safe_sheet_title(variant.name))
            sheet.append(
                ["Question", "Answer Full", "Source Title", "Source Score", "Source Preview", "Source Full Content"]
            )
            for question in variant.questions:
                if not question.sources:
                    sheet.append([question.question, question.answer_full, "", "", "", ""])
                    continue
                for source in question.sources:
                    sheet.append(
                        [
                            question.question,
                            question.answer_full,
                            source.title,
                            source.score,
                            source.preview,
                            source.content_full,
                        ]
                    )

        stream = BytesIO()
        workbook.save(stream)
        return stream.getvalue()

    async def export_word(self, session_id: str) -> bytes:
        session = await self.get(session_id)
        if session is None:
            raise KeyError(session_id)

        from docx import Document

        document = Document()
        document.add_heading("RAG Lab Evaluation Report", level=1)
        document.add_paragraph(f"Session ID: {session.session_id}")
        document.add_paragraph(
            f"Documents: {session.document_count} | Questions: {session.question_count}"
        )

        for variant in session.variants:
            document.add_heading(variant.name, level=2)
            document.add_paragraph(
                f"chunk_size={variant.chunk_size}, overlap_ratio={variant.chunk_overlap_ratio}, "
                f"retrieval_k={variant.retrieval_k}, rerank_k={variant.rerank_k}, temperature={variant.temperature}"
            )
            for index, question in enumerate(variant.questions, start=1):
                document.add_heading(f"Q{index}. {question.question}", level=3)
                document.add_paragraph(question.answer_full or "[No answer]")
                if not question.sources:
                    document.add_paragraph("No sources returned.")
                    continue
                for source_index, source in enumerate(question.sources, start=1):
                    document.add_paragraph(
                        f"[{source_index}] {source.title} | score={source.score:.4f}",
                        style="List Bullet",
                    )
                    document.add_paragraph(source.content_full)

        stream = BytesIO()
        document.save(stream)
        return stream.getvalue()

    async def _evaluate_variant(
        self,
        *,
        variant: RAGLabVariant,
        questions: list[str],
        documents: list[RAGLabDocument],
    ) -> RAGLabVariantResult:
        chunk_overlap = int(round(variant.chunk_size * variant.chunk_overlap_ratio))
        chunk_records = self._build_chunk_records(
            documents=documents,
            chunk_size=variant.chunk_size,
            chunk_overlap=chunk_overlap,
        )
        embeddings = await self._encode_texts([record["content"] for record in chunk_records])
        for record, embedding in zip(chunk_records, embeddings):
            record["embedding"] = embedding

        question_results: list[RAGLabQuestionResult] = []
        for question in questions:
            sources = await self._retrieve_sources(
                question=question,
                chunk_records=chunk_records,
                retrieval_k=variant.retrieval_k,
                rerank_k=variant.rerank_k,
            )
            answer = await self._generate_answer(
                question=question,
                sources=sources,
                temperature=variant.temperature,
            )
            question_results.append(
                RAGLabQuestionResult(
                    question=question,
                    answer_preview=_shorten(answer, 160),
                    answer_full=answer,
                    sources=[
                        RAGLabSourcePreview(
                            document_id=source.document_id,
                            title=source.title,
                            preview=_shorten(source.content, 180),
                            content_full=source.content,
                            score=source.score,
                        )
                        for source in sources
                    ],
                )
            )

        return RAGLabVariantResult(
            variant_id=variant.variant_id,
            name=variant.name,
            chunk_size=variant.chunk_size,
            chunk_overlap_ratio=variant.chunk_overlap_ratio,
            retrieval_k=variant.retrieval_k,
            rerank_k=variant.rerank_k,
            temperature=variant.temperature,
            questions=question_results,
        )

    def _build_chunk_records(
        self,
        *,
        documents: list[RAGLabDocument],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        chunk_records: list[dict] = []
        for document_index, document in enumerate(documents, start=1):
            chunks = _chunk_text(document.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                chunks = [document.content.strip()]
            for chunk_index, chunk in enumerate(chunks, start=1):
                chunk_records.append(
                    {
                        "document_id": f"lab-{document_index}",
                        "title": document.title,
                        "content": chunk,
                        "chunk_id": f"lab-{document_index}:{chunk_index}",
                    }
                )
        return chunk_records

    async def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        embedder = self._embedder_provider.get_model()

        def _encode():
            vectors = embedder.encode(
                texts,
                batch_size=8,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vectors.tolist()

        return await asyncio.to_thread(_encode)

    async def _retrieve_sources(
        self,
        *,
        question: str,
        chunk_records: list[dict],
        retrieval_k: int,
        rerank_k: int,
    ) -> list[SourceChunk]:
        query_vector = (await self._encode_texts([question]))[0]
        scored = sorted(
            (
                (
                    _dot(query_vector, record["embedding"]),
                    record,
                )
                for record in chunk_records
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:retrieval_k]

        chunks = [
            SourceChunk(
                document_id=record["document_id"],
                title=record["title"],
                content=record["content"],
                score=float(score),
                metadata={"chunk_id": record["chunk_id"]},
            )
            for score, record in scored
        ]

        if rerank_k > 0 and self._reranker_provider is not None and chunks:
            reranker = self._reranker_provider.get_model()
            pairs = [(question, f"{chunk.title}\n{chunk.content}") for chunk in chunks]
            scores = await asyncio.to_thread(reranker.predict, pairs)
            reranked = sorted(
                zip(chunks, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            chunks = [
                SourceChunk(
                    document_id=chunk.document_id,
                    title=chunk.title,
                    content=chunk.content,
                    score=float(score),
                    metadata={
                        **chunk.metadata,
                        "retrieval_score": str(chunk.score),
                        "rerank_score": str(float(score)),
                    },
                )
                for chunk, score in reranked[:rerank_k]
            ]
        elif rerank_k > 0:
            chunks = chunks[: min(rerank_k, len(chunks))]

        return chunks

    async def _generate_answer(
        self,
        *,
        question: str,
        sources: list[SourceChunk],
        temperature: float,
    ) -> str:
        answer = await self._generation_service.generate(
            GenerationRequest(
                messages=[ChatMessage(role="user", content=question)],
                intent=IntentDecision(
                    intent="knowledge_qa",
                    need_rag=True,
                    rewrite_query=question,
                    rationale="Transient RAG lab evaluation",
                ),
                sources=sources,
                temperature=temperature,
            )
        )
        return answer


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _safe_sheet_title(title: str) -> str:
    invalid_chars = set('\\/*?:[]')
    cleaned = "".join("_" if char in invalid_chars else char for char in title).strip()
    cleaned = cleaned[:31]
    return cleaned or "Variant"
