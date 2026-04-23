from dataclasses import dataclass
import asyncio

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.persistence.knowledge_ingest_store import KnowledgeIngestStore
from app.persistence.trace_store import SQLTraceStore
from app.schemas.admin import RAGCompareVariant, RAGEvaluationCase, RAGRuntimeConfig
from app.schemas.chat import ChatMessage
from app.services.chat_pipeline import ChatPipeline
from app.services.embedding_service import CrossEncoderProvider, SentenceTransformerProvider
from app.services.generator_service import (
    MockGenerationBackend,
    QueuedGenerationService,
    RemoteGenerationBackend,
)
from app.services.infra_service import InfraService
from app.services.intent_service import (
    IntentService,
    MockIntentService,
    RemoteIntentService,
)
from app.services.knowledge_base import InMemoryKnowledgeBase, KnowledgeBase, QdrantKnowledgeBase
from app.services.knowledge_ingest_service import KnowledgeIngestService
from app.services.rag_lab_service import RAGLabService
from app.services.retriever_service import (
    InMemoryRetrieverService,
    QdrantRetrieverService,
    RetrieverService,
)
from app.services.trace_service import LangSmithObserver, TraceService


@dataclass
class ServiceContainer:
    settings: Settings
    knowledge_base: KnowledgeBase
    knowledge_ingest_service: KnowledgeIngestService
    infra_service: InfraService
    intent_service: IntentService
    retriever_service: RetrieverService
    generation_service: QueuedGenerationService
    trace_service: TraceService
    chat_pipeline: ChatPipeline
    rag_lab_service: RAGLabService
    embedder_provider: SentenceTransformerProvider | None = None
    reranker_provider: CrossEncoderProvider | None = None

    async def start(self) -> None:
        self.infra_service.setup()
        self.infra_service.ensure_minio_bucket()
        await self.knowledge_ingest_service.start()
        self.trace_service.setup()
        if self.embedder_provider is not None:
            await asyncio.to_thread(self.embedder_provider.preload)
        if self.reranker_provider is not None:
            await asyncio.to_thread(self.reranker_provider.preload)
        await self.intent_service.start()
        await self.generation_service.start()

    async def stop(self) -> None:
        await self.generation_service.stop()
        await self.intent_service.stop()
        await self.knowledge_ingest_service.stop()
        await self.trace_service.shutdown()
        self.infra_service.shutdown()

    def get_rag_runtime_config(self) -> RAGRuntimeConfig:
        top_k = self.settings.rag_top_k
        score_threshold = self.settings.rag_score_threshold
        candidate_multiplier = 3
        reranker_enabled = self.reranker_provider is not None

        if isinstance(self.retriever_service, QdrantRetrieverService):
            top_k = self.retriever_service.top_k
            score_threshold = self.retriever_service.score_threshold
            candidate_multiplier = self.retriever_service.candidate_multiplier
            reranker_enabled = self.retriever_service.reranker_enabled

        chunk_size = self.settings.rag_chunk_size
        chunk_overlap = self.settings.rag_chunk_overlap
        if isinstance(self.knowledge_base, QdrantKnowledgeBase):
            chunk_size = self.knowledge_base.chunk_size
            chunk_overlap = self.knowledge_base.chunk_overlap

        return RAGRuntimeConfig(
            top_k=top_k,
            score_threshold=score_threshold,
            candidate_multiplier=candidate_multiplier,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reranker_enabled=reranker_enabled,
            llm_temperature=self.settings.llm_temperature,
        )

    def update_rag_runtime_config(
        self,
        *,
        top_k: int,
        score_threshold: float,
        candidate_multiplier: int,
        chunk_size: int,
        chunk_overlap: int,
        reranker_enabled: bool,
        llm_temperature: float,
    ) -> RAGRuntimeConfig:
        self.settings.rag_top_k = top_k
        self.settings.rag_score_threshold = score_threshold
        self.settings.rag_chunk_size = chunk_size
        self.settings.rag_chunk_overlap = chunk_overlap
        self.settings.llm_temperature = llm_temperature
        self.chat_pipeline.update_generation_temperature(llm_temperature)

        if isinstance(self.retriever_service, QdrantRetrieverService):
            self.retriever_service.update_runtime_config(
                top_k=top_k,
                score_threshold=score_threshold,
                candidate_multiplier=candidate_multiplier,
                reranker_enabled=reranker_enabled,
                reranker_provider=self.reranker_provider,
            )
        if isinstance(self.knowledge_base, QdrantKnowledgeBase):
            self.knowledge_base.update_chunking(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return self.get_rag_runtime_config()

    async def compare_rag_variants(self, query: str, variants: list[RAGCompareVariant], generate_answer: bool):
        from app.schemas.admin import RAGCompareResponse, RAGCompareResult

        if not isinstance(self.retriever_service, QdrantRetrieverService):
            return RAGCompareResponse(query=query, results=[])

        original = self.get_rag_runtime_config()
        results: list[RAGCompareResult] = []
        try:
            for variant in variants:
                answer, sources = await self._run_variant_query(
                    query=query,
                    variant=variant,
                    generate_answer=generate_answer,
                )
                results.append(
                    RAGCompareResult(
                        name=variant.name,
                        answer=answer,
                        sources=sources,
                    )
                )
        finally:
            self.retriever_service.update_runtime_config(
                top_k=original.top_k,
                score_threshold=original.score_threshold,
                candidate_multiplier=original.candidate_multiplier,
                reranker_enabled=original.reranker_enabled,
                reranker_provider=self.reranker_provider,
            )
        return RAGCompareResponse(query=query, results=results)

    async def evaluate_rag_variants(
        self,
        cases: list[RAGEvaluationCase],
        variants: list[RAGCompareVariant],
        generate_answer: bool,
    ):
        from app.schemas.admin import (
            RAGEvaluationCaseResult,
            RAGEvaluationResponse,
            RAGEvaluationSummary,
            RAGEvaluationVariantResult,
        )

        if not isinstance(self.retriever_service, QdrantRetrieverService):
            return RAGEvaluationResponse(cases=[], summaries=[])

        original = self.get_rag_runtime_config()
        case_results: list[RAGEvaluationCaseResult] = []
        totals: dict[str, dict[str, float]] = {
            variant.name: {"cases": 0, "source_hits": 0, "answer_hits": 0, "returned_sources": 0}
            for variant in variants
        }

        try:
            for case in cases:
                variant_results: list[RAGEvaluationVariantResult] = []
                for variant in variants:
                    answer, sources = await self._run_variant_query(
                        query=case.query,
                        variant=variant,
                        generate_answer=generate_answer,
                    )
                    matched_sources = self._match_expected_sources(case.expected_sources, sources)
                    matched_answer_keywords = self._match_expected_answer_keywords(
                        case.expected_answer_keywords,
                        answer or "",
                    )
                    source_hit = bool(case.expected_sources) and len(matched_sources) == len(case.expected_sources)
                    answer_hit = bool(case.expected_answer_keywords) and (
                        len(matched_answer_keywords) == len(case.expected_answer_keywords)
                    )
                    if not case.expected_sources:
                        source_hit = len(sources) > 0
                    if not case.expected_answer_keywords:
                        answer_hit = bool(answer)

                    totals[variant.name]["cases"] += 1
                    totals[variant.name]["source_hits"] += 1 if source_hit else 0
                    totals[variant.name]["answer_hits"] += 1 if answer_hit else 0
                    totals[variant.name]["returned_sources"] += len(sources)

                    variant_results.append(
                        RAGEvaluationVariantResult(
                            name=variant.name,
                            answer=answer,
                            sources=sources,
                            source_hit=source_hit,
                            answer_hit=answer_hit,
                            matched_sources=matched_sources,
                            matched_answer_keywords=matched_answer_keywords,
                        )
                    )
                case_results.append(
                    RAGEvaluationCaseResult(
                        query=case.query,
                        expected_sources=case.expected_sources,
                        expected_answer_keywords=case.expected_answer_keywords,
                        variants=variant_results,
                    )
                )
        finally:
            self.retriever_service.update_runtime_config(
                top_k=original.top_k,
                score_threshold=original.score_threshold,
                candidate_multiplier=original.candidate_multiplier,
                reranker_enabled=original.reranker_enabled,
                reranker_provider=self.reranker_provider,
            )

        summaries = [
            RAGEvaluationSummary(
                name=variant.name,
                total_cases=int(totals[variant.name]["cases"]),
                source_hit_cases=int(totals[variant.name]["source_hits"]),
                answer_hit_cases=int(totals[variant.name]["answer_hits"]),
                source_hit_rate=(
                    totals[variant.name]["source_hits"] / totals[variant.name]["cases"]
                    if totals[variant.name]["cases"]
                    else 0.0
                ),
                answer_hit_rate=(
                    totals[variant.name]["answer_hits"] / totals[variant.name]["cases"]
                    if totals[variant.name]["cases"]
                    else 0.0
                ),
                average_returned_sources=(
                    totals[variant.name]["returned_sources"] / totals[variant.name]["cases"]
                    if totals[variant.name]["cases"]
                    else 0.0
                ),
            )
            for variant in variants
        ]
        return RAGEvaluationResponse(cases=case_results, summaries=summaries)

    async def apply_rag_lab_variant(self, session_id: str, variant_id: str):
        from app.schemas.admin import RAGLabApplyResponse

        variant = await self.rag_lab_service.get_variant(session_id, variant_id)
        if variant is None:
            raise KeyError(f"Unknown RAG lab variant: {session_id}/{variant_id}")

        if variant.rerank_k > 0:
            top_k = variant.rerank_k
            candidate_multiplier = max(1, -(-variant.retrieval_k // max(variant.rerank_k, 1)))
            reranker_enabled = True
        else:
            top_k = variant.retrieval_k
            candidate_multiplier = 1
            reranker_enabled = False

        overlap_chars = int(round(variant.chunk_size * variant.chunk_overlap_ratio))
        applied = self.update_rag_runtime_config(
            top_k=top_k,
            score_threshold=self.get_rag_runtime_config().score_threshold,
            candidate_multiplier=candidate_multiplier,
            chunk_size=variant.chunk_size,
            chunk_overlap=overlap_chars,
            reranker_enabled=reranker_enabled,
            llm_temperature=variant.temperature,
        )
        return RAGLabApplyResponse(
            session_id=session_id,
            variant_id=variant_id,
            applied_config=applied,
            note="chunk 参数只会作用于后续新导入文档；现有知识库不会自动重切分。",
        )

    async def _run_variant_query(
        self,
        *,
        query: str,
        variant: RAGCompareVariant,
        generate_answer: bool,
    ) -> tuple[str | None, list]:
        from app.schemas.chat import IntentDecision
        from app.services.generator_service import GenerationRequest

        self.retriever_service.update_runtime_config(
            top_k=variant.top_k,
            score_threshold=variant.score_threshold,
            candidate_multiplier=variant.candidate_multiplier,
            reranker_enabled=variant.reranker_enabled,
            reranker_provider=self.reranker_provider,
        )
        sources = await self.retriever_service.retrieve(query)
        answer = None
        if generate_answer:
            answer = await self.generation_service.generate(
                GenerationRequest(
                    messages=[ChatMessage(role="user", content=query)],
                    intent=IntentDecision(
                        intent="knowledge_qa",
                        need_rag=True,
                        rewrite_query=query,
                        rationale=f"Admin evaluation variant: {variant.name}",
                    ),
                    sources=sources,
                )
            )
        return answer, sources

    @staticmethod
    def _match_expected_sources(expected_sources: list[str], sources: list) -> list[str]:
        matched: list[str] = []
        haystacks = [
            " ".join(
                filter(
                    None,
                    [
                        source.title,
                        source.document_id,
                        source.content,
                    ],
                )
            ).lower()
            for source in sources
        ]
        for expected in expected_sources:
            normalized = expected.strip().lower()
            if normalized and any(normalized in haystack for haystack in haystacks):
                matched.append(expected)
        return matched

    @staticmethod
    def _match_expected_answer_keywords(expected_keywords: list[str], answer: str) -> list[str]:
        normalized_answer = answer.lower()
        matched: list[str] = []
        for keyword in expected_keywords:
            normalized = keyword.strip().lower()
            if normalized and normalized in normalized_answer:
                matched.append(keyword)
        return matched


def build_service_container(settings: Settings) -> ServiceContainer:
    infra_service = InfraService(settings=settings)
    embedder_provider: SentenceTransformerProvider | None = None
    reranker_provider: CrossEncoderProvider | None = None
    if settings.qdrant_url and settings.embedding_model_path:
        qdrant_client = QdrantClient(url=settings.qdrant_url)
        embedder_provider = SentenceTransformerProvider(
            model_path=settings.embedding_model_path,
            device="cpu",
        )
        reranker_provider = (
            CrossEncoderProvider(
                model_path=settings.reranker_model_path,
                device="cpu",
            )
            if settings.reranker_model_path
            else None
        )
        knowledge_base = QdrantKnowledgeBase(
            qdrant_client=qdrant_client,
            embedder_provider=embedder_provider,
            collection_name=settings.rag_collection_name,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        retriever_service = QdrantRetrieverService(
            qdrant_client=qdrant_client,
            embedder_provider=embedder_provider,
            collection_name=settings.rag_collection_name,
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
            reranker_provider=reranker_provider,
        )
    else:
        knowledge_base = InMemoryKnowledgeBase()
        retriever_service = InMemoryRetrieverService(
            knowledge_base=knowledge_base,
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
        )
    trace_store = SQLTraceStore(settings.trace_store_dsn)
    langsmith_observer = LangSmithObserver(
        enabled=settings.langsmith_enabled,
        project_name=settings.langsmith_project,
        endpoint=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key,
    )
    trace_service = TraceService(
        store=trace_store,
        observer=langsmith_observer,
    )
    if settings.runtime_mode == "remote_inference":
        intent_service: IntentService = RemoteIntentService(
            base_url=settings.inference_service_url,
            timeout_seconds=settings.inference_timeout_seconds,
        )
        generation_backend = RemoteGenerationBackend(
            base_url=settings.inference_service_url,
            timeout_seconds=settings.inference_timeout_seconds,
            max_new_tokens=settings.llm_max_new_tokens,
        )
    else:
        intent_service = MockIntentService()
        generation_backend = MockGenerationBackend()

    generation_service = QueuedGenerationService(
        backend=generation_backend,
        maxsize=settings.gpu_queue_maxsize,
    )
    knowledge_ingest_store = KnowledgeIngestStore(settings.trace_store_dsn)
    knowledge_ingest_service = KnowledgeIngestService(
        knowledge_base=knowledge_base,
        store=knowledge_ingest_store,
    )
    chat_pipeline = ChatPipeline(
        intent_service=intent_service,
        retriever_service=retriever_service,
        generation_service=generation_service,
        trace_service=trace_service,
        generation_temperature=settings.llm_temperature,
    )
    rag_lab_service = RAGLabService(
        embedder_provider=embedder_provider,
        reranker_provider=reranker_provider,
        generation_service=generation_service,
    )
    return ServiceContainer(
        settings=settings,
        knowledge_base=knowledge_base,
        knowledge_ingest_service=knowledge_ingest_service,
        infra_service=infra_service,
        intent_service=intent_service,
        retriever_service=retriever_service,
        generation_service=generation_service,
        trace_service=trace_service,
        chat_pipeline=chat_pipeline,
        rag_lab_service=rag_lab_service,
        embedder_provider=embedder_provider,
        reranker_provider=reranker_provider,
    )
