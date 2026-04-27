from __future__ import annotations

import asyncio
from contextvars import Token
from contextvars import ContextVar
import math
import re
from time import perf_counter
from typing import Literal, Protocol
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient

from app.schemas.chat import SourceChunk
from app.services.bm25_index_store import SQLiteBM25IndexStore
from app.services.embedding_service import CrossEncoderProvider, SentenceTransformerProvider
from app.services.knowledge_base import InMemoryKnowledgeBase, KnowledgeRecord
from app.services.rag_snapshot_service import record_retrieval_query, record_retrieval_stage

RetrievalMode = Literal["dense", "bm25", "hybrid"]

SIMILAR_CHUNK_THRESHOLD = 0.85
MAX_MERGED_SOURCE_LENGTH = 2400
MAX_COARSE_CHUNKS_PER_DOCUMENT = 2
_retrieval_timings: ContextVar[dict[str, int] | None] = ContextVar(
    "retrieval_timings",
    default=None,
)


def begin_retrieval_timings() -> Token:
    return _retrieval_timings.set({"embedding": 0, "qdrant_search": 0, "rerank": 0})


def get_retrieval_timings() -> dict[str, int]:
    return dict(_retrieval_timings.get() or {})


def end_retrieval_timings(token: Token) -> None:
    _retrieval_timings.reset(token)


def _record_timing(name: str, started_at: float) -> None:
    timings = _retrieval_timings.get()
    if timings is not None:
        timings[name] = timings.get(name, 0) + int((perf_counter() - started_at) * 1000)


def _normalize_filter_knowledge_base_id(value: str | None) -> str | None:
    normalized = (value or "").strip()
    if not normalized or normalized == "default":
        return None
    return normalized


def _build_knowledge_base_filter(knowledge_base_id: str | None):
    normalized = _normalize_filter_knowledge_base_id(knowledge_base_id)
    if normalized is None:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="knowledge_base_id",
                match=models.MatchValue(value=normalized),
            )
        ]
    )


class RetrieverService(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        use_reranker: bool | None = None,
        knowledge_base_id: str | None = None,
    ) -> list[SourceChunk]:
        ...


class InMemoryRetrieverService:
    def __init__(
        self,
        knowledge_base: InMemoryKnowledgeBase,
        top_k: int = 4,
        score_threshold: float = 0.1,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._top_k = top_k
        self._score_threshold = score_threshold

    async def retrieve(
        self,
        query: str,
        *,
        use_reranker: bool | None = None,
        knowledge_base_id: str | None = None,
    ) -> list[SourceChunk]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored_records: list[tuple[float, KnowledgeRecord]] = []
        for record in self._knowledge_base.list_documents():
            if knowledge_base_id and record.metadata.get("knowledge_base_id", "default") != knowledge_base_id:
                continue
            document_terms = self._tokenize(f"{record.title} {record.content}")
            overlap = len(query_terms.intersection(document_terms))
            score = overlap / max(len(query_terms), 1)
            if score >= self._score_threshold:
                scored_records.append((score, record))

        scored_records.sort(key=lambda item: item[0], reverse=True)
        return [
            SourceChunk(
                document_id=record.document_id,
                title=record.title,
                content=record.content[:500],
                score=score,
                metadata=record.metadata,
            )
            for score, record in scored_records[: self._top_k]
        ]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.split(r"[\s,.;:!?锛屻€傦紱锛氾紒锛?\\]+", text) if token}


class QdrantRetrieverService:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedder_provider: SentenceTransformerProvider,
        collection_name: str,
        top_k: int = 4,
        score_threshold: float = 0.1,
        rerank_candidate_limit: int = 12,
        reranker_provider: CrossEncoderProvider | None = None,
    ) -> None:
        self._qdrant_client = qdrant_client
        self._embedder_provider = embedder_provider
        self._collection_name = collection_name
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._rerank_candidate_limit = rerank_candidate_limit
        self._reranker_provider = reranker_provider
        self._candidate_multiplier = 3

    async def retrieve(
        self,
        query: str,
        *,
        use_reranker: bool | None = None,
        knowledge_base_id: str | None = None,
    ) -> list[SourceChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        record_retrieval_query(normalized_query)
        chunks = await self.retrieve_candidates(normalized_query, knowledge_base_id=knowledge_base_id)
        chunks = _coarse_deduplicate_candidates(chunks)
        record_retrieval_stage("coarse_deduped", chunks)
        if use_reranker is not False and self._reranker_provider is not None and chunks:
            chunks = await _rerank_chunks(
                normalized_query,
                chunks,
                self._reranker_provider,
                max_candidates=self._rerank_candidate_limit,
                min_score=self._score_threshold,
            )
        elif use_reranker is False:
            chunks = _filter_by_score(chunks, min_score=self._score_threshold)
            record_retrieval_stage(
                "score_filtered",
                chunks,
                metadata={"min_score": self._score_threshold},
            )
        chunks = await self.post_process_sources(chunks)
        record_retrieval_stage("final_sources", chunks[: self._top_k])
        return chunks[: self._top_k]

    async def retrieve_candidates(
        self,
        query: str,
        *,
        limit: int | None = None,
        knowledge_base_id: str | None = None,
    ) -> list[SourceChunk]:
        normalized_query = query.strip()
        if not normalized_query or not self._collection_exists():
            return []

        started_at = perf_counter()
        vector = self._get_embedder().encode(
            normalized_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        _record_timing("embedding", started_at)

        started_at = perf_counter()
        results = self._search_points(
            vector.tolist(),
            limit=limit or max(self._top_k, self._top_k * self._candidate_multiplier),
            knowledge_base_id=knowledge_base_id,
        )
        _record_timing("qdrant_search", started_at)
        chunks = [
            SourceChunk(
                document_id=str(hit.payload.get("document_id", "")),
                title=str(hit.payload.get("title", "")),
                content=str(hit.payload.get("content", "")),
                score=float(hit.score),
                metadata={
                    **{
                        key: str(value)
                        for key, value in (hit.payload.get("metadata") or {}).items()
                    },
                        "chunk_id": str(hit.payload.get("chunk_id", "")),
                        "knowledge_base_id": str(hit.payload.get("knowledge_base_id") or "default"),
                        "knowledge_base_name": str(hit.payload.get("knowledge_base_name") or "默认知识库"),
                        "retrieval_mode": "dense",
                },
            )
            for hit in results
        ]
        record_retrieval_stage(
            "dense_candidates",
            chunks,
            metadata={"limit": limit or max(self._top_k, self._top_k * self._candidate_multiplier)},
        )
        return chunks

    def _search_points(self, query_vector: list[float], *, limit: int, knowledge_base_id: str | None = None):
        query_filter = _build_knowledge_base_filter(knowledge_base_id)
        if hasattr(self._qdrant_client, "search"):
            return self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=self._score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

        query_response = self._qdrant_client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=self._score_threshold,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        return query_response.points

    def _get_embedder(self):
        return self._embedder_provider.get_model()

    async def post_process_sources(self, chunks: list[SourceChunk]) -> list[SourceChunk]:
        return await _post_process_sources(
            chunks,
            vector_lookup=self.fetch_chunk_vectors,
        )

    async def fetch_chunk_vectors(self, chunks: list[SourceChunk]) -> dict[str, list[float]]:
        chunk_ids = [
            str(chunk.metadata.get("chunk_id", ""))
            for chunk in chunks
            if chunk.metadata.get("chunk_id")
        ]
        if not chunk_ids or not self._collection_exists():
            return {}

        point_ids = [str(uuid5(NAMESPACE_URL, chunk_id)) for chunk_id in chunk_ids]
        point_to_chunk = dict(zip(point_ids, chunk_ids))

        def retrieve_points():
            return self._qdrant_client.retrieve(
                collection_name=self._collection_name,
                ids=point_ids,
                with_payload=True,
                with_vectors=True,
            )

        try:
            points = await asyncio.to_thread(retrieve_points)
        except Exception:
            return {}

        vectors: dict[str, list[float]] = {}
        for point in points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id") or point_to_chunk.get(str(point.id), ""))
            vector = _extract_point_vector(getattr(point, "vector", None))
            if chunk_id and vector:
                vectors[chunk_id] = vector
        return vectors

    def _collection_exists(self) -> bool:
        collection_names = {
            collection.name for collection in self._qdrant_client.get_collections().collections
        }
        return self._collection_name in collection_names

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    @property
    def candidate_multiplier(self) -> int:
        return self._candidate_multiplier

    @property
    def rerank_candidate_limit(self) -> int:
        return self._rerank_candidate_limit

    @property
    def reranker_enabled(self) -> bool:
        return self._reranker_provider is not None

    def update_runtime_config(
        self,
        *,
        top_k: int,
        score_threshold: float,
        candidate_multiplier: int,
        reranker_enabled: bool,
        rerank_candidate_limit: int | None = None,
        reranker_provider: CrossEncoderProvider | None = None,
        retrieval_mode: RetrievalMode | None = None,
        bm25_top_k: int | None = None,
        bm25_title_boost: float | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._candidate_multiplier = candidate_multiplier
        if rerank_candidate_limit is not None:
            self._rerank_candidate_limit = rerank_candidate_limit
        self._reranker_provider = reranker_provider if reranker_enabled else None


class BM25RetrieverService:
    def __init__(
        self,
        index_store: SQLiteBM25IndexStore,
        top_k: int = 8,
        title_boost: float = 2.0,
        rerank_candidate_limit: int = 12,
        reranker_provider: CrossEncoderProvider | None = None,
    ) -> None:
        self._index_store = index_store
        self._top_k = top_k
        self._title_boost = title_boost
        self._rerank_candidate_limit = rerank_candidate_limit
        self._reranker_provider = reranker_provider

    async def retrieve(
        self,
        query: str,
        *,
        use_reranker: bool | None = None,
        knowledge_base_id: str | None = None,
    ) -> list[SourceChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        record_retrieval_query(normalized_query)
        chunks = self.retrieve_candidates(normalized_query, knowledge_base_id=knowledge_base_id)
        chunks = _coarse_deduplicate_candidates(chunks)
        record_retrieval_stage("coarse_deduped", chunks)
        if use_reranker is False:
            chunks = _filter_by_score(chunks, min_score=0.1)
            record_retrieval_stage("score_filtered", chunks, metadata={"min_score": 0.1})
        if use_reranker is not False and self._reranker_provider is not None and chunks:
            chunks = await _rerank_chunks(
                normalized_query,
                chunks,
                self._reranker_provider,
                max_candidates=self._rerank_candidate_limit,
                min_score=0.1,
            )
        record_retrieval_stage("final_sources", chunks[: self._top_k])
        return chunks[: self._top_k]

    def retrieve_candidates(
        self,
        query: str,
        *,
        limit: int | None = None,
        knowledge_base_id: str | None = None,
    ) -> list[SourceChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        results = self._index_store.search(
            normalized_query,
            limit=limit or self._top_k,
            title_boost=self._title_boost,
            knowledge_base_id=knowledge_base_id,
        )
        chunks = [
            SourceChunk(
                document_id=result.document_id,
                title=result.title,
                content=result.content,
                score=result.score,
                metadata={
                    **result.metadata,
                    "chunk_id": result.chunk_id,
                    "knowledge_base_id": result.metadata.get("knowledge_base_id", "default"),
                    "knowledge_base_name": result.metadata.get("knowledge_base_name", "默认知识库"),
                    "retrieval_mode": "bm25",
                },
            )
            for result in results
        ]
        record_retrieval_stage(
            "bm25_candidates",
            chunks,
            metadata={"limit": limit or self._top_k, "title_boost": self._title_boost},
        )
        return chunks

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def title_boost(self) -> float:
        return self._title_boost

    @property
    def rerank_candidate_limit(self) -> int:
        return self._rerank_candidate_limit

    @property
    def reranker_enabled(self) -> bool:
        return self._reranker_provider is not None

    def update_runtime_config(
        self,
        *,
        top_k: int,
        title_boost: float,
        reranker_enabled: bool,
        rerank_candidate_limit: int | None = None,
        reranker_provider: CrossEncoderProvider | None = None,
    ) -> None:
        self._top_k = top_k
        self._title_boost = title_boost
        if rerank_candidate_limit is not None:
            self._rerank_candidate_limit = rerank_candidate_limit
        self._reranker_provider = reranker_provider if reranker_enabled else None


class HybridRetrieverService:
    def __init__(
        self,
        dense_retriever: QdrantRetrieverService,
        bm25_retriever: BM25RetrieverService,
        *,
        top_k: int = 4,
        score_threshold: float = 0.1,
        candidate_multiplier: int = 3,
        rerank_candidate_limit: int = 12,
        rrf_min_score: float = 0.0,
        reranker_provider: CrossEncoderProvider | None = None,
        retrieval_mode: RetrievalMode = "hybrid",
        bm25_top_k: int = 8,
        bm25_title_boost: float = 2.0,
        rrf_k: int = 60,
    ) -> None:
        self._dense_retriever = dense_retriever
        self._bm25_retriever = bm25_retriever
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._candidate_multiplier = candidate_multiplier
        self._rerank_candidate_limit = rerank_candidate_limit
        self._rrf_min_score = rrf_min_score
        self._reranker_provider = reranker_provider
        self._retrieval_mode = retrieval_mode
        self._bm25_top_k = bm25_top_k
        self._bm25_title_boost = bm25_title_boost
        self._rrf_k = rrf_k
        self._sync_children()

    async def retrieve(
        self,
        query: str,
        *,
        use_reranker: bool | None = None,
        knowledge_base_id: str | None = None,
    ) -> list[SourceChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        record_retrieval_query(normalized_query)
        scoped_knowledge_base_id = _normalize_filter_knowledge_base_id(knowledge_base_id)

        if self._retrieval_mode == "dense":
            chunks = await self._dense_retriever.retrieve_candidates(
                normalized_query,
                limit=max(self._top_k, self._top_k * self._candidate_multiplier),
                knowledge_base_id=scoped_knowledge_base_id,
            )
        elif self._retrieval_mode == "bm25":
            chunks = self._bm25_retriever.retrieve_candidates(
                normalized_query,
                limit=max(self._top_k, self._bm25_top_k),
                knowledge_base_id=scoped_knowledge_base_id,
            )
        else:
            dense_chunks = await self._dense_retriever.retrieve_candidates(
                normalized_query,
                limit=max(self._top_k, self._top_k * self._candidate_multiplier),
                knowledge_base_id=scoped_knowledge_base_id,
            )
            bm25_chunks = self._bm25_retriever.retrieve_candidates(
                normalized_query,
                limit=max(self._top_k, self._bm25_top_k),
                knowledge_base_id=scoped_knowledge_base_id,
            )
            chunks = self._fuse_with_rrf(dense_chunks, bm25_chunks)
            record_retrieval_stage(
                "rrf_fused",
                chunks,
                metadata={"rrf_k": self._rrf_k},
            )

        chunks = _coarse_deduplicate_candidates(chunks)
        record_retrieval_stage(
            "coarse_deduped",
            chunks,
            metadata={"max_chunks_per_document": MAX_COARSE_CHUNKS_PER_DOCUMENT},
        )
        if use_reranker is not False and self._reranker_provider is not None and chunks:
            chunks = await _rerank_chunks(
                normalized_query,
                chunks,
                self._reranker_provider,
                max_candidates=self._rerank_candidate_limit,
                min_score=self._score_threshold,
            )
        elif self._retrieval_mode == "hybrid":
            chunks = _filter_by_score(chunks, min_score=self._rrf_min_score)
            record_retrieval_stage(
                "score_filtered",
                chunks,
                metadata={"min_score": self._rrf_min_score, "score_type": "rrf"},
            )
        elif self._retrieval_mode == "bm25":
            chunks = _filter_by_score(chunks, min_score=self._score_threshold)
            record_retrieval_stage(
                "score_filtered",
                chunks,
                metadata={"min_score": self._score_threshold, "score_type": "bm25"},
            )
        chunks = await self._dense_retriever.post_process_sources(chunks)
        record_retrieval_stage("final_sources", chunks[: self._top_k])
        return chunks[: self._top_k]

    def _fuse_with_rrf(
        self,
        dense_chunks: list[SourceChunk],
        bm25_chunks: list[SourceChunk],
    ) -> list[SourceChunk]:
        merged: dict[str, dict[str, object]] = {}

        def accumulate(chunks: list[SourceChunk], source_name: str) -> None:
            for rank, chunk in enumerate(chunks, start=1):
                key = _chunk_key(chunk)
                if key not in merged:
                    merged[key] = {
                        "chunk": chunk,
                        "score": 0.0,
                        "sources": [],
                        "dense_score": "",
                        "bm25_score": "",
                    }
                bucket = merged[key]
                bucket["score"] = float(bucket["score"]) + (1.0 / (self._rrf_k + rank))
                sources = list(bucket["sources"])
                if source_name not in sources:
                    sources.append(source_name)
                bucket["sources"] = sources
                if source_name == "dense":
                    bucket["dense_score"] = str(chunk.score)
                if source_name == "bm25":
                    bucket["bm25_score"] = str(chunk.score)

        accumulate(dense_chunks, "dense")
        accumulate(bm25_chunks, "bm25")

        fused: list[SourceChunk] = []
        for bucket in sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True):
            chunk = bucket["chunk"]
            metadata = {
                **chunk.metadata,
                "retrieval_mode": "hybrid",
                "fusion_sources": ",".join(bucket["sources"]),
                "fusion_score": str(bucket["score"]),
            }
            if bucket["dense_score"]:
                metadata["dense_score"] = str(bucket["dense_score"])
            if bucket["bm25_score"]:
                metadata["bm25_score"] = str(bucket["bm25_score"])
            fused.append(
                SourceChunk(
                    document_id=chunk.document_id,
                    title=chunk.title,
                    content=chunk.content,
                    score=float(bucket["score"]),
                    metadata=metadata,
                )
            )
        return fused

    def _sync_children(self) -> None:
        self._dense_retriever.update_runtime_config(
            top_k=self._top_k,
            score_threshold=self._score_threshold,
            candidate_multiplier=self._candidate_multiplier,
            rerank_candidate_limit=self._rerank_candidate_limit,
            reranker_enabled=False,
            reranker_provider=None,
        )
        self._bm25_retriever.update_runtime_config(
            top_k=self._bm25_top_k,
            title_boost=self._bm25_title_boost,
            rerank_candidate_limit=self._rerank_candidate_limit,
            reranker_enabled=False,
            reranker_provider=None,
        )

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    @property
    def candidate_multiplier(self) -> int:
        return self._candidate_multiplier

    @property
    def rerank_candidate_limit(self) -> int:
        return self._rerank_candidate_limit

    @property
    def reranker_enabled(self) -> bool:
        return self._reranker_provider is not None

    @property
    def retrieval_mode(self) -> RetrievalMode:
        return self._retrieval_mode

    @property
    def bm25_top_k(self) -> int:
        return self._bm25_top_k

    @property
    def bm25_title_boost(self) -> float:
        return self._bm25_title_boost

    @property
    def rrf_k(self) -> int:
        return self._rrf_k

    @property
    def rrf_min_score(self) -> float:
        return self._rrf_min_score

    def update_runtime_config(
        self,
        *,
        top_k: int,
        score_threshold: float,
        candidate_multiplier: int,
        reranker_enabled: bool,
        rerank_candidate_limit: int | None = None,
        rrf_min_score: float | None = None,
        reranker_provider: CrossEncoderProvider | None = None,
        retrieval_mode: RetrievalMode | None = None,
        bm25_top_k: int | None = None,
        bm25_title_boost: float | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._candidate_multiplier = candidate_multiplier
        if rerank_candidate_limit is not None:
            self._rerank_candidate_limit = rerank_candidate_limit
        if rrf_min_score is not None:
            self._rrf_min_score = rrf_min_score
        self._reranker_provider = reranker_provider if reranker_enabled else None
        if retrieval_mode is not None:
            self._retrieval_mode = retrieval_mode
        if bm25_top_k is not None:
            self._bm25_top_k = bm25_top_k
        if bm25_title_boost is not None:
            self._bm25_title_boost = bm25_title_boost
        if rrf_k is not None:
            self._rrf_k = rrf_k
        self._sync_children()


async def _rerank_chunks(
    query: str,
    chunks: list[SourceChunk],
    reranker_provider: CrossEncoderProvider,
    *,
    max_candidates: int | None = None,
    min_score: float | None = None,
) -> list[SourceChunk]:
    if max_candidates is not None:
        chunks = chunks[:max(1, max_candidates)]
    record_retrieval_stage("rerank_input", chunks, metadata={"max_candidates": max_candidates})
    reranker = reranker_provider.get_model()
    pairs = [(query, f"{chunk.title}\n{chunk.content}") for chunk in chunks]
    started_at = perf_counter()
    scores = await asyncio.to_thread(reranker.predict, pairs)
    _record_timing("rerank", started_at)
    reranked = sorted(
        zip(chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    ranked_chunks = [
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
        for chunk, score in reranked
    ]
    record_retrieval_stage("rerank_output", ranked_chunks)
    if min_score is None:
        return ranked_chunks
    filtered = [chunk for chunk in ranked_chunks if chunk.score >= min_score]
    record_retrieval_stage("rerank_score_filtered", filtered, metadata={"min_score": min_score})
    return filtered


def _chunk_key(chunk: SourceChunk) -> str:
    chunk_id = chunk.metadata.get("chunk_id")
    if chunk_id:
        return chunk_id
    return f"{chunk.document_id}:{chunk.title}:{hash(chunk.content)}"


def _coarse_deduplicate_candidates(chunks: list[SourceChunk]) -> list[SourceChunk]:
    """Cheaply trim obvious duplicate candidates before scoring or rerank."""
    by_key: dict[str, SourceChunk] = {}
    for chunk in chunks:
        key = _coarse_chunk_key(chunk)
        existing = by_key.get(key)
        if existing is None or chunk.score > existing.score:
            by_key[key] = chunk

    unique_chunks = sorted(by_key.values(), key=lambda chunk: chunk.score, reverse=True)
    document_counts: dict[str, int] = {}
    kept: list[SourceChunk] = []
    for chunk in unique_chunks:
        document_key = chunk.document_id or chunk.title
        count = document_counts.get(document_key, 0)
        if count >= MAX_COARSE_CHUNKS_PER_DOCUMENT:
            continue
        kept.append(chunk)
        document_counts[document_key] = count + 1

    return kept


def _coarse_chunk_key(chunk: SourceChunk) -> str:
    chunk_id = chunk.metadata.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    normalized_content = re.sub(r"\s+", "", chunk.content or "")
    return f"content:{chunk.document_id}:{hash(normalized_content)}"


def _filter_by_score(chunks: list[SourceChunk], *, min_score: float) -> list[SourceChunk]:
    if min_score <= 0:
        return chunks
    return [chunk for chunk in chunks if chunk.score >= min_score]


async def _post_process_sources(
    chunks: list[SourceChunk],
    *,
    vector_lookup,
) -> list[SourceChunk]:
    if not chunks:
        return []
    record_retrieval_stage("postprocess_input", chunks)
    vectors = await vector_lookup(chunks)
    deduplicated = _deduplicate_similar_chunks(chunks, vectors)
    record_retrieval_stage(
        "vector_deduped",
        deduplicated,
        metadata={"similarity_threshold": SIMILAR_CHUNK_THRESHOLD},
    )
    merged = _merge_adjacent_chunks(deduplicated)
    record_retrieval_stage("adjacent_merged", merged)
    cited = _assign_citation_indices(merged)
    record_retrieval_stage("citation_mapped", cited)
    return cited


def _deduplicate_similar_chunks(
    chunks: list[SourceChunk],
    vectors: dict[str, list[float]],
) -> list[SourceChunk]:
    kept: list[SourceChunk] = []
    kept_vectors: list[list[float] | None] = []

    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id", "")
        vector = vectors.get(chunk_id)
        duplicate_index = None
        for index, existing in enumerate(kept):
            if existing.document_id != chunk.document_id:
                continue
            existing_vector = kept_vectors[index]
            if vector is None or existing_vector is None:
                continue
            if _cosine_similarity(vector, existing_vector) > SIMILAR_CHUNK_THRESHOLD:
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(chunk)
            kept_vectors.append(vector)
            continue

        if chunk.score > kept[duplicate_index].score:
            kept[duplicate_index] = chunk
            kept_vectors[duplicate_index] = vector

    return kept


def _merge_adjacent_chunks(chunks: list[SourceChunk]) -> list[SourceChunk]:
    indexed: list[tuple[int, int, SourceChunk]] = []
    passthrough: list[tuple[int, SourceChunk]] = []
    for rank, chunk in enumerate(chunks):
        chunk_index = _parse_chunk_index(chunk)
        if chunk_index is None:
            passthrough.append((rank, chunk))
        else:
            indexed.append((rank, chunk_index, chunk))

    groups: list[dict[str, object]] = []
    for document_id in {chunk.document_id for _, _, chunk in indexed}:
        document_chunks = sorted(
            (item for item in indexed if item[2].document_id == document_id),
            key=lambda item: item[1],
        )
        current: list[tuple[int, int, SourceChunk]] = []
        for item in document_chunks:
            if not current or item[1] - current[-1][1] <= 1:
                current.append(item)
                continue
            groups.append(_build_merge_group(current))
            current = [item]
        if current:
            groups.append(_build_merge_group(current))

    groups.extend(
        {
            "rank": rank,
            "source": _with_metadata(
                chunk,
                {
                    "chunk_ids": chunk.metadata.get("chunk_id", ""),
                    "merged_chunk_count": "1",
                },
            ),
        }
        for rank, chunk in passthrough
    )

    groups.sort(key=lambda group: int(group["rank"]))
    return [group["source"] for group in groups]


def _build_merge_group(items: list[tuple[int, int, SourceChunk]]) -> dict[str, object]:
    rank = min(item[0] for item in items)
    chunks_in_order = [item[2] for item in sorted(items, key=lambda item: item[1])]
    best_chunk = max(chunks_in_order, key=lambda chunk: chunk.score)
    chunk_ids = [chunk.metadata.get("chunk_id", "") for chunk in chunks_in_order]
    chunk_indexes = [str(item[1]) for item in sorted(items, key=lambda item: item[1])]
    content = _merge_content([chunk.content for chunk in chunks_in_order])
    metadata = {
        **best_chunk.metadata,
        "chunk_ids": ",".join(chunk_id for chunk_id in chunk_ids if chunk_id),
        "chunk_index_range": f"{chunk_indexes[0]}-{chunk_indexes[-1]}",
        "merged_chunk_count": str(len(chunks_in_order)),
    }
    if len(chunks_in_order) > 1:
        metadata["source_governance"] = "adjacent_merged"
    return {
        "rank": rank,
        "source": SourceChunk(
            document_id=best_chunk.document_id,
            title=best_chunk.title,
            content=content,
            score=best_chunk.score,
            metadata={key: str(value) for key, value in metadata.items()},
        ),
    }


def _assign_citation_indices(chunks: list[SourceChunk]) -> list[SourceChunk]:
    return [
        _with_metadata(chunk, {"citation_index": str(index)})
        for index, chunk in enumerate(chunks, start=1)
    ]


def _with_metadata(chunk: SourceChunk, metadata: dict[str, str]) -> SourceChunk:
    return SourceChunk(
        document_id=chunk.document_id,
        title=chunk.title,
        content=chunk.content,
        score=chunk.score,
        metadata={**chunk.metadata, **{key: str(value) for key, value in metadata.items()}},
    )


def _parse_chunk_index(chunk: SourceChunk) -> int | None:
    chunk_id = chunk.metadata.get("chunk_id", "")
    if ":" not in chunk_id:
        return None
    suffix = chunk_id.rsplit(":", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _merge_content(parts: list[str]) -> str:
    merged = "\n\n".join(part.strip() for part in parts if part.strip())
    if len(merged) <= MAX_MERGED_SOURCE_LENGTH:
        return merged
    return merged[: MAX_MERGED_SOURCE_LENGTH - 1].rstrip() + "…"


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _extract_point_vector(vector) -> list[float] | None:
    if vector is None:
        return None
    if isinstance(vector, dict):
        first_vector = next(iter(vector.values()), None)
        return _extract_point_vector(first_vector)
    if isinstance(vector, list):
        return [float(value) for value in vector]
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return None
