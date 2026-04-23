from __future__ import annotations

import asyncio
import re
from typing import Protocol

from qdrant_client import QdrantClient

from app.schemas.chat import SourceChunk
from app.services.embedding_service import CrossEncoderProvider, SentenceTransformerProvider
from app.services.knowledge_base import InMemoryKnowledgeBase, KnowledgeRecord


class RetrieverService(Protocol):
    async def retrieve(self, query: str) -> list[SourceChunk]:
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

    async def retrieve(self, query: str) -> list[SourceChunk]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored_records: list[tuple[float, KnowledgeRecord]] = []
        for record in self._knowledge_base.list_documents():
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
        reranker_provider: CrossEncoderProvider | None = None,
    ) -> None:
        self._qdrant_client = qdrant_client
        self._embedder_provider = embedder_provider
        self._collection_name = collection_name
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._reranker_provider = reranker_provider
        self._candidate_multiplier = 3

    async def retrieve(self, query: str) -> list[SourceChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        if not self._collection_exists():
            return []

        vector = self._get_embedder().encode(
            normalized_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        results = self._search_points(vector.tolist())
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
                },
            )
            for hit in results
        ]
        if self._reranker_provider is not None and chunks:
            chunks = await self._rerank(normalized_query, chunks)
        return chunks[: self._top_k]

    def _search_points(self, query_vector: list[float]):
        if hasattr(self._qdrant_client, "search"):
            return self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=max(self._top_k, self._top_k * self._candidate_multiplier),
                score_threshold=self._score_threshold,
                with_payload=True,
                with_vectors=False,
            )

        query_response = self._qdrant_client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=max(self._top_k, self._top_k * self._candidate_multiplier),
            score_threshold=self._score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return query_response.points

    def _get_embedder(self):
        return self._embedder_provider.get_model()

    async def _rerank(self, query: str, chunks: list[SourceChunk]) -> list[SourceChunk]:
        reranker = self._reranker_provider.get_model()
        pairs = [(query, f"{chunk.title}\n{chunk.content}") for chunk in chunks]
        scores = await asyncio.to_thread(reranker.predict, pairs)
        reranked = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
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
    def reranker_enabled(self) -> bool:
        return self._reranker_provider is not None

    def update_runtime_config(
        self,
        *,
        top_k: int,
        score_threshold: float,
        candidate_multiplier: int,
        reranker_enabled: bool,
        reranker_provider: CrossEncoderProvider | None = None,
    ) -> None:
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._candidate_multiplier = candidate_multiplier
        self._reranker_provider = reranker_provider if reranker_enabled else None
