from __future__ import annotations

import re
from typing import Protocol

from qdrant_client import QdrantClient

from app.schemas.chat import SourceChunk
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
        embedding_model_path: str,
        collection_name: str,
        top_k: int = 4,
        score_threshold: float = 0.1,
    ) -> None:
        self._qdrant_client = qdrant_client
        self._embedding_model_path = embedding_model_path
        self._collection_name = collection_name
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._embedder = None

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
        results = self._qdrant_client.search(
            collection_name=self._collection_name,
            query_vector=vector.tolist(),
            limit=self._top_k,
            score_threshold=self._score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [
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

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self._embedding_model_path, device="cpu")
        return self._embedder

    def _collection_exists(self) -> bool:
        collection_names = {
            collection.name for collection in self._qdrant_client.get_collections().collections
        }
        return self._collection_name in collection_names
