from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from qdrant_client import QdrantClient, models

from app.schemas.knowledge import KnowledgeDocument


@dataclass
class KnowledgeRecord:
    document_id: str
    title: str
    content: str
    metadata: dict[str, str]


class KnowledgeBase(Protocol):
    @property
    def count(self) -> int:
        ...

    def add_documents(self, documents: list[KnowledgeDocument]) -> list[str]:
        ...

    def list_documents(self) -> list[KnowledgeRecord]:
        ...


class InMemoryKnowledgeBase:
    def __init__(self) -> None:
        self._documents: dict[str, KnowledgeRecord] = {}

    @property
    def count(self) -> int:
        return len(self._documents)

    def add_documents(self, documents: list[KnowledgeDocument]) -> list[str]:
        document_ids: list[str] = []
        for document in documents:
            document_id = uuid4().hex
            self._documents[document_id] = KnowledgeRecord(
                document_id=document_id,
                title=document.title,
                content=document.content,
                metadata=document.metadata,
            )
            document_ids.append(document_id)
        return document_ids

    def list_documents(self) -> list[KnowledgeRecord]:
        return list(self._documents.values())


class QdrantKnowledgeBase:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model_path: str,
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ) -> None:
        self._qdrant_client = qdrant_client
        self._embedding_model_path = embedding_model_path
        self._collection_name = collection_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedder = None
        self._collection_ready = False

    @property
    def count(self) -> int:
        return len(self.list_documents())

    def add_documents(self, documents: list[KnowledgeDocument]) -> list[str]:
        embedder = self._get_embedder()
        self._ensure_collection(embedder)

        document_ids: list[str] = []
        for document in documents:
            document_id = uuid4().hex
            document_ids.append(document_id)
            chunks = self._chunk_text(document.content)
            if not chunks:
                chunks = [document.content]

            embeddings = embedder.encode(
                chunks,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            points: list[models.PointStruct] = []
            for index, (chunk, vector) in enumerate(zip(chunks, embeddings), start=1):
                chunk_id = f"{document_id}:{index}"
                payload = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "title": document.title,
                    "content": chunk,
                    "metadata": {key: str(value) for key, value in document.metadata.items()},
                }
                points.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )

            self._qdrant_client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
        return document_ids

    def list_documents(self) -> list[KnowledgeRecord]:
        if not self._collection_exists():
            return []
        seen: dict[str, KnowledgeRecord] = {}
        offset = None

        while True:
            points, offset = self._qdrant_client.scroll(
                collection_name=self._collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                document_id = str(payload.get("document_id", ""))
                if not document_id or document_id in seen:
                    continue
                seen[document_id] = KnowledgeRecord(
                    document_id=document_id,
                    title=str(payload.get("title", "")),
                    content=str(payload.get("content", "")),
                    metadata={
                        key: str(value)
                        for key, value in (payload.get("metadata") or {}).items()
                    },
                )
            if offset is None:
                break

        return list(seen.values())

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self._embedding_model_path, device="cpu")
        return self._embedder

    def _ensure_collection(self, embedder) -> None:
        if self._collection_ready:
            return
        if not self._collection_exists():
            vector_size = int(embedder.get_sentence_embedding_dimension())
            self._qdrant_client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
            )
        self._collection_ready = True

    def _collection_exists(self) -> bool:
        collection_names = {
            collection.name for collection in self._qdrant_client.get_collections().collections
        }
        exists = self._collection_name in collection_names
        if exists:
            self._collection_ready = True
        return exists

    def _chunk_text(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        text_length = len(normalized)
        while start < text_length:
            end = min(start + self._chunk_size, text_length)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start = max(end - self._chunk_overlap, start + 1)
        return chunks
