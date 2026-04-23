from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

from qdrant_client import QdrantClient, models

from app.schemas.knowledge import KnowledgeDocument
from app.services.embedding_service import SentenceTransformerProvider


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

    def add_documents(
        self,
        documents: list[KnowledgeDocument],
        progress_callback: Callable[..., None] | None = None,
    ) -> list[str]:
        ...

    def list_documents(self) -> list[KnowledgeRecord]:
        ...

    def delete_document(self, document_id: str) -> bool:
        ...


class InMemoryKnowledgeBase:
    def __init__(self) -> None:
        self._documents: dict[str, KnowledgeRecord] = {}

    @property
    def count(self) -> int:
        return len(self._documents)

    def add_documents(
        self,
        documents: list[KnowledgeDocument],
        progress_callback: Callable[..., None] | None = None,
    ) -> list[str]:
        document_ids: list[str] = []
        processed_documents = 0
        for document in documents:
            if progress_callback is not None:
                progress_callback(
                    current_stage="processing",
                    current_title=document.title,
                    processed_documents=processed_documents,
                )
            document_id = uuid4().hex
            self._documents[document_id] = KnowledgeRecord(
                document_id=document_id,
                title=document.title,
                content=document.content,
                metadata=document.metadata,
            )
            document_ids.append(document_id)
            processed_documents += 1
            if progress_callback is not None:
                progress_callback(
                    current_stage="completed",
                    current_title=document.title,
                    processed_documents=processed_documents,
                )
        return document_ids

    def list_documents(self) -> list[KnowledgeRecord]:
        return list(self._documents.values())

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self._documents:
            return False
        del self._documents[document_id]
        return True


class QdrantKnowledgeBase:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedder_provider: SentenceTransformerProvider,
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ) -> None:
        self._qdrant_client = qdrant_client
        self._embedder_provider = embedder_provider
        self._collection_name = collection_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._collection_ready = False
        from threading import Lock

        self._write_lock = Lock()
        self._embedding_batch_size = 8

    @property
    def count(self) -> int:
        return len(self.list_documents())

    def add_documents(
        self,
        documents: list[KnowledgeDocument],
        progress_callback: Callable[..., None] | None = None,
    ) -> list[str]:
        with self._write_lock:
            embedder = self._get_embedder()
            self._ensure_collection(embedder)

            document_ids: list[str] = []
            processed_documents = 0
            total_chunks = 0
            processed_chunks = 0

            chunk_sets: list[list[str]] = []
            for document in documents:
                chunks = self._chunk_text(document.content)
                if not chunks:
                    chunks = [document.content]
                chunk_sets.append(chunks)
                total_chunks += len(chunks)

            if progress_callback is not None:
                progress_callback(
                    current_stage="chunking",
                    total_chunks=total_chunks,
                    processed_chunks=0,
                    processed_documents=0,
                )

            for document, chunks in zip(documents, chunk_sets):
                if progress_callback is not None:
                    progress_callback(
                        current_stage="embedding",
                        current_title=document.title,
                        processed_documents=processed_documents,
                        total_chunks=total_chunks,
                        processed_chunks=processed_chunks,
                    )
                document_id = uuid4().hex
                document_ids.append(document_id)

                vectors: list[list[float]] = []
                for batch_start in range(0, len(chunks), self._embedding_batch_size):
                    batch_chunks = chunks[batch_start : batch_start + self._embedding_batch_size]
                    batch_embeddings = embedder.encode(
                        batch_chunks,
                        batch_size=self._embedding_batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    vectors.extend(batch_embeddings.tolist())
                    processed_chunks += len(batch_chunks)
                    if progress_callback is not None:
                        progress_callback(
                            current_stage="embedding",
                            current_title=document.title,
                            processed_documents=processed_documents,
                            total_chunks=total_chunks,
                            processed_chunks=processed_chunks,
                        )

                points: list[models.PointStruct] = []
                if progress_callback is not None:
                    progress_callback(
                        current_stage="upserting",
                        current_title=document.title,
                        processed_documents=processed_documents,
                        total_chunks=total_chunks,
                        processed_chunks=processed_chunks,
                    )
                for index, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
                    chunk_id = f"{document_id}:{index}"
                    point_id = str(uuid5(NAMESPACE_URL, chunk_id))
                    payload = {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "title": document.title,
                        "content": chunk,
                        "metadata": {key: str(value) for key, value in document.metadata.items()},
                    }
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        )
                    )

                self._qdrant_client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                    wait=True,
                )
                processed_documents += 1
                if progress_callback is not None:
                    progress_callback(
                        current_stage="upserting",
                        current_title=document.title,
                        processed_documents=processed_documents,
                        total_chunks=total_chunks,
                        processed_chunks=processed_chunks,
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

    def delete_document(self, document_id: str) -> bool:
        if not self._collection_exists():
            return False

        matches = self._qdrant_client.count(
            collection_name=self._collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )
        if matches.count <= 0:
            return False

        self._qdrant_client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        return True

    def _get_embedder(self):
        return self._embedder_provider.get_model()

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

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    def update_chunking(self, *, chunk_size: int, chunk_overlap: int) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
