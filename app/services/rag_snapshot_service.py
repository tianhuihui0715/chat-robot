from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from app.schemas.chat import SourceChunk


MAX_SNAPSHOT_CHUNKS = 80
_active_snapshot: ContextVar["RAGSnapshotService | None"] = ContextVar(
    "active_rag_snapshot",
    default=None,
)
_active_request_id: ContextVar[str | None] = ContextVar("active_rag_snapshot_request_id", default=None)


class RAGSnapshotService:
    """In-memory request snapshots for debugging one request end-to-end."""

    def __init__(self, max_snapshots: int = 200) -> None:
        self._max_snapshots = max_snapshots
        self._snapshots: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = RLock()

    def start_request(self, request_id: str, *, user_query: str, session_id: str | None) -> tuple[Token, Token]:
        snapshot = {
            "request_id": request_id,
            "session_id": session_id,
            "user_query": user_query,
            "created_at": _utcnow(),
            "intent": None,
            "retrieval": {
                "query": None,
                "stages": [],
            },
            "generation": {
                "prompt_messages": [],
                "sources": [],
                "raw_llm_output": "",
                "final_output": "",
            },
        }
        with self._lock:
            self._snapshots[request_id] = snapshot
            self._snapshots.move_to_end(request_id)
            while len(self._snapshots) > self._max_snapshots:
                self._snapshots.popitem(last=False)

        request_token = _active_request_id.set(request_id)
        service_token = _active_snapshot.set(self)
        return service_token, request_token

    def activate_request(self, request_id: str) -> tuple[Token, Token]:
        request_token = _active_request_id.set(request_id)
        service_token = _active_snapshot.set(self)
        return service_token, request_token

    def end_request(self, tokens: tuple[Token, Token]) -> None:
        service_token, request_token = tokens
        _active_snapshot.reset(service_token)
        _active_request_id.reset(request_token)

    def get_snapshot(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            snapshot = self._snapshots.get(request_id)
            if snapshot is None:
                return None
            self._snapshots.move_to_end(request_id)
            return snapshot

    def update_intent(self, request_id: str, *, intent_payload: dict[str, Any]) -> None:
        self._update(request_id, ["intent"], intent_payload)

    def update_retrieval_query(self, request_id: str, *, query: str) -> None:
        self._update(request_id, ["retrieval", "query"], query)

    def add_retrieval_stage(
        self,
        request_id: str,
        *,
        name: str,
        chunks: list[SourceChunk],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        stage = {
            "name": name,
            "metadata": metadata or {},
            "count": len(chunks),
            "chunks": [_chunk_to_snapshot(chunk, rank=index) for index, chunk in enumerate(chunks, start=1)][
                :MAX_SNAPSHOT_CHUNKS
            ],
        }
        with self._lock:
            snapshot = self._snapshots.get(request_id)
            if snapshot is None:
                return
            snapshot.setdefault("retrieval", {}).setdefault("stages", []).append(stage)

    def update_generation(self, request_id: str, *, generation_payload: dict[str, Any]) -> None:
        with self._lock:
            snapshot = self._snapshots.get(request_id)
            if snapshot is None:
                return
            snapshot.setdefault("generation", {}).update(generation_payload)

    def _update(self, request_id: str, path: list[str], value: Any) -> None:
        with self._lock:
            current = self._snapshots.get(request_id)
            if current is None:
                return
            for key in path[:-1]:
                current = current.setdefault(key, {})
            current[path[-1]] = value


def record_retrieval_query(query: str) -> None:
    service, request_id = _active()
    if service is None or request_id is None:
        return
    service.update_retrieval_query(request_id, query=query)


def record_retrieval_stage(
    name: str,
    chunks: list[SourceChunk],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    service, request_id = _active()
    if service is None or request_id is None:
        return
    service.add_retrieval_stage(request_id, name=name, chunks=chunks, metadata=metadata)


def _active() -> tuple[RAGSnapshotService | None, str | None]:
    return _active_snapshot.get(), _active_request_id.get()


def _chunk_to_snapshot(chunk: SourceChunk, *, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "chunk_id": chunk.metadata.get("chunk_id") or chunk.metadata.get("chunk_ids"),
        "score": chunk.score,
        "content": chunk.content,
        "metadata": dict(chunk.metadata),
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
