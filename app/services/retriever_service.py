import re
from typing import Protocol

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
        return {token for token in re.split(r"[\s,.;:!?，。；：！？/\\]+", text) if token}
