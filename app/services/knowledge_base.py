from dataclasses import dataclass
from uuid import uuid4

from app.schemas.knowledge import KnowledgeDocument


@dataclass
class KnowledgeRecord:
    document_id: str
    title: str
    content: str
    metadata: dict[str, str]


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
