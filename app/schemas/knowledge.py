from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class KnowledgeDocumentSummary(BaseModel):
    document_id: str
    title: str
    metadata: dict[str, str] = Field(default_factory=dict)


class KnowledgeIngestRequest(BaseModel):
    documents: list[KnowledgeDocument] = Field(min_length=1)


class KnowledgeIngestResponse(BaseModel):
    ingested_count: int
    document_ids: list[str]
    total_documents: int
