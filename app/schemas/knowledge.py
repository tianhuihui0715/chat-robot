from typing import Literal

from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class KnowledgeDocumentSummary(BaseModel):
    document_id: str
    title: str
    metadata: dict[str, str] = Field(default_factory=dict)


class KnowledgeDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    vector_deleted: bool
    relational_deleted: bool
    synchronized: bool


class KnowledgeIngestRequest(BaseModel):
    documents: list[KnowledgeDocument] = Field(min_length=1)


class KnowledgeIngestResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    submitted_documents: int
    processed_documents: int = 0
    ingested_count: int = 0
    total_chunks: int | None = None
    processed_chunks: int = 0
    current_stage: str = "queued"
    current_title: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    total_documents: int | None = None
    error: str | None = None


class KnowledgeIngestStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    submitted_documents: int
    processed_documents: int
    ingested_count: int
    total_chunks: int | None = None
    processed_chunks: int = 0
    current_stage: str = "queued"
    current_title: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    total_documents: int | None = None
    error: str | None = None
