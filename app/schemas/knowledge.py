from typing import Literal

from pydantic import BaseModel, Field, field_validator


class KnowledgeDocument(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    knowledge_base_id: str = "default"
    knowledge_base_name: str = "默认知识库"

    @field_validator("content")
    @classmethod
    def reject_binary_office_payload(cls, value: str) -> str:
        # .docx/.xlsx/.pptx files are zip containers. If they reach the JSON
        # ingest endpoint as text, they create thousands of meaningless chunks.
        if value.lstrip().startswith("PK\x03\x04"):
            raise ValueError("Office files must be uploaded through /knowledge/ingest/upload.")
        return value


class KnowledgeDocumentSummary(BaseModel):
    document_id: str
    title: str
    knowledge_base_id: str = "default"
    knowledge_base_name: str = "默认知识库"
    metadata: dict[str, str] = Field(default_factory=dict)


class KnowledgeDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    vector_deleted: bool
    relational_deleted: bool
    synchronized: bool


class KnowledgeIngestRequest(BaseModel):
    documents: list[KnowledgeDocument] = Field(min_length=1)
    knowledge_base_id: str = "default"
    knowledge_base_name: str = "默认知识库"


KnowledgeIngestStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class KnowledgeIngestResponse(BaseModel):
    job_id: str
    status: KnowledgeIngestStatus
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
    status: KnowledgeIngestStatus
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
