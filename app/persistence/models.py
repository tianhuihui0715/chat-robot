from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """SQLAlchemy base model."""


class RequestTrace(Base):
    __tablename__ = "request_traces"

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    langsmith_trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    need_rag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    final_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list["TraceStep"]] = relationship(
        back_populates="request_trace",
        cascade="all, delete-orphan",
    )


class TraceStep(Base):
    __tablename__ = "trace_steps"

    step_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("request_traces.request_id"),
        index=True,
    )
    step_type: Mapped[str] = mapped_column(String(64), index=True)
    step_order: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="running")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    record_ref_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    record_ref_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    langsmith_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    request_trace: Mapped["RequestTrace"] = relationship(back_populates="steps")


class IntentRecord(Base):
    __tablename__ = "intent_records"

    intent_record_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(64), nullable=False)
    need_rag: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rewrite_query: Mapped[str] = mapped_column(Text, nullable=False)
    model_output: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RetrievalRecord(Base):
    __tablename__ = "retrieval_records"

    retrieval_record_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GenerationRecord(Base):
    __tablename__ = "generation_records"

    generation_record_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    used_source_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    llm_output: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KnowledgeIngestJobRecord(Base):
    __tablename__ = "knowledge_ingest_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    submitted_documents: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_documents: Mapped[int] = mapped_column(Integer, default=0)
    ingested_count: Mapped[int] = mapped_column(Integer, default=0)
    total_documents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    current_stage: Mapped[str] = mapped_column(String(32), default="queued")
    current_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    documents: Mapped[list["KnowledgeIngestDocumentRecord"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )


class KnowledgeIngestDocumentRecord(Base):
    __tablename__ = "knowledge_ingest_documents"

    ingest_document_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_ingest_jobs.job_id"),
        index=True,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    document_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped["KnowledgeIngestJobRecord"] = relationship(back_populates="documents")
