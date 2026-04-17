from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TraceStepItem(BaseModel):
    step_id: str
    request_id: str
    step_type: str
    step_order: int
    status: str
    latency_ms: int | None = None
    record_ref_type: str | None = None
    record_ref_id: str | None = None
    langsmith_run_id: str | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class TraceIntentRecord(BaseModel):
    intent_record_id: str
    request_id: str
    input_text: str
    intent: str
    need_rag: bool
    rewrite_query: str
    model_output: dict
    created_at: datetime


class TraceRetrievalRecord(BaseModel):
    retrieval_record_id: str
    request_id: str
    query: str
    retrieved_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class TraceGenerationRecord(BaseModel):
    generation_record_id: str
    request_id: str
    user_input: str
    used_source_ids: list[str] = Field(default_factory=list)
    llm_output: str
    created_at: datetime


class TraceSummary(BaseModel):
    request_id: str
    session_id: str | None = None
    langsmith_trace_id: str | None = None
    user_input: str
    intent: str | None = None
    need_rag: bool | None = None
    status: str
    total_latency_ms: int | None = None
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    step_count: int = 0


class TraceDetail(TraceSummary):
    final_output: str | None = None
    steps: list[TraceStepItem] = Field(default_factory=list)
    intent_record: TraceIntentRecord | None = None
    retrieval_record: TraceRetrievalRecord | None = None
    generation_record: TraceGenerationRecord | None = None


class TraceListResponse(BaseModel):
    items: list[TraceSummary] = Field(default_factory=list)
    total: int
    page: int
    page_size: int
