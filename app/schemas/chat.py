from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    session_id: str | None = None


class SourceChunk(BaseModel):
    document_id: str
    title: str
    content: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class IntentDecision(BaseModel):
    intent: Literal["chat", "knowledge_qa", "task", "follow_up", "reject"]
    need_rag: bool
    rewrite_query: str
    rationale: str


class ChatResponse(BaseModel):
    answer: str
    intent: IntentDecision
    sources: list[SourceChunk] = Field(default_factory=list)
    queue_size: int = 0
    request_id: str | None = None
