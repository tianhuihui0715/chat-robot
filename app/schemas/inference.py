from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatMessage, IntentDecision, SourceChunk


class InferenceGenerateRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    intent: IntentDecision
    sources: list[SourceChunk] = Field(default_factory=list)
    max_new_tokens: int | None = Field(default=None, ge=1)


class InferenceGenerateResponse(BaseModel):
    answer: str
    model_name: str | None = None


class InferenceHealthResponse(BaseModel):
    status: Literal["ok"]
    runtime_mode: Literal["mock", "local_hf"]
    model_loaded: bool
    model_name: str | None = None
