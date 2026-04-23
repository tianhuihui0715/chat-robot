from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatMessage, IntentDecision, SourceChunk


class InferenceGenerateRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    intent: IntentDecision
    sources: list[SourceChunk] = Field(default_factory=list)
    max_new_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class InferenceGenerateResponse(BaseModel):
    answer: str
    model_name: str | None = None


class InferenceIntentRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class InferenceIntentResponse(BaseModel):
    decision: IntentDecision
    model_name: str | None = None
    raw_output: str | None = None


class InferenceHealthResponse(BaseModel):
    status: Literal["ok"]
    runtime_mode: Literal["mock", "local_hf"]
    model_loaded: bool
    model_name: str | None = None
    intent_model_loaded: bool | None = None
    intent_model_name: str | None = None
