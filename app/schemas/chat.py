from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    session_id: str | None = None
    use_reranker: bool | None = None
    execution_strategy: Literal["off", "auto", "force"] = "auto"
    react_mode: Literal["off", "auto", "force"] | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_execution_strategy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("execution_strategy") in {None, ""} and data.get("react_mode") in {
            "off",
            "auto",
            "force",
        }:
            payload = dict(data)
            payload["execution_strategy"] = data["react_mode"]
            return payload
        return data


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
    execution_mode: Literal["direct", "rag", "plan_execute"] = "direct"
    should_clarify: bool = False
    clarify_question: str | None = None
    candidate_tools: list[str] = Field(default_factory=list)
    planner_hint: str | None = None
    knowledge_base_id: str | None = None
    knowledge_base_name: str | None = None


class PlanTask(BaseModel):
    task_id: str
    goal: str
    query: str
    depends_on: list[str] = Field(default_factory=list)


class SubtaskResult(BaseModel):
    task_id: str
    goal: str
    query: str
    depends_on: list[str] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)
    count: int | None = None
    summary: str | None = None
    notes: str | None = None
    confidence: float | None = None
    coverage_hint: str | None = None
    needs_retry: bool = False
    retry_count: int = 0
    source_ids: list[str] = Field(default_factory=list)


class AggregateResult(BaseModel):
    mode: Literal["union", "intersection", "compare", "dedupe_union", "rank", "group_by"]
    items: list[str] = Field(default_factory=list)
    grouped_items: dict[str, list[str]] = Field(default_factory=dict)
    ranked_task_ids: list[str] = Field(default_factory=list)
    left_task_id: str | None = None
    right_task_id: str | None = None
    left_count: int | None = None
    right_count: int | None = None
    winner_task_id: str | None = None
    notes: str | None = None
    confidence: float | None = None
    needs_retry: bool = False


class ChatResponse(BaseModel):
    answer: str
    intent: IntentDecision
    sources: list[SourceChunk] = Field(default_factory=list)
    planner: dict[str, object] | None = None
    subtask_results: list[SubtaskResult] = Field(default_factory=list)
    aggregate_result: AggregateResult | None = None
    execution_steps: list[dict[str, object]] = Field(default_factory=list)
    queue_size: int = 0
    request_id: str | None = None
