from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    runtime_mode: Literal["mock", "remote_inference"]
    queued_requests: int
    knowledge_documents: int
    trace_requests: int
