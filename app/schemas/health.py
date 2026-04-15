from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    runtime_mode: Literal["mock", "local_hf"]
    queued_requests: int
    knowledge_documents: int
