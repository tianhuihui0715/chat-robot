from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.schemas.health import HealthResponse
from app.services.container import ServiceContainer

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    container: ServiceContainer = Depends(get_container),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        runtime_mode=container.settings.runtime_mode,
        queued_requests=container.generation_service.queue_size,
        knowledge_documents=container.knowledge_base.count,
    )
