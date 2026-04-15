from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.container import ServiceContainer

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    container: ServiceContainer = Depends(get_container),
) -> ChatResponse:
    return await container.chat_pipeline.run(request)
