from app.schemas.chat import ChatRequest, ChatResponse
from app.services.generator_service import GenerationRequest, QueuedGenerationService
from app.services.intent_service import IntentService
from app.services.retriever_service import RetrieverService


class ChatPipeline:
    def __init__(
        self,
        intent_service: IntentService,
        retriever_service: RetrieverService,
        generation_service: QueuedGenerationService,
    ) -> None:
        self._intent_service = intent_service
        self._retriever_service = retriever_service
        self._generation_service = generation_service

    async def run(self, request: ChatRequest) -> ChatResponse:
        decision = await self._intent_service.decide(request.messages)
        sources = []

        if decision.need_rag:
            sources = await self._retriever_service.retrieve(decision.rewrite_query)

        answer = await self._generation_service.generate(
            GenerationRequest(
                messages=request.messages,
                intent=decision,
                sources=sources,
            )
        )

        return ChatResponse(
            answer=answer,
            intent=decision,
            sources=sources,
            queue_size=self._generation_service.queue_size,
        )
