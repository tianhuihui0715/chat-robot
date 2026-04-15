from dataclasses import dataclass

from app.core.config import Settings
from app.services.chat_pipeline import ChatPipeline
from app.services.generator_service import (
    MockGenerationBackend,
    QueuedGenerationService,
)
from app.services.intent_service import MockIntentService
from app.services.knowledge_base import InMemoryKnowledgeBase
from app.services.retriever_service import InMemoryRetrieverService


@dataclass
class ServiceContainer:
    settings: Settings
    knowledge_base: InMemoryKnowledgeBase
    intent_service: MockIntentService
    retriever_service: InMemoryRetrieverService
    generation_service: QueuedGenerationService
    chat_pipeline: ChatPipeline

    async def start(self) -> None:
        await self.generation_service.start()

    async def stop(self) -> None:
        await self.generation_service.stop()


def build_service_container(settings: Settings) -> ServiceContainer:
    knowledge_base = InMemoryKnowledgeBase()
    intent_service = MockIntentService()
    retriever_service = InMemoryRetrieverService(
        knowledge_base=knowledge_base,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )
    generation_service = QueuedGenerationService(
        backend=MockGenerationBackend(),
        maxsize=settings.gpu_queue_maxsize,
    )
    chat_pipeline = ChatPipeline(
        intent_service=intent_service,
        retriever_service=retriever_service,
        generation_service=generation_service,
    )
    return ServiceContainer(
        settings=settings,
        knowledge_base=knowledge_base,
        intent_service=intent_service,
        retriever_service=retriever_service,
        generation_service=generation_service,
        chat_pipeline=chat_pipeline,
    )
