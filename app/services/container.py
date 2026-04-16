from dataclasses import dataclass

from app.core.config import Settings
from app.persistence.trace_store import SQLTraceStore
from app.services.chat_pipeline import ChatPipeline
from app.services.generator_service import (
    MockGenerationBackend,
    QueuedGenerationService,
    RemoteGenerationBackend,
)
from app.services.intent_service import MockIntentService
from app.services.knowledge_base import InMemoryKnowledgeBase
from app.services.retriever_service import InMemoryRetrieverService
from app.services.trace_service import LangSmithObserver, TraceService


@dataclass
class ServiceContainer:
    settings: Settings
    knowledge_base: InMemoryKnowledgeBase
    intent_service: MockIntentService
    retriever_service: InMemoryRetrieverService
    generation_service: QueuedGenerationService
    trace_service: TraceService
    chat_pipeline: ChatPipeline

    async def start(self) -> None:
        self.trace_service.setup()
        await self.generation_service.start()

    async def stop(self) -> None:
        await self.generation_service.stop()
        await self.trace_service.shutdown()


def build_service_container(settings: Settings) -> ServiceContainer:
    knowledge_base = InMemoryKnowledgeBase()
    intent_service = MockIntentService()
    retriever_service = InMemoryRetrieverService(
        knowledge_base=knowledge_base,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )
    trace_store = SQLTraceStore(settings.trace_store_dsn)
    langsmith_observer = LangSmithObserver(
        enabled=settings.langsmith_enabled,
        project_name=settings.langsmith_project,
        endpoint=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key,
    )
    trace_service = TraceService(
        store=trace_store,
        observer=langsmith_observer,
    )
    if settings.runtime_mode == "remote_inference":
        generation_backend = RemoteGenerationBackend(
            base_url=settings.inference_service_url,
            timeout_seconds=settings.inference_timeout_seconds,
            max_new_tokens=settings.llm_max_new_tokens,
        )
    else:
        generation_backend = MockGenerationBackend()

    generation_service = QueuedGenerationService(
        backend=generation_backend,
        maxsize=settings.gpu_queue_maxsize,
    )
    chat_pipeline = ChatPipeline(
        intent_service=intent_service,
        retriever_service=retriever_service,
        generation_service=generation_service,
        trace_service=trace_service,
    )
    return ServiceContainer(
        settings=settings,
        knowledge_base=knowledge_base,
        intent_service=intent_service,
        retriever_service=retriever_service,
        generation_service=generation_service,
        trace_service=trace_service,
        chat_pipeline=chat_pipeline,
    )
