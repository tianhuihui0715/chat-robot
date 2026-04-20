from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.persistence.trace_store import SQLTraceStore
from app.services.chat_pipeline import ChatPipeline
from app.services.generator_service import (
    MockGenerationBackend,
    QueuedGenerationService,
    RemoteGenerationBackend,
)
from app.services.infra_service import InfraService
from app.services.intent_service import (
    IntentService,
    MockIntentService,
    RemoteIntentService,
)
from app.services.knowledge_base import InMemoryKnowledgeBase, KnowledgeBase, QdrantKnowledgeBase
from app.services.retriever_service import (
    InMemoryRetrieverService,
    QdrantRetrieverService,
    RetrieverService,
)
from app.services.trace_service import LangSmithObserver, TraceService


@dataclass
class ServiceContainer:
    settings: Settings
    knowledge_base: KnowledgeBase
    infra_service: InfraService
    intent_service: IntentService
    retriever_service: RetrieverService
    generation_service: QueuedGenerationService
    trace_service: TraceService
    chat_pipeline: ChatPipeline

    async def start(self) -> None:
        self.infra_service.setup()
        self.infra_service.ensure_minio_bucket()
        self.trace_service.setup()
        await self.intent_service.start()
        await self.generation_service.start()

    async def stop(self) -> None:
        await self.generation_service.stop()
        await self.intent_service.stop()
        await self.trace_service.shutdown()
        self.infra_service.shutdown()


def build_service_container(settings: Settings) -> ServiceContainer:
    infra_service = InfraService(settings=settings)
    if settings.qdrant_url and settings.embedding_model_path:
        qdrant_client = QdrantClient(url=settings.qdrant_url)
        knowledge_base = QdrantKnowledgeBase(
            qdrant_client=qdrant_client,
            embedding_model_path=settings.embedding_model_path,
            collection_name=settings.rag_collection_name,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )
        retriever_service = QdrantRetrieverService(
            qdrant_client=qdrant_client,
            embedding_model_path=settings.embedding_model_path,
            collection_name=settings.rag_collection_name,
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
        )
    else:
        knowledge_base = InMemoryKnowledgeBase()
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
        intent_service: IntentService = RemoteIntentService(
            base_url=settings.inference_service_url,
            timeout_seconds=settings.inference_timeout_seconds,
        )
        generation_backend = RemoteGenerationBackend(
            base_url=settings.inference_service_url,
            timeout_seconds=settings.inference_timeout_seconds,
            max_new_tokens=settings.llm_max_new_tokens,
        )
    else:
        intent_service = MockIntentService()
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
        infra_service=infra_service,
        intent_service=intent_service,
        retriever_service=retriever_service,
        generation_service=generation_service,
        trace_service=trace_service,
        chat_pipeline=chat_pipeline,
    )
