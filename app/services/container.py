from dataclasses import dataclass, field
import asyncio

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.persistence.knowledge_ingest_store import KnowledgeIngestStore
from app.persistence.trace_store import SQLTraceStore
from app.schemas.admin import RAGCompareVariant, RAGEvaluationCase, RAGRuntimeConfig
from app.schemas.chat import ChatMessage
from app.services.bm25_index_store import SQLiteBM25IndexStore
from app.services.chat_pipeline import ChatPipeline
from app.services.embedding_service import CrossEncoderProvider, SentenceTransformerProvider
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
from app.services.knowledge_ingest_service import KnowledgeIngestService
from app.services.rag_lab_service import RAGLabService
from app.services.rag_snapshot_service import RAGSnapshotService
from app.services.retriever_service import (
    BM25RetrieverService,
    HybridRetrieverService,
    InMemoryRetrieverService,
    QdrantRetrieverService,
    RetrievalMode,
    RetrieverService,
)
from app.services.trace_service import LangSmithObserver, TraceService


@dataclass
class ServiceContainer:
    settings: Settings
    knowledge_base: KnowledgeBase
    knowledge_ingest_service: KnowledgeIngestService
    infra_service: InfraService
    intent_service: IntentService
    retriever_service: RetrieverService
    generation_service: QueuedGenerationService
    trace_service: TraceService
    rag_snapshot_service: RAGSnapshotService
    chat_pipeline: ChatPipeline
    rag_lab_service: RAGLabService
    embedder_provider: SentenceTransformerProvider | None = None
    reranker_provider: CrossEncoderProvider | None = None
    bm25_index_store: SQLiteBM25IndexStore | None = None
    _variant_config_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def start(self) -> None:
        self.infra_service.setup()
        self.infra_service.ensure_minio_bucket()
        await self.knowledge_ingest_service.start()
        self.trace_service.setup()
        if self.bm25_index_store is not None:
            self.bm25_index_store.setup()
            if (
                isinstance(self.knowledge_base, QdrantKnowledgeBase)
                and self.settings.rag_retrieval_mode in {"bm25", "hybrid"}
                and self.bm25_index_store.count_chunks() == 0
            ):
                await asyncio.to_thread(self.knowledge_base.sync_lexical_index)
        if self.embedder_provider is not None:
            await asyncio.to_thread(self.embedder_provider.preload)
        if self.reranker_provider is not None:
            await asyncio.to_thread(self.reranker_provider.preload)
        await self.intent_service.start()
        await self.generation_service.start()

    async def stop(self) -> None:
        await self.generation_service.stop()
        await self.intent_service.stop()
        await self.knowledge_ingest_service.stop()
        await self.trace_service.shutdown()
        self.infra_service.shutdown()

    def get_rag_runtime_config(self) -> RAGRuntimeConfig:
        top_k = self.settings.rag_top_k
        score_threshold = self.settings.rag_score_threshold
        candidate_multiplier = 3
        rerank_candidate_limit = self.settings.rag_rerank_candidate_limit
        reranker_enabled = self.reranker_provider is not None
        retrieval_mode: RetrievalMode = self.settings.rag_retrieval_mode
        bm25_top_k = self.settings.rag_bm25_top_k
        bm25_title_boost = self.settings.rag_bm25_title_boost
        rrf_k = self.settings.rag_rrf_k
        rrf_min_score = self.settings.rag_rrf_min_score

        if isinstance(self.retriever_service, QdrantRetrieverService):
            top_k = self.retriever_service.top_k
            score_threshold = self.retriever_service.score_threshold
            candidate_multiplier = self.retriever_service.candidate_multiplier
            rerank_candidate_limit = self.retriever_service.rerank_candidate_limit
            reranker_enabled = self.retriever_service.reranker_enabled
        elif isinstance(self.retriever_service, HybridRetrieverService):
            top_k = self.retriever_service.top_k
            score_threshold = self.retriever_service.score_threshold
            candidate_multiplier = self.retriever_service.candidate_multiplier
            rerank_candidate_limit = self.retriever_service.rerank_candidate_limit
            reranker_enabled = self.retriever_service.reranker_enabled
            retrieval_mode = self.retriever_service.retrieval_mode
            bm25_top_k = self.retriever_service.bm25_top_k
            bm25_title_boost = self.retriever_service.bm25_title_boost
            rrf_k = self.retriever_service.rrf_k
            rrf_min_score = self.retriever_service.rrf_min_score

        chunk_size = self.settings.rag_chunk_size
        chunk_overlap = self.settings.rag_chunk_overlap
        if isinstance(self.knowledge_base, QdrantKnowledgeBase):
            chunk_size = self.knowledge_base.chunk_size
            chunk_overlap = self.knowledge_base.chunk_overlap

        return RAGRuntimeConfig(
            top_k=top_k,
            score_threshold=score_threshold,
            candidate_multiplier=candidate_multiplier,
            rerank_candidate_limit=rerank_candidate_limit,
            retrieval_mode=retrieval_mode,
            bm25_top_k=bm25_top_k,
            bm25_title_boost=bm25_title_boost,
            rrf_k=rrf_k,
            rrf_min_score=rrf_min_score,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reranker_enabled=reranker_enabled,
            llm_temperature=self.settings.llm_temperature,
        )

    def update_rag_runtime_config(
        self,
        *,
        top_k: int,
        score_threshold: float,
        candidate_multiplier: int,
        rerank_candidate_limit: int,
        retrieval_mode: RetrievalMode,
        bm25_top_k: int,
        bm25_title_boost: float,
        rrf_k: int,
        rrf_min_score: float,
        chunk_size: int,
        chunk_overlap: int,
        reranker_enabled: bool,
        llm_temperature: float,
    ) -> RAGRuntimeConfig:
        self.settings.rag_top_k = top_k
        self.settings.rag_score_threshold = score_threshold
        self.settings.rag_rerank_candidate_limit = rerank_candidate_limit
        self.settings.rag_retrieval_mode = retrieval_mode
        self.settings.rag_bm25_top_k = bm25_top_k
        self.settings.rag_bm25_title_boost = bm25_title_boost
        self.settings.rag_rrf_k = rrf_k
        self.settings.rag_rrf_min_score = rrf_min_score
        self.settings.rag_chunk_size = chunk_size
        self.settings.rag_chunk_overlap = chunk_overlap
        self.settings.llm_temperature = llm_temperature
        self.chat_pipeline.update_generation_temperature(llm_temperature)

        if isinstance(self.retriever_service, QdrantRetrieverService):
            self.retriever_service.update_runtime_config(
                top_k=top_k,
                score_threshold=score_threshold,
                candidate_multiplier=candidate_multiplier,
                rerank_candidate_limit=rerank_candidate_limit,
                reranker_enabled=reranker_enabled,
                reranker_provider=self.reranker_provider,
            )
        elif isinstance(self.retriever_service, HybridRetrieverService):
            self.retriever_service.update_runtime_config(
                top_k=top_k,
                score_threshold=score_threshold,
                candidate_multiplier=candidate_multiplier,
                rerank_candidate_limit=rerank_candidate_limit,
                rrf_min_score=rrf_min_score,
                reranker_enabled=reranker_enabled,
                reranker_provider=self.reranker_provider,
                retrieval_mode=retrieval_mode,
                bm25_top_k=bm25_top_k,
                bm25_title_boost=bm25_title_boost,
                rrf_k=rrf_k,
            )
        if isinstance(self.knowledge_base, QdrantKnowledgeBase):
            self.knowledge_base.update_chunking(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return self.get_rag_runtime_config()

    async def compare_rag_variants(self, query: str, variants: list[RAGCompareVariant], generate_answer: bool):
        from app.schemas.admin import RAGCompareResponse, RAGCompareResult

        if not isinstance(self.retriever_service, (QdrantRetrieverService, HybridRetrieverService)):
            return RAGCompareResponse(query=query, results=[])

        async with self._variant_config_lock:
            original = self.get_rag_runtime_config()
            results: list[RAGCompareResult] = []
            try:
                for variant in variants:
                    answer, sources = await self._run_variant_query(
                        query=query,
                        variant=variant,
                        generate_answer=generate_answer,
                    )
                    results.append(
                        RAGCompareResult(
                            name=variant.name,
                            answer=answer,
                            sources=sources,
                        )
                    )
            finally:
                self._restore_retriever_runtime_config(original)
        return RAGCompareResponse(query=query, results=results)

    async def evaluate_rag_variants(
        self,
        cases: list[RAGEvaluationCase],
        variants: list[RAGCompareVariant],
        generate_answer: bool,
    ):
        from app.schemas.admin import (
            RAGEvaluationCaseResult,
            RAGEvaluationResponse,
            RAGEvaluationSummary,
            RAGEvaluationVariantResult,
        )

        if not isinstance(self.retriever_service, (QdrantRetrieverService, HybridRetrieverService)):
            return RAGEvaluationResponse(cases=[], summaries=[])

        async with self._variant_config_lock:
            original = self.get_rag_runtime_config()
            case_results: list[RAGEvaluationCaseResult] = []
            totals: dict[str, dict[str, float]] = {
                variant.name: {"cases": 0, "source_hits": 0, "answer_hits": 0, "returned_sources": 0}
                for variant in variants
            }

            try:
                for case in cases:
                    variant_results: list[RAGEvaluationVariantResult] = []
                    for variant in variants:
                        answer, sources = await self._run_variant_query(
                            query=case.query,
                            variant=variant,
                            generate_answer=generate_answer,
                        )
                        matched_sources = self._match_expected_sources(case.expected_sources, sources)
                        matched_answer_keywords = self._match_expected_answer_keywords(
                            case.expected_answer_keywords,
                            answer or "",
                        )
                        source_hit = bool(case.expected_sources) and len(matched_sources) == len(case.expected_sources)
                        answer_hit = bool(case.expected_answer_keywords) and (
                            len(matched_answer_keywords) == len(case.expected_answer_keywords)
                        )
                        if not case.expected_sources:
                            source_hit = len(sources) > 0
                        if not case.expected_answer_keywords:
                            answer_hit = bool(answer)

                        totals[variant.name]["cases"] += 1
                        totals[variant.name]["source_hits"] += 1 if source_hit else 0
                        totals[variant.name]["answer_hits"] += 1 if answer_hit else 0
                        totals[variant.name]["returned_sources"] += len(sources)

                        variant_results.append(
                            RAGEvaluationVariantResult(
                                name=variant.name,
                                answer=answer,
                                sources=sources,
                                source_hit=source_hit,
                                answer_hit=answer_hit,
                                matched_sources=matched_sources,
                                matched_answer_keywords=matched_answer_keywords,
                            )
                        )
                    case_results.append(
                        RAGEvaluationCaseResult(
                            query=case.query,
                            expected_sources=case.expected_sources,
                            expected_answer_keywords=case.expected_answer_keywords,
                            variants=variant_results,
                        )
                    )
            finally:
                self._restore_retriever_runtime_config(original)

        summaries = [
            RAGEvaluationSummary(
                name=variant.name,
                total_cases=int(totals[variant.name]["cases"]),
                source_hit_cases=int(totals[variant.name]["source_hits"]),
                answer_hit_cases=int(totals[variant.name]["answer_hits"]),
                source_hit_rate=(
                    totals[variant.name]["source_hits"] / totals[variant.name]["cases"]
                    if totals[variant.name]["cases"]
                    else 0.0
                ),
                answer_hit_rate=(
                    totals[variant.name]["answer_hits"] / totals[variant.name]["cases"]
                    if totals[variant.name]["cases"]
                    else 0.0
                ),
                average_returned_sources=(
                    totals[variant.name]["returned_sources"] / totals[variant.name]["cases"]
                    if totals[variant.name]["cases"]
                    else 0.0
                ),
            )
            for variant in variants
        ]
        return RAGEvaluationResponse(cases=case_results, summaries=summaries)

    async def apply_rag_lab_variant(self, session_id: str, variant_id: str):
        from app.schemas.admin import RAGLabApplyResponse

        variant = await self.rag_lab_service.get_variant(session_id, variant_id)
        if variant is None:
            raise KeyError(f"Unknown RAG lab variant: {session_id}/{variant_id}")

        if variant.rerank_k > 0:
            top_k = variant.rerank_k
            candidate_multiplier = max(1, -(-variant.retrieval_k // max(variant.rerank_k, 1)))
            reranker_enabled = True
        else:
            top_k = variant.retrieval_k
            candidate_multiplier = 1
            reranker_enabled = False

        overlap_chars = int(round(variant.chunk_size * variant.chunk_overlap_ratio))
        current = self.get_rag_runtime_config()
        applied = self.update_rag_runtime_config(
            top_k=top_k,
            score_threshold=current.score_threshold,
            candidate_multiplier=candidate_multiplier,
            rerank_candidate_limit=variant.rerank_k or current.rerank_candidate_limit,
            retrieval_mode=variant.retrieval_mode,
            bm25_top_k=variant.bm25_top_k,
            bm25_title_boost=variant.bm25_title_boost,
            rrf_k=variant.rrf_k,
            rrf_min_score=current.rrf_min_score,
            chunk_size=variant.chunk_size,
            chunk_overlap=overlap_chars,
            reranker_enabled=reranker_enabled,
            llm_temperature=variant.temperature,
        )
        return RAGLabApplyResponse(
            session_id=session_id,
            variant_id=variant_id,
            applied_config=applied,
            note="chunk 参数只会作用于后续新导入文档；现有知识库不会自动重切分。",
        )

    async def _run_variant_query(
        self,
        *,
        query: str,
        variant: RAGCompareVariant,
        generate_answer: bool,
    ) -> tuple[str | None, list]:
        from app.schemas.chat import IntentDecision
        from app.services.generator_service import GenerationRequest

        self.retriever_service.update_runtime_config(
            top_k=variant.top_k,
            score_threshold=variant.score_threshold,
            candidate_multiplier=variant.candidate_multiplier,
            rerank_candidate_limit=variant.rerank_candidate_limit,
            rrf_min_score=variant.rrf_min_score,
            reranker_enabled=variant.reranker_enabled,
            reranker_provider=self.reranker_provider,
            retrieval_mode=variant.retrieval_mode,
            bm25_top_k=variant.bm25_top_k,
            bm25_title_boost=variant.bm25_title_boost,
            rrf_k=variant.rrf_k,
        )
        sources = await self.retriever_service.retrieve(query)
        answer = None
        if generate_answer:
            answer = await self.generation_service.generate(
                GenerationRequest(
                    messages=[ChatMessage(role="user", content=query)],
                    intent=IntentDecision(
                        intent="knowledge_qa",
                        need_rag=True,
                        rewrite_query=query,
                        rationale=f"Admin evaluation variant: {variant.name}",
                    ),
                    sources=sources,
                )
            )
        return answer, sources

    @staticmethod
    def _match_expected_sources(expected_sources: list[str], sources: list) -> list[str]:
        matched: list[str] = []
        haystacks = [
            " ".join(
                filter(
                    None,
                    [
                        source.title,
                        source.document_id,
                        source.content,
                    ],
                )
            ).lower()
            for source in sources
        ]
        for expected in expected_sources:
            normalized = expected.strip().lower()
            if normalized and any(normalized in haystack for haystack in haystacks):
                matched.append(expected)
        return matched

    @staticmethod
    def _match_expected_answer_keywords(expected_keywords: list[str], answer: str) -> list[str]:
        normalized_answer = answer.lower()
        matched: list[str] = []
        for keyword in expected_keywords:
            normalized = keyword.strip().lower()
            if normalized and normalized in normalized_answer:
                matched.append(keyword)
        return matched

    def _restore_retriever_runtime_config(self, config: RAGRuntimeConfig) -> None:
        self.retriever_service.update_runtime_config(
            top_k=config.top_k,
            score_threshold=config.score_threshold,
            candidate_multiplier=config.candidate_multiplier,
            rerank_candidate_limit=config.rerank_candidate_limit,
            rrf_min_score=config.rrf_min_score,
            reranker_enabled=config.reranker_enabled,
            reranker_provider=self.reranker_provider,
            retrieval_mode=config.retrieval_mode,
            bm25_top_k=config.bm25_top_k,
            bm25_title_boost=config.bm25_title_boost,
            rrf_k=config.rrf_k,
        )


def build_service_container(settings: Settings) -> ServiceContainer:
    infra_service = InfraService(settings=settings)
    embedder_provider: SentenceTransformerProvider | None = None
    reranker_provider: CrossEncoderProvider | None = None
    bm25_index_store: SQLiteBM25IndexStore | None = None
    if settings.qdrant_url and settings.embedding_model_path:
        qdrant_client = QdrantClient(url=settings.qdrant_url)
        embedder_provider = SentenceTransformerProvider(
            model_path=settings.embedding_model_path,
            device="cpu",
        )
        reranker_provider = (
            CrossEncoderProvider(
                model_path=settings.reranker_model_path,
                device="cpu",
            )
            if settings.reranker_model_path
            else None
        )
        bm25_index_store = SQLiteBM25IndexStore(settings.rag_lexical_index_path)
        knowledge_base = QdrantKnowledgeBase(
            qdrant_client=qdrant_client,
            embedder_provider=embedder_provider,
            collection_name=settings.rag_collection_name,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            bm25_index_store=bm25_index_store,
        )
        dense_retriever = QdrantRetrieverService(
            qdrant_client=qdrant_client,
            embedder_provider=embedder_provider,
            collection_name=settings.rag_collection_name,
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
            rerank_candidate_limit=settings.rag_rerank_candidate_limit,
        )
        bm25_retriever = BM25RetrieverService(
            index_store=bm25_index_store,
            top_k=settings.rag_bm25_top_k,
            title_boost=settings.rag_bm25_title_boost,
            rerank_candidate_limit=settings.rag_rerank_candidate_limit,
        )
        retriever_service = HybridRetrieverService(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
            candidate_multiplier=3,
            rerank_candidate_limit=settings.rag_rerank_candidate_limit,
            rrf_min_score=settings.rag_rrf_min_score,
            reranker_provider=reranker_provider,
            retrieval_mode=settings.rag_retrieval_mode,
            bm25_top_k=settings.rag_bm25_top_k,
            bm25_title_boost=settings.rag_bm25_title_boost,
            rrf_k=settings.rag_rrf_k,
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
    rag_snapshot_service = RAGSnapshotService()
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
    knowledge_ingest_store = KnowledgeIngestStore(settings.trace_store_dsn)
    knowledge_ingest_service = KnowledgeIngestService(
        knowledge_base=knowledge_base,
        store=knowledge_ingest_store,
    )
    chat_pipeline = ChatPipeline(
        intent_service=intent_service,
        knowledge_base=knowledge_base,
        retriever_service=retriever_service,
        generation_service=generation_service,
        trace_service=trace_service,
        rag_snapshot_service=rag_snapshot_service,
        generation_temperature=settings.llm_temperature,
    )
    rag_lab_service = RAGLabService(
        embedder_provider=embedder_provider,
        reranker_provider=reranker_provider,
        generation_service=generation_service,
    )
    return ServiceContainer(
        settings=settings,
        knowledge_base=knowledge_base,
        knowledge_ingest_service=knowledge_ingest_service,
        infra_service=infra_service,
        intent_service=intent_service,
        retriever_service=retriever_service,
        generation_service=generation_service,
        trace_service=trace_service,
        rag_snapshot_service=rag_snapshot_service,
        chat_pipeline=chat_pipeline,
        rag_lab_service=rag_lab_service,
        embedder_provider=embedder_provider,
        reranker_provider=reranker_provider,
        bm25_index_store=bm25_index_store,
    )
