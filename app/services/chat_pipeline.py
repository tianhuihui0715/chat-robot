import re
from time import perf_counter
from typing import AsyncIterator

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.generator_service import (
    GenerationRequest,
    QueuedGenerationService,
    build_generation_prompt_messages,
)
from app.services.intent_service import IntentService
from app.services.knowledge_base import KnowledgeBase
from app.services.rag_snapshot_service import RAGSnapshotService
from app.services.retriever_service import (
    RetrieverService,
    begin_retrieval_timings,
    end_retrieval_timings,
    get_retrieval_timings,
)
from app.services.trace_service import TraceService


class ChatPipeline:
    def __init__(
        self,
        intent_service: IntentService,
        knowledge_base: KnowledgeBase,
        retriever_service: RetrieverService,
        generation_service: QueuedGenerationService,
        trace_service: TraceService,
        rag_snapshot_service: RAGSnapshotService,
        generation_temperature: float = 0.0,
    ) -> None:
        self._intent_service = intent_service
        self._knowledge_base = knowledge_base
        self._retriever_service = retriever_service
        self._generation_service = generation_service
        self._trace_service = trace_service
        self._rag_snapshot_service = rag_snapshot_service
        self._generation_temperature = generation_temperature

    async def run(self, request: ChatRequest) -> ChatResponse:
        latest_user_message = self._extract_latest_user_message(request)

        with self._trace_service.request_trace(
            session_id=request.session_id,
            user_input=latest_user_message,
        ) as active_trace:
            snapshot_tokens = self._rag_snapshot_service.start_request(
                active_trace.request_id,
                user_query=latest_user_message,
                session_id=request.session_id,
            )
            self._rag_snapshot_service.end_request(snapshot_tokens)
            with self._trace_service.step_trace(
                active_trace=active_trace,
                step_type="intent_decision",
                run_type="chain",
                inputs={"user_input": latest_user_message},
            ) as intent_step:
                decision = await self._intent_service.decide(request.messages)
                decision = self._route_knowledge_base(decision)
                self._trace_service.complete_intent_step(
                    active_trace=active_trace,
                    active_step=intent_step,
                    input_text=latest_user_message,
                    intent=decision.intent,
                    need_rag=decision.need_rag,
                    rewrite_query=decision.rewrite_query,
                    rationale=decision.rationale,
                )
                self._rag_snapshot_service.update_intent(
                    active_trace.request_id,
                    intent_payload=decision.model_dump(mode="json"),
                )

            sources = []
            if decision.need_rag:
                with self._trace_service.step_trace(
                    active_trace=active_trace,
                    step_type="retrieval",
                    run_type="retriever",
                    inputs={"query": decision.rewrite_query},
                ) as retrieval_step:
                    timing_token = begin_retrieval_timings()
                    retrieval_snapshot_tokens = self._rag_snapshot_service.activate_request(
                        active_trace.request_id
                    )
                    try:
                        sources = await self._retriever_service.retrieve(
                            decision.rewrite_query,
                            use_reranker=request.use_reranker,
                            knowledge_base_id=decision.knowledge_base_id,
                        )
                        retrieval_timings = get_retrieval_timings()
                    finally:
                        self._rag_snapshot_service.end_request(retrieval_snapshot_tokens)
                        end_retrieval_timings(timing_token)
                    self._trace_service.complete_retrieval_step(
                        active_step=retrieval_step,
                        request_id=active_trace.request_id,
                        query=decision.rewrite_query,
                        retrieved_ids=[source.document_id for source in sources],
                    )
                    self._record_retrieval_timing_steps(active_trace, retrieval_timings)

            with self._trace_service.step_trace(
                active_trace=active_trace,
                step_type="llm_generation",
                run_type="llm",
                inputs={"user_input": latest_user_message},
            ) as generation_step:
                generation_request = GenerationRequest(
                    messages=request.messages,
                    intent=decision,
                    sources=sources,
                    temperature=self._generation_temperature,
                )
                self._rag_snapshot_service.update_generation(
                    active_trace.request_id,
                    generation_payload={
                        "prompt_messages": build_generation_prompt_messages(generation_request),
                        "sources": [source.model_dump(mode="json") for source in sources],
                    },
                )
                raw_answer = await self._generation_service.generate(generation_request)
                answer = _ensure_rag_citations(raw_answer, sources)
                self._rag_snapshot_service.update_generation(
                    active_trace.request_id,
                    generation_payload={
                        "raw_llm_output": raw_answer,
                        "final_output": answer,
                    },
                )
                self._trace_service.complete_generation_step(
                    active_step=generation_step,
                    request_id=active_trace.request_id,
                    user_input=latest_user_message,
                    used_source_ids=[source.document_id for source in sources],
                    llm_output=answer,
                )

            self._trace_service.complete_request_trace(
                active_trace=active_trace,
                intent=decision.intent,
                need_rag=decision.need_rag,
                final_output=answer,
            )

            return ChatResponse(
                answer=answer,
                intent=decision,
                sources=sources,
                queue_size=self._generation_service.queue_size,
                request_id=active_trace.request_id,
            )

    async def run_stream(self, request: ChatRequest) -> AsyncIterator[dict]:
        latest_user_message = self._extract_latest_user_message(request)

        with self._trace_service.request_trace(
            session_id=request.session_id,
            user_input=latest_user_message,
        ) as active_trace:
            snapshot_tokens = self._rag_snapshot_service.start_request(
                active_trace.request_id,
                user_query=latest_user_message,
                session_id=request.session_id,
            )
            self._rag_snapshot_service.end_request(snapshot_tokens)
            yield {"type": "status", "stage": "thinking", "message": "正在思考"}

            with self._trace_service.step_trace(
                active_trace=active_trace,
                step_type="intent_decision",
                run_type="chain",
                inputs={"user_input": latest_user_message},
            ) as intent_step:
                decision = await self._intent_service.decide(request.messages)
                decision = self._route_knowledge_base(decision)
                self._trace_service.complete_intent_step(
                    active_trace=active_trace,
                    active_step=intent_step,
                    input_text=latest_user_message,
                    intent=decision.intent,
                    need_rag=decision.need_rag,
                    rewrite_query=decision.rewrite_query,
                    rationale=decision.rationale,
                )
                self._rag_snapshot_service.update_intent(
                    active_trace.request_id,
                    intent_payload=decision.model_dump(mode="json"),
                )

            sources = []
            if decision.need_rag:
                with self._trace_service.step_trace(
                    active_trace=active_trace,
                    step_type="retrieval",
                    run_type="retriever",
                    inputs={"query": decision.rewrite_query},
                ) as retrieval_step:
                    timing_token = begin_retrieval_timings()
                    retrieval_snapshot_tokens = self._rag_snapshot_service.activate_request(
                        active_trace.request_id
                    )
                    try:
                        sources = await self._retriever_service.retrieve(
                            decision.rewrite_query,
                            use_reranker=request.use_reranker,
                            knowledge_base_id=decision.knowledge_base_id,
                        )
                        retrieval_timings = get_retrieval_timings()
                    finally:
                        self._rag_snapshot_service.end_request(retrieval_snapshot_tokens)
                        end_retrieval_timings(timing_token)
                    self._trace_service.complete_retrieval_step(
                        active_step=retrieval_step,
                        request_id=active_trace.request_id,
                        query=decision.rewrite_query,
                        retrieved_ids=[source.document_id for source in sources],
                    )
                    self._record_retrieval_timing_steps(active_trace, retrieval_timings)

            answer_parts: list[str] = []
            with self._trace_service.step_trace(
                active_trace=active_trace,
                step_type="llm_generation",
                run_type="llm",
                inputs={"user_input": latest_user_message},
            ) as generation_step:
                generation_started_at = perf_counter()
                first_token_recorded = False
                generation_request = GenerationRequest(
                    messages=request.messages,
                    intent=decision,
                    sources=sources,
                    temperature=self._generation_temperature,
                )
                self._rag_snapshot_service.update_generation(
                    active_trace.request_id,
                    generation_payload={
                        "prompt_messages": build_generation_prompt_messages(generation_request),
                        "sources": [source.model_dump(mode="json") for source in sources],
                    },
                )
                async for delta in self._generation_service.generate_stream(
                    generation_request
                ):
                    if delta and not first_token_recorded:
                        self._trace_service.record_timing_step(
                            active_trace=active_trace,
                            step_type="first_token_generation",
                            latency_ms=int((perf_counter() - generation_started_at) * 1000),
                        )
                        first_token_recorded = True
                    answer_parts.append(delta)
                    yield {"type": "delta", "delta": delta}

                raw_answer = "".join(answer_parts)
                answer = _ensure_rag_citations(raw_answer, sources)
                self._rag_snapshot_service.update_generation(
                    active_trace.request_id,
                    generation_payload={
                        "raw_llm_output": raw_answer,
                        "final_output": answer,
                    },
                )
                self._trace_service.complete_generation_step(
                    active_step=generation_step,
                    request_id=active_trace.request_id,
                    user_input=latest_user_message,
                    used_source_ids=[source.document_id for source in sources],
                    llm_output=answer,
                )

            self._trace_service.complete_request_trace(
                active_trace=active_trace,
                intent=decision.intent,
                need_rag=decision.need_rag,
                final_output=answer,
            )

            yield {
                "type": "response",
                "data": ChatResponse(
                    answer=answer,
                    intent=decision,
                    sources=sources,
                    queue_size=self._generation_service.queue_size,
                    request_id=active_trace.request_id,
                ).model_dump(mode="json"),
            }

    @staticmethod
    def _extract_latest_user_message(request: ChatRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content.strip()
        return request.messages[-1].content.strip()

    def update_generation_temperature(self, temperature: float) -> None:
        self._generation_temperature = temperature

    def _route_knowledge_base(self, decision):
        if not decision.need_rag:
            return decision

        selected = self._select_knowledge_base(decision.rewrite_query)
        if selected is None:
            return decision

        knowledge_base_id, knowledge_base_name = selected
        return decision.model_copy(
            update={
                "knowledge_base_id": knowledge_base_id,
                "knowledge_base_name": knowledge_base_name,
                "rationale": (
                    f"{decision.rationale} 已根据问题内容路由到知识库："
                    f"{knowledge_base_name} ({knowledge_base_id})。"
                ),
            }
        )

    def _select_knowledge_base(self, query: str) -> tuple[str, str] | None:
        documents = self._knowledge_base.list_documents()
        if not documents:
            return None

        catalog: dict[str, dict[str, object]] = {}
        for document in documents:
            knowledge_base_id = document.metadata.get("knowledge_base_id", "default")
            knowledge_base_name = document.metadata.get("knowledge_base_name", "默认知识库")
            bucket = catalog.setdefault(
                knowledge_base_id,
                {"name": knowledge_base_name, "titles": []},
            )
            bucket["titles"].append(document.title)

        if len(catalog) == 1:
            knowledge_base_id, payload = next(iter(catalog.items()))
            return knowledge_base_id, str(payload["name"])

        normalized_query = query.lower().replace(" ", "")
        best: tuple[int, str, str] | None = None
        for knowledge_base_id, payload in catalog.items():
            knowledge_base_name = str(payload["name"])
            score = 0
            compact_id = knowledge_base_id.lower().replace(" ", "")
            compact_name = knowledge_base_name.lower().replace(" ", "")
            if compact_id and compact_id in normalized_query:
                score += 20
            if compact_name and compact_name in normalized_query:
                score += 20
            for token in _extract_route_tokens(knowledge_base_name):
                if token in normalized_query:
                    score += 3
            for title in payload["titles"]:
                compact_title = str(title).lower().replace(" ", "")
                if compact_title and compact_title in normalized_query:
                    score += 8
                for token in _extract_route_tokens(str(title)):
                    if token in normalized_query:
                        score += 1
            if best is None or score > best[0]:
                best = (score, knowledge_base_id, knowledge_base_name)

        if best and best[0] > 0:
            return best[1], best[2]
        if "default" in catalog:
            return "default", str(catalog["default"]["name"])
        return None

    def _record_retrieval_timing_steps(self, active_trace, timings: dict[str, int]) -> None:
        self._trace_service.record_timing_step(
            active_trace=active_trace,
            step_type="embedding",
            latency_ms=timings.get("embedding"),
        )
        self._trace_service.record_timing_step(
            active_trace=active_trace,
            step_type="qdrant_search",
            latency_ms=timings.get("qdrant_search"),
        )
        self._trace_service.record_timing_step(
            active_trace=active_trace,
            step_type="rerank",
            latency_ms=timings.get("rerank"),
        )


def _extract_route_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_+-]+|[\u4e00-\u9fff]{2,}", text)
        if len(token.strip()) >= 2
    }


def _ensure_rag_citations(answer: str, sources: list) -> str:
    if not answer.strip() or not sources:
        return answer
    if re.search(r"【\d+】", answer):
        return answer

    citation_tokens = [
        f"【{source.metadata.get('citation_index', str(index))}】"
        for index, source in enumerate(sources, start=1)
    ]
    if not citation_tokens:
        return answer

    lines = answer.splitlines()
    item_indexes = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*(?:[-*]\s+|\d+[.、]\s+|[一二三四五六七八九十]+[、.]\s*)", line)
    ]
    if item_indexes:
        for offset, line_index in enumerate(item_indexes):
            token = citation_tokens[min(offset, len(citation_tokens) - 1)]
            lines[line_index] = f"{lines[line_index].rstrip()}{token}"
        return "\n".join(lines)

    return f"{answer.rstrip()}{''.join(citation_tokens)}"
