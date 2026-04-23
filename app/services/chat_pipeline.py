from typing import AsyncIterator

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.generator_service import GenerationRequest, QueuedGenerationService
from app.services.intent_service import IntentService
from app.services.retriever_service import RetrieverService
from app.services.trace_service import TraceService


class ChatPipeline:
    def __init__(
        self,
        intent_service: IntentService,
        retriever_service: RetrieverService,
        generation_service: QueuedGenerationService,
        trace_service: TraceService,
        generation_temperature: float = 0.0,
    ) -> None:
        self._intent_service = intent_service
        self._retriever_service = retriever_service
        self._generation_service = generation_service
        self._trace_service = trace_service
        self._generation_temperature = generation_temperature

    async def run(self, request: ChatRequest) -> ChatResponse:
        latest_user_message = self._extract_latest_user_message(request)

        with self._trace_service.request_trace(
            session_id=request.session_id,
            user_input=latest_user_message,
        ) as active_trace:
            with self._trace_service.step_trace(
                active_trace=active_trace,
                step_type="intent_decision",
                run_type="chain",
                inputs={"user_input": latest_user_message},
            ) as intent_step:
                decision = await self._intent_service.decide(request.messages)
                self._trace_service.complete_intent_step(
                    active_trace=active_trace,
                    active_step=intent_step,
                    input_text=latest_user_message,
                    intent=decision.intent,
                    need_rag=decision.need_rag,
                    rewrite_query=decision.rewrite_query,
                    rationale=decision.rationale,
                )

            sources = []
            if decision.need_rag:
                with self._trace_service.step_trace(
                    active_trace=active_trace,
                    step_type="retrieval",
                    run_type="retriever",
                    inputs={"query": decision.rewrite_query},
                ) as retrieval_step:
                    sources = await self._retriever_service.retrieve(decision.rewrite_query)
                    self._trace_service.complete_retrieval_step(
                        active_step=retrieval_step,
                        request_id=active_trace.request_id,
                        query=decision.rewrite_query,
                        retrieved_ids=[source.document_id for source in sources],
                    )

            with self._trace_service.step_trace(
                active_trace=active_trace,
                step_type="llm_generation",
                run_type="llm",
                inputs={"user_input": latest_user_message},
            ) as generation_step:
                answer = await self._generation_service.generate(
                    GenerationRequest(
                        messages=request.messages,
                        intent=decision,
                        sources=sources,
                        temperature=self._generation_temperature,
                    )
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
            yield {"type": "status", "stage": "thinking", "message": "正在思考"}

            with self._trace_service.step_trace(
                active_trace=active_trace,
                step_type="intent_decision",
                run_type="chain",
                inputs={"user_input": latest_user_message},
            ) as intent_step:
                decision = await self._intent_service.decide(request.messages)
                self._trace_service.complete_intent_step(
                    active_trace=active_trace,
                    active_step=intent_step,
                    input_text=latest_user_message,
                    intent=decision.intent,
                    need_rag=decision.need_rag,
                    rewrite_query=decision.rewrite_query,
                    rationale=decision.rationale,
                )

            sources = []
            if decision.need_rag:
                with self._trace_service.step_trace(
                    active_trace=active_trace,
                    step_type="retrieval",
                    run_type="retriever",
                    inputs={"query": decision.rewrite_query},
                ) as retrieval_step:
                    sources = await self._retriever_service.retrieve(decision.rewrite_query)
                    self._trace_service.complete_retrieval_step(
                        active_step=retrieval_step,
                        request_id=active_trace.request_id,
                        query=decision.rewrite_query,
                        retrieved_ids=[source.document_id for source in sources],
                    )

            answer_parts: list[str] = []
            with self._trace_service.step_trace(
                active_trace=active_trace,
                step_type="llm_generation",
                run_type="llm",
                inputs={"user_input": latest_user_message},
            ) as generation_step:
                async for delta in self._generation_service.generate_stream(
                    GenerationRequest(
                        messages=request.messages,
                        intent=decision,
                        sources=sources,
                        temperature=self._generation_temperature,
                    )
                ):
                    answer_parts.append(delta)
                    yield {"type": "delta", "delta": delta}

                answer = "".join(answer_parts)
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
