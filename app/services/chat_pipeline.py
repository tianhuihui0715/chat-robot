from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import AsyncIterator

from app.schemas.chat import (
    AggregateResult,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    IntentDecision,
    PlanTask,
    SourceChunk,
    SubtaskResult,
)
from app.services.generator_service import (
    GenerationRequest,
    QueuedGenerationService,
    build_generation_prompt_messages,
)
from app.services.intent_service import IntentService
from app.services.knowledge_base import KnowledgeBase
from app.services.planner_service import PlannerPlan, PlannerService
from app.services.rag_snapshot_service import RAGSnapshotService
from app.services.retriever_service import (
    RetrieverService,
    begin_retrieval_timings,
    end_retrieval_timings,
    get_retrieval_timings,
)
from app.services.trace_service import TraceService
from app.services.tool_runtime import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolRuntime,
    render_tool_catalog,
)


@dataclass
class PreparedGeneration:
    sources: list[SourceChunk]
    messages: list[ChatMessage]
    planner_plan: PlannerPlan | None = None
    planner_queries: list[str] = field(default_factory=list)
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    aggregate_result: AggregateResult | None = None
    execution_steps: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class TaskExecutionIntent:
    intent: str
    query: str
    reason: str
    source_hint: str | None = None
    target: str | None = None
    knowledge_base_id: str | None = None


class ChatPipeline:
    def __init__(
        self,
        intent_service: IntentService,
        knowledge_base: KnowledgeBase,
        retriever_service: RetrieverService,
        generation_service: QueuedGenerationService,
        planner_service: PlannerService,
        trace_service: TraceService,
        tool_runtime: ToolRuntime,
        rag_snapshot_service: RAGSnapshotService,
        generation_temperature: float = 0.0,
        plan_execute_top_k: int = 8,
        plan_execute_candidate_multiplier: int = 4,
        plan_execute_rerank_candidate_limit: int = 24,
        plan_execute_bm25_top_k: int = 16,
        plan_execute_max_retries: int = 1,
        plan_execute_retry_multiplier: int = 2,
    ) -> None:
        self._intent_service = intent_service
        self._knowledge_base = knowledge_base
        self._retriever_service = retriever_service
        self._generation_service = generation_service
        self._planner_service = planner_service
        self._trace_service = trace_service
        self._tool_runtime = tool_runtime
        self._rag_snapshot_service = rag_snapshot_service
        self._generation_temperature = generation_temperature
        self._plan_execute_top_k = plan_execute_top_k
        self._plan_execute_candidate_multiplier = plan_execute_candidate_multiplier
        self._plan_execute_rerank_candidate_limit = plan_execute_rerank_candidate_limit
        self._plan_execute_bm25_top_k = plan_execute_bm25_top_k
        self._plan_execute_max_retries = plan_execute_max_retries
        self._plan_execute_retry_multiplier = plan_execute_retry_multiplier

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
                inputs={
                    "user_input": latest_user_message,
                    "execution_strategy": request.execution_strategy,
                    "react_mode_legacy": request.react_mode,
                },
            ) as intent_step:
                decision = await self._intent_service.decide(request.messages)
                decision = self._route_knowledge_base(decision)
                decision = self._apply_execution_mode_override(decision, request.execution_strategy)
                self._trace_service.complete_intent_step(
                    active_trace=active_trace,
                    active_step=intent_step,
                    input_text=latest_user_message,
                    intent=decision.intent,
                    need_rag=decision.need_rag,
                    rewrite_query=decision.rewrite_query,
                    rationale=decision.rationale,
                    model_output_extra=self._build_intent_trace_payload(
                        decision,
                        request.execution_strategy,
                    ),
                )
                self._rag_snapshot_service.update_intent(
                    active_trace.request_id,
                    intent_payload=decision.model_dump(mode="json"),
                )

            if decision.should_clarify and decision.clarify_question:
                answer = decision.clarify_question
                with self._trace_service.step_trace(
                    active_trace=active_trace,
                    step_type="clarify_response",
                    run_type="chain",
                    inputs={"question": answer},
                ) as clarify_step:
                    self._trace_service.complete_generic_step(
                        clarify_step,
                        outputs={"clarify_question": answer},
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
                    sources=[],
                    planner=None,
                    subtask_results=[],
                    aggregate_result=None,
                    execution_steps=[],
                    queue_size=self._generation_service.queue_size,
                    request_id=active_trace.request_id,
                )

            prepared = await self._prepare_generation(
                request=request,
                decision=decision,
                active_trace=active_trace,
                latest_user_message=latest_user_message,
            )

            with self._trace_service.step_trace(
                active_trace=active_trace,
                step_type="llm_generation",
                run_type="llm",
                inputs={"user_input": latest_user_message},
            ) as generation_step:
                generation_request = GenerationRequest(
                    messages=prepared.messages,
                    intent=decision,
                    sources=prepared.sources,
                    temperature=self._generation_temperature,
                )
                self._rag_snapshot_service.update_generation(
                    active_trace.request_id,
                    generation_payload={
                        "prompt_messages": build_generation_prompt_messages(generation_request),
                        "sources": [source.model_dump(mode="json") for source in prepared.sources],
                        "planner_plan": (
                            _planner_plan_to_payload(prepared.planner_plan)
                            if prepared.planner_plan is not None
                            else None
                        ),
                        "subtask_results": [
                            result.model_dump(mode="json")
                            for result in prepared.subtask_results
                        ],
                        "aggregate_result": (
                            prepared.aggregate_result.model_dump(mode="json")
                            if prepared.aggregate_result is not None
                            else None
                        ),
                        "plan_execute_profile": self._plan_execute_retrieval_profile(),
                    },
                )
                raw_answer = await self._generation_service.generate(generation_request)
                answer = _ensure_rag_citations(raw_answer, prepared.sources)
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
                    used_source_ids=[source.document_id for source in prepared.sources],
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
                sources=prepared.sources,
                planner=_planner_plan_to_payload(prepared.planner_plan),
                subtask_results=prepared.subtask_results,
                aggregate_result=prepared.aggregate_result,
                execution_steps=prepared.execution_steps,
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
                inputs={
                    "user_input": latest_user_message,
                    "execution_strategy": request.execution_strategy,
                    "react_mode_legacy": request.react_mode,
                },
            ) as intent_step:
                decision = await self._intent_service.decide(request.messages)
                decision = self._route_knowledge_base(decision)
                decision = self._apply_execution_mode_override(decision, request.execution_strategy)
                self._trace_service.complete_intent_step(
                    active_trace=active_trace,
                    active_step=intent_step,
                    input_text=latest_user_message,
                    intent=decision.intent,
                    need_rag=decision.need_rag,
                    rewrite_query=decision.rewrite_query,
                    rationale=decision.rationale,
                    model_output_extra=self._build_intent_trace_payload(
                        decision,
                        request.execution_strategy,
                    ),
                )
                self._rag_snapshot_service.update_intent(
                    active_trace.request_id,
                    intent_payload=decision.model_dump(mode="json"),
                )

            if decision.should_clarify and decision.clarify_question:
                answer = decision.clarify_question
                yield {"type": "status", "stage": "clarify", "message": "需要补充信息"}
                with self._trace_service.step_trace(
                    active_trace=active_trace,
                    step_type="clarify_response",
                    run_type="chain",
                    inputs={"question": answer},
                ) as clarify_step:
                    self._trace_service.complete_generic_step(
                        clarify_step,
                        outputs={"clarify_question": answer},
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
                        sources=[],
                        planner=None,
                        subtask_results=[],
                        aggregate_result=None,
                        execution_steps=[],
                        queue_size=self._generation_service.queue_size,
                        request_id=active_trace.request_id,
                    ).model_dump(mode="json"),
                }
                return

            progress_events: list[dict] = []

            async def emit_event(event: dict) -> None:
                progress_events.append(event)

            prepared = await self._prepare_generation(
                request=request,
                decision=decision,
                active_trace=active_trace,
                latest_user_message=latest_user_message,
                emit_event=emit_event,
            )
            for event in progress_events:
                yield event

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
                    messages=prepared.messages,
                    intent=decision,
                    sources=prepared.sources,
                    temperature=self._generation_temperature,
                )
                self._rag_snapshot_service.update_generation(
                    active_trace.request_id,
                    generation_payload={
                        "prompt_messages": build_generation_prompt_messages(generation_request),
                        "sources": [source.model_dump(mode="json") for source in prepared.sources],
                        "planner_plan": (
                            _planner_plan_to_payload(prepared.planner_plan)
                            if prepared.planner_plan is not None
                            else None
                        ),
                        "subtask_results": [
                            result.model_dump(mode="json")
                            for result in prepared.subtask_results
                        ],
                        "aggregate_result": (
                            prepared.aggregate_result.model_dump(mode="json")
                            if prepared.aggregate_result is not None
                            else None
                        ),
                        "plan_execute_profile": self._plan_execute_retrieval_profile(),
                    },
                )
                async for delta in self._generation_service.generate_stream(generation_request):
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
                answer = _ensure_rag_citations(raw_answer, prepared.sources)
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
                    used_source_ids=[source.document_id for source in prepared.sources],
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
                    sources=prepared.sources,
                    planner=_planner_plan_to_payload(prepared.planner_plan),
                    subtask_results=prepared.subtask_results,
                    aggregate_result=prepared.aggregate_result,
                    execution_steps=prepared.execution_steps,
                    queue_size=self._generation_service.queue_size,
                    request_id=active_trace.request_id,
                ).model_dump(mode="json"),
            }

    async def _prepare_generation(
        self,
        *,
        request: ChatRequest,
        decision: IntentDecision,
        active_trace,
        latest_user_message: str,
        emit_event=None,
    ) -> PreparedGeneration:
        if decision.execution_mode == "plan_execute":
            return await self._run_plan_execute(
                request=request,
                decision=decision,
                active_trace=active_trace,
                latest_user_message=latest_user_message,
                emit_event=emit_event,
            )
        if not decision.need_rag:
            return PreparedGeneration(sources=[], messages=request.messages)

        retrieval_result = await self._execute_runtime_tool(
            active_trace=active_trace,
            tool_name="retrieval.search",
            arguments={"query": decision.rewrite_query},
            context=self._build_tool_context(
                active_trace=active_trace,
                request=request,
                knowledge_base_id=decision.knowledge_base_id,
            ),
        )
        return PreparedGeneration(sources=retrieval_result.sources, messages=request.messages)

    async def _run_plan_execute(
        self,
        *,
        request: ChatRequest,
        decision: IntentDecision,
        active_trace,
        latest_user_message: str,
        emit_event=None,
    ) -> PreparedGeneration:
        execution_steps: list[dict[str, object]] = []
        with self._trace_service.step_trace(
            active_trace=active_trace,
            step_type="plan_execution_planning",
            run_type="llm",
            inputs={
                "user_input": latest_user_message,
                "rewrite_query": decision.rewrite_query,
                "candidate_tools": decision.candidate_tools,
                "planner_hint": decision.planner_hint,
                "retrieval_profile": self._plan_execute_retrieval_profile(),
            },
        ) as planning_step:
            plan = await self._planner_service.plan(
                messages=request.messages,
                decision=decision,
            )
            self._rag_snapshot_service.update_planner(
                active_trace.request_id,
                planner_payload=_planner_plan_to_payload(plan, include_debug=True) or {},
            )
            self._trace_service.complete_generic_step(
                planning_step,
                outputs=_planner_plan_to_payload(plan),
            )
        planning_payload = _planner_plan_to_payload(plan) or {}
        execution_steps.append(
            {
                "type": "planner",
                "status": "completed",
                **planning_payload,
            }
        )
        if emit_event is not None:
            await emit_event(
                {
                    "type": "planner",
                    "data": planning_payload,
                }
            )

        if plan.mode == "answer_direct":
            return PreparedGeneration(
                sources=[],
                messages=_append_planner_system_message(request.messages, plan),
                planner_plan=plan,
                execution_steps=execution_steps,
            )

        if plan.mode == "single_retrieval":
            execution_steps.append(
                {
                    "type": "retrieval",
                    "status": "planned",
                    "query": plan.primary_query,
                    "retrieval_profile": self._plan_execute_retrieval_profile(),
                }
            )
            if emit_event is not None:
                await emit_event(
                    {
                        "type": "execution",
                        "stage": "retrieval_planned",
                        "message": f"准备检索：{plan.primary_query}",
                        "query": plan.primary_query,
                        "retrieval_profile": self._plan_execute_retrieval_profile(),
                    }
                )
            retrieval_result = await self._execute_runtime_tool(
                active_trace=active_trace,
                tool_name="retrieval.search",
                arguments={"query": plan.primary_query},
                context=self._build_tool_context(
                    active_trace=active_trace,
                    request=request,
                    knowledge_base_id=decision.knowledge_base_id,
                    retrieval_profile=self._plan_execute_retrieval_profile(),
                ),
            )
            sources = retrieval_result.sources
            return PreparedGeneration(
                sources=sources,
                messages=_append_planner_system_message(request.messages, plan),
                planner_plan=plan,
                planner_queries=[plan.primary_query],
                subtask_results=[],
                aggregate_result=None,
                execution_steps=execution_steps
                + [
                    {
                        "type": "retrieval",
                        "status": "completed",
                        "query": plan.primary_query,
                        "retrieved_count": len(sources),
                        "retrieval_profile": self._plan_execute_retrieval_profile(),
                    }
                ],
            )

        tasks = _build_plan_tasks(plan)
        task_results, sources = await self._execute_plan_tasks(
            active_trace=active_trace,
            tasks=tasks,
            use_reranker=request.use_reranker,
            knowledge_base_id=decision.knowledge_base_id,
            knowledge_base_name=decision.knowledge_base_name,
            execution_steps=execution_steps,
            emit_event=emit_event,
        )
        aggregate_result = _aggregate_subtask_results(plan, tasks, task_results)
        execution_steps.append(
            {
                "type": "aggregate",
                "status": "completed",
                "mode": aggregate_result.mode,
                "items": aggregate_result.items,
                "grouped_items": aggregate_result.grouped_items,
                "ranked_task_ids": aggregate_result.ranked_task_ids,
                "left_count": aggregate_result.left_count,
                "right_count": aggregate_result.right_count,
                "winner_task_id": aggregate_result.winner_task_id,
            }
        )
        if emit_event is not None:
            await emit_event(
                {
                    "type": "execution",
                    "stage": "aggregate_completed",
                    "message": f"聚合完成：{aggregate_result.mode}",
                    "mode": aggregate_result.mode,
                    "items": aggregate_result.items,
                    "grouped_items": aggregate_result.grouped_items,
                    "ranked_task_ids": aggregate_result.ranked_task_ids,
                    "left_count": aggregate_result.left_count,
                    "right_count": aggregate_result.right_count,
                    "confidence": aggregate_result.confidence,
                }
            )
        return PreparedGeneration(
            sources=sources,
            messages=_append_execution_context_messages(
                request.messages,
                plan=plan,
                task_results=task_results,
                aggregate_result=aggregate_result,
            ),
            planner_plan=plan,
            planner_queries=plan.subqueries,
            subtask_results=task_results,
            aggregate_result=aggregate_result,
            execution_steps=execution_steps,
        )

    async def _execute_plan_tasks(
        self,
        *,
        active_trace,
        tasks: list[PlanTask],
        use_reranker: bool | None,
        knowledge_base_id: str | None,
        knowledge_base_name: str | None,
        execution_steps: list[dict[str, object]],
        emit_event=None,
    ) -> tuple[list[SubtaskResult], list[SourceChunk]]:
        task_results: list[SubtaskResult] = []
        task_results_by_id: dict[str, SubtaskResult] = {}
        merged: list[SourceChunk] = []
        seen_keys: set[tuple[str, str]] = set()
        try:
            plan_levels = _build_plan_execution_levels(tasks)
        except ValueError:
            tasks = [
                task.model_copy(update={"depends_on": []})
                for task in tasks
            ]
            plan_levels = [tasks]

        with self._trace_service.step_trace(
            active_trace=active_trace,
            step_type="subtask_execution",
            run_type="chain",
            inputs={
                "tasks": [task.model_dump(mode="json") for task in tasks],
                "execution_levels": [
                    [task.task_id for task in level]
                    for level in plan_levels
                ],
            },
        ) as execution_step:
            for index, task in enumerate(tasks, start=1):
                planned_step = {
                    "type": "subtask",
                    "status": "planned",
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "query": task.query,
                    "depends_on": task.depends_on,
                }
                execution_steps.append(planned_step)
                if emit_event is not None:
                    await emit_event(
                        {
                            "type": "execution",
                            "stage": "subtask_planned",
                            "message": f"任务 {index}/{len(tasks)}：{task.goal}",
                            "query_index": index,
                            "task_id": task.task_id,
                            "goal": task.goal,
                            "query": task.query,
                            "depends_on": task.depends_on,
                        }
                    )

            task_position_by_id = {
                task.task_id: index
                for index, task in enumerate(tasks, start=1)
            }
            for level_index, level_tasks in enumerate(plan_levels, start=1):
                subtask_outputs = await asyncio.gather(
                    *[
                        self._execute_single_task(
                            active_trace=active_trace,
                            task=task,
                            use_reranker=use_reranker,
                            knowledge_base_id=knowledge_base_id,
                            knowledge_base_name=knowledge_base_name,
                            task_index=task_position_by_id[task.task_id],
                            task_count=len(tasks),
                            dependency_results={
                                dependency_id: task_results_by_id[dependency_id]
                                for dependency_id in task.depends_on
                                if dependency_id in task_results_by_id
                            },
                            emit_event=emit_event,
                        )
                        for task in level_tasks
                    ]
                )

                for task, sources, task_result, task_execution_steps in subtask_outputs:
                    execution_steps.extend(task_execution_steps)
                    task_results.append(task_result)
                    task_results_by_id[task.task_id] = task_result
                    for source in sources:
                        enriched = source.model_copy(
                            update={
                                "metadata": {
                                    **source.metadata,
                                    "planner_query_index": task.task_id,
                                    "planner_query": task.query,
                                    "planner_task_id": task.task_id,
                                }
                            }
                        )
                        key = (enriched.document_id, task.task_id)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        merged.append(enriched)
            self._trace_service.complete_generic_step(
                execution_step,
                outputs={
                    "task_count": len(tasks),
                    "execution_levels": [
                        [task.task_id for task in level]
                        for level in plan_levels
                    ],
                    "retrieved_count": len(merged),
                    "subtask_results": [
                        result.model_dump(mode="json") for result in task_results
                    ],
                },
            )
        return task_results, merged

    async def _run_retrieval(
        self,
        *,
        active_trace,
        query: str,
        use_reranker: bool | None,
        knowledge_base_id: str | None,
        top_k_override: int | None = None,
        candidate_multiplier_override: int | None = None,
        rerank_candidate_limit_override: int | None = None,
        bm25_top_k_override: int | None = None,
    ) -> list[SourceChunk]:
        with self._trace_service.step_trace(
            active_trace=active_trace,
            step_type="retrieval",
            run_type="retriever",
            inputs={
                "query": query,
                "top_k_override": top_k_override,
                "candidate_multiplier_override": candidate_multiplier_override,
                "rerank_candidate_limit_override": rerank_candidate_limit_override,
                "bm25_top_k_override": bm25_top_k_override,
            },
        ) as retrieval_step:
            timing_token = begin_retrieval_timings()
            retrieval_snapshot_tokens = self._rag_snapshot_service.activate_request(
                active_trace.request_id
            )
            try:
                sources = await self._retriever_service.retrieve(
                    query,
                    use_reranker=use_reranker,
                    knowledge_base_id=knowledge_base_id,
                    top_k_override=top_k_override,
                    candidate_multiplier_override=candidate_multiplier_override,
                    rerank_candidate_limit_override=rerank_candidate_limit_override,
                    bm25_top_k_override=bm25_top_k_override,
                )
                retrieval_timings = get_retrieval_timings()
            finally:
                self._rag_snapshot_service.end_request(retrieval_snapshot_tokens)
                end_retrieval_timings(timing_token)
            self._trace_service.complete_retrieval_step(
                active_step=retrieval_step,
                request_id=active_trace.request_id,
                query=query,
                retrieved_ids=[source.document_id for source in sources],
            )
            self._record_retrieval_timing_steps(active_trace, retrieval_timings)
            return sources

    async def _execute_single_task(
        self,
        *,
        active_trace,
        task: PlanTask,
        use_reranker: bool | None,
        knowledge_base_id: str | None,
        knowledge_base_name: str | None,
        task_index: int,
        task_count: int,
        dependency_results: dict[str, SubtaskResult],
        emit_event=None,
    ) -> tuple[PlanTask, list[SourceChunk], SubtaskResult, list[dict[str, object]]]:
        task_execution_steps: list[dict[str, object]] = []
        task_intent = await self._classify_task_intent(
            active_trace=active_trace,
            task=task,
            dependency_results=dependency_results,
        )
        task_execution_steps.append(
            {
                "type": "task_intent",
                "status": "completed",
                "task_id": task.task_id,
                "goal": task.goal,
                "intent": task_intent.intent,
                "query": task_intent.query,
                "source_hint": task_intent.source_hint,
                "target": task_intent.target,
                "knowledge_base_id": task_intent.knowledge_base_id,
                "reason": task_intent.reason,
            }
        )
        effective_task = task.model_copy(update={"query": task_intent.query or task.query})
        selected_tool = await self._select_task_tool(
            active_trace=active_trace,
            task=effective_task,
            task_intent=task_intent,
            dependency_results=dependency_results,
        )
        task_execution_steps.append(
            {
                "type": "tool_selection",
                "status": "completed",
                "task_id": effective_task.task_id,
                "goal": effective_task.goal,
                "task_intent": task_intent.intent,
                "tool_name": selected_tool["tool"],
                "arguments": selected_tool["arguments"],
                "reason": selected_tool["reason"],
            }
        )
        selected_knowledge_base_id = knowledge_base_id
        selected_knowledge_base_name = knowledge_base_name
        if self._tool_requires_knowledge_base(str(selected_tool["tool"])):
            selected_knowledge_base_id, selected_knowledge_base_name, kb_reason = await self._select_task_knowledge_base(
                active_trace=active_trace,
                task=effective_task,
                task_intent=task_intent,
                selected_tool_name=str(selected_tool["tool"]),
                dependency_results=dependency_results,
                default_knowledge_base_id=knowledge_base_id,
                default_knowledge_base_name=knowledge_base_name,
            )
            task_execution_steps.append(
                {
                    "type": "knowledge_base_selection",
                    "status": "completed",
                    "task_id": effective_task.task_id,
                    "goal": effective_task.goal,
                    "tool_name": selected_tool["tool"],
                    "knowledge_base_id": selected_knowledge_base_id,
                    "knowledge_base_name": selected_knowledge_base_name,
                    "reason": kb_reason,
                }
            )
        if task.depends_on:
            task_execution_steps.append(
                {
                    "type": "subtask_dependency_wait",
                    "status": "completed",
                    "task_id": effective_task.task_id,
                    "goal": effective_task.goal,
                    "depends_on": task.depends_on,
                    "dependency_result_count": len(dependency_results),
                }
            )
        sources: list[SourceChunk] = []
        if selected_tool["tool"] == "retrieval.search":
            retrieval_result = await self._execute_runtime_tool(
                active_trace=active_trace,
                tool_name="retrieval.search",
                arguments=selected_tool["arguments"],
                context=ToolExecutionContext(
                    request_id=active_trace.request_id,
                    session_id=active_trace.session_id,
                    knowledge_base_id=selected_knowledge_base_id,
                    use_reranker=use_reranker,
                    retrieval_profile=self._plan_execute_retrieval_profile(),
                ),
            )
            sources = retrieval_result.sources
        elif selected_tool["tool"] != "answer.direct":
            await self._execute_runtime_tool(
                active_trace=active_trace,
                tool_name=selected_tool["tool"],
                arguments=selected_tool["arguments"],
                context=ToolExecutionContext(
                    request_id=active_trace.request_id,
                    session_id=active_trace.session_id,
                    knowledge_base_id=selected_knowledge_base_id,
                    use_reranker=use_reranker,
                    retrieval_profile=self._plan_execute_retrieval_profile(),
                ),
            )
        task_result = await self._run_subtask_completion(
            active_trace=active_trace,
            task=effective_task,
            task_intent=task_intent,
            sources=sources,
            dependency_results=dependency_results,
        )
        retry_count = 0
        while (
            selected_tool["tool"] == "retrieval.search"
            and task_result.needs_retry
            and retry_count < self._plan_execute_max_retries
        ):
            retry_count += 1
            retry_query = _build_retry_query(effective_task.query, task_result)
            retry_profile = self._plan_execute_retry_profile(retry_count)
            task_execution_steps.append(
                {
                    "type": "subtask_retry",
                    "status": "planned",
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "retry_count": retry_count,
                    "query": retry_query,
                    "reason": task_result.coverage_hint or task_result.notes,
                    "retrieval_profile": retry_profile,
                }
            )
            if emit_event is not None:
                await emit_event(
                    {
                        "type": "execution",
                        "stage": "subtask_retry_planned",
                        "task_id": task.task_id,
                        "goal": task.goal,
                        "retry_count": retry_count,
                        "query": retry_query,
                        "message": f"任务 {task_index}/{task_count} 触发补检索：{retry_query}",
                        "retrieval_profile": retry_profile,
                    }
                )
            retry_result = await self._execute_runtime_tool(
                active_trace=active_trace,
                tool_name="retrieval.search",
                arguments={"query": retry_query},
                context=ToolExecutionContext(
                    request_id=active_trace.request_id,
                    session_id=active_trace.session_id,
                    knowledge_base_id=selected_knowledge_base_id,
                    use_reranker=use_reranker,
                    retrieval_profile=retry_profile,
                ),
            )
            retry_sources = retry_result.sources
            merged_retry_sources = _merge_task_sources(sources, retry_sources)
            task_result = await self._run_subtask_completion(
                active_trace=active_trace,
                task=effective_task.model_copy(update={"query": retry_query}),
                task_intent=task_intent,
                sources=merged_retry_sources,
                dependency_results=dependency_results,
                retry_count=retry_count,
            )
            sources = merged_retry_sources
            task_execution_steps.append(
                {
                    "type": "subtask_retry",
                    "status": "completed",
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "retry_count": retry_count,
                    "query": retry_query,
                    "retrieved_count": len(retry_sources),
                    "items": task_result.items,
                    "count": task_result.count,
                    "confidence": task_result.confidence,
                    "coverage_hint": task_result.coverage_hint,
                    "retrieval_profile": retry_profile,
                }
            )
            if emit_event is not None:
                await emit_event(
                    {
                        "type": "execution",
                        "stage": "subtask_retry_completed",
                        "task_id": task.task_id,
                        "goal": task.goal,
                        "retry_count": retry_count,
                        "query": retry_query,
                        "retrieved_count": len(retry_sources),
                        "item_count": len(task_result.items),
                        "count": task_result.count,
                        "items": task_result.items[:8],
                        "confidence": task_result.confidence,
                        "coverage_hint": task_result.coverage_hint,
                        "message": f"任务 {task_index}/{task_count} 补检索完成，新增 {len(retry_sources)} 条来源",
                        "retrieval_profile": retry_profile,
                    }
                )

        task_execution_steps.append(
            {
                "type": "subtask",
                "status": "completed",
                "task_id": task.task_id,
                "goal": task.goal,
                "query": effective_task.query,
                "depends_on": task.depends_on,
                "task_intent": task_intent.intent,
                "source_hint": task_intent.source_hint,
                "target": task_intent.target,
                "retrieved_count": len(sources),
                "items": task_result.items,
                "count": task_result.count,
                "confidence": task_result.confidence,
                "coverage_hint": task_result.coverage_hint,
                "needs_retry": task_result.needs_retry,
                "retry_count": retry_count,
                "retrieval_profile": self._plan_execute_retrieval_profile(),
            }
        )
        if emit_event is not None:
            await emit_event(
                {
                    "type": "execution",
                    "stage": "subtask_completed",
                    "message": f"任务 {task_index}/{task_count} 完成，召回 {len(sources)} 条来源，提取 {len(task_result.items)} 项",
                    "query_index": task_index,
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "query": effective_task.query,
                    "depends_on": task.depends_on,
                    "task_intent": task_intent.intent,
                    "source_hint": task_intent.source_hint,
                    "target": task_intent.target,
                    "retrieved_count": len(sources),
                    "item_count": len(task_result.items),
                    "items": task_result.items[:8],
                    "count": task_result.count,
                    "confidence": task_result.confidence,
                    "coverage_hint": task_result.coverage_hint,
                    "retry_count": retry_count,
                    "retrieval_profile": self._plan_execute_retrieval_profile(),
                }
            )
        return task, sources, task_result, task_execution_steps

    async def _run_subtask_completion(
        self,
        *,
        active_trace,
        task: PlanTask,
        task_intent: TaskExecutionIntent,
        sources: list[SourceChunk],
        dependency_results: dict[str, SubtaskResult],
        retry_count: int = 0,
    ) -> SubtaskResult:
        dependency_payload = (
            json.dumps(
                {
                    task_id: result.model_dump(mode="json")
                    for task_id, result in dependency_results.items()
                },
                ensure_ascii=False,
            )
            if dependency_results
            else ""
        )
        with self._trace_service.step_trace(
            active_trace=active_trace,
            step_type="subtask_completion",
            run_type="llm",
            inputs={
                "task_id": task.task_id,
                "goal": task.goal,
                "query": task.query,
                "task_intent": task_intent.intent,
                "source_hint": task_intent.source_hint,
                "target": task_intent.target,
                "source_count": len(sources),
                "depends_on": task.depends_on,
                "dependency_result_count": len(dependency_results),
            },
        ) as subtask_step:
            raw_output = await self._generation_service.generate(
                GenerationRequest(
                    messages=[
                        ChatMessage(
                            role="user",
                            content=(
                                f"任务目标：{task.goal}\n"
                                f"执行查询：{task.query}\n"
                                + (
                                    f"前置结果：{dependency_payload}\n"
                                    if dependency_payload
                                    else ""
                                )
                                + (
                                    f"抽取对象：{task_intent.target}\n"
                                    if task_intent.target
                                    else ""
                                )
                                + (
                                    f"来源提示：{task_intent.source_hint}\n"
                                    if task_intent.source_hint
                                    else ""
                                )
                                + 
                                "请只输出 subtask JSON。"
                            ),
                        )
                    ],
                    intent=IntentDecision(
                        intent="knowledge_qa",
                        need_rag=True,
                        rewrite_query=task.query,
                        rationale=f"Subtask execution for {task.task_id}",
                        execution_mode="rag",
                    ),
                    sources=sources,
                    temperature=0.0,
                    response_mode="json",
                    system_prompt_override=_build_subtask_executor_prompt(
                        task,
                        task_intent=task_intent,
                        has_dependency_results=bool(dependency_results),
                    ),
                )
            )
            result = _parse_subtask_result(
                raw_output=raw_output,
                task=task,
                sources=sources,
                dependency_mode=bool(dependency_results),
                retry_count=retry_count,
            )
            self._trace_service.complete_generic_step(
                subtask_step,
                outputs=result.model_dump(mode="json"),
            )
            return result

    @staticmethod
    def _build_intent_trace_payload(
        decision: IntentDecision,
        execution_strategy: str,
    ) -> dict[str, object]:
        return {
            "execution_strategy": execution_strategy,
            "execution_mode": decision.execution_mode,
            "should_clarify": decision.should_clarify,
            "clarify_question": decision.clarify_question,
            "candidate_tools": decision.candidate_tools,
            "planner_hint": decision.planner_hint,
            "knowledge_base_id": decision.knowledge_base_id,
            "knowledge_base_name": decision.knowledge_base_name,
        }

    def _apply_execution_mode_override(
        self,
        decision: IntentDecision,
        execution_strategy: str,
    ) -> IntentDecision:
        default_planner_tools = (
            self._tool_runtime.registry.planner_tool_names()
            or ["retrieval.search", "answer.direct"]
        )
        if execution_strategy == "auto":
            if decision.execution_mode != "plan_execute" or decision.candidate_tools:
                return decision
            return decision.model_copy(update={"candidate_tools": default_planner_tools})
        if decision.intent == "reject" or decision.should_clarify:
            return decision
        if execution_strategy == "off":
            return decision.model_copy(
                update={
                    "execution_mode": "rag" if decision.need_rag else "direct",
                    "candidate_tools": [],
                    "planner_hint": None,
                }
            )
        return decision.model_copy(
            update={
                "execution_mode": "plan_execute",
                "candidate_tools": decision.candidate_tools or default_planner_tools,
            }
        )

    @staticmethod
    def _extract_latest_user_message(request: ChatRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content.strip()
        return request.messages[-1].content.strip()

    def update_generation_temperature(self, temperature: float) -> None:
        self._generation_temperature = temperature

    @staticmethod
    def _build_tool_context(
        *,
        active_trace,
        request: ChatRequest,
        knowledge_base_id: str | None,
        retrieval_profile: dict[str, int] | None = None,
    ) -> ToolExecutionContext:
        return ToolExecutionContext(
            request_id=active_trace.request_id,
            session_id=active_trace.session_id,
            knowledge_base_id=knowledge_base_id,
            use_reranker=request.use_reranker,
            retrieval_profile=retrieval_profile or {},
        )

    def update_plan_execute_runtime_config(
        self,
        *,
        top_k: int,
        candidate_multiplier: int,
        rerank_candidate_limit: int,
        bm25_top_k: int,
        max_retries: int,
        retry_multiplier: int,
    ) -> None:
        self._plan_execute_top_k = top_k
        self._plan_execute_candidate_multiplier = candidate_multiplier
        self._plan_execute_rerank_candidate_limit = rerank_candidate_limit
        self._plan_execute_bm25_top_k = bm25_top_k
        self._plan_execute_max_retries = max_retries
        self._plan_execute_retry_multiplier = retry_multiplier

    def _plan_execute_retrieval_profile(self) -> dict[str, int]:
        return {
            "top_k": self._plan_execute_top_k,
            "candidate_multiplier": self._plan_execute_candidate_multiplier,
            "rerank_candidate_limit": self._plan_execute_rerank_candidate_limit,
            "bm25_top_k": self._plan_execute_bm25_top_k,
            "max_retries": self._plan_execute_max_retries,
            "retry_multiplier": self._plan_execute_retry_multiplier,
        }

    async def _classify_task_intent(
        self,
        *,
        active_trace,
        task: PlanTask,
        dependency_results: dict[str, SubtaskResult],
    ) -> TaskExecutionIntent:
        catalog = self._knowledge_base_catalog()
        dependency_payload = (
            json.dumps(
                {
                    task_id: {
                        "goal": result.goal,
                        "items": result.items[:8],
                        "summary": result.summary,
                    }
                    for task_id, result in dependency_results.items()
                },
                ensure_ascii=False,
            )
            if dependency_results
            else ""
        )
        kb_catalog = ""
        if len(catalog) > 1:
            kb_catalog = "\n可用知识库：\n" + _render_knowledge_base_catalog(catalog)
        system_prompt = (
            "你是一个中文子任务意图识别器。你只能输出一行 JSON，禁止输出解释、前后缀、代码块或任何额外文字。\n"
            "固定输出字段：intent, query, source_hint, target, knowledge_base_id, reason。\n"
            "intent 只允许 direct、retrieval、extraction、aggregation。\n"
            "如果任务主要基于前置结果做比较、交集、排序、分组、去重或汇总，选择 aggregation。\n"
            "如果任务需要从知识库中列出、统计、抽取某类对象，选择 extraction。\n"
            "如果任务是普通单次查询，选择 retrieval。\n"
            "query 用简洁中文短句；source_hint 只写来源主题、作品、文档或对象范围；target 只写待抽取对象类型；knowledge_base_id 不确定时写空字符串。"
            f"{kb_catalog}"
        )
        user_prompt = (
            f"任务ID：{task.task_id}\n"
            f"任务目标：{task.goal}\n"
            f"建议查询：{task.query}\n"
            f"依赖任务：{', '.join(task.depends_on) if task.depends_on else '无'}\n"
            + (f"前置结果：{dependency_payload}\n" if dependency_payload else "")
            + "请只输出 task intent JSON。"
        )
        with self._trace_service.step_trace(
            active_trace=active_trace,
            step_type="task_intent",
            run_type="llm",
            inputs={
                "task_id": task.task_id,
                "goal": task.goal,
                "query": task.query,
                "depends_on": task.depends_on,
                "dependency_result_count": len(dependency_results),
            },
        ) as intent_step:
            raw_output = await self._generation_service.generate(
                GenerationRequest(
                    messages=[ChatMessage(role="user", content=user_prompt)],
                    intent=IntentDecision(
                        intent="task",
                        need_rag=not bool(task.depends_on),
                        rewrite_query=task.query,
                        rationale=f"Task intent classification for {task.task_id}",
                        execution_mode="plan_execute",
                    ),
                    sources=[],
                    temperature=0.0,
                    response_mode="json",
                    system_prompt_override=system_prompt,
                )
            )
            task_intent = _parse_task_intent(
                raw_output=raw_output,
                task=task,
                dependency_results=dependency_results,
                catalog=catalog,
            )
            self._trace_service.complete_generic_step(
                intent_step,
                outputs={
                    "raw_output": raw_output,
                    **_task_intent_to_payload(task_intent),
                },
            )
            return task_intent

    async def _select_task_tool(
        self,
        *,
        active_trace,
        task: PlanTask,
        task_intent: TaskExecutionIntent,
        dependency_results: dict[str, SubtaskResult],
    ) -> dict[str, object]:
        tool_definitions = self._tool_runtime.registry.list_tools()
        tool_catalog = render_tool_catalog(tool_definitions)
        dependency_payload = (
            json.dumps(
                {
                    task_id: result.model_dump(mode="json")
                    for task_id, result in dependency_results.items()
                },
                ensure_ascii=False,
            )
            if dependency_results
            else ""
        )
        system_prompt = (
            "你是一个中文任务工具选择器。你只能输出一行 JSON，禁止输出解释、前后缀、代码块或任何额外文字。\n"
            "固定输出字段：tool, arguments, reason。\n"
            "tool 必须从给定工具列表中选择；arguments 必须是对象；reason 用一句中文简短说明原因。\n"
            "如果 task_intent=aggregation 或 direct，优先选择 answer.direct。\n"
            "如果 task_intent=extraction，优先选择 retrieval.search，并围绕 source_hint 和 target 组织 query。\n"
            "如果任务主要依赖前置结果推导，不需要外部工具，优先选择 answer.direct。\n"
            "如果任务需要从知识库查询片段，优先选择 retrieval.search，并提供 query。\n"
            "如果任务是定位文档，选择 kb.document_lookup，并提供 keyword。\n"
            "可用工具列表：\n"
            f"{tool_catalog}"
        )
        user_prompt = (
            f"任务ID：{task.task_id}\n"
            f"任务目标：{task.goal}\n"
            f"建议查询：{task.query}\n"
            f"task_intent：{task_intent.intent}\n"
            + (f"来源提示：{task_intent.source_hint}\n" if task_intent.source_hint else "")
            + (f"抽取对象：{task_intent.target}\n" if task_intent.target else "")
            +
            f"依赖任务：{', '.join(task.depends_on) if task.depends_on else '无'}\n"
            + (
                f"前置结果：{dependency_payload}\n"
                if dependency_payload
                else ""
            )
            + "请只输出 tool JSON。"
        )
        with self._trace_service.step_trace(
            active_trace=active_trace,
            step_type="tool_selection",
            run_type="llm",
            inputs={
                "task_id": task.task_id,
                "goal": task.goal,
                "query": task.query,
                "task_intent": task_intent.intent,
                "depends_on": task.depends_on,
                "dependency_result_count": len(dependency_results),
            },
        ) as selection_step:
            raw_output = await self._generation_service.generate(
                GenerationRequest(
                    messages=[ChatMessage(role="user", content=user_prompt)],
                    intent=IntentDecision(
                        intent="task",
                        need_rag=not bool(task.depends_on),
                        rewrite_query=task.query,
                        rationale=f"Tool selection for {task.task_id}",
                        execution_mode="plan_execute",
                    ),
                    sources=[],
                    temperature=0.0,
                    response_mode="json",
                    system_prompt_override=system_prompt,
                )
            )
            selection = _parse_tool_selection(
                raw_output=raw_output,
                task=task,
                task_intent=task_intent,
                available_tools=tool_definitions,
                dependency_results=dependency_results,
            )
            self._trace_service.complete_generic_step(
                selection_step,
                outputs={
                    "raw_output": raw_output,
                    **selection,
                },
            )
            return selection

    def _tool_requires_knowledge_base(self, tool_name: str) -> bool:
        tool = self._tool_runtime.registry.get(tool_name)
        if tool is None:
            return tool_name in {"retrieval.search", "kb.document_lookup"}
        return tool.definition.requires_knowledge_base

    async def _select_task_knowledge_base(
        self,
        *,
        active_trace,
        task: PlanTask,
        task_intent: TaskExecutionIntent,
        selected_tool_name: str,
        dependency_results: dict[str, SubtaskResult],
        default_knowledge_base_id: str | None,
        default_knowledge_base_name: str | None,
    ) -> tuple[str | None, str | None, str]:
        catalog = self._knowledge_base_catalog()
        if not catalog:
            return default_knowledge_base_id, default_knowledge_base_name, "当前没有可用知识库，沿用默认设置。"

        if len(catalog) == 1:
            knowledge_base_id, payload = next(iter(catalog.items()))
            return knowledge_base_id, str(payload["name"]), "当前只有一个知识库，直接使用该知识库。"

        preferred_kb_id = (task_intent.knowledge_base_id or "").strip()
        if preferred_kb_id and preferred_kb_id in catalog:
            return (
                preferred_kb_id,
                str(catalog[preferred_kb_id]["name"]),
                "子任务抽取意图已经明确指定知识库，直接沿用该知识库。",
            )

        heuristic = self._select_knowledge_base(
            "\n".join(
                part
                for part in [
                    task.goal,
                    task.query,
                    task_intent.source_hint or "",
                    task_intent.target or "",
                ]
                if part
            ),
            catalog=catalog,
        )
        fallback_id = default_knowledge_base_id
        fallback_name = default_knowledge_base_name
        if heuristic is not None:
            fallback_id, fallback_name = heuristic
        elif fallback_id is None:
            knowledge_base_id, payload = next(iter(catalog.items()))
            fallback_id = knowledge_base_id
            fallback_name = str(payload["name"])

        system_prompt = (
            "你是一个中文知识库选择器。你只能输出一行 JSON，禁止输出解释、前后缀、代码块或任何额外文字。\n"
            "固定输出字段：knowledge_base_id, reason。\n"
            "knowledge_base_id 必须从给定知识库列表中选择；reason 用一句中文简短说明原因。\n"
            "如果任务和某个知识库的标题、描述明显对应，就选那个知识库。\n"
            "可用知识库：\n"
            f"{_render_knowledge_base_catalog(catalog)}"
        )
        dependency_payload = (
            json.dumps(
                {
                    task_id: {
                        "goal": result.goal,
                        "summary": result.summary,
                        "items": result.items[:6],
                    }
                    for task_id, result in dependency_results.items()
                },
                ensure_ascii=False,
            )
            if dependency_results
            else ""
        )
        user_prompt = (
            f"工具：{selected_tool_name}\n"
            f"任务目标：{task.goal}\n"
            f"建议查询：{task.query}\n"
            f"task_intent：{task_intent.intent}\n"
            + (f"来源提示：{task_intent.source_hint}\n" if task_intent.source_hint else "")
            + (f"抽取对象：{task_intent.target}\n" if task_intent.target else "")
            + (f"前置结果：{dependency_payload}\n" if dependency_payload else "")
            + "请只输出 knowledge base JSON。"
        )
        with self._trace_service.step_trace(
            active_trace=active_trace,
            step_type="knowledge_base_selection",
            run_type="llm",
            inputs={
                "task_id": task.task_id,
                "tool_name": selected_tool_name,
                "goal": task.goal,
                "query": task.query,
                "task_intent": task_intent.intent,
            },
        ) as kb_step:
            raw_output = await self._generation_service.generate(
                GenerationRequest(
                    messages=[ChatMessage(role="user", content=user_prompt)],
                    intent=IntentDecision(
                        intent="knowledge_qa",
                        need_rag=True,
                        rewrite_query=task.query,
                        rationale=f"Knowledge base selection for {task.task_id}",
                        execution_mode="plan_execute",
                    ),
                    sources=[],
                    temperature=0.0,
                    response_mode="json",
                    system_prompt_override=system_prompt,
                )
            )
            selected_id, selected_name, reason = _parse_knowledge_base_selection(
                raw_output=raw_output,
                catalog=catalog,
                fallback_id=fallback_id,
                fallback_name=fallback_name,
                fallback_reason=(
                    "模型未返回合法知识库选择，已按启发式结果回退。"
                    if heuristic is not None
                    else "模型未返回合法知识库选择，已按默认知识库回退。"
                ),
            )
            self._trace_service.complete_generic_step(
                kb_step,
                outputs={
                    "raw_output": raw_output,
                    "knowledge_base_id": selected_id,
                    "knowledge_base_name": selected_name,
                    "reason": reason,
                },
            )
            return selected_id, selected_name, reason

    async def _execute_runtime_tool(
        self,
        *,
        active_trace,
        tool_name: str,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        with self._trace_service.step_trace(
            active_trace=active_trace,
            step_type="tool_call",
            run_type="tool",
            inputs={"tool_name": tool_name, "arguments": arguments},
        ) as tool_step:
            if tool_name == "retrieval.search":
                query = str(arguments.get("query", "")).strip()
                if not query:
                    result = ToolExecutionResult(
                        tool_name=tool_name,
                        ok=False,
                        payload={"error": "query is required"},
                        summary="缺少检索 query。",
                    )
                else:
                    profile = context.retrieval_profile or {}
                    explicit_top_k = arguments.get("top_k")
                    top_k_override = (
                        explicit_top_k
                        if isinstance(explicit_top_k, int)
                        else profile.get("top_k")
                    )
                    sources = await self._run_retrieval(
                        active_trace=active_trace,
                        query=query,
                        use_reranker=context.use_reranker,
                        knowledge_base_id=context.knowledge_base_id,
                        top_k_override=top_k_override,
                        candidate_multiplier_override=profile.get("candidate_multiplier"),
                        rerank_candidate_limit_override=profile.get("rerank_candidate_limit"),
                        bm25_top_k_override=profile.get("bm25_top_k"),
                    )
                    result = ToolExecutionResult(
                        tool_name=tool_name,
                        ok=True,
                        payload={
                            "query": query,
                            "retrieved_count": len(sources),
                            "source_ids": [source.document_id for source in sources],
                        },
                        summary=f"检索完成，返回 {len(sources)} 条来源。",
                        sources=sources,
                    )
            else:
                result = await self._tool_runtime.execute(tool_name, arguments, context)
            self._trace_service.complete_generic_step(
                tool_step,
                outputs={
                    "tool_name": result.tool_name,
                    "ok": result.ok,
                    "summary": result.summary,
                    **result.payload,
                },
            )
            return result

    def _plan_execute_retry_profile(self, retry_count: int) -> dict[str, int]:
        scale = max(1, self._plan_execute_retry_multiplier) ** max(0, retry_count)
        return {
            "top_k": max(self._plan_execute_top_k, int(self._plan_execute_top_k * scale)),
            "candidate_multiplier": max(
                self._plan_execute_candidate_multiplier,
                int(self._plan_execute_candidate_multiplier * scale),
            ),
            "rerank_candidate_limit": max(
                self._plan_execute_rerank_candidate_limit,
                int(self._plan_execute_rerank_candidate_limit * scale),
            ),
            "bm25_top_k": max(
                self._plan_execute_bm25_top_k,
                int(self._plan_execute_bm25_top_k * scale),
            ),
            "retry_count": retry_count,
            "scale": int(scale),
        }

    def _route_knowledge_base(self, decision: IntentDecision) -> IntentDecision:
        if not decision.need_rag:
            return decision

        selected = self._select_knowledge_base(decision.rewrite_query)
        if selected is None:
            return self._downgrade_out_of_domain_query(decision)

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

    def _knowledge_base_catalog(self) -> dict[str, dict[str, object]]:
        documents = self._knowledge_base.list_documents()
        if not documents:
            return {}

        catalog: dict[str, dict[str, object]] = {}
        for document in documents:
            knowledge_base_id = document.metadata.get("knowledge_base_id", "default")
            knowledge_base_name = document.metadata.get("knowledge_base_name", "默认知识库")
            bucket = catalog.setdefault(
                knowledge_base_id,
                {"name": knowledge_base_name, "titles": [], "snippets": []},
            )
            bucket["titles"].append(document.title)
            snippet = document.content[:400].strip()
            if snippet:
                bucket["snippets"].append(snippet)
        return catalog

    def _select_knowledge_base(
        self,
        query: str,
        *,
        catalog: dict[str, dict[str, object]] | None = None,
    ) -> tuple[str, str] | None:
        catalog = catalog or self._knowledge_base_catalog()
        if not catalog:
            return None

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
            for snippet in payload.get("snippets", []):
                compact_snippet = str(snippet).lower().replace(" ", "")
                if compact_snippet and compact_snippet in normalized_query:
                    score += 6
                for token in _extract_route_tokens(str(snippet)):
                    if token in normalized_query:
                        score += 1
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
        return None

    @staticmethod
    def _downgrade_out_of_domain_query(decision: IntentDecision) -> IntentDecision:
        return decision.model_copy(
            update={
                "need_rag": False,
                "execution_mode": "direct",
                "candidate_tools": [],
                "planner_hint": None,
                "knowledge_base_id": None,
                "knowledge_base_name": None,
                "rationale": (
                    f"{decision.rationale} 当前知识库没有命中明显领域线索，"
                    "已降级为直接回答，避免被无关知识库片段误导。"
                ),
            }
        )

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


def _planner_plan_to_payload(
    plan: PlannerPlan | None,
    *,
    include_debug: bool = False,
) -> dict[str, object] | None:
    if plan is None:
        return None
    payload: dict[str, object] = {
        "mode": plan.mode,
        "reason": plan.reason,
        "primary_query": plan.primary_query,
        "subqueries": plan.subqueries,
        "merge_strategy": plan.merge_strategy,
        "answer_style": plan.answer_style,
        "planner_model": plan.planner_model,
        "planner_source": plan.planner_source,
        "tasks": [
            task.model_dump(mode="json")
            for task in (plan.tasks or _build_plan_tasks(plan))
        ],
    }
    if include_debug:
        payload.update(
            {
                "system_prompt": plan.system_prompt,
                "user_prompt": plan.user_prompt,
                "raw_output": plan.raw_output,
            }
        )
    return payload


def _build_plan_tasks(plan: PlannerPlan) -> list[PlanTask]:
    if plan.tasks:
        return [
            task.model_copy(
                update={
                    "query": task.query or task.goal,
                    "depends_on": task.depends_on,
                }
            )
            for task in plan.tasks
        ]
    tasks: list[PlanTask] = []
    for index, query in enumerate(plan.subqueries, start=1):
        tasks.append(
            PlanTask(
                task_id=f"task_{index}",
                goal=f"完成子任务 {index}: {query}",
                query=query,
                depends_on=[],
            )
        )
    return tasks


def _build_subtask_executor_prompt(
    task: PlanTask,
    *,
    task_intent: TaskExecutionIntent,
    has_dependency_results: bool = False,
) -> str:
    dependency_hint = (
        "你会同时拿到前置任务结果，请优先基于这些前置结果完成当前任务；只有在来源片段里有补充证据时再引用来源。\n"
        if has_dependency_results
        else ""
    )
    extraction_hint = ""
    if task_intent.intent == "extraction":
        extraction_hint = (
            "当前任务是抽取型任务。请优先输出明确、稳定、可命名的候选项，不要输出描述性句子片段。\n"
            + (
                f"来源范围：{task_intent.source_hint}\n"
                if task_intent.source_hint
                else ""
            )
            + (
                f"抽取对象：{task_intent.target}\n"
                if task_intent.target
                else ""
            )
        )
    return (
        "你是一个中文子任务执行器。你只能输出一行 JSON，禁止输出解释、前后缀、代码块或额外文字。\n"
        "固定输出字段：items, count, summary, notes。\n"
        "items 是当前任务产出的候选条目列表；count 是 items 的数量；summary 是一句简短总结；notes 用于说明证据是否不足。\n"
        "如果证据不足，也要输出合法 JSON，items 可以为空数组。\n"
        f"{dependency_hint}"
        f"{extraction_hint}"
        f"当前子任务：{task.goal}\n"
        f"当前查询：{task.query}"
    )


def _parse_subtask_result(
    *,
    raw_output: str,
    task: PlanTask,
    sources: list[SourceChunk],
    dependency_mode: bool = False,
    retry_count: int = 0,
) -> SubtaskResult:
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in subtask output.")
        payload = json.loads(raw_output[start : end + 1])
        raw_items = payload.get("items", [])
        items = []
        for item in raw_items if isinstance(raw_items, list) else []:
            compact = str(item).strip()
            if compact and compact not in items:
                items.append(compact)
        count = payload.get("count")
        count = int(count) if isinstance(count, (int, float, str)) and str(count).strip().isdigit() else None
        summary = str(payload.get("summary", "")).strip() or None
        notes = str(payload.get("notes", "")).strip() or None
    except Exception:
        items = _fallback_extract_items(task.query, sources)
        count = len(items)
        summary = "模型未返回合法子任务 JSON，已按规则从来源片段提取候选项。"
        notes = "fallback_subtask_extraction"

    items = _clean_candidate_items(items, query=task.query)
    if not items:
        items = _fallback_extract_items(task.query, sources)
        items = _clean_candidate_items(items, query=task.query)
        if items and count in {None, 0}:
            count = len(items)
            notes = notes or "fallback_subtask_extraction"
    if count is None:
        count = len(items)
    confidence, coverage_hint, needs_retry = _assess_subtask_quality(
        query=task.query,
        sources=sources,
        items=items,
        summary=summary,
        notes=notes,
        dependency_mode=dependency_mode,
    )
    return SubtaskResult(
        task_id=task.task_id,
        goal=task.goal,
        query=task.query,
        depends_on=task.depends_on,
        items=items[:24],
        count=count,
        summary=summary,
        notes=notes,
        confidence=confidence,
        coverage_hint=coverage_hint,
        needs_retry=needs_retry,
        retry_count=retry_count,
        source_ids=[source.document_id for source in sources],
    )


def _task_intent_to_payload(task_intent: TaskExecutionIntent) -> dict[str, object]:
    return {
        "intent": task_intent.intent,
        "query": task_intent.query,
        "source_hint": task_intent.source_hint,
        "target": task_intent.target,
        "knowledge_base_id": task_intent.knowledge_base_id,
        "reason": task_intent.reason,
    }


def _parse_task_intent(
    *,
    raw_output: str,
    task: PlanTask,
    dependency_results: dict[str, SubtaskResult],
    catalog: dict[str, dict[str, object]],
) -> TaskExecutionIntent:
    fallback = _fallback_task_intent(
        task=task,
        dependency_results=dependency_results,
        catalog=catalog,
        reason="模型未返回合法任务意图，已按规则回退。",
    )
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in task intent output.")
        payload = json.loads(raw_output[start : end + 1])
    except Exception:
        return fallback

    intent = str(payload.get("intent", "")).strip().lower()
    if intent not in {"direct", "retrieval", "extraction", "aggregation"}:
        return fallback

    query = str(payload.get("query", "")).strip() or task.query
    source_hint = str(payload.get("source_hint", "")).strip() or None
    target = str(payload.get("target", "")).strip() or None
    knowledge_base_id = str(payload.get("knowledge_base_id", "")).strip() or None
    reason = str(payload.get("reason", "")).strip() or "模型未提供原因。"
    if knowledge_base_id and knowledge_base_id not in catalog:
        knowledge_base_id = None
    if intent == "extraction" and source_hint is None:
        source_hint = _infer_source_hint(query) or _infer_source_hint(task.goal)
    if intent == "extraction" and target is None:
        target = _infer_extraction_target(query) or _infer_extraction_target(task.goal)
    return TaskExecutionIntent(
        intent=intent,
        query=query,
        source_hint=source_hint,
        target=target,
        knowledge_base_id=knowledge_base_id,
        reason=reason,
    )


def _fallback_task_intent(
    *,
    task: PlanTask,
    dependency_results: dict[str, SubtaskResult],
    catalog: dict[str, dict[str, object]],
    reason: str,
) -> TaskExecutionIntent:
    query = task.query
    goal = task.goal
    source_hint = _infer_source_hint(query) or _infer_source_hint(goal)
    target = _infer_extraction_target(query) or _infer_extraction_target(goal)
    knowledge_base_id = None
    if source_hint:
        selected = _select_catalog_entry_for_text(source_hint, catalog)
        if selected is not None:
            knowledge_base_id = selected[0]
    if dependency_results and any(
        token in query + goal for token in ("比较", "交集", "共同", "排序", "分组", "去重", "汇总")
    ):
        return TaskExecutionIntent(
            intent="aggregation",
            query=query,
            source_hint=None,
            target=None,
            knowledge_base_id=None,
            reason=reason,
        )
    if _looks_like_extraction_task(query, goal):
        return TaskExecutionIntent(
            intent="extraction",
            query=query,
            source_hint=source_hint,
            target=target,
            knowledge_base_id=knowledge_base_id,
            reason=reason,
        )
    if any(token in query + goal for token in ("文档", "文件", "资料", "配置", "接口", "部署")):
        return TaskExecutionIntent(
            intent="retrieval",
            query=query,
            source_hint=source_hint,
            target=target,
            knowledge_base_id=knowledge_base_id,
            reason=reason,
        )
    if dependency_results:
        return TaskExecutionIntent(
            intent="direct",
            query=query,
            source_hint=None,
            target=None,
            knowledge_base_id=None,
            reason=reason,
        )
    return TaskExecutionIntent(
        intent="retrieval",
        query=query,
        source_hint=source_hint,
        target=target,
        knowledge_base_id=knowledge_base_id,
        reason=reason,
    )


def _looks_like_extraction_task(query: str, goal: str) -> bool:
    haystack = query + goal
    return any(token in haystack for token in ("列出", "统计", "出现过", "提取", "抽取", "有哪些", "哪些"))


def _infer_source_hint(text: str) -> str | None:
    quoted_match = re.search(r"《([^》]{1,30})》", text)
    if quoted_match:
        return quoted_match.group(1).strip()
    range_match = re.search(r"([A-Za-z0-9_\-\u4e00-\u9fff]{2,30})中", text)
    if range_match:
        return range_match.group(1).strip("：:，,。 ")
    named_match = re.search(r"(射雕英雄传|神雕侠侣|默认知识库|项目文档库|运维FAQ库)", text)
    if named_match:
        return named_match.group(1)
    return None


def _infer_extraction_target(text: str) -> str | None:
    candidates = (
        "武功",
        "功夫",
        "人物",
        "角色",
        "配置项",
        "接口",
        "错误",
        "部署方式",
        "命令",
        "参数",
        "文档",
    )
    for candidate in candidates:
        if candidate in text:
            return candidate
    return None


def _select_catalog_entry_for_text(
    text: str,
    catalog: dict[str, dict[str, object]],
) -> tuple[str, str] | None:
    normalized = text.lower().replace(" ", "")
    for knowledge_base_id, payload in catalog.items():
        knowledge_base_name = str(payload.get("name", knowledge_base_id))
        if normalized and (
            normalized in knowledge_base_id.lower().replace(" ", "")
            or normalized in knowledge_base_name.lower().replace(" ", "")
        ):
            return knowledge_base_id, knowledge_base_name
        for title in payload.get("titles", []):
            title_text = str(title).lower().replace(" ", "")
            if normalized and normalized in title_text:
                return knowledge_base_id, knowledge_base_name
    return None


def _fallback_extract_items(query: str, sources: list[SourceChunk]) -> list[str]:
    candidates: list[str] = []
    target_hint = "功" if any(token in query for token in ("功", "武功", "功夫", "掌", "剑", "拳")) else ""
    patterns = [
        r"[\u4e00-\u9fff]{2,8}(?:神功|真经|剑法|剑术|剑招|掌法|掌|拳法|拳|刀法|刀|棍法|棍|指法|指|爪功|爪|鞭法|步法|轻功|内功|功)",
    ]
    for source in sources:
        for pattern in patterns:
            for match in re.findall(pattern, source.content):
                compact = match.strip()
                if target_hint and target_hint not in compact and not compact.endswith(("掌", "拳", "功")):
                    continue
                if compact and compact not in candidates:
                    candidates.append(compact)
    return candidates[:24]


def _clean_candidate_items(items: list[str], *, query: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        candidate = str(item).strip()
        if not candidate:
            continue
        candidate = re.sub(r"^(?:这套|这门|这种|这个|这项|那套|那门|一种|一个|一门)", "", candidate)
        candidate = re.sub(r"(?:中最后一章的|本来|便似|也不能说|只是|说着).*$", "", candidate)
        candidate = candidate.strip(" ，,。；;：:")
        if not candidate:
            continue
        if _looks_like_dirty_candidate(candidate, query=query):
            continue
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(candidate)
    return cleaned[:24]


def _looks_like_dirty_candidate(item: str, *, query: str) -> bool:
    if len(item) <= 1:
        return True
    if len(item) > 12:
        return True
    generic_markers = (
        "绝世武功",
        "武功",
        "内功",
        "功夫",
        "高手的掌",
        "不会武功",
        "修习内功",
        "上乘内功",
    )
    if item in generic_markers:
        return True
    if any(marker in item for marker in ("不得", "不能", "本来", "便似", "只是", "说着", "究竟", "时他已")):
        return True
    if " " in item:
        return True
    if _looks_like_enumeration_query(query):
        required_suffixes = ("功", "掌", "拳", "剑", "刀", "指", "爪", "鞭", "棍", "真经", "步法", "轻功")
        if not item.endswith(required_suffixes):
            return True
    return False


def _assess_subtask_quality(
    *,
    query: str,
    sources: list[SourceChunk],
    items: list[str],
    summary: str | None,
    notes: str | None,
    dependency_mode: bool = False,
) -> tuple[float, str, bool]:
    source_count = len(sources)
    item_count = len(items)
    quality_penalty = 0.0
    reasons: list[str] = []

    if not dependency_mode and source_count == 0:
        quality_penalty += 0.6
        reasons.append("未检索到来源")
    elif not dependency_mode and source_count <= 2:
        quality_penalty += 0.25
        reasons.append("来源较少")

    if item_count == 0:
        quality_penalty += 0.5
        reasons.append("未提取到候选项")
    elif _looks_like_enumeration_query(query) and item_count <= 2:
        quality_penalty += 0.2
        reasons.append("候选项偏少")

    textual_haystack = " ".join(filter(None, [summary or "", notes or ""])).lower()
    if any(marker in textual_haystack for marker in ("不足", "可能", "部分", "fallback")):
        quality_penalty += 0.1
        reasons.append("结果带有不确定性")

    confidence = max(0.05, min(0.98, 1.0 - quality_penalty))
    needs_retry = (
        (not dependency_mode and source_count <= 2)
        or item_count == 0
        or (_looks_like_enumeration_query(query) and item_count <= 1)
    )
    if dependency_mode and (summary or item_count > 0):
        needs_retry = False
    coverage_hint = "；".join(reasons) if reasons else "来源和候选项覆盖正常"
    return confidence, coverage_hint, needs_retry


def _looks_like_enumeration_query(query: str) -> bool:
    return any(token in query for token in ("哪些", "列出", "统计", "出现过", "更多", "更少"))


def _build_retry_query(query: str, result: SubtaskResult) -> str:
    if any(token in query for token in ("哪些", "列出", "统计")):
        return f"{query}，请尽量列出全部候选项"
    if result.count == 0:
        return f"{query}，补充相关名称和列表"
    return f"{query}，补充更多细节"


def _merge_task_sources(primary: list[SourceChunk], secondary: list[SourceChunk]) -> list[SourceChunk]:
    merged: list[SourceChunk] = []
    seen: set[tuple[str, str]] = set()
    for source in [*primary, *secondary]:
        key = (source.document_id, source.metadata.get("chunk_id", source.title))
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
    return merged


def _build_plan_execution_levels(tasks: list[PlanTask]) -> list[list[PlanTask]]:
    if not tasks:
        return []
    tasks_by_id = {task.task_id: task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError("Duplicate task_id detected in plan tasks.")

    indegree = {task.task_id: 0 for task in tasks}
    dependents: dict[str, list[str]] = {task.task_id: [] for task in tasks}
    for task in tasks:
        for dependency_id in task.depends_on:
            if dependency_id == task.task_id:
                raise ValueError("Task cannot depend on itself.")
            if dependency_id not in tasks_by_id:
                raise ValueError("Task depends on unknown task.")
            indegree[task.task_id] += 1
            dependents[dependency_id].append(task.task_id)

    levels: list[list[PlanTask]] = []
    ready = [task.task_id for task in tasks if indegree[task.task_id] == 0]
    visited = 0
    while ready:
        current_ids = ready
        ready = []
        current_level = [tasks_by_id[task_id] for task_id in current_ids]
        levels.append(current_level)
        visited += len(current_level)
        for task_id in current_ids:
            for dependent_id in dependents[task_id]:
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    ready.append(dependent_id)

    if visited != len(tasks):
        raise ValueError("Cycle detected in task dependency graph.")
    return levels


def _select_aggregate_seed_results(
    tasks: list[PlanTask],
    task_results: list[SubtaskResult],
) -> list[SubtaskResult]:
    if not tasks or not task_results:
        return task_results
    source_task_ids = [task.task_id for task in tasks if not task.depends_on]
    if not source_task_ids:
        return task_results
    result_by_id = {result.task_id: result for result in task_results}
    selected = [
        result_by_id[task_id]
        for task_id in source_task_ids
        if task_id in result_by_id
    ]
    return selected or task_results


def _aggregate_subtask_results(
    plan: PlannerPlan,
    tasks: list[PlanTask] | list[SubtaskResult],
    task_results: list[SubtaskResult] | None = None,
) -> AggregateResult:
    if task_results is None:
        aggregated_results = list(tasks)  # type: ignore[list-item]
    else:
        aggregated_results = _select_aggregate_seed_results(tasks, task_results)  # type: ignore[arg-type]
    if plan.merge_strategy == "dedupe_union":
        merged_items: list[str] = []
        for result in aggregated_results:
            for item in result.items:
                if item not in merged_items:
                    merged_items.append(item)
        return AggregateResult(
            mode="dedupe_union",
            items=merged_items,
            notes="已对多个子任务结果做去重汇总。",
            confidence=_aggregate_confidence(aggregated_results, has_items=bool(merged_items)),
            needs_retry=any(result.needs_retry for result in aggregated_results) and not merged_items,
        )

    if plan.merge_strategy == "intersection":
        if len(aggregated_results) < 2:
            return AggregateResult(
                mode="intersection",
                items=aggregated_results[0].items if aggregated_results else [],
                notes="仅有一个子任务结果，无法严格求交集。",
            )
        common_items = set(aggregated_results[0].items)
        for result in aggregated_results[1:]:
            common_items &= set(result.items)
        ordered_items = [item for item in aggregated_results[0].items if item in common_items]
        return AggregateResult(
            mode="intersection",
            items=ordered_items,
            left_task_id=aggregated_results[0].task_id if aggregated_results else None,
            right_task_id=aggregated_results[1].task_id if len(aggregated_results) > 1 else None,
            notes="已按子任务结果求交集。",
            confidence=_aggregate_confidence(aggregated_results, has_items=bool(ordered_items)),
            needs_retry=any(result.needs_retry for result in aggregated_results) and not ordered_items,
        )

    if plan.merge_strategy == "compare":
        if len(aggregated_results) < 2:
            return AggregateResult(
                mode="compare",
                left_task_id=aggregated_results[0].task_id if aggregated_results else None,
                left_count=aggregated_results[0].count if aggregated_results else None,
                notes="仅有一个子任务结果，无法完成对比。",
            )
        left, right = aggregated_results[0], aggregated_results[1]
        left_count = left.count or len(left.items)
        right_count = right.count or len(right.items)
        if left_count > right_count:
            winner_task_id = left.task_id
        elif right_count > left_count:
            winner_task_id = right.task_id
        else:
            winner_task_id = None
        return AggregateResult(
            mode="compare",
            left_task_id=left.task_id,
            right_task_id=right.task_id,
            left_count=left_count,
            right_count=right_count,
            winner_task_id=winner_task_id,
            notes="已按子任务结果比较数量。",
            confidence=_aggregate_confidence(aggregated_results, has_items=(left_count + right_count) > 0),
            needs_retry=any(result.needs_retry for result in aggregated_results) and (left_count + right_count) == 0,
        )

    if plan.merge_strategy == "rank":
        ranked_results = sorted(
            aggregated_results,
            key=lambda result: (
                result.count or len(result.items),
                result.confidence or 0.0,
            ),
            reverse=True,
        )
        ranked_items = [
            f"{result.goal}（{result.count or len(result.items)}）"
            for result in ranked_results
        ]
        return AggregateResult(
            mode="rank",
            items=ranked_items,
            ranked_task_ids=[result.task_id for result in ranked_results],
            notes="已按子任务结果数量从高到低排序。",
            confidence=_aggregate_confidence(aggregated_results, has_items=bool(ranked_items)),
            needs_retry=any(result.needs_retry for result in aggregated_results) and not ranked_items,
        )

    if plan.merge_strategy == "group_by":
        grouped_items = {
            (result.goal or result.task_id): result.items
            for result in aggregated_results
        }
        flattened_items: list[str] = []
        for items in grouped_items.values():
            for item in items:
                if item not in flattened_items:
                    flattened_items.append(item)
        return AggregateResult(
            mode="group_by",
            items=flattened_items,
            grouped_items=grouped_items,
            notes="已按子任务分组保留结果。",
            confidence=_aggregate_confidence(aggregated_results, has_items=bool(grouped_items)),
            needs_retry=any(result.needs_retry for result in aggregated_results) and not flattened_items,
        )

    merged_items: list[str] = []
    for result in aggregated_results:
        for item in result.items:
            if item not in merged_items:
                merged_items.append(item)
    return AggregateResult(
        mode="union",
        items=merged_items,
        notes="已合并子任务结果。",
        confidence=_aggregate_confidence(aggregated_results, has_items=bool(merged_items)),
        needs_retry=any(result.needs_retry for result in aggregated_results) and not merged_items,
    )


def _aggregate_confidence(task_results: list[SubtaskResult], *, has_items: bool) -> float:
    if not task_results:
        return 0.05
    base = sum(result.confidence or 0.5 for result in task_results) / len(task_results)
    if not has_items:
        base -= 0.2
    return max(0.05, min(0.99, round(base, 3)))


def _append_planner_system_message(
    messages: list[ChatMessage],
    plan: PlannerPlan,
) -> list[ChatMessage]:
    plan_payload = json.dumps(_planner_plan_to_payload(plan), ensure_ascii=False)
    planner_message = ChatMessage(
        role="system",
        content=(
            "你已经拿到了规划结果，请严格按规划组织答案，不要输出规划 JSON。\n"
            f"规划结果：{plan_payload}\n"
            "如果 merge_strategy=intersection，只保留多个子查询都能支持的共同信息；"
            "如果 merge_strategy=dedupe_union，要先去重再汇总；"
            "如果 merge_strategy=rank，要按数量或覆盖度排序；"
            "如果 merge_strategy=group_by，要按子任务分组展示；"
            "如果证据不足，要明确说明。"
        ),
    )
    return [*messages, planner_message]


def _append_execution_context_messages(
    messages: list[ChatMessage],
    *,
    plan: PlannerPlan,
    task_results: list[SubtaskResult],
    aggregate_result: AggregateResult,
) -> list[ChatMessage]:
    plan_payload = json.dumps(_planner_plan_to_payload(plan), ensure_ascii=False)
    results_payload = json.dumps(
        [result.model_dump(mode="json") for result in task_results],
        ensure_ascii=False,
    )
    aggregate_payload = json.dumps(
        aggregate_result.model_dump(mode="json"),
        ensure_ascii=False,
    )
    execution_message = ChatMessage(
        role="system",
        content=(
            "你正在根据结构化执行结果生成最终答案。\n"
            f"规划结果：{plan_payload}\n"
            f"子任务结果：{results_payload}\n"
            f"聚合结果：{aggregate_payload}\n"
            "请优先依据聚合结果回答；如果是 intersection，只输出共同项；"
            "如果是 compare，明确说明哪个更多以及比较依据；"
            "如果是 dedupe_union，优先输出去重后的总列表；"
            "如果是 rank，按照聚合结果给出的排序输出；"
            "如果是 group_by，按分组标题分段展示；"
            "如果 aggregate_result.needs_retry=true 或 confidence 较低，要明确提示当前结论可能不完整。"
        ),
    )
    return [*messages, execution_message]


def _parse_action_selection(raw_output: str, fallback_query: str) -> dict[str, str]:
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in action output.")
        payload = json.loads(raw_output[start : end + 1])
        action = str(payload.get("action", "")).strip()
        query = str(payload.get("query", "")).strip() or fallback_query
        reason = str(payload.get("reason", "")).strip()
    except Exception:
        return {
            "action": "retrieval.search",
            "query": fallback_query,
            "reason": "模型未返回合法 action，回退到检索。",
        }

    if action not in {"retrieval.search", "answer.direct"}:
        action = "retrieval.search"
    return {
        "action": action,
        "query": query,
        "reason": reason or "模型未提供原因。",
    }


def _parse_tool_selection(
    *,
    raw_output: str,
    task: PlanTask,
    task_intent: TaskExecutionIntent,
    available_tools: list,
    dependency_results: dict[str, SubtaskResult],
) -> dict[str, object]:
    available_tool_names = {
        definition.name
        for definition in available_tools
        if definition.enabled
    }
    fallback = _fallback_tool_selection(
        task=task,
        task_intent=task_intent,
        dependency_results=dependency_results,
        available_tool_names=available_tool_names,
        reason="模型未返回合法工具选择，已按规则回退。",
    )
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in tool selection output.")
        payload = json.loads(raw_output[start : end + 1])
    except Exception:
        return fallback

    tool_name = str(payload.get("tool", "")).strip()
    arguments = payload.get("arguments", {})
    reason = str(payload.get("reason", "")).strip() or "模型未提供原因。"
    if tool_name not in available_tool_names or not isinstance(arguments, dict):
        return _fallback_tool_selection(
            task=task,
            task_intent=task_intent,
            dependency_results=dependency_results,
            available_tool_names=available_tool_names,
            reason="模型返回了不可用工具或非法参数，已按规则回退。",
        )

    normalized_arguments = {str(key): value for key, value in arguments.items()}
    if tool_name == "retrieval.search":
        query = str(normalized_arguments.get("query", "")).strip() or task_intent.query or task.query
        normalized_arguments = {"query": query}
    elif tool_name == "answer.direct":
        normalized_arguments = {
            "query": str(normalized_arguments.get("query", "")).strip()
            or task_intent.query
            or task.goal
        }
    elif tool_name == "kb.document_lookup":
        normalized_arguments = {
            "keyword": str(normalized_arguments.get("keyword", "")).strip()
            or task_intent.query
            or task.query
        }
    elif tool_name == "trace.lookup":
        normalized_arguments = {
            "request_id": str(normalized_arguments.get("request_id", "")).strip()
        }
        if not normalized_arguments["request_id"]:
            return fallback

    return {
        "tool": tool_name,
        "arguments": normalized_arguments,
        "reason": reason,
    }


def _fallback_tool_selection(
    *,
    task: PlanTask,
    task_intent: TaskExecutionIntent,
    dependency_results: dict[str, SubtaskResult],
    available_tool_names: set[str],
    reason: str,
) -> dict[str, object]:
    fallback_query = task_intent.query or task.query
    if task_intent.intent in {"aggregation", "direct"} and "answer.direct" in available_tool_names:
        return {
            "tool": "answer.direct",
            "arguments": {"query": fallback_query},
            "reason": reason,
        }
    if dependency_results and "answer.direct" in available_tool_names:
        return {
            "tool": "answer.direct",
            "arguments": {"query": fallback_query},
            "reason": reason,
        }
    if (
        any(token in task.goal for token in ("文档", "文件", "资料"))
        and "kb.document_lookup" in available_tool_names
    ):
        return {
            "tool": "kb.document_lookup",
            "arguments": {"keyword": fallback_query},
            "reason": reason,
        }
    if "retrieval.search" in available_tool_names:
        return {
            "tool": "retrieval.search",
            "arguments": {"query": fallback_query},
            "reason": reason,
        }
    return {
        "tool": "answer.direct",
        "arguments": {"query": fallback_query},
        "reason": reason,
    }


def _render_knowledge_base_catalog(catalog: dict[str, dict[str, object]]) -> str:
    sections: list[str] = []
    for knowledge_base_id, payload in sorted(catalog.items(), key=lambda item: item[0]):
        name = str(payload.get("name", knowledge_base_id))
        titles = [str(title) for title in payload.get("titles", [])[:3]]
        title_text = " / ".join(titles) if titles else "无标题示例"
        sections.append(f"- {knowledge_base_id}: {name} | titles={title_text}")
    return "\n".join(sections)


def _parse_knowledge_base_selection(
    *,
    raw_output: str,
    catalog: dict[str, dict[str, object]],
    fallback_id: str | None,
    fallback_name: str | None,
    fallback_reason: str,
) -> tuple[str | None, str | None, str]:
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in knowledge base selection output.")
        payload = json.loads(raw_output[start : end + 1])
    except Exception:
        return fallback_id, fallback_name, fallback_reason

    knowledge_base_id = str(payload.get("knowledge_base_id", "")).strip()
    reason = str(payload.get("reason", "")).strip() or "模型未提供原因。"
    if knowledge_base_id not in catalog:
        return fallback_id, fallback_name, fallback_reason
    return knowledge_base_id, str(catalog[knowledge_base_id].get("name", knowledge_base_id)), reason


def _ensure_rag_citations(answer: str, sources: list[SourceChunk]) -> str:
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
