from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.persistence.models import GenerationRecord, IntentRecord, RequestTrace, RetrievalRecord, TraceStep
from app.persistence.trace_store import SQLTraceStore
from app.schemas.traces import (
    TraceDetail,
    TraceGenerationRecord,
    TraceIntentRecord,
    TraceListResponse,
    TraceRetrievalRecord,
    TraceStepItem,
    TraceSummary,
)


@dataclass
class LangSmithRunHandle:
    run_id: str | None = None
    trace_id: str | None = None
    _run_tree: Any | None = None

    def end(self, outputs: dict[str, Any]) -> None:
        if self._run_tree is not None:
            self._run_tree.end(outputs=outputs)


class LangSmithObserver:
    def __init__(
        self,
        enabled: bool,
        project_name: str,
        endpoint: str,
        api_key: str | None,
    ) -> None:
        self._available = False
        self._project_name = project_name
        self._client = None
        self._ls = None

        if not enabled or not api_key:
            return

        try:
            import langsmith as ls
            from langsmith import Client
        except ImportError:
            return

        self._ls = ls
        self._client = Client(api_key=api_key, api_url=endpoint)
        self._available = True

    @contextmanager
    def root_trace(self, name: str, inputs: dict[str, Any]):
        if not self._available:
            yield LangSmithRunHandle()
            return

        with self._ls.tracing_context(  # type: ignore[union-attr]
            client=self._client,
            project_name=self._project_name,
            enabled=True,
        ):
            with self._ls.trace(name, "chain", inputs=inputs) as run_tree:  # type: ignore[union-attr]
                yield LangSmithRunHandle(
                    run_id=str(getattr(run_tree, "id", "")) or None,
                    trace_id=str(getattr(run_tree, "trace_id", "")) or None,
                    _run_tree=run_tree,
                )

    @contextmanager
    def child_trace(self, name: str, run_type: str, inputs: dict[str, Any]):
        if not self._available:
            yield LangSmithRunHandle()
            return

        with self._ls.trace(name, run_type, inputs=inputs) as run_tree:  # type: ignore[union-attr]
            yield LangSmithRunHandle(
                run_id=str(getattr(run_tree, "id", "")) or None,
                trace_id=str(getattr(run_tree, "trace_id", "")) or None,
                _run_tree=run_tree,
            )

    async def flush(self) -> None:
        if self._client is None:
            return
        flush = getattr(self._client, "flush", None)
        if flush is None:
            return
        maybe_awaitable = flush()
        if hasattr(maybe_awaitable, "__await__"):
            await maybe_awaitable


@dataclass
class ActiveTrace:
    request_id: str
    session_id: str | None
    user_input: str
    langsmith_trace_id: str | None
    started_at: float
    observer_handle: LangSmithRunHandle
    _step_order: int = 0
    completed: bool = False

    def next_step_order(self) -> int:
        self._step_order += 1
        return self._step_order


@dataclass
class ActiveStep:
    step_id: str
    request_id: str
    step_type: str
    step_order: int
    started_at: float
    observer_handle: LangSmithRunHandle
    completed: bool = False


class TraceService:
    def __init__(self, store: SQLTraceStore, observer: LangSmithObserver) -> None:
        self._store = store
        self._observer = observer

    def setup(self) -> None:
        self._store.setup()

    async def shutdown(self) -> None:
        await self._observer.flush()

    def count_request_traces(self) -> int:
        return self._store.count_request_traces()

    def list_traces(
        self,
        *,
        page: int,
        page_size: int,
        session_id: str | None = None,
        status: str | None = None,
    ) -> TraceListResponse:
        offset = (page - 1) * page_size
        traces, total = self._store.list_request_traces(
            limit=page_size,
            offset=offset,
            session_id=session_id,
            status=status,
        )
        items = [self._to_trace_summary(trace) for trace in traces]
        return TraceListResponse(items=items, total=total, page=page, page_size=page_size)

    def get_trace_detail(self, request_id: str) -> TraceDetail | None:
        trace = self._store.get_request_trace(request_id)
        if trace is None:
            return None
        intent_record = self._store.get_intent_record(request_id)
        retrieval_record = self._store.get_retrieval_record(request_id)
        generation_record = self._store.get_generation_record(request_id)
        return self._to_trace_detail(
            trace=trace,
            intent_record=intent_record,
            retrieval_record=retrieval_record,
            generation_record=generation_record,
        )

    @contextmanager
    def request_trace(self, session_id: str | None, user_input: str):
        with self._observer.root_trace(
            "chat_pipeline",
            {
                "user_input": user_input,
                "session_id": session_id,
            },
        ) as observer_handle:
            request_id = self._store.create_request_trace(
                session_id=session_id,
                user_input=user_input,
                langsmith_trace_id=observer_handle.trace_id,
            )
            active_trace = ActiveTrace(
                request_id=request_id,
                session_id=session_id,
                user_input=user_input,
                langsmith_trace_id=observer_handle.trace_id,
                started_at=perf_counter(),
                observer_handle=observer_handle,
            )
            try:
                yield active_trace
            except Exception as exc:
                if not active_trace.completed:
                    self.fail_request_trace(active_trace, str(exc))
                    observer_handle.end(outputs={"status": "error", "error": str(exc)})
                raise

    @contextmanager
    def step_trace(
        self,
        active_trace: ActiveTrace,
        step_type: str,
        run_type: str,
        inputs: dict[str, Any],
    ):
        step_order = active_trace.next_step_order()
        with self._observer.child_trace(step_type, run_type, inputs) as observer_handle:
            step_id = self._store.create_step(
                request_id=active_trace.request_id,
                step_type=step_type,
                step_order=step_order,
                langsmith_run_id=observer_handle.run_id,
            )
            active_step = ActiveStep(
                step_id=step_id,
                request_id=active_trace.request_id,
                step_type=step_type,
                step_order=step_order,
                started_at=perf_counter(),
                observer_handle=observer_handle,
            )
            try:
                yield active_step
            except Exception as exc:
                if not active_step.completed:
                    self.fail_step(active_step, str(exc))
                    observer_handle.end(outputs={"status": "error", "error": str(exc)})
                raise

    def complete_intent_step(
        self,
        active_trace: ActiveTrace,
        active_step: ActiveStep,
        input_text: str,
        intent: str,
        need_rag: bool,
        rewrite_query: str,
        rationale: str,
    ) -> None:
        record_id = self._store.create_intent_record(
            request_id=active_trace.request_id,
            input_text=input_text,
            intent=intent,
            need_rag=need_rag,
            rewrite_query=rewrite_query,
            model_output={
                "intent": intent,
                "need_rag": need_rag,
                "rewrite_query": rewrite_query,
                "rationale": rationale,
            },
        )
        self._complete_step(
            active_step=active_step,
            record_ref_type="intent_record",
            record_ref_id=record_id,
            outputs={
                "intent": intent,
                "need_rag": need_rag,
                "rewrite_query": rewrite_query,
            },
        )

    def complete_retrieval_step(
        self,
        active_step: ActiveStep,
        request_id: str,
        query: str,
        retrieved_ids: list[str],
    ) -> None:
        record_id = self._store.create_retrieval_record(
            request_id=request_id,
            query=query,
            retrieved_ids=retrieved_ids,
        )
        self._complete_step(
            active_step=active_step,
            record_ref_type="retrieval_record",
            record_ref_id=record_id,
            outputs={
                "retrieved_ids": retrieved_ids,
                "retrieved_count": len(retrieved_ids),
            },
        )

    def complete_generation_step(
        self,
        active_step: ActiveStep,
        request_id: str,
        user_input: str,
        used_source_ids: list[str],
        llm_output: str,
    ) -> None:
        record_id = self._store.create_generation_record(
            request_id=request_id,
            user_input=user_input,
            used_source_ids=used_source_ids,
            llm_output=llm_output,
        )
        self._complete_step(
            active_step=active_step,
            record_ref_type="generation_record",
            record_ref_id=record_id,
            outputs={"output": llm_output},
        )

    def record_timing_step(
        self,
        active_trace: ActiveTrace,
        step_type: str,
        latency_ms: int | None,
    ) -> None:
        if latency_ms is None:
            return
        self._store.create_completed_step(
            request_id=active_trace.request_id,
            step_type=step_type,
            step_order=active_trace.next_step_order(),
            latency_ms=max(0, int(latency_ms)),
        )

    def complete_request_trace(
        self,
        active_trace: ActiveTrace,
        intent: str | None,
        need_rag: bool | None,
        final_output: str,
    ) -> None:
        total_latency_ms = self._elapsed_ms(active_trace.started_at)
        self._store.complete_request_trace(
            request_id=active_trace.request_id,
            intent=intent,
            need_rag=need_rag,
            final_output=final_output,
            total_latency_ms=total_latency_ms,
        )
        active_trace.observer_handle.end(
            outputs={
                "intent": intent,
                "need_rag": need_rag,
                "final_output": final_output,
                "request_id": active_trace.request_id,
            }
        )
        active_trace.completed = True

    def fail_request_trace(self, active_trace: ActiveTrace, error_message: str) -> None:
        self._store.fail_request_trace(
            request_id=active_trace.request_id,
            error_message=error_message,
            total_latency_ms=self._elapsed_ms(active_trace.started_at),
        )
        active_trace.completed = True

    def fail_step(self, active_step: ActiveStep, error_message: str) -> None:
        self._store.fail_step(
            step_id=active_step.step_id,
            latency_ms=self._elapsed_ms(active_step.started_at),
            error_message=error_message,
        )
        active_step.completed = True

    def _complete_step(
        self,
        active_step: ActiveStep,
        record_ref_type: str,
        record_ref_id: str,
        outputs: dict[str, Any],
    ) -> None:
        self._store.complete_step(
            step_id=active_step.step_id,
            latency_ms=self._elapsed_ms(active_step.started_at),
            record_ref_type=record_ref_type,
            record_ref_id=record_ref_id,
        )
        active_step.completed = True
        active_step.observer_handle.end(outputs=outputs)

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return int((perf_counter() - started_at) * 1000)

    @staticmethod
    def _to_trace_summary(trace: RequestTrace) -> TraceSummary:
        return TraceSummary(
            request_id=trace.request_id,
            session_id=trace.session_id,
            langsmith_trace_id=trace.langsmith_trace_id,
            user_input=trace.user_input,
            intent=trace.intent,
            need_rag=trace.need_rag,
            status=trace.status,
            total_latency_ms=trace.total_latency_ms,
            error_message=trace.error_message,
            created_at=trace.created_at,
            completed_at=trace.completed_at,
            step_count=len(trace.steps),
        )

    @staticmethod
    def _to_trace_detail(
        trace: RequestTrace,
        intent_record: IntentRecord | None,
        retrieval_record: RetrievalRecord | None,
        generation_record: GenerationRecord | None,
    ) -> TraceDetail:
        steps = sorted(trace.steps, key=lambda step: (step.step_order, step.created_at))
        return TraceDetail(
            request_id=trace.request_id,
            session_id=trace.session_id,
            langsmith_trace_id=trace.langsmith_trace_id,
            user_input=trace.user_input,
            intent=trace.intent,
            need_rag=trace.need_rag,
            final_output=trace.final_output,
            status=trace.status,
            total_latency_ms=trace.total_latency_ms,
            error_message=trace.error_message,
            created_at=trace.created_at,
            completed_at=trace.completed_at,
            step_count=len(steps),
            steps=[TraceService._to_trace_step(step) for step in steps],
            intent_record=TraceService._to_intent_record(intent_record),
            retrieval_record=TraceService._to_retrieval_record(retrieval_record),
            generation_record=TraceService._to_generation_record(generation_record),
        )

    @staticmethod
    def _to_trace_step(step: TraceStep) -> TraceStepItem:
        return TraceStepItem(
            step_id=step.step_id,
            request_id=step.request_id,
            step_type=step.step_type,
            step_order=step.step_order,
            status=step.status,
            latency_ms=step.latency_ms,
            record_ref_type=step.record_ref_type,
            record_ref_id=step.record_ref_id,
            langsmith_run_id=step.langsmith_run_id,
            error_message=step.error_message,
            created_at=step.created_at,
            completed_at=step.completed_at,
        )

    @staticmethod
    def _to_intent_record(record: IntentRecord | None) -> TraceIntentRecord | None:
        if record is None:
            return None
        return TraceIntentRecord(
            intent_record_id=record.intent_record_id,
            request_id=record.request_id,
            input_text=record.input_text,
            intent=record.intent,
            need_rag=record.need_rag,
            rewrite_query=record.rewrite_query,
            model_output=record.model_output,
            created_at=record.created_at,
        )

    @staticmethod
    def _to_retrieval_record(record: RetrievalRecord | None) -> TraceRetrievalRecord | None:
        if record is None:
            return None
        return TraceRetrievalRecord(
            retrieval_record_id=record.retrieval_record_id,
            request_id=record.request_id,
            query=record.query,
            retrieved_ids=record.retrieved_ids,
            created_at=record.created_at,
        )

    @staticmethod
    def _to_generation_record(record: GenerationRecord | None) -> TraceGenerationRecord | None:
        if record is None:
            return None
        return TraceGenerationRecord(
            generation_record_id=record.generation_record_id,
            request_id=record.request_id,
            user_input=record.user_input,
            used_source_ids=record.used_source_ids,
            llm_output=record.llm_output,
            created_at=record.created_at,
        )
