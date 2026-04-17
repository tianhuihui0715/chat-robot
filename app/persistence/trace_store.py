from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.persistence.models import (
    Base,
    GenerationRecord,
    IntentRecord,
    RequestTrace,
    RetrievalRecord,
    TraceStep,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SQLTraceStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._ensure_sqlite_parent_dir()
        connect_args = {"check_same_thread": False} if dsn.startswith("sqlite") else {}
        self._engine = create_engine(dsn, future=True, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def setup(self) -> None:
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Session:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_request_trace(
        self,
        session_id: str | None,
        user_input: str,
        langsmith_trace_id: str | None,
    ) -> str:
        request_id = str(uuid4())
        request_trace = RequestTrace(
            request_id=request_id,
            session_id=session_id,
            user_input=user_input,
            langsmith_trace_id=langsmith_trace_id,
        )
        with self.session() as session:
            session.add(request_trace)
        return request_id

    def complete_request_trace(
        self,
        request_id: str,
        intent: str | None,
        need_rag: bool | None,
        final_output: str,
        total_latency_ms: int,
    ) -> None:
        with self.session() as session:
            request_trace = session.get(RequestTrace, request_id)
            if request_trace is None:
                return
            request_trace.intent = intent
            request_trace.need_rag = need_rag
            request_trace.final_output = final_output
            request_trace.status = "completed"
            request_trace.total_latency_ms = total_latency_ms
            request_trace.completed_at = utcnow()

    def fail_request_trace(
        self,
        request_id: str,
        error_message: str,
        total_latency_ms: int,
    ) -> None:
        with self.session() as session:
            request_trace = session.get(RequestTrace, request_id)
            if request_trace is None:
                return
            request_trace.status = "error"
            request_trace.error_message = error_message
            request_trace.total_latency_ms = total_latency_ms
            request_trace.completed_at = utcnow()

    def create_step(
        self,
        request_id: str,
        step_type: str,
        step_order: int,
        langsmith_run_id: str | None,
    ) -> str:
        step_id = str(uuid4())
        step = TraceStep(
            step_id=step_id,
            request_id=request_id,
            step_type=step_type,
            step_order=step_order,
            langsmith_run_id=langsmith_run_id,
        )
        with self.session() as session:
            session.add(step)
        return step_id

    def complete_step(
        self,
        step_id: str,
        latency_ms: int,
        record_ref_type: str | None = None,
        record_ref_id: str | None = None,
    ) -> None:
        with self.session() as session:
            step = session.get(TraceStep, step_id)
            if step is None:
                return
            step.status = "completed"
            step.latency_ms = latency_ms
            step.record_ref_type = record_ref_type
            step.record_ref_id = record_ref_id
            step.completed_at = utcnow()

    def fail_step(self, step_id: str, latency_ms: int, error_message: str) -> None:
        with self.session() as session:
            step = session.get(TraceStep, step_id)
            if step is None:
                return
            step.status = "error"
            step.latency_ms = latency_ms
            step.error_message = error_message
            step.completed_at = utcnow()

    def create_intent_record(
        self,
        request_id: str,
        input_text: str,
        intent: str,
        need_rag: bool,
        rewrite_query: str,
        model_output: dict,
    ) -> str:
        record_id = str(uuid4())
        record = IntentRecord(
            intent_record_id=record_id,
            request_id=request_id,
            input_text=input_text,
            intent=intent,
            need_rag=need_rag,
            rewrite_query=rewrite_query,
            model_output=model_output,
        )
        with self.session() as session:
            session.add(record)
        return record_id

    def create_retrieval_record(
        self,
        request_id: str,
        query: str,
        retrieved_ids: list[str],
    ) -> str:
        record_id = str(uuid4())
        record = RetrievalRecord(
            retrieval_record_id=record_id,
            request_id=request_id,
            query=query,
            retrieved_ids=retrieved_ids,
        )
        with self.session() as session:
            session.add(record)
        return record_id

    def create_generation_record(
        self,
        request_id: str,
        user_input: str,
        used_source_ids: list[str],
        llm_output: str,
    ) -> str:
        record_id = str(uuid4())
        record = GenerationRecord(
            generation_record_id=record_id,
            request_id=request_id,
            user_input=user_input,
            used_source_ids=used_source_ids,
            llm_output=llm_output,
        )
        with self.session() as session:
            session.add(record)
        return record_id

    def count_request_traces(self) -> int:
        with self.session() as session:
            return len(session.scalars(select(RequestTrace.request_id)).all())

    def list_request_traces(
        self,
        *,
        limit: int,
        offset: int,
        session_id: str | None = None,
        status: str | None = None,
    ) -> tuple[list[RequestTrace], int]:
        with self.session() as session:
            stmt = select(RequestTrace)
            if session_id is not None:
                stmt = stmt.where(RequestTrace.session_id == session_id)
            if status is not None:
                stmt = stmt.where(RequestTrace.status == status)

            total = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
            items = session.scalars(
                stmt.options(selectinload(RequestTrace.steps))
                .order_by(RequestTrace.created_at.desc(), RequestTrace.request_id.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            return items, int(total)

    def get_request_trace(self, request_id: str) -> RequestTrace | None:
        with self.session() as session:
            stmt = (
                select(RequestTrace)
                .where(RequestTrace.request_id == request_id)
                .options(selectinload(RequestTrace.steps))
            )
            return session.scalars(stmt).first()

    def get_intent_record(self, request_id: str) -> IntentRecord | None:
        with self.session() as session:
            stmt = select(IntentRecord).where(IntentRecord.request_id == request_id)
            return session.scalars(stmt).first()

    def get_retrieval_record(self, request_id: str) -> RetrievalRecord | None:
        with self.session() as session:
            stmt = select(RetrievalRecord).where(RetrievalRecord.request_id == request_id)
            return session.scalars(stmt).first()

    def get_generation_record(self, request_id: str) -> GenerationRecord | None:
        with self.session() as session:
            stmt = select(GenerationRecord).where(GenerationRecord.request_id == request_id)
            return session.scalars(stmt).first()

    def _ensure_sqlite_parent_dir(self) -> None:
        url = make_url(self._dsn)
        if url.drivername != "sqlite" or not url.database:
            return
        database_path = Path(url.database)
        if not database_path.is_absolute():
            database_path = Path.cwd() / database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
