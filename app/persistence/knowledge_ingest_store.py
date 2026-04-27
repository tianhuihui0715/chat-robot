from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.persistence.models import Base, KnowledgeIngestDocumentRecord, KnowledgeIngestJobRecord
from app.schemas.knowledge import KnowledgeDocument


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PersistedKnowledgeIngestJob:
    job_id: str
    status: str
    submitted_documents: int
    processed_documents: int
    ingested_count: int
    total_documents: int | None
    total_chunks: int | None
    processed_chunks: int
    current_stage: str
    current_title: str | None
    document_ids: list[str]
    error: str | None
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    completed_at: datetime | None


class KnowledgeIngestStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._ensure_sqlite_parent_dir()
        connect_args = {"check_same_thread": False} if dsn.startswith("sqlite") else {}
        self._engine = create_engine(dsn, future=True, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def setup(self) -> None:
        Base.metadata.create_all(self._engine)

    def create_job(self, job_id: str, documents: list[KnowledgeDocument]) -> PersistedKnowledgeIngestJob:
        job = KnowledgeIngestJobRecord(
            job_id=job_id,
            status="queued",
            submitted_documents=len(documents),
            current_stage="queued",
            updated_at=utcnow(),
        )
        document_rows = [
            KnowledgeIngestDocumentRecord(
                ingest_document_id=str(uuid4()),
                job_id=job_id,
                title=document.title,
                content=document.content,
                metadata_json={
                    **document.metadata,
                    "knowledge_base_id": document.knowledge_base_id,
                    "knowledge_base_name": document.knowledge_base_name,
                },
                document_order=index,
            )
            for index, document in enumerate(documents, start=1)
        ]
        with self._session_factory() as session:
            session.add(job)
            session.add_all(document_rows)
            session.commit()
            session.refresh(job)
            return self._to_job(job)

    def get_job(self, job_id: str) -> PersistedKnowledgeIngestJob | None:
        with self._session_factory() as session:
            record = session.get(KnowledgeIngestJobRecord, job_id)
            if record is None:
                return None
            return self._to_job(record)

    def has_job(self, job_id: str) -> bool:
        with self._session_factory() as session:
            return session.get(KnowledgeIngestJobRecord, job_id) is not None

    def list_pending_job_ids(self) -> list[str]:
        with self._session_factory() as session:
            stmt = (
                select(KnowledgeIngestJobRecord.job_id)
                .where(KnowledgeIngestJobRecord.status.in_(("queued", "running")))
                .order_by(KnowledgeIngestJobRecord.created_at.asc())
            )
            return list(session.scalars(stmt).all())

    def list_active_jobs(self) -> list[PersistedKnowledgeIngestJob]:
        with self._session_factory() as session:
            stmt = (
                select(KnowledgeIngestJobRecord)
                .where(KnowledgeIngestJobRecord.status.in_(("queued", "running")))
                .order_by(KnowledgeIngestJobRecord.created_at.asc())
            )
            return [self._to_job(record) for record in session.scalars(stmt).all()]

    def get_latest_active_job(self) -> PersistedKnowledgeIngestJob | None:
        with self._session_factory() as session:
            stmt = (
                select(KnowledgeIngestJobRecord)
                .where(KnowledgeIngestJobRecord.status.in_(("queued", "running")))
                .order_by(KnowledgeIngestJobRecord.status.desc(), KnowledgeIngestJobRecord.created_at.asc())
            )
            record = session.scalars(stmt).first()
            if record is None:
                return None
            return self._to_job(record)

    def load_documents(self, job_id: str) -> list[KnowledgeDocument]:
        with self._session_factory() as session:
            stmt = (
                select(KnowledgeIngestDocumentRecord)
                .where(KnowledgeIngestDocumentRecord.job_id == job_id)
                .order_by(KnowledgeIngestDocumentRecord.document_order.asc())
            )
            rows = session.scalars(stmt).all()
            return [
                KnowledgeDocument(
                    title=row.title,
                    content=row.content,
                    metadata={key: str(value) for key, value in (row.metadata_json or {}).items()},
                    knowledge_base_id=str((row.metadata_json or {}).get("knowledge_base_id") or "default"),
                    knowledge_base_name=str((row.metadata_json or {}).get("knowledge_base_name") or "默认知识库"),
                )
                for row in rows
            ]

    def mark_job_running(self, job_id: str) -> PersistedKnowledgeIngestJob:
        with self._session_factory() as session:
            record = session.get(KnowledgeIngestJobRecord, job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status == "cancelled":
                raise RuntimeError("Ingest job has been cancelled.")
            now = utcnow()
            record.status = "running"
            record.current_stage = "preparing"
            record.started_at = record.started_at or now
            record.updated_at = now
            record.error = None
            session.commit()
            session.refresh(record)
            return self._to_job(record)

    def update_progress(
        self,
        job_id: str,
        *,
        current_stage: str | None = None,
        current_title: str | None = None,
        processed_documents: int | None = None,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
    ) -> PersistedKnowledgeIngestJob:
        with self._session_factory() as session:
            record = session.get(KnowledgeIngestJobRecord, job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status == "cancelled":
                raise RuntimeError("Ingest job has been cancelled.")
            if current_stage is not None:
                record.current_stage = current_stage
            if current_title is not None:
                record.current_title = current_title
            if processed_documents is not None:
                record.processed_documents = processed_documents
            if total_chunks is not None:
                record.total_chunks = total_chunks
            if processed_chunks is not None:
                record.processed_chunks = processed_chunks
            record.updated_at = utcnow()
            session.commit()
            session.refresh(record)
            return self._to_job(record)

    def mark_job_completed(
        self,
        job_id: str,
        *,
        document_ids: list[str],
        total_documents: int,
    ) -> PersistedKnowledgeIngestJob:
        with self._session_factory() as session:
            record = session.get(KnowledgeIngestJobRecord, job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status == "cancelled":
                return self._to_job(record)
            now = utcnow()
            record.status = "completed"
            record.current_stage = "completed"
            record.current_title = None
            record.processed_documents = record.submitted_documents
            record.ingested_count = len(document_ids)
            record.document_ids = document_ids
            record.total_documents = total_documents
            record.updated_at = now
            record.completed_at = now
            session.commit()
            session.refresh(record)
            return self._to_job(record)

    def mark_job_cancelled(self, job_id: str) -> PersistedKnowledgeIngestJob:
        with self._session_factory() as session:
            record = session.get(KnowledgeIngestJobRecord, job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status in {"completed", "failed", "cancelled"}:
                return self._to_job(record)
            now = utcnow()
            record.status = "cancelled"
            record.current_stage = "cancelled"
            record.error = "Cancelled by user."
            record.updated_at = now
            record.completed_at = now
            session.commit()
            session.refresh(record)
            return self._to_job(record)

    def mark_job_failed(self, job_id: str, error: str) -> PersistedKnowledgeIngestJob:
        with self._session_factory() as session:
            record = session.get(KnowledgeIngestJobRecord, job_id)
            if record is None:
                raise KeyError(job_id)
            now = utcnow()
            record.status = "failed"
            record.current_stage = "failed"
            record.error = error
            record.updated_at = now
            record.completed_at = now
            session.commit()
            session.refresh(record)
            return self._to_job(record)

    def reset_interrupted_jobs(self) -> None:
        with self._session_factory() as session:
            stmt = select(KnowledgeIngestJobRecord).where(KnowledgeIngestJobRecord.status == "running")
            for record in session.scalars(stmt).all():
                record.status = "queued"
                record.current_stage = "queued"
                record.error = None
                record.updated_at = utcnow()
            session.commit()

    def delete_document(self, document_id: str) -> bool:
        deleted = False
        with self._session_factory() as session:
            stmt = select(KnowledgeIngestJobRecord).order_by(KnowledgeIngestJobRecord.created_at.asc())
            for record in session.scalars(stmt).all():
                document_ids = list(record.document_ids or [])
                if document_id not in document_ids:
                    continue

                document_index = document_ids.index(document_id)
                document_order = document_index + 1

                doc_stmt = (
                    select(KnowledgeIngestDocumentRecord)
                    .where(KnowledgeIngestDocumentRecord.job_id == record.job_id)
                    .where(KnowledgeIngestDocumentRecord.document_order == document_order)
                )
                document_row = session.scalars(doc_stmt).first()
                if document_row is not None:
                    session.delete(document_row)

                record.document_ids = [value for value in document_ids if value != document_id]
                if record.ingested_count > 0:
                    record.ingested_count = max(0, record.ingested_count - 1)
                record.updated_at = utcnow()
                deleted = True
            if deleted:
                session.commit()
        return deleted

    @staticmethod
    def _to_job(record: KnowledgeIngestJobRecord) -> PersistedKnowledgeIngestJob:
        return PersistedKnowledgeIngestJob(
            job_id=record.job_id,
            status=record.status,
            submitted_documents=record.submitted_documents,
            processed_documents=record.processed_documents,
            ingested_count=record.ingested_count,
            total_documents=record.total_documents,
            total_chunks=record.total_chunks,
            processed_chunks=record.processed_chunks,
            current_stage=record.current_stage,
            current_title=record.current_title,
            document_ids=list(record.document_ids or []),
            error=record.error,
            created_at=record.created_at,
            started_at=record.started_at,
            updated_at=record.updated_at,
            completed_at=record.completed_at,
        )

    def _ensure_sqlite_parent_dir(self) -> None:
        url = make_url(self._dsn)
        if url.drivername != "sqlite" or not url.database:
            return
        database_path = Path(url.database)
        if not database_path.is_absolute():
            database_path = Path.cwd() / database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
