from __future__ import annotations

import asyncio
import contextlib
import logging
from uuid import uuid4

from app.persistence.knowledge_ingest_store import KnowledgeIngestStore, PersistedKnowledgeIngestJob
from app.schemas.knowledge import KnowledgeDocument, KnowledgeIngestStatusResponse
from app.services.knowledge_base import KnowledgeBase


logger = logging.getLogger(__name__)


class KnowledgeIngestService:
    def __init__(self, knowledge_base: KnowledgeBase, store: KnowledgeIngestStore) -> None:
        self._knowledge_base = knowledge_base
        self._store = store
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._queued_job_ids: set[str] = set()
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await asyncio.to_thread(self._store.setup)
        await asyncio.to_thread(self._store.reset_interrupted_jobs)
        await self._recover_pending_jobs()
        await self._ensure_worker()

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

    async def submit(self, documents: list[KnowledgeDocument]) -> KnowledgeIngestStatusResponse:
        await self._ensure_worker()
        job_id = str(uuid4())
        await asyncio.to_thread(self._store.create_job, job_id, documents)
        await self._enqueue_job(job_id)
        return self.get_status(job_id)

    def get_status(self, job_id: str) -> KnowledgeIngestStatusResponse:
        job = self._store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return self._to_status(job)

    def has_job(self, job_id: str) -> bool:
        return self._store.has_job(job_id)

    def get_latest_active_job(self) -> KnowledgeIngestStatusResponse | None:
        job = self._store.get_latest_active_job()
        if job is None:
            return None
        return self._to_status(job)

    async def delete_document(self, document_id: str) -> bool:
        return await asyncio.to_thread(self._store.delete_document, document_id)

    async def _recover_pending_jobs(self) -> None:
        pending_job_ids = await asyncio.to_thread(self._store.list_pending_job_ids)
        for job_id in pending_job_ids:
            await self._enqueue_job(job_id)
        if pending_job_ids:
            logger.info("Recovered %s pending knowledge ingest jobs.", len(pending_job_ids))

    async def _enqueue_job(self, job_id: str) -> None:
        if job_id in self._queued_job_ids:
            return
        self._queued_job_ids.add(job_id)
        await self._queue.put(job_id)

    async def _ensure_worker(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        if self._worker_task is not None and self._worker_task.done():
            with contextlib.suppress(Exception):
                self._worker_task.result()
        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name="knowledge-ingest-worker",
        )
        logger.info("Knowledge ingest worker started.")

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            self._queued_job_ids.discard(job_id)
            logger.info("Knowledge ingest job %s dequeued.", job_id)
            try:
                await asyncio.to_thread(self._store.mark_job_running, job_id)
                documents = await asyncio.to_thread(self._store.load_documents, job_id)
                logger.info("Knowledge ingest job %s started with %s documents.", job_id, len(documents))

                document_ids = await asyncio.to_thread(
                    self._knowledge_base.add_documents,
                    documents,
                    self._build_progress_callback(job_id),
                )
                total_documents = await asyncio.to_thread(lambda: self._knowledge_base.count)
                await asyncio.to_thread(
                    self._store.mark_job_completed,
                    job_id,
                    document_ids=document_ids,
                    total_documents=total_documents,
                )
                logger.info(
                    "Knowledge ingest job %s completed: %s documents ingested.",
                    job_id,
                    len(document_ids),
                )
            except Exception as exc:
                await asyncio.to_thread(self._store.mark_job_failed, job_id, str(exc))
                logger.exception("Knowledge ingest job %s failed.", job_id)
            finally:
                self._queue.task_done()

    def _build_progress_callback(self, job_id: str):
        def _callback(**kwargs) -> None:
            self._store.update_progress(job_id, **kwargs)

        return _callback

    @staticmethod
    def _to_status(job: PersistedKnowledgeIngestJob) -> KnowledgeIngestStatusResponse:
        return KnowledgeIngestStatusResponse(
            job_id=job.job_id,
            status=job.status,
            submitted_documents=job.submitted_documents,
            processed_documents=job.processed_documents,
            ingested_count=job.ingested_count,
            total_chunks=job.total_chunks,
            processed_chunks=job.processed_chunks,
            current_stage=job.current_stage,
            current_title=job.current_title,
            document_ids=job.document_ids,
            total_documents=job.total_documents,
            error=job.error,
        )
