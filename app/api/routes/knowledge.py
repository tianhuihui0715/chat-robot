from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_container
from app.schemas.knowledge import (
    KnowledgeDeleteResponse,
    KnowledgeDocumentSummary,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeIngestStatusResponse,
)
from app.services.container import ServiceContainer

router = APIRouter()


@router.post("/knowledge/ingest", response_model=KnowledgeIngestResponse)
async def ingest_documents(
    request: KnowledgeIngestRequest,
    container: ServiceContainer = Depends(get_container),
) -> KnowledgeIngestResponse:
    status = await container.knowledge_ingest_service.submit(request.documents)
    return KnowledgeIngestResponse(**status.model_dump())


@router.get("/knowledge/ingest/latest", response_model=KnowledgeIngestStatusResponse)
async def get_latest_ingest_status(
    container: ServiceContainer = Depends(get_container),
) -> KnowledgeIngestStatusResponse:
    status = container.knowledge_ingest_service.get_latest_active_job()
    if status is None:
        raise HTTPException(status_code=404, detail="No active ingest job.")
    return status


@router.get("/knowledge/ingest/{job_id}", response_model=KnowledgeIngestStatusResponse)
async def get_ingest_status(
    job_id: str,
    container: ServiceContainer = Depends(get_container),
) -> KnowledgeIngestStatusResponse:
    if not container.knowledge_ingest_service.has_job(job_id):
        raise HTTPException(status_code=404, detail="Ingest job not found.")
    return container.knowledge_ingest_service.get_status(job_id)


@router.get("/knowledge/documents", response_model=list[KnowledgeDocumentSummary])
async def list_documents(
    container: ServiceContainer = Depends(get_container),
) -> list[KnowledgeDocumentSummary]:
    documents = await run_in_threadpool(container.knowledge_base.list_documents)
    return [
        KnowledgeDocumentSummary(
            document_id=document.document_id,
            title=document.title,
            metadata=document.metadata,
        )
        for document in documents
    ]


@router.delete("/knowledge/documents/{document_id}", response_model=KnowledgeDeleteResponse)
async def delete_document(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
) -> KnowledgeDeleteResponse:
    vector_deleted = await run_in_threadpool(container.knowledge_base.delete_document, document_id)
    relational_deleted = await container.knowledge_ingest_service.delete_document(document_id)
    deleted = vector_deleted or relational_deleted
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return KnowledgeDeleteResponse(
        document_id=document_id,
        deleted=True,
        vector_deleted=vector_deleted,
        relational_deleted=relational_deleted,
        synchronized=True,
    )
