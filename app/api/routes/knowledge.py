from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_container
from app.schemas.knowledge import (
    KnowledgeDeleteResponse,
    KnowledgeDocument,
    KnowledgeDocumentSummary,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeIngestStatusResponse,
)
from app.services.container import ServiceContainer
from app.services.document_parsing_service import parse_lab_document

router = APIRouter()


@router.post("/knowledge/ingest", response_model=KnowledgeIngestResponse)
async def ingest_documents(
    request: KnowledgeIngestRequest,
    container: ServiceContainer = Depends(get_container),
) -> KnowledgeIngestResponse:
    documents = [
        document.model_copy(
            update={
                "knowledge_base_id": (
                    request.knowledge_base_id
                    if request.knowledge_base_id != "default"
                    else document.knowledge_base_id
                ),
                "knowledge_base_name": (
                    request.knowledge_base_name
                    if request.knowledge_base_name != "默认知识库"
                    else document.knowledge_base_name
                ),
            }
        )
        for document in request.documents
    ]
    status = await container.knowledge_ingest_service.submit(documents)
    return KnowledgeIngestResponse(**status.model_dump())


@router.post("/knowledge/ingest/upload", response_model=KnowledgeIngestResponse)
async def ingest_uploaded_documents(
    files: list[UploadFile] | None = File(default=None),
    knowledge_base_id: str = Form(default="default"),
    knowledge_base_name: str = Form(default="默认知识库"),
    manual_title: str | None = Form(default=None),
    manual_content: str | None = Form(default=None),
    container: ServiceContainer = Depends(get_container),
) -> KnowledgeIngestResponse:
    documents: list[KnowledgeDocument] = []

    for upload in files or []:
        payload = await upload.read()
        if not payload:
            continue
        try:
            parsed = parse_lab_document(upload.filename or "uploaded-file", payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        documents.append(
            KnowledgeDocument(
                title=parsed.title,
                content=parsed.content,
                knowledge_base_id=knowledge_base_id.strip() or "default",
                knowledge_base_name=knowledge_base_name.strip() or "默认知识库",
                metadata={
                    "source": "upload",
                    "file_name": upload.filename or parsed.title,
                },
            )
        )

    if manual_content and manual_content.strip():
        documents.append(
            KnowledgeDocument(
                title=(manual_title or "").strip() or "manual-entry",
                content=manual_content.strip(),
                knowledge_base_id=knowledge_base_id.strip() or "default",
                knowledge_base_name=knowledge_base_name.strip() or "默认知识库",
                metadata={"source": "manual"},
            )
        )

    if not documents:
        raise HTTPException(status_code=400, detail="No valid document content provided.")

    status = await container.knowledge_ingest_service.submit(documents)
    return KnowledgeIngestResponse(**status.model_dump())


@router.get("/knowledge/ingest", response_model=list[KnowledgeIngestStatusResponse])
async def list_active_ingest_jobs(
    container: ServiceContainer = Depends(get_container),
) -> list[KnowledgeIngestStatusResponse]:
    return container.knowledge_ingest_service.list_active_jobs()


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


@router.post("/knowledge/ingest/{job_id}/cancel", response_model=KnowledgeIngestStatusResponse)
async def cancel_ingest_job(
    job_id: str,
    container: ServiceContainer = Depends(get_container),
) -> KnowledgeIngestStatusResponse:
    if not container.knowledge_ingest_service.has_job(job_id):
        raise HTTPException(status_code=404, detail="Ingest job not found.")
    return await container.knowledge_ingest_service.cancel_job(job_id)


@router.get("/knowledge/documents", response_model=list[KnowledgeDocumentSummary])
async def list_documents(
    container: ServiceContainer = Depends(get_container),
) -> list[KnowledgeDocumentSummary]:
    documents = await run_in_threadpool(container.knowledge_base.list_documents)
    return [
        KnowledgeDocumentSummary(
            document_id=document.document_id,
            title=document.title,
            knowledge_base_id=document.metadata.get("knowledge_base_id", "default"),
            knowledge_base_name=document.metadata.get("knowledge_base_name", "默认知识库"),
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
