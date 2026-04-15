from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.schemas.knowledge import (
    KnowledgeDocument,
    KnowledgeDocumentSummary,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
)
from app.services.container import ServiceContainer

router = APIRouter()


@router.post("/knowledge/ingest", response_model=KnowledgeIngestResponse)
async def ingest_documents(
    request: KnowledgeIngestRequest,
    container: ServiceContainer = Depends(get_container),
) -> KnowledgeIngestResponse:
    ingested_ids = container.knowledge_base.add_documents(request.documents)
    return KnowledgeIngestResponse(
        ingested_count=len(ingested_ids),
        document_ids=ingested_ids,
        total_documents=container.knowledge_base.count,
    )


@router.get("/knowledge/documents", response_model=list[KnowledgeDocumentSummary])
async def list_documents(
    container: ServiceContainer = Depends(get_container),
) -> list[KnowledgeDocumentSummary]:
    return [
        KnowledgeDocumentSummary(
            document_id=document.document_id,
            title=document.title,
            metadata=document.metadata,
        )
        for document in container.knowledge_base.list_documents()
    ]
