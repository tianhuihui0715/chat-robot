from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_container
from app.schemas.traces import TraceDetail, TraceListResponse
from app.services.container import ServiceContainer

router = APIRouter()


@router.get("/traces", response_model=TraceListResponse)
async def list_traces(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session_id: str | None = None,
    status: str | None = None,
    container: ServiceContainer = Depends(get_container),
) -> TraceListResponse:
    return container.trace_service.list_traces(
        page=page,
        page_size=page_size,
        session_id=session_id,
        status=status,
    )


@router.get("/traces/{request_id}", response_model=TraceDetail)
async def get_trace_detail(
    request_id: str,
    container: ServiceContainer = Depends(get_container),
) -> TraceDetail:
    trace = container.trace_service.get_trace_detail(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
