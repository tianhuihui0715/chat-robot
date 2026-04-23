from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_container
from app.schemas.admin import TraceDetailPreview, TracePageResponse
from app.schemas.traces import TraceDetail, TraceListResponse
from app.services.container import ServiceContainer

router = APIRouter()


@router.get("/traces", response_model=TracePageResponse)
async def list_traces(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session_id: str | None = None,
    status: str | None = None,
    container: ServiceContainer = Depends(get_container),
) -> TracePageResponse:
    response = container.trace_service.list_traces(
        page=page,
        page_size=page_size,
        session_id=session_id,
        status=status,
    )
    return TracePageResponse(**response.model_dump())


@router.get("/traces/{request_id}", response_model=TraceDetailPreview)
async def get_trace_detail(
    request_id: str,
    step_limit: int = Query(default=3, ge=1, le=100),
    view_all: bool = False,
    container: ServiceContainer = Depends(get_container),
) -> TraceDetailPreview:
    trace = container.trace_service.get_trace_detail(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    full_trace = TraceDetail.model_validate(trace.model_dump())
    has_more_steps = len(full_trace.steps) > step_limit
    output_truncated = False

    if not view_all:
        full_trace.steps = full_trace.steps[:step_limit]
        output_value = full_trace.final_output or ""
        if len(output_value) > 1200:
            full_trace.final_output = output_value[:1200] + "\n\n[已截断，点击查看全部展开]"
            output_truncated = True
        if full_trace.generation_record and len(full_trace.generation_record.llm_output) > 1200:
            full_trace.generation_record.llm_output = (
                full_trace.generation_record.llm_output[:1200] + "\n\n[已截断，点击查看全部展开]"
            )

    return TraceDetailPreview(
        trace=full_trace,
        step_limit=step_limit,
        has_more_steps=has_more_steps and not view_all,
        output_truncated=output_truncated and not view_all,
    )
