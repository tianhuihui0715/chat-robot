import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.api.deps import get_container
from app.schemas.admin import (
    RAGLabApplyResponse,
    RAGLabRequest,
    RAGLabSessionResponse,
    RAGCompareRequest,
    RAGCompareResponse,
    RAGEvaluationRequest,
    RAGEvaluationResponse,
    RAGRuntimeConfig,
    RAGRuntimeConfigUpdate,
)
from app.services.document_parsing_service import parse_lab_document
from app.services.container import ServiceContainer

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/rag/config", response_model=RAGRuntimeConfig)
async def get_rag_config(
    container: ServiceContainer = Depends(get_container),
) -> RAGRuntimeConfig:
    return container.get_rag_runtime_config()


@router.put("/rag/config", response_model=RAGRuntimeConfig)
async def update_rag_config(
    request: RAGRuntimeConfigUpdate,
    container: ServiceContainer = Depends(get_container),
) -> RAGRuntimeConfig:
    return container.update_rag_runtime_config(
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        candidate_multiplier=request.candidate_multiplier,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        reranker_enabled=request.reranker_enabled,
        llm_temperature=request.llm_temperature,
    )


@router.post("/rag/compare", response_model=RAGCompareResponse)
async def compare_rag_variants(
    request: RAGCompareRequest,
    container: ServiceContainer = Depends(get_container),
) -> RAGCompareResponse:
    return await container.compare_rag_variants(
        query=request.query,
        variants=request.variants,
        generate_answer=request.generate_answer,
    )


@router.post("/rag/evaluate", response_model=RAGEvaluationResponse)
async def evaluate_rag_variants(
    request: RAGEvaluationRequest,
    container: ServiceContainer = Depends(get_container),
) -> RAGEvaluationResponse:
    return await container.evaluate_rag_variants(
        cases=request.cases,
        variants=request.variants,
        generate_answer=request.generate_answer,
    )


@router.post("/rag/lab/run", response_model=RAGLabSessionResponse)
async def run_rag_lab(
    request: RAGLabRequest,
    container: ServiceContainer = Depends(get_container),
) -> RAGLabSessionResponse:
    try:
        return await container.rag_lab_service.run(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/rag/lab/run-upload", response_model=RAGLabSessionResponse)
async def run_rag_lab_upload(
    questions: str = Form(...),
    variants: str = Form(...),
    files: list[UploadFile] = File(...),
    container: ServiceContainer = Depends(get_container),
) -> RAGLabSessionResponse:
    try:
        parsed_questions = [line.strip() for line in questions.splitlines() if line.strip()]
        parsed_variants = json.loads(variants)
        parsed_documents = []
        for upload in files:
            payload = await upload.read()
            if not payload:
                continue
            parsed_documents.append(parse_lab_document(upload.filename or "uploaded-file", payload))
        request = RAGLabRequest(
            questions=parsed_questions,
            variants=parsed_variants,
            documents=parsed_documents,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid RAG lab input: {exc}") from exc

    try:
        return await container.rag_lab_service.run(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/rag/lab/sessions/{session_id}", response_model=RAGLabSessionResponse)
async def get_rag_lab_session(
    session_id: str,
    container: ServiceContainer = Depends(get_container),
) -> RAGLabSessionResponse:
    session = await container.rag_lab_service.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="RAG lab session not found")
    return session


@router.post("/rag/lab/sessions/{session_id}/apply/{variant_id}", response_model=RAGLabApplyResponse)
async def apply_rag_lab_variant(
    session_id: str,
    variant_id: str,
    container: ServiceContainer = Depends(get_container),
) -> RAGLabApplyResponse:
    try:
        return await container.apply_rag_lab_variant(session_id, variant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/rag/lab/sessions/{session_id}/export.xlsx")
async def export_rag_lab_excel(
    session_id: str,
    container: ServiceContainer = Depends(get_container),
) -> Response:
    try:
        content = await container.rag_lab_service.export_excel(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="RAG lab session not found") from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Excel export dependency missing: {exc.name}") from exc
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="rag-lab-{session_id}.xlsx"',
        },
    )


@router.get("/rag/lab/sessions/{session_id}/export.docx")
async def export_rag_lab_word(
    session_id: str,
    container: ServiceContainer = Depends(get_container),
) -> Response:
    try:
        content = await container.rag_lab_service.export_word(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="RAG lab session not found") from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Word export dependency missing: {exc.name}") from exc
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="rag-lab-{session_id}.docx"',
        },
    )
