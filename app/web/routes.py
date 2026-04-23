from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

router = APIRouter()


@router.get("/", include_in_schema=False)
async def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@router.get("/knowledge", include_in_schema=False)
async def knowledge_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "knowledge.html")


@router.get("/traces", include_in_schema=False)
async def traces_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin-traces.html")


@router.get("/admin", include_in_schema=False)
async def admin_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin-dashboard.html")


@router.get("/admin/traces", include_in_schema=False)
async def admin_traces_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin-traces.html")


@router.get("/admin/rag", include_in_schema=False)
async def admin_rag_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin-rag.html")


@router.get("/admin/compare", include_in_schema=False)
async def admin_compare_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin-compare.html")
