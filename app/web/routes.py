from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

router = APIRouter()


@router.get("/", include_in_schema=False)
async def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@router.get("/traces", include_in_schema=False)
async def traces_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "traces.html")

