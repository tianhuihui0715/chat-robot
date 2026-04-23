import asyncio
import contextlib
import json
from time import monotonic

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_container
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.container import ServiceContainer

router = APIRouter()


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    container: ServiceContainer = Depends(get_container),
) -> ChatResponse:
    return await container.chat_pipeline.run(request)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    container: ServiceContainer = Depends(get_container),
) -> StreamingResponse:
    async def event_stream():
        event_queue: asyncio.Queue[dict | None] = asyncio.Queue()
        started_at = monotonic()

        async def produce_events() -> None:
            try:
                async for event in container.chat_pipeline.run_stream(request):
                    await event_queue.put(event)
            except Exception as exc:
                await event_queue.put({"type": "error", "message": str(exc)})
            finally:
                await event_queue.put(None)

        producer = asyncio.create_task(produce_events())

        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield _sse_event(
                        {
                            "type": "status",
                            "stage": "thinking",
                            "message": "正在思考",
                            "elapsed_seconds": int(monotonic() - started_at),
                        }
                    )
                    continue

                if event is None:
                    break

                if event.get("type") == "status":
                    event["elapsed_seconds"] = int(monotonic() - started_at)
                yield _sse_event(event)

            yield _sse_event({"type": "done"})
        finally:
            if not producer.done():
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
