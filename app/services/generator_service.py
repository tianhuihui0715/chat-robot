import asyncio
import contextlib
from dataclasses import dataclass
from typing import Protocol

from app.schemas.chat import ChatMessage, IntentDecision, SourceChunk
from app.schemas.inference import (
    InferenceGenerateRequest,
    InferenceGenerateResponse,
)


@dataclass
class GenerationRequest:
    messages: list[ChatMessage]
    intent: IntentDecision
    sources: list[SourceChunk]


class GenerationBackend(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def generate(self, request: GenerationRequest) -> str:
        ...


class MockGenerationBackend:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def generate(self, request: GenerationRequest) -> str:
        latest_user_message = next(
            (message.content for message in reversed(request.messages) if message.role == "user"),
            "",
        )

        if request.sources:
            source_titles = ", ".join(source.title for source in request.sources[:3])
            return (
                "这是骨架阶段的 mock 回答。"
                f"我会围绕你的问题“{latest_user_message}”进行回答，"
                f"并参考知识库片段：{source_titles}。"
            )

        return (
            "这是骨架阶段的 mock 回答。"
            f"当前没有命中知识库，我收到的问题是：{latest_user_message}"
        )


class RemoteGenerationBackend:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        max_new_tokens: int,
        transport: object | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_new_tokens = max_new_tokens
        self._transport = transport
        self._client = None
        self._httpx = None

    async def start(self) -> None:
        import httpx

        self._httpx = httpx
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def generate(self, request: GenerationRequest) -> str:
        if self._client is None:
            raise RuntimeError("RemoteGenerationBackend has not been started.")

        payload = InferenceGenerateRequest(
            messages=request.messages,
            intent=request.intent,
            sources=request.sources,
            max_new_tokens=self._max_new_tokens,
        )
        response = await self._client.post(
            "/generate",
            json=payload.model_dump(mode="json"),
        )
        response.raise_for_status()
        body = InferenceGenerateResponse.model_validate(response.json())
        return body.answer


@dataclass
class _GenerationJob:
    request: GenerationRequest
    future: asyncio.Future[str]


class QueuedGenerationService:
    def __init__(self, backend: GenerationBackend, maxsize: int = 100) -> None:
        self._backend = backend
        self._queue: asyncio.Queue[_GenerationJob] = asyncio.Queue(maxsize=maxsize)
        self._worker_task: asyncio.Task[None] | None = None

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        await self._backend.start()
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        await self._backend.stop()

    async def generate(self, request: GenerationRequest) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        await self._queue.put(_GenerationJob(request=request, future=future))
        return await future

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                result = await self._backend.generate(job.request)
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
            else:
                if not job.future.done():
                    job.future.set_result(result)
            finally:
                self._queue.task_done()
