import asyncio
import contextlib
from dataclasses import dataclass
from typing import Protocol

from app.schemas.chat import ChatMessage, IntentDecision, SourceChunk


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
            source_titles = "、".join(source.title for source in request.sources[:3])
            return (
                f"这是骨架阶段的 mock 回答。"
                f"我会围绕你的问题“{latest_user_message}”进行回答，"
                f"并参考知识库片段：{source_titles}。"
            )

        return (
            f"这是骨架阶段的 mock 回答。"
            f"当前没有命中知识库，我收到的问题是：{latest_user_message}"
        )


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
