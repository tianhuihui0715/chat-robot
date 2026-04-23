import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from app.schemas.chat import ChatMessage, IntentDecision, SourceChunk
from app.schemas.inference import InferenceGenerateRequest, InferenceGenerateResponse


@dataclass
class GenerationRequest:
    messages: list[ChatMessage]
    intent: IntentDecision
    sources: list[SourceChunk]
    temperature: float | None = None


class GenerationBackend(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def generate(self, request: GenerationRequest) -> str:
        ...

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
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
                f"我会围绕你的问题“{latest_user_message}”进行回答，并参考知识库片段：{source_titles}。"
            )

        return f"这是骨架阶段的 mock 回答。当前没有命中知识库，你的问题是：{latest_user_message}"

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        answer = await self.generate(request)
        for index in range(0, len(answer), 12):
            await asyncio.sleep(0.03)
            yield answer[index : index + 12]


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
            temperature=request.temperature,
        )
        response = await self._client.post("/generate", json=payload.model_dump(mode="json"))
        response.raise_for_status()
        body = InferenceGenerateResponse.model_validate(response.json())
        return body.answer

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError("RemoteGenerationBackend has not been started.")

        payload = InferenceGenerateRequest(
            messages=request.messages,
            intent=request.intent,
            sources=request.sources,
            max_new_tokens=self._max_new_tokens,
            temperature=request.temperature,
        )

        async with self._client.stream(
            "POST",
            "/generate/stream",
            json=payload.model_dump(mode="json"),
        ) as response:
            response.raise_for_status()
            async for payload in _iter_sse_payloads(response):
                event_type = payload.get("type")
                if event_type == "delta":
                    delta = str(payload.get("delta", ""))
                    if delta:
                        yield delta
                elif event_type == "error":
                    raise RuntimeError(str(payload.get("message", "Generation stream failed.")))


@dataclass
class _GenerationJob:
    request: GenerationRequest
    future: asyncio.Future[str]


@dataclass
class _GenerationStreamChunk:
    delta: str | None = None
    error: Exception | None = None
    done: bool = False


@dataclass
class _GenerationStreamJob:
    request: GenerationRequest
    queue: asyncio.Queue[_GenerationStreamChunk]


class QueuedGenerationService:
    def __init__(self, backend: GenerationBackend, maxsize: int = 100) -> None:
        self._backend = backend
        self._queue: asyncio.Queue[_GenerationJob | _GenerationStreamJob] = asyncio.Queue(
            maxsize=maxsize
        )
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

    async def generate_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        queue: asyncio.Queue[_GenerationStreamChunk] = asyncio.Queue()
        await self._queue.put(_GenerationStreamJob(request=request, queue=queue))

        while True:
            item = await queue.get()
            if item.error is not None:
                raise item.error
            if item.done:
                break
            if item.delta:
                yield item.delta

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if isinstance(job, _GenerationStreamJob):
                    await self._handle_stream_job(job)
                else:
                    result = await self._backend.generate(job.request)
                    if not job.future.done():
                        job.future.set_result(result)
            except Exception as exc:
                if isinstance(job, _GenerationStreamJob):
                    await job.queue.put(_GenerationStreamChunk(error=exc))
                elif not job.future.done():
                    job.future.set_exception(exc)
            finally:
                if isinstance(job, _GenerationStreamJob):
                    await job.queue.put(_GenerationStreamChunk(done=True))
                self._queue.task_done()

    async def _handle_stream_job(self, job: _GenerationStreamJob) -> None:
        async for delta in self._backend.generate_stream(job.request):
            if delta:
                await job.queue.put(_GenerationStreamChunk(delta=delta))


async def _iter_sse_payloads(response) -> AsyncIterator[dict]:
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        events = buffer.split("\n\n")
        buffer = events.pop() or ""
        for event in events:
            payload_line = next(
                (line for line in event.splitlines() if line.startswith("data: ")),
                None,
            )
            if payload_line is None:
                continue
            yield json.loads(payload_line[6:])
