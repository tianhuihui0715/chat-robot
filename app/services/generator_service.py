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
    max_new_tokens: int | None = None
    response_mode: str = "answer"
    system_prompt_override: str | None = None


def build_generation_prompt_messages(request: GenerationRequest) -> list[dict[str, str]]:
    if request.system_prompt_override:
        system_parts = [request.system_prompt_override.strip()]
    else:
        system_parts = [
            "你是一个本地部署的中文 AI 助手。",
            "请直接给出答案，不要输出思考过程，也不要输出 <think> 标签。",
        ]

    if request.sources:
        source_sections = []
        for index, source in enumerate(request.sources, start=1):
            citation_index = source.metadata.get("citation_index", str(index))
            source_sections.append(
                f"[{citation_index}] id={source.document_id} title={source.title}\n{source.content}"
            )
        system_parts.append(
            "回答时优先参考以下知识片段；如果依据不足，请明确说明。"
            "凡是依据知识片段生成的事实、结论或列表项，必须在对应句子末尾标注来源编号，格式为【1】、【2】。"
            "如果同一句同时依据多个来源，可以写成【1】【3】。\n\n"
            + "\n\n".join(source_sections)
        )

    extra_system_messages = [
        message.content.strip()
        for message in request.messages
        if message.role == "system" and message.content.strip()
    ]
    if extra_system_messages:
        system_parts.append("额外约束：\n" + "\n".join(extra_system_messages))

    prompt_messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]
    prompt_messages.extend(
        {"role": message.role, "content": message.content}
        for message in request.messages
        if message.role in {"user", "assistant"}
    )
    return prompt_messages


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
        if request.response_mode == "json":
            return _mock_json_response(request)

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
            max_new_tokens=request.max_new_tokens or self._max_new_tokens,
            temperature=request.temperature,
            response_mode=request.response_mode,
            system_prompt_override=request.system_prompt_override,
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
            max_new_tokens=request.max_new_tokens or self._max_new_tokens,
            temperature=request.temperature,
            response_mode=request.response_mode,
            system_prompt_override=request.system_prompt_override,
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


def _mock_json_response(request: GenerationRequest) -> str:
    latest_user_message = next(
        (message.content for message in reversed(request.messages) if message.role == "user"),
        "",
    ).strip()
    normalized = latest_user_message.replace(" ", "")
    system_prompt = (request.system_prompt_override or "").lower()

    if "子任务意图识别器" in system_prompt or "task intent json" in system_prompt:
        latest_task_message = latest_user_message.replace(" ", "")
        if "前置结果" in latest_user_message and any(
            token in latest_task_message
            for token in ("比较", "交集", "共同", "排序", "分组", "去重", "汇总")
        ):
            return json.dumps(
                {
                    "intent": "aggregation",
                    "query": request.intent.rewrite_query or latest_user_message,
                    "source_hint": "",
                    "target": "",
                    "knowledge_base_id": "",
                    "reason": "当前任务主要基于前置结果做聚合处理。",
                },
                ensure_ascii=False,
            )
        if any(token in latest_task_message for token in ("列出", "统计", "出现过", "抽取", "提取", "哪些")):
            source_hint = ""
            if "射雕" in latest_task_message:
                source_hint = "射雕英雄传"
            elif "神雕" in latest_task_message:
                source_hint = "神雕侠侣"
            target = "武功" if any(token in latest_task_message for token in ("武功", "功夫")) else ""
            return json.dumps(
                {
                    "intent": "extraction",
                    "query": request.intent.rewrite_query or latest_user_message,
                    "source_hint": source_hint,
                    "target": target,
                    "knowledge_base_id": "default" if source_hint else "",
                    "reason": "当前任务需要从知识库中抽取特定对象列表。",
                },
                ensure_ascii=False,
            )
        if any(token in latest_task_message for token in ("文档", "文件", "资料", "配置", "接口", "部署")):
            return json.dumps(
                {
                    "intent": "retrieval",
                    "query": request.intent.rewrite_query or latest_user_message,
                    "source_hint": "",
                    "target": "",
                    "knowledge_base_id": "",
                    "reason": "当前任务更适合普通知识检索。",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "intent": "direct",
                "query": request.intent.rewrite_query or latest_user_message,
                "source_hint": "",
                "target": "",
                "knowledge_base_id": "",
                "reason": "当前任务不需要额外检索。",
            },
            ensure_ascii=False,
        )

    if "工具选择器" in system_prompt or "tool json" in system_prompt:
        latest_task_message = latest_user_message.replace(" ", "")
        if "前置结果" in latest_user_message and any(
            token in latest_task_message
            for token in ("比较", "交集", "共同", "排序", "分组", "去重", "汇总")
        ):
            return json.dumps(
                {
                    "tool": "answer.direct",
                    "arguments": {
                        "query": request.intent.rewrite_query or latest_user_message,
                    },
                    "reason": "当前任务主要基于前置结果推导，不需要额外工具。",
                },
                ensure_ascii=False,
            )
        if any(token in latest_task_message for token in ("文档", "文件", "资料")):
            return json.dumps(
                {
                    "tool": "kb.document_lookup",
                    "arguments": {"keyword": request.intent.rewrite_query or latest_user_message},
                    "reason": "任务更像文档定位，优先查文档列表。",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "tool": "retrieval.search",
                "arguments": {
                    "query": request.intent.rewrite_query or latest_user_message,
                },
                "reason": "当前任务需要先检索知识库片段。",
            },
            ensure_ascii=False,
        )

    if "知识库选择器" in system_prompt or "knowledge base json" in system_prompt:
        latest_task_message = latest_user_message.replace(" ", "")
        if any(token in latest_task_message for token in ("射雕", "神雕", "武功", "功夫")):
            return json.dumps(
                {
                    "knowledge_base_id": "default",
                    "reason": "任务涉及当前默认小说知识库中的作品和武功内容。",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "knowledge_base_id": "default",
                "reason": "默认选择当前最相关的知识库。",
            },
            ensure_ascii=False,
        )

    if "检索规划器" in system_prompt or "plan json" in system_prompt:
        if _looks_like_planner_question(normalized):
            merge_strategy, answer_style = _mock_planner_strategy(normalized)
            return json.dumps(
                {
                    "mode": "plan_rag",
                    "reason": "问题涉及多个对象和共同点统计，适合拆成多次检索后汇总。",
                    "primary_query": request.intent.rewrite_query or latest_user_message,
                    "subqueries": _build_mock_subqueries(request.intent.rewrite_query or latest_user_message),
                    "merge_strategy": merge_strategy,
                    "answer_style": answer_style,
                },
                ensure_ascii=False,
            )
        if any(token in normalized for token in ("文档", "配置", "部署", "接口", "知识库", "trace")):
            return json.dumps(
                {
                    "mode": "single_retrieval",
                    "reason": "问题依赖项目知识或配置细节，单次检索即可。",
                    "primary_query": request.intent.rewrite_query or latest_user_message,
                    "subqueries": [],
                    "merge_strategy": "union",
                    "answer_style": "summary",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "mode": "answer_direct",
                "reason": "问题可以直接回答。",
                "primary_query": request.intent.rewrite_query or latest_user_message,
                "subqueries": [],
                "merge_strategy": "union",
                "answer_style": "summary",
            },
            ensure_ascii=False,
        )

    if "子任务执行器" in system_prompt or "subtask json" in system_prompt:
        items = _extract_mock_items_from_sources(request.sources)
        return json.dumps(
            {
                "items": items,
                "count": len(items),
                "summary": "已根据子任务来源提取候选结果。",
                "notes": "mock 子任务执行结果。",
            },
            ensure_ascii=False,
        )

    if any(token in normalized for token in ("文档", "配置", "部署", "接口", "知识库", "trace")):
        return json.dumps(
            {
                "action": "retrieval.search",
                "query": request.intent.rewrite_query or latest_user_message,
                "reason": "问题依赖项目知识或配置细节，先检索再回答。",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "action": "answer.direct",
            "query": request.intent.rewrite_query or latest_user_message,
            "reason": "问题可以直接回答，不需要额外检索。",
        },
        ensure_ascii=False,
    )


def _looks_like_planner_question(normalized_text: str) -> bool:
    markers = ("对比", "比较", "差异", "统计", "汇总", "共同", "都", "分别", "同时", "并且", "也在", "排序", "分组", "去重")
    hit_count = sum(1 for marker in markers if marker in normalized_text)
    multi_entity = any(separator in normalized_text for separator in ("和", "与", "及", "、", "并且"))
    return hit_count >= 2 or (multi_entity and hit_count >= 1)


def _mock_planner_strategy(normalized_text: str) -> tuple[str, str]:
    if any(marker in normalized_text for marker in ("排序", "排名", "最多", "最少")):
        return "rank", "table"
    if any(marker in normalized_text for marker in ("分组", "分别", "各自", "归类")):
        return "group_by", "table"
    if any(marker in normalized_text for marker in ("去重", "汇总", "合并")):
        return "dedupe_union", "list"
    if any(marker in normalized_text for marker in ("共同", "都", "交集", "同时", "并且", "也在")):
        return "intersection", "list"
    return "union", "summary"


def _build_mock_subqueries(query: str) -> list[str]:
    separators = ("和", "与", "及", "、", "并且")
    for separator in separators:
        if separator in query:
            left, right = query.split(separator, 1)
            left = left.strip(" ，,。；;：:")
            right = right.strip(" ，,。；;：:")
            if left and right:
                return [left, right]
    return [query]


def _extract_mock_items_from_sources(sources: list[SourceChunk]) -> list[str]:
    items: list[str] = []
    for source in sources:
        for token in ("九隂神功", "九阴真经", "降龙十八掌", "黯然销魂掌", "玉女剑法", "独孤九剑", "空明拳", "蛤蟆功"):
            if token in source.content and token not in items:
                items.append(token)
    return items[:12]
