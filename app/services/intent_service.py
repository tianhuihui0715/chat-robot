from typing import Protocol

from app.schemas.chat import ChatMessage, IntentDecision
from app.schemas.inference import (
    InferenceIntentRequest,
    InferenceIntentResponse,
)


class IntentService(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def decide(self, messages: list[ChatMessage]) -> IntentDecision:
        ...


class MockIntentService:
    _rag_keywords = (
        "文档",
        "知识库",
        "部署",
        "说明",
        "配置",
        "接口",
        "怎么",
        "如何",
        "查询",
        "项目",
    )

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def decide(self, messages: list[ChatMessage]) -> IntentDecision:
        latest_user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        normalized = latest_user_message.strip()
        need_rag = any(keyword in normalized for keyword in self._rag_keywords)

        if need_rag:
            intent = "knowledge_qa"
            rationale = "命中了知识问答关键词，优先尝试走检索增强生成。"
        elif len(messages) > 1:
            intent = "follow_up"
            rationale = "检测到多轮上下文，先按追问处理。"
        else:
            intent = "chat"
            rationale = "未命中检索关键词，按普通对话处理。"

        return IntentDecision(
            intent=intent,
            need_rag=need_rag,
            rewrite_query=normalized,
            rationale=rationale,
        )


class RemoteIntentService:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        transport: object | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._client = None

    async def start(self) -> None:
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def decide(self, messages: list[ChatMessage]) -> IntentDecision:
        if self._client is None:
            raise RuntimeError("RemoteIntentService has not been started.")

        payload = InferenceIntentRequest(messages=messages)
        response = await self._client.post(
            "/intent",
            json=payload.model_dump(mode="json"),
        )
        response.raise_for_status()
        body = InferenceIntentResponse.model_validate(response.json())
        return body.decision
