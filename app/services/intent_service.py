from __future__ import annotations

import re
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
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def decide(self, messages: list[ChatMessage]) -> IntentDecision:
        return _heuristic_intent_decision(messages)


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


def _heuristic_intent_decision(messages: list[ChatMessage]) -> IntentDecision:
    decision = _rule_based_intent_decision(messages)
    if decision is not None:
        return decision

    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    if not latest_user_message:
        return _build_intent_decision(
            intent="chat",
            need_rag=False,
            rewrite_query="",
            rationale="没有可用的用户输入，回退为 chat。",
            messages=messages,
        )

    if _should_use_plan_execute(latest_user_message):
        return _build_intent_decision(
            intent="knowledge_qa",
            need_rag=True,
            rewrite_query=_normalize_knowledge_query(latest_user_message),
            rationale="命中复杂问题特征，回退为 knowledge_qa + plan_execute。",
            messages=messages,
        )

    return _build_intent_decision(
        intent="chat",
        need_rag=False,
        rewrite_query=latest_user_message,
        rationale="未命中明确规则，回退为 chat。",
        messages=messages,
    )


def _rule_based_intent_decision(messages: list[ChatMessage]) -> IntentDecision | None:
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    if not latest_user_message:
        return None

    if _matches_reject_rule(latest_user_message):
        return _build_intent_decision(
            intent="reject",
            need_rag=False,
            rewrite_query=latest_user_message,
            rationale="命中高风险请求规则，判定为 reject。",
            messages=messages,
        )

    if _matches_chat_rule(latest_user_message):
        return _build_intent_decision(
            intent="chat",
            need_rag=False,
            rewrite_query=latest_user_message,
            rationale="命中问候或闲聊规则，判定为 chat。",
            messages=messages,
        )

    if _looks_like_follow_up_by_rules(messages):
        expanded_query = _expand_follow_up_query(latest_user_message, messages)
        need_rag = _follow_up_needs_rag(expanded_query, messages)
        return _build_intent_decision(
            intent="follow_up",
            need_rag=need_rag,
            rewrite_query=expanded_query,
            rationale="命中上下文追问规则，判定为 follow_up。",
            messages=messages,
        )

    if _matches_task_rule(latest_user_message):
        return _build_intent_decision(
            intent="task",
            need_rag=False,
            rewrite_query=latest_user_message,
            rationale="命中任务执行规则，判定为 task。",
            messages=messages,
        )

    if _matches_knowledge_rule(latest_user_message) or _should_use_plan_execute(latest_user_message):
        return _build_intent_decision(
            intent="knowledge_qa",
            need_rag=True,
            rewrite_query=_normalize_knowledge_query(latest_user_message),
            rationale="命中知识问答或复杂问题规则，判定为 knowledge_qa。",
            messages=messages,
        )

    return None


def _build_intent_decision(
    *,
    intent: str,
    need_rag: bool,
    rewrite_query: str,
    rationale: str,
    messages: list[ChatMessage],
) -> IntentDecision:
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    should_clarify = False
    clarify_question = None
    if intent not in {"reject", "chat"}:
        clarify_question = _maybe_build_clarify_question(
            latest_user_message=latest_user_message,
            messages=messages,
            intent=intent,
        )
        should_clarify = clarify_question is not None

    if intent == "reject":
        return IntentDecision(
            intent="reject",
            need_rag=False,
            rewrite_query=rewrite_query,
            rationale=rationale,
            execution_mode="direct",
        )
    if should_clarify:
        return IntentDecision(
            intent=intent,
            need_rag=need_rag,
            rewrite_query=rewrite_query,
            rationale=rationale,
            execution_mode="direct",
            should_clarify=True,
            clarify_question=clarify_question,
        )
    if intent == "chat":
        return IntentDecision(
            intent="chat",
            need_rag=False,
            rewrite_query=rewrite_query,
            rationale=rationale,
            execution_mode="direct",
        )
    if need_rag:
        execution_mode = "plan_execute" if _should_use_plan_execute(rewrite_query or latest_user_message) else "rag"
        return IntentDecision(
            intent=intent,
            need_rag=True,
            rewrite_query=rewrite_query,
            rationale=rationale,
            execution_mode=execution_mode,
            candidate_tools=["retrieval.search", "answer.direct"] if execution_mode == "plan_execute" else [],
            planner_hint=_build_planner_hint(rewrite_query or latest_user_message) if execution_mode == "plan_execute" else None,
        )
    return IntentDecision(
        intent=intent,
        need_rag=False,
        rewrite_query=rewrite_query,
        rationale=rationale,
        execution_mode="direct",
    )


def _matches_reject_rule(text: str) -> bool:
    normalized = text.lower().replace(" ", "")
    return any(
        keyword in normalized
        for keyword in ("炸弹", "窃取密码", "窃取账号", "入侵服务器", "恶意脚本", "钓鱼话术")
    )


def _matches_chat_rule(text: str) -> bool:
    normalized = text.strip().lower()
    if len(normalized) > 30:
        return False
    return any(
        re.match(pattern, normalized)
        for pattern in (
            r"^(你好|您好|嗨|hi|hello)[，,。.!？?]*$",
            r"^(谢谢|感谢|辛苦了|多谢)[，,。.!？?]*$",
            r"^(再见|拜拜|晚安)[，,。.!？?]*$",
            r"^你是谁|能做什么.*$",
        )
    )


def _matches_task_rule(text: str) -> bool:
    normalized = text.strip()
    return any(
        re.match(pattern, normalized)
        for pattern in (
            r"^帮我(写|生成|整理|总结|改写|翻译|列出|制定|设计|规划).+",
            r"^请(帮我)?(写|生成|整理|总结|改写|翻译|列出|制定|设计|规划).+",
            r"^帮我做.+",
            r"^帮我把.+",
        )
    )


def _matches_knowledge_rule(text: str) -> bool:
    normalized = text.strip()
    knowledge_keywords = (
        "部署方式",
        "配置文件",
        "接口",
        "Qdrant",
        "怎么启动",
        "如何配置",
        "参数",
        "说明文档",
        "有哪些",
        "统计",
        "对比",
        "比较",
    )
    knowledge_patterns = (
        r"^.+(是什么|是做什么用的|有哪些|怎么配|放哪|放哪里|如何配置|怎么启动|部署方式).*$",
        r"^配置文件里.+应该填什么.*$",
        r"^.+接口.*(有哪些|是什么).*$",
    )
    return any(keyword in normalized for keyword in knowledge_keywords) or any(
        re.match(pattern, normalized) for pattern in knowledge_patterns
    )


def _looks_like_follow_up_by_rules(messages: list[ChatMessage]) -> bool:
    latest_user_message = next(
        (message.content for message in reversed(messages) if message.role == "user"),
        "",
    ).strip()
    if not latest_user_message:
        return False
    if not any(message.role in {"user", "assistant"} and message.content.strip() for message in messages[:-1]):
        return False
    return _looks_like_follow_up_fragment(latest_user_message)


def _follow_up_needs_rag(expanded_query: str, messages: list[ChatMessage]) -> bool:
    previous_user = ""
    for message in reversed(messages[:-1]):
        if message.role == "user" and message.content.strip():
            previous_user = message.content.strip()
            break
    haystack = f"{expanded_query}\n{previous_user}".lower()
    return any(
        keyword in haystack
        for keyword in ("文档", "知识库", "部署", "说明", "配置", "接口", "参数", "docker", "qdrant")
    )


def _expand_follow_up_query(candidate: str, messages: list[ChatMessage]) -> str:
    if not _looks_like_follow_up_fragment(candidate):
        return candidate
    previous_user = ""
    for message in reversed(messages[:-1]):
        if message.role == "user" and message.content.strip():
            previous_user = message.content.strip()
            break
    if not previous_user:
        return candidate
    return f"{previous_user}；补充问题：{candidate}"


def _looks_like_follow_up_fragment(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) <= 10:
        return True
    return any(
        normalized.startswith(marker)
        for marker in ("那", "这个", "那个", "继续", "然后", "详细说说", "为什么", "怎么配", "放哪")
    )


def _maybe_build_clarify_question(
    *,
    latest_user_message: str,
    messages: list[ChatMessage],
    intent: str,
) -> str | None:
    if intent in {"reject", "chat"}:
        return None
    if any(message.role in {"user", "assistant"} and message.content.strip() for message in messages[:-1]):
        return None
    normalized = latest_user_message.strip().lower()
    if not any(
        pattern in normalized
        for pattern in ("帮我看看", "帮我分析", "部署问题", "配置问题", "帮我排查", "看下这个", "看看这个")
    ):
        return None
    if any(keyword in normalized for keyword in ("部署", "docker", "wsl", "windows", "linux")):
        return "你想看哪种部署场景？可以补充一下运行环境、目标平台或具体报错。"
    if any(keyword in normalized for keyword in ("配置", "参数", "环境变量", "模型")):
        return "你想确认哪个配置项或运行环境？可以补充配置名、场景或报错信息。"
    return "可以再补充一下你的具体目标、使用环境或报错信息吗？这样我更容易继续分析。"


def _should_use_plan_execute(rewrite_query: str) -> bool:
    normalized = rewrite_query.replace(" ", "")
    if not normalized:
        return False
    markers = (
        "对比",
        "比较",
        "评估",
        "分析",
        "分别",
        "综合",
        "汇总",
        "差异",
        "共同",
        "都",
        "哪些",
        "统计",
        "交集",
        "排序",
        "分组",
        "去重",
    )
    hit_count = sum(1 for marker in markers if marker in normalized)
    multi_entity = any(separator in normalized for separator in ("和", "与", "及", "、", "并且"))
    return hit_count >= 2 or (multi_entity and hit_count >= 1)


def _build_planner_hint(query: str) -> str | None:
    normalized = query.replace(" ", "")
    if not normalized:
        return None
    if any(marker in normalized for marker in ("共同", "都", "交集", "同时出现", "并且", "也在")):
        return "问题更像集合交集，请优先拆出多个检索对象并保留共同信息。"
    if any(marker in normalized for marker in ("排序", "排名", "最多", "最少")):
        return "问题更像排序题，请拆成多个对象分别检索，再按数量或覆盖度排序。"
    if any(marker in normalized for marker in ("分组", "分别", "各自", "归类")):
        return "问题更像分组题，请拆成多个对象分别检索，并按对象分组组织答案。"
    if any(marker in normalized for marker in ("去重", "汇总", "合并")):
        return "问题更像去重汇总题，请拆成多个对象分别检索，再做去重合并。"
    if any(marker in normalized for marker in ("对比", "比较", "差异")):
        return "问题更像对比题，请拆成多个对象分别检索，再按差异或共同点组织答案。"
    if any(marker in normalized for marker in ("统计", "汇总", "列出全部", "综合")):
        return "问题需要汇总多个检索结果，请尽量拆成清晰的子查询。"
    return None


def _normalize_knowledge_query(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact or text
