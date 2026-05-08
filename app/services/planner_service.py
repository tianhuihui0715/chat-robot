from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from app.schemas.chat import ChatMessage, IntentDecision, PlanTask
from app.services.generator_service import GenerationRequest, QueuedGenerationService

PlannerMode = Literal["answer_direct", "single_retrieval", "plan_rag"]
MergeStrategy = Literal["union", "intersection", "compare", "dedupe_union", "rank", "group_by"]
AnswerStyle = Literal["summary", "list", "table"]
PlannerTaskType = Literal["direct", "single_retrieval", "intersection", "compare", "dedupe_union", "rank", "group_by"]


@dataclass
class PlannerPlan:
    mode: PlannerMode
    reason: str
    primary_query: str
    subqueries: list[str]
    merge_strategy: MergeStrategy
    answer_style: AnswerStyle
    planner_model: str = "default"
    planner_source: str = "model"
    tasks: list[PlanTask] | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    raw_output: str | None = None


class PlannerService:
    def __init__(
        self,
        generation_service: QueuedGenerationService,
        *,
        enabled: bool,
        max_new_tokens: int = 256,
        model_label: str = "default",
    ) -> None:
        self._generation_service = generation_service
        self._enabled = enabled
        self._max_new_tokens = max_new_tokens
        self._model_label = model_label

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def plan(
        self,
        *,
        messages: list[ChatMessage],
        decision: IntentDecision,
    ) -> PlannerPlan:
        latest_user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        ).strip()
        fallback_query = decision.rewrite_query or latest_user_message

        if not self._enabled:
            return _fallback_plan(fallback_query, self._model_label)

        system_prompt = _build_planner_prompt(decision)
        user_prompt = _build_planner_user_prompt(latest_user_message, fallback_query)
        raw_output = await self._generation_service.generate(
            GenerationRequest(
                messages=[
                    ChatMessage(
                        role="user",
                        content=user_prompt,
                    )
                ],
                intent=decision,
                sources=[],
                temperature=0.0,
                max_new_tokens=self._max_new_tokens,
                response_mode="json",
                system_prompt_override=system_prompt,
            )
        )
        plan = _parse_planner_plan(raw_output, fallback_query)
        guardrail_reason = None
        guardrail_source = None
        if plan.reason.startswith("planner 未返回合法 JSON"):
            guardrail_reason = plan.reason
            guardrail_source = "guardrail_invalid_json"
        elif _should_upgrade_to_guardrail_plan(
            plan=plan,
            query=fallback_query,
            planner_hint=decision.planner_hint,
        ):
            guardrail_reason = "planner 给出了过于保守的单检索计划。"
            guardrail_source = "guardrail_override_conservative"
        if guardrail_reason is not None:
            guardrail_plan = _build_guardrail_plan(
                query=fallback_query,
                planner_hint=decision.planner_hint,
                fallback_reason=guardrail_reason,
                guardrail_source=guardrail_source or "guardrail",
            )
            if guardrail_plan is not None:
                plan = guardrail_plan
        plan.planner_model = self._model_label
        plan.system_prompt = system_prompt
        plan.user_prompt = user_prompt
        plan.raw_output = raw_output
        return plan


def _build_planner_prompt(decision: IntentDecision) -> str:
    knowledge_base_hint = ""
    if decision.knowledge_base_name:
        knowledge_base_hint = (
            f"\n如果需要检索，优先围绕知识库“{decision.knowledge_base_name}”拆解任务。"
        )
    planner_hint = f"\n额外提示：{decision.planner_hint}" if decision.planner_hint else ""
    return (
        "你是一个中文任务规划器。你只能输出一行 JSON，禁止输出解释、前后缀、代码块、空行或任何额外文字。\n"
        "固定输出字段只有 task_type 和 tasks。\n"
        "task_type 只允许：direct、single_retrieval、intersection、compare、dedupe_union、rank、group_by。\n"
        "tasks 必须是对象数组；每一项都只允许 task_id、goal、depends_on 三个字段。\n"
        "goal 是可执行的子任务描述；depends_on 是字符串数组，表示该任务依赖哪些前置 task_id；最多 4 项任务。\n"
        "如果问题只需要一次检索，输出 single_retrieval，并给出 1 个独立任务。\n"
        "如果问题涉及共同项、都出现过、交集，输出 intersection。\n"
        "如果问题涉及对比、哪个更多、差异，输出 compare。\n"
        "如果问题涉及排序、排名，输出 rank。\n"
        "如果问题涉及分组、分别说明，输出 group_by。\n"
        "如果问题涉及去重汇总，输出 dedupe_union。\n"
        "如果问题不需要检索，输出 direct，tasks 为空数组。\n"
        "独立任务的 depends_on 必须是空数组。\n"
        "如果某个任务要等前面任务结果出来后才能执行，才给它填写 depends_on。\n"
        "示例1：{\"task_type\":\"single_retrieval\",\"tasks\":[{\"task_id\":\"task_1\",\"goal\":\"说明这个项目的部署方式\",\"depends_on\":[]}]} \n"
        "示例2：{\"task_type\":\"intersection\",\"tasks\":[{\"task_id\":\"task_1\",\"goal\":\"列出射雕英雄传中出现过的武功\",\"depends_on\":[]},{\"task_id\":\"task_2\",\"goal\":\"列出神雕侠侣中出现过的武功\",\"depends_on\":[]}]} \n"
        "示例3：{\"task_type\":\"compare\",\"tasks\":[{\"task_id\":\"task_1\",\"goal\":\"列出射雕英雄传中出现过的武功\",\"depends_on\":[]},{\"task_id\":\"task_2\",\"goal\":\"列出神雕侠侣中出现过的武功\",\"depends_on\":[]},{\"task_id\":\"task_3\",\"goal\":\"比较 task_1 和 task_2 的结果\",\"depends_on\":[\"task_1\",\"task_2\"]}]} \n"
        "不要遗漏大括号，不要输出多余字段。"
        f"{knowledge_base_hint}{planner_hint}"
    )


def _build_planner_user_prompt(latest_user_message: str, fallback_query: str) -> str:
    if fallback_query and fallback_query != latest_user_message:
        return (
            f"问题：{latest_user_message}\n"
            f"检索改写：{fallback_query}\n"
            "请只输出 plan JSON。"
        )
    return f"问题：{fallback_query or latest_user_message}\n请只输出 plan JSON。"


def _fallback_plan(fallback_query: str, model_label: str) -> PlannerPlan:
    return PlannerPlan(
        mode="single_retrieval",
        reason="planner 未启用，回退到单次检索。",
        primary_query=fallback_query,
        subqueries=[],
        merge_strategy="union",
        answer_style="summary",
        planner_model=model_label,
        planner_source="disabled_fallback",
        tasks=[],
    )


def _parse_planner_plan(raw_output: str, fallback_query: str) -> PlannerPlan:
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in planner output.")
        payload = json.loads(raw_output[start : end + 1])
    except Exception:
        return PlannerPlan(
            mode="single_retrieval",
            reason="planner 未返回合法 JSON，回退到单次检索。",
            primary_query=fallback_query,
            subqueries=[],
            merge_strategy="union",
            answer_style="summary",
            planner_model="default",
            planner_source="invalid_json_fallback",
            tasks=[],
        )

    task_style_plan = _parse_task_style_payload(payload, fallback_query)
    if task_style_plan is not None:
        return task_style_plan

    mode = str(payload.get("mode", "")).strip()
    reason = str(payload.get("reason", "")).strip() or "规划器未提供原因。"
    primary_query = str(payload.get("primary_query", "")).strip() or fallback_query
    subqueries = [
        str(item).strip()
        for item in payload.get("subqueries", [])
        if str(item).strip()
    ]
    merge_strategy = str(payload.get("merge_strategy", "union")).strip()
    answer_style = str(payload.get("answer_style", "summary")).strip()

    if mode not in {"answer_direct", "single_retrieval", "plan_rag"}:
        mode = "single_retrieval"
    if merge_strategy not in {"union", "intersection", "compare", "dedupe_union", "rank", "group_by"}:
        merge_strategy = "union"
    if answer_style not in {"summary", "list", "table"}:
        answer_style = "summary"

    if mode == "plan_rag" and not subqueries:
        subqueries = [primary_query]
        mode = "single_retrieval"

    return PlannerPlan(
        mode=mode,
        reason=reason,
        primary_query=primary_query,
        subqueries=subqueries[:3],
        merge_strategy=merge_strategy,
        answer_style=answer_style,
        planner_source="model",
        tasks=[
            PlanTask(
                task_id=f"task_{index}",
                goal=query,
                query=query,
                depends_on=[],
            )
            for index, query in enumerate(subqueries[:3], start=1)
        ],
    )


def _parse_task_style_payload(
    payload: dict[str, object],
    fallback_query: str,
) -> PlannerPlan | None:
    task_type = str(payload.get("task_type", "")).strip()
    if task_type not in {
        "direct",
        "single_retrieval",
        "intersection",
        "compare",
        "dedupe_union",
        "rank",
        "group_by",
    }:
        return None

    parsed_tasks: list[PlanTask] = []
    for index, raw_task in enumerate(payload.get("tasks", []), start=1):
        if isinstance(raw_task, str):
            goal = raw_task.strip()
            if goal:
                parsed_tasks.append(
                    PlanTask(
                        task_id=f"task_{index}",
                        goal=goal,
                        query=goal,
                        depends_on=[],
                    )
                )
            continue
        if not isinstance(raw_task, dict):
            continue
        goal = str(raw_task.get("goal", "")).strip()
        if not goal:
            continue
        task_id = str(raw_task.get("task_id", "")).strip() or f"task_{index}"
        query = str(raw_task.get("query", "")).strip() or goal
        depends_on = [
            str(item).strip()
            for item in raw_task.get("depends_on", [])
            if str(item).strip()
        ]
        parsed_tasks.append(
            PlanTask(
                task_id=task_id,
                goal=goal,
                query=query,
                depends_on=depends_on,
            )
        )
    parsed_tasks = parsed_tasks[:4]
    if task_type == "direct":
        return PlannerPlan(
            mode="answer_direct",
            reason="规划器判定该问题可直接回答。",
            primary_query=fallback_query,
            subqueries=[],
            merge_strategy="union",
            answer_style="summary",
            planner_source="model",
            tasks=[],
        )
    if task_type == "single_retrieval":
        primary_query = parsed_tasks[0].query if parsed_tasks else fallback_query
        return PlannerPlan(
            mode="single_retrieval",
            reason="规划器判定该问题只需要一次检索。",
            primary_query=primary_query,
            subqueries=[],
            merge_strategy="union",
            answer_style="summary",
            planner_source="model",
            tasks=parsed_tasks,
        )
    if not parsed_tasks:
        return PlannerPlan(
            mode="single_retrieval",
            reason="规划器未给出可执行任务，回退到单次检索。",
            primary_query=fallback_query,
            subqueries=[],
            merge_strategy="union",
            answer_style="summary",
            planner_source="invalid_json_fallback",
            tasks=[],
        )
    merge_strategy: MergeStrategy = task_type  # type: ignore[assignment]
    answer_style: AnswerStyle = "table" if task_type in {"rank", "group_by"} else "list"
    return PlannerPlan(
        mode="plan_rag",
        reason=_task_type_reason(task_type),
        primary_query=fallback_query,
        subqueries=[task.query for task in parsed_tasks],
        merge_strategy=merge_strategy,
        answer_style=answer_style,
        planner_source="model",
        tasks=parsed_tasks,
    )


def _task_type_reason(task_type: PlannerTaskType) -> str:
    mapping = {
        "direct": "规划器判定该问题可直接回答。",
        "single_retrieval": "规划器判定该问题只需要一次检索。",
        "intersection": "规划器判定该问题需要多个任务后求交集。",
        "compare": "规划器判定该问题需要多个任务后做对比。",
        "dedupe_union": "规划器判定该问题需要多个任务后去重汇总。",
        "rank": "规划器判定该问题需要多个任务后排序。",
        "group_by": "规划器判定该问题需要多个任务后分组展示。",
    }
    return mapping[task_type]


def _build_guardrail_plan(
    *,
    query: str,
    planner_hint: str | None,
    fallback_reason: str,
    guardrail_source: str,
) -> PlannerPlan | None:
    normalized = query.replace(" ", "")
    hint = planner_hint or ""
    entities = _extract_plan_entities(query)

    if _looks_like_intersection_problem(normalized, hint) and len(entities) >= 2:
        subqueries = [f"{entity}中出现过哪些武功" for entity in entities[:3]]
        return PlannerPlan(
            mode="plan_rag",
            reason=f"{fallback_reason} 命中交集模式，按规则构造多子查询规划。",
            primary_query=query,
            subqueries=subqueries,
            merge_strategy="intersection",
            answer_style="list",
            planner_source=f"{guardrail_source}_intersection",
        )

    if _looks_like_compare_problem(normalized, hint) and len(entities) >= 2:
        target = _extract_compare_target(query)
        subqueries = [f"{entity}中出现过哪些{target}" for entity in entities[:3]]
        return PlannerPlan(
            mode="plan_rag",
            reason=f"{fallback_reason} 命中对比模式，按规则构造多子查询规划。",
            primary_query=query,
            subqueries=subqueries,
            merge_strategy="compare",
            answer_style="list",
            planner_source=f"{guardrail_source}_compare",
        )

    if _looks_like_rank_problem(normalized, hint) and len(entities) >= 2:
        target = _extract_compare_target(query)
        subqueries = [f"{entity}中出现过哪些{target}" for entity in entities[:3]]
        return PlannerPlan(
            mode="plan_rag",
            reason=f"{fallback_reason} 命中排序模式，按规则构造多子查询规划。",
            primary_query=query,
            subqueries=subqueries,
            merge_strategy="rank",
            answer_style="table",
            planner_source=f"{guardrail_source}_rank",
        )

    if _looks_like_group_problem(normalized, hint) and len(entities) >= 2:
        subqueries = [f"{entity}的核心信息、差异与限制" for entity in entities[:3]]
        return PlannerPlan(
            mode="plan_rag",
            reason=f"{fallback_reason} 命中分组模式，按规则构造多子查询规划。",
            primary_query=query,
            subqueries=subqueries,
            merge_strategy="group_by",
            answer_style="table",
            planner_source=f"{guardrail_source}_group_by",
        )

    if _looks_like_dedupe_union_problem(normalized, hint) and len(entities) >= 2:
        subqueries = [f"{entity}相关的候选项和要点" for entity in entities[:3]]
        return PlannerPlan(
            mode="plan_rag",
            reason=f"{fallback_reason} 命中去重汇总模式，按规则构造多子查询规划。",
            primary_query=query,
            subqueries=subqueries,
            merge_strategy="dedupe_union",
            answer_style="list",
            planner_source=f"{guardrail_source}_dedupe_union",
        )

    return None


def _should_upgrade_to_guardrail_plan(
    *,
    plan: PlannerPlan,
    query: str,
    planner_hint: str | None,
) -> bool:
    if plan.mode != "single_retrieval":
        return False
    normalized = query.replace(" ", "")
    hint = planner_hint or ""
    entities = _extract_plan_entities(query)
    if len(entities) < 2:
        return False
    return (
        _looks_like_intersection_problem(normalized, hint)
        or _looks_like_compare_problem(normalized, hint)
        or _looks_like_rank_problem(normalized, hint)
        or _looks_like_group_problem(normalized, hint)
        or _looks_like_dedupe_union_problem(normalized, hint)
    )


def _extract_plan_entities(query: str) -> list[str]:
    entities: list[str] = []

    titled_entities = re.findall(r"《([^》]+)》", query)
    for entity in titled_entities:
        compact = entity.strip()
        if compact and compact not in entities:
            entities.append(compact)

    patterns = (
        r"(?:对比下|对比一下|比较下|比较一下|对比|比较)?(?P<a>[\u4e00-\u9fffA-Za-z0-9_\-·]{2,30})(?:和|与|及|、)(?P<b>[\u4e00-\u9fffA-Za-z0-9_\-·]{2,30})哪",
        r"在(?P<a>[\u4e00-\u9fffA-Za-z0-9_\-·]{2,30})出现过.*?(?:并且|同时|并|且).{0,8}?在(?P<b>[\u4e00-\u9fffA-Za-z0-9_\-·]{2,30})(?:中)?(?:也)?出现",
        r"(?P<a>[\u4e00-\u9fffA-Za-z0-9_\-·]{2,30})(?:和|与|及|、)(?P<b>[\u4e00-\u9fffA-Za-z0-9_\-·]{2,30})",
    )
    for pattern in patterns:
        match = re.search(pattern, query)
        if not match:
            continue
        for key in ("a", "b"):
            compact = _normalize_entity_text((match.group(key) or ""))
            if compact and compact not in entities:
                entities.append(compact)

    return entities


def _normalize_entity_text(text: str) -> str:
    compact = text.strip(" ，,。；;：:")
    compact = re.sub(r"^(对比下|对比一下|比较下|比较一下|对比|比较)", "", compact)
    compact = re.sub(r"(中也|中)$", "", compact)
    compact = re.sub(r"(里也|里)$", "", compact)
    compact = re.sub(r"(哪部小说.*|哪一个.*|哪个.*)$", "", compact)
    return compact.strip(" ，,。；;：:")


def _looks_like_intersection_problem(normalized_query: str, planner_hint: str) -> bool:
    if "交集" in planner_hint:
        return True
    markers = ("共同", "都", "并且", "也在", "同时出现", "重复", "交集")
    return sum(1 for marker in markers if marker in normalized_query) >= 2


def _looks_like_compare_problem(normalized_query: str, planner_hint: str) -> bool:
    if "对比" in planner_hint:
        return True
    markers = ("对比", "比较", "差异", "分别", "优缺点", "更多", "更少", "哪个", "哪部")
    return sum(1 for marker in markers if marker in normalized_query) >= 2


def _looks_like_rank_problem(normalized_query: str, planner_hint: str) -> bool:
    if "排序" in planner_hint:
        return True
    markers = ("排序", "排名", "排行", "先后", "从多到少", "从少到多", "最多", "最少")
    return sum(1 for marker in markers if marker in normalized_query) >= 1


def _looks_like_group_problem(normalized_query: str, planner_hint: str) -> bool:
    if "分组" in planner_hint:
        return True
    markers = ("分别", "各自", "按", "分组", "归类")
    return sum(1 for marker in markers if marker in normalized_query) >= 2


def _looks_like_dedupe_union_problem(normalized_query: str, planner_hint: str) -> bool:
    if "去重" in planner_hint:
        return True
    markers = ("汇总", "合并", "去重", "整理", "全集", "全部")
    return sum(1 for marker in markers if marker in normalized_query) >= 2


def _extract_compare_target(query: str) -> str:
    target_patterns = (
        r"出现的(?P<target>[\u4e00-\u9fffA-Za-z0-9_\-·]{1,12})(?:更多|更少)",
        r"哪些(?P<target>[\u4e00-\u9fffA-Za-z0-9_\-·]{1,12})",
    )
    for pattern in target_patterns:
        match = re.search(pattern, query)
        if match:
            target = (match.group("target") or "").strip(" ，,。；;：:")
            if target:
                return target
    if "功夫" in query:
        return "功夫"
    if "武功" in query:
        return "武功"
    return "内容"
