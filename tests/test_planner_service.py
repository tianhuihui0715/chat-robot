from __future__ import annotations

import asyncio
import unittest

from app.schemas.chat import ChatMessage, IntentDecision
from app.services.generator_service import MockGenerationBackend, QueuedGenerationService
from app.services.planner_service import PlannerService, _parse_planner_plan


class InvalidJsonBackend:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def generate(self, request) -> str:
        return "not-json"

    async def generate_stream(self, request):
        if False:
            yield ""
        await asyncio.sleep(0)


class ConservativeSingleRetrievalBackend:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def generate(self, request) -> str:
        return '{"mode":"single_retrieval","reason":"问题可以直接检索","primary_query":"原问题","subqueries":[],"merge_strategy":"union","answer_style":"summary"}'

    async def generate_stream(self, request):
        if False:
            yield ""
        await asyncio.sleep(0)


class PlannerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_planner_returns_plan_rag_for_intersection_question(self) -> None:
        generation_service = QueuedGenerationService(backend=MockGenerationBackend())
        await generation_service.start()
        try:
            planner_service = PlannerService(
                generation_service=generation_service,
                enabled=True,
                max_new_tokens=128,
                model_label="mock-planner",
            )
            decision = IntentDecision(
                intent="knowledge_qa",
                need_rag=True,
                rewrite_query="统计下有哪些武功是在射雕英雄传出现过并且也在神雕侠侣中也出现的",
                rationale="test",
                execution_mode="plan_execute",
                candidate_tools=["retrieval.search", "answer.direct"],
                planner_hint="问题更像集合交集，请优先拆出多个检索对象并保留共同信息。",
            )

            plan = await planner_service.plan(
                messages=[ChatMessage(role="user", content=decision.rewrite_query)],
                decision=decision,
            )
        finally:
            await generation_service.stop()

        self.assertEqual(plan.mode, "plan_rag")
        self.assertEqual(plan.merge_strategy, "intersection")
        self.assertGreaterEqual(len(plan.subqueries), 1)
        self.assertIn(
            plan.planner_source,
            {"model", "guardrail_override_conservative_intersection"},
        )

    def test_parse_planner_plan_falls_back_on_invalid_json(self) -> None:
        plan = _parse_planner_plan("not-json", "fallback query")

        self.assertEqual(plan.mode, "single_retrieval")
        self.assertEqual(plan.primary_query, "fallback query")
        self.assertEqual(plan.reason, "planner 未返回合法 JSON，回退到单次检索。")
        self.assertEqual(plan.planner_source, "invalid_json_fallback")

    def test_parse_planner_plan_accepts_group_by_strategy(self) -> None:
        plan = _parse_planner_plan(
            '{"mode":"plan_rag","reason":"需要分组展示","primary_query":"分别说明两种部署方式","subqueries":["Docker Compose 部署方式","Windows API-only 部署方式"],"merge_strategy":"group_by","answer_style":"table"}',
            "fallback query",
        )

        self.assertEqual(plan.mode, "plan_rag")
        self.assertEqual(plan.merge_strategy, "group_by")
        self.assertEqual(plan.answer_style, "table")
        self.assertEqual(
            plan.subqueries,
            ["Docker Compose 部署方式", "Windows API-only 部署方式"],
        )

    def test_parse_planner_plan_accepts_task_style_payload(self) -> None:
        plan = _parse_planner_plan(
            '{"task_type":"intersection","tasks":["列出射雕英雄传中出现过的武功","列出神雕侠侣中出现过的武功"]}',
            "fallback query",
        )

        self.assertEqual(plan.mode, "plan_rag")
        self.assertEqual(plan.merge_strategy, "intersection")
        self.assertEqual(
            plan.subqueries,
            ["列出射雕英雄传中出现过的武功", "列出神雕侠侣中出现过的武功"],
        )
        self.assertEqual(plan.primary_query, "fallback query")
        self.assertIsNotNone(plan.tasks)
        self.assertEqual(plan.tasks[0].depends_on, [])

    def test_parse_planner_plan_accepts_task_objects_with_dependencies(self) -> None:
        plan = _parse_planner_plan(
            '{"task_type":"compare","tasks":[{"task_id":"task_1","goal":"列出射雕英雄传中出现过的武功","depends_on":[]},{"task_id":"task_2","goal":"列出神雕侠侣中出现过的武功","depends_on":[]},{"task_id":"task_3","goal":"比较 task_1 和 task_2 的结果","depends_on":["task_1","task_2"]}]}',
            "fallback query",
        )

        self.assertEqual(plan.mode, "plan_rag")
        self.assertEqual(plan.merge_strategy, "compare")
        self.assertIsNotNone(plan.tasks)
        self.assertEqual(len(plan.tasks), 3)
        self.assertEqual(plan.tasks[2].task_id, "task_3")
        self.assertEqual(plan.tasks[2].depends_on, ["task_1", "task_2"])

    async def test_invalid_json_uses_guardrail_plan_for_intersection_question(self) -> None:
        generation_service = QueuedGenerationService(backend=InvalidJsonBackend())
        await generation_service.start()
        try:
            planner_service = PlannerService(
                generation_service=generation_service,
                enabled=True,
                max_new_tokens=128,
                model_label="mock-planner",
            )
            decision = IntentDecision(
                intent="knowledge_qa",
                need_rag=True,
                rewrite_query="统计下有哪些武功是在射雕英雄传出现过并且也在神雕侠侣中也出现的",
                rationale="test",
                execution_mode="plan_execute",
                candidate_tools=["retrieval.search", "answer.direct"],
                planner_hint="问题更像集合交集，请优先拆出多个检索对象并保留共同信息。",
            )

            plan = await planner_service.plan(
                messages=[ChatMessage(role="user", content=decision.rewrite_query)],
                decision=decision,
            )
        finally:
            await generation_service.stop()

        self.assertEqual(plan.mode, "plan_rag")
        self.assertEqual(plan.merge_strategy, "intersection")
        self.assertEqual(
            plan.subqueries[:2],
            ["射雕英雄传中出现过哪些武功", "神雕侠侣中出现过哪些武功"],
        )
        self.assertIn("按规则构造多子查询规划", plan.reason)
        self.assertEqual(plan.planner_source, "guardrail_invalid_json_intersection")

    async def test_invalid_json_uses_compare_guardrail_for_count_question(self) -> None:
        generation_service = QueuedGenerationService(backend=InvalidJsonBackend())
        await generation_service.start()
        try:
            planner_service = PlannerService(
                generation_service=generation_service,
                enabled=True,
                max_new_tokens=128,
                model_label="mock-planner",
            )
            decision = IntentDecision(
                intent="knowledge_qa",
                need_rag=True,
                rewrite_query="对比下神雕侠侣和射雕英雄传哪部小说里出现的功夫更多",
                rationale="test",
                execution_mode="plan_execute",
                candidate_tools=["retrieval.search", "answer.direct"],
                planner_hint="问题更像对比题，请拆成多个对象分别检索，再按差异或共同点组织答案。",
            )

            plan = await planner_service.plan(
                messages=[ChatMessage(role="user", content=decision.rewrite_query)],
                decision=decision,
            )
        finally:
            await generation_service.stop()

        self.assertEqual(plan.mode, "plan_rag")
        self.assertEqual(plan.merge_strategy, "compare")
        self.assertEqual(
            plan.subqueries[:2],
            ["神雕侠侣中出现过哪些功夫", "射雕英雄传中出现过哪些功夫"],
        )
        self.assertEqual(plan.planner_source, "guardrail_invalid_json_compare")

    async def test_conservative_single_retrieval_is_upgraded_for_complex_question(self) -> None:
        generation_service = QueuedGenerationService(backend=ConservativeSingleRetrievalBackend())
        await generation_service.start()
        try:
            planner_service = PlannerService(
                generation_service=generation_service,
                enabled=True,
                max_new_tokens=128,
                model_label="mock-planner",
            )
            decision = IntentDecision(
                intent="knowledge_qa",
                need_rag=True,
                rewrite_query="统计下有哪些武功是在射雕英雄传出现过并且也在神雕侠侣中也出现的",
                rationale="test",
                execution_mode="plan_execute",
                candidate_tools=["retrieval.search", "answer.direct"],
                planner_hint="问题更像集合交集，请优先拆出多个检索对象并保留共同信息。",
            )
            plan = await planner_service.plan(
                messages=[ChatMessage(role="user", content=decision.rewrite_query)],
                decision=decision,
            )
        finally:
            await generation_service.stop()

        self.assertEqual(plan.mode, "plan_rag")
        self.assertEqual(plan.merge_strategy, "intersection")
        self.assertIn("过于保守", plan.reason)
        self.assertEqual(plan.planner_source, "guardrail_override_conservative_intersection")


if __name__ == "__main__":
    unittest.main()
