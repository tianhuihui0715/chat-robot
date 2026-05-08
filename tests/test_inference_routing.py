from __future__ import annotations

import json
import os
import unittest

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.inference.main import app as inference_app
from app.schemas.chat import PlanTask, SubtaskResult
from app.services.chat_pipeline import (
    _parse_action_selection,
    _parse_task_intent,
    _parse_knowledge_base_selection,
    _parse_tool_selection,
)
from app.services.chat_pipeline import TaskExecutionIntent
from app.services.tool_runtime import ToolDefinition, ToolParameter


class InferenceRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._previous_inference_runtime_mode = os.environ.get("INFERENCE_RUNTIME_MODE")
        os.environ["INFERENCE_RUNTIME_MODE"] = "mock"
        get_settings.cache_clear()
        cls.client = TestClient(inference_app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        if cls._previous_inference_runtime_mode is None:
            os.environ.pop("INFERENCE_RUNTIME_MODE", None)
        else:
            os.environ["INFERENCE_RUNTIME_MODE"] = cls._previous_inference_runtime_mode
        get_settings.cache_clear()

    def test_intent_returns_clarify_for_vague_deploy_issue(self) -> None:
        response = self.client.post(
            "/intent",
            json={"messages": [{"role": "user", "content": "帮我看看部署问题。"}]},
        )
        self.assertEqual(response.status_code, 200)
        decision = response.json()["decision"]

        self.assertEqual(decision["intent"], "knowledge_qa")
        self.assertTrue(decision["need_rag"])
        self.assertTrue(decision["should_clarify"])
        self.assertEqual(decision["execution_mode"], "direct")
        self.assertIn("部署场景", decision["clarify_question"])
        self.assertEqual(decision["candidate_tools"], [])

    def test_intent_returns_rag_for_direct_knowledge_question(self) -> None:
        response = self.client.post(
            "/intent",
            json={"messages": [{"role": "user", "content": "这个项目的部署方式是什么？"}]},
        )
        self.assertEqual(response.status_code, 200)
        decision = response.json()["decision"]

        self.assertEqual(decision["intent"], "knowledge_qa")
        self.assertTrue(decision["need_rag"])
        self.assertFalse(decision["should_clarify"])
        self.assertEqual(decision["execution_mode"], "rag")
        self.assertEqual(decision["candidate_tools"], [])

    def test_intent_returns_plan_execute_for_complex_comparison(self) -> None:
        response = self.client.post(
            "/intent",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "帮我对比一下这个项目 Docker Compose 部署和 Windows API-only 部署的差异，再说下各自容易踩的坑。",
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        decision = response.json()["decision"]

        self.assertEqual(decision["intent"], "knowledge_qa")
        self.assertTrue(decision["need_rag"])
        self.assertFalse(decision["should_clarify"])
        self.assertEqual(decision["execution_mode"], "plan_execute")
        self.assertEqual(decision["candidate_tools"], ["retrieval.search", "answer.direct"])

    def test_intent_returns_plan_execute_for_cross_source_intersection_question(self) -> None:
        response = self.client.post(
            "/intent",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "统计下有哪些武功是在射雕英雄传出现过并且也在神雕侠侣中也出现的",
                    }
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        decision = response.json()["decision"]

        self.assertEqual(decision["intent"], "knowledge_qa")
        self.assertTrue(decision["need_rag"])
        self.assertEqual(decision["execution_mode"], "plan_execute")
        self.assertIn("交集", decision["planner_hint"])

    def test_generate_json_mode_returns_action_payload(self) -> None:
        response = self.client.post(
            "/generate",
            json={
                "messages": [{"role": "user", "content": "这个项目的部署方式是什么？"}],
                "intent": {
                    "intent": "knowledge_qa",
                    "need_rag": True,
                    "rewrite_query": "这个项目的部署方式是什么？",
                    "rationale": "测试",
                    "execution_mode": "plan_execute",
                    "should_clarify": False,
                    "clarify_question": None,
                    "candidate_tools": ["retrieval.search", "answer.direct"],
                    "knowledge_base_id": None,
                    "knowledge_base_name": None,
                },
                "sources": [],
                "response_mode": "json",
                "system_prompt_override": "router",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.json()["answer"])

        self.assertIn(payload["action"], {"retrieval.search", "answer.direct"})
        self.assertTrue(payload["query"])
        self.assertTrue(payload["reason"])

    def test_parse_action_selection_falls_back_on_invalid_json(self) -> None:
        action = _parse_action_selection("not-json", "fallback query")
        self.assertEqual(
            action,
            {
                "action": "retrieval.search",
                "query": "fallback query",
                "reason": "模型未返回合法 action，回退到检索。",
            },
        )

    def test_parse_tool_selection_accepts_valid_retrieval_payload(self) -> None:
        selection = _parse_tool_selection(
            raw_output='{"tool":"retrieval.search","arguments":{"query":"射雕英雄传中出现过哪些武功","top_k":5},"reason":"需要检索"}',
            task=PlanTask(task_id="task_1", goal="列出射雕里的武功", query="射雕英雄传中出现过哪些武功"),
            task_intent=TaskExecutionIntent(
                intent="extraction",
                query="射雕英雄传中出现过哪些武功",
                source_hint="射雕英雄传",
                target="武功",
                knowledge_base_id="default",
                reason="抽取型任务",
            ),
            available_tools=[
                ToolDefinition(
                    name="retrieval.search",
                    description="检索",
                    parameters=(ToolParameter(name="query", type="string", description="q"),),
                ),
                ToolDefinition(
                    name="answer.direct",
                    description="直接回答",
                    parameters=(ToolParameter(name="query", type="string", description="q"),),
                ),
            ],
            dependency_results={},
        )

        self.assertEqual(selection["tool"], "retrieval.search")
        self.assertEqual(selection["arguments"]["query"], "射雕英雄传中出现过哪些武功")
        self.assertNotIn("top_k", selection["arguments"])
        self.assertEqual(selection["reason"], "需要检索")

    def test_parse_tool_selection_falls_back_to_answer_direct_for_dependency_task(self) -> None:
        selection = _parse_tool_selection(
            raw_output="not-json",
            task=PlanTask(
                task_id="task_3",
                goal="比较 task_1 和 task_2 的结果",
                query="比较 task_1 和 task_2 的结果",
                depends_on=["task_1", "task_2"],
            ),
            task_intent=TaskExecutionIntent(
                intent="aggregation",
                query="比较 task_1 和 task_2 的结果",
                reason="聚合任务",
            ),
            available_tools=[
                ToolDefinition(
                    name="retrieval.search",
                    description="检索",
                    parameters=(ToolParameter(name="query", type="string", description="q"),),
                ),
                ToolDefinition(
                    name="answer.direct",
                    description="直接回答",
                    parameters=(ToolParameter(name="query", type="string", description="q"),),
                ),
            ],
            dependency_results={
                "task_1": SubtaskResult(task_id="task_1", goal="A", query="A", items=["a"]),
                "task_2": SubtaskResult(task_id="task_2", goal="B", query="B", items=["b"]),
            },
        )

        self.assertEqual(selection["tool"], "answer.direct")
        self.assertEqual(selection["arguments"]["query"], "比较 task_1 和 task_2 的结果")
        self.assertIn("回退", selection["reason"])

    def test_parse_task_intent_accepts_extraction_payload(self) -> None:
        task_intent = _parse_task_intent(
            raw_output=(
                '{"intent":"extraction","query":"射雕英雄传中出现过哪些武功",'
                '"source_hint":"射雕英雄传","target":"武功","knowledge_base_id":"default","reason":"需要抽取武功列表"}'
            ),
            task=PlanTask(
                task_id="task_1",
                goal="列出射雕英雄传中出现过的武功",
                query="列出射雕英雄传中出现过的武功",
            ),
            dependency_results={},
            catalog={"default": {"name": "默认知识库", "titles": ["射雕英雄传"]}},
        )

        self.assertEqual(task_intent.intent, "extraction")
        self.assertEqual(task_intent.source_hint, "射雕英雄传")
        self.assertEqual(task_intent.target, "武功")
        self.assertEqual(task_intent.knowledge_base_id, "default")

    def test_parse_task_intent_falls_back_to_aggregation_for_dependency_task(self) -> None:
        task_intent = _parse_task_intent(
            raw_output="not-json",
            task=PlanTask(
                task_id="task_3",
                goal="比较 task_1 和 task_2 的结果",
                query="比较 task_1 和 task_2 的结果",
                depends_on=["task_1", "task_2"],
            ),
            dependency_results={
                "task_1": SubtaskResult(task_id="task_1", goal="A", query="A", items=["a"]),
                "task_2": SubtaskResult(task_id="task_2", goal="B", query="B", items=["b"]),
            },
            catalog={"default": {"name": "默认知识库", "titles": []}},
        )

        self.assertEqual(task_intent.intent, "aggregation")
        self.assertEqual(task_intent.query, "比较 task_1 和 task_2 的结果")

    def test_parse_knowledge_base_selection_accepts_known_catalog_entry(self) -> None:
        selected_id, selected_name, reason = _parse_knowledge_base_selection(
            raw_output='{"knowledge_base_id":"novels","reason":"任务涉及武侠小说内容"}',
            catalog={
                "default": {"name": "默认知识库", "titles": []},
                "novels": {"name": "金庸小说库", "titles": []},
            },
            fallback_id="default",
            fallback_name="默认知识库",
            fallback_reason="fallback",
        )

        self.assertEqual(selected_id, "novels")
        self.assertEqual(selected_name, "金庸小说库")
        self.assertEqual(reason, "任务涉及武侠小说内容")

    def test_parse_knowledge_base_selection_falls_back_on_invalid_json(self) -> None:
        selected_id, selected_name, reason = _parse_knowledge_base_selection(
            raw_output="not-json",
            catalog={
                "default": {"name": "默认知识库", "titles": []},
                "novels": {"name": "金庸小说库", "titles": []},
            },
            fallback_id="default",
            fallback_name="默认知识库",
            fallback_reason="模型未返回合法知识库选择，已按默认知识库回退。",
        )

        self.assertEqual(selected_id, "default")
        self.assertEqual(selected_name, "默认知识库")
        self.assertIn("回退", reason)


if __name__ == "__main__":
    unittest.main()
