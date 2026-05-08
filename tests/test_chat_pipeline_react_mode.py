from __future__ import annotations

import unittest
from unittest.mock import Mock

from app.schemas.chat import IntentDecision
from app.services.chat_pipeline import ChatPipeline
from app.services.knowledge_base import InMemoryKnowledgeBase
from app.services.retriever_service import InMemoryRetrieverService
from app.services.tool_runtime import build_default_tool_runtime


def _build_tool_runtime():
    knowledge_base = InMemoryKnowledgeBase()
    retriever_service = InMemoryRetrieverService(knowledge_base)
    trace_service = Mock()
    trace_service.get_trace_detail.return_value = None
    return build_default_tool_runtime(
        retriever_service=retriever_service,
        knowledge_base=knowledge_base,
        trace_service=trace_service,
    )


class ChatPipelineExecutionStrategyTests(unittest.TestCase):
    def test_auto_keeps_original_execution_mode(self) -> None:
        pipeline = ChatPipeline(
            intent_service=None,  # type: ignore[arg-type]
            knowledge_base=InMemoryKnowledgeBase(),
            retriever_service=None,  # type: ignore[arg-type]
            generation_service=None,  # type: ignore[arg-type]
            planner_service=None,  # type: ignore[arg-type]
            trace_service=None,  # type: ignore[arg-type]
            tool_runtime=_build_tool_runtime(),
            rag_snapshot_service=None,  # type: ignore[arg-type]
        )
        decision = IntentDecision(
            intent="knowledge_qa",
            need_rag=True,
            rewrite_query="这个项目的部署方式是什么？",
            rationale="test",
            execution_mode="rag",
            candidate_tools=[],
        )

        updated = pipeline._apply_execution_mode_override(decision, "auto")

        self.assertEqual(updated.execution_mode, "rag")
        self.assertEqual(updated.candidate_tools, [])

    def test_off_disables_plan_execute(self) -> None:
        pipeline = ChatPipeline(
            intent_service=None,  # type: ignore[arg-type]
            knowledge_base=InMemoryKnowledgeBase(),
            retriever_service=None,  # type: ignore[arg-type]
            generation_service=None,  # type: ignore[arg-type]
            planner_service=None,  # type: ignore[arg-type]
            trace_service=None,  # type: ignore[arg-type]
            tool_runtime=_build_tool_runtime(),
            rag_snapshot_service=None,  # type: ignore[arg-type]
        )
        decision = IntentDecision(
            intent="knowledge_qa",
            need_rag=True,
            rewrite_query="复杂知识问题",
            rationale="test",
            execution_mode="plan_execute",
            candidate_tools=["retrieval.search", "answer.direct"],
        )

        updated = pipeline._apply_execution_mode_override(decision, "off")

        self.assertEqual(updated.execution_mode, "rag")
        self.assertEqual(updated.candidate_tools, [])

    def test_force_enables_plan_execute_and_default_tools(self) -> None:
        pipeline = ChatPipeline(
            intent_service=None,  # type: ignore[arg-type]
            knowledge_base=InMemoryKnowledgeBase(),
            retriever_service=None,  # type: ignore[arg-type]
            generation_service=None,  # type: ignore[arg-type]
            planner_service=None,  # type: ignore[arg-type]
            trace_service=None,  # type: ignore[arg-type]
            tool_runtime=_build_tool_runtime(),
            rag_snapshot_service=None,  # type: ignore[arg-type]
        )
        decision = IntentDecision(
            intent="task",
            need_rag=False,
            rewrite_query="帮我整理一下",
            rationale="test",
            execution_mode="direct",
            candidate_tools=[],
        )

        updated = pipeline._apply_execution_mode_override(decision, "force")

        self.assertEqual(updated.execution_mode, "plan_execute")
        self.assertEqual(updated.candidate_tools, ["answer.direct", "retrieval.search"])

    def test_force_does_not_override_clarify(self) -> None:
        pipeline = ChatPipeline(
            intent_service=None,  # type: ignore[arg-type]
            knowledge_base=InMemoryKnowledgeBase(),
            retriever_service=None,  # type: ignore[arg-type]
            generation_service=None,  # type: ignore[arg-type]
            planner_service=None,  # type: ignore[arg-type]
            trace_service=None,  # type: ignore[arg-type]
            tool_runtime=_build_tool_runtime(),
            rag_snapshot_service=None,  # type: ignore[arg-type]
        )
        decision = IntentDecision(
            intent="knowledge_qa",
            need_rag=True,
            rewrite_query="帮我看看部署问题",
            rationale="test",
            execution_mode="direct",
            should_clarify=True,
            clarify_question="你想看哪种部署场景？",
            candidate_tools=[],
        )

        updated = pipeline._apply_execution_mode_override(decision, "force")

        self.assertTrue(updated.should_clarify)
        self.assertEqual(updated.execution_mode, "direct")
        self.assertEqual(updated.clarify_question, "你想看哪种部署场景？")

    def test_auto_fills_runtime_tools_for_plan_execute(self) -> None:
        pipeline = ChatPipeline(
            intent_service=None,  # type: ignore[arg-type]
            knowledge_base=InMemoryKnowledgeBase(),
            retriever_service=None,  # type: ignore[arg-type]
            generation_service=None,  # type: ignore[arg-type]
            planner_service=None,  # type: ignore[arg-type]
            trace_service=None,  # type: ignore[arg-type]
            tool_runtime=_build_tool_runtime(),
            rag_snapshot_service=None,  # type: ignore[arg-type]
        )
        decision = IntentDecision(
            intent="knowledge_qa",
            need_rag=True,
            rewrite_query="复杂知识问题",
            rationale="test",
            execution_mode="plan_execute",
            candidate_tools=[],
        )

        updated = pipeline._apply_execution_mode_override(decision, "auto")

        self.assertEqual(updated.execution_mode, "plan_execute")
        self.assertEqual(updated.candidate_tools, ["answer.direct", "retrieval.search"])

    def test_retry_profile_scales_with_retry_budget(self) -> None:
        pipeline = ChatPipeline(
            intent_service=None,  # type: ignore[arg-type]
            knowledge_base=None,  # type: ignore[arg-type]
            retriever_service=None,  # type: ignore[arg-type]
            generation_service=None,  # type: ignore[arg-type]
            planner_service=None,  # type: ignore[arg-type]
            trace_service=None,  # type: ignore[arg-type]
            tool_runtime=_build_tool_runtime(),
            rag_snapshot_service=None,  # type: ignore[arg-type]
            plan_execute_top_k=6,
            plan_execute_candidate_multiplier=3,
            plan_execute_rerank_candidate_limit=18,
            plan_execute_bm25_top_k=10,
            plan_execute_max_retries=2,
            plan_execute_retry_multiplier=2,
        )

        retry_profile = pipeline._plan_execute_retry_profile(1)

        self.assertEqual(retry_profile["top_k"], 12)
        self.assertEqual(retry_profile["candidate_multiplier"], 6)
        self.assertEqual(retry_profile["rerank_candidate_limit"], 36)
        self.assertEqual(retry_profile["bm25_top_k"], 20)
        self.assertEqual(retry_profile["retry_count"], 1)
        self.assertEqual(retry_profile["scale"], 2)


if __name__ == "__main__":
    unittest.main()
