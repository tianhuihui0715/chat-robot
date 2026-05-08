from __future__ import annotations

from datetime import datetime
import unittest
from unittest.mock import Mock

from app.schemas.knowledge import KnowledgeDocument
from app.schemas.traces import TraceDetail
from app.services.knowledge_base import InMemoryKnowledgeBase
from app.services.retriever_service import InMemoryRetrieverService
from app.services.tool_runtime import ToolExecutionContext, build_default_tool_runtime


class ToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.knowledge_base = InMemoryKnowledgeBase()
        self.knowledge_base.add_documents(
            [
                KnowledgeDocument(
                    title="射雕英雄传",
                    content="降龙十八掌和九阴真经都在这本文档里出现。",
                ),
                KnowledgeDocument(
                    title="神雕侠侣",
                    content="黯然销魂掌与降龙十八掌都在这本文档里出现。",
                ),
            ]
        )
        self.retriever_service = InMemoryRetrieverService(self.knowledge_base, top_k=2)
        self.trace_service = Mock()
        self.runtime = build_default_tool_runtime(
            retriever_service=self.retriever_service,
            knowledge_base=self.knowledge_base,
            trace_service=self.trace_service,
        )
        self.context = ToolExecutionContext(
            request_id="req-1",
            session_id="sess-1",
            knowledge_base_id="default",
            use_reranker=False,
        )

    def test_registry_exposes_planner_tools(self) -> None:
        self.assertEqual(
            self.runtime.registry.planner_tool_names(),
            ["answer.direct", "retrieval.search"],
        )
        definitions = {definition.name: definition for definition in self.runtime.registry.list_tools()}
        self.assertTrue(definitions["retrieval.search"].requires_knowledge_base)
        self.assertFalse(definitions["answer.direct"].requires_knowledge_base)

    async def test_retrieval_search_returns_sources(self) -> None:
        result = await self.runtime.execute(
            "retrieval.search",
            {"query": "射雕英雄传"},
            self.context,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.tool_name, "retrieval.search")
        self.assertGreaterEqual(len(result.sources), 1)
        self.assertIn("retrieved_count", result.payload)

    async def test_document_lookup_matches_titles_and_content(self) -> None:
        result = await self.runtime.execute(
            "kb.document_lookup",
            {"keyword": "神雕", "limit": 2},
            self.context,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.payload["match_count"], 1)
        self.assertEqual(result.payload["matches"][0]["title"], "神雕侠侣")

    async def test_trace_lookup_returns_summary_payload(self) -> None:
        self.trace_service.get_trace_detail.return_value = TraceDetail(
            request_id="trace-1",
            session_id="sess-1",
            langsmith_trace_id=None,
            user_input="测试",
            intent="knowledge_qa",
            need_rag=True,
            status="completed",
            total_latency_ms=123,
            error_message=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
            step_count=4,
            final_output="ok",
            steps=[],
            intent_record=None,
            retrieval_record=None,
            generation_record=None,
        )

        result = await self.runtime.execute(
            "trace.lookup",
            {"request_id": "trace-1"},
            self.context,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.payload["request_id"], "trace-1")
        self.assertEqual(result.payload["step_count"], 4)

    async def test_unknown_tool_fails_gracefully(self) -> None:
        result = await self.runtime.execute("tool.missing", {}, self.context)

        self.assertFalse(result.ok)
        self.assertIn("Unknown tool", result.payload["error"])


if __name__ == "__main__":
    unittest.main()
