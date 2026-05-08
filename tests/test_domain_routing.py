from __future__ import annotations

import unittest
from unittest.mock import Mock

from app.schemas.chat import IntentDecision
from app.schemas.knowledge import KnowledgeDocument
from app.services.chat_pipeline import ChatPipeline
from app.services.knowledge_base import InMemoryKnowledgeBase
from app.services.retriever_service import InMemoryRetrieverService
from app.services.tool_runtime import build_default_tool_runtime


class DomainRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        knowledge_base = InMemoryKnowledgeBase()
        knowledge_base.add_documents(
            [
                KnowledgeDocument(
                    title="部署说明",
                    content="这个项目支持 Docker Compose 部署，也支持 Windows API-only 部署。",
                ),
                KnowledgeDocument(
                    title="RAG 架构说明",
                    content="系统使用 Qdrant 负责向量检索，BM25 负责关键词召回，最后做 hybrid 融合。",
                ),
            ]
        )
        retriever_service = InMemoryRetrieverService(knowledge_base)
        trace_service = Mock()
        trace_service.get_trace_detail.return_value = None
        self.pipeline = ChatPipeline(
            intent_service=None,  # type: ignore[arg-type]
            knowledge_base=knowledge_base,
            retriever_service=retriever_service,
            generation_service=None,  # type: ignore[arg-type]
            planner_service=None,  # type: ignore[arg-type]
            trace_service=trace_service,  # type: ignore[arg-type]
            tool_runtime=build_default_tool_runtime(
                retriever_service=retriever_service,
                knowledge_base=knowledge_base,
                trace_service=trace_service,
            ),
            rag_snapshot_service=None,  # type: ignore[arg-type]
        )

    def test_route_knowledge_base_keeps_in_domain_question(self) -> None:
        decision = IntentDecision(
            intent="knowledge_qa",
            need_rag=True,
            rewrite_query="Qdrant 在这个系统里是做什么用的？",
            rationale="test",
            execution_mode="rag",
        )

        routed = self.pipeline._route_knowledge_base(decision)

        self.assertTrue(routed.need_rag)
        self.assertEqual(routed.execution_mode, "rag")
        self.assertEqual(routed.knowledge_base_id, "default")
        self.assertEqual(routed.knowledge_base_name, "默认知识库")

    def test_route_knowledge_base_downgrades_out_of_domain_question(self) -> None:
        decision = IntentDecision(
            intent="knowledge_qa",
            need_rag=True,
            rewrite_query="明天上海天气怎么样？",
            rationale="test",
            execution_mode="rag",
        )

        routed = self.pipeline._route_knowledge_base(decision)

        self.assertFalse(routed.need_rag)
        self.assertEqual(routed.execution_mode, "direct")
        self.assertIsNone(routed.knowledge_base_id)
        self.assertIn("降级为直接回答", routed.rationale)


if __name__ == "__main__":
    unittest.main()
