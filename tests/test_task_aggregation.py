from __future__ import annotations

import unittest

from app.schemas.chat import PlanTask, SourceChunk, SubtaskResult
from app.services.chat_pipeline import (
    _aggregate_subtask_results,
    _clean_candidate_items,
    _parse_subtask_result,
)
from app.services.planner_service import PlannerPlan


class TaskAggregationTests(unittest.TestCase):
    def test_intersection_aggregation_keeps_common_items(self) -> None:
        plan = PlannerPlan(
            mode="plan_rag",
            reason="test",
            primary_query="哪些武功同时出现",
            subqueries=["射雕英雄传中出现过哪些武功", "神雕侠侣中出现过哪些武功"],
            merge_strategy="intersection",
            answer_style="list",
        )
        task_results = [
            SubtaskResult(
                task_id="task_1",
                goal="射雕",
                query="射雕英雄传中出现过哪些武功",
                items=["降龙十八掌", "九阴真经", "空明拳"],
                count=3,
            ),
            SubtaskResult(
                task_id="task_2",
                goal="神雕",
                query="神雕侠侣中出现过哪些武功",
                items=["九阴真经", "降龙十八掌", "玉女剑法"],
                count=3,
            ),
        ]

        aggregate = _aggregate_subtask_results(plan, task_results)

        self.assertEqual(aggregate.mode, "intersection")
        self.assertEqual(aggregate.items, ["降龙十八掌", "九阴真经"])

    def test_compare_aggregation_marks_winner(self) -> None:
        plan = PlannerPlan(
            mode="plan_rag",
            reason="test",
            primary_query="哪部小说里出现的功夫更多",
            subqueries=["神雕侠侣中出现过哪些功夫", "射雕英雄传中出现过哪些功夫"],
            merge_strategy="compare",
            answer_style="list",
        )
        task_results = [
            SubtaskResult(
                task_id="task_1",
                goal="神雕",
                query="神雕侠侣中出现过哪些功夫",
                items=["黯然销魂掌", "玉女剑法", "玄铁剑法", "九阴真经"],
                count=4,
            ),
            SubtaskResult(
                task_id="task_2",
                goal="射雕",
                query="射雕英雄传中出现过哪些功夫",
                items=["降龙十八掌", "九阴真经"],
                count=2,
            ),
        ]

        aggregate = _aggregate_subtask_results(plan, task_results)

        self.assertEqual(aggregate.mode, "compare")
        self.assertEqual(aggregate.left_count, 4)
        self.assertEqual(aggregate.right_count, 2)
        self.assertEqual(aggregate.winner_task_id, "task_1")

    def test_dedupe_union_aggregation_removes_duplicates(self) -> None:
        plan = PlannerPlan(
            mode="plan_rag",
            reason="test",
            primary_query="汇总两部作品里的武功并去重",
            subqueries=["射雕英雄传中出现过哪些武功", "神雕侠侣中出现过哪些武功"],
            merge_strategy="dedupe_union",
            answer_style="list",
        )
        task_results = [
            SubtaskResult(
                task_id="task_1",
                goal="射雕",
                query="射雕英雄传中出现过哪些武功",
                items=["降龙十八掌", "九阴真经"],
                count=2,
            ),
            SubtaskResult(
                task_id="task_2",
                goal="神雕",
                query="神雕侠侣中出现过哪些武功",
                items=["九阴真经", "玉女剑法"],
                count=2,
            ),
        ]

        aggregate = _aggregate_subtask_results(plan, task_results)

        self.assertEqual(aggregate.mode, "dedupe_union")
        self.assertEqual(aggregate.items, ["降龙十八掌", "九阴真经", "玉女剑法"])

    def test_rank_aggregation_orders_tasks_by_count(self) -> None:
        plan = PlannerPlan(
            mode="plan_rag",
            reason="test",
            primary_query="按数量给三种方案排序",
            subqueries=["方案A", "方案B", "方案C"],
            merge_strategy="rank",
            answer_style="table",
        )
        task_results = [
            SubtaskResult(task_id="task_1", goal="方案A", query="方案A", items=["a1"], count=1, confidence=0.7),
            SubtaskResult(task_id="task_2", goal="方案B", query="方案B", items=["b1", "b2", "b3"], count=3, confidence=0.8),
            SubtaskResult(task_id="task_3", goal="方案C", query="方案C", items=["c1", "c2"], count=2, confidence=0.9),
        ]

        aggregate = _aggregate_subtask_results(plan, task_results)

        self.assertEqual(aggregate.mode, "rank")
        self.assertEqual(aggregate.ranked_task_ids, ["task_2", "task_3", "task_1"])
        self.assertEqual(aggregate.items, ["方案B（3）", "方案C（2）", "方案A（1）"])

    def test_group_by_aggregation_preserves_groups(self) -> None:
        plan = PlannerPlan(
            mode="plan_rag",
            reason="test",
            primary_query="分别总结两种部署方式",
            subqueries=["Docker Compose 部署方式", "Windows API-only 部署方式"],
            merge_strategy="group_by",
            answer_style="table",
        )
        task_results = [
            SubtaskResult(
                task_id="task_1",
                goal="Docker Compose",
                query="Docker Compose 部署方式",
                items=["容器编排", "统一启动"],
                count=2,
            ),
            SubtaskResult(
                task_id="task_2",
                goal="Windows API-only",
                query="Windows API-only 部署方式",
                items=["只启动 API", "适合本机调试"],
                count=2,
            ),
        ]

        aggregate = _aggregate_subtask_results(plan, task_results)

        self.assertEqual(aggregate.mode, "group_by")
        self.assertEqual(
            aggregate.grouped_items,
            {
                "Docker Compose": ["容器编排", "统一启动"],
                "Windows API-only": ["只启动 API", "适合本机调试"],
            },
        )
        self.assertEqual(
            aggregate.items,
            ["容器编排", "统一启动", "只启动 API", "适合本机调试"],
        )

    def test_subtask_fallback_marks_low_coverage_for_retry(self) -> None:
        task = PlanTask(
            task_id="task_1",
            goal="列出射雕中的武功",
            query="射雕英雄传中出现过哪些武功",
        )
        sources = [
            SourceChunk(
                document_id="doc-1",
                title="射雕",
                content="郭靖学会了降龙十八掌。",
                score=0.9,
                metadata={},
            )
        ]

        result = _parse_subtask_result(
            raw_output="not-json",
            task=task,
            sources=sources,
        )

        self.assertTrue(result.needs_retry)
        self.assertLess(result.confidence or 1.0, 0.9)
        self.assertTrue(bool(result.coverage_hint))

    def test_clean_candidate_items_filters_obvious_dirty_phrases(self) -> None:
        cleaned = _clean_candidate_items(
            [
                "绝世武功",
                "便似数位高手的掌",
                "降龙十八掌",
                "九阴真经",
                "本来静坐修习内功",
            ],
            query="射雕英雄传中出现过哪些武功",
        )

        self.assertEqual(cleaned, ["降龙十八掌", "九阴真经"])

    def test_aggregate_confidence_drops_when_subtasks_are_low_quality(self) -> None:
        plan = PlannerPlan(
            mode="plan_rag",
            reason="test",
            primary_query="哪些武功同时出现",
            subqueries=["射雕英雄传中出现过哪些武功", "神雕侠侣中出现过哪些武功"],
            merge_strategy="intersection",
            answer_style="list",
        )
        task_results = [
            SubtaskResult(
                task_id="task_1",
                goal="射雕",
                query="射雕英雄传中出现过哪些武功",
                items=["降龙十八掌"],
                count=1,
                confidence=0.45,
                needs_retry=True,
            ),
            SubtaskResult(
                task_id="task_2",
                goal="神雕",
                query="神雕侠侣中出现过哪些武功",
                items=[],
                count=0,
                confidence=0.2,
                needs_retry=True,
            ),
        ]

        aggregate = _aggregate_subtask_results(plan, task_results)

        self.assertTrue(aggregate.needs_retry)
        self.assertLess(aggregate.confidence or 1.0, 0.5)


if __name__ == "__main__":
    unittest.main()
