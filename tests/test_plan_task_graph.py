from __future__ import annotations

import unittest

from app.schemas.chat import PlanTask, SubtaskResult
from app.services.chat_pipeline import (
    _build_plan_execution_levels,
    _select_aggregate_seed_results,
)


class PlanTaskGraphTests(unittest.TestCase):
    def test_build_plan_execution_levels_groups_dependencies(self) -> None:
        tasks = [
            PlanTask(task_id="task_1", goal="A", query="A", depends_on=[]),
            PlanTask(task_id="task_2", goal="B", query="B", depends_on=[]),
            PlanTask(task_id="task_3", goal="C", query="C", depends_on=["task_1", "task_2"]),
            PlanTask(task_id="task_4", goal="D", query="D", depends_on=["task_3"]),
        ]

        levels = _build_plan_execution_levels(tasks)

        self.assertEqual([[task.task_id for task in level] for level in levels], [
            ["task_1", "task_2"],
            ["task_3"],
            ["task_4"],
        ])

    def test_build_plan_execution_levels_rejects_cycle(self) -> None:
        tasks = [
            PlanTask(task_id="task_1", goal="A", query="A", depends_on=["task_2"]),
            PlanTask(task_id="task_2", goal="B", query="B", depends_on=["task_1"]),
        ]

        with self.assertRaises(ValueError):
            _build_plan_execution_levels(tasks)

    def test_select_aggregate_seed_results_prefers_independent_tasks(self) -> None:
        tasks = [
            PlanTask(task_id="task_1", goal="A", query="A", depends_on=[]),
            PlanTask(task_id="task_2", goal="B", query="B", depends_on=[]),
            PlanTask(task_id="task_3", goal="Compare", query="Compare", depends_on=["task_1", "task_2"]),
        ]
        results = [
            SubtaskResult(task_id="task_1", goal="A", query="A", items=["a1"]),
            SubtaskResult(task_id="task_2", goal="B", query="B", items=["b1"]),
            SubtaskResult(task_id="task_3", goal="Compare", query="Compare", items=["cmp"]),
        ]

        selected = _select_aggregate_seed_results(tasks, results)

        self.assertEqual([result.task_id for result in selected], ["task_1", "task_2"])


if __name__ == "__main__":
    unittest.main()
