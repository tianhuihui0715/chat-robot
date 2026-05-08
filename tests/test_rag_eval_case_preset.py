from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.schemas.admin import RAGEvaluationCase


class RAGEvalCasePresetTests(unittest.TestCase):
    def test_plan_execute_eval_case_preset_is_valid(self) -> None:
        preset_path = Path("config/rag_eval_cases.plan_execute.json")
        payload = json.loads(preset_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["name"], "plan_execute_complex")
        self.assertTrue(payload["description"])
        self.assertGreaterEqual(len(payload["cases"]), 4)

        for raw_case in payload["cases"]:
            case = RAGEvaluationCase.model_validate(raw_case)
            self.assertTrue(case.query.strip())
            self.assertLessEqual(len(case.expected_sources), 4)
            self.assertLessEqual(len(case.expected_answer_keywords), 4)


if __name__ == "__main__":
    unittest.main()
