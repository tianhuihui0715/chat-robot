from __future__ import annotations

import unittest

from app.schemas.chat import ChatRequest


class ChatRequestExecutionStrategyTests(unittest.TestCase):
    def test_defaults_to_auto(self) -> None:
        request = ChatRequest(
            messages=[{"role": "user", "content": "你好"}],
        )

        self.assertEqual(request.execution_strategy, "auto")
        self.assertIsNone(request.react_mode)

    def test_accepts_execution_strategy(self) -> None:
        request = ChatRequest(
            messages=[{"role": "user", "content": "你好"}],
            execution_strategy="force",
        )

        self.assertEqual(request.execution_strategy, "force")

    def test_maps_legacy_react_mode(self) -> None:
        request = ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "你好"}],
                "react_mode": "off",
            }
        )

        self.assertEqual(request.execution_strategy, "off")
        self.assertEqual(request.react_mode, "off")

    def test_execution_strategy_wins_when_both_present(self) -> None:
        request = ChatRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "你好"}],
                "execution_strategy": "force",
                "react_mode": "off",
            }
        )

        self.assertEqual(request.execution_strategy, "force")
        self.assertEqual(request.react_mode, "off")


if __name__ == "__main__":
    unittest.main()
