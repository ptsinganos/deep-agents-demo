import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.orchestrator import (
    ORCHESTRATOR_PROMPT,
    _orchestrator_permissions,
    build_orchestrator,
)
from src.utils import ResearchConfig


class OrchestratorTests(unittest.TestCase):
    def test_orchestrator_defaults_to_strong_model(self):
        fake_model = FakeListChatModel(responses=["done"])
        with patch(
            "src.agents.orchestrator.resolve_model", return_value=fake_model
        ) as resolve_model:
            build_orchestrator()

        resolve_model.assert_called_once_with(None, default_name="strong_model")

    def test_orchestrator_builds_complete_hierarchy(self):
        agent = build_orchestrator(
            FakeListChatModel(responses=["done"]),
            config=ResearchConfig(max_papers_per_topic=2),
        )
        self.assertEqual(agent.name, "LangGraph")

    def test_orchestrator_is_read_only_under_research(self):
        permissions = _orchestrator_permissions()
        self.assertEqual(permissions[0].operations, ["read"])
        self.assertEqual(permissions[1].operations, ["write"])
        self.assertIn("Dispatch all tasks before waiting", ORCHESTRATOR_PROMPT)
