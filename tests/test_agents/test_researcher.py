import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.researcher import (
    RESEARCHER_PROMPT,
    _researcher_permissions,
    build_researcher,
)
from src.utils import ResearchConfig


class ResearcherTests(unittest.TestCase):
    def test_researcher_defaults_to_strong_model(self):
        fake_model = FakeListChatModel(responses=["done"])
        with patch(
            "src.agents.researcher.resolve_model", return_value=fake_model
        ) as resolve_model:
            build_researcher()

        resolve_model.assert_called_once_with(None, default_name="strong_model")

    def test_researcher_builds_independently(self):
        agent = build_researcher(
            FakeListChatModel(responses=["done"]),
            config=ResearchConfig(max_papers_per_topic=2),
        )
        self.assertEqual(agent.name, "LangGraph")

    def test_researcher_contract(self):
        self.assertIn("one assistant run", RESEARCHER_PROMPT)
        self.assertIn("errors.md", RESEARCHER_PROMPT)
        self.assertEqual(_researcher_permissions()[0].operations, ["read", "write"])
