import unittest

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.summarizer import (
    SUMMARIZER_PROMPT,
    _summarizer_permissions,
    build_summarizer,
)


class SummarizerTests(unittest.TestCase):
    def test_summarizer_builds_independently(self):
        agent = build_summarizer(FakeListChatModel(responses=["done"]))
        self.assertEqual(agent.name, "LangGraph")

    def test_pdf_access_contract(self):
        permissions = _summarizer_permissions()
        self.assertEqual(permissions[0].operations, ["write"])
        self.assertEqual(permissions[1].mode, "deny")
        self.assertIn("extract_pdf_text", SUMMARIZER_PROMPT)
