import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import FilesystemBackend

from src.agents.tools import (
    create_download_paper_tool,
    create_search_recent_papers_tool,
)
from src.utils import Paper, ResearchConfig


class ToolFactoryTests(unittest.TestCase):
    def test_download_tool_writes_to_supplied_path(self):
        class PdfResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                return b"%PDF-1.7 test"

        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            tool = create_download_paper_tool(backend)
            paper_path = "/research/topic-id/2610.11920v1.pdf"
            with patch(
                "src.agents.tools.urllib.request.urlopen", return_value=PdfResponse()
            ):
                result = tool.invoke(
                    {
                        "url": "https://arxiv.org/pdf/2610.11920v1",
                        "paper_path": paper_path,
                    }
                )

            downloaded = backend.read(paper_path)

        self.assertEqual(result, paper_path)
        self.assertEqual(downloaded.file_data["encoding"], "base64")

    def test_search_tool_passes_its_date_inputs_to_the_search(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            expected = Paper(
                "Paper",
                ("Author",),
                "2026-01-01",
                "id",
                "https://example.test/paper.pdf",
            )
            received: dict[str, object] = {}

            def search(topic, config, published_after, published_before):
                received.update(
                    published_after=published_after, published_before=published_before
                )
                return [expected]

            tool = create_search_recent_papers_tool(
                ResearchConfig(), backend, search=search
            )

            result = json.loads(
                tool.invoke(
                    {
                        "topic": "agents",
                        "published_after": "2026-01-01",
                        "published_before": "2026-01-31",
                    }
                )
            )

        self.assertEqual(result[0]["title"], "Paper")
        self.assertEqual(received["published_after"], date(2026, 1, 1))
        self.assertEqual(received["published_before"], date(2026, 1, 31))

    def test_search_failure_records_error_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FilesystemBackend(root_dir=Path(directory), virtual_mode=True)
            tool = create_search_recent_papers_tool(
                ResearchConfig(),
                backend,
                search=lambda topic, config, published_after, published_before: (
                    _ for _ in ()
                ).throw(RuntimeError("offline")),
            )
            result = json.loads(tool.invoke({"topic": "agents"}))
            error = backend.read(result["errors_path"])

        self.assertEqual(result["status"], "terminal_error")
        self.assertIn("offline", error.file_data["content"])
