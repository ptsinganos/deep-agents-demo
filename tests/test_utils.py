import unittest

from src.utils import (
    Paper,
    ResearchConfig,
    summary_path,
    topic_directory,
    topic_slug,
    validate_config,
)


class UtilsTests(unittest.TestCase):
    def test_deterministic_artifact_paths(self):
        paper = Paper(
            "Paper",
            ("Author",),
            "2026-01-01",
            "arXiv:2601.12345v2",
            "https://arxiv.org/abs/2601.12345",
        )
        self.assertEqual(
            topic_slug("Retrieval-Augmented Generation!"),
            "retrieval_augmented_generation",
        )
        self.assertEqual(
            topic_directory("Retrieval-Augmented Generation!"),
            "/research/retrieval_augmented_generation",
        )
        self.assertEqual(
            summary_path("Retrieval-Augmented Generation!", paper),
            "/research/retrieval_augmented_generation/arXiv_2601.12345v2_summary.md",
        )

    def test_paper_tool_payload_is_explicit_and_json_compatible(self):
        paper = Paper(
            "Paper",
            ("Author One", "Author Two"),
            "2026-01-01",
            "http://arxiv.org/abs/2609.23986v1",
            "https://arxiv.org/pdf/2609.23986v1",
            "Abstract",
        )

        self.assertEqual(
            paper.to_tool_payload(),
            {
                "paper_id": "2609.23986v1",
                "title": "Paper",
                "authors": ["Author One", "Author Two"],
                "published": "2026-01-01",
                "identifier": "http://arxiv.org/abs/2609.23986v1",
                "url": "https://arxiv.org/pdf/2609.23986v1",
                "abstract": "Abstract",
            },
        )

    def test_invalid_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_search_attempts"):
            validate_config(ResearchConfig(max_search_attempts_per_topic=0))
