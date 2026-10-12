import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, TypedDict

from deepagents.backends import BackendProtocol, FilesystemBackend
from pydantic import BaseModel


@dataclass(frozen=True)
class ResearchConfig:
    """Controls search freshness and bounded parallelism."""

    max_papers_per_topic: int = 3
    max_search_attempts_per_topic: int = 1
    arxiv_num_retries: int = 0
    arxiv_delay_seconds: float = 3.0
    published_after: date | None = None
    published_before: date | None = None


@dataclass(frozen=True)
class PaperToolPayload(TypedDict):
    """Paper metadata exposed to the researcher tool."""

    paper_id: str
    title: str
    authors: list[str]
    published: str
    identifier: str
    url: str
    abstract: str


@dataclass(frozen=True)
class Paper:
    title: str
    authors: tuple[str, ...]
    published: str
    identifier: str
    url: str
    abstract: str = ""

    @property
    def paper_id(self) -> str:
        """A stable filename-safe identifier, preferring the arXiv ID."""
        value = self.identifier.rsplit("/", 1)[-1]
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.")
        return value or hashlib.sha256(self.url.encode()).hexdigest()[:12]

    def to_tool_payload(self) -> PaperToolPayload:
        """Return the explicit, JSON-compatible contract for search results."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "published": self.published,
            "identifier": self.identifier,
            "url": self.url,
            "abstract": self.abstract,
        }


class ResearcherOutput(BaseModel):
    status: Literal["success", "fail"]
    count: int
    error_path: str | None = None
    report_path: str | None = None


class SummarizerOutput(BaseModel):
    status: Literal["success", "fail"]
    summary_path: str | None = None


def topic_slug(topic: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return value or "untitled_topic"


def topic_directory(topic: str) -> str:
    return f"/research/{topic_slug(topic)}"


def summary_path(topic: str, paper: Paper) -> str:
    return f"{topic_directory(topic)}/{paper.paper_id}_summary.md"


def validate_config(config: ResearchConfig) -> None:
    """Raise a helpful error before an agent graph is constructed."""
    if config.max_papers_per_topic < 1:
        raise ValueError("max_papers_per_topic must be at least 1")
    if config.max_search_attempts_per_topic < 1:
        raise ValueError("max_search_attempts_per_topic must be at least 1")
    if config.arxiv_num_retries < 0:
        raise ValueError("arxiv_num_retries cannot be negative")
    if config.arxiv_delay_seconds < 0:
        raise ValueError("arxiv_delay_seconds cannot be negative")


def default_backend() -> BackendProtocol:
    """Create the local artifact backend used when one is not supplied."""
    artifact_root = Path(__file__).resolve().parents[1] / "output"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return FilesystemBackend(root_dir=artifact_root, virtual_mode=True)


def resolve_model(
    model: object | None, *, default_name: Literal["model", "strong_model"] = "model"
) -> object:
    """Use the requested configured model only when a caller did not supply one."""
    if model is not None:
        return model
    from configs import models

    return getattr(models, default_name)
