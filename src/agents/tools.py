import base64
import json
import subprocess
import urllib.request
from collections.abc import Callable
from datetime import date

import arxiv
from deepagents.backends import BackendProtocol
from langchain.tools import tool

from ..utils import Paper, ResearchConfig, topic_directory

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_PDF_CHARS = 60_000
SEARCH_OVERFETCH_FACTOR = 3
HTTP_TIMEOUT_SECONDS = 30
PDF_TEXT_TIMEOUT_SECONDS = 30


def _format_tool_error(operation: str, error: BaseException | str) -> str:
    """Return the shared error format for tools with string results."""
    if isinstance(error, BaseException):
        detail = f"{type(error).__name__}: {error}"
    else:
        detail = error
    return f"{operation}_ERROR: {detail}"


def _terminal_search_error(error: BaseException, error_path: str) -> str:
    """Return the stable terminal-error contract for failed searches."""
    return json.dumps(
        {
            "status": "terminal_error",
            "error": f"search failed: {error}",
            "errors_path": error_path,
            "instruction": "Do not retry search_recent_papers for this topic. Finish the topic with the recorded error.",
        }
    )


def search_arxiv(
    topic: str,
    config: ResearchConfig,
    published_after: date | None = None,
    published_before: date | None = None,
) -> list[Paper]:
    """Search arXiv; only the researcher receives this tool in production."""
    query = f'all:"{topic}"'
    search = arxiv.Search(
        query=query,
        max_results=config.max_papers_per_topic * SEARCH_OVERFETCH_FACTOR,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers: list[Paper] = []
    client = arxiv.Client(
        page_size=max(1, config.max_papers_per_topic * SEARCH_OVERFETCH_FACTOR),
        delay_seconds=config.arxiv_delay_seconds,
        num_retries=config.arxiv_num_retries,
    )
    for result in client.results(search):
        published = result.published.date()
        if published_after and published < published_after:
            continue
        if published_before and published > published_before:
            continue
        papers.append(
            Paper(
                result.title,
                tuple(author.name for author in result.authors),
                str(published),
                result.entry_id,
                result.pdf_url,
                result.summary,
            )
        )
        if len(papers) >= config.max_papers_per_topic:
            break
    return papers


def record_topic_error(backend: BackendProtocol, topic: str, message: str) -> str:
    """Append a tool-detected failure to the deterministic topic error log."""
    path = f"{topic_directory(topic)}/errors.md"
    existing = backend.read(path)
    previous = ""
    if existing.file_data and existing.file_data["encoding"] == "utf-8":
        previous = existing.file_data["content"].rstrip()
    content = f"{previous}\n" if previous else "# Research errors\n"
    backend.write(path, f"{content}- {message}\n")
    return path


def create_search_recent_papers_tool(
    config: ResearchConfig,
    backend: BackendProtocol,
    *,
    search: Callable[
        [str, ResearchConfig, date | None, date | None], list[Paper]
    ] = search_arxiv,
):
    """Create a search tool bound to one agent's config and artifact backend."""

    @tool
    def search_recent_papers(
        topic: str,
        published_after: date | None = None,
        published_before: date | None = None,
    ) -> str:
        """Find arXiv papers for a topic within inclusive publication-date bounds.

        Args:
            topic: Search terms used to query arXiv.
            published_after: Inclusive lower publication-date bound. Pass ``None``
                to include papers from any earlier date.
            published_before: Inclusive upper publication-date bound. Pass ``None``
                to include papers from any later date.

        Returns:
            A JSON array of compact paper metadata on success. On failure, returns
            JSON with ``status: terminal_error``, an error message, and the path
            to the recorded topic error artifact.
        """
        try:
            return json.dumps(
                [
                    paper.to_tool_payload()
                    for paper in search(
                        topic, config, published_after, published_before
                    )
                ],
                default=str,
            )
        except Exception as exc:
            error_path = record_topic_error(
                backend, topic, f"arXiv search failed: {type(exc).__name__}: {exc}"
            )
            return _terminal_search_error(exc, error_path)

    return search_recent_papers


def create_download_paper_tool(backend: BackendProtocol):
    """Create the PDF download tool bound to the shared artifact backend."""

    @tool
    def download_paper(url: str, paper_path: str) -> str:
        """Download a PDF URL to the exact supplied filesystem path.

        Args:
            url: HTTP(S) URL that serves the source PDF.
            paper_path: Destination virtual path for the PDF. It must end in
                ``.pdf``; the path is used exactly as supplied.

        Returns:
            The supplied ``paper_path`` after a successful upload. Otherwise, an
            error string beginning with ``DOWNLOAD_ERROR:``.
        """
        if not paper_path.endswith(".pdf"):
            return _format_tool_error("DOWNLOAD", "paper_path must point to a PDF")
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310
                raw = response.read(MAX_PDF_BYTES + 1)
            if len(raw) > MAX_PDF_BYTES:
                return _format_tool_error(
                    "DOWNLOAD", f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MiB limit"
                )
            if not raw.startswith(b"%PDF"):
                return _format_tool_error("DOWNLOAD", f"source at {url} was not a PDF")
            backend.upload_files([(paper_path, raw)])
            return paper_path
        except Exception as exc:
            return _format_tool_error("DOWNLOAD", exc)

    return download_paper


def create_extract_pdf_text_tool(backend: BackendProtocol):
    """Create the local PDF text extraction tool bound to the artifact backend."""

    @tool
    def extract_pdf_text(paper_path: str) -> str:
        """Extract plain text from a downloaded PDF without sending it to the model.

        Args:
            paper_path: Virtual path of the downloaded PDF to extract. It must
                end in ``.pdf`` and refer to a binary PDF in the shared backend.

        Returns:
            Extracted PDF text, truncated to ``MAX_EXTRACTED_PDF_CHARS`` when
            necessary. On failure, an error string beginning with
            ``EXTRACTION_ERROR:``.
        """
        if not paper_path.endswith(".pdf"):
            return _format_tool_error("EXTRACTION", "paper_path must point to a PDF")

        result = backend.read(paper_path)
        if result.error or result.file_data is None:
            return _format_tool_error(
                "EXTRACTION", result.error or "PDF is unavailable"
            )
        if result.file_data["encoding"] != "base64":
            return _format_tool_error(
                "EXTRACTION", "assigned file was not read as binary PDF data"
            )

        try:
            pdf_bytes = base64.b64decode(result.file_data["content"], validate=True)
            completed = subprocess.run(
                ["pdftotext", "-layout", "-", "-"],
                input=pdf_bytes,
                capture_output=True,
                check=True,
                timeout=PDF_TEXT_TIMEOUT_SECONDS,
            )
            text = completed.stdout.decode("utf-8", errors="replace").strip()
        except (ValueError, subprocess.SubprocessError) as exc:
            return _format_tool_error("EXTRACTION", exc)

        if not text:
            return _format_tool_error(
                "EXTRACTION", "no extractable text was found in the PDF"
            )
        if len(text) > MAX_EXTRACTED_PDF_CHARS:
            text = f"{text[:MAX_EXTRACTED_PDF_CHARS]}\n\n[Text truncated at {MAX_EXTRACTED_PDF_CHARS:,} characters.]"
        return text

    return extract_pdf_text
