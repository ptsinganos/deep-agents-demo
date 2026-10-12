from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import BackendProtocol
from deepagents.middleware.filesystem import FilesystemPermission

from ..utils import SummarizerOutput, default_backend, resolve_model
from .tools import create_extract_pdf_text_tool

SUMMARIZER_PROMPT = """
Summarize EXACTLY ONE assigned paper and save the summary to the specified output file.

1. Validate inputs: Require paper_id, paper_path, output_path, paper metadata, and the assigned topic. 
If any are missing, return fail immediately. Never infer paths or search the filesystem.

2. Read: Call `extract_pdf_text` on exactly paper_path. Never use read_file for a PDF. 
Do not search directories, the web, or external sources.

3. Summarize: Use only the extracted paper text and supplied metadata. Never invent facts. 
Include these sections in order:
```
- Title
- Metadata (authors, published, arXiv/DOI, URL)
- Research Problem
- Motivation
- Method
- Contributions
- Experiments and Results
- Limitations
- Key Takeaways
- Relevance to <topic>
```
State "Not specified in the paper" when information is unavailable.

4. WRITE — MANDATORY: Call `write_file` with the exact supplied `output_path` and the complete summary as its content. Do not stop after drafting the summary. 
If the PDF is unavailable or invalid, write a concise failure message to output_path if possible.

5. Verify: Confirm that write_file reports success. If writing fails or success is uncertain, return fail. 
Never claim success without a successful write result.

Never access unrelated files, write to another path, or return the summary in your response.

**Final response only:**
- success — <output_path>
- fail — <output_path or unavailable> — <brief reason>

Completion rule: No successful write, no success status.
"""


def _summarizer_permissions() -> list[FilesystemPermission]:
    return [
        FilesystemPermission(
            operations=["write"], paths=["/research/**"], mode="allow"
        ),
        FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
    ]


def build_summarizer(
    model: Any | None = None,
    *,
    backend: BackendProtocol | None = None,
):
    """Build the paper summarizer graph for standalone or nested use."""
    model = resolve_model(model)
    backend = backend or default_backend()
    return create_deep_agent(
        model=model,
        backend=backend,
        tools=[create_extract_pdf_text_tool(backend)],
        system_prompt=SUMMARIZER_PROMPT,
        permissions=_summarizer_permissions(),
        response_format=SummarizerOutput,
    )


def summarizer_subagent(
    model: Any | None = None, *, backend: BackendProtocol | None = None
) -> dict[str, Any]:
    """Return the DeepAgents subagent descriptor for a standalone summarizer."""
    return {
        "name": "summarizer",
        "description": "Summarize exactly one paper and write only its assigned summary artifact.",
        "runnable": build_summarizer(model, backend=backend),
    }
