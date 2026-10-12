from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import BackendProtocol
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import ToolCallLimitMiddleware

from ..utils import (
    ResearchConfig,
    ResearcherOutput,
    default_backend,
    resolve_model,
    validate_config,
)
from .summarizer import summarizer_subagent
from .tools import create_download_paper_tool, create_search_recent_papers_tool

RESEARCHER_PROMPT = """
You are a workflow-execution agent. Your job is to research the assigned topic by executing the tools below in order.

## Inputs
Your task description specifies:
- Topic ID and research question
- Scope and constraints
- Configured recency scope
- Report path: /research/<topic-id>/report.md

Treat these inputs as authoritative. Research only the assigned topic and stay within its scope.

1. Extract search terms. 
Before calling any tool, identify **2–3 short, specific search keywords** or phrases directly from the topic. 
Prefer named methods, tasks, techniques, or domain terms. Do not use full sentences, generic filler, or unrelated expansions. 
Do not convert the topic into a sentence or append generic phrases.
Use these extracted terms as the queries for `search_recent_papers`. Do not skip keyword extraction.

2. Search and select. 
Call `search_recent_papers` at most {max_search_attempts} times. 
Every search call must include `published_after={published_after}` and
`published_before={published_before}` (use null when a bound is absent); these
are tool inputs, not implicit defaults.
Deduplicate results by stable paper ID, prioritize the configured date range, and select at most {max_papers} papers. 
If the tool returns status: terminal_error, do not retry. Write /research/<topic-id>/report.md stating research was unavailable, then return failure.

3. Download papers. 
Call `download_paper` for every selected paper before delegation with exactly two fields:
`url` (the paper's PDF URL) and `paper_path` (the full destination path, for example
`/research/<topic-id>/<paper_id>.pdf`). Do not pass a topic or paper ID separately.
Record failed or invalid downloads in errors.md and continue.

4. Delegate summaries. 
In one assistant run, call `task(summarizer)` once per successfully downloaded paper, using parallel fan-out. 
The task tool accepts only the subagent type and one description argument. Put ALL fields inside that description string; never pass them as separate arguments:
```
topic: <topic>
paper_id: <stable paper ID>
paper_path: /research/<topic-id>/<paper_id>.pdf
output_path: /research/<topic-id>/<paper_id>_summary.md
title: <title>
authors: <authors>
published: <publication date>
identifier: <arXiv/DOI>
url: <primary URL>
```

Never use titles as filenames, omit paths, or ask the summarizer to discover files.

5. Compile the report. 
After all summarizers finish, read only generated *_summary.md files. 
Write /research/<topic-id>/report.md using the required report sections. 
Record every failed, duplicate, omitted, or unavailable item in /research/<topic-id>/errors.md. Continue despite individual failures.

6. Return only this status structure. Never include full summaries. 
```
status: success | fail
count: <number of successfully summarized papers>
error: None | /research/<topic-id>/errors.md
report: None | /research/<topic-id>/report.md
```

**Critical rules:** 
- Extract keywords before searching. 
- Never exceed tool-call limits. 
- Never claim a paper was summarized unless its summary file was successfully generated.
- Do not return a final status until the workflow has been executed as far as possible.
"""


def _researcher_permissions() -> list[FilesystemPermission]:
    return [
        FilesystemPermission(
            operations=["read", "write"], paths=["/research/**"], mode="allow"
        ),
        FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
    ]


def build_researcher(
    model: Any | None = None,
    *,
    config: ResearchConfig | None = None,
    backend: BackendProtocol | None = None,
):
    """Build the per-topic researcher graph for standalone or nested use."""
    config = config or ResearchConfig()
    validate_config(config)
    model = resolve_model(model, default_name="strong_model")
    backend = backend or default_backend()
    return create_deep_agent(
        model=model,
        backend=backend,
        tools=[
            create_search_recent_papers_tool(config, backend),
            create_download_paper_tool(backend),
        ],
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="search_recent_papers",
                run_limit=config.max_search_attempts_per_topic,
                exit_behavior="end",
            )
        ],
        system_prompt=RESEARCHER_PROMPT.format(
            max_papers=config.max_papers_per_topic,
            max_search_attempts=config.max_search_attempts_per_topic,
            published_after=config.published_after.isoformat()
            if config.published_after
            else "null",
            published_before=config.published_before.isoformat()
            if config.published_before
            else "null",
        ),
        subagents=[summarizer_subagent(model, backend=backend)],
        permissions=_researcher_permissions(),
        response_format=ResearcherOutput,
    )


def researcher_subagent(
    model: Any | None = None,
    *,
    config: ResearchConfig | None = None,
    backend: BackendProtocol | None = None,
) -> dict[str, Any]:
    """Return the DeepAgents subagent descriptor for a standalone researcher."""
    return {
        "name": "researcher",
        "description": "Research one topic, fan out paper summaries, and write its report.",
        "runnable": build_researcher(model, config=config, backend=backend),
    }
