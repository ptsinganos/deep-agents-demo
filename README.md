# DeepAgents fan-out research workflow

`src.agents` provides a nested DeepAgents workflow:

`orchestrator → researcher (one/topic) → summarizer (one/paper)`.

Artifacts use `FilesystemBackend` mounted at the local `output/` directory:

```text
/research/<topic_id>/<paper_id>_summary.md
/research/<topic_id>/<paper_id>.pdf
/research/<topic_id>/report.md
/research/<topic_id>/errors.md   # only when needed
```

## Run with an LLM

```python
from configs.models import strong_model
from src.agents import ResearchConfig, build_orchestrator

agent = build_orchestrator(
    strong_model,
    config=ResearchConfig(
        max_papers_per_topic=3,
        published_after=None,  # or datetime.date(2024, 1, 1)
    ),
)
result = agent.invoke({"messages": [{"role": "user", "content": "Run a workflow researching RAG evaluation and agent memory."}]})
print(result["messages"][-1].content)
```

## Build an individual agent

Every agent has its own builder, so scripts and notebooks can exercise a
single level with a real model or a test double.  Supply a shared backend when
you want a researcher and its summarizer to use the same artifacts.

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents import ResearchConfig, build_orchestrator, build_researcher, build_summarizer

model = FakeListChatModel(responses=["done"])
summarizer = build_summarizer(model)
researcher = build_researcher(model, config=ResearchConfig(max_papers_per_topic=2))
orchestrator = build_orchestrator(model, config=ResearchConfig(max_papers_per_topic=2))
```

`build_orchestrator` builds the complete hierarchy. Individual builders remain
available for exercising a single agent level.

The orchestrator has a read-only `/research/**` permission rule and an FS-tool allowlist with no write/edit/delete operation. The researcher and summarizer use their own permission rules and prompts; all artifact reads and writes go through the DeepAgents harness filesystem.

DeepAgents task calls are requested in a single turn at both levels, allowing the framework to run topic and paper fan-out in parallel. `ResearchConfig` bounds each stage and makes recency configurable. Search/download failures and failed summaries are isolated and recorded in deterministic `errors.md` artifacts.

## Test

```bash
uv run python -m unittest discover -s tests -v
```

Tests require no network/API keys and validate the native DeepAgents graph construction, filesystem permissions, deterministic artifact paths, and the two fan-out delegation contracts. Integration runs exercise the model-driven task calls and harness filesystem end to end.
