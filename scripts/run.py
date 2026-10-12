import datetime
import os
from pathlib import Path
from typing import Any

from deepagents.backends import FilesystemBackend
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client

from configs.environment import load_environment
from src.agents import ResearchConfig, build_orchestrator

load_environment()


def log(message: str) -> None:
    """Emit progress immediately, including while a model/tool call is pending."""
    print(f"[research-workflow] {message}", flush=True)


class WorkflowProgressHandler(BaseCallbackHandler):
    """Concise local feedback; LangSmith tracing remains handled separately."""

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **_: Any,
    ) -> None:
        if parent_run_id is None:
            log(f"graph started (run={run_id})")

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **_: Any,
    ) -> None:
        if parent_run_id is None:
            log(f"graph completed (run={run_id})")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **_: Any,
    ) -> None:
        if parent_run_id is None:
            log(f"graph failed (run={run_id}): {type(error).__name__}: {error}")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        **_: Any,
    ) -> None:
        name = serialized.get("name", "chat model")
        log(f"model request started: {name} (run={run_id})")

    def on_llm_end(self, response: Any, *, run_id: Any, **_: Any) -> None:
        log(f"model response received (run={run_id})")

    def on_llm_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        log(f"model request failed (run={run_id}): {type(error).__name__}: {error}")

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *, run_id: Any, **_: Any
    ) -> None:
        log(f"tool started: {serialized.get('name', 'unknown')} (run={run_id})")

    def on_tool_end(self, output: Any, *, run_id: Any, **_: Any) -> None:
        log(f"tool completed (run={run_id})")

    def on_tool_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        log(f"tool failed (run={run_id}): {type(error).__name__}: {error}")


def log_tracing_error(error: Exception) -> None:
    log(f"LangSmith submission error: {type(error).__name__}: {error}")


custom_client = Client(
    timeout_ms=(10000, 60000),
    tracing_error_callback=log_tracing_error,
)

# 3. Create the tracer instance
custom_tracer = LangChainTracer(
    project_name=os.getenv("LANGSMITH_PROJECT", "default"),
    client=custom_client,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUT_DIR.mkdir(exist_ok=True)
backend = FilesystemBackend(root_dir=OUT_DIR, virtual_mode=True)
log(f"artifact filesystem mounted: / -> {OUT_DIR}")
log("building DeepAgents research workflow")
agent = build_orchestrator(
    config=ResearchConfig(
        max_papers_per_topic=3,
        published_after=datetime.date(2025, 1, 1),
        max_search_attempts_per_topic=2,
    ),
    backend=backend,
)
log(
    "LangSmith client configured "
    f"(endpoint={custom_client.api_url}, api_key_present={bool(custom_client.api_key)}, "
    f"project={custom_tracer.project_name})"
)

# Pass the custom tracer through the standard Runnable config. It propagates
# through the DeepAgents graph and all nested researcher/summarizer task runs.
try:
    log("invoking workflow")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Run a workflow researching 'agent memory' in 2024-2026.",
                }
            ]
        },
        config={
            "callbacks": [
                custom_tracer,
                # WorkflowProgressHandler()
            ]
        },
    )
except Exception as exc:
    log(f"workflow stopped: {type(exc).__name__}: {exc}")
    raise
else:
    log("workflow returned a final state")
    print(result["messages"][-1].content, flush=True)

artifacts = sorted(path for path in OUT_DIR.rglob("*") if path.is_file())
print("\nArtifacts written directly to output/:")
for path in artifacts:
    print(f"  {path.relative_to(OUT_DIR)}  ({path.stat().st_size:,} bytes)")


log("flushing pending LangSmith traces")
custom_client.flush()
log("LangSmith trace flush complete")
