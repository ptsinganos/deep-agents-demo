"""Project-wide environment configuration."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_environment() -> bool:
    """Load the repository's ``.env`` file without replacing shell variables.

    Call this once at the start of executable scripts. It is safe for modules
    such as ``configs.models`` to call it again because loading is idempotent.
    """
    return load_dotenv(DOTENV_PATH, override=False)
