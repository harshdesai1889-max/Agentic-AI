import os
from pathlib import Path
from functools import lru_cache

# Base directory for prompts
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=16)
def load_prompt(filename: str, fallback: str = "") -> str:
    """Safely load a prompt from the prompts directory.

    Args:
        filename: Name of the prompt file (e.g., 'assistant.txt').
        fallback: Default string returned if file cannot be read.

    Returns:
        The content of the prompt file as a string.
    """
    path = _PROMPTS_DIR / filename
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return fallback.strip()


def clear_prompt_cache():
    """Clear the cached prompt contents."""
    load_prompt.cache_clear()

