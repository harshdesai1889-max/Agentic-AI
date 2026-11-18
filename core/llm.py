from openai import OpenAI

from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL1,
    MODEL2,
    get_api_key,
)

# Re-export model constants for backward compatibility with existing code
__all__ = [
    "client",
    "get_client",
    "MODEL1",
    "MODEL2",
    "OPENROUTER_BASE_URL",
]


def get_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """Create and return a configured OpenAI client for OpenRouter."""
    resolved_key = api_key or get_api_key(required=True)
    resolved_base_url = base_url or OPENROUTER_BASE_URL
    return OpenAI(
        base_url=resolved_base_url,
        api_key=resolved_key,
    )


# Shared client instance initialized with environment settings
_key = OPENROUTER_API_KEY or "dummy_key_for_testing"
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=_key,
)
