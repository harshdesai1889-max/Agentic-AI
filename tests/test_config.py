import pytest
from config.settings import (
    MODEL1,
    MODEL2,
    OPENROUTER_BASE_URL,
    get_api_key,
)
from core.model_router import select_model


def test_model_identifiers():
    """Verify that Nemotron model identifiers remain unchanged."""
    assert MODEL1 == "nvidia/nemotron-3.5-lightning:free"
    assert MODEL2 == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_missing_api_key_error(monkeypatch):
    """Verify that a missing API key produces a clear RuntimeError."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY missing from .env"):
        get_api_key(required=True)


def test_model_router_modes():
    """Verify model routing for fast and advanced/complex modes."""
    assert select_model("fast") == MODEL1
    assert select_model("simple") == MODEL1
    assert select_model("routine") == MODEL1
    assert select_model("advanced") == MODEL2
    assert select_model("complex") == MODEL2
    assert select_model("planning") == MODEL2
    assert select_model("unknown_mode") == MODEL1

