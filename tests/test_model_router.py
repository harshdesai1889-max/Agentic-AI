import pytest

from config.settings import MODEL1, MODEL2, GEMINI_MODEL
from core.model_router import (
    ModelRouter,
    ModelRoute,
    get_route,
    get_provider,
    select_model,
)
from providers.openrouter_provider import OpenRouterProvider
from providers.gemini_provider import GeminiProvider


def test_default_provider_selection():
    """Verify default routing selects OpenRouter with MODEL1."""
    router = ModelRouter()
    route = router.route(mode="fast")
    assert route.provider_name == "openrouter"
    assert isinstance(route.provider, OpenRouterProvider)
    assert route.model == MODEL1


def test_explicit_provider_selection():
    """Verify explicit provider and model overrides."""
    router = ModelRouter()
    route = router.route(mode="manual", provider="openrouter", model=MODEL2)
    assert route.provider_name == "openrouter"
    assert route.model == MODEL2

    # Explicit selection with alias
    route_alias = router.route(mode="manual", provider="nvidia", model=MODEL1)
    assert route_alias.provider_name == "openrouter"


def test_fast_routing():
    """Verify fast mode routes to Nemotron Lightning (MODEL1)."""
    route = get_route(mode="fast")
    assert route.provider_name == "openrouter"
    assert route.model == MODEL1
    assert route.mode == "fast"


def test_advanced_and_complex_routing():
    """Verify advanced and complex modes route to Nemotron Ultra (MODEL2)."""
    route_adv = get_route(mode="advanced")
    assert route_adv.provider_name == "openrouter"
    assert route_adv.model == MODEL2

    route_complex = get_route(mode="complex")
    assert route_complex.provider_name == "openrouter"
    assert route_complex.model == MODEL2


def test_missing_gemini_api_key_raises_error_without_fallback(monkeypatch):
    """Verify requesting unconfigured Gemini without fallback raises clear RuntimeError."""
    router = ModelRouter(allow_fallback=False)
    monkeypatch.setattr(router, "is_provider_configured", lambda p: False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is missing from environment"):
        router.get_provider("gemini", allow_fallback=False)


def test_unknown_provider_raises_error():
    """Verify requesting an unsupported provider raises ValueError."""
    router = ModelRouter(allow_fallback=False)
    with pytest.raises(ValueError, match="Unknown provider 'claude_ai'"):
        router.get_provider("claude_ai", allow_fallback=False)


def test_invalid_model_validation():
    """Verify empty or non-string model identifiers are rejected."""
    router = ModelRouter()
    with pytest.raises(ValueError, match="Model identifier must be a non-empty string"):
        router.validate_model("openrouter", "")

    with pytest.raises(ValueError, match="Model identifier must be a non-empty string"):
        router.validate_model("gemini", "   ")


def test_provider_initialization_failure():
    """Verify clear error on bad provider specification."""
    router = ModelRouter(allow_fallback=False)
    with pytest.raises(ValueError):
        router.get_provider("invalid_provider_name")


def test_configurable_fallback(monkeypatch):
    """Verify observable fallback to OpenRouter when enabled."""
    router = ModelRouter(allow_fallback=True)
    monkeypatch.setattr(router, "is_provider_configured", lambda p: False)

    # Requesting gemini falls back to default openrouter observably
    route = router.route(mode="gemini", allow_fallback=True)
    assert route.provider_name == "openrouter"
    assert isinstance(route.provider, OpenRouterProvider)
    assert route.model == MODEL1


def test_nvidia_default_workflow_without_gemini(monkeypatch):
    """Verify default OpenRouter/NVIDIA workflow remains 100% operational when Gemini is unconfigured."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    router = ModelRouter()
    route = router.route(mode="fast")
    assert route.provider_name == "openrouter"
    assert route.model == MODEL1


def test_backward_compatible_select_model():
    """Verify original select_model() function returns correct strings."""
    assert select_model("fast") == MODEL1
    assert select_model("advanced") == MODEL2
    assert select_model("complex") == MODEL2
    assert select_model("unknown") == MODEL1

