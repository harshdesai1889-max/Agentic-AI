import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from config.settings import (
    MODEL1,
    MODEL2,
    GEMINI_MODEL,
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
)
from config.model_config import (
    ProviderType,
    DEFAULT_PROVIDER,
    MODE_FAST,
    MODE_ADVANCED,
    MODE_COMPLEX,
    MODE_MANUAL,
    KNOWN_OPENROUTER_MODELS,
    KNOWN_GEMINI_MODELS,
    ALLOW_PROVIDER_FALLBACK,
)
from core.provider_interface import LLMProvider
from utils.logging_config import get_logger

logger = get_logger()


@dataclass
class ModelRoute:
    """Represents a resolved provider and model selection."""
    provider_name: str
    provider: LLMProvider
    model: str
    mode: str


class ModelRouter:
    """Centralized provider and model routing system."""

    def __init__(
        self,
        default_provider: str = DEFAULT_PROVIDER.value,
        allow_fallback: bool = ALLOW_PROVIDER_FALLBACK,
    ):
        """Initialize the ModelRouter.

        Args:
            default_provider: Default provider name ('openrouter' or 'gemini').
            allow_fallback: Whether to allow observable fallback to default provider
                            if an optional provider is not configured.
        """
        self.default_provider = self._normalize_provider_name(default_provider)
        self.allow_fallback = allow_fallback
        self._provider_cache: Dict[str, LLMProvider] = {}

    @staticmethod
    def _normalize_provider_name(provider_name: Optional[str]) -> str:
        """Normalize various aliases to canonical provider names."""
        if not provider_name:
            return DEFAULT_PROVIDER.value

        cleaned = str(provider_name).strip().lower()
        if cleaned in ("openrouter", "nvidia", "nemotron", "default"):
            return "openrouter"
        if cleaned in ("gemini", "google", "genai"):
            return "gemini"

        return cleaned

    def validate_model(self, provider_name: str, model: str) -> None:
        """Validate that a model name is valid for the given provider."""
        if not model or not isinstance(model, str) or not model.strip():
            raise ValueError("Model identifier must be a non-empty string.")

        canonical_provider = self._normalize_provider_name(provider_name)
        model_str = model.strip()

        if canonical_provider == "openrouter":
            if model_str not in KNOWN_OPENROUTER_MODELS and not (
                model_str.startswith("nvidia/") or "/" in model_str
            ):
                logger.warning(
                    f"[MODEL ROUTER] Model '{model_str}' is not in known OpenRouter list."
                )

        elif canonical_provider == "gemini":
            if model_str not in KNOWN_GEMINI_MODELS and not model_str.startswith("gemini-"):
                logger.warning(
                    f"[MODEL ROUTER] Model '{model_str}' is not in known Gemini list."
                )

    def is_provider_configured(self, provider_name: str) -> bool:
        """Check whether a provider has required environment credentials configured."""
        canonical = self._normalize_provider_name(provider_name)
        if canonical == "openrouter":
            return bool(OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY"))
        if canonical == "gemini":
            return bool(GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"))
        return False

    def get_provider(
        self,
        provider_name: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Any = None,
        allow_fallback: Optional[bool] = None,
    ) -> LLMProvider:
        """Retrieve or instantiate a concrete LLMProvider.

        Args:
            provider_name: 'openrouter' or 'gemini' (defaults to default_provider).
            api_key: Optional explicit API key.
            client: Optional pre-constructed client instance.
            allow_fallback: Override fallback behavior for this request.

        Returns:
            An instantiated LLMProvider.
        """
        target_provider = self._normalize_provider_name(provider_name)
        fallback_enabled = (
            allow_fallback if allow_fallback is not None else self.allow_fallback
        )

        # 1. Check for unknown provider
        if target_provider not in ("openrouter", "gemini"):
            if fallback_enabled:
                logger.warning(
                    f"[MODEL ROUTER] Unknown provider '{target_provider}'. "
                    f"Observably falling back to default provider '{self.default_provider}'."
                )
                target_provider = self.default_provider
            else:
                raise ValueError(
                    f"Unknown provider '{target_provider}'. Supported providers: 'openrouter', 'gemini'."
                )

        # 2. Check credentials for Gemini
        if target_provider == "gemini":
            has_credentials = bool(api_key or client or self.is_provider_configured("gemini"))
            if not has_credentials:
                if fallback_enabled:
                    logger.warning(
                        "[MODEL ROUTER] Gemini provider is requested but GEMINI_API_KEY is not configured. "
                        "Observably falling back to default provider 'openrouter'."
                    )
                    target_provider = "openrouter"
                else:
                    raise RuntimeError(
                        "GEMINI_API_KEY is missing from environment. "
                        "Configure GEMINI_API_KEY to use Gemini, or enable fallback."
                    )

        # 3. Instantiate requested provider
        if target_provider == "openrouter":
            from providers.openrouter_provider import OpenRouterProvider
            return OpenRouterProvider(api_key=api_key, client=client)
        elif target_provider == "gemini":
            from providers.gemini_provider import GeminiProvider
            return GeminiProvider(api_key=api_key, client=client)

        raise ValueError(f"Could not initialize provider '{target_provider}'.")

    def route(
        self,
        mode: str = MODE_FAST,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        allow_fallback: Optional[bool] = None,
    ) -> ModelRoute:
        """Route a request based on mode, explicit provider, and model.

        Routing modes:
            - 'fast': OpenRouter + Nemotron Lightning (MODEL1)
            - 'advanced' / 'complex': OpenRouter + Nemotron Ultra (MODEL2)
            - 'gemini': Gemini + gemini-3.8-flash (GEMINI_MODEL)
            - 'manual': Uses user-specified provider and model

        Returns:
            ModelRoute containing provider_name, provider instance, model, and mode.
        """
        normalized_mode = str(mode).strip().lower()

        resolved_provider_name: str
        resolved_model: str

        # Manual explicit override takes precedence
        if normalized_mode == MODE_MANUAL or (provider is not None or model is not None):
            resolved_provider_name = self._normalize_provider_name(
                provider or self.default_provider
            )
            if model:
                resolved_model = model.strip()
            else:
                resolved_model = (
                    GEMINI_MODEL if resolved_provider_name == "gemini" else MODEL1
                )
        elif normalized_mode in (MODE_ADVANCED, MODE_COMPLEX, "planning", "ultra", "2"):
            resolved_provider_name = "openrouter"
            resolved_model = MODEL2
        elif normalized_mode in ("gemini", "flash", "google", "3"):
            resolved_provider_name = "gemini"
            resolved_model = GEMINI_MODEL
        else:
            # Default to fast mode
            resolved_provider_name = "openrouter"
            resolved_model = MODEL1

        # Validate model
        self.validate_model(resolved_provider_name, resolved_model)

        # Retrieve provider instance with fallback handling
        provider_instance = self.get_provider(
            provider_name=resolved_provider_name,
            allow_fallback=allow_fallback,
        )

        # If fallback occurred, adjust resolved provider name and default model accordingly
        actual_provider_name = getattr(provider_instance, "provider_name", None)
        if actual_provider_name is None:
            cls_name = provider_instance.__class__.__name__.lower()
            if "openrouter" in cls_name:
                actual_provider_name = "openrouter"
            elif "gemini" in cls_name:
                actual_provider_name = "gemini"

        if actual_provider_name != resolved_provider_name:
            resolved_provider_name = actual_provider_name
            if resolved_provider_name == "openrouter" and resolved_model == GEMINI_MODEL:
                resolved_model = MODEL1

        logger.info(
            f"[MODEL ROUTER] Routed task mode='{normalized_mode}' -> "
            f"provider='{resolved_provider_name}', model='{resolved_model}'"
        )

        return ModelRoute(
            provider_name=resolved_provider_name,
            provider=provider_instance,
            model=resolved_model,
            mode=normalized_mode,
        )


# Global default router instance
_default_router = ModelRouter()


def get_router() -> ModelRouter:
    """Retrieve the global default ModelRouter instance."""
    return _default_router


def get_route(
    mode: str = MODE_FAST,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    allow_fallback: Optional[bool] = None,
) -> ModelRoute:
    """Convenience function to route a task using the global router."""
    return _default_router.route(
        mode=mode,
        provider=provider,
        model=model,
        allow_fallback=allow_fallback,
    )


def get_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Any = None,
    allow_fallback: Optional[bool] = None,
) -> LLMProvider:
    """Convenience function to get a provider from the global router."""
    return _default_router.get_provider(
        provider_name=provider_name,
        api_key=api_key,
        client=client,
        allow_fallback=allow_fallback,
    )


def select_model(mode: str = "fast") -> str:
    """Backward-compatible model selection returning model string.

    - 'fast' -> MODEL1 (Nemotron Lightning)
    - 'advanced' / 'complex' -> MODEL2 (Nemotron Ultra)
    """
    normalized = str(mode).strip().lower()
    if normalized in ("advanced", "complex", "planning", "ultra", "2"):
        return MODEL2
    return MODEL1
