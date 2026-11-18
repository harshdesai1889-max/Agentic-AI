"""Model and provider routing configuration for the multi-provider architecture."""

from enum import Enum
from typing import Dict, List

from config.settings import MODEL1, MODEL2, GEMINI_MODEL


class ProviderType(str, Enum):
    """Supported LLM Providers."""
    OPENROUTER = "openrouter"
    GEMINI = "gemini"


DEFAULT_PROVIDER = ProviderType.OPENROUTER

# Routing Modes
MODE_FAST = "fast"
MODE_ADVANCED = "advanced"
MODE_COMPLEX = "complex"
MODE_MANUAL = "manual"

# Known supported models per provider
KNOWN_OPENROUTER_MODELS: List[str] = [
    MODEL1,
    MODEL2,
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

KNOWN_GEMINI_MODELS: List[str] = [
    GEMINI_MODEL,
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Default routing profiles (mode -> (provider, model))
DEFAULT_ROUTING_TABLE: Dict[str, tuple[ProviderType, str]] = {
    MODE_FAST: (ProviderType.OPENROUTER, MODEL1),
    MODE_ADVANCED: (ProviderType.OPENROUTER, MODEL2),
    MODE_COMPLEX: (ProviderType.OPENROUTER, MODEL2),
    "gemini": (ProviderType.GEMINI, GEMINI_MODEL),
}

# Configurable provider fallback setting (disabled by default to prevent silent behavior changes)
ALLOW_PROVIDER_FALLBACK = False

