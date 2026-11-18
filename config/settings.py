import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically locate and load .env from the project root or current working directory
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.is_file():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

# OpenRouter / NVIDIA Nemotron Settings (Default Provider)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL1 = "nvidia/nemotron-3.5-lightning:free"
MODEL2 = "nvidia/nemotron-3-ultra-550b-a55b:free"
MAX_TOOL_ROUNDS = 5

# Google Gemini Settings (Optional Provider)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_api_key(required: bool = True) -> str:
    """Retrieve and validate the OpenRouter API key."""
    key = os.getenv("OPENROUTER_API_KEY")
    if required and not key:
        raise RuntimeError("OPENROUTER_API_KEY missing from .env")
    return key or ""


def get_gemini_api_key(required: bool = False) -> str:
    """Retrieve optional Gemini API key."""
    key = os.getenv("GEMINI_API_KEY")
    if required and not key:
        raise RuntimeError("GEMINI_API_KEY missing from .env")
    return key or ""


def validate_config():
    """Validate that all essential default configuration values are set.
    
    Note: Only OPENROUTER_API_KEY is required for the default NVIDIA Nemotron workflow.
    GEMINI_API_KEY is optional.
    """
    get_api_key(required=True)


# Module-level API key variable for backward compatibility
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
