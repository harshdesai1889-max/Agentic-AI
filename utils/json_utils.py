import json
import re
from typing import Any, Tuple, Optional


def clean_json_string(raw: str) -> str:
    """Extract and clean JSON content from raw LLM output."""
    text = raw.strip()

    # Match markdown code block ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    return text


def parse_json_safely(raw: str) -> Tuple[bool, Any, Optional[str]]:
    """Safely parse JSON text from model output or tool arguments.

    Returns:
        (success, data, error_message)
    """
    if not raw or not isinstance(raw, str):
        return False, None, "Input must be a non-empty string"

    cleaned = clean_json_string(raw)
    try:
        data = json.loads(cleaned)
        return True, data, None
    except json.JSONDecodeError as exc:
        return False, None, f"JSON decode error: {exc.msg} at line {exc.lineno} col {exc.colno}"
    except Exception as exc:
        return False, None, f"Unexpected parsing error: {str(exc)}"


def parse_json_from_llm(raw: str) -> Optional[Any]:
    """Parse JSON from LLM output string, returning parsed object or None."""
    success, data, _ = parse_json_safely(raw)
    return data if success else None

