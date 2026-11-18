from typing import Callable, Dict
from tools.basic_tools import (
    calculator,
    get_datetime,
    word_counter,
)
from tools.schemas import TOOLS

# Map of tool names to Python callable implementations
TOOL_REGISTRY: Dict[str, Callable] = {
    "calculator": calculator,
    "get_datetime": get_datetime,
    "word_counter": word_counter,
}


def get_tool(tool_name: str) -> Callable | None:
    """Retrieve a tool function by its registered name."""
    return TOOL_REGISTRY.get(tool_name)


def verify_registry_consistency() -> bool:
    """Verify that every tool in TOOLS schema exists in TOOL_REGISTRY and vice-versa."""
    schema_names = {tool["function"]["name"] for tool in TOOLS}
    registry_names = set(TOOL_REGISTRY.keys())
    return schema_names == registry_names
