"""
MCP Server: basic_tools_server.py

Exposes the existing basic tools (calculator, get_datetime, word_counter)
as MCP tools using the official Python MCP SDK (mcp>=2.x, MCPServer).

Transport: stdio (default) — suitable for local agent integration.

Usage:
    cd C:\\AgenticAI\\day223_agents
    python mcp_servers/basic_tools_server.py

This server is intentionally decoupled from the agent engine and provider
architecture. It delegates all business logic to tools/basic_tools.py.
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any, Optional

# Allow running from the project root or the mcp_servers/ directory.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.server.mcpserver import MCPServer

# Reuse existing implementations — no business logic duplication.
from tools.basic_tools import calculator, get_datetime, word_counter


def _exec_mcp_tool(tool_func, tool_name: str, args: dict, agent_role: str = "ExecutorAgent", security: Any = None) -> dict:
    """Helper executing tool with security authorization and transparent observability."""
    try:
        from core.security import get_active_security_manager, SecurityDecision
        sec = security or get_active_security_manager()
        if sec:
            decision, reason = sec.check_tool_permission(
                agent_role=agent_role,
                tool_name=tool_name,
                tool_args=args,
            )
            if decision != SecurityDecision.ALLOW:
                return {"error": f"Security policy rejected tool execution: {reason}"}
    except Exception:
        pass

    try:
        from core.observability import get_active_observability_manager, instrument_tool_execution
        obs = get_active_observability_manager()
        if obs:
            res = instrument_tool_execution(
                tool_func=tool_func,
                tool_name=tool_name,
                tool_args=args,
                observability=obs,
                agent_name="mcp_server",
            )
        else:
            res = tool_func(**args)
    except Exception:
        res = tool_func(**args)

    try:
        from core.security import get_active_security_manager, TrustClassification
        sec = security or get_active_security_manager()
        if sec:
            sec.inspect_content(content=res, source_trust=TrustClassification.TOOL_OUTPUT_UNTRUSTED)
    except Exception:
        pass

    return res


execute_mcp_tool = _exec_mcp_tool

# ---------------------------------------------------------------------------
# Server definition
# ---------------------------------------------------------------------------

server = MCPServer(
    name="basic-tools-server",
    title="Basic Tools MCP Server",
    description=(
        "Exposes calculator, datetime, and word-counter utilities "
        "from the day223_agents project as MCP tools."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Tool: calculate
# ---------------------------------------------------------------------------

@server.tool(
    name="calculate",
    description=(
        "Evaluate a mathematical expression using a secure AST-based evaluator. "
        "Supports +, -, *, /, ** (power), and parentheses. "
        "Returns {\"result\": <number>} on success, or {\"error\": \"<message>\"} on failure."
    ),
)
def calculate(expression: str) -> dict:
    """Calculate a mathematical expression.

    Args:
        expression: A mathematical expression string, e.g. '2 + 3 * 4'.

    Returns:
        dict with 'result' (numeric) or 'error' (string).
    """
    if not expression or not expression.strip():
        return {"error": "Expression must not be empty."}
    return _exec_mcp_tool(calculator, "calculator", {"expression": expression})


# ---------------------------------------------------------------------------
# Tool: current_datetime
# ---------------------------------------------------------------------------

@server.tool(
    name="current_datetime",
    description=(
        "Return the current date and time in ISO 8601 format. "
        "Returns {\"datetime\": \"<ISO string>\"}."
    ),
)
def current_datetime() -> dict:
    """Get the current date and time.

    Returns:
        dict with 'datetime' key containing an ISO 8601 formatted string.
    """
    return _exec_mcp_tool(get_datetime, "get_datetime", {})


# ---------------------------------------------------------------------------
# Tool: count_words
# ---------------------------------------------------------------------------

@server.tool(
    name="count_words",
    description=(
        "Count words and characters in the provided text. "
        "Returns {\"word_count\": <int>, \"character_count\": <int>}."
    ),
)
def count_words(text: str) -> dict:
    """Count words and characters in text.

    Args:
        text: The text to analyse.

    Returns:
        dict with 'word_count' and 'character_count' keys.
    """
    if text is None:
        text = ""
    return _exec_mcp_tool(word_counter, "word_counter", {"text": text})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with stdio transport by default.
    server.run(transport="stdio")
