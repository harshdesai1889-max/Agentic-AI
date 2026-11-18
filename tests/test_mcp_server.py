"""
tests/test_mcp_server.py

Tests for mcp_servers/basic_tools_server.py.

Strategy:
- Use server._tool_manager (sync) for tool registration assertions.
- Call tool wrapper functions directly for unit-level tests.
- Use asyncio.run(server.call_tool(...)) for integration tests.
- No live MCP stdio transport started; no paid API calls.
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import pytest

# Ensure project root is in sys.path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import the server module — registers tools as a side effect.
import mcp_servers.basic_tools_server as mcp_module
from mcp_servers.basic_tools_server import server, calculate, current_datetime, count_words


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_list_tools():
    """Return the list of registered Tool objects synchronously."""
    return server._tool_manager.list_tools()


def _run_call_tool(name: str, arguments: dict):
    """Run server.call_tool asynchronously and return the result."""
    return asyncio.run(server.call_tool(name, arguments))


def _result_to_dict(result) -> dict:
    """
    Parse a CallToolResult into a Python dict.
    MCPServer returns dict results as JSON text in content[0].text.
    """
    if hasattr(result, "content") and result.content:
        item = result.content[0]
        text = item.text if hasattr(item, "text") else str(item)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"raw": text}
    return {}


# ---------------------------------------------------------------------------
# Server initialisation
# ---------------------------------------------------------------------------

class TestServerInitialisation:
    def test_server_instance_created(self):
        """The module-level server must be an MCPServer."""
        from mcp.server.mcpserver import MCPServer
        assert isinstance(server, MCPServer)

    def test_server_name(self):
        assert server.name == "basic-tools-server"

    def test_server_has_title(self):
        assert server.title == "Basic Tools MCP Server"

    def test_server_module_exposes_server(self):
        """The module attribute 'server' must exist and not be None."""
        assert mcp_module.server is not None


# ---------------------------------------------------------------------------
# Tool registration (sync via _tool_manager)
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_three_tools_registered(self):
        tools = _sync_list_tools()
        names = [t.name for t in tools]
        assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}: {names}"

    def test_calculate_registered(self):
        names = {t.name for t in _sync_list_tools()}
        assert "calculate" in names

    def test_current_datetime_registered(self):
        names = {t.name for t in _sync_list_tools()}
        assert "current_datetime" in names

    def test_count_words_registered(self):
        names = {t.name for t in _sync_list_tools()}
        assert "count_words" in names

    def test_tool_descriptions_are_non_empty(self):
        for tool in _sync_list_tools():
            assert tool.description, f"Tool '{tool.name}' has an empty description."

    def test_tool_names_match_expected_set(self):
        names = {t.name for t in _sync_list_tools()}
        assert names == {"calculate", "current_datetime", "count_words"}


# ---------------------------------------------------------------------------
# Direct function-level tests (bypass MCP transport entirely)
# ---------------------------------------------------------------------------

class TestCalculateDirect:
    def test_simple_addition(self):
        assert calculate("2 + 2") == {"result": 4}

    def test_multiplication(self):
        assert calculate("3 * 7") == {"result": 21}

    def test_power(self):
        assert calculate("2 ** 10") == {"result": 1024}

    def test_float_result(self):
        result = calculate("10 / 4")
        assert "result" in result
        assert abs(result["result"] - 2.5) < 1e-9

    def test_complex_expression(self):
        assert calculate("(10 + 5) * 2 - 3") == {"result": 27}

    def test_empty_expression_returns_error(self):
        result = calculate("")
        assert "error" in result

    def test_blank_expression_returns_error(self):
        result = calculate("   ")
        assert "error" in result

    def test_unsafe_expression_returns_error(self):
        result = calculate("__import__('os').system('echo hi')")
        assert "error" in result


class TestCurrentDatetimeDirect:
    def test_returns_datetime_key(self):
        result = current_datetime()
        assert "datetime" in result, f"Expected 'datetime' key, got: {result}"

    def test_datetime_is_iso_string(self):
        from datetime import datetime
        dt_str = current_datetime()["datetime"]
        parsed = datetime.fromisoformat(dt_str)
        assert parsed is not None

    def test_datetime_is_non_empty(self):
        assert current_datetime()["datetime"].strip() != ""


class TestCountWordsDirect:
    def test_simple_sentence(self):
        result = count_words("hello world")
        assert result["word_count"] == 2
        assert result["character_count"] == 11

    def test_single_word(self):
        assert count_words("Python")["word_count"] == 1

    def test_empty_string(self):
        result = count_words("")
        assert result["word_count"] == 0

    def test_none_treated_as_empty(self):
        result = count_words(None)
        assert result["word_count"] == 0

    def test_multi_word(self):
        assert count_words("the quick brown fox jumps")["word_count"] == 5

    def test_character_count_includes_spaces(self):
        # 'ab cd' has 5 characters (a, b, space, c, d)
        assert count_words("ab cd")["character_count"] == 5


# ---------------------------------------------------------------------------
# call_tool() integration tests (async via asyncio.run)
# ---------------------------------------------------------------------------

class TestCallToolIntegration:
    def test_calculate_via_call_tool(self):
        result = _run_call_tool("calculate", {"expression": "6 * 7"})
        data = _result_to_dict(result)
        assert data.get("result") == 42, f"Expected 42, got: {data}"

    def test_current_datetime_via_call_tool(self):
        result = _run_call_tool("current_datetime", {})
        data = _result_to_dict(result)
        assert "datetime" in data, f"Expected 'datetime' key, got: {data}"

    def test_count_words_via_call_tool(self):
        result = _run_call_tool("count_words", {"text": "MCP is working"})
        data = _result_to_dict(result)
        assert data.get("word_count") == 3, f"Expected 3, got: {data}"

    def test_calculate_addition_via_call_tool(self):
        result = _run_call_tool("calculate", {"expression": "100 + 23"})
        data = _result_to_dict(result)
        assert data.get("result") == 123

    def test_calculate_invalid_not_fatal(self):
        """
        An invalid expression should return an error dict, not crash the server.
        MCPServer surfaces tool errors as content (is_error may be True or False
        depending on whether the tool raises or returns an error dict).
        """
        result = _run_call_tool("calculate", {"expression": "not_a_number!!!"})
        data = _result_to_dict(result)
        # Either the result dict has 'error', or is_error is set.
        is_err = getattr(result, "is_error", False)
        has_error_key = "error" in data
        assert is_err or has_error_key, f"Expected error signal, got is_error={is_err}, data={data}"
