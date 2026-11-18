import json
from types import SimpleNamespace
from tools.executor import execute_tool
from tools.registry import verify_registry_consistency


def _make_tool_call(name: str, args_str: str, call_id: str = "call_123"):
    """Helper to mock an OpenAI ToolCall object."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=args_str),
    )


def test_schema_and_registry_consistency():
    """Verify that all schemas match registered tools."""
    assert verify_registry_consistency() is True


def test_execute_valid_calculator():
    """Verify executing a valid calculator tool call."""
    call = _make_tool_call("calculator", json.dumps({"expression": "100 / 4"}))
    result = execute_tool(call)
    assert result == {"result": 25}


def test_execute_valid_word_counter():
    """Verify executing a valid word counter tool call."""
    call = _make_tool_call("word_counter", json.dumps({"text": "Alpha Beta Gamma"}))
    result = execute_tool(call)
    assert result == {"word_count": 3, "character_count": 16}


def test_execute_unknown_tool():
    """Verify error handling for an unknown tool name."""
    call = _make_tool_call("secret_backdoor_tool", "{}")
    result = execute_tool(call)
    assert result == {"error": "Unknown tool requested"}


def test_execute_invalid_json():
    """Verify error handling for malformed JSON arguments."""
    call = _make_tool_call("calculator", "{broken json:")
    result = execute_tool(call)
    assert result == {"error": "Invalid JSON arguments"}


def test_execute_incorrect_arguments():
    """Verify error handling for mismatched function signature parameters."""
    call = _make_tool_call("calculator", json.dumps({"unexpected_arg": 42}))
    result = execute_tool(call)
    assert result == {"error": "Incorrect tool parameters"}


def test_execute_tool_from_dict():
    """Verify tool execution from a standard dictionary structure."""
    dict_call = {
        "function": {
            "name": "calculator",
            "arguments": '{"expression": "50 * 2"}',
        }
    }
    result = execute_tool(dict_call)
    assert result == {"result": 100}

