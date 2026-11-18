import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from config.settings import MODEL1, MODEL2, GEMINI_MODEL
from core.agent_engine import AgentEngine, run_agent
from core.model_router import ModelRouter, get_route
from core.provider_interface import LLMMessage, ToolCallRequest
from providers.openrouter_provider import OpenRouterProvider
from providers.gemini_provider import GeminiProvider
from google.genai import types as gemini_types


# =============================================================================
# A. OPENROUTER INTEGRATED WORKFLOW
# =============================================================================

def test_openrouter_end_to_end_workflow():
    """Verify complete OpenRouter tool-calling workflow through AgentEngine."""
    mock_client = MagicMock()

    # Turn 1: Model requests calculator call
    tool_call = SimpleNamespace(
        id="call_openrouter_1",
        type="function",
        function=SimpleNamespace(name="calculator", arguments='{"expression": "8 * 8"}'),
    )
    turn1_msg = SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call])

    # Turn 2: Model returns synthesized answer
    turn2_msg = SimpleNamespace(role="assistant", content="8 times 8 is 64.", tool_calls=None)

    mock_client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=turn1_msg)]),
        SimpleNamespace(choices=[SimpleNamespace(message=turn2_msg)]),
    ]

    provider = OpenRouterProvider(client=mock_client)
    engine = AgentEngine(provider=provider, model=MODEL1)

    result = engine.run("What is 8 times 8?")

    assert result == "8 times 8 is 64."
    assert mock_client.chat.completions.create.call_count == 2

    # Verify tool result was delivered to the model
    second_call_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_turn = next(m for m in second_call_messages if m["role"] == "tool")
    assert tool_turn["tool_call_id"] == "call_openrouter_1"
    assert json.loads(tool_turn["content"]) == {"result": 64}


# =============================================================================
# B. GEMINI INTEGRATED WORKFLOW
# =============================================================================

def test_gemini_end_to_end_workflow():
    """Verify complete Gemini function-calling workflow through AgentEngine."""
    mock_gemini_client = MagicMock()

    # Turn 1: Model requests word counter
    fc = gemini_types.FunctionCall(name="word_counter", args={"text": "OpenRouter and Gemini unified"})
    cand1 = gemini_types.Candidate(content=gemini_types.Content(role="model", parts=[gemini_types.Part(function_call=fc)]))

    # Turn 2: Model returns synthesized answer
    cand2 = gemini_types.Candidate(content=gemini_types.Content(role="model", parts=[gemini_types.Part.from_text(text="There are 4 words in the phrase.")]))

    mock_gemini_client.models.generate_content.side_effect = [
        gemini_types.GenerateContentResponse(candidates=[cand1]),
        gemini_types.GenerateContentResponse(candidates=[cand2]),
    ]

    provider = GeminiProvider(client=mock_gemini_client)
    engine = AgentEngine(provider=provider, model=GEMINI_MODEL)

    result = engine.run("Count words in 'OpenRouter and Gemini unified'")

    assert result == "There are 4 words in the phrase."
    assert mock_gemini_client.models.generate_content.call_count == 2

    # Verify function response was delivered to Gemini
    second_call_contents = mock_gemini_client.models.generate_content.call_args_list[1].kwargs["contents"]
    resp_part = second_call_contents[-1].parts[0]
    assert resp_part.function_response.name == "word_counter"
    assert resp_part.function_response.response == {"word_count": 4, "character_count": 29}


# =============================================================================
# C. ROUTING INTEGRATION
# =============================================================================

def test_routing_fast_mode_integration():
    """Verify fast routing creates an engine with OpenRouter and Nemotron Lightning."""
    route = get_route(mode="fast")
    assert route.provider_name == "openrouter"
    assert route.model == MODEL1


def test_routing_advanced_mode_integration():
    """Verify advanced routing creates an engine with OpenRouter and Nemotron Ultra."""
    route = get_route(mode="advanced")
    assert route.provider_name == "openrouter"
    assert route.model == MODEL2


def test_routing_explicit_provider_selection():
    """Verify explicit provider selection routes to the specified provider."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Manual route response", tool_calls=None))]
    )
    custom_provider = OpenRouterProvider(client=mock_client)

    result = run_agent(
        user_task="Test manual provider override",
        model=MODEL2,
        provider=custom_provider,
    )
    assert result == "Manual route response"


# =============================================================================
# D. TOOL EXECUTION DIVERSITY
# =============================================================================

def test_integrated_tool_execution_datetime():
    """Verify get_datetime tool execution through the agent loop."""
    mock_client = MagicMock()
    tool_call = SimpleNamespace(
        id="call_date_1",
        type="function",
        function=SimpleNamespace(name="get_datetime", arguments="{}"),
    )
    mock_client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="The current time was fetched.", tool_calls=None))]),
    ]
    provider = OpenRouterProvider(client=mock_client)
    res = run_agent("What time is it?", provider=provider)
    assert res == "The current time was fetched."

    tool_msg = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    parsed_content = json.loads(tool_msg["content"])
    assert "datetime" in parsed_content


def test_integrated_tool_execution_invalid_tool_handled():
    """Verify requesting an unknown tool is handled safely without crashing the loop."""
    mock_client = MagicMock()
    bad_tool = SimpleNamespace(
        id="call_unknown_99",
        type="function",
        function=SimpleNamespace(name="non_existent_tool", arguments="{}"),
    )
    mock_client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=None, tool_calls=[bad_tool]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Handled unknown tool.", tool_calls=None))]),
    ]
    provider = OpenRouterProvider(client=mock_client)
    res = run_agent("Call secret tool", provider=provider)
    assert res == "Handled unknown tool."

    tool_msg = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    parsed = json.loads(tool_msg["content"])
    assert parsed == {"error": "Unknown tool requested"}


# =============================================================================
# E. FAILURE HANDLING & SAFETY
# =============================================================================

def test_integrated_failure_network_timeout():
    """Verify network timeout returns clean error string without crashing."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Connection timed out after 30s")
    provider = OpenRouterProvider(client=mock_client)

    result = run_agent("Task during network drop", provider=provider)
    assert "Agent execution error" in result
    assert "Connection timed out" in result


def test_integrated_failure_missing_api_key(monkeypatch):
    """Verify unconfigured provider returns clear error without attempting network calls."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("providers.openrouter_provider.OPENROUTER_API_KEY", None)
    provider = OpenRouterProvider(api_key="", client=None)

    result = run_agent("Task without API key", provider=provider)
    assert "Agent execution error" in result
    assert "OPENROUTER_API_KEY missing" in result


def test_integrated_failure_maximum_tool_rounds_breaker():
    """Verify agent loop terminates and synthesizes when max_tool_rounds is reached."""
    mock_client = MagicMock()
    infinite_tool = SimpleNamespace(
        id="call_inf",
        type="function",
        function=SimpleNamespace(name="get_datetime", arguments="{}"),
    )
    mock_client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=None, tool_calls=[infinite_tool]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=None, tool_calls=[infinite_tool]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Loop broken cleanly at max rounds.", tool_calls=None))]),
    ]
    provider = OpenRouterProvider(client=mock_client)
    result = run_agent("Loop test", provider=provider, max_tool_rounds=2)
    assert result == "Loop broken cleanly at max rounds."
    assert mock_client.chat.completions.create.call_count == 3
