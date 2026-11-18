import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from config.settings import MODEL1
from core.agent_engine import AgentEngine, run_agent
from core.provider_interface import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from providers.gemini_provider import GeminiProvider
from google.genai import types as gemini_types


# =============================================================================
# LEGACY TESTS (BACKWARD COMPATIBILITY WITH OPENAI CLIENT_INSTANCE INJECTION)
# =============================================================================

def _mock_tool_call(name: str, arguments: str, call_id: str = "call_123"):
    """Helper to mock an OpenAI ToolCall."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content=None, tool_calls=None):
    """Helper to mock an OpenAI chat completion response."""
    msg = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        model_dump=lambda exclude_none=True: {
            k: v for k, v in {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (tool_calls or [])
                ] or None,
            }.items()
            if not exclude_none or v is not None
        },
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_agent_direct_answer_without_tools():
    """Agent answers directly using internal intelligence when no tool is needed."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response(
        content="Photosynthesis is the process by which green plants convert light into chemical energy."
    )

    result = run_agent(
        user_task="Explain photosynthesis",
        model=MODEL1,
        client_instance=mock_client,
    )

    assert "Photosynthesis" in result
    assert mock_client.chat.completions.create.call_count == 1
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert "tools" in call_kwargs
    assert call_kwargs["tool_choice"] == "auto"


def test_agent_single_tool_call():
    """Agent calls a tool, receives the result, and returns final answer."""
    mock_client = MagicMock()

    tool_call = _mock_tool_call("calculator", json.dumps({"expression": "45 * 2"}), "call_calc_1")

    mock_client.chat.completions.create.side_effect = [
        _mock_response(content=None, tool_calls=[tool_call]),
        _mock_response(content="The product of 45 and 2 is 90."),
    ]

    result = run_agent(
        user_task="What is 45 times 2?",
        client_instance=mock_client,
    )

    assert result == "The product of 45 and 2 is 90."
    assert mock_client.chat.completions.create.call_count == 2

    second_call_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
    tool_message = next(m for m in second_call_messages if m.get("role") == "tool")
    assert tool_message["tool_call_id"] == "call_calc_1"
    parsed_tool_result = json.loads(tool_message["content"])
    assert parsed_tool_result == {"result": 90}


def test_agent_multiple_tool_calls_in_one_response():
    """Agent handles multiple tool calls returned in a single assistant turn."""
    mock_client = MagicMock()

    call1 = _mock_tool_call("calculator", json.dumps({"expression": "10 + 20"}), "call_1")
    call2 = _mock_tool_call("word_counter", json.dumps({"text": "apple banana"}), "call_2")

    mock_client.chat.completions.create.side_effect = [
        _mock_response(content=None, tool_calls=[call1, call2]),
        _mock_response(content="Sum is 30, word count is 2."),
    ]

    result = run_agent(
        user_task="Calculate 10+20 and count words in 'apple banana'",
        client_instance=mock_client,
    )

    assert result == "Sum is 30, word count is 2."
    second_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
    tool_messages = [m for m in second_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert tool_messages[1]["tool_call_id"] == "call_2"


def test_agent_multiple_tool_rounds():
    """Agent handles sequential tool calls across multiple rounds."""
    mock_client = MagicMock()

    call1 = _mock_tool_call("calculator", json.dumps({"expression": "5 * 5"}), "call_r1")
    call2 = _mock_tool_call("word_counter", json.dumps({"text": "result twenty-five"}), "call_r2")

    mock_client.chat.completions.create.side_effect = [
        _mock_response(content=None, tool_calls=[call1]),
        _mock_response(content=None, tool_calls=[call2]),
        _mock_response(content="Calculated 25 and counted 2 words."),
    ]

    result = run_agent(
        user_task="Multi-step tool round task",
        client_instance=mock_client,
    )

    assert result == "Calculated 25 and counted 2 words."
    assert mock_client.chat.completions.create.call_count == 3


def test_agent_tool_execution_failure_handled():
    """Agent gracefully passes tool errors back to the model rather than crashing."""
    mock_client = MagicMock()

    bad_call = _mock_tool_call("unknown_tool", "{}", "call_fail_1")

    mock_client.chat.completions.create.side_effect = [
        _mock_response(content=None, tool_calls=[bad_call]),
        _mock_response(content="I apologize, the requested tool was not found."),
    ]

    result = run_agent(
        user_task="Trigger unknown tool",
        client_instance=mock_client,
    )

    assert "not found" in result
    second_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
    tool_msg = next(m for m in second_messages if m.get("role") == "tool")
    assert "Unknown tool requested" in tool_msg["content"]


def test_agent_maximum_tool_round_handling():
    """Agent breaks out when max_tool_rounds is reached and synthesizes a final response."""
    mock_client = MagicMock()

    continuous_tool_call = _mock_tool_call("get_datetime", "{}", "loop_call")

    mock_client.chat.completions.create.side_effect = [
        _mock_response(content=None, tool_calls=[continuous_tool_call]),
        _mock_response(content=None, tool_calls=[continuous_tool_call]),
        _mock_response(content="Final summary after hitting max tool rounds."),
    ]

    result = run_agent(
        user_task="Run continuously",
        client_instance=mock_client,
        max_tool_rounds=2,
    )

    assert result == "Final summary after hitting max tool rounds."
    assert mock_client.chat.completions.create.call_count == 3


# =============================================================================
# NEW PROVIDER-INDEPENDENT AGENT ENGINE TESTS
# =============================================================================

def test_provider_engine_normal_text_response():
    """Engine returns direct text response using normalized provider interface."""
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate.return_value = LLMResponse(
        content="Reinforcement learning learns via trial and error rewards.",
    )

    engine = AgentEngine(provider=mock_provider, model="any-model")
    result = engine.run("Explain RL")

    assert result == "Reinforcement learning learns via trial and error rewards."
    assert mock_provider.generate.call_count == 1
    call_args = mock_provider.generate.call_args
    assert call_args.kwargs["model"] == "any-model"
    assert call_args.kwargs["tool_choice"] == "auto"


def test_provider_engine_single_tool_call():
    """Engine processes a normalized ToolCallRequest, executes tool, and returns result."""
    mock_provider = MagicMock(spec=LLMProvider)

    call_req = ToolCallRequest(
        id="calc_id_101",
        name="calculator",
        arguments={"expression": "15 * 3"},
    )

    # Turn 1: Model requests calculator
    # Turn 2: Model returns synthesized answer
    mock_provider.generate.side_effect = [
        LLMResponse(tool_calls=[call_req]),
        LLMResponse(content="15 times 3 is 45."),
    ]

    engine = AgentEngine(provider=mock_provider, model="any-model")
    result = engine.run("Compute 15 * 3")

    assert result == "15 times 3 is 45."
    assert mock_provider.generate.call_count == 2

    # Verify conversation history passed in turn 2
    turn2_messages = mock_provider.generate.call_args_list[1].kwargs["messages"]
    tool_msg = next(m for m in turn2_messages if m.role == "tool")
    assert tool_msg.tool_call_id == "calc_id_101"
    assert tool_msg.tool_name == "calculator"
    assert json.loads(tool_msg.content) == {"result": 45}


def test_provider_engine_multiple_tool_calls_in_one_turn():
    """Engine processes multiple ToolCallRequests in a single response turn."""
    mock_provider = MagicMock(spec=LLMProvider)

    tc1 = ToolCallRequest(id="c1", name="calculator", arguments={"expression": "20 + 30"})
    tc2 = ToolCallRequest(id="c2", name="word_counter", arguments={"text": "hello agent world"})

    mock_provider.generate.side_effect = [
        LLMResponse(tool_calls=[tc1, tc2]),
        LLMResponse(content="Calculation is 50 and word count is 3."),
    ]

    engine = AgentEngine(provider=mock_provider, model="any-model")
    result = engine.run("Calculate and count words")

    assert result == "Calculation is 50 and word count is 3."
    turn2_messages = mock_provider.generate.call_args_list[1].kwargs["messages"]
    tool_msgs = [m for m in turn2_messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0].tool_call_id == "c1"
    assert json.loads(tool_msgs[0].content) == {"result": 50}
    assert tool_msgs[1].tool_call_id == "c2"
    assert json.loads(tool_msgs[1].content) == {"word_count": 3, "character_count": 17}


def test_provider_engine_multiple_tool_rounds():
    """Engine iterates through sequential tool rounds up to completion."""
    mock_provider = MagicMock(spec=LLMProvider)

    t1 = ToolCallRequest(id="round1_call", name="calculator", arguments={"expression": "100 / 2"})
    t2 = ToolCallRequest(id="round2_call", name="word_counter", arguments={"text": "50"})

    mock_provider.generate.side_effect = [
        LLMResponse(tool_calls=[t1]),
        LLMResponse(tool_calls=[t2]),
        LLMResponse(content="Round 1 got 50, Round 2 verified length."),
    ]

    engine = AgentEngine(provider=mock_provider, model="any-model")
    result = engine.run("Run two rounds")

    assert result == "Round 1 got 50, Round 2 verified length."
    assert mock_provider.generate.call_count == 3


def test_provider_engine_maximum_tool_round_handling():
    """Engine enforces max_tool_rounds limit and executes final synthesis call."""
    mock_provider = MagicMock(spec=LLMProvider)

    continuous_tool = ToolCallRequest(id="loop_id", name="get_datetime", arguments={})

    # Provider constantly returns tool calls
    mock_provider.generate.side_effect = [
        LLMResponse(tool_calls=[continuous_tool]),  # round 1
        LLMResponse(tool_calls=[continuous_tool]),  # round 2
        LLMResponse(content="Final summary forced after max rounds reached."),  # final synthesis
    ]

    engine = AgentEngine(provider=mock_provider, model="any-model", max_tool_rounds=2)
    result = engine.run("Run forever")

    assert result == "Final summary forced after max rounds reached."
    assert mock_provider.generate.call_count == 3
    final_call_kwargs = mock_provider.generate.call_args_list[2].kwargs
    assert final_call_kwargs["tools"] is None
    assert final_call_kwargs["tool_choice"] == "none"


def test_provider_engine_handles_provider_errors():
    """Engine gracefully intercepts and reports provider errors without crashing."""
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate.return_value = LLMResponse(
        error="429 ResourceExhausted: Model rate limit exceeded."
    )

    engine = AgentEngine(provider=mock_provider, model="any-model")
    result = engine.run("Trigger error")

    assert "Agent execution error" in result
    assert "429 ResourceExhausted" in result


def test_provider_engine_handles_malformed_response():
    """Engine handles None or empty response objects safely."""
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.generate.return_value = None

    engine = AgentEngine(provider=mock_provider, model="any-model")
    result = engine.run("Trigger None")

    assert "Agent execution error" in result
    assert "Provider returned None response" in result


def test_provider_engine_with_gemini_provider():
    """Engine works end-to-end with GeminiProvider using mocked google-genai client."""
    mock_gemini_client = MagicMock()
    # Mock Gemini returning a FunctionCall part
    fc = gemini_types.FunctionCall(name="calculator", args={"expression": "7 * 7"})
    cand1 = gemini_types.Candidate(content=gemini_types.Content(role="model", parts=[gemini_types.Part(function_call=fc)]))

    # Mock Gemini returning final text
    cand2 = gemini_types.Candidate(content=gemini_types.Content(role="model", parts=[gemini_types.Part.from_text(text="7 * 7 is 49.")]))

    mock_gemini_client.models.generate_content.side_effect = [
        gemini_types.GenerateContentResponse(candidates=[cand1]),
        gemini_types.GenerateContentResponse(candidates=[cand2]),
    ]

    gemini_provider = GeminiProvider(client=mock_gemini_client)
    result = run_agent(
        user_task="What is 7 * 7?",
        model="gemini-3.8-flash",
        provider=gemini_provider,
    )

    assert result == "7 * 7 is 49."
    assert mock_gemini_client.models.generate_content.call_count == 2
