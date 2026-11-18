import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from config.settings import MODEL1, MODEL2
from core.provider_interface import LLMMessage, ToolCallRequest
from providers.openrouter_provider import OpenRouterProvider
from tools.schemas import TOOLS


def _create_mock_completion(content=None, tool_calls=None, usage=None):
    """Helper to construct a mock OpenAI chat completion response."""
    msg = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(
        choices=[choice] if content is not None or tool_calls is not None else [],
        usage=usage,
    )


def test_openrouter_normal_text_response():
    """1. Test direct text answer without tool calls."""
    mock_client = MagicMock()
    usage = SimpleNamespace(prompt_tokens=15, completion_tokens=35, total_tokens=50)
    mock_client.chat.completions.create.return_value = _create_mock_completion(
        content="Transformers utilize multi-head self-attention mechanisms.",
        usage=usage,
    )

    provider = OpenRouterProvider(client=mock_client)
    messages = [
        LLMMessage(role="system", content="You are an expert AI researcher."),
        LLMMessage(role="user", content="How do transformers work?"),
    ]

    response = provider.generate(messages=messages, model=MODEL1)

    assert response.error is None
    assert response.content == "Transformers utilize multi-head self-attention mechanisms."
    assert response.tool_calls == []
    assert response.usage == {
        "prompt_tokens": 15,
        "completion_tokens": 35,
        "total_tokens": 50,
    }

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == MODEL1
    assert len(call_kwargs["messages"]) == 2


def test_openrouter_single_tool_call():
    """2. Test single tool call parsing and normalization."""
    mock_client = MagicMock()
    tool_call = SimpleNamespace(
        id="call_calc_42",
        type="function",
        function=SimpleNamespace(name="calculator", arguments='{"expression": "100 / 5"}'),
    )
    mock_client.chat.completions.create.return_value = _create_mock_completion(
        tool_calls=[tool_call]
    )

    provider = OpenRouterProvider(client=mock_client)
    messages = [LLMMessage(role="user", content="Divide 100 by 5")]

    response = provider.generate(messages=messages, model=MODEL1, tools=TOOLS)

    assert response.error is None
    assert response.content is None
    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc.name == "calculator"
    assert tc.arguments == {"expression": "100 / 5"}
    assert tc.id == "call_calc_42"


def test_openrouter_multiple_tool_calls():
    """3. Test response containing multiple tool calls in one assistant turn."""
    mock_client = MagicMock()
    tc1 = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="calculator", arguments='{"expression": "40 + 2"}'),
    )
    tc2 = SimpleNamespace(
        id="call_2",
        type="function",
        function=SimpleNamespace(name="word_counter", arguments='{"text": "answer is 42"}'),
    )
    mock_client.chat.completions.create.return_value = _create_mock_completion(
        tool_calls=[tc1, tc2]
    )

    provider = OpenRouterProvider(client=mock_client)
    messages = [LLMMessage(role="user", content="Compute and count words")]

    response = provider.generate(messages=messages, model=MODEL2, tools=TOOLS)

    assert response.error is None
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"expression": "40 + 2"}
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[1].name == "word_counter"
    assert response.tool_calls[1].arguments == {"text": "answer is 42"}
    assert response.tool_calls[1].id == "call_2"


def test_openrouter_tool_response_history():
    """4. Test message conversion when conversation history includes tool responses."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _create_mock_completion(
        content="The result of 5 * 5 is 25."
    )

    provider = OpenRouterProvider(client=mock_client)
    messages = [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="What is 5 * 5?"),
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCallRequest(id="call_5x5", name="calculator", arguments={"expression": "5 * 5"})
            ],
        ),
        LLMMessage(
            role="tool",
            tool_call_id="call_5x5",
            tool_name="calculator",
            content='{"result": 25}',
        ),
    ]

    response = provider.generate(messages=messages, model=MODEL1, tools=TOOLS)

    assert response.error is None
    assert response.content == "The result of 5 * 5 is 25."

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    converted = call_kwargs["messages"]
    assert len(converted) == 4

    assert converted[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert converted[1] == {"role": "user", "content": "What is 5 * 5?"}
    assert converted[2]["role"] == "assistant"
    assert converted[2]["tool_calls"][0]["id"] == "call_5x5"
    assert json.loads(converted[2]["tool_calls"][0]["function"]["arguments"]) == {"expression": "5 * 5"}
    assert converted[3] == {"role": "tool", "tool_call_id": "call_5x5", "content": '{"result": 25}'}


def test_openrouter_missing_api_key(monkeypatch):
    """5. Test clear configuration error when OPENROUTER_API_KEY is missing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("providers.openrouter_provider.OPENROUTER_API_KEY", None)
    provider = OpenRouterProvider(api_key="", client=None)

    response = provider.generate(
        messages=[LLMMessage(role="user", content="Hello")],
        model=MODEL1,
    )

    assert response.content is None
    assert response.error is not None
    assert "OPENROUTER_API_KEY missing from environment" in response.error


def test_openrouter_api_failure():
    """6. Test graceful exception handling when client API raises an exception."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("502 Bad Gateway / Connection timeout")

    provider = OpenRouterProvider(client=mock_client)
    messages = [LLMMessage(role="user", content="Trigger network error")]

    response = provider.generate(messages=messages, model=MODEL1)

    assert response.content is None
    assert response.error is not None
    assert "OpenRouter API error" in response.error
    assert "502 Bad Gateway" in response.error


def test_openrouter_empty_or_malformed_response():
    """7. Test handling of empty choices, malformed tool arguments, and invalid models."""
    mock_client = MagicMock()

    # Empty choices
    mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    provider = OpenRouterProvider(client=mock_client)
    messages = [LLMMessage(role="user", content="Check empty response")]

    resp_empty = provider.generate(messages=messages, model=MODEL1)
    assert resp_empty.error is not None
    assert "Empty response received" in resp_empty.error

    # Malformed JSON in tool call arguments
    bad_tool_call = SimpleNamespace(
        id="call_bad",
        type="function",
        function=SimpleNamespace(name="calculator", arguments="{broken json:"),
    )
    mock_client.chat.completions.create.return_value = _create_mock_completion(
        tool_calls=[bad_tool_call]
    )
    resp_bad_json = provider.generate(messages=messages, model=MODEL1)
    assert resp_bad_json.error is not None
    assert "Tool-call argument parsing error" in resp_bad_json.error

    # Invalid empty model name
    resp_invalid_model = provider.generate(messages=messages, model="   ")
    assert resp_invalid_model.error is not None
    assert "Invalid model specified" in resp_invalid_model.error

    # Unsupported message role
    resp_bad_role = provider.generate(
        messages=[LLMMessage(role="unsupported_role", content="text")],
        model=MODEL1,
    )
    assert resp_bad_role.error is not None
    assert "Message conversion error" in resp_bad_role.error

