from unittest.mock import MagicMock
from google.genai import types

from core.provider_interface import LLMMessage, ToolCallRequest
from providers.gemini_provider import GeminiProvider
from tools.schemas import TOOLS


def _create_mock_response(text=None, function_calls=None, usage_metadata=None):
    """Helper to construct a mock GenerateContentResponse."""
    parts = []
    if text:
        parts.append(types.Part.from_text(text=text))
    if function_calls:
        for fc in function_calls:
            parts.append(types.Part(function_call=fc))

    candidate = types.Candidate(content=types.Content(role="model", parts=parts))
    return types.GenerateContentResponse(
        candidates=[candidate] if parts else [],
        usage_metadata=usage_metadata,
    )


def test_gemini_normal_text_response():
    """1. Test standard direct text answer without function calls."""
    mock_client = MagicMock()
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=20,
        total_token_count=30,
    )
    mock_client.models.generate_content.return_value = _create_mock_response(
        text="Deep learning is a subset of machine learning based on neural networks.",
        usage_metadata=usage,
    )

    provider = GeminiProvider(client=mock_client)
    messages = [
        LLMMessage(role="system", content="You are a helpful science tutor."),
        LLMMessage(role="user", content="What is deep learning?"),
    ]

    response = provider.generate(messages=messages, model="gemini-3.8-flash")

    assert response.error is None
    assert response.content == "Deep learning is a subset of machine learning based on neural networks."
    assert response.tool_calls == []
    assert response.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }

    # Verify generate_content parameters
    call_args = mock_client.models.generate_content.call_args
    assert call_args.kwargs["model"] == "gemini-3.8-flash"
    assert call_args.kwargs["config"].system_instruction == "You are a helpful science tutor."


def test_gemini_single_function_call():
    """2. Test single function call parsing and normalization."""
    mock_client = MagicMock()
    fc = types.FunctionCall(name="calculator", args={"expression": "12 * 12"})
    mock_client.models.generate_content.return_value = _create_mock_response(function_calls=[fc])

    provider = GeminiProvider(client=mock_client)
    messages = [LLMMessage(role="user", content="Calculate 12 times 12")]

    response = provider.generate(messages=messages, model="gemini-3.8-flash", tools=TOOLS)

    assert response.error is None
    assert response.content is None
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call.name == "calculator"
    assert tool_call.arguments == {"expression": "12 * 12"}
    assert tool_call.id.startswith("gemini_calculator_")


def test_gemini_multiple_function_calls():
    """3. Test response containing multiple function calls in one turn."""
    mock_client = MagicMock()
    fc1 = types.FunctionCall(name="calculator", args={"expression": "50 + 50"})
    fc2 = types.FunctionCall(name="word_counter", args={"text": "agentic AI system"})
    mock_client.models.generate_content.return_value = _create_mock_response(
        function_calls=[fc1, fc2]
    )

    provider = GeminiProvider(client=mock_client)
    messages = [LLMMessage(role="user", content="Calculate 50+50 and count words in 'agentic AI system'")]

    response = provider.generate(messages=messages, model="gemini-3.8-flash", tools=TOOLS)

    assert response.error is None
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"expression": "50 + 50"}
    assert response.tool_calls[1].name == "word_counter"
    assert response.tool_calls[1].arguments == {"text": "agentic AI system"}


def test_gemini_tool_response_history():
    """4. Test message conversion when conversation history includes tool responses."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _create_mock_response(
        text="The calculation result is 144."
    )

    provider = GeminiProvider(client=mock_client)
    messages = [
        LLMMessage(role="system", content="You are a precise calculator assistant."),
        LLMMessage(role="user", content="What is 12 * 12?"),
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCallRequest(id="call_calc_99", name="calculator", arguments={"expression": "12 * 12"})
            ],
        ),
        LLMMessage(
            role="tool",
            tool_call_id="call_calc_99",
            tool_name="calculator",
            content='{"result": 144}',
        ),
    ]

    response = provider.generate(messages=messages, model="gemini-3.8-flash", tools=TOOLS)

    assert response.error is None
    assert response.content == "The calculation result is 144."

    # Inspect generated contents passed to the mock
    call_args = mock_client.models.generate_content.call_args
    contents = call_args.kwargs["contents"]
    assert len(contents) == 3

    # Turn 1: user text
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "What is 12 * 12?"

    # Turn 2: model function call
    assert contents[1].role == "model"
    assert contents[1].parts[0].function_call.name == "calculator"
    assert contents[1].parts[0].function_call.args == {"expression": "12 * 12"}

    # Turn 3: user function response
    assert contents[2].role == "user"
    assert contents[2].parts[0].function_response.name == "calculator"
    assert contents[2].parts[0].function_response.response == {"result": 144}


def test_gemini_missing_api_key(monkeypatch):
    """5. Test clear configuration error when GEMINI_API_KEY is not configured."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("providers.gemini_provider.GEMINI_API_KEY", None)
    provider = GeminiProvider(api_key="", client=None)

    response = provider.generate(
        messages=[LLMMessage(role="user", content="Hello")],
        model="gemini-3.8-flash",
    )

    assert response.content is None
    assert response.error is not None
    assert "GEMINI_API_KEY missing from environment" in response.error


def test_gemini_api_error_handling():
    """6. Test graceful exception handling when client API raises an exception."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("ResourceExhausted: Quota exceeded")

    provider = GeminiProvider(client=mock_client)
    messages = [LLMMessage(role="user", content="Trigger error")]

    response = provider.generate(messages=messages, model="gemini-3.8-flash")

    assert response.content is None
    assert response.error is not None
    assert "Gemini API error" in response.error
    assert "Quota exceeded" in response.error


def test_gemini_empty_or_malformed_model_response():
    """7. Test handling of empty candidates or invalid model parameters."""
    mock_client = MagicMock()
    # Empty candidates list
    mock_client.models.generate_content.return_value = types.GenerateContentResponse(candidates=[])

    provider = GeminiProvider(client=mock_client)
    messages = [LLMMessage(role="user", content="Check empty response")]

    response = provider.generate(messages=messages, model="gemini-3.8-flash")
    assert response.error is not None
    assert "Empty response received" in response.error

    # Invalid empty model name
    resp_invalid_model = provider.generate(messages=messages, model="   ")
    assert resp_invalid_model.error is not None
    assert "Invalid model specified" in resp_invalid_model.error

    # Unsupported message role
    resp_bad_role = provider.generate(
        messages=[LLMMessage(role="unsupported_role", content="text")],
        model="gemini-3.8-flash",
    )
    assert resp_bad_role.error is not None
    assert "Message conversion error" in resp_bad_role.error


def test_gemini_thought_signature_preserved():
    """8. Test that thought_signature is cached and preserved during multi-turn function calling."""
    mock_client = MagicMock()
    # First turn: Gemini returns a function call with thought_signature
    sig_bytes = b"encrypted_thought_signature_token"
    part = types.Part(
        function_call=types.FunctionCall(name="calculator", args={"expression": "10 * 10"}, id="call_calc_42"),
        thought_signature=sig_bytes,
    )
    mock_client.models.generate_content.return_value = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))]
    )

    provider = GeminiProvider(client=mock_client)
    res1 = provider.generate(
        messages=[LLMMessage(role="user", content="Calculate 10 * 10")],
        model="gemini-3.8-flash",
        tools=TOOLS,
    )

    assert len(res1.tool_calls) == 1
    assert res1.tool_calls[0].thought_signature == sig_bytes
    assert res1.tool_calls[0].id == "call_calc_42"

    # Second turn: sending back tool result
    mock_client.models.generate_content.return_value = _create_mock_response(text="100")
    res2 = provider.generate(
        messages=[
            LLMMessage(role="user", content="Calculate 10 * 10"),
            LLMMessage(role="assistant", tool_calls=res1.tool_calls),
            LLMMessage(role="tool", tool_call_id="call_calc_42", tool_name="calculator", content='{"result": 100}'),
        ],
        model="gemini-3.8-flash",
        tools=TOOLS,
    )

    assert res2.error is None
    assert res2.content == "100"

    # Verify that the model part sent in round 2 retained thought_signature
    call_args = mock_client.models.generate_content.call_args
    sent_contents = call_args.kwargs["contents"]
    model_turn_parts = sent_contents[1].parts
    assert model_turn_parts[0].thought_signature == sig_bytes
