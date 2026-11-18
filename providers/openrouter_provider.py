import json
import os
import uuid
from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
)
from config.settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL1,
    MODEL2,
)


class OpenRouterProvider(LLMProvider):
    """Provider adapter for OpenRouter (NVIDIA Nemotron models) using OpenAI SDK."""

    provider_name: str = "openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Any = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the OpenRouter provider.

        Args:
            api_key: Optional OpenRouter API key. Defaults to OPENROUTER_API_KEY from settings.
            client: Optional injected OpenAI client (useful for unit testing).
            base_url: Optional OpenRouter base URL. Defaults to OPENROUTER_BASE_URL.
        """
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")

        self.base_url = base_url or OPENROUTER_BASE_URL

        if client is not None:
            self.client = client
        elif self.api_key and self.api_key.strip():
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key.strip(),
            )
        else:
            self.client = None

    # =========================================================================
    # CONVERT COMMON MESSAGES TO OPENAI/OPENROUTER FORMAT
    # =========================================================================

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert normalized LLMMessage objects into OpenRouter/OpenAI message format."""
        converted: List[Dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, LLMMessage):
                raise ValueError("All messages must be instances of LLMMessage")

            if message.role == "assistant":
                item: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                }

                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments)
                                if isinstance(call.arguments, dict)
                                else str(call.arguments),
                            },
                        }
                        for call in message.tool_calls
                    ]

                converted.append(item)

            elif message.role == "tool":
                converted.append({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": message.content or "",
                })

            elif message.role in ("system", "user"):
                converted.append({
                    "role": message.role,
                    "content": message.content or "",
                })

            else:
                raise ValueError(f"Unsupported message role: {message.role}")

        return converted

    # =========================================================================
    # GENERATE MODEL RESPONSE
    # =========================================================================

    def generate(
        self,
        messages: List[LLMMessage],
        model: str = MODEL1,
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Generate a response using OpenRouter models (NVIDIA Nemotron).

        Args:
            messages: List of normalized LLMMessage objects.
            model: Model identifier (defaults to MODEL1 - Nemotron Lightning).
            tools: Optional OpenAI-compatible tool definitions (e.g. from tools/schemas.py).
            tool_choice: Tool calling strategy ('auto', 'none', 'required').

        Returns:
            Normalized LLMResponse containing content, tool calls, usage, and error.
        """
        # 1. Configuration check
        if self.client is None:
            return LLMResponse(
                error="OPENROUTER_API_KEY missing from environment. Configure OPENROUTER_API_KEY to use OpenRouter."
            )

        # 2. Model argument validation
        if not model or not isinstance(model, str) or not model.strip():
            return LLMResponse(
                error="Invalid model specified. Model name must be a non-empty string."
            )

        # 3. Messages validation
        if not isinstance(messages, list):
            return LLMResponse(error="Invalid messages: must be a list of LLMMessage.")

        if not messages:
            return LLMResponse(error="Messages list cannot be empty.")

        # 4. Message conversion
        try:
            converted_messages = self._convert_messages(messages)
        except ValueError as val_err:
            return LLMResponse(error=f"Message conversion error: {str(val_err)}")
        except Exception as exc:
            return LLMResponse(error=f"Unexpected error converting messages: {str(exc)}")

        # 5. Build request payload
        kwargs: Dict[str, Any] = {
            "model": model.strip(),
            "messages": converted_messages,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # 6. Execute model call
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            return LLMResponse(error=f"OpenRouter API error: {str(exc)}")

        # 7. Normalize response into LLMResponse
        if not response or not getattr(response, "choices", None) or len(response.choices) == 0:
            return LLMResponse(
                raw_response=response,
                error="Empty response received from OpenRouter model: no choices returned.",
            )

        message = response.choices[0].message
        tool_calls: List[ToolCallRequest] = []

        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                call_id = getattr(call, "id", "") or f"call_{uuid.uuid4().hex[:8]}"
                fn = getattr(call, "function", None)
                fn_name = getattr(fn, "name", "") if fn else ""
                raw_arguments = getattr(fn, "arguments", "{}") if fn else "{}"

                if isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                elif isinstance(raw_arguments, str):
                    try:
                        arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
                        if not isinstance(arguments, dict):
                            arguments = {"argument": arguments}
                    except json.JSONDecodeError as json_err:
                        return LLMResponse(
                            raw_response=response,
                            error=f"Tool-call argument parsing error for '{fn_name}': {str(json_err)}",
                        )
                else:
                    arguments = {}

                tool_calls.append(
                    ToolCallRequest(
                        id=call_id,
                        name=fn_name,
                        arguments=arguments,
                    )
                )

        content = message.content

        # Handle empty response (no content and no tool calls)
        if not content and not tool_calls:
            return LLMResponse(
                raw_response=response,
                error="Empty response received from OpenRouter model (no content or tool calls).",
            )

        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw_response=response,
        )
