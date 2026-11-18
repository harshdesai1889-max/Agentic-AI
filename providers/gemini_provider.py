import json
import os
import uuid
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from core.provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
)
from config.settings import GEMINI_API_KEY


class GeminiProvider(LLMProvider):
    """Provider adapter for Google Gemini models using the official google-genai SDK."""

    provider_name: str = "gemini"

    def __init__(self, api_key: Optional[str] = None, client: Any = None):
        """Initialize the Gemini provider.

        Args:
            api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY from environment.
            client: Optional injected genai.Client (useful for unit testing).
        """
        self._call_cache: Dict[str, Any] = {}

        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY")

        if client is not None:
            self.client = client
        elif self.api_key and self.api_key.strip():
            self.client = genai.Client(api_key=self.api_key.strip())
        else:
            self.client = None

    # =========================================================================
    # TOOL CONVERSION
    # =========================================================================

    def _convert_tools(self, tools: Optional[List[dict]]) -> Optional[List[types.Tool]]:
        """Convert OpenAI-compatible tool schemas into Gemini Tool declarations."""
        if not tools:
            return None

        declarations: List[types.FunctionDeclaration] = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            fn = tool.get("function", tool) if "function" in tool else tool
            if not isinstance(fn, dict):
                continue

            name = fn.get("name")
            if not name:
                continue

            description = fn.get("description", "")
            parameters = fn.get("parameters", {})

            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            )

        if not declarations:
            return None

        return [types.Tool(function_declarations=declarations)]

    # =========================================================================
    # TOOL CHOICE CONVERSION
    # =========================================================================

    def _convert_tool_choice(self, tool_choice: str) -> Optional[types.ToolConfig]:
        """Convert standard tool_choice strings into Gemini ToolConfig."""
        if not tool_choice or tool_choice == "auto":
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )
        if tool_choice == "none":
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            )
        if tool_choice in ("any", "required"):
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )
        return None

    # =========================================================================
    # MESSAGE CONVERSION
    # =========================================================================

    def _convert_messages(
        self, messages: List[LLMMessage]
    ) -> tuple[Optional[str], List[types.Content]]:
        """Convert internal LLMMessages into Gemini system instruction and Content items."""
        system_parts: List[str] = []
        contents: List[types.Content] = []
        tool_call_map: Dict[str, str] = {}

        for message in messages:
            if not isinstance(message, LLMMessage):
                raise ValueError("All messages must be instances of LLMMessage")

            # 1. System messages -> mapped to system_instruction
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)

            # 2. User messages
            elif message.role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=message.content or "")],
                    )
                )

            # 3. Assistant / Model messages
            elif message.role == "assistant":
                parts: List[types.Part] = []
                if message.content:
                    parts.append(types.Part.from_text(text=message.content))

                if message.tool_calls:
                    for call in message.tool_calls:
                        if call.id:
                            tool_call_map[call.id] = call.name

                        cached_part = self._call_cache.get(call.id)
                        if cached_part is None and call.name and not str(call.id).startswith("call_"):
                            cached_part = self._call_cache.get(call.name)

                        if cached_part is not None:
                            parts.append(cached_part)
                        else:
                            args = call.arguments if isinstance(call.arguments, dict) else {}
                            fc_kwargs: Dict[str, Any] = {"name": call.name, "args": args}
                            if call.id and not call.id.startswith("gemini_"):
                                fc_kwargs["id"] = call.id
                            fc = types.FunctionCall(**fc_kwargs)
                            part_kwargs: Dict[str, Any] = {"function_call": fc}
                            sig = getattr(call, "thought_signature", None)
                            if sig:
                                part_kwargs["thought_signature"] = sig
                            parts.append(types.Part(**part_kwargs))

                if parts:
                    contents.append(types.Content(role="model", parts=parts))

            # 4. Tool execution response messages
            elif message.role == "tool":
                tool_name = message.tool_name or tool_call_map.get(
                    message.tool_call_id or "", "unknown_tool"
                )

                # Parse JSON string content into a dictionary if needed
                if isinstance(message.content, dict):
                    response_payload = message.content
                elif isinstance(message.content, str):
                    try:
                        parsed = json.loads(message.content)
                        response_payload = parsed if isinstance(parsed, dict) else {"result": parsed}
                    except Exception:
                        response_payload = {"output": message.content}
                else:
                    response_payload = {"output": str(message.content or "")}

                response_part = types.Part.from_function_response(
                    name=tool_name,
                    response=response_payload,
                )

                # Group consecutive tool responses in the same user turn if applicable
                if contents and contents[-1].role == "user" and any(
                    getattr(p, "function_response", None) for p in contents[-1].parts
                ):
                    contents[-1].parts.append(response_part)
                else:
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[response_part],
                        )
                    )

            else:
                raise ValueError(f"Unsupported message role: {message.role}")

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    # =========================================================================
    # GENERATE MODEL RESPONSE
    # =========================================================================

    def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Generate a response using Google Gemini through the official google-genai SDK.

        Args:
            messages: List of normalized LLMMessage objects.
            model: Gemini model identifier (e.g. 'gemini-3.8-flash').
            tools: Optional OpenAI-format tool definitions.
            tool_choice: Tool calling strategy ('auto', 'none', 'any').

        Returns:
            Normalized LLMResponse containing content, tool calls, usage, and error.
        """
        # 1. Configuration check
        if self.client is None:
            return LLMResponse(
                error="GEMINI_API_KEY missing from environment. Configure GEMINI_API_KEY to use Gemini."
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

        # 4. Message and tool conversion
        try:
            system_instruction, contents = self._convert_messages(messages)
        except ValueError as val_err:
            return LLMResponse(error=f"Message conversion error: {str(val_err)}")
        except Exception as exc:
            return LLMResponse(error=f"Unexpected error converting messages: {str(exc)}")

        if not contents:
            return LLMResponse(error="No conversation content found to send to Gemini.")

        # 5. Build GenerateContentConfig
        config_kwargs: Dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools:
            gemini_tools = self._convert_tools(tools)
            if gemini_tools:
                config_kwargs["tools"] = gemini_tools
                tool_config = self._convert_tool_choice(tool_choice)
                if tool_config:
                    config_kwargs["tool_config"] = tool_config

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        # 6. Execute model call
        try:
            response = self.client.models.generate_content(
                model=model.strip(),
                contents=contents,
                config=config,
            )
        except Exception as exc:
            return LLMResponse(error=f"Gemini API error: {str(exc)}")

        # 7. Normalize Gemini response into LLMResponse
        if not response or not getattr(response, "candidates", None):
            return LLMResponse(
                raw_response=response,
                error="Empty response received from Gemini model.",
            )

        text_parts: List[str] = []
        tool_calls: List[ToolCallRequest] = []

        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "text", None):
                        text_parts.append(part.text)
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        fc_name = getattr(fc, "name", "")
                        if not fc_name:
                            continue

                        raw_args = getattr(fc, "args", {})
                        if isinstance(raw_args, dict):
                            args = raw_args
                        elif hasattr(raw_args, "__iter__"):
                            try:
                                args = dict(raw_args)
                            except Exception:
                                args = {}
                        else:
                            args = {}

                        call_id = getattr(fc, "id", None) or f"gemini_{fc_name}_{uuid.uuid4().hex[:8]}"
                        sig = getattr(part, "thought_signature", None)

                        # Cache original Part so subsequent message conversions preserve thought_signature
                        self._call_cache[call_id] = part
                        self._call_cache[fc_name] = part

                        tool_calls.append(
                            ToolCallRequest(
                                id=call_id,
                                name=fc_name,
                                arguments=args,
                                thought_signature=sig,
                            )
                        )

        # Fallback to response.function_calls accessor if candidate parts were not directly exposed
        if not tool_calls and getattr(response, "function_calls", None):
            for fc in response.function_calls:
                fc_name = getattr(fc, "name", "")
                if not fc_name:
                    continue
                raw_args = getattr(fc, "args", {})
                args = dict(raw_args) if isinstance(raw_args, dict) else {}
                call_id = getattr(fc, "id", None) or f"gemini_{fc_name}_{uuid.uuid4().hex[:8]}"
                tool_calls.append(
                    ToolCallRequest(
                        id=call_id,
                        name=fc_name,
                        arguments=args,
                    )
                )

        content = "".join(text_parts).strip() if text_parts else None
        if not content and not tool_calls:
            fallback_text = getattr(response, "text", None)
            if fallback_text:
                content = fallback_text.strip()

        if not content and not tool_calls:
            return LLMResponse(
                raw_response=response,
                error="Empty response received from Gemini model (no text or tool calls).",
            )

        # Extract usage metadata
        usage: Optional[Dict[str, int]] = None
        if getattr(response, "usage_metadata", None):
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw_response=response,
        )
