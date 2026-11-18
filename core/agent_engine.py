import json
from typing import Any, List, Optional

from config.settings import MODEL1, MAX_TOOL_ROUNDS
from core.provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
)
from tools.schemas import TOOLS
from tools.executor import execute_tool
from utils.prompt_loader import load_prompt
from utils.logging_config import get_logger

logger = get_logger()


class AgentEngine:
    """Provider-independent agent execution engine.

    Orchestrates multi-turn model interaction and tool execution through
    the normalized LLMProvider abstraction.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: str = MODEL1,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        system_prompt: Optional[str] = None,
        tools: Optional[List[dict]] = None,
    ):
        """Initialize the AgentEngine.

        Args:
            provider: Concrete LLMProvider instance (defaults to OpenRouterProvider).
            model: Model identifier for the provider.
            max_tool_rounds: Maximum number of tool execution rounds before forced synthesis.
            system_prompt: Base system prompt (defaults to assistant.txt).
            tools: Canonical list of tool schemas (defaults to TOOLS).
        """
        if provider is not None:
            self.provider = provider
        else:
            from providers.openrouter_provider import OpenRouterProvider
            self.provider = OpenRouterProvider()
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else TOOLS

    def run(self, user_task: str) -> str:
        """Execute the agent loop for a user task.

        Args:
            user_task: The user instruction or question.

        Returns:
            The agent's final text response.
        """
        if not user_task or not user_task.strip():
            return "Task cannot be empty."

        system_instruction = (
            self.system_prompt
            if self.system_prompt is not None
            else load_prompt("assistant.txt")
        )

        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=system_instruction),
            LLMMessage(role="user", content=user_task.strip()),
        ]

        try:
            for round_number in range(self.max_tool_rounds):
                response: LLMResponse = self.provider.generate(
                    messages=messages,
                    model=self.model,
                    tools=self.tools,
                    tool_choice="auto",
                )

                if response is None:
                    return "Agent execution error: Provider returned None response."

                # Handle provider errors explicitly
                if response.error:
                    logger.error(f"Provider error on round {round_number + 1}: {response.error}")
                    return f"Agent execution error: {response.error}"

                # CASE 1: No tool call requested - model answered using internal intelligence
                if not response.tool_calls:
                    return response.content or "No response generated."

                # CASE 2: Tool calls requested - preserve assistant turn in history
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                # Execute each requested tool and record results
                for tool_call in response.tool_calls:
                    result = execute_tool(tool_call)

                    messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            content=json.dumps(result),
                        )
                    )

            # Final synthesis round if tool round limit was reached
            final_response = self.provider.generate(
                messages=messages,
                model=self.model,
                tools=None,
                tool_choice="none",
            )

            if final_response is None:
                return "Agent execution error: Provider returned None during final synthesis."

            if final_response.error:
                logger.error(f"Provider error during final synthesis: {final_response.error}")
                return f"Agent execution error: {final_response.error}"

            return final_response.content or "No response generated."

        except Exception as exc:
            logger.error(f"Unexpected error during agent execution: {str(exc)}")
            return f"Agent execution error: {str(exc)}"


def run_agent(
    user_task: str,
    model: str = MODEL1,
    client_instance: Any = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    system_prompt: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    tools: Optional[List[dict]] = None,
) -> str:
    """Public wrapper maintaining 100% backward compatibility for all existing callers.

    Supports both legacy client_instance injection and modern provider injection.
    """
    resolved_provider: LLMProvider

    if provider is not None:
        resolved_provider = provider
    elif client_instance is not None:
        if isinstance(client_instance, LLMProvider):
            resolved_provider = client_instance
        else:
            # Wrap legacy OpenAI client / mock in OpenRouterProvider
            from providers.openrouter_provider import OpenRouterProvider
            resolved_provider = OpenRouterProvider(client=client_instance)
    else:
        from providers.openrouter_provider import OpenRouterProvider
        resolved_provider = OpenRouterProvider()

    engine = AgentEngine(
        provider=resolved_provider,
        model=model,
        max_tool_rounds=max_tool_rounds,
        system_prompt=system_prompt,
        tools=tools,
    )

    return engine.run(user_task=user_task)
