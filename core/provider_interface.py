from abc import ABC, abstractmethod

from dataclasses import dataclass, field

from typing import Any, Optional


# ==========================================
# 1. NORMALIZED TOOL CALL
# ==========================================

@dataclass
class ToolCallRequest:

    id: str

    name: str

    arguments: dict[str, Any] = field(
        default_factory=dict
    )

    thought_signature: Optional[Any] = None


# ==========================================
# 2. NORMALIZED MESSAGE
# ==========================================

@dataclass
class LLMMessage:

    role: str

    content: Optional[str] = None

    tool_calls: list[ToolCallRequest] = field(
        default_factory=list
    )

    tool_call_id: Optional[str] = None

    tool_name: Optional[str] = None


# ==========================================
# 3. NORMALIZED MODEL RESPONSE
# ==========================================

@dataclass
class LLMResponse:

    content: Optional[str] = None

    tool_calls: list[ToolCallRequest] = field(
        default_factory=list
    )

    usage: Optional[dict[str, Any]] = None

    raw_response: Any = None

    error: Optional[str] = None


# ==========================================
# 4. ABSTRACT PROVIDER INTERFACE
# ==========================================

class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        model: str,
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto"
    ) -> LLMResponse:

        """
        Generate a response using the selected LLM.

        Every provider must implement this method.
        """

        raise NotImplementedError