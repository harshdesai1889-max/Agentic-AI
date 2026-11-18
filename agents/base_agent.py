"""
agents/base_agent.py

Abstract base class establishing a consistent architecture for all specialized agents:
- Common identity (name, role, description)
- Consistent initialization and provider resolution
- Unified access to LLMProvider and system prompts
- Standardized error handling and logging
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from config.settings import MODEL1
from core.provider_interface import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
)
from providers.openrouter_provider import OpenRouterProvider
from utils.prompt_loader import load_prompt
from utils.logging_config import get_logger

logger = get_logger()


class BaseAgent(ABC):
    """Abstract base class establishing a consistent agent architecture.

    Every specialized agent has:
    1. A clearly defined identity (name, role, description).
    2. A specialized system instruction set.
    3. An injected LLMProvider (or legacy client adapter).
    4. A designated default model.
    5. A uniform execution contract via run().
    """

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        provider: Optional[LLMProvider] = None,
        model: str = MODEL1,
        system_prompt: Optional[str] = None,
        client_instance: Any = None,
        tools: Optional[List[dict]] = None,
    ):
        """Initialize the BaseAgent.

        Args:
            name: Human-readable agent identifier (e.g. 'PlannerAgent').
            role: Primary responsibility summary.
            description: Detailed capability description.
            provider: Concrete LLMProvider instance.
            model: Model identifier for this agent.
            system_prompt: Specialized system prompt string.
            client_instance: Optional legacy OpenAI client or mock.
            tools: Optional list of authorized tool schemas.
        """
        self.name = name
        self.role = role
        self.description = description
        self.model = model
        self.system_prompt = system_prompt or ""
        self.tools = tools
        self.loaded_skills: List[Any] = []

        # Resolve provider consistently without duplicating resolution logic
        if provider is not None:
            self.provider = provider
        elif client_instance is not None:
            if isinstance(client_instance, LLMProvider):
                self.provider = client_instance
            else:
                self.provider = OpenRouterProvider(client=client_instance)
        else:
            self.provider = OpenRouterProvider()

    def get_identity(self) -> Dict[str, str]:
        """Return the agent's identity metadata."""
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "model": self.model,
        }

    def load_skill(self, skill: Union[str, Any]) -> bool:
        """Attach a reusable skill to this agent's contextual execution."""
        from core.skills_registry import SkillRegistry, Skill
        if isinstance(skill, str):
            target_skill = SkillRegistry.get_instance().get_skill(skill)
            if not target_skill:
                logger.warning(f"[{self.name}] Skill '{skill}' not found in registry.")
                return False
        elif hasattr(skill, "instructions"):
            target_skill = skill
        else:
            return False

        if not hasattr(self, "loaded_skills") or self.loaded_skills is None:
            self.loaded_skills = []
        if target_skill not in self.loaded_skills:
            self.loaded_skills.append(target_skill)
        logger.info(f"[{self.name}] Attached skill '{target_skill.name}'.")
        return True

    def get_loaded_skills(self) -> List[Any]:
        """Return the list of currently attached skills."""
        return list(getattr(self, "loaded_skills", []))

    def get_effective_system_prompt(self) -> str:
        """Compose the agent's base system prompt with all currently loaded skills."""
        base = self.system_prompt or ""
        skills = self.get_loaded_skills()
        if not skills:
            return base

        skill_blocks = []
        for s in skills:
            name = getattr(s, "name", "Skill")
            instructions = getattr(s, "instructions", "")
            skill_blocks.append(f"### Skill Guidance: {name}\n{instructions}")

        return f"{base}\n\n## Attached Professional Skills\n\n" + "\n\n".join(skill_blocks)

    def execute_with_skill_lifecycle(
        self,
        task_input: str,
        task_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> Any:
        """Execute a task following the complete skill discovery & application lifecycle:
        1. Agent receives task.
        2. Skill discovery runs.
        3. Relevant skill is selected.
        4. Skill instructions are loaded into the agent.
        5. Effective system prompt incorporates skill instructions.
        6. Agent executes.
        7. Applied skills are recorded in AgentExecutionResult.
        """
        import uuid
        from core.skills_registry import SkillRegistry
        from core.agent_contracts import AgentExecutionResult

        tid = task_id or f"task_{uuid.uuid4().hex[:8]}"
        wfid = workflow_id or f"wf_{uuid.uuid4().hex[:8]}"

        # 1-3. Discover & select relevant skill
        registry = SkillRegistry.get_instance()
        discovered = registry.discover_skills_for_task(
            task_description=task_input,
            agent_role=self.name,
            max_results=2,
        )

        # 4. Load skill instructions
        for s in discovered:
            self.load_skill(s)

        # 5. Effective system prompt now incorporates skill instructions
        # 6. Execute agent
        applied_skills = [getattr(s, "name", str(s)) for s in self.get_loaded_skills()]
        try:
            output = self.run(task_input)
            is_success = bool(output) and not (isinstance(output, dict) and output.get("error"))
            return AgentExecutionResult(
                task_id=tid,
                workflow_id=wfid,
                agent_name=self.name,
                status="completed" if is_success else "failed",
                result=output,
                skills_applied=applied_skills,
                confidence=0.95 if is_success else 0.0,
            )
        except Exception as exc:
            return AgentExecutionResult(
                task_id=tid,
                workflow_id=wfid,
                agent_name=self.name,
                status="failed",
                result=None,
                errors=str(exc),
                skills_applied=applied_skills,
                confidence=0.0,
            )

    def _call_llm(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Call the resolved LLMProvider with error logging."""
        target_model = model or self.model
        try:
            response = self.provider.generate(
                messages=messages,
                model=target_model,
                tools=tools,
                tool_choice=tool_choice,
            )
            if response is None:
                return LLMResponse(error="Provider returned None response.")
            return response
        except Exception as exc:
            logger.error(f"[{self.name}] LLM invocation failed: {str(exc)}")
            return LLMResponse(error=f"LLM invocation failed: {str(exc)}")

    @abstractmethod
    def run(self, user_input: str) -> Any:
        """Execute the agent on a user input.

        Must be implemented by every specialized agent subclass.
        """
        raise NotImplementedError

    def receive_message(self, message: Any) -> Any:
        """Process an incoming AgentMessage and produce a response AgentMessage.

        Default base implementation runs the agent on message.input_data and wraps
        the result in a responsive AgentMessage.
        """
        from core.orchestrator import AgentMessage, MessageType
        user_input = ""
        if isinstance(message.input_data, str):
            user_input = message.input_data
        elif isinstance(message.input_data, dict):
            user_input = message.input_data.get("query") or message.input_data.get("instruction") or str(message.input_data)
        else:
            user_input = str(message.input_data)

        try:
            result = self.run(user_input)
            return AgentMessage(
                task_id=message.task_id,
                agent_name=self.name,
                sender=self.name,
                receiver=message.sender,
                stage=message.stage or "execution",
                status="success",
                input_data=message.input_data,
                output_data=result,
                correlation_id=message.correlation_id or message.task_id,
                message_type=MessageType.TASK_RESULT,
            )
        except Exception as exc:
            return AgentMessage(
                task_id=message.task_id,
                agent_name=self.name,
                sender=self.name,
                receiver=message.sender,
                stage=message.stage or "execution",
                status="failed",
                input_data=message.input_data,
                output_data=None,
                error=str(exc),
                correlation_id=message.correlation_id or message.task_id,
                message_type=MessageType.ERROR_ALERT,
            )
