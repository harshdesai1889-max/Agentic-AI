"""
agents/planner.py

Specialized Planner Agent for task decomposition and execution strategy.
Adheres to the specialized BaseAgent architecture while preserving 100%
backward compatibility for existing functional callers.
"""

from typing import Any, Dict, List, Optional

from config.settings import MODEL2
from core.provider_interface import LLMProvider, LLMMessage
from agents.base_agent import BaseAgent
from utils.prompt_loader import load_prompt
from utils.json_utils import parse_json_safely
from utils.logging_config import get_logger

logger = get_logger()

MAX_ALLOWED_TASKS = 10


def validate_plan(data: Any, max_tasks: int = MAX_ALLOWED_TASKS) -> Dict[str, Any]:
    """Validate and sanitize the raw parsed plan data.

    Ensures:
    - Data is a dictionary with a 'tasks' list.
    - Each task has a valid ID and non-empty description.
    - Total task count does not exceed max_tasks.
    - Tasks are ordered sequentially.
    - Optional planning metadata (required_tools, dependencies) is validated.
    """
    if not isinstance(data, dict):
        return {"tasks": [], "error": "Plan must be a JSON object"}

    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list):
        return {"tasks": [], "error": "Plan must contain a 'tasks' list"}

    validated_tasks: List[Dict[str, Any]] = []

    for index, item in enumerate(raw_tasks, start=1):
        if not isinstance(item, dict):
            continue

        raw_desc = item.get("description")
        if not raw_desc or not isinstance(raw_desc, str) or not raw_desc.strip():
            continue

        raw_id = item.get("id")
        try:
            task_id = int(raw_id) if raw_id is not None else index
        except (ValueError, TypeError):
            task_id = index

        task_entry: Dict[str, Any] = {
            "id": task_id,
            "description": raw_desc.strip(),
        }

        # Optional structured planning attributes
        if "required_tools" in item and isinstance(item["required_tools"], list):
            task_entry["required_tools"] = [str(t) for t in item["required_tools"]]
        if "dependencies" in item and isinstance(item["dependencies"], list):
            task_entry["dependencies"] = [
                dep for dep in item["dependencies"] if isinstance(dep, int)
            ]

        validated_tasks.append(task_entry)

        if len(validated_tasks) >= max_tasks:
            logger.warning(f"Plan exceeded maximum task limit ({max_tasks}); truncated.")
            break

    result: Dict[str, Any] = {"tasks": validated_tasks}
    if "strategy_notes" in data and isinstance(data["strategy_notes"], str):
        result["strategy_notes"] = data["strategy_notes"]
    if "estimated_steps" in data and isinstance(data["estimated_steps"], int):
        result["estimated_steps"] = data["estimated_steps"]

    return result


class PlannerAgent(BaseAgent):
    """Specialized agent for task planning, decomposition, and execution strategy.

    Primary Responsibility:
    - Convert complex objectives into structured, ordered, actionable plans.
    - Identify required tools, resources, and dependencies.
    - Recognize potential constraints and execution risks.
    - The Planner plans tasks — it does NOT independently execute actions.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: str = MODEL2,
        system_prompt: Optional[str] = None,
        client_instance: Any = None,
        max_tasks: int = MAX_ALLOWED_TASKS,
    ):
        """Initialize the PlannerAgent with specialized instructions."""
        resolved_prompt = system_prompt or load_prompt("planner.txt")
        super().__init__(
            name="PlannerAgent",
            role="Planning, task decomposition, and execution strategy",
            description=(
                "Analyzes complex goals and decomposes them into structured, "
                "ordered, actionable execution plans with tool and dependency mapping."
            ),
            provider=provider,
            model=model,
            system_prompt=resolved_prompt,
            client_instance=client_instance,
            tools=None,  # Planner does not execute tools
        )
        self.max_tasks = max_tasks

    def plan(
        self,
        objective: str,
        max_tasks: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a structured, validated execution plan from an objective.

        Args:
            objective: The goal or request to decompose.
            max_tasks: Optional upper bound on generated tasks.
            task_id: Optional identifier for tracking consistent task handoffs.

        Returns:
            Dictionary with 'tasks' list on success, or empty tasks + 'error'.
        """
        if not objective or not objective.strip():
            return {"tasks": [], "error": "Objective cannot be empty", "task_id": task_id}

        limit = max_tasks if max_tasks is not None else self.max_tasks

        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=self.get_effective_system_prompt()),
            LLMMessage(role="user", content=objective.strip()),
        ]

        try:
            response = self._call_llm(
                messages=messages,
                model=self.model,
                tools=None,
                tool_choice="none",
            )

            if response.error:
                logger.error(f"Planner execution failed: {response.error}")
                return {"tasks": [], "error": f"Planner failed: {response.error}", "task_id": task_id}

            raw_content = response.content or ""
            success, parsed, err = parse_json_safely(raw_content)

            if not success:
                logger.error(f"Failed to parse planner JSON: {err}")
                return {"tasks": [], "error": f"Invalid planner JSON: {err}", "task_id": task_id}

            validated = validate_plan(parsed, max_tasks=limit)
            if task_id is not None:
                validated["task_id"] = task_id
                for item in validated.get("tasks", []):
                    item["task_id"] = f"{task_id}.{item.get('id')}"
            return validated

        except Exception as exc:
            logger.error(f"Planner execution failed: {str(exc)}")
            return {"tasks": [], "error": f"Planner failed: {str(exc)}", "task_id": task_id}

    def run(self, user_input: str) -> Dict[str, Any]:
        """Execute the planning workflow on the given objective."""
        return self.plan(user_input)

    def receive_message(self, message: Any) -> Any:
        """Handle incoming AgentMessage for task planning and decomposition.

        Accepts research findings or direct objectives and returns structured plan tasks.
        """
        from core.orchestrator import AgentMessage, MessageType
        objective = ""
        if isinstance(message.input_data, str):
            objective = message.input_data
        elif isinstance(message.input_data, dict):
            objective = (
                message.input_data.get("findings")
                or message.input_data.get("query")
                or message.input_data.get("objective")
                or message.input_data.get("instruction")
                or str(message.input_data)
            )
        else:
            objective = str(message.input_data)

        plan_report = self.plan(objective, task_id=message.task_id)
        is_success = bool(plan_report.get("tasks")) and not plan_report.get("error")

        return AgentMessage(
            task_id=message.task_id,
            agent_name=self.name,
            sender=self.name,
            receiver=message.sender,
            stage="planning",
            status="success" if is_success else "failed",
            input_data=message.input_data,
            output_data=plan_report,
            error=plan_report.get("error"),
            correlation_id=message.correlation_id or message.task_id,
            message_type=MessageType.TASK_RESULT,
        )


def create_plan(
    objective: str,
    client_instance: Any = None,
    model: str = MODEL2,
    max_tasks: int = MAX_ALLOWED_TASKS,
    provider: Optional[LLMProvider] = None,
) -> Dict[str, Any]:
    """Generate a structured, executable task plan from a complex objective.

    Maintains 100% backward compatibility for all existing callers.
    """
    agent = PlannerAgent(
        provider=provider,
        client_instance=client_instance,
        model=model,
        max_tasks=max_tasks,
    )
    return agent.plan(objective=objective, max_tasks=max_tasks)


# Backward compatibility alias
create_research_plan = create_plan
