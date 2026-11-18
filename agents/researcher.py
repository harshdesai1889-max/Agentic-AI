"""
agents/researcher.py

Specialized Researcher Agent for information gathering, verification, and analysis.
Adheres to the specialized BaseAgent architecture while preserving 100%
backward compatibility for existing multi-step workflows.
"""

import json
from typing import Any, Dict, List, Optional

from config.settings import MODEL1, MODEL2
from core.provider_interface import LLMProvider
from core.llm import client as default_client
from core.agent_engine import run_agent
from agents.base_agent import BaseAgent
from agents.planner import create_plan
from agents.executor import execute_task
from tools.schemas import TOOLS
from utils.prompt_loader import load_prompt
from utils.logging_config import get_logger

logger = get_logger()


class ResearcherAgent(BaseAgent):
    """Specialized agent for research, information gathering, and analysis.

    Primary Responsibility:
    - Process research-oriented requests and extract core information requirements.
    - Gather facts and perform calculations using authorized tools.
    - Organize findings into a clear, logical structure.
    - Rigorously distinguish verified information from assumptions.
    - Identify data gaps, uncertainties, and research limitations.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: str = MODEL1,
        system_prompt: Optional[str] = None,
        client_instance: Any = None,
        tools: Optional[List[dict]] = None,
    ):
        """Initialize the ResearcherAgent with specialized research instructions."""
        resolved_prompt = system_prompt or load_prompt("researcher.txt", fallback="")
        if not resolved_prompt:
            resolved_prompt = load_prompt("assistant.txt")

        # Authorized tools for factual verification (calculator, datetime, word_counter)
        authorized_tools = tools if tools is not None else TOOLS

        super().__init__(
            name="ResearcherAgent",
            role="Research, information gathering, and analysis",
            description=(
                "Gathers and analyzes information using internal knowledge and "
                "authorized tools, distinguishing verified facts from assumptions "
                "and clearly noting research limitations."
            ),
            provider=provider,
            model=model,
            system_prompt=resolved_prompt,
            client_instance=client_instance,
            tools=authorized_tools,
        )

    def research(self, query: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Conduct research on a topic or query and return structured findings.

        Args:
            query: The research topic, question, or data gathering task.
            task_id: Optional identifier for tracking consistent task handoffs.

        Returns:
            Dictionary containing 'task_id', 'query', 'status', 'findings', and 'error'.
        """
        if not query or not query.strip():
            return {
                "task_id": task_id,
                "query": "",
                "status": "failed",
                "findings": "",
                "error": "Research query cannot be empty.",
            }

        cleaned_query = query.strip()
        logger.info(f"[{self.name}] Researching query (task_id={task_id}): {cleaned_query[:80]}...")

        try:
            findings = run_agent(
                user_task=cleaned_query,
                model=self.model,
                client_instance=None,
                system_prompt=self.get_effective_system_prompt(),
                provider=self.provider,
                tools=self.tools,
            )

            is_error = findings.startswith("Agent execution error:")

            return {
                "task_id": task_id,
                "query": cleaned_query,
                "status": "failed" if is_error else "completed",
                "findings": findings,
                "error": findings if is_error else None,
            }

        except Exception as exc:
            logger.error(f"[{self.name}] Research error: {str(exc)}")
            return {
                "task_id": task_id,
                "query": cleaned_query,
                "status": "failed",
                "findings": f"Research failed: {str(exc)}",
                "error": str(exc),
            }

    def run(self, user_input: str) -> str:
        """Execute the research inquiry and return the text findings."""
        report = self.research(user_input)
        return report.get("findings", "")

    def receive_message(self, message: Any) -> Any:
        """Handle incoming AgentMessage for research tasks.

        Preserves task_id and correlation_id while returning structured findings.
        """
        from core.orchestrator import AgentMessage, MessageType
        query = ""
        if isinstance(message.input_data, str):
            query = message.input_data
        elif isinstance(message.input_data, dict):
            query = (
                message.input_data.get("query")
                or message.input_data.get("instruction")
                or message.input_data.get("task")
                or str(message.input_data)
            )
        else:
            query = str(message.input_data)

        report = self.research(query, task_id=message.task_id)
        is_success = report.get("status") == "completed"

        return AgentMessage(
            task_id=message.task_id,
            agent_name=self.name,
            sender=self.name,
            receiver=message.sender,
            stage="research",
            status="success" if is_success else "failed",
            input_data=message.input_data,
            output_data=report,
            error=report.get("error"),
            correlation_id=message.correlation_id or message.task_id,
            message_type=MessageType.TASK_RESULT,
        )


def run_multi_step_agent(
    objective: str,
    client_instance: Any = None,
    verbose: bool = True,
) -> str:
    """Orchestrate the multi-step agent workflow:
    1. Plan tasks using MODEL2.
    2. Validate and execute each task using execute_task (MODEL1 + tools).
    3. Synthesize all findings with MODEL2.

    Maintains 100% backward compatibility for all existing callers.
    """
    active_client = client_instance or default_client

    if verbose:
        print("\n[PLANNER] MODEL2 creating execution plan...")

    plan = create_plan(objective, client_instance=active_client, model=MODEL2)
    tasks = plan.get("tasks", [])

    if not tasks:
        error_detail = plan.get("error", "No executable tasks were generated.")
        logger.warning(f"Planning failed: {error_detail}")
        return f"No executable tasks were generated: {error_detail}"

    results: List[Dict[str, Any]] = []

    if verbose:
        print(f"\n[EXECUTION STARTED] - {len(tasks)} tasks scheduled")

    for task in tasks:
        task_id = task.get("id")
        task_description = task.get("description", "")

        if verbose:
            print(f"\n[EXECUTING TASK {task_id}]")
            print(f"Description: {task_description}")

        try:
            result = execute_task(
                task=task_description,
                model=MODEL1,
                client_instance=active_client,
            )
        except Exception as exc:
            logger.error(f"Task {task_id} failed: {str(exc)}")
            result = f"Task execution failed: {str(exc)}"

        results.append({
            "task_id": task_id,
            "description": task_description,
            "result": result,
        })

        if verbose:
            print(f"[TASK {task_id} COMPLETED]")

    if verbose:
        print("\n[FINAL ANALYSIS] MODEL2 synthesizing results...")

    synthesizer_prompt = load_prompt("synthesizer.txt")

    try:
        final_response = active_client.chat.completions.create(
            model=MODEL2,
            messages=[
                {
                    "role": "system",
                    "content": synthesizer_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"Original objective:\n{objective}\n\n"
                        f"Task results:\n{json.dumps(results, indent=2)}"
                    ),
                },
            ],
        )
        return final_response.choices[0].message.content or "No response generated."
    except Exception as exc:
        logger.error(f"Synthesis failed: {str(exc)}")
        return f"Synthesis error: {str(exc)}"


# Backward compatibility alias
run_research_agent = run_multi_step_agent
