"""
agents/executor.py

Specialized Executor Agent for executing approved tasks using authorized tools.
Adheres to the specialized BaseAgent architecture while preserving 100%
backward compatibility for existing functional callers.
"""

import re
from typing import Any, Dict, List, Optional, Union

from config.settings import MODEL1
from core.provider_interface import LLMProvider
from core.agent_engine import run_agent
from agents.base_agent import BaseAgent
from tools.schemas import TOOLS
from utils.prompt_loader import load_prompt
from utils.logging_config import get_logger

logger = get_logger()

# Sensitive operation patterns enforced at the tool execution boundary
SENSITIVE_PATTERNS = [
    r"\b(delete|remove|drop|truncate|destroy|purge|erase|wipe)\b",
    r"\b(modify\s+config|change\s+configuration|overwrite\s+settings)\b",
    r"\b(send\s+email|send\s+message|send\s+notification|post\s+tweet)\b",
    r"\b(transfer\s+funds?|process\s+payment|billing|credit\s+card)\b",
    r"\b(chmod|chown|grant\s+permission|revoke\s+permission|sudo)\b",
]


def is_sensitive_operation(text: str) -> bool:
    """Check if a task description indicates a sensitive or destructive operation."""
    if not text or not isinstance(text, str):
        return False
    return any(re.search(pat, text, re.IGNORECASE) for pat in SENSITIVE_PATTERNS)


class ExecutorAgent(BaseAgent):
    """Specialized agent for executing approved tasks using authorized tools.

    Primary Responsibility:
    - Receive approved tasks or structured task items.
    - Select appropriate registered tools from the canonical registry.
    - Execute tasks safely through the AgentEngine without bypassing tool boundaries.
    - Handle tool errors gracefully and report execution status.
    - Respect authorization and safety restrictions (no arbitrary commands).
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: str = MODEL1,
        system_prompt: Optional[str] = None,
        client_instance: Any = None,
        tools: Optional[List[dict]] = None,
        approval_manager: Optional[Any] = None,
        risk_engine: Optional[Any] = None,
        audit_trail: Optional[Any] = None,
    ):
        """Initialize the ExecutorAgent with specialized execution instructions."""
        resolved_prompt = system_prompt or load_prompt("executor.txt", fallback="")
        if not resolved_prompt:
            resolved_prompt = load_prompt("assistant.txt")

        # Canonical tools from tools/schemas.py (calculator, datetime, word_counter)
        authorized_tools = tools if tools is not None else TOOLS

        super().__init__(
            name="ExecutorAgent",
            role="Executing approved tasks and using authorized tools",
            description=(
                "Executes approved tasks using registered, authorized tools "
                "through the agent engine, handling errors and reporting factual results."
            ),
            provider=provider,
            model=model,
            system_prompt=resolved_prompt,
            client_instance=client_instance,
            tools=authorized_tools,
        )
        self.approval_manager = approval_manager
        self.risk_engine = risk_engine
        self.audit_trail = audit_trail

    def execute(
        self,
        task: Union[str, Dict[str, Any]],
        is_authorized: bool = False,
        expected_task_id: Optional[Union[str, int]] = None,
        approval_token: Optional[str] = None,
        approval_manager: Optional[Any] = None,
        risk_engine: Optional[Any] = None,
        audit_trail: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute an approved task and return a structured execution report.

        Args:
            task: Either a string task description or a dictionary with
                  'description' (and optional 'id', 'task_id', 'authorized', 'approval_token').
            is_authorized: Explicit authorization flag for sensitive tasks.
            expected_task_id: Optional identifier for tracking consistent task handoffs.
            approval_token: Optional cryptographic approval token issued by HumanApprovalManager.
            approval_manager: Optional HumanApprovalManager instance for boundary verification.
            risk_engine: Optional RiskAssessmentEngine instance for pre-execution risk classification.
            audit_trail: Optional AuditTrail instance for audit event logging.

        Returns:
            Dictionary with 'task_id', 'task', 'status', 'result', and 'error'.
        """
        task_id: Optional[Union[str, int]] = expected_task_id
        task_description: str = ""
        task_authorized: bool = is_authorized
        token_to_verify: Optional[str] = approval_token
        active_approval_manager = approval_manager or getattr(self, "approval_manager", None)
        active_risk_engine = risk_engine or getattr(self, "risk_engine", None)
        active_audit_trail = audit_trail or getattr(self, "audit_trail", None)

        if isinstance(task, dict):
            task_id = task.get("task_id") if task.get("task_id") is not None else task.get("id")
            if task_id is None:
                task_id = expected_task_id
            raw_desc = task.get("description", "")
            task_description = str(raw_desc).strip() if raw_desc else ""
            if task.get("authorized") is True:
                task_authorized = True
            if not token_to_verify:
                token_to_verify = task.get("approval_token") or task.get("token")
        elif isinstance(task, str):
            task_description = task.strip()
        else:
            return {
                "task_id": task_id,
                "task": "",
                "status": "failed",
                "result": "",
                "error": "Task must be a string or dictionary.",
            }

        if not task_description:
            return {
                "task_id": task_id,
                "task": "",
                "status": "failed",
                "result": "",
                "error": "Task description cannot be empty.",
            }

        # Step 5: Authorization & Risk enforcement at the tool-execution boundary
        requires_approval_check = is_sensitive_operation(task_description) or (
            isinstance(task, dict) and task.get("requires_approval") is True
        )

        # Risk assessment evaluation before execution
        if active_risk_engine is not None:
            target_task_str = str(task_id) if task_id is not None else ""
            risk_res = active_risk_engine.evaluate_action(
                task_id=target_task_str,
                action=task_description,
                explicit_requires_approval=requires_approval_check,
            )
            if risk_res.approval_required:
                requires_approval_check = True

        if active_approval_manager is not None and requires_approval_check:
            # Human approval is strictly required - bypass via is_authorized is rejected
            target_task_str = str(task_id) if task_id is not None else ""
            is_valid, err_msg = active_approval_manager.is_approved(
                task_id=target_task_str,
                action=task_description,
                token=token_to_verify,
            )
            if not is_valid:
                logger.warning(
                    f"[{self.name}] Human approval boundary check failed for task '{task_id}': {err_msg}"
                )
                fail_res = {
                    "task_id": task_id,
                    "task": task_description,
                    "status": "failed",
                    "result": f"Execution rejected at boundary: {err_msg}",
                    "error": f"Human approval boundary check failed: {err_msg}",
                }
                if active_audit_trail is not None and hasattr(active_audit_trail, "record_event"):
                    from core.hitl import AuditEventType
                    active_audit_trail.record_event(
                        task_id=target_task_str,
                        event_type=AuditEventType.APPROVAL_REJECTION,
                        description=f"Execution blocked at boundary: {err_msg}",
                        agent_name=self.name,
                        payload=fail_res,
                    )
                return fail_res
        elif is_sensitive_operation(task_description) and not task_authorized:
            logger.warning(
                f"[{self.name}] Sensitive operation blocked at tool boundary: {task_description}"
            )
            return {
                "task_id": task_id,
                "task": task_description,
                "status": "failed",
                "result": "Unauthorized sensitive operation.",
                "error": "Unauthorized sensitive operation: sensitive actions require explicit authorization at the tool boundary.",
            }

        logger.info(f"[{self.name}] Executing task: {task_description[:80]}...")

        try:
            result_str = execute_task(
                task=task_description,
                model=self.model,
                client_instance=None,
                system_prompt=self.get_effective_system_prompt(),
                provider=self.provider,
                tools=self.tools,
            )

            is_error = result_str.startswith("Agent execution error:")
            exec_res = {
                "task_id": task_id,
                "task": task_description,
                "status": "failed" if is_error else "completed",
                "result": result_str,
                "error": result_str if is_error else None,
            }

            if active_audit_trail is not None and hasattr(active_audit_trail, "record_event"):
                from core.hitl import AuditEventType
                active_audit_trail.record_event(
                    task_id=str(task_id) if task_id is not None else "unknown",
                    event_type=AuditEventType.EXECUTION_ATTEMPT,
                    description=f"Execution {exec_res['status']}: {task_description[:50]}",
                    agent_name=self.name,
                    payload={"task_id": task_id, "status": exec_res["status"]},
                )

            return exec_res

        except Exception as exc:
            logger.error(f"[{self.name}] Execution error: {str(exc)}")
            err_res = {
                "task_id": task_id,
                "task": task_description,
                "status": "failed",
                "result": f"Execution failed: {str(exc)}",
                "error": str(exc),
            }
            if active_audit_trail is not None and hasattr(active_audit_trail, "record_event"):
                from core.hitl import AuditEventType
                active_audit_trail.record_event(
                    task_id=str(task_id) if task_id is not None else "unknown",
                    event_type=AuditEventType.ERROR_ALERT,
                    description=f"Execution exception: {str(exc)[:60]}",
                    agent_name=self.name,
                    payload=err_res,
                )
            return err_res

    def run(self, user_input: str) -> str:
        """Execute the task and return the text result."""
        report = self.execute(user_input)
        return report.get("result", "")

    def receive_message(self, message: Any) -> Any:
        """Handle incoming AgentMessage for task execution.

        Executes the authorized task using registered tools and returns structured execution results.
        """
        from core.orchestrator import AgentMessage, MessageType
        task_input = message.input_data
        approval_token = None
        if hasattr(message, "metadata") and isinstance(message.metadata, dict):
            approval_token = message.metadata.get("approval_token") or message.metadata.get("token")
        if not approval_token and isinstance(task_input, dict):
            approval_token = task_input.get("approval_token") or task_input.get("token")

        exec_report = self.execute(
            task_input,
            is_authorized=True,
            expected_task_id=message.task_id,
            approval_token=approval_token,
        )
        is_success = exec_report.get("status") == "completed"

        return AgentMessage(
            task_id=message.task_id,
            agent_name=self.name,
            sender=self.name,
            receiver=message.sender,
            stage="execution",
            status="success" if is_success else "failed",
            input_data=message.input_data,
            output_data=exec_report,
            error=exec_report.get("error"),
            correlation_id=message.correlation_id or message.task_id,
            message_type=MessageType.TASK_RESULT,
        )


def execute_task(
    task: str,
    model: str = MODEL1,
    client_instance: Any = None,
    system_prompt: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    tools: Optional[List[dict]] = None,
) -> str:
    """Execute a single planned task using the tool-enabled agent engine.

    Preserves both internal model intelligence and external tool execution.
    Does NOT bypass the tool-enabled agent engine.
    Maintains 100% backward compatibility for all existing callers.
    """
    if system_prompt is None:
        system_prompt = load_prompt("executor.txt", fallback="")
        if not system_prompt:
            system_prompt = load_prompt("assistant.txt")

    return run_agent(
        user_task=task,
        model=model,
        client_instance=client_instance,
        system_prompt=system_prompt,
        provider=provider,
        tools=tools,
    )


# Backward compatibility alias
execute_research_task = execute_task
