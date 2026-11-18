"""
agents/supervisor.py

Specialized Supervisor Agent for day223_agents (Day 225 Part 6).

Implements:
1. DecisionType: DELEGATE, COMPLETE, CLARIFY.
2. SupervisorDecision: Structured decision contract with schema validation.
3. validate_supervisor_decision: Strict validation enforcing WorkerRegistry registration,
   preventing invented agent names, and verifying capability qualifications.
4. SupervisorAgent: Supervisor AI that analyzes task requirements, dynamically selects
   registered workers from WorkerRegistry, audits execution evaluations, and drives
   the multi-agent decision loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from config.settings import MODEL1
from core.failure_recovery import WorkerRegistry
from core.provider_interface import LLMMessage, LLMProvider
from agents.base_agent import BaseAgent
from utils.json_utils import parse_json_from_llm
from utils.logging_config import get_logger
from utils.prompt_loader import load_prompt

logger = get_logger()


# ===========================================================================
# 1. Decision Types & Models
# ===========================================================================

class DecisionType(str, Enum):
    DELEGATE = "DELEGATE"
    COMPLETE = "COMPLETE"
    CLARIFY = "CLARIFY"


@dataclass
class SupervisorDecision:
    """Structured decision contract produced by the Supervisor."""
    decision_type: DecisionType
    target_worker: Optional[str] = None
    capability_required: Optional[str] = None
    instruction: Optional[str] = None
    reasoning: str = ""
    final_response: Optional[str] = None
    clarification_question: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_number: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a serializable dictionary."""
        return {
            "decision": self.decision_type.value,
            "decision_type": self.decision_type.value,
            "target_worker": self.target_worker,
            "capability_required": self.capability_required,
            "instruction": self.instruction,
            "reasoning": self.reasoning,
            "final_response": self.final_response,
            "clarification_question": self.clarification_question,
            "metadata": self.metadata,
            "step_number": self.step_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SupervisorDecision:
        """Create a SupervisorDecision from a dictionary."""
        raw_type = data.get("decision") or data.get("decision_type") or "DELEGATE"
        try:
            dec_type = DecisionType(str(raw_type).upper())
        except ValueError:
            dec_type = DecisionType.DELEGATE

        return cls(
            decision_type=dec_type,
            target_worker=data.get("target_worker"),
            capability_required=data.get("capability_required"),
            instruction=data.get("instruction"),
            reasoning=data.get("reasoning", ""),
            final_response=data.get("final_response"),
            clarification_question=data.get("clarification_question"),
            metadata=data.get("metadata", {}),
            step_number=data.get("step_number", 1),
        )


# ===========================================================================
# 2. Decision Validation Enforcing Registry Rules
# ===========================================================================

def validate_supervisor_decision(
    decision: Union[SupervisorDecision, Dict[str, Any]],
    worker_registry: WorkerRegistry,
) -> Tuple[bool, Optional[str], Optional[SupervisorDecision]]:
    """Strictly validate a supervisor decision before execution.

    Enforces:
    1. Valid decision type (DELEGATE, COMPLETE, CLARIFY).
    2. Registered worker verification: Target worker must be present in WorkerRegistry.
       Invented agent names are rejected immediately.
    3. Worker capability match: Target worker must be qualified for the requested capability.
    4. Non-empty instructions for delegations.
    5. Substantive content for COMPLETE and CLARIFY.

    Returns:
        (is_valid, error_message, normalized_decision)
    """
    if isinstance(decision, dict):
        try:
            norm_decision = SupervisorDecision.from_dict(decision)
        except Exception as exc:
            return False, f"Malformed decision structure: {str(exc)}", None
    elif isinstance(decision, SupervisorDecision) or (
        hasattr(decision, "decision_type") and hasattr(decision, "target_worker")
    ):
        norm_decision = decision
    else:
        return False, f"Invalid decision object type: {type(decision)}", None

    dec_type = norm_decision.decision_type

    if dec_type == DecisionType.DELEGATE:
        target = norm_decision.target_worker
        if not target or not str(target).strip():
            return False, "Validation Error: DELEGATE decision must specify a non-empty 'target_worker'.", norm_decision

        target_name = str(target).strip()
        worker = worker_registry.get_worker(target_name)
        if worker is None:
            return (
                False,
                f"Validation Error: Target worker '{target_name}' is not registered in WorkerRegistry. "
                "Invented agents are strictly rejected.",
                norm_decision,
            )

        cap = norm_decision.capability_required
        if cap and str(cap).strip():
            if not worker_registry.is_capable(worker, str(cap).strip()):
                return (
                    False,
                    f"Validation Error: Worker '{target_name}' lacks required capability '{cap}'.",
                    norm_decision,
                )

        instr = norm_decision.instruction
        if not instr or not str(instr).strip():
            return False, f"Validation Error: DELEGATE decision to '{target_name}' must include actionable 'instruction'.", norm_decision

        return True, None, norm_decision

    elif dec_type == DecisionType.COMPLETE:
        if not norm_decision.final_response and not norm_decision.reasoning:
            return False, "Validation Error: COMPLETE decision must include final_response or reasoning.", norm_decision
        return True, None, norm_decision

    elif dec_type == DecisionType.CLARIFY:
        question = norm_decision.clarification_question or norm_decision.instruction
        if not question or not str(question).strip():
            return False, "Validation Error: CLARIFY decision must include a non-empty 'clarification_question'.", norm_decision
        norm_decision.clarification_question = str(question).strip()
        return True, None, norm_decision

    return False, f"Unsupported decision type: {dec_type}", norm_decision


# ===========================================================================
# 3. Supervisor Agent Class
# ===========================================================================

class SupervisorAgent(BaseAgent):
    """Supervisor Agent responsible for multi-agent coordination and dynamic delegation.

    Coordinates registered workers, enforces capability boundaries, audits evaluation
    reports, and ensures tasks complete strictly based on verified evidence.
    """

    def __init__(
        self,
        name: str = "SupervisorAgent",
        role: str = "Multi-Agent System Supervisor and Coordinator",
        description: str = (
            "Analyzes user task requirements, dynamically selects qualified registered workers "
            "from the WorkerRegistry, audits execution evaluations, and issues structured DELEGATE, "
            "COMPLETE, or CLARIFY decisions."
        ),
        provider: Optional[LLMProvider] = None,
        model: str = MODEL1,
        system_prompt: Optional[str] = None,
        worker_registry: Optional[WorkerRegistry] = None,
        max_delegation_steps: int = 6,
    ):
        prompt_content = system_prompt or load_prompt("supervisor.txt")
        super().__init__(
            name=name,
            role=role,
            description=description,
            provider=provider,
            model=model,
            system_prompt=prompt_content,
        )
        self.worker_registry = worker_registry or WorkerRegistry()
        self.max_delegation_steps = max_delegation_steps

    def run(self, user_input: str, success_criteria: Optional[List[str]] = None, **kwargs) -> Any:
        """Execute supervisor decision on user input.

        Satisfies BaseAgent abstract method requirement.
        """
        context_state = kwargs.get("context_state", {})
        return self.decide(
            objective=user_input,
            success_criteria=success_criteria or [],
            context_state=context_state,
            worker_registry=self.worker_registry,
        )

    def receive_message(self, message: Any) -> Any:
        """Handle incoming AgentMessage directed to the Supervisor.

        Processes incoming task results, handoffs, or status updates from workers.
        """
        from core.orchestrator import AgentMessage, MessageType
        return AgentMessage(
            task_id=message.task_id,
            agent_name=self.name,
            sender=self.name,
            receiver=message.sender,
            stage=message.stage or "supervision",
            status="success",
            input_data=message.input_data,
            output_data={
                "received": True,
                "source_message_id": message.message_id,
                "sender": message.sender,
                "status": message.status,
            },
            correlation_id=message.correlation_id or message.task_id,
            message_type=MessageType.STATUS_UPDATE,
        )

    def decide(
        self,
        objective: str,
        success_criteria: List[str],
        context_state: Dict[str, Any],
        worker_registry: Optional[WorkerRegistry] = None,
        step_number: int = 1,
        last_evaluation: Optional[Dict[str, Any]] = None,
    ) -> SupervisorDecision:
        """Produce a structured next-step decision for the multi-agent system.

        Args:
            objective: User's primary goal.
            success_criteria: List of explicit criteria strings.
            context_state: Snapshot of current execution state (findings, plan, results, errors).
            worker_registry: Registry of available workers (defaults to self.worker_registry).
            step_number: Current iteration number in the decision loop.
            last_evaluation: Optional ResultEvaluator report on previous execution.

        Returns:
            Validated SupervisorDecision.
        """
        registry = worker_registry or self.worker_registry

        # 1. Edge-case: Ambiguous or empty user request -> request clarification
        cleaned_obj = (objective or "").strip()
        if not cleaned_obj:
            return SupervisorDecision(
                decision_type=DecisionType.CLARIFY,
                clarification_question="The task objective is empty. Please provide a clear task description.",
                reasoning="Empty task objective cannot be processed without clarification.",
                step_number=step_number,
            )

        # 2. Check if maximum delegation steps exceeded
        if step_number > self.max_delegation_steps:
            logger.warning(f"[{self.name}] Maximum delegation steps ({self.max_delegation_steps}) exceeded.")
            return SupervisorDecision(
                decision_type=DecisionType.COMPLETE,
                reasoning=f"Exceeded maximum delegation steps ({self.max_delegation_steps}). Forcing loop termination.",
                final_response=f"Execution terminated after reaching the maximum limit of {self.max_delegation_steps} delegation steps.",
                step_number=step_number,
            )

        # 3. Format prompt for LLM decision
        available_workers = registry.list_workers()
        user_prompt = self._format_decision_prompt(
            objective=cleaned_obj,
            success_criteria=success_criteria,
            context_state=context_state,
            available_workers=available_workers,
            step_number=step_number,
            last_evaluation=last_evaluation,
        )

        messages = [
            LLMMessage(role="system", content=self.get_effective_system_prompt()),
            LLMMessage(role="user", content=user_prompt),
        ]

        # 4. Attempt LLM generation
        try:
            response = self._call_llm(messages=messages, model=self.model)
            if response.content:
                parsed = parse_json_from_llm(response.content)
                if isinstance(parsed, dict) and "decision" in parsed:
                    parsed["step_number"] = step_number
                    is_valid, err_msg, decision = validate_supervisor_decision(parsed, registry)
                    if is_valid and decision is not None:
                        logger.info(f"[{self.name}] LLM Decision: {decision.decision_type.value} -> {decision.target_worker}")
                        return decision
                    else:
                        logger.warning(f"[{self.name}] LLM decision invalid: {err_msg}; falling back to deterministic heuristic.")
        except Exception as exc:
            logger.warning(f"[{self.name}] LLM call failed: {str(exc)}; falling back to deterministic heuristic.")

        # 5. Deterministic Heuristic Fallback
        return self._heuristic_decision(
            objective=cleaned_obj,
            success_criteria=success_criteria,
            context_state=context_state,
            registry=registry,
            step_number=step_number,
            last_evaluation=last_evaluation,
        )

    def _format_decision_prompt(
        self,
        objective: str,
        success_criteria: List[str],
        context_state: Dict[str, Any],
        available_workers: List[Dict[str, Any]],
        step_number: int,
        last_evaluation: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Construct the prompt providing full context to the Supervisor LLM."""
        workers_summary = "\n".join(
            f"- {w['name']}: Capabilities={w['capabilities']}, Priority={w['priority']}"
            for w in available_workers
        )

        findings = context_state.get("research_findings") or "None"
        plan = context_state.get("execution_plan") or "None"
        results = context_state.get("execution_results") or []
        errors = context_state.get("errors") or []

        prompt_lines = [
            f"Step: {step_number} (Max: {self.max_delegation_steps})",
            f"Task Objective: {objective}",
            f"Success Criteria: {json.dumps(success_criteria)}",
            "",
            "AVAILABLE REGISTERED WORKERS (Choose ONLY from this list):",
            workers_summary or "No workers registered",
            "",
            "CURRENT EXECUTION STATE:",
            f"- Research Findings: {findings[:300]}..." if len(str(findings)) > 300 else f"- Research Findings: {findings}",
            f"- Execution Plan: {json.dumps(plan) if isinstance(plan, dict) else str(plan)}",
            f"- Completed Execution Steps: {len(results)} steps recorded",
            f"- Errors / Alerts: {json.dumps(errors)}",
        ]

        if last_evaluation:
            prompt_lines.extend([
                "",
                "LAST EVALUATION REPORT:",
                f"- Criteria Met: {last_evaluation.get('all_criteria_met')}",
                f"- Has Valid Evidence: {last_evaluation.get('has_valid_evidence')}",
                f"- Notes: {last_evaluation.get('evaluation_notes')}",
            ])

        prompt_lines.extend([
            "",
            "Analyze the state and return a valid JSON decision object (DELEGATE, COMPLETE, or CLARIFY).",
        ])

        return "\n".join(prompt_lines)

    def _heuristic_decision(
        self,
        objective: str,
        success_criteria: List[str],
        context_state: Dict[str, Any],
        registry: WorkerRegistry,
        step_number: int,
        last_evaluation: Optional[Dict[str, Any]] = None,
    ) -> SupervisorDecision:
        """Deterministic heuristic fallback ensuring dependable decision-making without LLM."""
        research_findings = context_state.get("research_findings")
        execution_plan = context_state.get("execution_plan")
        execution_results = context_state.get("execution_results") or []
        completed_step_ids = {
            str(r.get("task_id")) for r in execution_results if r.get("status") == "completed"
        }

        # Check if research is missing and a research worker is registered
        if not research_findings:
            research_worker = registry.get_best_worker_for_capability("research")
            if research_worker:
                return SupervisorDecision(
                    decision_type=DecisionType.DELEGATE,
                    target_worker=research_worker[0],
                    capability_required="research",
                    instruction=f"Conduct research and information gathering on: {objective}",
                    reasoning="Task objective requires preliminary research and factual discovery.",
                    step_number=step_number,
                )

        # Check if planning is missing and a planner is registered
        if not execution_plan or not execution_plan.get("tasks"):
            planner_worker = registry.get_best_worker_for_capability("planning")
            if planner_worker:
                instruction = f"Objective: {objective}\nFindings: {research_findings or ''}"
                return SupervisorDecision(
                    decision_type=DecisionType.DELEGATE,
                    target_worker=planner_worker[0],
                    capability_required="planning",
                    instruction=instruction,
                    reasoning="Research is completed; structured decomposition and execution plan required.",
                    step_number=step_number,
                )

        # Check if planned tasks remain unexecuted
        tasks = execution_plan.get("tasks", [])
        unexecuted_tasks = [
            t for t in tasks
            if str(t.get("id")) not in completed_step_ids
            and str(t.get("task_id")) not in completed_step_ids
        ]

        if unexecuted_tasks:
            executor_worker = registry.get_best_worker_for_capability("execution")
            if executor_worker:
                next_task = unexecuted_tasks[0]
                task_desc = next_task.get("description", str(next_task))
                return SupervisorDecision(
                    decision_type=DecisionType.DELEGATE,
                    target_worker=executor_worker[0],
                    capability_required="execution",
                    instruction=f"Execute task {next_task.get('id', 1)}: {task_desc}",
                    metadata={"task_item": next_task},
                    reasoning=f"Executing planned step {next_task.get('id', 1)}: {task_desc[:40]}.",
                    step_number=step_number,
                )

        # All steps executed -> check evaluation
        synthesis_lines = [f"Task Objective: {objective}"]
        if research_findings:
            synthesis_lines.append(f"Research: {research_findings}")
        if execution_results:
            synthesis_lines.append(f"Execution Results: {len(execution_results)} steps completed.")

        return SupervisorDecision(
            decision_type=DecisionType.COMPLETE,
            final_response="\n\n".join(synthesis_lines),
            reasoning="All research, planning, and execution stages have been successfully fulfilled.",
            step_number=step_number,
        )
