"""
core/task_manager.py

Practical Task Assignment, Execution, and Verification System (Day 224 Part 6).

Implements:
1. TaskSpecification: Structured contract for incoming task assignments.
2. TaskStatus: Comprehensive lifecycle states.
3. Sensitive Operation Detection: Checkpoints for potentially destructive actions.
4. VerificationResult: Rigorous post-execution comparison against success criteria.
5. TaskManager: Full lifecycle manager coordinating Researcher, Planner, and Executor.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from config.settings import MODEL1, MODEL2
from core.provider_interface import LLMProvider, LLMMessage, LLMResponse
from utils.prompt_loader import load_prompt
from utils.logging_config import get_logger

logger = get_logger()

# Patterns identifying potentially sensitive or destructive operations
SENSITIVE_PATTERNS = [
    r"\b(delete|remove|drop|truncate|destroy|purge|erase|wipe)\b",
    r"\b(modify\s+config|change\s+configuration|overwrite\s+settings)\b",
    r"\b(send\s+email|send\s+message|send\s+notification|post\s+tweet)\b",
    r"\b(transfer\s+funds?|process\s+payment|billing|credit\s+card)\b",
    r"\b(chmod|chown|grant\s+permission|revoke\s+permission|sudo)\b",
]


# ===========================================================================
# 1. Task Status Enum
# ===========================================================================

class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    VALIDATED = "validated"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    REJECTED = "rejected"


# ===========================================================================
# 2. Task Specification Dataclass
# ===========================================================================

@dataclass
class TaskSpecification:
    """Structured contract defining a real-world task assignment.

    Contains all requirements, boundaries, assigned agents, and execution state.
    """
    objective: str
    success_criteria: List[str]
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    input_data: Any = None
    constraints: List[str] = field(default_factory=list)
    expected_output: str = ""
    status: TaskStatus = TaskStatus.SUBMITTED
    assigned_agents: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    requires_approval: bool = False
    is_sensitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to a serializable dictionary."""
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "input_data": self.input_data,
            "constraints": self.constraints,
            "expected_output": self.expected_output,
            "success_criteria": self.success_criteria,
            "status": self.status.value,
            "assigned_agents": self.assigned_agents,
            "results": self.results,
            "errors": self.errors,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "requires_approval": self.requires_approval,
            "is_sensitive": self.is_sensitive,
        }


# ===========================================================================
# 3. Verification Result Dataclass
# ===========================================================================

@dataclass
class VerificationResult:
    """Detailed audit of execution results evaluated against success criteria."""
    all_criteria_met: bool
    passed_criteria: List[str] = field(default_factory=list)
    failed_criteria: List[str] = field(default_factory=list)
    completed_steps: List[Dict[str, Any]] = field(default_factory=list)
    incomplete_steps: List[Dict[str, Any]] = field(default_factory=list)
    verification_notes: str = ""
    has_valid_evidence: bool = True
    missing_fields: List[str] = field(default_factory=list)
    is_eligible_for_completion: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_criteria_met": self.all_criteria_met,
            "passed_criteria": self.passed_criteria,
            "failed_criteria": self.failed_criteria,
            "completed_steps_count": len(self.completed_steps),
            "incomplete_steps_count": len(self.incomplete_steps),
            "verification_notes": self.verification_notes,
            "has_valid_evidence": self.has_valid_evidence,
            "missing_fields": self.missing_fields,
            "is_eligible_for_completion": self.is_eligible_for_completion,
        }


# ===========================================================================
# 4. Task Manager Class
# ===========================================================================

class TaskManager:
    """Manages practical task assignments, multi-agent execution, and verification.

    Responsibilities:
    - Task creation and TaskSpecification validation.
    - Sensitivity detection and approval gate governance.
    - Delegation to WorkflowOrchestrator for sequential multi-agent execution.
    - Deterministic verification of outcomes against success criteria.
    - Final completion status assignment and task report generation.
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        researcher: Optional[Any] = None,
        planner: Optional[Any] = None,
        executor: Optional[Any] = None,
        provider: Optional[LLMProvider] = None,
        approval_callback: Optional[Callable[..., bool]] = None,
        model_synthesis: str = MODEL2,
        max_retries: int = 2,
        backoff_factor: float = 0.0,
        worker_registry: Optional[Any] = None,
        supervisor: Optional[Any] = None,
    ):
        if provider is not None:
            self.provider = provider
        else:
            from providers.openrouter_provider import OpenRouterProvider
            self.provider = OpenRouterProvider()

        # Wire or instantiate WorkflowOrchestrator
        if orchestrator is not None:
            self.orchestrator = orchestrator
        else:
            from core.orchestrator import WorkflowOrchestrator
            self.orchestrator = WorkflowOrchestrator(
                researcher=researcher,
                planner=planner,
                executor=executor,
                provider=self.provider,
                model_synthesis=model_synthesis,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                worker_registry=worker_registry,
                supervisor=supervisor,
            )

        # Retain direct access to specialized agents for backward compatibility
        self.researcher = getattr(self.orchestrator, "researcher", researcher)
        self.planner = getattr(self.orchestrator, "planner", planner)
        self.executor = getattr(self.orchestrator, "executor", executor)
        self.supervisor = supervisor or getattr(self.orchestrator, "supervisor", None)
        if self.supervisor and self.orchestrator and not getattr(self.orchestrator, "supervisor", None):
            self.orchestrator.supervisor = self.supervisor

        self.approval_callback = approval_callback
        self.model_synthesis = model_synthesis

    # -----------------------------------------------------------------------
    # Sensitivity Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def detect_sensitive_operations(text: str) -> bool:
        """Scan text for keywords or patterns indicating sensitive operations."""
        if not text or not isinstance(text, str):
            return False
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate_specification(self, spec: TaskSpecification) -> Optional[str]:
        """Validate an incoming task specification.

        Returns an error string if invalid, or None if valid.
        """
        if not isinstance(spec, TaskSpecification):
            return "Assignment must be an instance of TaskSpecification."

        if not spec.objective or not isinstance(spec.objective, str) or not spec.objective.strip():
            return "Task objective cannot be empty."

        if not isinstance(spec.success_criteria, list) or len(spec.success_criteria) == 0:
            return "Task must specify at least one success criterion."

        for idx, criterion in enumerate(spec.success_criteria, start=1):
            if not criterion or not isinstance(criterion, str) or not criterion.strip():
                return f"Success criterion #{idx} cannot be empty."

        if not isinstance(spec.constraints, list):
            return "Constraints must be a list of strings."

        return None

    # -----------------------------------------------------------------------
    # Submission
    # -----------------------------------------------------------------------

    def submit_task(
        self,
        assignment: Union[TaskSpecification, Dict[str, Any]],
    ) -> TaskSpecification:
        """Accept a user assignment, validate it, and mark as validated or rejected."""
        if isinstance(assignment, dict):
            task_id = assignment.get("task_id") or f"task_{uuid.uuid4().hex[:8]}"
            spec = TaskSpecification(
                task_id=task_id,
                objective=assignment.get("objective", ""),
                input_data=assignment.get("input_data") or assignment.get("input"),
                constraints=assignment.get("constraints", []),
                expected_output=assignment.get("expected_output", ""),
                success_criteria=assignment.get("success_criteria", []),
                requires_approval=assignment.get("requires_approval", False),
            )
        elif isinstance(assignment, TaskSpecification):
            spec = assignment
        else:
            spec = TaskSpecification(
                objective="",
                success_criteria=[],
                status=TaskStatus.REJECTED,
                errors=["Invalid assignment format: expected TaskSpecification or dict."],
            )
            return spec

        error = self.validate_specification(spec)
        if error:
            spec.status = TaskStatus.REJECTED
            spec.errors.append(error)
            logger.warning(f"[{spec.task_id}] Assignment validation rejected: {error}")
            return spec

        # Check for inherent sensitivity in objective or constraints
        full_text = f"{spec.objective} {' '.join(spec.constraints)} {str(spec.input_data or '')}"
        if self.detect_sensitive_operations(full_text):
            spec.is_sensitive = True
            spec.requires_approval = True

        spec.status = TaskStatus.VALIDATED
        return spec

    # -----------------------------------------------------------------------
    # Result Verification & Evaluation
    # -----------------------------------------------------------------------

    def verify_results(self, spec: TaskSpecification) -> VerificationResult:
        """Deterministically verify completed execution steps against success criteria and evidence.

        Uses ResultEvaluator to execute the 6 core checks:
        1. Task completion
        2. Expected output fields
        3. Requirement / criteria satisfaction
        4. Substantive evidence validation
        5. Step failure detection
        6. Completion eligibility determination
        """
        from core.evaluator import ResultEvaluator

        evaluator = ResultEvaluator()
        execution_steps = spec.results.get("execution_steps", [])
        research_findings = spec.results.get("research", "")

        eval_res = evaluator.evaluate(
            task_id=spec.task_id,
            objective=spec.objective,
            success_criteria=spec.success_criteria,
            execution_steps=execution_steps,
            research_findings=research_findings,
            expected_output=spec.expected_output,
            errors=spec.errors,
        )

        return VerificationResult(
            all_criteria_met=eval_res.all_criteria_met,
            passed_criteria=eval_res.passed_criteria,
            failed_criteria=eval_res.failed_criteria,
            completed_steps=eval_res.completed_steps,
            incomplete_steps=eval_res.failed_steps,
            verification_notes=eval_res.evaluation_notes,
            has_valid_evidence=eval_res.has_valid_evidence,
            missing_fields=eval_res.missing_fields,
            is_eligible_for_completion=eval_res.is_eligible_for_completion,
        )

    # -----------------------------------------------------------------------
    # Full Execution Workflow
    # -----------------------------------------------------------------------

    def execute_assignment(
        self,
        assignment: Union[TaskSpecification, Dict[str, Any]],
    ) -> TaskSpecification:
        """Execute the full assignment lifecycle from submission to verification.

        Delegates sequential execution to the WorkflowOrchestrator to ensure
        clean separation of concerns between task management and orchestration.
        """
        from core.orchestrator import WorkflowStatus

        # 1. Submission & Validation
        spec = self.submit_task(assignment)
        if spec.status == TaskStatus.REJECTED:
            return spec

        spec.status = TaskStatus.IN_PROGRESS
        spec.assigned_agents = [
            getattr(self.researcher, "name", "ResearcherAgent"),
            getattr(self.planner, "name", "PlannerAgent"),
            getattr(self.executor, "name", "ExecutorAgent"),
        ]
        if self.supervisor:
            sup_name = getattr(self.supervisor, "name", "SupervisorAgent")
            if sup_name not in spec.assigned_agents:
                spec.assigned_agents.append(sup_name)

        logger.info(f"[{spec.task_id}] Executing assignment: {spec.objective[:60]}...")

        # 2. Bridge approval callback and sensitivity requirements to WorkflowOrchestrator
        def orchestrator_approval_gate(plan: Dict[str, Any]) -> bool:
            spec.is_sensitive = True
            spec.requires_approval = True
            spec.status = TaskStatus.AWAITING_APPROVAL

            if self.approval_callback is None:
                return True

            try:
                import inspect
                sig = inspect.signature(self.approval_callback)
                if len(sig.parameters) >= 2:
                    return bool(self.approval_callback(spec, plan))
                return bool(self.approval_callback(plan))
            except Exception as exc:
                logger.error(f"Approval callback failed: {str(exc)}")
                return False

        self.orchestrator.approval_callback = orchestrator_approval_gate
        self.orchestrator.require_approval = spec.requires_approval or spec.is_sensitive

        # 3. Format structured workflow request
        workflow_parts = [f"Objective: {spec.objective}"]
        if spec.input_data is not None:
            workflow_parts.append(f"Input Data: {spec.input_data}")
        if spec.constraints:
            workflow_parts.append(f"Constraints: {'; '.join(spec.constraints)}")
        if spec.success_criteria:
            workflow_parts.append(f"Success Criteria: {'; '.join(spec.success_criteria)}")
        workflow_request = "\n\n".join(workflow_parts)

        # 4. Delegate workflow execution to WorkflowOrchestrator with consistent task_id!
        context = self.orchestrator.run(
            workflow_request,
            task_id=spec.task_id,
            success_criteria=spec.success_criteria,
            supervisor=self.supervisor,
        )

        # 5. Synchronize artifacts, results, and messages into TaskSpecification
        spec.results["research"] = context.research_findings or ""
        spec.results["plan"] = context.execution_plan or {}
        spec.results["execution_steps"] = context.execution_results
        spec.results["messages"] = [m.to_dict() for m in context.messages]
        spec.results["context"] = context
        spec.results["supervisor_decisions"] = context.metadata.get("supervisor_decisions", [])
        if context.metadata.get("clarification_question"):
            spec.results["clarification_question"] = context.metadata.get("clarification_question")
        spec.results["recovery_history"] = (
            context.recovery_history.to_dict() if getattr(context, "recovery_history", None) else {}
        )
        spec.results["reassignment_history"] = (
            [r.to_dict() for r in context.recovery_history.reassignments]
            if getattr(context, "recovery_history", None)
            else []
        )

        for err in context.errors:
            if err not in spec.errors:
                spec.errors.append(err)

        if context.metadata.get("is_sensitive"):
            spec.is_sensitive = True
            spec.requires_approval = True

        # Check for clarification request from Supervisor
        if context.metadata.get("clarification_needed"):
            spec.status = TaskStatus.AWAITING_APPROVAL
            spec.results["final_report"] = context.final_response
            spec.completed_at = datetime.now(timezone.utc).isoformat()
            return spec

        # 6. Check for approval gate halt (stopped / unapproved)
        if context.status == WorkflowStatus.STOPPED:
            spec.status = TaskStatus.REJECTED
            approval_err = "Approval Checkpoint: Sensitive operation was not approved."
            if approval_err not in spec.errors:
                spec.errors.append(approval_err)
            spec.completed_at = datetime.now(timezone.utc).isoformat()
            return spec

        # 7. Check for early agent failures (Researcher or Planner failure)
        if context.status == WorkflowStatus.FAILED or (not context.research_findings and not context.execution_plan):
            spec.status = TaskStatus.FAILED
            spec.completed_at = datetime.now(timezone.utc).isoformat()
            return spec

        # 8. Deterministic Result Evaluation Against Success Criteria & Evidence
        verification = self.verify_results(spec)
        spec.results["verification"] = verification.to_dict()

        # Step 3 & 4 Completion Status Semantics
        if verification.is_eligible_for_completion:
            spec.status = TaskStatus.COMPLETED
        elif len(verification.completed_steps) > 0 and verification.has_valid_evidence:
            spec.status = TaskStatus.PARTIALLY_COMPLETED
        else:
            spec.status = TaskStatus.FAILED

        # 9. Final Report & Completion
        spec.results["final_report"] = context.final_response or self._synthesize_task_report(spec, verification)
        spec.completed_at = datetime.now(timezone.utc).isoformat()

        logger.info(f"[{spec.task_id}] Assignment finished with status: {spec.status.value}")
        return spec

        logger.info(f"[{spec.task_id}] Assignment finished with status: {spec.status.value}")
        return spec

    # -----------------------------------------------------------------------
    # Report Synthesis
    # -----------------------------------------------------------------------

    def _synthesize_task_report(
        self,
        spec: TaskSpecification,
        verification: VerificationResult,
    ) -> str:
        """Construct the final structured report."""
        lines = [
            f"# Assignment Report: {spec.task_id}",
            f"**Objective:** {spec.objective}",
            f"**Status:** {spec.status.value.upper()}",
            f"**Created At:** {spec.created_at}",
            f"**Completed At:** {spec.completed_at or 'In Progress'}",
            "",
            "## 1. Success Criteria Evaluation",
        ]
        for crit in spec.success_criteria:
            icon = "✅" if crit in verification.passed_criteria else "❌"
            lines.append(f"- {icon} {crit}")

        lines.extend([
            "",
            "## 2. Research Findings",
            spec.results.get("research", "No research performed."),
            "",
            "## 3. Execution Steps",
        ])

        for step in spec.results.get("execution_steps", []):
            s_id = step.get("task_id", "-")
            s_task = step.get("task", "")
            s_status = step.get("status", "unknown")
            s_out = step.get("result", "")
            lines.append(f"- **Step {s_id}** [{s_status}]: {s_task}\n  Result: {s_out}")

        if spec.errors:
            lines.extend([
                "",
                "## 4. Errors & Caveats",
            ])
            for err in spec.errors:
                lines.append(f"- ⚠️ {err}")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Report Synthesis
    # -----------------------------------------------------------------------

    def _synthesize_task_report(
        self,
        spec: TaskSpecification,
        verification: VerificationResult,
    ) -> str:
        """Construct the final structured report."""
        lines = [
            f"# Assignment Report: {spec.task_id}",
            f"**Objective:** {spec.objective}",
            f"**Status:** {spec.status.value.upper()}",
            f"**Created At:** {spec.created_at}",
            f"**Completed At:** {spec.completed_at or 'In Progress'}",
            "",
            "## 1. Success Criteria Evaluation",
        ]
        for crit in spec.success_criteria:
            icon = "✅" if crit in verification.passed_criteria else "❌"
            lines.append(f"- {icon} {crit}")

        lines.extend([
            "",
            "## 2. Research Findings",
            spec.results.get("research", "No research performed."),
            "",
            "## 3. Execution Steps",
        ])

        for step in spec.results.get("execution_steps", []):
            s_id = step.get("task_id", "-")
            s_task = step.get("task", "")
            s_status = step.get("status", "unknown")
            s_out = step.get("result", "")
            lines.append(f"- **Step {s_id}** [{s_status}]: {s_task}\n  Result: {s_out}")

        if spec.errors:
            lines.extend([
                "",
                "## 4. Errors & Caveats",
            ])
            for err in spec.errors:
                lines.append(f"- ⚠️ {err}")

        return "\n".join(lines)
