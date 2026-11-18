"""
tests/test_day225_part6_supervisor_system.py

Comprehensive Test Suite for Day 225 Part 6:
Building a Supervisor-Based Multi-Agent System.

Validates:
1. SupervisorAgent initialization, identity, and system prompt.
2. Structured decision models (DELEGATE, COMPLETE, CLARIFY).
3. Rejection of invented agent names (WorkerRegistry enforcement).
4. Enforcement of worker capability restrictions.
5. Handling of clarification requests for ambiguous or incomplete inputs.
6. End-to-end dynamic delegation (Supervisor -> Researcher -> Planner -> Executor -> Synthesis).
7. ResultEvaluator verification gate preventing completion without verified evidence.
8. Prevention of infinite loops via maximum delegation steps.
9. Duplicate assignment loop detection and termination.
10. Preservation of approval gate and safety checks for sensitive operations.
11. Failure recovery and worker reassignment during supervised execution.
12. Non-idempotent operation protections under supervisor delegation.
13. TaskManager end-to-end integration with Supervisor.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agents.executor import ExecutorAgent
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.supervisor import (
    DecisionType,
    SupervisorAgent,
    SupervisorDecision,
    validate_supervisor_decision,
)
from core.evaluator import ResultEvaluator
from core.failure_recovery import FailureType, WorkerRegistry
from core.orchestrator import AgentMessage, TaskContext, WorkflowOrchestrator, WorkflowStatus
from core.provider_interface import LLMMessage, LLMProvider, LLMResponse
from core.task_manager import TaskManager, TaskSpecification, TaskStatus


# ---------------------------------------------------------------------------
# Mock Helpers
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """Predictable mock provider returning predetermined responses."""

    def __init__(self, response_text: str = "Mock synthesis report."):
        self.response_text = response_text
        self.call_count = 0

    def generate(
        self,
        messages: list[LLMMessage],
        model: str,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=self.response_text)


# ===========================================================================
# 1. Supervisor Initialization & Identity
# ===========================================================================

def test_1_supervisor_initialization_and_identity():
    """Verify SupervisorAgent initialization, identity properties, and prompt loading."""
    registry = WorkerRegistry()
    supervisor = SupervisorAgent(
        name="ChiefSupervisor",
        worker_registry=registry,
        max_delegation_steps=8,
    )

    ident = supervisor.get_identity()
    assert ident["name"] == "ChiefSupervisor"
    assert "Supervisor" in ident["role"]
    assert supervisor.max_delegation_steps == 8
    assert supervisor.worker_registry is registry
    assert "DELEGATE" in supervisor.system_prompt
    assert "COMPLETE" in supervisor.system_prompt
    assert "CLARIFY" in supervisor.system_prompt


# ===========================================================================
# 2. Structured Decision Models & Serialization
# ===========================================================================

def test_2_structured_decision_models_and_serialization():
    """Verify SupervisorDecision creation, serialization to dict, and from_dict parsing."""
    dec_delegate = SupervisorDecision(
        decision_type=DecisionType.DELEGATE,
        target_worker="ResearcherAgent",
        capability_required="research",
        instruction="Gather latest GPU benchmarks",
        reasoning="Need fresh benchmarks to plan comparison",
        step_number=1,
    )
    d1 = dec_delegate.to_dict()
    assert d1["decision"] == "DELEGATE"
    assert d1["target_worker"] == "ResearcherAgent"
    assert d1["capability_required"] == "research"

    restored_delegate = SupervisorDecision.from_dict(d1)
    assert restored_delegate.decision_type == DecisionType.DELEGATE
    assert restored_delegate.target_worker == "ResearcherAgent"

    dec_complete = SupervisorDecision(
        decision_type=DecisionType.COMPLETE,
        final_response="All benchmarks successfully summarized.",
        reasoning="All criteria verified with concrete figures.",
    )
    d2 = dec_complete.to_dict()
    assert d2["decision"] == "COMPLETE"
    assert "benchmarks" in d2["final_response"]

    dec_clarify = SupervisorDecision(
        decision_type=DecisionType.CLARIFY,
        clarification_question="Which fiscal year should be analyzed?",
        reasoning="Timeframe missing from user objective.",
    )
    d3 = dec_clarify.to_dict()
    assert d3["decision"] == "CLARIFY"
    assert "fiscal year" in d3["clarification_question"]


# ===========================================================================
# 3. Rejection of Invented Agent Names
# ===========================================================================

def test_3_rejection_of_invented_agent_names():
    """Verify that decisions specifying an unregistered agent are strictly rejected."""
    registry = WorkerRegistry()
    mock_worker = MagicMock()
    mock_worker.name = "LegitimateWorker"
    registry.register(mock_worker, capabilities=["research"])

    # Decision specifying an invented agent name
    decision_invented = SupervisorDecision(
        decision_type=DecisionType.DELEGATE,
        target_worker="InventedMagicAgent",
        capability_required="research",
        instruction="Perform impossible magic",
    )

    is_valid, err_msg, _ = validate_supervisor_decision(decision_invented, registry)
    assert is_valid is False
    assert "not registered in WorkerRegistry" in err_msg
    assert "Invented agents are strictly rejected" in err_msg

    # Legitimate registered worker passes
    decision_valid = SupervisorDecision(
        decision_type=DecisionType.DELEGATE,
        target_worker="LegitimateWorker",
        capability_required="research",
        instruction="Analyze verified documents",
    )
    is_valid, err_msg, _ = validate_supervisor_decision(decision_valid, registry)
    assert is_valid is True
    assert err_msg is None


# ===========================================================================
# 4. Enforcement of Worker Capability Boundaries
# ===========================================================================

def test_4_enforcement_of_worker_capability_boundaries():
    """Verify that delegating an unverified capability to a worker is rejected."""
    registry = WorkerRegistry()
    mock_researcher = MagicMock()
    mock_researcher.name = "ResearcherAgent"
    # Only registered for research, NOT tool execution!
    registry.register(mock_researcher, capabilities=["research", "analysis"])

    decision_unqualified = SupervisorDecision(
        decision_type=DecisionType.DELEGATE,
        target_worker="ResearcherAgent",
        capability_required="tool_execution",  # Not possessed!
        instruction="Execute system shell command",
    )

    is_valid, err_msg, _ = validate_supervisor_decision(decision_unqualified, registry)
    assert is_valid is False
    assert "lacks required capability 'tool_execution'" in err_msg


# ===========================================================================
# 5. Clarification Decision for Ambiguous or Incomplete Task
# ===========================================================================

def test_5_clarification_decision_for_ambiguous_task():
    """Verify that when a task is ambiguous or empty, Supervisor requests clarification."""
    supervisor = SupervisorAgent(provider=MockProvider())
    decision = supervisor.decide(
        objective="",  # Empty objective
        success_criteria=[],
        context_state={},
    )
    assert decision.decision_type == DecisionType.CLARIFY
    assert decision.clarification_question is not None
    assert "empty" in decision.clarification_question.lower()

    # Now verify that WorkflowOrchestrator handles CLARIFY by stopping cleanly
    orchestrator = WorkflowOrchestrator(supervisor=supervisor)
    context = orchestrator.run_supervised(user_request="   ")
    assert context.status == WorkflowStatus.FAILED
    assert "cannot be empty" in context.errors[0]


# ===========================================================================
# 6. End-to-End Dynamic Delegation Loop
# ===========================================================================

def test_6_dynamic_delegation_sequential_progression():
    """Verify full end-to-end multi-agent execution driven by the Supervisor.

    Step 1: Supervisor delegates to ResearcherAgent.
    Step 2: Supervisor delegates to PlannerAgent.
    Step 3: Supervisor delegates to ExecutorAgent.
    Step 4: Supervisor issues COMPLETE with verified criteria.
    """
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Global solar installation grew by 24% in 2026 reaching 500 GW.",
        "task_id": "task_sup_006",
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Calculate total capacity added: 500 GW", "authorized": True}
        ],
        "task_id": "task_sup_006",
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Calculate total capacity added: 500 GW",
        "status": "completed",
        "result": "Calculation confirmed: 500 GW added globally (growth rate 24%).",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockProvider("Solar market report finalized."),
    )
    supervisor = SupervisorAgent(
        worker_registry=orchestrator.worker_registry,
        provider=MockProvider("Solar market report finalized."),
    )
    orchestrator.supervisor = supervisor

    context = orchestrator.run(
        user_request="Analyze global solar capacity in 2026",
        task_id="task_sup_006",
        success_criteria=["500 GW", "24%"],
    )

    assert context.status == WorkflowStatus.COMPLETED
    assert context.research_findings is not None
    assert "24%" in context.research_findings
    assert len(context.execution_results) == 1
    assert context.execution_results[0]["status"] == "completed"
    assert "supervisor_decisions" in context.metadata
    assert len(context.metadata["supervisor_decisions"]) >= 3
    assert context.final_response is not None


# ===========================================================================
# 7. Verification Gate Rejects Unsupported Completion
# ===========================================================================

def test_7_verification_gate_rejects_unsupported_completion():
    """Verify that if the Supervisor attempts to complete, but criteria are unmet, completion is rejected."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "General clean energy report.",
        "task_id": "task_gate_007",
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Compute ROI", "authorized": True}],
        "task_id": "task_gate_007",
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    # Executor produces result missing the required criteria "Target ROI: 15%"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Compute ROI",
        "status": "completed",
        "result": "Done without numerical details",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockProvider("Report"),
    )
    supervisor = SupervisorAgent(
        worker_registry=orchestrator.worker_registry,
        provider=MockProvider("Report"),
    )
    orchestrator.supervisor = supervisor

    # Success criteria requires exact token "Target ROI: 15%" which is missing!
    context = orchestrator.run(
        user_request="Compute investment ROI",
        task_id="task_gate_007",
        success_criteria=['Target ROI: 15%'],
    )

    # Must NOT report COMPLETED because criteria were unmet!
    assert context.status in (WorkflowStatus.PARTIALLY_COMPLETED, WorkflowStatus.FAILED)
    assert any("verification" in err.lower() or "criteria" in err.lower() for err in context.errors)


# ===========================================================================
# 8. Loop Prevention: Maximum Delegation Steps
# ===========================================================================

def test_8_loop_prevention_maximum_delegation_steps():
    """Verify that a runaway supervisor loop is forcibly stopped at max_delegation_steps."""
    registry = WorkerRegistry()
    mock_worker = MagicMock()
    mock_worker.name = "EndlessWorker"
    mock_worker.run.return_value = "Step done."
    registry.register(mock_worker, capabilities=["looping"])

    # Mock a supervisor that constantly issues DELEGATE
    mock_supervisor = MagicMock(spec=SupervisorAgent)
    mock_supervisor.name = "LoopingSupervisor"
    mock_supervisor.max_delegation_steps = 3
    mock_supervisor.decide.side_effect = lambda **kwargs: SupervisorDecision(
        decision_type=DecisionType.DELEGATE,
        target_worker="EndlessWorker",
        capability_required="looping",
        # Vary instruction slightly to avoid the immediate duplicate-assignment detector
        instruction=f"Instruction variation #{kwargs.get('step_number', 1)}",
        step_number=kwargs.get("step_number", 1),
    )

    orchestrator = WorkflowOrchestrator(
        worker_registry=registry,
        supervisor=mock_supervisor,
        provider=MockProvider("Fallback report"),
    )

    context = orchestrator.run(
        user_request="Run endless task",
        task_id="task_loop_008",
        success_criteria=["Complete"],
    )

    assert context.status == WorkflowStatus.FAILED
    assert any("maximum delegation steps" in err.lower() for err in context.errors)


# ===========================================================================
# 9. Prevention of Duplicate Assignment Loops
# ===========================================================================

def test_9_prevention_of_duplicate_assignment_loops():
    """Verify that repeatedly delegating the exact same assignment without state progress is stopped."""
    registry = WorkerRegistry()
    mock_worker = MagicMock()
    mock_worker.name = "WorkerA"
    mock_worker.run.return_value = "Result"
    registry.register(mock_worker, capabilities=["generic"])

    mock_supervisor = MagicMock(spec=SupervisorAgent)
    mock_supervisor.name = "StuckSupervisor"
    mock_supervisor.max_delegation_steps = 10
    # Supervisor issues the exact same assignment repeatedly
    mock_supervisor.decide.return_value = SupervisorDecision(
        decision_type=DecisionType.DELEGATE,
        target_worker="WorkerA",
        capability_required="generic",
        instruction="Do identical calculation",
    )

    orchestrator = WorkflowOrchestrator(
        worker_registry=registry,
        supervisor=mock_supervisor,
        provider=MockProvider("Report"),
    )

    context = orchestrator.run("Test duplicate loop", task_id="task_dup_009")

    assert context.status == WorkflowStatus.FAILED
    assert any("infinite loop prevented" in err.lower() or "duplicate assignment" in err.lower() for err in context.errors)


# ===========================================================================
# 10. Safety & Approval Gate Enforcement Under Supervision
# ===========================================================================

def test_10_safety_and_approval_gate_enforcement_under_supervision():
    """Verify that sensitive operations planned under supervision enforce approval and halt on rejection."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Findings", "task_id": "task_safe_010"}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            # Sensitive operation: database drop
            {"id": 1, "description": "drop database production_analytics", "authorized": False}
        ],
        "task_id": "task_safe_010",
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        # Approval callback explicitly rejects the sensitive operation!
        approval_callback=lambda plan: False,
        provider=MockProvider("Report"),
    )
    supervisor = SupervisorAgent(
        worker_registry=orchestrator.worker_registry,
        provider=MockProvider("Report"),
    )
    orchestrator.supervisor = supervisor

    context = orchestrator.run("Drop analytics database", task_id="task_safe_010")

    # Workflow must be stopped at the approval gate!
    assert context.status == WorkflowStatus.STOPPED
    assert any("sensitive operation was not approved" in err.lower() or "approval" in err.lower() for err in context.errors)
    # Executor must NEVER have been called!
    assert mock_executor.execute.call_count == 0


# ===========================================================================
# 11. Failure Recovery & Worker Reassignment Under Supervision
# ===========================================================================

def test_11_failure_recovery_and_worker_reassignment_under_supervision():
    """Verify that Part 5 failure recovery and reassignment operate seamlessly during supervised delegation."""
    registry = WorkerRegistry()

    # Primary researcher that fails with transient timeout
    failing_researcher = MagicMock(spec=ResearcherAgent)
    failing_researcher.name = "FailingResearcher"
    failing_researcher.research.side_effect = [
        {"status": "failed", "error": "TimeoutError: API timed out", "findings": ""},
        {"status": "failed", "error": "TimeoutError: API timed out", "findings": ""},
    ]

    # Backup researcher that succeeds
    backup_researcher = MagicMock(spec=ResearcherAgent)
    backup_researcher.name = "BackupResearcher"
    backup_researcher.research.return_value = {
        "status": "completed",
        "findings": "Backup research recovered data successfully.",
        "task_id": "task_rec_011",
    }

    registry.register(failing_researcher, capabilities=["research"], priority=10)
    registry.register(backup_researcher, capabilities=["research"], priority=15)

    orchestrator = WorkflowOrchestrator(
        worker_registry=registry,
        max_retries=1,
        backoff_factor=0.0,
        provider=MockProvider("Report"),
    )

    # Supervisor delegates to FailingResearcher
    mock_supervisor = MagicMock(spec=SupervisorAgent)
    mock_supervisor.name = "SupervisorAgent"
    mock_supervisor.max_delegation_steps = 2
    mock_supervisor.decide.side_effect = [
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="FailingResearcher",
            capability_required="research",
            instruction="Research target sector",
        ),
        SupervisorDecision(
            decision_type=DecisionType.COMPLETE,
            final_response="Completed via backup worker.",
        ),
    ]

    context = orchestrator.run(
        user_request="Research target sector",
        task_id="task_rec_011",
        supervisor=mock_supervisor,
    )

    # Reassignment occurred and backup worker findings preserved
    assert context.research_findings == "Backup research recovered data successfully."
    reassignments = context.recovery_history.reassignments
    assert len(reassignments) == 1
    assert reassignments[0].original_worker == "FailingResearcher"
    assert reassignments[0].replacement_worker == "BackupResearcher"


# ===========================================================================
# 12. Non-Idempotent Operation Protection Under Supervision
# ===========================================================================

def test_12_non_idempotent_operation_protection_under_supervision():
    """Verify that non-idempotent operations (payment, fund transfer) are not automatically retried."""
    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "transfer funds of 1000 USD to contractor",
        "status": "failed",
        "result": "",
        "error": "Gateway timeout during fund transfer",
    }

    registry = WorkerRegistry()
    registry.register(mock_executor, capabilities=["execution"])

    mock_supervisor = MagicMock(spec=SupervisorAgent)
    mock_supervisor.max_delegation_steps = 2
    mock_supervisor.decide.side_effect = [
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="ExecutorAgent",
            capability_required="execution",
            instruction="transfer funds of 1000 USD to contractor",
            metadata={"task_item": {"id": 1, "description": "transfer funds of 1000 USD to contractor"}},
        ),
        SupervisorDecision(
            decision_type=DecisionType.COMPLETE,
            final_response="Halted non-idempotent operation.",
        ),
    ]

    orchestrator = WorkflowOrchestrator(
        executor=mock_executor,
        worker_registry=registry,
        max_retries=2,  # Retries configured, but must NOT retry non-idempotent operation!
        backoff_factor=0.0,
        provider=MockProvider("Report"),
        supervisor=mock_supervisor,
    )

    context = orchestrator.run("Process payment", task_id="task_idem_012")

    # Executor was invoked strictly once; automatic retrying of side-effects is prevented
    assert mock_executor.execute.call_count == 1
    assert any("non-idempotent" in a.error_message.lower() or "timeout" in (a.error_message or "").lower() for a in context.recovery_history.attempts)


# ===========================================================================
# 13. TaskManager End-to-End Integration With Supervisor
# ===========================================================================

def test_13_task_manager_integration_with_supervisor():
    """Verify TaskManager integrates seamlessly with Supervisor, producing full lifecycle records."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Revenue grew by 18% in Q4 2026.",
        "task_id": "task_tm_013",
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Format quarterly summary: 18% growth", "authorized": True}],
        "task_id": "task_tm_013",
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Format quarterly summary: 18% growth",
        "status": "completed",
        "result": "Quarterly summary formatted: confirmed 18% growth in Q4.",
    }

    supervisor = SupervisorAgent(provider=MockProvider("Final Q4 Executive Brief."))

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        supervisor=supervisor,
        provider=MockProvider("Final Q4 Executive Brief."),
    )

    spec = tm.execute_assignment({
        "task_id": "task_tm_013",
        "objective": "Audit Q4 financial performance",
        "success_criteria": ["18% growth"],
    })

    assert spec.status == TaskStatus.COMPLETED
    assert "SupervisorAgent" in spec.assigned_agents
    assert "supervisor_decisions" in spec.results
    assert len(spec.results["supervisor_decisions"]) >= 3
    assert "final_report" in spec.results
    assert spec.results["verification"]["all_criteria_met"] is True
