"""
tests/test_day225_part5_failure_recovery.py

Comprehensive Test Suite for Day 225 Part 5: Failure Recovery and Reassignment.

Covers all 12 required scenarios:
1. Successful execution without recovery.
2. Recovery from a transient failure.
3. Maximum retry limit enforcement.
4. Permanent failure without retry.
5. Authorization failure without automatic retry.
6. Successful Worker reassignment.
7. No suitable replacement Worker.
8. Preservation of previous successful outputs.
9. Correct task ID continuity.
10. Prevention of duplicate non-idempotent execution.
11. Partial completion after an unrecoverable failure.
12. Accurate recovery history.
"""

from unittest.mock import MagicMock
import pytest

from core.failure_recovery import (
    FailureType,
    FailureDetails,
    AttemptRecord,
    ReassignmentRecord,
    RecoveryHistory,
    FailureClassifier,
    WorkerRegistry,
    is_non_idempotent_operation,
)
from core.orchestrator import (
    WorkflowOrchestrator,
    TaskContext,
    WorkflowStatus,
)
from core.task_manager import (
    TaskManager,
    TaskSpecification,
    TaskStatus,
)
from core.provider_interface import LLMProvider, LLMMessage, LLMResponse
from agents.researcher import ResearcherAgent
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent


class MockSynthesisProvider(LLMProvider):
    """Deterministic mock synthesis provider."""

    def __init__(self, content: str = "Synthesized report.", error: str = None):
        self.content = content
        self.error = error

    def generate(self, messages, model, tools=None, tool_choice="auto"):
        if self.error:
            return LLMResponse(error=self.error)
        return LLMResponse(content=self.content)


# ===========================================================================
# 1. Successful execution without recovery
# ===========================================================================

def test_1_successful_execution_without_recovery():
    """Test standard multi-agent execution completing cleanly without needing recovery."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Solar panel efficiency data collected.",
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Calculate average output"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Calculate average output",
        "status": "completed",
        "result": "Calculated average: 320 kWh",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Solar report complete."),
        max_retries=2,
        backoff_factor=0.0,
    )

    context = orchestrator.run("Assess solar output", task_id="task_success_001")

    assert context.status == WorkflowStatus.COMPLETED
    assert context.recovery_history.total_retries == 0
    assert context.recovery_history.recovered is False
    assert len(context.recovery_history.attempts) >= 3
    assert all(a.status == "success" for a in context.recovery_history.attempts)
    assert len(context.recovery_history.reassignments) == 0


# ===========================================================================
# 2. Recovery from a transient failure
# ===========================================================================

def test_2_recovery_from_transient_failure():
    """Test successful recovery when an agent encounters a transient network timeout on attempt 1."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    # Attempt 1 fails with a transient error, Attempt 2 succeeds
    mock_researcher.research.side_effect = [
        {"status": "failed", "error": "503 Service Unavailable: temporary spike in demand", "findings": ""},
        {"status": "completed", "findings": "Recovered findings on second attempt.", "task_id": "task_retry_002"},
    ]

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Step 1"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Step 1",
        "status": "completed",
        "result": "Done",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Report"),
        max_retries=2,
        backoff_factor=0.0,
    )

    context = orchestrator.run("Query data", task_id="task_retry_002")

    assert context.status == WorkflowStatus.COMPLETED
    assert context.recovery_history.total_retries == 1
    assert context.recovery_history.recovered is True
    # Attempt 1 was classified as TRANSIENT
    assert context.recovery_history.attempts[0].status == "failed"
    assert context.recovery_history.attempts[0].failure_type == FailureType.TRANSIENT
    # Attempt 2 succeeded
    assert context.recovery_history.attempts[1].status == "success"
    assert context.research_findings == "Recovered findings on second attempt."


# ===========================================================================
# 3. Maximum retry limit enforcement
# ===========================================================================

def test_3_maximum_retry_limit_enforcement():
    """Test that retries halt strictly after max_retries without entering an infinite loop."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    # Always fails with a transient error
    mock_researcher.research.return_value = {
        "status": "failed",
        "error": "RateLimitError: 429 Too Many Requests",
        "findings": "",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        provider=MockSynthesisProvider("Fallback"),
        max_retries=2,  # 1 initial attempt + 2 retries = 3 total attempts
        backoff_factor=0.0,
    )

    context = orchestrator.run("Query rate limited service", task_id="task_max_retry_003")

    assert context.status == WorkflowStatus.FAILED
    research_attempts = [a for a in context.recovery_history.attempts if a.stage == "research"]
    assert len(research_attempts) == 3
    assert context.recovery_history.total_retries == 2
    assert mock_researcher.research.call_count == 3


# ===========================================================================
# 4. Permanent failure without retry
# ===========================================================================

def test_4_permanent_failure_without_retry():
    """Test that permanent non-recoverable errors are rejected from automatic retry immediately."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "failed",
        "error": "Permanent failure: 404 Model not found or deprecated.",
        "findings": "",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        provider=MockSynthesisProvider("Report"),
        max_retries=2,
        backoff_factor=0.0,
    )

    context = orchestrator.run("Query missing model", task_id="task_perm_004")

    assert context.status == WorkflowStatus.FAILED
    research_attempts = [a for a in context.recovery_history.attempts if a.stage == "research"]
    # Only 1 attempt occurred; permanent errors must not be retried!
    assert len(research_attempts) == 1
    assert research_attempts[0].failure_type == FailureType.PERMANENT
    assert context.recovery_history.total_retries == 0
    assert mock_researcher.research.call_count == 1


# ===========================================================================
# 5. Authorization failure without automatic retry
# ===========================================================================

def test_5_authorization_failure_without_automatic_retry():
    """Test that authorization or approval checkpoint rejections are never automatically retried."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Findings."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "delete production database"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"

    # Human-in-the-loop approval gate rejects the sensitive plan
    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        require_approval=True,
        approval_callback=lambda plan: False,  # Rejected!
        max_retries=2,
        backoff_factor=0.0,
    )

    context = orchestrator.run("Purge data", task_id="task_auth_005")

    assert context.status == WorkflowStatus.STOPPED
    approval_attempts = [a for a in context.recovery_history.attempts if a.stage == "approval"]
    assert len(approval_attempts) == 1
    assert approval_attempts[0].status == "failed"
    assert approval_attempts[0].failure_type == FailureType.AUTHORIZATION
    # Executor was never invoked!
    assert mock_executor.execute.call_count == 0


# ===========================================================================
# 6. Successful Worker reassignment
# ===========================================================================

def test_6_successful_worker_reassignment():
    """Test that when a primary worker fails, a registered capable replacement worker is reassigned and succeeds."""
    mock_primary_researcher = MagicMock(spec=ResearcherAgent)
    mock_primary_researcher.name = "PrimaryResearcher"
    mock_primary_researcher.research.return_value = {
        "status": "failed",
        "error": "Validation error: Malformed researcher output",
        "findings": "",
    }

    mock_backup_researcher = MagicMock(spec=ResearcherAgent)
    mock_backup_researcher.name = "BackupResearcher"
    mock_backup_researcher.research.return_value = {
        "status": "completed",
        "findings": "Verified research findings from backup worker.",
        "task_id": "task_reassign_006",
    }

    registry = WorkerRegistry()
    registry.register(mock_primary_researcher, capabilities=["research"], priority=10, name="PrimaryResearcher")
    registry.register(mock_backup_researcher, capabilities=["research"], priority=5, name="BackupResearcher")

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {"tasks": [{"id": 1, "description": "Step 1"}]}

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {"task_id": 1, "task": "Step 1", "status": "completed", "result": "Done"}

    orchestrator = WorkflowOrchestrator(
        researcher=mock_primary_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Report"),
        max_retries=1,
        backoff_factor=0.0,
        worker_registry=registry,
    )

    context = orchestrator.run("Research advanced physics", task_id="task_reassign_006")

    assert context.status == WorkflowStatus.COMPLETED
    assert context.recovery_history.recovered is True
    assert len(context.recovery_history.reassignments) == 1
    reassignment = context.recovery_history.reassignments[0]
    assert reassignment.status == "reassigned"
    assert reassignment.original_worker == "PrimaryResearcher"
    assert reassignment.replacement_worker == "BackupResearcher"
    assert context.research_findings == "Verified research findings from backup worker."


# ===========================================================================
# 7. No suitable replacement Worker
# ===========================================================================

def test_7_no_suitable_replacement_worker():
    """Test that when a worker fails and no capable replacement is registered, the failure is recorded without inventing agents."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "SoleResearcher"
    mock_researcher.research.return_value = {
        "status": "failed",
        "error": "Validation error: Missing required fields",
        "findings": "",
    }

    # Registry only contains the failing researcher and unrelated executor
    registry = WorkerRegistry()
    registry.register(mock_researcher, capabilities=["research"], priority=10, name="SoleResearcher")

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        provider=MockSynthesisProvider("Report"),
        max_retries=1,
        backoff_factor=0.0,
        worker_registry=registry,
    )

    context = orchestrator.run("Research quantum computing", task_id="task_no_worker_007")

    assert context.status == WorkflowStatus.FAILED
    assert len(context.recovery_history.reassignments) == 1
    reassignment = context.recovery_history.reassignments[0]
    assert reassignment.status == "failed_no_worker"
    assert reassignment.replacement_worker is None
    assert "No capable replacement worker" in reassignment.reason


# ===========================================================================
# 8. Preservation of previous successful outputs
# ===========================================================================

def test_8_preservation_of_previous_successful_outputs():
    """Test that when a downstream stage fails, artifacts from upstream successful stages are preserved in context."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Critical solar power market data from 2026.",
        "task_id": "task_preserve_008",
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    # Planner fails completely
    mock_planner.plan.return_value = {
        "tasks": [],
        "error": "Planner failed to decompose objective into valid JSON",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        provider=MockSynthesisProvider("Report"),
        max_retries=1,
        backoff_factor=0.0,
    )

    context = orchestrator.run("Analyze market", task_id="task_preserve_008")

    assert context.status == WorkflowStatus.FAILED
    # Research findings MUST be preserved intact!
    assert context.research_findings == "Critical solar power market data from 2026."
    assert any("Planner failed" in err or "Planning stage failed" in err for err in context.errors)


# ===========================================================================
# 9. Correct task ID continuity
# ===========================================================================

def test_9_correct_task_id_continuity():
    """Test that all attempts, retries, and messages maintain continuous task ID tracking."""
    test_id = "task_continuity_audit_999"

    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.side_effect = [
        {"status": "failed", "error": "TimeoutError: Network timeout", "findings": ""},
        {"status": "completed", "findings": "Findings OK.", "task_id": test_id},
    ]

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "task_id": f"{test_id}_task_1", "description": "Step 1"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": f"{test_id}_task_1",
        "task": "Step 1",
        "status": "completed",
        "result": "Success",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Synthesized"),
        max_retries=2,
        backoff_factor=0.0,
    )

    context = orchestrator.run("Perform continuous audit", task_id=test_id)

    assert context.status == WorkflowStatus.COMPLETED
    assert context.task_id == test_id
    # Every attempt recorded carries the expected task ID
    assert all(a.task_id == test_id or a.task_id.startswith(test_id) for a in context.recovery_history.attempts)
    # Every message carries the expected task ID
    assert all(m.task_id == test_id or m.task_id.startswith(test_id) for m in context.messages)


# ===========================================================================
# 10. Prevention of duplicate non-idempotent execution
# ===========================================================================

def test_10_prevention_of_duplicate_non_idempotent_execution():
    """Test that sensitive non-idempotent operations (e.g. fund transfer, email) are never retried automatically."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Findings."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            # Non-idempotent operation with external side-effects
            {"id": 1, "description": "transfer funds of 500 dollars to vendor", "authorized": True}
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "transfer funds of 500 dollars to vendor",
        "status": "failed",
        "result": "Network timeout during payment gateway call",
        "error": "Network timeout during payment gateway call",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Report"),
        approval_callback=lambda plan: True,  # Approved sensitive plan
        max_retries=2,  # Retries are configured, but non-idempotent must NOT retry!
        backoff_factor=0.0,
    )

    context = orchestrator.run("Process transfer", task_id="task_idempotency_010")

    # Executor was invoked strictly ONCE; automatic repetition of side-effects is prevented!
    assert mock_executor.execute.call_count == 1
    assert any("timeout" in (a.error_message or "").lower() for a in context.recovery_history.attempts)


# ===========================================================================
# 11. Partial completion after an unrecoverable failure
# ===========================================================================

def test_11_partial_completion_after_an_unrecoverable_failure():
    """Test that when one execution task fails unrecoverably while others succeed, TaskManager records PARTIALLY_COMPLETED."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Data."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Step 1: Compute metrics"},
            {"id": 2, "description": "Step 2: Database purge"},
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    # Step 1 succeeds, Step 2 fails permanently
    mock_executor.execute.side_effect = [
        {"task_id": 1, "task": "Step 1: Compute metrics", "status": "completed", "result": "Metrics: 42"},
        {"task_id": 2, "task": "Step 2: Database purge", "status": "failed", "result": "Disk error", "error": "Disk error: Write failed"},
    ]

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Report after partial completion"),
        max_retries=1,
        backoff_factor=0.0,
    )

    spec = tm.execute_assignment({
        "objective": "Compute metrics and purge database",
        "success_criteria": ["Metrics: 42", "Database purged"],
    })

    assert spec.status == TaskStatus.PARTIALLY_COMPLETED
    assert len(spec.results["execution_steps"]) == 2
    assert spec.results["execution_steps"][0]["status"] == "completed"
    assert spec.results["execution_steps"][1]["status"] == "failed"
    assert "final_report" in spec.results


# ===========================================================================
# 12. Accurate recovery history
# ===========================================================================

def test_12_accurate_recovery_history():
    """Test that recovery history contains structured timestamps, attempt counts, and reassignment records."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.side_effect = [
        {"status": "failed", "error": "TimeoutError: Request timed out", "findings": ""},
        {"status": "completed", "findings": "Findings on attempt 2.", "task_id": "task_history_012"},
    ]

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {"tasks": [{"id": 1, "description": "Step 1"}]}

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {"task_id": 1, "task": "Step 1", "status": "completed", "result": "Success"}

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Final report"),
        max_retries=2,
        backoff_factor=0.0,
    )

    spec = tm.execute_assignment({
        "task_id": "task_history_012",
        "objective": "Audit recovery history",
        "success_criteria": ["Success"],
    })

    rec_hist = spec.results.get("recovery_history", {})
    assert rec_hist.get("total_retries") == 1
    assert rec_hist.get("recovered") is True
    assert "attempts" in rec_hist
    attempts = rec_hist["attempts"]
    assert len(attempts) >= 4  # research attempt 1, research attempt 2, planning attempt 1, execution attempt 1

    # Verify attempt fields
    first_attempt = attempts[0]
    assert first_attempt["attempt_number"] == 1
    assert first_attempt["status"] == "failed"
    assert first_attempt["failure_type"] == FailureType.TRANSIENT.value
    assert "timestamp" in first_attempt
    assert first_attempt["task_id"] == spec.task_id

    second_attempt = attempts[1]
    assert second_attempt["attempt_number"] == 2
    assert second_attempt["status"] == "success"
    assert "timestamp" in second_attempt
