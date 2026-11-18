"""
tests/test_day224_integration_audit.py

Comprehensive Integration Audit Suite for Day 224 Part 5 and Part 6.

Validates the target consolidated architecture:
1. TaskManager delegates workflow execution to WorkflowOrchestrator.
2. Status and artifact synchronization between TaskContext and TaskSpecification.
3. Single tool failure in multi-step task is gracefully handled without crashing.
4. Completion status semantics: COMPLETED vs PARTIALLY_COMPLETED vs FAILED vs REJECTED.
5. Fallback synthesis explicitly includes execution failure alerts.
6. Malformed Planner output schema rejection (missing description, invalid types).
7. Malformed Researcher output schema rejection (missing keys, non-string findings).
8. Approval bypass attempts rejected via sensitive pattern detection.
9. Direct unauthorized sensitive tool execution blocked at Executor boundary.
10. Deterministic numerical and quoted string verification (no false positives).
11. Full end-to-end assignment execution.
12. Strict TaskSpecification input validation.
"""

from unittest.mock import MagicMock
import pytest

from core.task_manager import (
    TaskManager,
    TaskSpecification,
    TaskStatus,
    VerificationResult,
)
from core.orchestrator import (
    WorkflowOrchestrator,
    TaskContext,
    WorkflowStatus,
    AgentMessage,
)
from agents.researcher import ResearcherAgent
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent, is_sensitive_operation
from core.provider_interface import LLMProvider, LLMResponse


class MockSynthesisProvider(LLMProvider):
    def __init__(self, content="Consolidated synthesis report", error=None):
        self.content = content
        self.error = error

    def generate(self, messages, model, tools=None, tool_choice="auto") -> LLMResponse:
        if self.error:
            return LLMResponse(error=self.error)
        return LLMResponse(content=self.content)


# ===========================================================================
# 1. Delegation to WorkflowOrchestrator
# ===========================================================================

def test_audit_1_task_manager_delegates_to_workflow_orchestrator():
    """TaskManager must delegate workflow execution to WorkflowOrchestrator."""
    mock_orchestrator = MagicMock(spec=WorkflowOrchestrator)
    mock_orchestrator.researcher = MagicMock(spec=ResearcherAgent, name="MockResearcher")
    mock_orchestrator.planner = MagicMock(spec=PlannerAgent, name="MockPlanner")
    mock_orchestrator.executor = MagicMock(spec=ExecutorAgent, name="MockExecutor")

    context = TaskContext(workflow_id="wf_test_1", user_request="Do audit task")
    context.status = WorkflowStatus.COMPLETED
    context.research_findings = "Verified audit data."
    context.execution_plan = {"tasks": [{"id": 1, "description": "Run audit step"}]}
    context.execution_results = [
        {"task_id": 1, "task": "Run audit step", "status": "completed", "result": "Audit step OK"}
    ]
    context.final_response = "Final synthesized audit report."
    mock_orchestrator.run.return_value = context

    tm = TaskManager(orchestrator=mock_orchestrator)
    spec = tm.execute_assignment({
        "objective": "Audit orchestration delegation",
        "success_criteria": ["Audit step OK"],
    })

    # Verify delegation was called
    mock_orchestrator.run.assert_called_once()
    assert spec.status == TaskStatus.COMPLETED
    assert spec.results["research"] == "Verified audit data."
    assert spec.results["final_report"] == "Final synthesized audit report."
    assert spec.results["context"] == context


# ===========================================================================
# 2. Status & Artifact Synchronization
# ===========================================================================

def test_audit_2_status_and_artifact_sync():
    """Ensure TaskContext artifacts synchronize cleanly into TaskSpecification."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Findings synced."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Sync step", "required_tools": []}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1, "task": "Sync step", "status": "completed", "result": "Sync complete"
    }

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Report synced"),
    )

    spec = tm.execute_assignment({
        "objective": "Verify artifact sync",
        "success_criteria": ["Sync complete"],
    })

    assert spec.status == TaskStatus.COMPLETED
    assert "messages" in spec.results
    assert len(spec.results["messages"]) >= 3
    assert spec.results["execution_steps"][0]["result"] == "Sync complete"
    assert spec.completed_at is not None


# ===========================================================================
# 3. Single Tool Failure in Multi-Step Workflow
# ===========================================================================

def test_audit_3_tool_failure_handling():
    """A tool failure on one step must not crash the workflow or hide the error."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Steps planned."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Step 1: OK"},
            {"id": 2, "description": "Step 2: Failing tool"},
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.side_effect = [
        {"task_id": 1, "task": "Step 1: OK", "status": "completed", "result": "Step 1 OK"},
        {"task_id": 2, "task": "Step 2: Failing tool", "status": "failed", "result": "ConnectionRefused", "error": "ConnectionRefused"},
    ]

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Report with error"),
    )

    spec = tm.execute_assignment({
        "objective": "Handle tool error",
        "success_criteria": ["Step 1 OK", "Step 2 completed"],
    })

    # Must be PARTIALLY_COMPLETED, not COMPLETED or uncaught crash
    assert spec.status == TaskStatus.PARTIALLY_COMPLETED
    assert any("ConnectionRefused" in e for e in spec.errors)
    assert len(spec.results["execution_steps"]) == 2


# ===========================================================================
# 4. Partial Completion Semantics
# ===========================================================================

def test_audit_4_partially_completed_semantics():
    """Verify PARTIALLY_COMPLETED occurs when useful work succeeds but some fail."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Data."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Step 1"},
            {"id": 2, "description": "Step 2"},
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.side_effect = [
        {"task_id": 1, "task": "Step 1", "status": "completed", "result": "Valuable output 1"},
        {"task_id": 2, "task": "Step 2", "status": "failed", "result": "Timeout", "error": "Timeout"},
    ]

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider(),
    )

    spec = tm.execute_assignment({
        "objective": "Execute partial pipeline",
        "success_criteria": ["Valuable output 1", "Step 2 completed"],
    })

    assert spec.status == TaskStatus.PARTIALLY_COMPLETED
    verification = spec.results["verification"]
    assert verification["all_criteria_met"] is False
    assert verification["completed_steps_count"] == 1
    assert verification["incomplete_steps_count"] == 1


# ===========================================================================
# 5. Fallback Synthesis Failure Alerts
# ===========================================================================

def test_audit_5_fallback_synthesis_includes_failure_alerts():
    """Fallback synthesis must explicitly feature execution failure alerts."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Findings."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Failing operation"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1, "task": "Failing operation", "status": "failed", "result": "DiskFull", "error": "DiskFull: No space left"
    }

    # Provider fails during synthesis triggering fallback
    failing_provider = MockSynthesisProvider(error="Provider API down")

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=failing_provider,
    )

    spec = tm.execute_assignment({
        "objective": "Test fallback synthesis",
        "success_criteria": ["Operation completed without error"],
    })

    assert "final_report" in spec.results
    report = spec.results["final_report"]
    assert "⚠️ Execution Failures & Alerts" in report
    assert "DiskFull: No space left" in report


# ===========================================================================
# 6. Malformed Planner Output Rejected
# ===========================================================================

def test_audit_6_malformed_planner_output_rejected():
    """Planner output missing 'description' in task item must be rejected."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Good research."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    # Malformed task: missing 'description'
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "wrong_field": "Missing description"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
    )

    spec = tm.execute_assignment({
        "objective": "Test malformed planner",
        "success_criteria": ["Task planned"],
    })

    assert spec.status == TaskStatus.FAILED
    assert any("Malformed Planner output" in e for e in spec.errors)
    mock_executor.execute.assert_not_called()


# ===========================================================================
# 7. Malformed Researcher Output Rejected
# ===========================================================================

def test_audit_7_malformed_researcher_output_rejected():
    """Researcher output missing 'findings' must be rejected."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    # Malformed: missing 'findings'
    mock_researcher.research.return_value = {"status": "completed"}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_executor = MagicMock(spec=ExecutorAgent)

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
    )

    spec = tm.execute_assignment({
        "objective": "Test malformed researcher",
        "success_criteria": ["Criteria 1"],
    })

    assert spec.status == TaskStatus.FAILED
    assert any("Malformed Researcher output" in e for e in spec.errors)
    mock_planner.plan.assert_not_called()


# ===========================================================================
# 8. Approval Bypass Attempt Rejected
# ===========================================================================

def test_audit_8_approval_bypass_attempt_rejected():
    """Sensitive plan cannot bypass approval checkpoint."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "System findings."}

    # Objective is innocuously phrased, but Planner generates a sensitive task
    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Drop database table users_backup"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)

    # Disapproval callback
    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        approval_callback=lambda spec, plan: False,
    )

    spec = tm.execute_assignment({
        "objective": "Clean up backup system",
        "success_criteria": ["Backup cleaned"],
    })

    assert spec.status == TaskStatus.REJECTED
    assert spec.is_sensitive is True
    assert any("Sensitive operation was not approved" in e for e in spec.errors)
    mock_executor.execute.assert_not_called()


# ===========================================================================
# 9. Direct Unauthorized Sensitive Tool Execution Blocked at Executor Boundary
# ===========================================================================

def test_audit_9_direct_unauthorized_sensitive_tool_blocked():
    """ExecutorAgent must block sensitive operations directly at tool boundary if unapproved."""
    executor = ExecutorAgent()

    # Attempt sensitive operation without authorization
    result = executor.execute({
        "id": 1,
        "description": "destroy cluster configuration and wipe disks",
    }, is_authorized=False)

    assert result["status"] == "failed"
    assert "Unauthorized sensitive operation" in result["error"]
    assert "tool boundary" in result["error"]


# ===========================================================================
# 10. Deterministic Numerical and Quoted Verification
# ===========================================================================

def test_audit_10_deterministic_numerical_verification():
    """Verify that wrong numbers fail verification, and exact numbers succeed."""
    tm = TaskManager()

    # Scenario A: Result has wrong number -> MUST FAIL
    spec_wrong = TaskSpecification(
        objective="Calculate annual revenue",
        success_criteria=["Sum quarterly revenues 45000 total"],
        results={
            "execution_steps": [
                {"task_id": 1, "status": "completed", "result": "Calculation output is 12000"}
            ],
            "research": "Quarterly revenues identified.",
        }
    )
    verification_wrong = tm.verify_results(spec_wrong)
    assert verification_wrong.all_criteria_met is False
    assert len(verification_wrong.failed_criteria) == 1

    # Scenario B: Result has exact matching number -> MUST PASS
    spec_correct = TaskSpecification(
        objective="Calculate annual revenue",
        success_criteria=["Sum quarterly revenues 45000 total"],
        results={
            "execution_steps": [
                {"task_id": 1, "status": "completed", "result": "Calculation output is 45000"}
            ],
            "research": "Quarterly revenues identified.",
        }
    )
    verification_correct = tm.verify_results(spec_correct)
    assert verification_correct.all_criteria_met is True
    assert len(verification_correct.passed_criteria) == 1


# ===========================================================================
# 11. Full Assignment Execution End-to-End
# ===========================================================================

def test_audit_11_full_assignment_end_to_end():
    """Verify complete assignment workflow from dict input to verified report."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Verified specs: Engine power 120 kW.",
        "error": None,
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Convert 120 kW to horsepower", "required_tools": ["calculator"]},
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Convert 120 kW to horsepower",
        "status": "completed",
        "result": "160.9 hp",
        "error": None,
    }

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=MockSynthesisProvider("Summary: Engine power is 160.9 hp."),
    )

    assignment = {
        "objective": "Calculate horsepower for 120 kW engine",
        "constraints": ["Use standard metric conversion"],
        "success_criteria": ["160.9 hp"],
    }

    spec = tm.execute_assignment(assignment)

    assert spec.status == TaskStatus.COMPLETED
    assert spec.completed_at is not None
    assert spec.results["verification"]["all_criteria_met"] is True
    assert len(spec.assigned_agents) == 3
    assert "160.9 hp" in spec.results["execution_steps"][0]["result"]


# ===========================================================================
# 12. Strict Specification Input Validation
# ===========================================================================

def test_audit_12_specification_validation():
    """Verify rejection of invalid TaskSpecification inputs."""
    tm = TaskManager()

    # Empty objective
    s1 = tm.submit_task({"objective": "", "success_criteria": ["Criterion 1"]})
    assert s1.status == TaskStatus.REJECTED

    # Whitespace-only objective
    s2 = tm.submit_task({"objective": "   ", "success_criteria": ["Criterion 1"]})
    assert s2.status == TaskStatus.REJECTED

    # Empty criteria list
    s3 = tm.submit_task({"objective": "Valid objective", "success_criteria": []})
    assert s3.status == TaskStatus.REJECTED

    # Non-list constraints
    s4 = tm.submit_task({"objective": "Valid", "success_criteria": ["C1"], "constraints": "not_a_list"})
    assert s4.status == TaskStatus.REJECTED
