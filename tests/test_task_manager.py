"""
tests/test_task_manager.py

Comprehensive tests for Day 224 Part 6: Practical Task Assignment,
Execution, and Verification using TaskManager.

Covers the 10 required practical test cases:
1. Research a technical topic and prepare a report.
2. Analyse a structured dataset and produce findings.
3. Create an implementation plan for a simple software task.
4. Execute an authorized harmless tool operation.
5. Handle an invalid assignment.
6. Handle a Researcher failure.
7. Handle a Planner failure.
8. Handle an Executor failure.
9. Require approval for a sensitive operation.
10. Verify incomplete task execution.
"""

from unittest.mock import MagicMock
import pytest

from core.task_manager import (
    TaskManager,
    TaskSpecification,
    TaskStatus,
    VerificationResult,
)
from agents.researcher import ResearcherAgent
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from core.provider_interface import LLMProvider, LLMResponse


# ===========================================================================
# Deterministic Mock Provider
# ===========================================================================

class DeterministicMockProvider(LLMProvider):
    def __init__(self, content="Default mock synthesis report"):
        self.content = content

    def generate(self, messages, model, tools=None, tool_choice="auto") -> LLMResponse:
        return LLMResponse(content=self.content)


# ===========================================================================
# Test Case 1: Research a technical topic and prepare a report
# ===========================================================================

def test_1_research_technical_topic_and_report():
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Vector embeddings represent text as dense vectors in high-dimensional space.",
        "error": None,
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Compile research into report", "required_tools": []}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Compile research into report",
        "status": "completed",
        "result": "Report generated with vector embedding findings.",
        "error": None,
    }

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=DeterministicMockProvider(),
    )

    spec = tm.execute_assignment({
        "objective": "Research vector embeddings and compile a technical summary",
        "success_criteria": [
            "Research vector embeddings findings",
            "Generate report without errors",
        ],
    })

    assert spec.status == TaskStatus.COMPLETED
    assert "Vector embeddings" in spec.results["research"]
    assert len(spec.results["execution_steps"]) == 1
    assert spec.results["execution_steps"][0]["status"] == "completed"
    assert spec.completed_at is not None


# ===========================================================================
# Test Case 2: Analyse a structured dataset and produce findings
# ===========================================================================

def test_2_analyse_structured_dataset():
    dataset = [
        {"month": "Jan", "revenue": 10000},
        {"month": "Feb", "revenue": 15000},
        {"month": "Mar", "revenue": 20000},
    ]

    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Total dataset contains 3 months. Positive monthly trend observed.",
        "error": None,
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Sum revenues 10000 + 15000 + 20000", "required_tools": ["calculator"]},
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Sum revenues 10000 + 15000 + 20000",
        "status": "completed",
        "result": "45000",
        "error": None,
    }

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=DeterministicMockProvider(),
    )

    spec = tm.execute_assignment(TaskSpecification(
        objective="Analyse structured revenue dataset and sum quarterly total",
        input_data=dataset,
        success_criteria=["Sum revenues 45000 quarterly total"],
    ))

    assert spec.status == TaskStatus.COMPLETED
    assert "45000" in spec.results["execution_steps"][0]["result"]


# ===========================================================================
# Test Case 3: Create an implementation plan for a software task
# ===========================================================================

def test_3_create_implementation_plan():
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Authentication requires token generation, password hashing, and session lookup.",
        "error": None,
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Define User model and password hashing schema", "required_tools": []},
            {"id": 2, "description": "Implement JWT issue and verify functions", "required_tools": []},
            {"id": 3, "description": "Add login and register route handlers", "required_tools": []},
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.side_effect = [
        {"task_id": 1, "task": "Define User model", "status": "completed", "result": "Model spec created"},
        {"task_id": 2, "task": "Implement JWT", "status": "completed", "result": "JWT helper created"},
        {"task_id": 3, "task": "Add routes", "status": "completed", "result": "Routes mapped"},
    ]

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=DeterministicMockProvider(),
    )

    spec = tm.execute_assignment({
        "objective": "Create an implementation plan for a REST authentication service",
        "success_criteria": ["Model spec created", "JWT helper created", "Routes mapped"],
    })

    assert spec.status == TaskStatus.COMPLETED
    assert len(spec.results["plan"]["tasks"]) == 3


# ===========================================================================
# Test Case 4: Execute an authorized harmless tool operation
# ===========================================================================

def test_4_execute_authorized_harmless_tool():
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Expression to evaluate is 128 * 8.",
        "error": None,
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Evaluate 128 * 8 using AST calculator", "required_tools": ["calculator"]},
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Evaluate 128 * 8 using AST calculator",
        "status": "completed",
        "result": "1024",
        "error": None,
    }

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=DeterministicMockProvider(),
    )

    spec = tm.execute_assignment({
        "objective": "Perform math calculation 128 * 8",
        "success_criteria": ["Evaluate 1024 calculator result"],
    })

    assert spec.status == TaskStatus.COMPLETED
    assert spec.results["execution_steps"][0]["result"] == "1024"


# ===========================================================================
# Test Case 5: Handle an invalid assignment
# ===========================================================================

def test_5_handle_invalid_assignment():
    tm = TaskManager()

    # Empty objective
    spec1 = tm.execute_assignment({
        "objective": "",
        "success_criteria": ["Valid criteria"],
    })
    assert spec1.status == TaskStatus.REJECTED
    assert any("objective cannot be empty" in e.lower() for e in spec1.errors)

    # Empty success criteria
    spec2 = tm.execute_assignment({
        "objective": "Do something",
        "success_criteria": [],
    })
    assert spec2.status == TaskStatus.REJECTED
    assert any("at least one success criterion" in e.lower() for e in spec2.errors)


# ===========================================================================
# Test Case 6: Handle a Researcher failure
# ===========================================================================

def test_6_handle_researcher_failure():
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "failed",
        "findings": "",
        "error": "Database lookup timed out.",
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_executor = MagicMock(spec=ExecutorAgent)

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
    )

    spec = tm.execute_assignment({
        "objective": "Research market competitors",
        "success_criteria": ["Competitor analysis ready"],
    })

    assert spec.status == TaskStatus.FAILED
    assert any("Database lookup timed out" in e for e in spec.errors)
    mock_planner.plan.assert_not_called()
    mock_executor.execute.assert_not_called()


# ===========================================================================
# Test Case 7: Handle a Planner failure
# ===========================================================================

def test_7_handle_planner_failure():
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Facts found."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [],
        "error": "Failed to parse planning instructions.",
    }

    mock_executor = MagicMock(spec=ExecutorAgent)

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
    )

    spec = tm.execute_assignment({
        "objective": "Plan construction",
        "success_criteria": ["Construction planned"],
    })

    assert spec.status == TaskStatus.FAILED
    assert any("Failed to parse planning instructions" in e for e in spec.errors)
    mock_executor.execute.assert_not_called()


# ===========================================================================
# Test Case 8: Handle an Executor failure
# ===========================================================================

def test_8_handle_executor_failure():
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Target file located."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Execute critical calculation"}]
    }

    # Step execution fails with error
    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Execute critical calculation",
        "status": "failed",
        "result": "ZeroDivisionError: Division by zero",
        "error": "ZeroDivisionError: Division by zero",
    }

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=DeterministicMockProvider(),
    )

    spec = tm.execute_assignment({
        "objective": "Perform division operation",
        "success_criteria": ["Execute critical calculation without division by zero error"],
    })

    # When all steps fail, task status must be FAILED, not completed
    assert spec.status == TaskStatus.FAILED
    assert any("ZeroDivisionError" in e for e in spec.errors)


# ===========================================================================
# Test Case 9: Require approval for a sensitive operation
# ===========================================================================

def test_9_require_approval_for_sensitive_operation():
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Records to delete identified."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    # Plan contains sensitive action: "delete old database records"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Delete old database records from production"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)

    # 1. Rejection case: callback denies approval
    tm_reject = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        approval_callback=lambda spec, plan: False,  # Denied!
    )

    spec_denied = tm_reject.execute_assignment({
        "objective": "Purge outdated database records",
        "success_criteria": ["Records purged"],
    })

    assert spec_denied.status == TaskStatus.REJECTED
    assert spec_denied.is_sensitive is True
    assert any("Sensitive operation was not approved" in e for e in spec_denied.errors)
    mock_executor.execute.assert_not_called()

    # 2. Approved case: callback grants approval
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Delete old database records",
        "status": "completed",
        "result": "Deleted 10 records",
    }
    tm_approved = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        approval_callback=lambda spec, plan: True,  # Approved!
        provider=DeterministicMockProvider(),
    )

    spec_approved = tm_approved.execute_assignment({
        "objective": "Purge outdated database records",
        "success_criteria": ["Deleted 10 records"],
    })

    assert spec_approved.status == TaskStatus.COMPLETED
    mock_executor.execute.assert_called_once()


# ===========================================================================
# Test Case 10: Verify incomplete task execution
# ===========================================================================

def test_10_verify_incomplete_task_execution():
    """Verify that when 1 out of 2 tasks succeeds, status is PARTIALLY_COMPLETED, not COMPLETED."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Two tasks needed."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Task 1: Fetch user info"},
            {"id": 2, "description": "Task 2: Send confirmation"},
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.side_effect = [
        {"task_id": 1, "task": "Fetch user info", "status": "completed", "result": "User info fetched"},
        {"task_id": 2, "task": "Send confirmation", "status": "failed", "result": "Network timeout", "error": "Network timeout"},
    ]

    tm = TaskManager(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=DeterministicMockProvider(),
    )

    spec = tm.execute_assignment({
        "objective": "Complete two user workflows",
        "success_criteria": [
            "Fetch user info completed",
            "Send confirmation completed",
        ],
    })

    # Must NOT claim success!
    assert spec.status == TaskStatus.PARTIALLY_COMPLETED
    assert "verification" in spec.results
    verification = spec.results["verification"]
    assert verification["all_criteria_met"] is False
    assert verification["completed_steps_count"] == 1
    assert verification["incomplete_steps_count"] == 1
