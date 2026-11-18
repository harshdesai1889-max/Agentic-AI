"""
tests/test_day225_part4_worker_coordination.py

Comprehensive tests for Day 225 Part 4:
Worker Coordination and Result Evaluation.

Covers the 10 required test cases:
1. Successful Researcher-to-Planner handoff.
2. Successful Planner-to-Executor handoff.
3. Invalid Researcher output.
4. Invalid Planner output.
5. Missing required output fields.
6. Incorrect task IDs.
7. Failed Executor operation.
8. Successful result evaluation.
9. Failed result evaluation (rejecting vacuous self-claims).
10. Partial completion.
"""

from unittest.mock import MagicMock
import pytest

from core.evaluator import (
    ResultEvaluator,
    EvaluationResult,
    has_substantive_evidence,
    is_vacuous_claim,
)
from core.orchestrator import (
    WorkflowOrchestrator,
    TaskContext,
    WorkflowStatus,
    AgentMessage,
)
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


class DeterministicMockProvider(LLMProvider):
    def __init__(self, content="Default mock synthesis report"):
        self.content = content

    def generate(self, messages, model, tools=None, tool_choice="auto") -> LLMResponse:
        return LLMResponse(content=self.content)


# ===========================================================================
# Test 1: Successful Researcher-to-Planner Handoff
# ===========================================================================

def test_1_successful_researcher_to_planner_handoff():
    """Verify structured handoff from ResearcherAgent to PlannerAgent."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Verified: Li-ion cell operating range is 3.0V to 4.2V.",
        "task_id": "test_handoff_1",
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "description": "Calculate nominal voltage"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.return_value = {
        "task_id": 1,
        "task": "Calculate nominal voltage",
        "status": "completed",
        "result": "3.6V nominal",
    }

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=DeterministicMockProvider(),
    )

    context = orchestrator.run("Analyze battery range", task_id="test_handoff_1")

    assert context.status == WorkflowStatus.COMPLETED
    assert context.research_findings == "Verified: Li-ion cell operating range is 3.0V to 4.2V."

    # Planner must have received the research findings in its input
    planner_prompt = mock_planner.plan.call_args[0][0]
    assert "Li-ion cell operating range is 3.0V to 4.2V" in planner_prompt
    assert "Analyze battery range" in planner_prompt

    # Handoff messages properly recorded in audit history
    research_msg = next(m for m in context.messages if m.stage == "research")
    assert research_msg.status == "success"
    assert research_msg.agent_name == "ResearcherAgent"


# ===========================================================================
# Test 2: Successful Planner-to-Executor Handoff
# ===========================================================================

def test_2_successful_planner_to_executor_handoff():
    """Verify structured handoff from PlannerAgent to ExecutorAgent with task IDs."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {
        "status": "completed",
        "findings": "Server cluster status healthy.",
    }

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    mock_planner.plan.return_value = {
        "tasks": [
            {"id": 1, "description": "Check CPU utilization", "required_tools": []},
            {"id": 2, "description": "Check Memory utilization", "required_tools": []},
        ]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)
    mock_executor.name = "ExecutorAgent"
    mock_executor.execute.side_effect = [
        {"task_id": 1, "task": "Check CPU utilization", "status": "completed", "result": "CPU: 22%"},
        {"task_id": 2, "task": "Check Memory utilization", "status": "completed", "result": "Memory: 41%"},
    ]

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        provider=DeterministicMockProvider(),
    )

    context = orchestrator.run("Cluster health check", task_id="test_handoff_2")

    assert context.status == WorkflowStatus.COMPLETED
    assert mock_executor.execute.call_count == 2
    assert len(context.execution_results) == 2
    assert context.execution_results[0]["result"] == "CPU: 22%"
    assert context.execution_results[1]["result"] == "Memory: 41%"

    # Executor messages recorded with consistent stage and success
    exec_msgs = [m for m in context.messages if m.stage == "execution"]
    assert len(exec_msgs) == 2
    assert all(m.status == "success" for m in exec_msgs)


# ===========================================================================
# Test 3: Invalid Researcher Output
# ===========================================================================

def test_3_invalid_researcher_output():
    """Handoff validation must reject invalid or malformed Researcher output."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    # Malformed: missing 'findings'
    mock_researcher.research.return_value = {"status": "completed"}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_executor = MagicMock(spec=ExecutorAgent)

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
    )

    context = orchestrator.run("Research query")

    assert context.status == WorkflowStatus.FAILED
    assert any("Malformed Researcher output" in err for err in context.errors)
    # Downstream agents must NOT be called
    mock_planner.plan.assert_not_called()
    mock_executor.execute.assert_not_called()


# ===========================================================================
# Test 4: Invalid Planner Output
# ===========================================================================

def test_4_invalid_planner_output():
    """Handoff validation must reject malformed Planner task items."""
    mock_researcher = MagicMock(spec=ResearcherAgent)
    mock_researcher.name = "ResearcherAgent"
    mock_researcher.research.return_value = {"status": "completed", "findings": "Findings valid."}

    mock_planner = MagicMock(spec=PlannerAgent)
    mock_planner.name = "PlannerAgent"
    # Malformed: task item is missing required 'description'
    mock_planner.plan.return_value = {
        "tasks": [{"id": 1, "invalid_key": "No description provided"}]
    }

    mock_executor = MagicMock(spec=ExecutorAgent)

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
    )

    context = orchestrator.run("Plan objective")

    assert context.status == WorkflowStatus.FAILED
    assert any("Malformed Planner output" in err for err in context.errors)
    mock_executor.execute.assert_not_called()


# ===========================================================================
# Test 5: Missing Required Output Fields
# ===========================================================================

def test_5_missing_required_output_fields():
    """ResultEvaluator must identify missing required schema fields."""
    evaluator = ResultEvaluator()

    # Step missing 'result' field
    step_with_missing_field = {
        "task_id": 1,
        "status": "completed",
        # 'result' field is missing!
    }

    step_eval = evaluator.evaluate_step(step_with_missing_field)
    assert step_eval["valid"] is False
    assert "result" in step_eval["missing_fields"]

    # Full task evaluation with missing required expected output token
    eval_res = evaluator.evaluate(
        task_id="test_missing_5",
        objective="Generate metrics table",
        success_criteria=["Metrics generated"],
        execution_steps=[
            {"task_id": 1, "status": "completed", "result": "Some text without the expected key"}
        ],
        expected_output="throughput_metric, latency_metric",
    )

    assert eval_res.is_eligible_for_completion is False
    assert any("throughput_metric" in mf for mf in eval_res.missing_fields)
    assert any("latency_metric" in mf for mf in eval_res.missing_fields)


# ===========================================================================
# Test 6: Incorrect Task IDs
# ===========================================================================

def test_6_incorrect_task_ids():
    """Validation must detect and flag task ID mismatches during handoffs."""
    evaluator = ResultEvaluator()

    # Step output has mismatched task ID
    step = {
        "task_id": "step_999",
        "status": "completed",
        "result": "Calculated value 42",
    }
    step_eval = evaluator.evaluate_step(step, expected_task_id="step_100")

    assert step_eval["valid"] is False
    assert any("Task ID mismatch" in issue for issue in step_eval["issues"])

    # Orchestrator handoff validation also catches task ID mismatch
    orchestrator = WorkflowOrchestrator()
    is_valid, err = orchestrator.validate_executor_output(
        {"task_id": "wrong_id", "status": "completed", "result": "42"},
        expected_task_id="expected_id",
    )
    assert is_valid is False
    assert "Task ID mismatch" in err


# ===========================================================================
# Test 7: Failed Executor Operation
# ===========================================================================

def test_7_failed_executor_operation():
    """Failed executor operations must be flagged and prevent full completion."""
    evaluator = ResultEvaluator()

    failed_step = {
        "task_id": 1,
        "task": "Execute network call",
        "status": "failed",
        "result": "TimeoutError: Request timed out after 30s",
        "error": "TimeoutError: Request timed out after 30s",
    }

    eval_res = evaluator.evaluate(
        task_id="task_fail_7",
        objective="Fetch remote data",
        success_criteria=["Fetch remote data without error"],
        execution_steps=[failed_step],
    )

    assert eval_res.is_eligible_for_completion is False
    assert len(eval_res.failed_steps) == 1
    assert len(eval_res.completed_steps) == 0
    assert eval_res.status == "failed"


# ===========================================================================
# Test 8: Successful Result Evaluation
# ===========================================================================

def test_8_successful_result_evaluation():
    """Complete execution with verified evidence qualifies for COMPLETED status."""
    evaluator = ResultEvaluator()

    steps = [
        {"task_id": 1, "task": "Calculate gross margin", "status": "completed", "result": "Gross margin is 45000 USD."},
        {"task_id": 2, "task": "Format currency", "status": "completed", "result": "Formatted: $45,000 USD"},
    ]

    eval_res = evaluator.evaluate(
        task_id="task_eval_8",
        objective="Calculate and format quarterly margin",
        success_criteria=["Gross margin 45000", "Formatted $45,000 USD"],
        execution_steps=steps,
        research_findings="Quarterly financials verified.",
    )

    assert eval_res.is_eligible_for_completion is True
    assert eval_res.all_criteria_met is True
    assert eval_res.has_valid_evidence is True
    assert len(eval_res.failed_steps) == 0
    assert len(eval_res.completed_steps) == 2
    assert eval_res.status == "completed"


# ===========================================================================
# Test 9: Failed Result Evaluation (Rejecting Vacuous Self-Claims)
# ===========================================================================

def test_9_failed_result_evaluation_vacuous_claim():
    """Vacuous self-claims of completion must be rejected as lacking substantive evidence."""
    evaluator = ResultEvaluator()

    # Agent claims it completed the task, but provides zero data/numbers/evidence
    vacuous_steps = [
        {"task_id": 1, "task": "Sum total revenue", "status": "completed", "result": "Task completed successfully."}
    ]

    # Verify helper directly
    assert is_vacuous_claim("Task completed successfully.") is True
    assert is_vacuous_claim("I have completed the task.") is True
    assert is_vacuous_claim("Done.") is True
    assert has_substantive_evidence("Task completed successfully.") is False

    eval_res = evaluator.evaluate(
        task_id="task_vacuous_9",
        objective="Sum quarterly revenue",
        success_criteria=["Total revenue 45000"],
        execution_steps=vacuous_steps,
    )

    # Must NOT be eligible for completion
    assert eval_res.is_eligible_for_completion is False
    assert eval_res.has_valid_evidence is False
    assert eval_res.status == "failed"


# ===========================================================================
# Test 10: Partial Completion Evaluation
# ===========================================================================

def test_10_partial_completion_evaluation():
    """When useful work succeeds but some steps fail, status must be partially_completed."""
    evaluator = ResultEvaluator()

    steps = [
        {"task_id": 1, "task": "Query user profile", "status": "completed", "result": "User: Alice, Tier: Enterprise"},
        {"task_id": 2, "task": "Dispatch webhook", "status": "failed", "result": "Connection refused", "error": "ConnectionRefusedError"},
    ]

    eval_res = evaluator.evaluate(
        task_id="task_partial_10",
        objective="User profile and webhook pipeline",
        success_criteria=["User Alice verified", "Webhook dispatched"],
        execution_steps=steps,
    )

    assert eval_res.is_eligible_for_completion is False
    assert len(eval_res.completed_steps) == 1
    assert len(eval_res.failed_steps) == 1
    assert eval_res.has_valid_evidence is True
    assert eval_res.status == "partially_completed"

    # Integration via TaskManager
    tm = TaskManager()
    spec = TaskSpecification(
        task_id="task_tm_partial",
        objective="Pipeline workflow",
        success_criteria=["User Alice verified", "Webhook dispatched"],
        results={"execution_steps": steps, "research": ""},
    )
    verification = tm.verify_results(spec)

    assert verification.all_criteria_met is False
    assert verification.is_eligible_for_completion is False
    assert len(verification.completed_steps) == 1
    assert len(verification.incomplete_steps) == 1
