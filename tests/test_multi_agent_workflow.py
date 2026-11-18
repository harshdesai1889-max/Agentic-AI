"""
tests/test_multi_agent_workflow.py

Comprehensive tests for Day 224 Part 5:
Multi-Agent Communication and Collaboration using:
1. ResearcherAgent
2. PlannerAgent
3. ExecutorAgent
4. WorkflowOrchestrator, TaskContext, and AgentMessage

Tests verify:
- Researcher-to-Planner communication
- Planner-to-Executor communication
- Structured data transfer and handoff validation
- Shared TaskContext updates
- Approval gate safety mechanism
- Invalid agent outputs and graceful termination
- Agent and tool failure handling
- Final report synthesis
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from core.provider_interface import LLMProvider, LLMResponse, LLMMessage
from core.orchestrator import (
    WorkflowOrchestrator,
    TaskContext,
    AgentMessage,
    WorkflowStatus,
)
from agents.researcher import ResearcherAgent
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from config.settings import MODEL1, MODEL2


# ===========================================================================
# Deterministic Mock Provider for Multi-Agent Workflow Tests
# ===========================================================================

class SequenceMockProvider(LLMProvider):
    """Returns sequential pre-configured responses for workflow steps."""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses) if responses else []
        self.error = error
        self.calls = []

    def generate(self, messages, model, tools=None, tool_choice="auto") -> LLMResponse:
        self.calls.append({
            "messages": messages,
            "model": model,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        if self.error:
            return LLMResponse(error=self.error)
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, LLMResponse):
                return item
            return LLMResponse(content=str(item))
        return LLMResponse(content="Default mock content")


# ===========================================================================
# 1. End-to-End Sequential Communication Tests
# ===========================================================================

class TestSequentialAgentCommunication:
    def test_complete_end_to_end_workflow(self):
        """Test the full pipeline: Researcher -> Planner -> Executor -> Synthesis."""
        # 1. Researcher output
        mock_research = {
            "query": "Assess battery efficiency",
            "status": "completed",
            "findings": "Verified: Li-ion cell voltage is 3.7V. Capacity: 5000mAh.",
            "error": None,
        }
        # 2. Planner output
        mock_plan = {
            "tasks": [
                {"id": 1, "description": "Multiply 3.7 by 5", "required_tools": ["calculator"]},
                {"id": 2, "description": "Format watt-hour output", "required_tools": []},
            ]
        }
        # 3. Executor mock responses (2 tasks)
        exec_step1 = {"task_id": 1, "task": "Multiply 3.7 by 5", "status": "completed", "result": "18.5", "error": None}
        exec_step2 = {"task_id": 2, "task": "Format watt-hour output", "status": "completed", "result": "18.5 Wh total", "error": None}

        # Setup mock agents
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = mock_research

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.name = "PlannerAgent"
        mock_planner.plan.return_value = mock_plan

        mock_executor = MagicMock(spec=ExecutorAgent)
        mock_executor.name = "ExecutorAgent"
        mock_executor.execute.side_effect = [exec_step1, exec_step2]

        # Synthesis provider
        synth_provider = SequenceMockProvider(responses=["Final Analysis: Total energy capacity is 18.5 Wh."])

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
            provider=synth_provider,
        )

        context = orchestrator.run("Assess battery efficiency")

        # Verify final status
        assert context.status == WorkflowStatus.COMPLETED
        assert context.final_response == "Final Analysis: Total energy capacity is 18.5 Wh."

        # Verify Researcher was called with original request
        mock_researcher.research.assert_called_once_with("Assess battery efficiency")

        # Verify Planner received BOTH original request AND research findings
        planner_arg = mock_planner.plan.call_args[0][0]
        assert "Assess battery efficiency" in planner_arg
        assert "Verified: Li-ion cell voltage is 3.7V" in planner_arg

        # Verify Executor received the exact tasks from the plan
        assert mock_executor.execute.call_count == 2
        assert mock_executor.execute.call_args_list[0][0][0] == mock_plan["tasks"][0]
        assert mock_executor.execute.call_args_list[1][0][0] == mock_plan["tasks"][1]

        # Verify execution results recorded in context
        assert len(context.execution_results) == 2
        assert context.execution_results[0]["result"] == "18.5"
        assert context.execution_results[1]["result"] == "18.5 Wh total"

    def test_run_to_text_returns_string(self):
        """Verify the convenience run_to_text method returns string content."""
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = {"status": "completed", "findings": "Facts found."}

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.name = "PlannerAgent"
        mock_planner.plan.return_value = {"tasks": [{"id": 1, "description": "Step 1"}]}

        mock_executor = MagicMock(spec=ExecutorAgent)
        mock_executor.name = "ExecutorAgent"
        mock_executor.execute.return_value = {"task_id": 1, "status": "completed", "result": "Done"}

        synth_provider = SequenceMockProvider(responses=["Consolidated Answer."])

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
            provider=synth_provider,
        )

        text_result = orchestrator.run_to_text("Quick request")
        assert text_result == "Consolidated Answer."


# ===========================================================================
# 2. Shared Context and Audit Trail Tests
# ===========================================================================

class TestSharedContextAndAuditTrail:
    def test_task_context_tracks_all_stages(self):
        """Verify that TaskContext records AgentMessages for each handoff."""
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = {"status": "completed", "findings": "Research OK"}

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.name = "PlannerAgent"
        mock_planner.plan.return_value = {"tasks": [{"id": 1, "description": "Task 1"}]}

        mock_executor = MagicMock(spec=ExecutorAgent)
        mock_executor.name = "ExecutorAgent"
        mock_executor.execute.return_value = {"task_id": 1, "status": "completed", "result": "Result 1"}

        synth_provider = SequenceMockProvider(responses=["Synthesis complete."])

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
            provider=synth_provider,
        )

        context = orchestrator.run("Trace stages")

        # Stages: research, planning, execution (1 task), synthesis = 4 messages
        stages = [msg.stage for msg in context.messages]
        assert "research" in stages
        assert "planning" in stages
        assert "execution" in stages
        assert "synthesis" in stages

        # Check message contents
        research_msg = next(m for m in context.messages if m.stage == "research")
        assert research_msg.agent_name == "ResearcherAgent"
        assert research_msg.status == "success"

        plan_msg = next(m for m in context.messages if m.stage == "planning")
        assert plan_msg.agent_name == "PlannerAgent"
        assert plan_msg.status == "success"

        exec_msg = next(m for m in context.messages if m.stage == "execution")
        assert exec_msg.agent_name == "ExecutorAgent"
        assert exec_msg.status == "success"

    def test_context_to_dict_serialization(self):
        """Verify TaskContext and AgentMessage convert cleanly to dicts."""
        msg = AgentMessage(
            task_id="t1",
            agent_name="AgentA",
            stage="research",
            status="success",
            input_data="query",
            output_data="findings",
        )
        d = msg.to_dict()
        assert d["task_id"] == "t1"
        assert d["agent_name"] == "AgentA"

        ctx = TaskContext(workflow_id="wf_123", user_request="Do task")
        ctx.add_message(msg)
        ctx_dict = ctx.to_dict()
        assert ctx_dict["workflow_id"] == "wf_123"
        assert ctx_dict["message_count"] == 1


# ===========================================================================
# 3. Input Validation and Failure Handling Tests
# ===========================================================================

class TestFailureHandlingAndValidation:
    def test_empty_user_request_fails_immediately(self):
        orchestrator = WorkflowOrchestrator()
        context = orchestrator.run("")

        assert context.status == WorkflowStatus.FAILED
        assert "User request cannot be empty." in context.errors

    def test_researcher_failure_halts_workflow(self):
        """If Researcher fails, Planner and Executor must NOT be called."""
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = {
            "status": "failed",
            "findings": "",
            "error": "External data source unavailable.",
        }

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_executor = MagicMock(spec=ExecutorAgent)

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
        )

        context = orchestrator.run("Research market")

        assert context.status == WorkflowStatus.FAILED
        assert "External data source unavailable." in context.errors
        mock_planner.plan.assert_not_called()
        mock_executor.execute.assert_not_called()

    def test_researcher_empty_findings_halts_workflow(self):
        """If Researcher returns empty findings string, workflow must halt safely."""
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = {
            "status": "completed",
            "findings": "   ",  # blank findings
            "error": None,
        }

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_executor = MagicMock(spec=ExecutorAgent)

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
        )

        context = orchestrator.run("Research topic")

        assert context.status == WorkflowStatus.FAILED
        assert any("empty findings" in err for err in context.errors)
        mock_planner.plan.assert_not_called()

    def test_planner_failure_halts_workflow(self):
        """If Planner fails or returns empty tasks, Executor must NOT be called."""
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = {"status": "completed", "findings": "Findings valid."}

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.name = "PlannerAgent"
        mock_planner.plan.return_value = {
            "tasks": [],
            "error": "Could not decompose objective into tasks.",
        }

        mock_executor = MagicMock(spec=ExecutorAgent)

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
        )

        context = orchestrator.run("Plan mission")

        assert context.status == WorkflowStatus.FAILED
        assert "Could not decompose objective into tasks." in context.errors
        mock_executor.execute.assert_not_called()

    def test_executor_task_failure_is_recorded_and_tolerated(self):
        """When an individual task fails in Executor, workflow records failure and continues."""
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = {"status": "completed", "findings": "Findings."}

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.name = "PlannerAgent"
        mock_planner.plan.return_value = {
            "tasks": [
                {"id": 1, "description": "Step 1"},
                {"id": 2, "description": "Step 2"},
            ]
        }

        # Step 1 fails, Step 2 succeeds
        mock_executor = MagicMock(spec=ExecutorAgent)
        mock_executor.name = "ExecutorAgent"
        mock_executor.execute.side_effect = [
            {"task_id": 1, "task": "Step 1", "status": "failed", "result": "Division by zero", "error": "Division by zero"},
            {"task_id": 2, "task": "Step 2", "status": "completed", "result": "Success", "error": None},
        ]

        synth_provider = SequenceMockProvider(responses=["Synthesis after partial failure."])

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
            provider=synth_provider,
        )

        context = orchestrator.run("Tolerate step failure")

        assert context.status == WorkflowStatus.COMPLETED
        assert any("Task 1 failed" in err for err in context.errors)
        assert len(context.execution_results) == 2
        assert context.execution_results[0]["status"] == "failed"
        assert context.execution_results[1]["status"] == "completed"


# ===========================================================================
# 4. Approval Gate (Safety Control) Tests
# ===========================================================================

class TestApprovalGate:
    def test_approval_gate_accepts_when_approved(self):
        """Workflow proceeds when approval callback returns True."""
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = {"status": "completed", "findings": "Research OK."}

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.name = "PlannerAgent"
        mock_planner.plan.return_value = {"tasks": [{"id": 1, "description": "Safe task"}]}

        mock_executor = MagicMock(spec=ExecutorAgent)
        mock_executor.name = "ExecutorAgent"
        mock_executor.execute.return_value = {"task_id": 1, "status": "completed", "result": "Executed"}

        synth_provider = SequenceMockProvider(responses=["Synthesis done."])

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
            provider=synth_provider,
            require_approval=True,
            approval_callback=lambda plan: True,  # Approved
        )

        context = orchestrator.run("Perform safe action")

        assert context.status == WorkflowStatus.COMPLETED
        mock_executor.execute.assert_called_once()

    def test_approval_gate_halts_when_rejected(self):
        """Workflow halts and Executor is NOT called when approval callback returns False."""
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = {"status": "completed", "findings": "Research OK."}

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.name = "PlannerAgent"
        mock_planner.plan.return_value = {"tasks": [{"id": 1, "description": "Dangerous task"}]}

        mock_executor = MagicMock(spec=ExecutorAgent)

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
            require_approval=True,
            approval_callback=lambda plan: False,  # Rejected!
        )

        context = orchestrator.run("Perform dangerous action")

        assert context.status == WorkflowStatus.STOPPED
        assert any("not approved" in err for err in context.errors)
        mock_executor.execute.assert_not_called()


# ===========================================================================
# 5. Fallback Synthesis Test
# ===========================================================================

class TestFallbackSynthesis:
    def test_fallback_synthesis_used_when_llm_synthesis_fails(self):
        """When the synthesis LLM call fails, the orchestrator constructs a structured report."""
        mock_researcher = MagicMock(spec=ResearcherAgent)
        mock_researcher.name = "ResearcherAgent"
        mock_researcher.research.return_value = {"status": "completed", "findings": "Market is growing."}

        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.name = "PlannerAgent"
        mock_planner.plan.return_value = {"tasks": [{"id": 1, "description": "Calculate growth"}]}

        mock_executor = MagicMock(spec=ExecutorAgent)
        mock_executor.name = "ExecutorAgent"
        mock_executor.execute.return_value = {"task_id": 1, "task": "Calculate growth", "status": "completed", "result": "12%"}

        # Provider fails during synthesis
        failing_provider = SequenceMockProvider(error="Synthesis API timeout")

        orchestrator = WorkflowOrchestrator(
            researcher=mock_researcher,
            planner=mock_planner,
            executor=mock_executor,
            provider=failing_provider,
        )

        context = orchestrator.run("Assess growth")

        assert context.status == WorkflowStatus.COMPLETED
        assert context.final_response is not None
        assert "Execution Report" in context.final_response
        assert "Market is growing." in context.final_response
        assert "Calculate growth" in context.final_response
