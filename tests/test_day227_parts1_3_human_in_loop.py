"""
tests/test_day227_parts1_3_human_in_loop.py

Comprehensive test suite for Day 227 Parts 1, 2, and 3:
- Part 1: HumanInteractionManager (Requests, Types, History, Context Attachment, SQLite Persistence & Restart Recovery).
- Part 2: HumanApprovalManager (Approval Gates, Expiration, Task/Action Binding, Execution Boundary Enforcement, Direct Bypass Prevention).
- Part 3: HumanFeedbackManager (Clarification Requests, Workflow Pause & State Snapshot, Validation, Resume Without Repetition).
- Integration: Complete HITL Workflow & Backward Compatibility.
"""

import os
import time
import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from core.hitl import (
    InteractionType,
    InteractionStatus,
    ApprovalDecision,
    HumanInteractionRequest,
    ApprovalRecord,
    HumanInteractionManager,
    HumanApprovalManager,
    HumanFeedbackManager,
    HumanInTheLoopEngine,
)
from core.persistent_storage import SQLiteMessageStorage
from core.orchestrator import (
    WorkflowOrchestrator,
    TaskContext,
    AgentMessage,
    WorkflowStatus,
    MessageType,
)
from core.message_router import MessageRouter
from agents.executor import ExecutorAgent
from agents.researcher import ResearcherAgent
from agents.planner import PlannerAgent
from agents.supervisor import SupervisorAgent
from core.provider_interface import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for fast and offline HITL test execution."""

    def __init__(self, canned_response: str = "Mock completed response"):
        self.canned_response = canned_response
        self.call_history = []

    def generate(self, messages, tools=None, **kwargs) -> LLMResponse:
        self.call_history.append((messages, tools, kwargs))
        return LLMResponse(
            content=self.canned_response,
            tool_calls=None,
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )


# ===========================================================================
# PART 1 TESTS: HumanInteractionManager
# ===========================================================================

class TestPart1HumanInteractionManager:
    """Test suite for Part 1: Structured HumanInteractionManager."""

    def test_interaction_request_creation(self):
        """Verify creation of structured interaction request with all metadata."""
        manager = HumanInteractionManager()
        req = manager.create_interaction(
            task_id="task_101",
            correlation_id="corr_101",
            requesting_agent="ResearcherAgent",
            interaction_type=InteractionType.CLARIFICATION_REQUEST,
            question_or_request="Please clarify which database to query.",
            relevant_context={"target": "users_db"},
            available_decisions=["PROVIDE_INFO", "CANCEL"],
            metadata={"priority": "high"},
        )

        assert req.interaction_id.startswith("hitl_")
        assert req.task_id == "task_101"
        assert req.correlation_id == "corr_101"
        assert req.requesting_agent == "ResearcherAgent"
        assert req.interaction_type == InteractionType.CLARIFICATION_REQUEST
        assert req.question_or_request == "Please clarify which database to query."
        assert req.relevant_context == {"target": "users_db"}
        assert req.status == InteractionStatus.PENDING
        assert req.is_pending() is True
        assert req.created_at is not None
        assert req.resolved_at is None
        assert req.decision is None

    def test_interaction_types_supported(self):
        """Verify all interaction types are supported."""
        manager = HumanInteractionManager()
        types = [
            InteractionType.APPROVAL_REQUEST,
            InteractionType.CLARIFICATION_REQUEST,
            InteractionType.REVIEW_REQUEST,
            InteractionType.FEEDBACK_REQUEST,
            InteractionType.INFORMATION_REQUEST,
            InteractionType.PLAN_MODIFICATION_REQUEST,
        ]

        for itype in types:
            req = manager.create_interaction(
                task_id="task_types",
                correlation_id="corr_types",
                requesting_agent="Supervisor",
                interaction_type=itype,
                question_or_request=f"Testing {itype.value}",
            )
            assert req.interaction_type == itype
            assert manager.get_interaction(req.interaction_id) is not None

    def test_interaction_history_and_filtering(self):
        """Verify history querying and filtering by task_id and status."""
        manager = HumanInteractionManager()
        r1 = manager.create_interaction("task_A", "corr_A", "Agent1", InteractionType.APPROVAL_REQUEST, "Approve 1")
        r2 = manager.create_interaction("task_A", "corr_A", "Agent1", InteractionType.CLARIFICATION_REQUEST, "Clarify 2")
        r3 = manager.create_interaction("task_B", "corr_B", "Agent2", InteractionType.FEEDBACK_REQUEST, "Feedback 3")

        # Resolve r1
        manager.resolve_interaction(r1.interaction_id, decision="APPROVED")

        # Filter by task
        task_a_list = manager.list_interactions(task_id="task_A")
        assert len(task_a_list) == 2
        assert {r.interaction_id for r in task_a_list} == {r1.interaction_id, r2.interaction_id}

        # Filter by status
        pending_list = manager.get_pending_interactions()
        assert len(pending_list) == 2
        assert r1.interaction_id not in [p.interaction_id for p in pending_list]

        pending_task_a = manager.get_pending_interactions(task_id="task_A")
        assert len(pending_task_a) == 1
        assert pending_task_a[0].interaction_id == r2.interaction_id

    def test_context_attachment(self):
        """Verify that creating an interaction attaches metadata and audit message to TaskContext."""
        manager = HumanInteractionManager()
        context = TaskContext(workflow_id="wf_audit", task_id="task_audit", user_request="Audit test")

        req = manager.create_interaction(
            task_id="task_audit",
            correlation_id="wf_audit",
            requesting_agent="ResearcherAgent",
            interaction_type=InteractionType.CLARIFICATION_REQUEST,
            question_or_request="Need more data",
            context=context,
        )

        assert "human_interactions" in context.metadata
        assert len(context.metadata["human_interactions"]) == 1
        assert context.metadata["human_interactions"][0]["interaction_id"] == req.interaction_id

        # Verify audit message in context
        audit_msgs = [m for m in context.messages if m.stage == "human_intervention"]
        assert len(audit_msgs) == 1
        assert audit_msgs[0].message_type == MessageType.CLARIFICATION_REQUEST
        assert audit_msgs[0].input_data == "Need more data"

    def test_sqlite_persistence_and_restart_recovery(self, tmp_path):
        """Verify interaction records persist to SQLite and survive manager restart."""
        db_path = str(tmp_path / "hitl_test.db")
        storage = SQLiteMessageStorage(db_path=db_path)

        manager1 = HumanInteractionManager(storage=storage)
        r1 = manager1.create_interaction("task_p1", "corr_p1", "Researcher", InteractionType.CLARIFICATION_REQUEST, "Q1")
        r2 = manager1.create_interaction("task_p2", "corr_p2", "Planner", InteractionType.APPROVAL_REQUEST, "Q2")

        manager1.resolve_interaction(r1.interaction_id, decision="PROVIDED", human_response="Answer 1")

        # Re-instantiate manager with same SQLite storage (simulate process restart)
        manager2 = HumanInteractionManager(storage=storage)

        recovered_r1 = manager2.get_interaction(r1.interaction_id)
        recovered_r2 = manager2.get_interaction(r2.interaction_id)

        assert recovered_r1 is not None
        assert recovered_r1.status == InteractionStatus.RESOLVED
        assert recovered_r1.decision == "PROVIDED"
        assert recovered_r1.human_response == "Answer 1"

        assert recovered_r2 is not None
        assert recovered_r2.status == InteractionStatus.PENDING
        assert recovered_r2.is_pending() is True


# ===========================================================================
# PART 2 TESTS: HumanApprovalManager & Execution Boundary
# ===========================================================================

class TestPart2HumanApprovalManager:
    """Test suite for Part 2: HumanApprovalManager and Execution Boundaries."""

    def test_approval_granted(self):
        """Verify approving an action issues an action-bound approval token."""
        approval_mgr = HumanApprovalManager()
        req = approval_mgr.request_approval(
            task_id="task_appr_1",
            correlation_id="corr_appr_1",
            requesting_agent="SupervisorAgent",
            requested_action="Deploy new index to production",
        )

        rec = approval_mgr.approve(req.interaction_id, feedback="Approved by SRE")

        assert rec.approval_status == "APPROVED"
        assert rec.decision == ApprovalDecision.APPROVED
        assert rec.human_feedback == "Approved by SRE"
        assert rec.token.startswith("tok_appr_")

        # Verify approval verification
        valid, err = approval_mgr.is_approved(
            task_id="task_appr_1",
            action="Deploy new index to production",
            token=rec.token,
        )
        assert valid is True
        assert err is None

    def test_approval_rejected(self):
        """Verify rejecting an action records rejection and prevents execution."""
        approval_mgr = HumanApprovalManager()
        req = approval_mgr.request_approval(
            task_id="task_rej_1",
            correlation_id="corr_rej_1",
            requesting_agent="SupervisorAgent",
            requested_action="Drop table users",
        )

        rec = approval_mgr.reject(req.interaction_id, reason="Destructive action prohibited")
        assert rec.approval_status == "REJECTED"
        assert rec.decision == ApprovalDecision.REJECTED

        valid, err = approval_mgr.is_approved("task_rej_1", action="Drop table users")
        assert valid is False
        assert "not approved" in err.lower()

    def test_approval_modification_requested(self):
        """Verify modification requested state captures human instructions."""
        approval_mgr = HumanApprovalManager()
        req = approval_mgr.request_approval(
            task_id="task_mod_1",
            correlation_id="corr_mod_1",
            requesting_agent="Planner",
            requested_action="Execute bulk update without dry run",
        )

        rec = approval_mgr.request_modification(req.interaction_id, instructions="Add dry run flag first")
        assert rec.approval_status == "MODIFICATION_REQUESTED"
        assert rec.human_feedback == "Add dry run flag first"

        valid, err = approval_mgr.is_approved("task_mod_1")
        assert valid is False

    def test_expired_approval(self):
        """Verify approvals past their TTL expire and are rejected."""
        approval_mgr = HumanApprovalManager(default_ttl_seconds=0.01)
        req = approval_mgr.request_approval(
            task_id="task_exp_1",
            correlation_id="corr_exp_1",
            requesting_agent="Supervisor",
            requested_action="Restart cluster",
            ttl_seconds=0.01,
        )

        # Wait for TTL expiration
        time.sleep(0.02)

        with pytest.raises(TimeoutError):
            approval_mgr.approve(req.interaction_id)

        valid, err = approval_mgr.is_approved("task_exp_1", action="Restart cluster")
        assert valid is False

    def test_missing_approval_rejection(self):
        """Verify rejection when no approval record exists for a task."""
        approval_mgr = HumanApprovalManager()
        valid, err = approval_mgr.is_approved("non_existent_task")
        assert valid is False
        assert "no approval record exists" in err.lower()

    def test_approval_bound_to_task_prevents_reuse(self):
        """Verify approval issued for task_A cannot be reused for task_B."""
        approval_mgr = HumanApprovalManager()
        req = approval_mgr.request_approval("task_A", "corr_A", "Supervisor", "Delete cache")
        rec = approval_mgr.approve(req.interaction_id)

        # Attempt to use token for task_B
        valid, err = approval_mgr.is_approved(task_id="task_B", action="Delete cache", token=rec.token)
        assert valid is False
        assert "task id mismatch" in err.lower()

    def test_execution_boundary_enforcement_in_executor(self):
        """Verify ExecutorAgent boundary rejects sensitive tasks when approval is missing."""
        approval_mgr = HumanApprovalManager()
        executor = ExecutorAgent(approval_manager=approval_mgr)

        sensitive_task = "delete customer accounts from database"
        res = executor.execute(sensitive_task, expected_task_id="task_sec_1")

        assert res["status"] == "failed"
        assert "Human approval boundary check failed" in res["error"]
        assert "no approval record exists" in res["error"].lower()

    def test_direct_execution_bypass_prevention(self):
        """Verify direct invocation with is_authorized=True CANNOT bypass boundary when approval_manager is set."""
        approval_mgr = HumanApprovalManager()
        executor = ExecutorAgent(approval_manager=approval_mgr)

        # Direct bypass attempt: passing is_authorized=True for sensitive operation
        res = executor.execute(
            {"description": "purge all audit logs", "task_id": "task_bypass_1", "authorized": True},
            is_authorized=True,
        )

        # Must still be rejected at tool boundary
        assert res["status"] == "failed"
        assert "Human approval boundary check failed" in res["error"]

    def test_authorized_execution_with_token(self):
        """Verify ExecutorAgent executes successfully when valid approval token is provided."""
        approval_mgr = HumanApprovalManager()
        executor = ExecutorAgent(approval_manager=approval_mgr)

        action = "calculate total users to remove"
        # Explicitly mark requires_approval
        req = approval_mgr.request_approval("task_ok_1", "corr_1", "Supervisor", action)
        rec = approval_mgr.approve(req.interaction_id)

        # Execute with valid approval token and mock run_agent
        with patch("agents.executor.execute_task", return_value="Calculation: 42 users"):
            res = executor.execute(
                {"description": action, "task_id": "task_ok_1", "requires_approval": True, "approval_token": rec.token},
                expected_task_id="task_ok_1",
            )

        assert res["status"] == "completed"
        assert "42 users" in res["result"]


# ===========================================================================
# PART 3 TESTS: HumanFeedbackManager & Workflow Pause/Resume
# ===========================================================================

class TestPart3HumanFeedbackManager:
    """Test suite for Part 3: HumanFeedbackManager and Workflow Resumption."""

    def test_clarification_request_and_workflow_pause(self):
        """Verify requesting clarification pauses the workflow and persists checkpoint."""
        feedback_mgr = HumanFeedbackManager()
        context = TaskContext(workflow_id="wf_pause_1", task_id="task_p1", user_request="Analyze data")
        context.research_findings = "Preliminary findings collected."
        context.status = WorkflowStatus.RESEARCHING

        req = feedback_mgr.request_clarification(
            task_id="task_p1",
            correlation_id="wf_pause_1",
            requesting_agent="ResearcherAgent",
            question="Which date range should be analyzed?",
            context=context,
        )

        assert context.status == WorkflowStatus.PAUSED
        assert context.metadata["clarification_pending"] is True
        assert context.metadata["pending_interaction_id"] == req.interaction_id
        assert context.research_findings == "Preliminary findings collected."

    def test_feedback_submission_and_validation(self):
        """Verify feedback submission validates content and attaches to context."""
        feedback_mgr = HumanFeedbackManager()
        context = TaskContext(workflow_id="wf_sub_1", task_id="task_sub_1", user_request="Build plan")
        req = feedback_mgr.request_clarification(
            task_id="task_sub_1",
            correlation_id="wf_sub_1",
            requesting_agent="Planner",
            question="Which budget constraint to apply?",
            context=context,
        )

        # 1. Empty feedback rejected
        ok, msg, _ = feedback_mgr.submit_feedback(req.interaction_id, "", task_id="task_sub_1", context=context)
        assert ok is False
        assert "cannot be empty" in msg.lower()

        # 2. Valid feedback accepted
        ok, msg, resolved_req = feedback_mgr.submit_feedback(
            req.interaction_id, "Budget is $5,000", task_id="task_sub_1", context=context
        )
        assert ok is True
        assert resolved_req.status == InteractionStatus.RESOLVED
        assert context.metadata["clarification_pending"] is False
        assert len(context.metadata["human_feedback"]) == 1
        assert context.metadata["human_feedback"][0]["response"] == "Budget is $5,000"

    def test_duplicate_response_prevention(self):
        """Verify that duplicate feedback for the same interaction is strictly rejected."""
        feedback_mgr = HumanFeedbackManager()
        req = feedback_mgr.request_clarification("task_dup_1", "corr_1", "Planner", "Question?")

        ok1, _, _ = feedback_mgr.submit_feedback(req.interaction_id, "First response")
        assert ok1 is True

        # Duplicate submission rejected
        ok2, msg2, _ = feedback_mgr.submit_feedback(req.interaction_id, "Duplicate response")
        assert ok2 is False
        assert "duplicate" in msg2.lower()

    def test_incorrect_task_response_rejection(self):
        """Verify feedback targeting the wrong task ID is rejected."""
        feedback_mgr = HumanFeedbackManager()
        req = feedback_mgr.request_clarification("task_target_A", "corr_1", "Researcher", "Question?")

        # Provide response with mismatching task_id
        ok, msg, _ = feedback_mgr.submit_feedback(req.interaction_id, "Feedback", task_id="task_target_B")
        assert ok is False
        assert "task mismatch" in msg.lower()

    def test_cancellation_handling(self):
        """Verify cancellation marks interaction cancelled and cannot resume stopped workflow."""
        feedback_mgr = HumanFeedbackManager()
        req = feedback_mgr.request_clarification("task_cancel_1", "corr_1", "Supervisor", "Question?")

        feedback_mgr.interaction_manager.cancel_interaction(req.interaction_id, reason="User cancelled task")
        cancelled_req = feedback_mgr.interaction_manager.get_interaction(req.interaction_id)
        assert cancelled_req.status == InteractionStatus.CANCELLED

        # Cannot submit feedback on cancelled interaction
        ok, msg, _ = feedback_mgr.submit_feedback(req.interaction_id, "Late response")
        assert ok is False

        # Cannot resume stopped workflow
        ctx = TaskContext(workflow_id="wf_stop", task_id="task_stop", user_request="Stopped")
        ctx.status = WorkflowStatus.STOPPED
        with pytest.raises(RuntimeError):
            feedback_mgr.resume_workflow(ctx, orchestrator=MagicMock())

    def test_workflow_resume_preserves_completed_work_without_repetition(self, tmp_path):
        """Verify that resuming a paused workflow preserves research and plan checkpoints."""
        db_path = str(tmp_path / "resume_test.db")
        storage = SQLiteMessageStorage(db_path=db_path)
        feedback_mgr = HumanFeedbackManager(storage=storage)

        context = TaskContext(
            workflow_id="wf_checkpoint_1",
            task_id="task_cp_1",
            user_request="Perform full audit",
        )
        context.research_findings = "Phase 1 research completed successfully."
        context.execution_plan = {"tasks": [{"id": 1, "description": "Execute audit step 1"}]}

        # Pause workflow
        req = feedback_mgr.request_clarification(
            task_id="task_cp_1",
            correlation_id="wf_checkpoint_1",
            requesting_agent="Researcher",
            question="Include legacy servers?",
            context=context,
        )

        # Submit feedback
        feedback_mgr.submit_feedback(req.interaction_id, "Yes, include legacy servers", context=context)

        # Resume from checkpoint
        mock_orch = MagicMock()
        mock_orch.run_communication_workflow.side_effect = lambda user_request, task_id, **kw: TaskContext(
            workflow_id=task_id,
            task_id=task_id,
            user_request=user_request,
            research_findings=context.research_findings,
            execution_plan=context.execution_plan,
        )

        resumed_ctx = feedback_mgr.resume_workflow(
            workflow_id_or_context="wf_checkpoint_1",
            orchestrator=mock_orch,
            feedback_response="Yes, include legacy servers",
        )

        assert resumed_ctx.research_findings == "Phase 1 research completed successfully."
        assert resumed_ctx.execution_plan["tasks"][0]["description"] == "Execute audit step 1"
        assert "Yes, include legacy servers" in resumed_ctx.user_request

    def test_multiple_clarifications_history(self):
        """Verify multiple rounds of clarification requests are tracked sequentially."""
        feedback_mgr = HumanFeedbackManager()
        context = TaskContext(workflow_id="wf_multi", task_id="task_multi", user_request="Multi clarify")

        # Round 1
        req1 = feedback_mgr.request_clarification("task_multi", "wf_multi", "Researcher", "Clarify Q1", context=context)
        feedback_mgr.submit_feedback(req1.interaction_id, "Answer A1", task_id="task_multi", context=context)

        # Round 2
        req2 = feedback_mgr.request_clarification("task_multi", "wf_multi", "Planner", "Clarify Q2", context=context)
        feedback_mgr.submit_feedback(req2.interaction_id, "Answer A2", task_id="task_multi", context=context)

        assert len(context.metadata["human_feedback"]) == 2
        assert context.metadata["human_feedback"][0]["response"] == "Answer A1"
        assert context.metadata["human_feedback"][1]["response"] == "Answer A2"


# ===========================================================================
# INTEGRATION TESTS: End-to-End HITL Workflow & Backward Compatibility
# ===========================================================================

class TestHITLIntegration:
    """Integration test suite connecting Parts 1, 2, and 3."""

    def test_complete_hitl_end_to_end_workflow(self, tmp_path):
        """Execute full HITL integration flow:
        Clarification Pause -> Human Feedback -> Resume -> Approval Gate -> Token Verification -> Execution.
        """
        db_path = str(tmp_path / "integration_hitl.db")
        storage = SQLiteMessageStorage(db_path=db_path)
        engine = HumanInTheLoopEngine(storage=storage)

        mock_provider = MockLLMProvider("Mock response")
        executor = ExecutorAgent(provider=mock_provider, approval_manager=engine.approvals)
        orchestrator = WorkflowOrchestrator(
            provider=mock_provider,
            executor=executor,
            hitl_engine=engine,
            require_approval=True,
        )

        task_id = "wf_hitl_e2e"
        context = TaskContext(workflow_id=task_id, task_id=task_id, user_request="Clean customer data")

        # Step 1: Researcher requests clarification
        clarify_req = engine.feedback.request_clarification(
            task_id=task_id,
            correlation_id=task_id,
            requesting_agent="ResearcherAgent",
            question="Which table should be cleaned?",
            context=context,
        )
        assert context.status == WorkflowStatus.PAUSED

        # Step 2: Human answers clarification
        engine.feedback.submit_feedback(
            interaction_id=clarify_req.interaction_id,
            response_text="Clean table 'customers_inactive'",
            task_id=task_id,
            context=context,
        )

        # Step 3: Planner generates plan
        context.research_findings = "Identified 50 inactive records in customers_inactive table."
        context.execution_plan = {
            "tasks": [
                {
                    "id": 1,
                    "description": "delete inactive records from customers_inactive",
                    "requires_approval": True,
                }
            ]
        }

        # Step 4: Supervisor triggers approval gate
        appr_req = engine.approvals.request_approval(
            task_id=task_id,
            correlation_id=task_id,
            requesting_agent="SupervisorAgent",
            requested_action="delete inactive records from customers_inactive",
            context=context,
        )

        # Human approves
        appr_rec = engine.approvals.approve(appr_req.interaction_id, feedback="Approved by Data Protection Officer")
        assert appr_rec.token is not None

        # Step 5: Executor validates token and executes sensitive task
        with patch("agents.executor.execute_task", return_value="50 inactive records removed"):
            exec_res = executor.execute(
                {
                    "description": "delete inactive records from customers_inactive",
                    "task_id": task_id,
                    "requires_approval": True,
                    "approval_token": appr_rec.token,
                },
                expected_task_id=task_id,
            )

        assert exec_res["status"] == "completed"
        assert "50 inactive records removed" in exec_res["result"]

    def test_backward_compatibility_without_hitl(self):
        """Verify that components operate normally without HITL engine configured."""
        mock_provider = MockLLMProvider("Normal execution")
        executor = ExecutorAgent(provider=mock_provider)  # No approval manager attached

        # Non-sensitive task executes normally
        with patch("agents.executor.execute_task", return_value="Calculation complete: 10"):
            res = executor.execute("calculate sum of 5 and 5")
            assert res["status"] == "completed"

        # Sensitive task with is_authorized=True executes normally when no approval manager attached
        with patch("agents.executor.execute_task", return_value="Deleted temporary file"):
            res_sens = executor.execute("delete temporary cache", is_authorized=True)
            assert res_sens["status"] == "completed"
