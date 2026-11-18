"""
tests/test_day227_parts4_6_production_hitl.py

Comprehensive test suite for Day 227 Parts 4, 5, and 6:
- Part 4: Risk-Based Human Intervention (Classifications, Policies, Mandatory Authorizations, Reassessments, Persistence).
- Part 5: Human Review, Audit Trail & Feedback Intelligence (Reviews, Outcomes, Append-only Audit, Pattern Discovery, Advisory Proposals).
- Part 6: Complete Production HITL Integration (18-step workflow, boundary enforcement, failure & recovery, restart recovery, backward compatibility).
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
    RiskLevel,
    RiskPolicyConfig,
    RiskAssessmentResult,
    RiskAssessmentEngine,
    ReviewOutcome,
    HumanReviewRecord,
    HumanReviewManager,
    AuditEventType,
    AuditEvent,
    AuditTrail,
    FeedbackSummary,
    PolicyChangeProposal,
    FeedbackAnalysisEngine,
    redact_sensitive_data,
)
from core.persistent_storage import SQLiteMessageStorage
from core.orchestrator import (
    WorkflowOrchestrator,
    TaskContext,
    AgentMessage,
    WorkflowStatus,
    MessageType,
)
from agents.executor import ExecutorAgent
from agents.researcher import ResearcherAgent
from agents.planner import PlannerAgent
from agents.supervisor import SupervisorAgent
from core.provider_interface import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for offline testing."""

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
# PART 4 TESTS: Risk-Based Human Intervention
# ===========================================================================

class TestPart4RiskBasedIntervention:
    """Test suite for Part 4: RiskAssessmentEngine and Risk-Based Policies."""

    def test_low_risk_action_classification(self):
        """Verify low-risk actions (read, search, calculate) classify as LOW without requiring approval."""
        engine = RiskAssessmentEngine()
        res = engine.evaluate_action(
            task_id="task_low_1",
            action="calculate 15 percent tip on 85 dollars",
        )

        assert res.risk_level == RiskLevel.LOW
        assert res.risk_score < 0.30
        assert res.approval_required is False
        assert "automated_execution" in res.required_controls

    def test_medium_risk_action_classification(self):
        """Verify medium-risk actions (write file, update records) classify as MEDIUM with parameter validation."""
        engine = RiskAssessmentEngine()
        res = engine.evaluate_action(
            task_id="task_med_1",
            action="update user profile display name in staging database",
        )

        assert res.risk_level == RiskLevel.MEDIUM
        assert 0.30 <= res.risk_score < 0.60
        assert "parameter_validation" in res.required_controls
        assert res.approval_required is False

    def test_high_risk_action_classification(self):
        """Verify high-risk actions (delete records, modify system config) classify as HIGH and require approval."""
        engine = RiskAssessmentEngine()
        res = engine.evaluate_action(
            task_id="task_high_1",
            action="delete inactive session records from cache",
        )

        assert res.risk_level == RiskLevel.HIGH
        assert res.risk_score >= 0.60
        assert res.approval_required is True
        assert "explicit_human_approval" in res.required_controls

    def test_critical_risk_action_classification(self):
        """Verify critical-risk actions (drop database, destroy cluster) classify as CRITICAL with stricter controls."""
        engine = RiskAssessmentEngine()
        res = engine.evaluate_action(
            task_id="task_crit_1",
            action="drop database production_analytics_db",
        )

        assert res.risk_level == RiskLevel.CRITICAL
        assert res.risk_score >= 0.85
        assert res.approval_required is True
        assert "explicit_human_authorization" in res.required_controls
        assert "stricter_approval_controls" in res.required_controls

    def test_approval_requirement_enforcement(self):
        """Verify custom rules can enforce approval requirements for designated actions."""
        policy = RiskPolicyConfig(custom_rules={r"deploy\s+service": RiskLevel.HIGH})
        engine = RiskAssessmentEngine(policy=policy)

        res = engine.evaluate_action("task_rule_1", "deploy service billing-v2")
        assert res.risk_level == RiskLevel.HIGH
        assert res.approval_required is True

    def test_mandatory_authorization_preservation(self):
        """Verify low-risk classification CANNOT bypass mandatory authorization when explicitly required."""
        engine = RiskAssessmentEngine()

        # Action is low-risk (read), but explicit_requires_approval=True
        res = engine.evaluate_action(
            task_id="task_mand_1",
            action="read confidential salary report file",
            explicit_requires_approval=True,
        )

        assert res.risk_level == RiskLevel.LOW
        assert res.approval_required is True
        assert "mandatory_authorization" in res.required_controls

    def test_modified_action_reassessment(self):
        """Verify that modifying an action invalidates prior approval and forces reassessment."""
        engine = RiskAssessmentEngine()

        initial_res = engine.evaluate_action("task_mod_1", "read user records")
        assert initial_res.approval_required is False

        # Action modified to destructive operation
        modified_action = "delete user records permanently"
        was_modified, new_res = engine.reassess_if_modified(initial_res, modified_action)

        assert was_modified is True
        assert new_res.action == modified_action
        assert new_res.risk_level == RiskLevel.HIGH
        assert new_res.approval_required is True
        assert new_res.metadata["modified_from"] == "read user records"

    def test_risk_assessment_persistence(self, tmp_path):
        """Verify risk assessments persist to SQLite and can be retrieved."""
        db_path = str(tmp_path / "risk_test.db")
        storage = SQLiteMessageStorage(db_path=db_path)
        engine = RiskAssessmentEngine(storage=storage)

        res = engine.evaluate_action("task_p_1", "drop table test_runs")

        # Query back from storage
        persisted = storage.get_risk_assessment(res.assessment_id)
        assert persisted is not None
        assert persisted["task_id"] == "task_p_1"
        assert persisted["risk_level"] == "CRITICAL"
        assert persisted["approval_required"] is True


# ===========================================================================
# PART 5 TESTS: Human Review, Audit Trail & Feedback Intelligence
# ===========================================================================

class TestPart5HumanReviewAuditIntelligence:
    """Test suite for Part 5: Reviews, Audit Trail, and Feedback Intelligence."""

    def test_review_submission(self):
        """Verify human review submission records outcome, round, and target metadata."""
        review_mgr = HumanReviewManager()
        rec = review_mgr.submit_review(
            task_id="task_rev_1",
            target_type="plan",
            target_data={"steps": ["step 1", "step 2"]},
            reviewer="LeadArchitect",
            outcome=ReviewOutcome.ACCEPTED,
            feedback="Plan is structured and sound.",
        )

        assert rec.review_id.startswith("rev_")
        assert rec.task_id == "task_rev_1"
        assert rec.reviewer == "LeadArchitect"
        assert rec.outcome == ReviewOutcome.ACCEPTED
        assert rec.review_round == 1

    def test_review_acceptance(self):
        """Verify acceptance records ACCEPTED outcome."""
        review_mgr = HumanReviewManager()
        rec = review_mgr.submit_review("task_acc", "research", {"data": "verified"}, "SRE", ReviewOutcome.ACCEPTED)
        assert rec.outcome == ReviewOutcome.ACCEPTED

    def test_review_rejection(self):
        """Verify review rejection records REJECTED outcome and feedback."""
        review_mgr = HumanReviewManager()
        rec = review_mgr.submit_review(
            "task_rej", "plan", {"tasks": []}, "SecurityOfficer", ReviewOutcome.REJECTED, feedback="Plan lacks rollback"
        )
        assert rec.outcome == ReviewOutcome.REJECTED
        assert "rollback" in rec.feedback

    def test_revision_request(self):
        """Verify revision request records NEEDS_REVISION outcome."""
        review_mgr = HumanReviewManager()
        rec = review_mgr.submit_review(
            "task_rev_req", "decision", {"action": "delete"}, "TechLead", ReviewOutcome.NEEDS_REVISION, feedback="Add backup first"
        )
        assert rec.outcome == ReviewOutcome.NEEDS_REVISION

    def test_multiple_review_rounds(self):
        """Verify successive reviews for the same target auto-increment the review round counter."""
        review_mgr = HumanReviewManager()
        rec1 = review_mgr.submit_review("task_rounds", "plan", {"v": 1}, "Reviewer1", ReviewOutcome.NEEDS_REVISION)
        assert rec1.review_round == 1

        rec2 = review_mgr.submit_review("task_rounds", "plan", {"v": 2}, "Reviewer1", ReviewOutcome.ACCEPTED)
        assert rec2.review_round == 2

    def test_audit_event_persistence(self, tmp_path):
        """Verify audit events are stored sequentially in SQLite without overwriting."""
        db_path = str(tmp_path / "audit_test.db")
        storage = SQLiteMessageStorage(db_path=db_path)
        audit = AuditTrail(storage=storage)

        e1 = audit.record_event("task_aud", AuditEventType.AGENT_ACTION, "Researcher started analysis", "Researcher")
        e2 = audit.record_event("task_aud", AuditEventType.APPROVAL_REQUEST, "Approval requested for drop table", "Supervisor")

        events = audit.query(task_id="task_aud")
        assert len(events) == 2
        assert events[0].description == "Researcher started analysis"
        assert events[1].description == "Approval requested for drop table"

    def test_audit_history_filtering(self, tmp_path):
        """Verify filtering audit events by task, agent, and event_type."""
        db_path = str(tmp_path / "audit_filter.db")
        storage = SQLiteMessageStorage(db_path=db_path)
        audit = AuditTrail(storage=storage)

        audit.record_event("task_A", AuditEventType.AGENT_ACTION, "Act A", "AgentA")
        audit.record_event("task_B", AuditEventType.APPROVAL_DECISION, "Appr B", "Supervisor")
        audit.record_event("task_A", AuditEventType.APPROVAL_DECISION, "Appr A", "Supervisor")

        # Filter by task
        res_a = audit.query(task_id="task_A")
        assert len(res_a) == 2

        # Filter by agent
        res_sup = audit.query(agent_name="Supervisor")
        assert len(res_sup) == 2

        # Filter by event type
        res_dec = audit.query(event_type=AuditEventType.APPROVAL_DECISION)
        assert len(res_dec) == 2

    def test_secret_redaction_in_audit_trail(self, tmp_path):
        """Verify sensitive API keys and tokens are automatically scrubbed in audit records."""
        db_path = str(tmp_path / "audit_redact.db")
        storage = SQLiteMessageStorage(db_path=db_path)
        audit = AuditTrail(storage=storage)

        secret_text = "Connecting to OpenRouter with api_key=sk-abcdef1234567890abcdef1234567890"
        payload = {"auth_token": "Bearer secret_jwt_token_12345", "user": "admin"}

        event = audit.record_event("task_sec", AuditEventType.AGENT_ACTION, secret_text, payload=payload)

        assert "sk-abcdef" not in event.description
        assert "[REDACTED" in event.description
        assert event.payload["auth_token"] == "[REDACTED]"
        assert event.payload["user"] == "admin"

    def test_feedback_pattern_identification(self):
        """Verify FeedbackAnalysisEngine discovers recurring failure patterns from historical reviews."""
        review_mgr = HumanReviewManager()
        # Seed multiple reviews with missing parameters
        review_mgr.submit_review("t1", "plan", {}, "Reviewer", ReviewOutcome.REJECTED, "Plan lacks missing parameters")
        review_mgr.submit_review("t2", "plan", {}, "Reviewer", ReviewOutcome.NEEDS_REVISION, "Missing date range")
        review_mgr.submit_review("t3", "research", {}, "Reviewer", ReviewOutcome.ACCEPTED, "Good findings")

        engine = FeedbackAnalysisEngine(review_manager=review_mgr)
        summary = engine.generate_summary()

        assert summary.total_reviews == 3
        assert "missing_information" in summary.recurring_issue_categories
        assert any("rejections" in p.lower() or "revision" in p.lower() for p in summary.observed_patterns)

    def test_feedback_summary_generation(self):
        """Verify structured feedback summary separates observed facts from advisory recommendations."""
        review_mgr = HumanReviewManager()
        review_mgr.submit_review("t_s1", "plan", {}, "Admin", ReviewOutcome.REJECTED, "Dangerous destructive action")

        engine = FeedbackAnalysisEngine(review_manager=review_mgr)
        summary = engine.generate_summary(title="Q4 Agent Quality Review")

        assert summary.summary_title == "Q4 Agent Quality Review"
        assert len(summary.observed_patterns) > 0
        assert len(summary.actionable_recommendations) > 0
        # Recommendations must be advisory
        assert all("advisory" in rec.lower() for rec in summary.actionable_recommendations)

    def test_unauthorized_policy_change_rejection(self):
        """Verify policy-change proposals strictly require human authorization before application."""
        engine = FeedbackAnalysisEngine()
        prop = engine.create_proposal(
            proposed_change="Mandate dry-run for all delete operations",
            target_policy="ExecutorPolicy",
            rationale="Identified multiple review rejections due to lack of rollback steps",
        )

        assert prop.status == "PROPOSED"

        # 1. Attempt to apply without authorization -> REJECTED
        ok, msg, _ = engine.apply_proposal(prop.proposal_id, authorized_by="")
        assert ok is False
        assert "unauthorized" in msg.lower()
        assert prop.status == "PROPOSED"

        # 2. Authorized by human administrator -> APPLIED
        ok, msg, applied_prop = engine.apply_proposal(prop.proposal_id, authorized_by="SRE_Lead")
        assert ok is True
        assert applied_prop.status == "APPLIED"
        assert applied_prop.authorized_by == "SRE_Lead"
        assert applied_prop.authorized_at is not None


# ===========================================================================
# PART 6 TESTS: Complete Production HITL Integration
# ===========================================================================

class TestPart6ProductionHITLIntegration:
    """Test suite for Part 6: Unified Human-in-the-Loop Integration."""

    def test_complete_18_step_hitl_workflow(self, tmp_path):
        """Simulate the complete 18-step production HITL workflow:
        Step 1: User submits task.
        Step 2: Supervisor evaluates request.
        Step 3: RiskAssessmentEngine evaluates proposed actions.
        Step 4: Supervisor delegates research to Researcher.
        Step 5: Researcher requests clarification.
        Step 6: Workflow pauses & persists.
        Step 7: Human supplies clarification.
        Step 8: Workflow resumes without repeating completed work.
        Step 9: Planner generates execution plan.
        Step 10: HumanReviewManager allows plan review.
        Step 11: RiskAssessmentEngine evaluates planned actions.
        Step 12: HumanApprovalManager requests approval.
        Step 13: Human approves & issues token.
        Step 14: Executor verifies token and executes.
        Step 15: HumanReviewManager evaluates execution results.
        Step 16: Supervisor produces final response.
        Step 17: Audit system records complete history.
        Step 18: FeedbackAnalysisEngine analyzes accumulated review data.
        """
        db_path = str(tmp_path / "full_hitl.db")
        storage = SQLiteMessageStorage(db_path=db_path)
        hitl = HumanInTheLoopEngine(storage=storage)

        mock_provider = MockLLMProvider("Verified report")
        executor = ExecutorAgent(
            provider=mock_provider,
            approval_manager=hitl.approvals,
            risk_engine=hitl.risk,
            audit_trail=hitl.audit,
        )
        orchestrator = WorkflowOrchestrator(
            provider=mock_provider,
            executor=executor,
            hitl_engine=hitl,
        )

        wf_id = "wf_prod_18"
        context = TaskContext(workflow_id=wf_id, task_id=wf_id, user_request="Prune old inactive users")

        # Step 1-3: Risk Assessment of initial request
        init_risk = hitl.assess_action(wf_id, context.user_request)
        assert init_risk is not None

        # Step 4-6: Researcher clarification & pause
        clarify = hitl.feedback.request_clarification(
            task_id=wf_id,
            correlation_id=wf_id,
            requesting_agent="Researcher",
            question="Which inactivity cutoff date should be used?",
            context=context,
        )
        assert context.status == WorkflowStatus.PAUSED

        # Step 7-8: Human provides clarification & workflow resumes
        hitl.feedback.submit_feedback(clarify.interaction_id, "Inactivity threshold is 365 days", context=context)
        context.status = WorkflowStatus.INITIALIZED
        context.research_findings = "Identified 120 users inactive for >365 days."

        # Step 9-10: Planner generates plan & Human reviews it
        context.execution_plan = {
            "tasks": [
                {
                    "id": 1,
                    "description": "delete inactive users older than 365 days",
                    "requires_approval": True,
                }
            ]
        }
        plan_rev = hitl.submit_review(
            task_id=wf_id,
            target_type="plan",
            target_data=context.execution_plan,
            reviewer="LeadArchitect",
            outcome=ReviewOutcome.ACCEPTED,
            feedback="Plan approved for execution with safety threshold",
            context=context,
        )
        assert plan_rev.outcome == ReviewOutcome.ACCEPTED

        # Step 11-13: Risk Assessment & Human Approval
        action_desc = "delete inactive users older than 365 days"
        plan_risk = hitl.assess_action(wf_id, action_desc)
        assert plan_risk.approval_required is True

        appr_req = hitl.approvals.request_approval(
            task_id=wf_id,
            correlation_id=wf_id,
            requesting_agent="Supervisor",
            requested_action=action_desc,
            context=context,
        )
        appr_rec = hitl.approvals.approve(appr_req.interaction_id, feedback="Approved by SRE on call")
        assert appr_rec.token is not None

        # Step 14: Executor verifies token and executes
        with patch("agents.executor.execute_task", return_value="120 inactive users pruned"):
            res = executor.execute(
                {
                    "description": action_desc,
                    "task_id": wf_id,
                    "requires_approval": True,
                    "approval_token": appr_rec.token,
                },
                expected_task_id=wf_id,
            )
        assert res["status"] == "completed"
        context.execution_results.append(res)

        # Step 15: Human reviews execution result
        exec_rev = hitl.submit_review(
            task_id=wf_id,
            target_type="execution",
            target_data=res,
            reviewer="SRE_Lead",
            outcome=ReviewOutcome.ACCEPTED,
            feedback="Execution clean and matches expected threshold",
            context=context,
        )
        assert exec_rev.outcome == ReviewOutcome.ACCEPTED

        # Step 16: Final synthesis
        context.final_response = "Pruning completed successfully for 120 users."
        context.status = WorkflowStatus.COMPLETED

        # Step 17: Audit history verified
        audit_events = hitl.query_audit_trail(task_id=wf_id)
        assert len(audit_events) >= 3

        # Step 18: Feedback intelligence analysis
        summary = hitl.analyze_feedback()
        assert summary.total_reviews >= 2

    def test_invalid_approval_token_rejection(self):
        """Verify that presentation of an invalid or forged approval token is rejected at boundary."""
        hitl = HumanInTheLoopEngine()
        executor = ExecutorAgent(approval_manager=hitl.approvals)

        res = executor.execute(
            {
                "description": "delete customer database",
                "task_id": "task_forged",
                "requires_approval": True,
                "approval_token": "tok_forged_invalid_123",
            },
            expected_task_id="task_forged",
        )

        assert res["status"] == "failed"
        assert "not recognized" in res["error"].lower()

    def test_modified_action_approval_mismatch_rejection(self):
        """Verify that approving action A does NOT authorize execution of modified action B."""
        hitl = HumanInTheLoopEngine()
        executor = ExecutorAgent(approval_manager=hitl.approvals)

        # Approve specific action
        req = hitl.approvals.request_approval("task_mismatch", "corr_1", "Supervisor", "delete staging logs")
        rec = hitl.approvals.approve(req.interaction_id)

        # Attempt to execute modified action with same token
        res = executor.execute(
            {
                "description": "delete production database customer records",
                "task_id": "task_mismatch",
                "requires_approval": True,
                "approval_token": rec.token,
            },
            expected_task_id="task_mismatch",
        )

        assert res["status"] == "failed"
        assert "action mismatch" in res["error"].lower()

    def test_workflow_restart_recovery_preserves_paused_state(self, tmp_path):
        """Verify that a paused workflow state survives process restart and can be resumed."""
        db_path = str(tmp_path / "restart_rec.db")
        storage = SQLiteMessageStorage(db_path=db_path)

        hitl1 = HumanInTheLoopEngine(storage=storage)
        ctx = TaskContext(workflow_id="wf_restart", task_id="task_res", user_request="Long migration")
        ctx.research_findings = "Schema validated."

        # Pause workflow
        req = hitl1.feedback.request_clarification(
            "task_res", "wf_restart", "Researcher", "Target cloud region?", context=ctx
        )
        assert ctx.status == WorkflowStatus.PAUSED

        # Simulate cold restart: instantiate new engine from same SQLite database
        hitl2 = HumanInTheLoopEngine(storage=storage)

        # Human provides feedback to recovered engine
        ok, _, resolved_req = hitl2.feedback.submit_feedback(req.interaction_id, "us-east-1", task_id="task_res")
        assert ok is True
        assert resolved_req.status == InteractionStatus.RESOLVED

        # Resume workflow from storage snapshot
        mock_orch = MagicMock()
        mock_orch.run_communication_workflow.side_effect = lambda user_request, task_id, **kw: TaskContext(
            workflow_id=task_id, task_id=task_id, user_request=user_request, research_findings="Schema validated."
        )
        mock_orch.run.side_effect = lambda user_request, task_id, **kw: TaskContext(
            workflow_id=task_id, task_id=task_id, user_request=user_request, research_findings="Schema validated."
        )

        resumed = hitl2.feedback.resume_workflow("wf_restart", orchestrator=mock_orch, feedback_response="us-east-1")
        assert resumed.research_findings == "Schema validated."
        assert "us-east-1" in resumed.user_request

    def test_backward_compatibility_when_hitl_disabled(self):
        """Verify that standard agents and workflows run unaffected when HITL engine is not configured."""
        mock_provider = MockLLMProvider("Regular response")
        executor = ExecutorAgent(provider=mock_provider)

        with patch("agents.executor.execute_task", return_value="Calculated: 100"):
            res = executor.execute("calculate total count")
            assert res["status"] == "completed"
            assert "100" in res["result"]
