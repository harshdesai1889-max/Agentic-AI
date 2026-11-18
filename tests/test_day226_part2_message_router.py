"""Test Suite for Day 226 Part 2: Agent Message Router and Controlled Delivery System.

Validates:
1. Supervisor-to-Worker routing.
2. Worker-to-Worker routing.
3. Worker-to-Supervisor routing.
4. Unknown recipient rejection.
5. Invalid message rejection.
6. Successful delivery tracking.
7. Failed delivery tracking.
8. Eligible retry behavior and limits.
9. Duplicate delivery protection (idempotency).
10. Task and correlation ID preservation.
11. Message history preservation.
12. Backward compatibility with existing AgentMessage objects.
13. End-to-end workflow execution (Supervisor -> Researcher -> Planner -> Supervisor -> Executor -> Supervisor).
"""

import copy
import pytest
from unittest.mock import MagicMock

from day223_agents.core.orchestrator import (
    AgentMessage,
    MessageType,
    TaskContext,
    WorkflowOrchestrator,
    WorkflowStatus,
)
from day223_agents.core.failure_recovery import WorkerRegistry
from day223_agents.core.message_router import (
    MessageRouter,
    DeliveryReceipt,
    DeliveryStatus,
    RoutingRules,
)
from day223_agents.agents.supervisor import (
    SupervisorAgent,
    DecisionType,
    SupervisorDecision,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def populated_worker_registry():
    """Worker registry with Researcher, Planner, and Executor registered."""
    registry = WorkerRegistry()
    mock_researcher = MagicMock()
    mock_researcher.name = "Researcher"
    mock_planner = MagicMock()
    mock_planner.name = "Planner"
    mock_executor = MagicMock()
    mock_executor.name = "Executor"

    registry.register(mock_researcher, capabilities=["research", "analysis"], priority=10, name="Researcher")
    registry.register(mock_planner, capabilities=["planning", "task_decomposition"], priority=10, name="Planner")
    registry.register(mock_executor, capabilities=["execution", "action"], priority=10, name="Executor")
    return registry


@pytest.fixture
def router(populated_worker_registry):
    """Instantiated MessageRouter with 0-delay backoff for speedy tests."""
    return MessageRouter(
        worker_registry=populated_worker_registry,
        max_retries=2,
        backoff_factor=0.0,
    )


# ===========================================================================
# 1. Supervisor-to-Worker Routing
# ===========================================================================

def test_1_supervisor_to_worker_routing(router):
    """Verify routing of TASK_ASSIGNMENT from Supervisor to registered Worker."""
    msg = AgentMessage(
        task_id="task_s2w_100",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Investigate distributed consensus protocols",
        output_data=None,
        correlation_id="wf_root_100",
    )

    receipt = router.route(msg)
    assert receipt.status == DeliveryStatus.DELIVERED
    assert receipt.sender == "Supervisor"
    assert receipt.receiver == "Researcher"
    assert receipt.attempts == 1
    assert receipt.error is None
    assert receipt.delivery_timestamp is not None
    assert receipt.correlation_id == "wf_root_100"


# ===========================================================================
# 2. Worker-to-Worker Routing
# ===========================================================================

def test_2_worker_to_worker_routing(router):
    """Verify routing of STAGE_HANDOFF from one Worker to another Worker."""
    msg = AgentMessage(
        task_id="task_w2w_200",
        agent_name="Researcher",
        sender="Researcher",
        receiver="Planner",
        stage="planning",
        status="success",
        message_type=MessageType.STAGE_HANDOFF,
        input_data={"findings": "Raft is simpler than Paxos"},
        output_data={"handoff_ready": True},
        correlation_id="wf_root_200",
    )

    receipt = router.route(msg)
    assert receipt.status == DeliveryStatus.DELIVERED
    assert receipt.sender == "Researcher"
    assert receipt.receiver == "Planner"
    assert receipt.message_type == MessageType.STAGE_HANDOFF.value


# ===========================================================================
# 3. Worker-to-Supervisor Routing
# ===========================================================================

def test_3_worker_to_supervisor_routing(router):
    """Verify routing of RESULT_RESPONSE from Worker back to Supervisor."""
    msg = AgentMessage(
        task_id="task_w2s_300",
        agent_name="Executor",
        sender="Executor",
        receiver="Supervisor",
        stage="execution",
        status="success",
        message_type=MessageType.RESULT_RESPONSE,
        input_data={"command": "run_cluster"},
        output_data={"cluster_status": "healthy"},
        correlation_id="wf_root_300",
    )

    receipt = router.route(msg)
    assert receipt.status == DeliveryStatus.DELIVERED
    assert receipt.sender == "Executor"
    assert receipt.receiver == "Supervisor"
    assert receipt.message_type in (MessageType.RESULT_RESPONSE.value, MessageType.TASK_RESULT.value)


# ===========================================================================
# 4. Unknown Recipient Rejection
# ===========================================================================

def test_4_unknown_recipient_rejection(router):
    """Verify that MessageRouter rejects delivery when recipient is not registered."""
    msg = AgentMessage(
        task_id="task_bad_recip_400",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="NonExistentAgent",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Do work",
        output_data=None,
    )

    receipt = router.route(msg)
    assert receipt.status == DeliveryStatus.FAILED
    assert "not registered in WorkerRegistry" in (receipt.error or "") or "is not a registered worker" in (receipt.error or "")
    assert receipt.attempts == 1


# ===========================================================================
# 5. Invalid Message Rejection
# ===========================================================================

def test_5_invalid_message_rejection(router):
    """Verify that messages with missing required fields are rejected."""
    msg_missing_id = AgentMessage(
        task_id="task_inv_500",
        agent_name="Planner",
        sender="Planner",
        receiver="Executor",
        stage="planning",
        status="success",
        input_data={},
        output_data={},
    )
    msg_missing_id.message_id = ""

    receipt = router.route(msg_missing_id)
    assert receipt.status == DeliveryStatus.FAILED
    assert "message_id cannot be empty" in (receipt.error or "")


# ===========================================================================
# 6. Successful Delivery Tracking and Latency
# ===========================================================================

def test_6_successful_delivery_tracking(router):
    """Verify delivery tracking, timestamp, and latency measurement."""
    msg = AgentMessage(
        task_id="task_track_600",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Planner",
        stage="planning",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Draft execution plan",
        output_data=None,
    )

    receipt = router.route(msg)
    assert receipt.status == DeliveryStatus.DELIVERED
    assert receipt.latency_seconds >= 0.0
    assert receipt.delivery_timestamp is not None

    stored_receipt = router.get_receipt(msg.message_id)
    assert stored_receipt is not None
    assert stored_receipt.message_id == msg.message_id


# ===========================================================================
# 7. Failed Delivery Tracking
# ===========================================================================

def test_7_failed_delivery_tracking(router):
    """Verify capture and tracking of delivery exceptions."""
    failing_handler = MagicMock(side_effect=RuntimeError("Worker socket connection refused"))
    router.register_recipient_handler("CrashingWorker", failing_handler)

    msg = AgentMessage(
        task_id="task_fail_700",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="CrashingWorker",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Process task",
        output_data=None,
    )

    # Allow CrashingWorker as valid system recipient for handler test
    RoutingRules.SYSTEM_ROLES.add("CrashingWorker")
    try:
        receipt = router.route(msg)
        assert receipt.status == DeliveryStatus.FAILED
        assert "Worker socket connection refused" in (receipt.error or "")
        failed_receipts = router.get_receipts_by_status(DeliveryStatus.FAILED)
        assert any(r.message_id == msg.message_id for r in failed_receipts)
    finally:
        RoutingRules.SYSTEM_ROLES.remove("CrashingWorker")


# ===========================================================================
# 8. Eligible Retry Behavior and Limits
# ===========================================================================

def test_8_eligible_retry_behavior_and_limits(router):
    """Verify that transient delivery failures are retried up to max_retries."""
    attempts_seen = []

    def flaky_handler(m):
        attempts_seen.append(len(attempts_seen) + 1)
        if len(attempts_seen) < 3:
            raise ConnectionError("Temporary connection reset by peer")
        return True

    router.register_recipient_handler("FlakyWorker", flaky_handler)
    RoutingRules.SYSTEM_ROLES.add("FlakyWorker")

    try:
        msg = AgentMessage(
            task_id="task_retry_800",
            agent_name="Supervisor",
            sender="Supervisor",
            receiver="FlakyWorker",
            stage="execution",
            status="waiting",
            message_type=MessageType.TASK_ASSIGNMENT,
            input_data="Run flaky pipeline",
            output_data=None,
        )

        receipt = router.route(msg)
        assert receipt.status == DeliveryStatus.DELIVERED
        assert receipt.attempts == 3
        assert len(attempts_seen) == 3
    finally:
        RoutingRules.SYSTEM_ROLES.remove("FlakyWorker")


# ===========================================================================
# 9. Duplicate Delivery Protection (Idempotency)
# ===========================================================================

def test_9_duplicate_delivery_protection(router):
    """Verify that re-sending an already delivered message does not duplicate delivery."""
    call_count = [0]

    def count_handler(m):
        call_count[0] += 1
        return True

    router.register_recipient_handler("IdempotentWorker", count_handler)
    RoutingRules.SYSTEM_ROLES.add("IdempotentWorker")

    try:
        msg = AgentMessage(
            task_id="task_dup_900",
            agent_name="Supervisor",
            sender="Supervisor",
            receiver="IdempotentWorker",
            stage="execution",
            status="waiting",
            message_type=MessageType.TASK_ASSIGNMENT,
            input_data="Send payment notice",
            output_data=None,
        )

        r1 = router.route(msg)
        assert r1.status == DeliveryStatus.DELIVERED
        assert call_count[0] == 1

        # Route the identical message again
        r2 = router.route(msg)
        assert r2.status == DeliveryStatus.DELIVERED
        assert r2.metadata.get("duplicate_suppressed") is True
        # Underlying handler was not invoked a second time
        assert call_count[0] == 1
    finally:
        RoutingRules.SYSTEM_ROLES.remove("IdempotentWorker")


# ===========================================================================
# 10. Task and Correlation ID Preservation
# ===========================================================================

def test_10_task_and_correlation_id_preservation(router):
    """Verify that task_id and correlation_id are strictly preserved across delivery receipts."""
    parent_corr = "parent_workflow_xyz_1000"
    task_sub = "child_task_1000"

    msg = AgentMessage(
        task_id=task_sub,
        agent_name="Planner",
        sender="Planner",
        receiver="Executor",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data={"step": "provision"},
        output_data=None,
        correlation_id=parent_corr,
    )

    receipt = router.route(msg)
    assert receipt.task_id == task_sub
    assert receipt.correlation_id == parent_corr

    # Lookup by correlation
    correlated_messages = router.get_messages_by_correlation(parent_corr)
    assert len(correlated_messages) >= 1
    assert correlated_messages[0].task_id == task_sub


# ===========================================================================
# 11. Message History Preservation
# ===========================================================================

def test_11_message_history_preservation(router):
    """Verify that all routed messages are preserved in router history and TaskContext."""
    context = TaskContext(workflow_id="wf_hist_1100", user_request="History preservation test")

    m1 = AgentMessage(
        task_id="task_h1",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="query",
        output_data=None,
    )
    m2 = AgentMessage(
        task_id="task_h2",
        agent_name="Researcher",
        sender="Researcher",
        receiver="Planner",
        stage="planning",
        status="success",
        message_type=MessageType.STAGE_HANDOFF,
        input_data="query",
        output_data="findings",
    )

    router.route(m1, context=context)
    router.route(m2, context=context)

    # Router history
    assert len(router._message_history) == 2
    # Context message history
    assert len(context.messages) == 2
    assert "delivery_receipt" in context.messages[0].metadata
    assert context.messages[0].metadata["delivery_receipt"]["status"] == "delivered"


# ===========================================================================
# 12. Backward Compatibility with Existing AgentMessage Objects
# ===========================================================================

def test_12_backward_compatibility(router):
    """Verify that legacy AgentMessage construction routes cleanly without error."""
    legacy_msg = AgentMessage(
        task_id="legacy_task_1200",
        agent_name="Researcher",
        stage="research",
        status="success",
        input_data="legacy query",
        output_data="legacy result",
    )
    # Give it a receiver to satisfy routing
    legacy_msg.receiver = "Planner"

    receipt = router.route(legacy_msg)
    assert receipt.status == DeliveryStatus.DELIVERED
    assert receipt.sender == "Researcher"
    assert receipt.receiver == "Planner"


# ===========================================================================
# 13. End-to-End Workflow Integration Demonstration
# (Supervisor -> Researcher -> Planner -> Supervisor -> Executor -> Supervisor)
# ===========================================================================

def test_13_workflow_integration_demonstration(populated_worker_registry):
    """Demonstrate the workflow:
    Supervisor -> Researcher -> Planner -> Supervisor -> Executor -> Supervisor
    using MessageRouter with existing specialized agents.
    """
    mock_researcher = populated_worker_registry.get_worker("Researcher")
    mock_researcher.research.return_value = {
        "task_id": "test_e2e_1300",
        "status": "completed",
        "findings": "Verified high performance characteristics.",
    }

    mock_planner = populated_worker_registry.get_worker("Planner")
    mock_planner.plan.return_value = {
        "task_id": "test_e2e_1300",
        "status": "completed",
        "tasks": [{"id": 1, "description": "Execute load test benchmark", "authorized": True}],
    }

    mock_executor = populated_worker_registry.get_worker("Executor")
    mock_executor.execute.return_value = {
        "task_id": "test_e2e_1300_task_1",
        "status": "completed",
        "result": "Load test completed with 0 errors.",
    }

    mock_supervisor = MagicMock(spec=SupervisorAgent)
    mock_supervisor.name = "SupervisorAgent"
    mock_supervisor.max_delegation_steps = 6

    # Sequential Supervisor decisions simulating:
    # 1. DELEGATE -> Researcher
    # 2. DELEGATE -> Planner
    # 3. DELEGATE -> Executor
    # 4. COMPLETE -> Supervisor
    decisions = [
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Researcher",
            capability_required="research",
            instruction="Research performance targets",
        ),
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Planner",
            capability_required="planning",
            instruction="Create execution plan for benchmark",
        ),
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Executor",
            capability_required="execution",
            instruction="Run benchmark execution",
        ),
        SupervisorDecision(
            decision_type=DecisionType.COMPLETE,
            final_response="Multi-agent benchmarking workflow fully accomplished.",
        ),
    ]
    mock_supervisor.decide.side_effect = decisions

    # Instantiate orchestrator with MessageRouter
    custom_router = MessageRouter(worker_registry=populated_worker_registry)
    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        supervisor=mock_supervisor,
        worker_registry=populated_worker_registry,
        message_router=custom_router,
    )

    context = orchestrator.run_supervised("Perform full system benchmark")

    assert context.status == WorkflowStatus.COMPLETED
    assert context.final_response is not None

    # Verify that MessageRouter routed each delegation step from Supervisor to Worker
    routed_assignments = [
        r for r in custom_router._delivery_receipts
        if r.sender in ("Supervisor", "SupervisorAgent")
    ]
    assert len(routed_assignments) >= 3

    receivers = [r.receiver for r in routed_assignments]
    assert "Researcher" in receivers
    assert "Planner" in receivers
    assert "Executor" in receivers

    for r in routed_assignments:
        assert r.status == DeliveryStatus.DELIVERED
        assert r.correlation_id == context.task_id
