"""Test Suite for Day 226 Part 5: Message Reliability, Acknowledgements, Dead-Letter Queues and Delivery Guarantees.

Validates:
1. Successful acknowledgement tracking (AckStatus.ACKNOWLEDGED).
2. Missing acknowledgement handling (UNACKNOWLEDGED default / timeout).
3. Rejected acknowledgement handling (AckStatus.REJECTED).
4. Distinction between message delivery status and execution processing outcome.
5. At-most-once delivery guarantee (retries disabled).
6. At-least-once delivery guarantee (retries enabled on transient failures).
7. Retry exhaustion leading to DeadLetterQueue escalation.
8. DLQ insertion preserving original message, task ID, correlation ID, and metadata.
9. DLQ persistence across restarts in SQLite storage.
10. DLQ message querying and inspection.
11. Authorized replay of eligible messages.
12. Unauthorized replay rejection.
13. Replay history preservation and incrementing replay_count.
14. Duplicate side-effect protection (refusal to replay non-idempotent operations).
15. Uncertain execution handling after crash or disconnection.
16. Reliability metrics calculation (delivered, acked, failed, retried, DLQ, average latency).
17. Full multi-agent workflow compatibility with existing specialized agents.
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
from day223_agents.core.message_queue import (
    MessageQueue,
    MessagePriority,
    QueueMessageState,
    QueueItem,
)
from day223_agents.core.persistent_storage import SQLiteMessageStorage
from day223_agents.core.reliability import (
    AckStatus,
    DeliveryGuarantee,
    MessageAck,
    DLQEntry,
    ReliabilityMetrics,
    DeadLetterQueue,
    ReliabilityCoordinator,
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
def temp_db(tmp_path):
    """Temporary SQLite database file path."""
    return str(tmp_path / "reliable_messages.db")


@pytest.fixture
def storage(temp_db):
    """Instantiated SQLite storage."""
    return SQLiteMessageStorage(db_path=temp_db)


@pytest.fixture
def worker_registry():
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
def coordinator(storage):
    """ReliabilityCoordinator with SQLite-backed DLQ."""
    dlq = DeadLetterQueue(storage=storage)
    metrics = ReliabilityMetrics()
    return ReliabilityCoordinator(
        default_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        dlq=dlq,
        metrics=metrics,
    )


@pytest.fixture
def reliable_queue(worker_registry, storage, coordinator):
    """MessageQueue wired with ReliabilityCoordinator, Router, and SQLite."""
    router = MessageRouter(
        worker_registry=worker_registry,
        max_retries=2,
        backoff_factor=0.0,
    )
    mq = MessageQueue(
        router=router,
        max_retries=2,
        storage=storage,
        auto_recover=True,
        reliability_coordinator=coordinator,
    )
    router.attach_queue(mq)
    return mq


# ===========================================================================
# 1. Successful Acknowledgement
# ===========================================================================

def test_1_successful_acknowledgement(coordinator):
    """Verify that recipient acknowledgement records ACKNOWLEDGED status and marks idempotency."""
    msg = AgentMessage(
        task_id="task_ack_100",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Collect metrics",
        output_data=None,
    )

    ack = coordinator.record_acknowledgement(
        message=msg,
        recipient="Researcher",
        status=AckStatus.ACKNOWLEDGED,
        reason="Operation processed successfully",
    )

    assert ack.status == AckStatus.ACKNOWLEDGED
    assert ack.message_id == msg.message_id
    assert coordinator.is_operation_completed(msg) is True
    assert coordinator.metrics.acknowledged_messages == 1


# ===========================================================================
# 2. Missing Acknowledgement Handling
# ===========================================================================

def test_2_missing_acknowledgement(coordinator):
    """Verify that unacknowledged messages return None or UNACKNOWLEDGED status safely."""
    msg = AgentMessage(
        task_id="task_unack_200",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Planner",
        stage="planning",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Plan step",
        output_data=None,
    )

    # Has not been acknowledged yet
    assert coordinator.get_acknowledgement(msg.message_id) is None
    assert coordinator.is_operation_completed(msg) is False


# ===========================================================================
# 3. Rejected Acknowledgement
# ===========================================================================

def test_3_rejected_acknowledgement(coordinator):
    """Verify that rejected acknowledgement updates processing failure metrics and does not complete idempotency."""
    msg = AgentMessage(
        task_id="task_rej_300",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Executor",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Invalid payload",
        output_data=None,
    )

    ack = coordinator.record_acknowledgement(
        message=msg,
        recipient="Executor",
        status=AckStatus.REJECTED,
        reason="Input validation contract violation",
    )

    assert ack.status == AckStatus.REJECTED
    assert coordinator.is_operation_completed(msg) is False
    assert coordinator.metrics.processing_failures == 1


# ===========================================================================
# 4. Delivery Status vs Processing Outcome
# ===========================================================================

def test_4_delivery_vs_processing_status(coordinator):
    """Verify distinction between delivery receipt (network/router) and recipient execution acknowledgement."""
    msg = AgentMessage(
        task_id="task_dist_400",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Executor",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Step action",
        output_data=None,
    )

    # 1. Delivery succeeded at network/router layer
    receipt = DeliveryReceipt(
        message_id=msg.message_id,
        task_id=msg.task_id,
        sender="Supervisor",
        receiver="Executor",
        message_type=str(msg.message_type),
        status=DeliveryStatus.DELIVERED,
    )
    assert receipt.status == DeliveryStatus.DELIVERED

    # 2. But agent processing fails internally
    ack = coordinator.record_acknowledgement(
        message=msg,
        recipient="Executor",
        status=AckStatus.REJECTED,
        reason="Execution exception thrown inside worker",
    )
    assert ack.status == AckStatus.REJECTED
    assert coordinator.is_operation_completed(msg) is False


# ===========================================================================
# 5. At-Most-Once Delivery Guarantee
# ===========================================================================

def test_5_at_most_once_behavior(worker_registry, storage):
    """Verify that AT_MOST_ONCE mode disables retries and fails directly on first error."""
    # Custom coordinator in AT_MOST_ONCE mode
    coord = ReliabilityCoordinator(
        default_guarantee=DeliveryGuarantee.AT_MOST_ONCE,
        dlq=DeadLetterQueue(storage=storage),
    )
    attempts = [0]

    def crash_handler(m):
        attempts[0] += 1
        raise ConnectionError("Transient drop")

    router = MessageRouter(worker_registry=worker_registry, max_retries=3, backoff_factor=0.0)
    router.register_recipient_handler("DropWorker", crash_handler)
    RoutingRules.SYSTEM_ROLES.add("DropWorker")

    try:
        q = MessageQueue(
            router=router,
            max_retries=3,
            storage=storage,
            reliability_coordinator=coord,
        )
        msg = AgentMessage(task_id="t_amo", agent_name="A", sender="Supervisor", receiver="DropWorker", stage="e", status="w", input_data="data", output_data=None)
        q.enqueue(msg)

        receipt = q.process_next()
        assert receipt.status == DeliveryStatus.FAILED
        # Exactly 1 attempt made: zero retries under at-most-once
        assert attempts[0] == 1
        assert q.get_state(msg.message_id) == QueueMessageState.FAILED
    finally:
        RoutingRules.SYSTEM_ROLES.remove("DropWorker")


# ===========================================================================
# 6. At-Least-Once Delivery Guarantee
# ===========================================================================

def test_6_at_least_once_behavior(worker_registry, storage):
    """Verify that AT_LEAST_ONCE mode retries transient failures until success."""
    coord = ReliabilityCoordinator(
        default_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        dlq=DeadLetterQueue(storage=storage),
    )
    attempts = [0]

    def flaky_handler(m):
        attempts[0] += 1
        if attempts[0] < 3:
            raise ConnectionError("Temporary connection drop")
        return True

    router = MessageRouter(worker_registry=worker_registry, max_retries=3, backoff_factor=0.0)
    router.register_recipient_handler("FlakyWorker", flaky_handler)
    RoutingRules.SYSTEM_ROLES.add("FlakyWorker")

    try:
        q = MessageQueue(
            router=router,
            max_retries=3,
            storage=storage,
            reliability_coordinator=coord,
        )
        msg = AgentMessage(task_id="t_alo", agent_name="A", sender="Supervisor", receiver="FlakyWorker", stage="e", status="w", input_data="data", output_data=None)
        q.enqueue(msg)

        # Process delivery with retry until success under at-least-once guarantee
        receipt = q.process_next()
        assert receipt.status == DeliveryStatus.DELIVERED
        assert attempts[0] == 3
        assert q.get_state(msg.message_id) == QueueMessageState.COMPLETED
    finally:
        RoutingRules.SYSTEM_ROLES.remove("FlakyWorker")


# ===========================================================================
# 7. Retry Exhaustion and DLQ Escalation
# ===========================================================================

def test_7_retry_exhaustion_and_dlq_insertion(reliable_queue):
    """Verify that exhausting retries escalates the message into the DeadLetterQueue."""
    def fatal_handler(m):
        raise ConnectionResetError("Connection reset by peer")

    reliable_queue.router.register_recipient_handler("FatalWorker", fatal_handler)
    RoutingRules.SYSTEM_ROLES.add("FatalWorker")

    try:
        msg = AgentMessage(
            task_id="t_exhaust",
            agent_name="Supervisor",
            sender="Supervisor",
            receiver="FatalWorker",
            stage="execution",
            status="waiting",
            message_type=MessageType.TASK_ASSIGNMENT,
            input_data="Exhaust retries",
            output_data=None,
        )
        reliable_queue.enqueue(msg, priority=MessagePriority.NORMAL)

        # Drain executes initial attempt + 2 retries = 3 attempts total, then escalates to DLQ
        receipts = reliable_queue.drain()
        assert len(receipts) == 1
        assert receipts[0].status == DeliveryStatus.FAILED
        assert receipts[0].attempts == 3

        # Check DLQ entry
        dlq = reliable_queue.reliability.dlq
        assert dlq.size() == 1
        entry = dlq.get(f"dlq_{msg.message_id}")
        assert entry is not None
        assert entry.message.message_id == msg.message_id
        assert entry.retry_count == 3
        assert "Connection reset by peer" in entry.reason
    finally:
        RoutingRules.SYSTEM_ROLES.remove("FatalWorker")


# ===========================================================================
# 8. DLQ Persistence in SQLite
# ===========================================================================

def test_8_dlq_persistence(reliable_queue, storage):
    """Verify that DLQ entries are persisted into the dead_letter_queue SQLite table."""
    msg = AgentMessage(
        task_id="t_dlq_pers",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Persist DLQ",
        output_data=None,
    )

    entry = reliable_queue.reliability.route_to_dlq(
        message=msg,
        reason="Unresolvable dependency",
        retry_count=3,
        failure_details={"code": 500},
    )

    persisted_entries = storage.get_all_dlq_entries()
    assert len(persisted_entries) >= 1
    p_entry = [e for e in persisted_entries if e.dlq_id == entry.dlq_id][0]
    assert p_entry.dlq_id == entry.dlq_id
    assert p_entry.reason == "Unresolvable dependency"
    assert p_entry.retry_count == 3


# ===========================================================================
# 9. DLQ Persistence After Restart
# ===========================================================================

def test_9_dlq_persistence_after_restart(temp_db):
    """Verify that DLQ entries reload properly from SQLite upon application restart."""
    storage1 = SQLiteMessageStorage(db_path=temp_db)
    dlq1 = DeadLetterQueue(storage=storage1)

    m = AgentMessage(task_id="t_rest", agent_name="A", sender="Supervisor", receiver="Planner", stage="p", status="w", input_data="data", output_data=None)
    dlq1.push(m, reason="Hard crash during execution", retry_count=2)
    assert dlq1.size() == 1

    # Restart: instantiate second storage and DLQ pointing to same database
    storage2 = SQLiteMessageStorage(db_path=temp_db)
    dlq2 = DeadLetterQueue(storage=storage2)

    assert dlq2.size() == 1
    recovered_entry = dlq2.get(f"dlq_{m.message_id}")
    assert recovered_entry is not None
    assert recovered_entry.message.task_id == "t_rest"
    assert recovered_entry.reason == "Hard crash during execution"


# ===========================================================================
# 10. DLQ Message Inspection
# ===========================================================================

def test_10_dlq_inspection(coordinator):
    """Verify querying and inspecting messages in the DeadLetterQueue."""
    m1 = AgentMessage(task_id="t1", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="q1", output_data=None)
    m2 = AgentMessage(task_id="t2", agent_name="A", sender="Supervisor", receiver="Planner", stage="p", status="w", input_data="q2", output_data=None)

    coordinator.route_to_dlq(m1, reason="Reason 1", retry_count=2)
    coordinator.route_to_dlq(m2, reason="Reason 2", retry_count=3)

    all_entries = coordinator.dlq.list_entries()
    assert len(all_entries) == 2
    assert {e.message.task_id for e in all_entries} == {"t1", "t2"}


# ===========================================================================
# 11. Authorized Replay
# ===========================================================================

def test_11_authorized_replay(reliable_queue):
    """Verify authorized replay of eligible DLQ message re-enqueues it cleanly."""
    msg = AgentMessage(
        task_id="t_replay_auth",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Safe search telemetry query",
        output_data=None,
    )

    entry = reliable_queue.reliability.route_to_dlq(msg, reason="Temporary cluster partition", retry_count=2)
    dlq_id = entry.dlq_id

    # Authorized replay
    success, explanation, replayed_msg = reliable_queue.reliability.dlq.replay(
        dlq_id=dlq_id,
        authorized=True,
        authorization_token="admin_sec_token_777",
        target_queue=reliable_queue,
    )

    assert success is True
    assert "successfully authorized" in explanation
    assert replayed_msg is not None
    assert replayed_msg.message_id == msg.message_id
    assert reliable_queue.pending_count() == 1

    updated_entry = reliable_queue.reliability.dlq.get(dlq_id)
    assert updated_entry.replayed is True
    assert updated_entry.replay_count == 1
    assert len(updated_entry.replay_history) == 1
    assert updated_entry.replay_history[0]["authorized_by"] == "admin_sec_token_777"


# ===========================================================================
# 12. Unauthorized Replay Rejection
# ===========================================================================

def test_12_unauthorized_replay_rejection(coordinator):
    """Verify that replay without explicit authorization and token is rejected."""
    msg = AgentMessage(task_id="t_unauth", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="query", output_data=None)
    entry = coordinator.route_to_dlq(msg, reason="Error", retry_count=1)

    success, explanation, _ = coordinator.dlq.replay(
        dlq_id=entry.dlq_id,
        authorized=False,
    )

    assert success is False
    assert "Explicit authorization and token required" in explanation


# ===========================================================================
# 13. Replay History Preservation
# ===========================================================================

def test_13_replay_history_preservation(coordinator):
    """Verify that multiple authorized replays preserve full replay audit history."""
    msg = AgentMessage(task_id="t_hist", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="safe query", output_data=None)
    entry = coordinator.route_to_dlq(msg, reason="Error", retry_count=1)

    coordinator.dlq.replay(dlq_id=entry.dlq_id, authorized=True, authorization_token="admin_1")
    coordinator.dlq.replay(dlq_id=entry.dlq_id, authorized=True, authorization_token="admin_2")

    assert entry.replay_count == 2
    assert len(entry.replay_history) == 2
    assert entry.replay_history[0]["authorized_by"] == "admin_1"
    assert entry.replay_history[1]["authorized_by"] == "admin_2"


# ===========================================================================
# 14. Duplicate Side-Effect Protection (Non-Idempotent Replay Defense)
# ===========================================================================

def test_14_non_idempotent_replay_defense(coordinator):
    """Verify that non-idempotent operations (billing, database deletion, email) are strictly rejected from replay."""
    msg = AgentMessage(
        task_id="t_danger",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Executor",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="process payment transaction for credit card",
        output_data=None,
    )

    entry = coordinator.route_to_dlq(msg, reason="Timeout during bank gateway handshake", retry_count=1)

    # Replay must be refused even with authorization!
    success, explanation, _ = coordinator.dlq.replay(
        dlq_id=entry.dlq_id,
        authorized=True,
        authorization_token="super_admin",
    )

    assert success is False
    assert "non-idempotent operation" in explanation
    assert entry.replayed is False


# ===========================================================================
# 15. Uncertain Execution Handling
# ===========================================================================

def test_15_uncertain_execution_handling(coordinator):
    """Verify that operations with uncertain execution require reconciliation."""
    msg = AgentMessage(
        task_id="t_uncertain",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Executor",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Execute external side-effect",
        output_data=None,
    )

    coordinator.mark_execution_uncertain(msg)
    assert coordinator.is_execution_uncertain(msg) is True

    # Reconciling completion clears uncertain state
    coordinator.mark_operation_completed(msg)
    assert coordinator.is_execution_uncertain(msg) is False
    assert coordinator.is_operation_completed(msg) is True


# ===========================================================================
# 16. Reliability Metric Calculations
# ===========================================================================

def test_16_reliability_metric_calculations():
    """Verify calculation of telemetry counters and average delivery latency."""
    metrics = ReliabilityMetrics()

    metrics.record_message()
    metrics.record_message()
    metrics.record_delivery(0.010)
    metrics.record_delivery(0.020)
    metrics.record_ack(AckStatus.ACKNOWLEDGED)
    metrics.record_ack(AckStatus.REJECTED)
    metrics.record_retry()
    metrics.record_dlq()

    summary = metrics.get_summary()
    assert summary["total_messages"] == 2
    assert summary["delivered_messages"] == 2
    assert summary["acknowledged_messages"] == 1
    assert summary["retried_messages"] == 1
    assert summary["dead_letter_messages"] == 1
    assert summary["processing_failures"] == 1
    assert summary["average_delivery_latency_seconds"] == 0.015


# ===========================================================================
# 17. Full Multi-Agent Workflow Compatibility
# ===========================================================================

def test_17_full_reliable_workflow_integration(worker_registry, storage, coordinator):
    """Verify that the complete multi-agent workflow runs cleanly with reliability tracking:
    Supervisor -> Persistent Queue -> Researcher -> Planner -> Supervisor -> Executor -> Supervisor
    """
    mock_researcher = worker_registry.get_worker("Researcher")
    mock_researcher.research.return_value = {
        "task_id": "test_reliable_e2e",
        "status": "completed",
        "findings": "Reliable messaging confirmed across all agents.",
    }

    mock_planner = worker_registry.get_worker("Planner")
    mock_planner.plan.return_value = {
        "task_id": "test_reliable_e2e",
        "status": "completed",
        "tasks": [{"id": 1, "description": "Execute reliability audit", "authorized": True}],
    }

    mock_executor = worker_registry.get_worker("Executor")
    mock_executor.execute.return_value = {
        "task_id": "test_reliable_e2e_task_1",
        "status": "completed",
        "result": "Reliability audit passed with 100% acknowledgements.",
    }

    mock_supervisor = MagicMock(spec=SupervisorAgent)
    mock_supervisor.name = "SupervisorAgent"
    mock_supervisor.max_delegation_steps = 6

    mock_supervisor.decide.side_effect = [
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Researcher",
            capability_required="research",
            instruction="Research message reliability guarantees",
        ),
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Planner",
            capability_required="planning",
            instruction="Plan acknowledgement protocols",
        ),
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Executor",
            capability_required="execution",
            instruction="Execute reliability benchmarks",
        ),
        SupervisorDecision(
            decision_type=DecisionType.COMPLETE,
            final_response="All reliable messaging components fully verified.",
        ),
    ]

    router = MessageRouter(worker_registry=worker_registry, max_retries=2, backoff_factor=0.0)
    queue = MessageQueue(
        router=router,
        max_retries=2,
        storage=storage,
        auto_recover=True,
        reliability_coordinator=coordinator,
    )
    router.attach_queue(queue)

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        supervisor=mock_supervisor,
        worker_registry=worker_registry,
        message_router=router,
        message_queue=queue,
    )

    context = orchestrator.run_supervised("Run reliable multi-agent system test")

    assert context.status == WorkflowStatus.COMPLETED
    assert context.final_response is not None

    # Check reliability telemetry
    metrics_summary = coordinator.metrics.get_summary()
    assert metrics_summary["delivered_messages"] >= 3
    assert metrics_summary["acknowledged_messages"] >= 3
    assert metrics_summary["dead_letter_messages"] == 0
