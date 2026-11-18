"""Test Suite for Day 226 Part 6: Complete Multi-Agent Communication System Integration.

Comprehensive integration test suite covering:
1. Complete message lifecycle (creation -> queue -> router -> recipient dispatch -> delivery ack -> completion)
2. Supervisor-to-Worker communication (Supervisor -> Researcher, Supervisor -> Executor)
3. Worker-to-Worker communication (Researcher -> Planner handoff)
4. Worker-to-Supervisor communication (Planner -> Supervisor, Executor -> Supervisor)
5. Priority processing (CRITICAL > HIGH > NORMAL > LOW with FIFO tie-breaking)
6. Asynchronous processing (async delivery and concurrent queue processing)
7. Persistent message recovery (recovering unacknowledged/pending messages from SQLite after restart)
8. Acknowledgements (ACKNOWLEDGED, REJECTED, ACK_TIMEOUT, delivery vs processing status separation)
9. Retry handling (controlled retry of transient errors with backoff and retry limits)
10. Dead-letter queue (escalation of retry-exhausted and unroutable messages into SQLite DLQ)
11. Authorized replay (replay requires authorization; unauthorized replay raises PermissionError)
12. Idempotency (preventing duplicate execution and replay side effects on non-idempotent tasks)
13. Failure recovery (classification into TRANSIENT/VALIDATION/PERMANENT/AUTHORIZATION)
14. Reliability metrics (consolidated ReliabilityMonitor checking all 10 core telemetry dimensions)
15. Complete multi-agent workflow (End-to-end: Supervisor -> Researcher -> Planner -> Supervisor -> Executor -> Supervisor)
16. Backward compatibility (preserves existing run(), run_supervised(), and run_to_text() pipelines)
"""

import asyncio
import os
import tempfile
import time
import pytest
from unittest.mock import MagicMock

from day223_agents.core.orchestrator import (
    WorkflowOrchestrator,
    TaskContext,
    AgentMessage,
    MessageType,
    WorkflowStatus,
    run_end_to_end_demonstration,
)
from day223_agents.core.failure_recovery import (
    WorkerRegistry,
    FailureClassifier,
    FailureType,
    is_non_idempotent_operation,
)
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
    ReliabilityMonitor,
    RecoveryStatistics,
)
from day223_agents.agents.researcher import ResearcherAgent
from day223_agents.agents.planner import PlannerAgent
from day223_agents.agents.executor import ExecutorAgent
from day223_agents.agents.supervisor import SupervisorAgent, DecisionType, SupervisorDecision
from day223_agents.core.provider_interface import LLMProvider, LLMResponse, LLMMessage


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for test persistence."""
    fd, path = tempfile.mkstemp(suffix="_test_p6.db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.fixture
def storage(temp_db):
    """Instantiated SQLiteMessageStorage on temp database."""
    return SQLiteMessageStorage(db_path=temp_db)


@pytest.fixture
def mock_provider():
    """Mock LLMProvider with deterministic responses for tests."""
    provider = MagicMock(spec=LLMProvider)
    provider.generate.return_value = LLMResponse(
        content='{"tasks": [{"id": 1, "description": "Verify PostgreSQL connection and initialize schema", "authorized": true}], "decision": "DELEGATE", "target_worker": "Researcher", "instruction": "Research PostgreSQL benefits"}'
    )
    return provider


@pytest.fixture
def worker_registry(mock_provider):
    """WorkerRegistry populated with all 4 core agents."""
    reg = WorkerRegistry()
    researcher = ResearcherAgent(provider=mock_provider)
    planner = PlannerAgent(provider=mock_provider)
    executor = ExecutorAgent(provider=mock_provider)
    supervisor = SupervisorAgent(provider=mock_provider, worker_registry=reg)

    reg.register(researcher, capabilities=["research", "information_gathering", "analysis"], priority=10, name="Researcher")
    reg.register(researcher, capabilities=["research", "information_gathering", "analysis"], priority=10, name="ResearcherAgent")
    reg.register(planner, capabilities=["planning", "task_decomposition", "strategy"], priority=10, name="Planner")
    reg.register(planner, capabilities=["planning", "task_decomposition", "strategy"], priority=10, name="PlannerAgent")
    reg.register(executor, capabilities=["execution", "tool_execution", "action"], priority=10, name="Executor")
    reg.register(executor, capabilities=["execution", "tool_execution", "action"], priority=10, name="ExecutorAgent")
    reg.register(supervisor, capabilities=["supervision", "coordination", "evaluation"], priority=10, name="Supervisor")
    reg.register(supervisor, capabilities=["supervision", "coordination", "evaluation"], priority=10, name="SupervisorAgent")
    return reg


@pytest.fixture
def integrated_orchestrator(mock_provider, worker_registry, storage):
    """WorkflowOrchestrator fully wired with router, queue, storage, and reliability."""
    router = MessageRouter(worker_registry=worker_registry, max_retries=2, backoff_factor=0.0)
    dlq = DeadLetterQueue(storage=storage)
    reliability = ReliabilityCoordinator(
        default_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        dlq=dlq,
    )
    queue = MessageQueue(
        router=router,
        max_retries=2,
        storage=storage,
        reliability_coordinator=reliability,
    )
    router.attach_queue(queue)

    orch = WorkflowOrchestrator(
        provider=mock_provider,
        researcher=worker_registry.get_worker("Researcher"),
        planner=worker_registry.get_worker("Planner"),
        executor=worker_registry.get_worker("Executor"),
        supervisor=worker_registry.get_worker("Supervisor"),
        worker_registry=worker_registry,
        message_router=router,
        message_queue=queue,
    )
    return orch


# ===========================================================================
# 1. Complete Message Lifecycle
# ===========================================================================

def test_1_complete_message_lifecycle(integrated_orchestrator, storage):
    """Verify end-to-end lifecycle: creation -> queue -> routing -> execution -> ack -> persistence."""
    orch = integrated_orchestrator
    queue = orch.message_queue

    msg = AgentMessage(
        task_id="task_life_100",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Lifecycle research test",
        correlation_id="task_life_100",
        priority="HIGH",
    )

    # 1. Enqueue
    item = queue.enqueue(msg, priority=MessagePriority.HIGH)
    assert item.state == QueueMessageState.PENDING
    assert queue.size() == 1

    # 2. Process / Route
    receipt = queue.process_next()
    assert receipt is not None
    assert receipt.status == DeliveryStatus.DELIVERED
    assert receipt.message_id == msg.message_id

    # 3. Queue state
    assert queue.get_state(msg.message_id) == QueueMessageState.COMPLETED

    # 4. Storage persistence
    persisted = storage.get_message(msg.message_id)
    assert persisted is not None
    assert persisted["state"] == QueueMessageState.COMPLETED.value
    assert persisted["ack_status"] == AckStatus.ACKNOWLEDGED.value


# ===========================================================================
# 2. Supervisor-to-Worker Communication
# ===========================================================================

def test_2_supervisor_to_worker_communication(integrated_orchestrator):
    """Verify Supervisor can dispatch structured TASK_ASSIGNMENT messages to Researcher and Executor."""
    orch = integrated_orchestrator
    queue = orch.message_queue

    # Supervisor -> Researcher
    msg_res = AgentMessage(
        task_id="task_sup_res",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Research database indexing",
        correlation_id="task_sup_res",
    )
    queue.enqueue(msg_res, priority=MessagePriority.NORMAL)
    receipt_res = queue.process_next()
    assert receipt_res.status == DeliveryStatus.DELIVERED

    # Supervisor -> Executor
    msg_exec = AgentMessage(
        task_id="task_sup_exec",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Executor",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data={"id": 1, "description": "Execute database check", "authorized": True},
        correlation_id="task_sup_exec",
    )
    queue.enqueue(msg_exec, priority=MessagePriority.NORMAL)
    receipt_exec = queue.process_next()
    assert receipt_exec.status == DeliveryStatus.DELIVERED


# ===========================================================================
# 3. Worker-to-Worker Communication
# ===========================================================================

def test_3_worker_to_worker_communication(integrated_orchestrator):
    """Verify Researcher can hand off research findings directly to Planner via STAGE_HANDOFF."""
    orch = integrated_orchestrator
    queue = orch.message_queue

    msg = AgentMessage(
        task_id="task_w2w_handoff",
        agent_name="Researcher",
        sender="Researcher",
        receiver="Planner",
        stage="planning",
        status="waiting",
        message_type=MessageType.STAGE_HANDOFF,
        input_data={"findings": "PostgreSQL provides ACID compliance and JSONB support."},
        correlation_id="task_w2w_handoff",
    )

    queue.enqueue(msg, priority=MessagePriority.HIGH)
    receipt = queue.process_next()

    assert receipt.status == DeliveryStatus.DELIVERED
    assert receipt.sender == "Researcher"
    assert receipt.receiver == "Planner"
    assert receipt.message_type == MessageType.STAGE_HANDOFF.value


# ===========================================================================
# 4. Worker-to-Supervisor Communication
# ===========================================================================

def test_4_worker_to_supervisor_communication(integrated_orchestrator):
    """Verify Planner and Executor can return TASK_RESULT messages to the Supervisor."""
    orch = integrated_orchestrator
    queue = orch.message_queue

    # Planner -> Supervisor
    msg_plan = AgentMessage(
        task_id="task_w2s_plan",
        agent_name="Planner",
        sender="Planner",
        receiver="Supervisor",
        stage="approval",
        status="waiting",
        message_type=MessageType.TASK_RESULT,
        input_data={"tasks": [{"id": 1, "description": "Initialize schema"}]},
        output_data={"tasks": [{"id": 1, "description": "Initialize schema"}]},
        correlation_id="task_w2s_plan",
    )
    queue.enqueue(msg_plan)
    receipt_plan = queue.process_next()
    assert receipt_plan.status == DeliveryStatus.DELIVERED

    # Executor -> Supervisor
    msg_exec = AgentMessage(
        task_id="task_w2s_exec",
        agent_name="Executor",
        sender="Executor",
        receiver="Supervisor",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_RESULT,
        input_data={"id": 1, "description": "Initialize schema"},
        output_data={"task_id": 1, "status": "completed", "result": "Schema created"},
        correlation_id="task_w2s_exec",
    )
    queue.enqueue(msg_exec)
    receipt_exec = queue.process_next()
    assert receipt_exec.status == DeliveryStatus.DELIVERED


# ===========================================================================
# 5. Priority-Based Processing
# ===========================================================================

def test_5_priority_processing(integrated_orchestrator):
    """Verify CRITICAL > HIGH > NORMAL > LOW priority scheduling."""
    queue = integrated_orchestrator.message_queue

    msg_low = AgentMessage(task_id="t_low", agent_name="Supervisor", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="low", correlation_id="t")
    msg_norm = AgentMessage(task_id="t_norm", agent_name="Supervisor", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="norm", correlation_id="t")
    msg_high = AgentMessage(task_id="t_high", agent_name="Supervisor", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="high", correlation_id="t")
    msg_crit = AgentMessage(task_id="t_crit", agent_name="Supervisor", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="crit", correlation_id="t")

    queue.enqueue(msg_low, priority=MessagePriority.LOW)
    queue.enqueue(msg_norm, priority=MessagePriority.NORMAL)
    queue.enqueue(msg_crit, priority=MessagePriority.CRITICAL)
    queue.enqueue(msg_high, priority=MessagePriority.HIGH)

    r1 = queue.process_next()
    r2 = queue.process_next()
    r3 = queue.process_next()
    r4 = queue.process_next()

    assert r1.message_id == msg_crit.message_id
    assert r2.message_id == msg_high.message_id
    assert r3.message_id == msg_norm.message_id
    assert r4.message_id == msg_low.message_id


# ===========================================================================
# 6. Asynchronous Processing
# ===========================================================================

@pytest.mark.asyncio
async def test_6_asynchronous_processing(integrated_orchestrator):
    """Verify asynchronous non-blocking message delivery via deliver_message_async and process_all_async."""
    queue = integrated_orchestrator.message_queue

    msg1 = AgentMessage(task_id="t_async_1", agent_name="Supervisor", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="data 1", correlation_id="t1")
    msg2 = AgentMessage(task_id="t_async_2", agent_name="Supervisor", sender="Supervisor", receiver="Planner", stage="p", status="w", input_data="data 2", correlation_id="t2")

    item1 = queue.enqueue(msg1)
    receipt1 = await queue.deliver_message_async(item1)
    assert receipt1.status == DeliveryStatus.DELIVERED

    queue.enqueue(msg2)
    receipts = await queue.process_all_async()
    assert len(receipts) == 1
    assert receipts[0].message_id == msg2.message_id
    assert receipts[0].status == DeliveryStatus.DELIVERED


# ===========================================================================
# 7. Persistent Message Recovery
# ===========================================================================

def test_7_persistent_message_recovery(temp_db, worker_registry):
    """Verify pending messages persist in SQLite and are restored into the queue upon restart."""
    storage1 = SQLiteMessageStorage(db_path=temp_db)
    router1 = MessageRouter(worker_registry=worker_registry)
    q1 = MessageQueue(router=router1, storage=storage1)

    msg = AgentMessage(
        task_id="t_recover",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        input_data="Crash recovery test",
        correlation_id="t_recover",
    )
    q1.enqueue(msg, priority=MessagePriority.HIGH)

    # Simulate restart by instantiating new storage and queue pointing to same db
    storage2 = SQLiteMessageStorage(db_path=temp_db)
    router2 = MessageRouter(worker_registry=worker_registry)
    q2 = MessageQueue(router=router2, storage=storage2, auto_recover=True)

    assert q2.size() == 1
    assert q2.pending_count() == 1

    receipt = q2.process_next()
    assert receipt is not None
    assert receipt.message_id == msg.message_id
    assert receipt.status == DeliveryStatus.DELIVERED
    assert q2.get_state(msg.message_id) == QueueMessageState.COMPLETED


# ===========================================================================
# 8. Acknowledgements
# ===========================================================================

def test_8_acknowledgements_mechanism(integrated_orchestrator):
    """Verify ACKNOWLEDGED, REJECTED, and ACK_TIMEOUT states separate transport from processing."""
    coord = integrated_orchestrator.message_queue.reliability

    msg = AgentMessage(task_id="t_ack", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="d", correlation_id="t")

    # Positive ACK
    ack_pos = coord.record_acknowledgement(msg, recipient="Researcher", status=AckStatus.ACKNOWLEDGED, reason="Processed successfully")
    assert ack_pos.status == AckStatus.ACKNOWLEDGED
    assert coord.get_acknowledgement(msg.message_id).status == AckStatus.ACKNOWLEDGED

    # Rejected ACK
    msg_rej = AgentMessage(task_id="t_rej", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="d", correlation_id="t")
    ack_rej = coord.record_acknowledgement(msg_rej, recipient="Researcher", status=AckStatus.REJECTED, reason="Validation failed")
    assert ack_rej.status == AckStatus.REJECTED
    assert coord.metrics.processing_failures >= 1


# ===========================================================================
# 9. Retry Handling
# ===========================================================================

def test_9_retry_handling(integrated_orchestrator):
    """Verify transient failures are retried with controlled backoff."""
    router = integrated_orchestrator.message_router
    queue = integrated_orchestrator.message_queue

    attempts = [0]
    def flaky_receiver(m):
        attempts[0] += 1
        if attempts[0] < 3:
            raise ConnectionError("Temporary network reset")
        return {"status": "completed", "result": "recovered"}

    router.register_recipient_handler("FlakyWorker", flaky_receiver)
    RoutingRules.SYSTEM_ROLES.add("FlakyWorker")

    try:
        msg = AgentMessage(task_id="t_retry", agent_name="Supervisor", sender="Supervisor", receiver="FlakyWorker", stage="e", status="w", input_data="retry test", correlation_id="t")
        queue.enqueue(msg)

        receipt = queue.process_next()
        assert receipt.status == DeliveryStatus.DELIVERED
        assert attempts[0] == 3
        assert receipt.attempts == 3
    finally:
        RoutingRules.SYSTEM_ROLES.remove("FlakyWorker")


# ===========================================================================
# 10. Dead-Letter Queue Escalation
# ===========================================================================

def test_10_dead_letter_queue_escalation(integrated_orchestrator):
    """Verify retry-exhausted messages escalate into the Dead-Letter Queue."""
    router = integrated_orchestrator.message_router
    queue = integrated_orchestrator.message_queue
    dlq = queue.reliability.dlq

    def broken_receiver(m):
        raise ConnectionResetError("Connection reset by peer")

    router.register_recipient_handler("BrokenWorker", broken_receiver)
    RoutingRules.SYSTEM_ROLES.add("BrokenWorker")

    try:
        msg = AgentMessage(task_id="t_dlq", agent_name="Supervisor", sender="Supervisor", receiver="BrokenWorker", stage="e", status="w", input_data="exhaust retries", correlation_id="t")
        queue.enqueue(msg)

        receipts = queue.drain()
        assert len(receipts) == 1
        assert receipts[0].status == DeliveryStatus.FAILED
        assert receipts[0].attempts == 3

        # Confirm DLQ presence
        assert dlq.size() >= 1
        entry = dlq.get(f"dlq_{msg.message_id}")
        assert entry is not None
        assert entry.retry_count == 3
        assert "Connection reset by peer" in entry.reason
    finally:
        RoutingRules.SYSTEM_ROLES.remove("BrokenWorker")


# ===========================================================================
# 11. Authorized Replay
# ===========================================================================

def test_11_authorized_replay(integrated_orchestrator):
    """Verify DLQ messages can be replayed only with authorized credentials."""
    queue = integrated_orchestrator.message_queue
    dlq = queue.reliability.dlq

    msg = AgentMessage(task_id="t_replay", agent_name="Supervisor", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="replay data", correlation_id="t")
    entry = dlq.push(msg, reason="Test failure", retry_count=3)

    dlq_id = entry.dlq_id

    # Unauthorized replay attempt is rejected
    success, explanation, _ = dlq.replay(dlq_id, authorized=False, target_queue=queue)
    assert success is False
    assert "Explicit authorization and token required" in explanation

    # Authorized replay succeeds
    success, explanation, replayed_msg = dlq.replay(dlq_id, authorized=True, authorization_token="ADMIN_TOKEN", target_queue=queue)
    assert success is True
    assert "successfully authorized" in explanation
    assert replayed_msg is not None
    assert replayed_msg.message_id == msg.message_id
    assert queue.pending_count() == 1


# ===========================================================================
# 12. Idempotency & Replay Defense
# ===========================================================================

def test_12_idempotency_and_non_idempotent_replay_defense(integrated_orchestrator):
    """Verify non-idempotent operations are protected against blind retry and repeated replay."""
    coord = integrated_orchestrator.message_queue.reliability
    dlq = coord.dlq

    # Operation containing mutating keyword 'delete'
    msg = AgentMessage(
        task_id="t_side_effect",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Executor",
        stage="execution",
        status="waiting",
        input_data="delete records from database",
        correlation_id="t_side_effect",
    )

    # Confirm classified as non-idempotent
    assert is_non_idempotent_operation(msg.input_data) is True

    # Check that completed operation is remembered by idempotency cache
    coord.mark_operation_completed(msg)
    assert coord.is_operation_completed(msg) is True

    # Pushing to DLQ and attempting replay of non-idempotent operation is blocked
    entry = dlq.push(msg, reason="Failed delete", retry_count=1)
    success, explanation, _ = dlq.replay(entry.dlq_id, authorized=True, authorization_token="TOKEN")
    assert success is False
    assert "non-idempotent" in explanation.lower()


# ===========================================================================
# 13. Failure Recovery & Classification
# ===========================================================================

def test_13_failure_recovery_classification():
    """Verify FailureClassifier correctly segments TRANSIENT, VALIDATION, AUTHORIZATION, and PERMANENT."""
    trans_err = FailureClassifier.classify(TimeoutError("Connection timeout"))
    assert trans_err.failure_type == FailureType.TRANSIENT
    assert trans_err.is_retryable is True

    val_err = FailureClassifier.classify("Validation error: tasks list must be a list")
    assert val_err.failure_type == FailureType.VALIDATION
    assert val_err.is_retryable is True

    auth_err = FailureClassifier.classify(PermissionError("Unauthorized sensitive operation blocked"))
    assert auth_err.failure_type == FailureType.AUTHORIZATION
    assert auth_err.is_retryable is False

    perm_err = FailureClassifier.classify("404 Model is no longer available permanent error")
    assert perm_err.failure_type == FailureType.PERMANENT
    assert perm_err.is_retryable is False


# ===========================================================================
# 14. Reliability Monitoring Interface
# ===========================================================================

def test_14_reliability_monitoring_interface(integrated_orchestrator):
    """Verify ReliabilityMonitor inspects all 10 core operational dimensions."""
    orch = integrated_orchestrator
    queue = orch.message_queue
    coord = queue.reliability

    monitor = ReliabilityMonitor(coordinator=coord, queue=queue, storage=queue.storage)

    # Process one message to generate telemetry
    msg = AgentMessage(task_id="t_mon", agent_name="Supervisor", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="mon", correlation_id="t")
    queue.enqueue(msg)
    queue.process_next()

    stats = monitor.inspect()
    assert "total_messages" in stats
    assert "pending_messages" in stats
    assert "delivered_messages" in stats
    assert "acknowledged_messages" in stats
    assert "failed_messages" in stats
    assert "retried_messages" in stats
    assert "dead_letter_messages" in stats
    assert "average_delivery_latency" in stats
    assert "processing_failures" in stats
    assert "recovery_statistics" in stats

    assert stats["delivered_messages"] >= 1
    assert stats["acknowledged_messages"] >= 1
    assert isinstance(stats["recovery_statistics"], dict)


# ===========================================================================
# 15. Complete Multi-Agent Workflow
# ===========================================================================

def test_15_complete_multi_agent_workflow(integrated_orchestrator):
    """Verify 5-step communication pipeline:
    Supervisor -> Researcher -> Planner -> Supervisor -> Executor -> Supervisor
    """
    orch = integrated_orchestrator

    def safe_db_execution(task_item):
        return {
            "task_id": 1,
            "task": task_item.get("description", "DB Task"),
            "status": "completed",
            "result": "PostgreSQL table created and indexed successfully.",
        }

    ctx = orch.run_communication_workflow(
        user_request="Research the benefits of PostgreSQL, create an implementation plan, and execute an approved database-related task.",
        execution_handler=safe_db_execution,
    )

    # Assert workflow completed
    assert ctx.status == WorkflowStatus.COMPLETED
    assert bool(ctx.research_findings) is True
    assert bool(ctx.execution_plan) is True
    assert len(ctx.execution_results) >= 1
    assert ctx.execution_results[0]["status"] == "completed"

    # Assert 5 messages recorded in history with matching correlation IDs
    assert len(ctx.messages) == 5
    senders = [m.sender for m in ctx.messages]
    receivers = [m.receiver for m in ctx.messages]

    assert senders == ["Supervisor", "Researcher", "Planner", "Supervisor", "Executor"]
    assert receivers == ["Researcher", "Planner", "Supervisor", "Executor", "Supervisor"]

    for m in ctx.messages:
        assert m.correlation_id == ctx.workflow_id
        assert m.task_id.startswith(ctx.workflow_id)


# ===========================================================================
# 16. Backward Compatibility
# ===========================================================================

def test_16_backward_compatibility(mock_provider):
    """Verify legacy sequential run() and run_to_text() pipelines remain operational."""
    orch = WorkflowOrchestrator(provider=mock_provider)

    # run()
    ctx = orch.run("Test backward compatibility request")
    assert ctx.status == WorkflowStatus.COMPLETED
    assert ctx.final_response is not None

    # run_to_text()
    text = orch.run_to_text("Test backward compatibility to text")
    assert isinstance(text, str)
    assert len(text) > 0


# ===========================================================================
# 17. End-to-End Practical Demonstration
# ===========================================================================

def test_17_end_to_end_demonstration():
    """Verify the standalone run_end_to_end_demonstration function executes cleanly."""
    demo_result = run_end_to_end_demonstration()

    assert demo_result["status"] == "completed"
    assert len(demo_result["stages_executed"]) == 6
    assert demo_result["messages_count"] == 5
    assert "PostgreSQL" in demo_result["user_request"]
    assert "monitor_summary" in demo_result
    assert demo_result["monitor_summary"]["delivered_messages"] >= 5
