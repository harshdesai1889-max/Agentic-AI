"""Test Suite for Day 226 Part 3: Message Queues, Prioritization and Asynchronous Processing.

Validates:
1. Message insertion and retrieval (QueueItem, PENDING state).
2. Priority ordering (CRITICAL > HIGH > NORMAL > LOW).
3. FIFO ordering for equal priorities.
4. Queue state transitions (PENDING -> PROCESSING -> COMPLETED/FAILED).
5. Asynchronous delivery (asyncio deliver_message_async).
6. Concurrent independent message processing (process_all_async with concurrency limit).
7. Duplicate-processing prevention and mutex isolation.
8. Failure handling and error capture.
9. Retry limits on transient failures.
10. Task and correlation context preservation across queue processing.
11. Lower-priority message starvation prevention via priority aging.
12. Compatibility with existing synchronous execution patterns (drain, process_next).
13. End-to-end multi-agent workflow integration:
    Supervisor -> MessageQueue -> Researcher
    Researcher -> MessageQueue -> Planner
    Planner -> MessageQueue -> Supervisor
    Supervisor -> MessageQueue -> Executor
    Executor -> MessageQueue -> Supervisor
"""

import asyncio
import copy
import time
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
def queue(populated_worker_registry):
    """Instantiated MessageQueue with attached MessageRouter."""
    router = MessageRouter(
        worker_registry=populated_worker_registry,
        max_retries=2,
        backoff_factor=0.0,
    )
    mq = MessageQueue(
        router=router,
        max_retries=2,
        starvation_threshold_seconds=0.1,  # Fast for anti-starvation test
        concurrency_limit=4,
    )
    router.attach_queue(mq)
    return mq


# ===========================================================================
# 1. Message Insertion and Retrieval
# ===========================================================================

def test_1_message_insertion_and_retrieval(queue):
    """Verify that messages are correctly enqueued and dequeued in PENDING/PROCESSING states."""
    msg = AgentMessage(
        task_id="task_q_100",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Collect metrics",
        output_data=None,
    )

    item = queue.enqueue(msg, priority=MessagePriority.HIGH)
    assert item is not None
    assert item.message.message_id == msg.message_id
    assert item.state == QueueMessageState.PENDING
    assert queue.pending_count() == 1

    # Dequeue
    popped = queue.dequeue()
    assert popped is not None
    assert popped.message.message_id == msg.message_id
    assert popped.state == QueueMessageState.PROCESSING
    assert queue.processing_count() == 1


# ===========================================================================
# 2. Priority Ordering (CRITICAL > HIGH > NORMAL > LOW)
# ===========================================================================

def test_2_priority_ordering(queue):
    """Verify that higher priority messages dequeue before lower priority messages."""
    m_low = AgentMessage(task_id="t1", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="low", output_data=None)
    m_norm = AgentMessage(task_id="t2", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="norm", output_data=None)
    m_high = AgentMessage(task_id="t3", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="high", output_data=None)
    m_crit = AgentMessage(task_id="t4", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="crit", output_data=None)

    # Enqueue in reverse priority order
    queue.enqueue(m_low, priority=MessagePriority.LOW)
    queue.enqueue(m_norm, priority=MessagePriority.NORMAL)
    queue.enqueue(m_high, priority=MessagePriority.HIGH)
    queue.enqueue(m_crit, priority=MessagePriority.CRITICAL)

    # Pop order must be: CRITICAL, HIGH, NORMAL, LOW
    p1 = queue.dequeue()
    p2 = queue.dequeue()
    p3 = queue.dequeue()
    p4 = queue.dequeue()

    assert p1.original_priority == MessagePriority.CRITICAL
    assert p2.original_priority == MessagePriority.HIGH
    assert p3.original_priority == MessagePriority.NORMAL
    assert p4.original_priority == MessagePriority.LOW


# ===========================================================================
# 3. FIFO Ordering for Equal Priorities
# ===========================================================================

def test_3_fifo_ordering_for_equal_priorities(queue):
    """Verify that messages with equal priority follow strict FIFO sequence."""
    m1 = AgentMessage(task_id="t1", agent_name="A", sender="Supervisor", receiver="Planner", stage="p", status="w", input_data="first", output_data=None)
    m2 = AgentMessage(task_id="t2", agent_name="A", sender="Supervisor", receiver="Planner", stage="p", status="w", input_data="second", output_data=None)
    m3 = AgentMessage(task_id="t3", agent_name="A", sender="Supervisor", receiver="Planner", stage="p", status="w", input_data="third", output_data=None)

    queue.enqueue(m1, priority=MessagePriority.HIGH)
    queue.enqueue(m2, priority=MessagePriority.HIGH)
    queue.enqueue(m3, priority=MessagePriority.HIGH)

    assert queue.dequeue().message.input_data == "first"
    assert queue.dequeue().message.input_data == "second"
    assert queue.dequeue().message.input_data == "third"


# ===========================================================================
# 4. Queue State Transitions
# ===========================================================================

def test_4_queue_state_transitions(queue):
    """Verify clean state progression: PENDING -> PROCESSING -> COMPLETED."""
    msg = AgentMessage(
        task_id="task_state_400",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="State transition test",
        output_data=None,
    )

    item = queue.enqueue(msg, priority=MessagePriority.NORMAL)
    assert queue.get_state(msg.message_id) == QueueMessageState.PENDING

    # Process next executes delivery
    receipt = queue.process_next()
    assert receipt.status == DeliveryStatus.DELIVERED
    assert queue.get_state(msg.message_id) == QueueMessageState.COMPLETED
    assert item.completed_at is not None


# ===========================================================================
# 5. Asynchronous Delivery
# ===========================================================================

@pytest.mark.asyncio
async def test_5_asynchronous_delivery(queue):
    """Verify asynchronous delivery via deliver_message_async."""
    msg = AgentMessage(
        task_id="task_async_500",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Planner",
        stage="planning",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Async planning task",
        output_data=None,
    )

    item = queue.enqueue(msg, priority=MessagePriority.HIGH)
    popped = queue.dequeue()

    receipt = await queue.deliver_message_async(popped)
    assert receipt.status == DeliveryStatus.DELIVERED
    assert queue.get_state(msg.message_id) == QueueMessageState.COMPLETED


# ===========================================================================
# 6. Concurrent Independent Messages
# ===========================================================================

@pytest.mark.asyncio
async def test_6_concurrent_independent_messages(queue):
    """Verify concurrent batch delivery with process_all_async."""
    for i in range(8):
        msg = AgentMessage(
            task_id=f"task_conc_{i}",
            agent_name="Supervisor",
            sender="Supervisor",
            receiver="Researcher",
            stage="research",
            status="waiting",
            message_type=MessageType.TASK_ASSIGNMENT,
            input_data=f"Concurrent payload {i}",
            output_data=None,
        )
        queue.enqueue(msg, priority=MessagePriority.NORMAL)

    assert queue.pending_count() == 8
    receipts = await queue.process_all_async()
    assert len(receipts) == 8
    for r in receipts:
        assert r.status == DeliveryStatus.DELIVERED
    assert queue.pending_count() == 0


# ===========================================================================
# 7. Duplicate-Processing Prevention
# ===========================================================================

def test_7_duplicate_processing_prevention(queue):
    """Verify that an active message cannot be re-enqueued or dequeued simultaneously."""
    msg = AgentMessage(
        task_id="task_dup_700",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Executor",
        stage="execution",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Single execution only",
        output_data=None,
    )

    item1 = queue.enqueue(msg, priority=MessagePriority.HIGH)
    # Attempting to enqueue same message while active returns existing item
    item2 = queue.enqueue(msg, priority=MessagePriority.HIGH)
    assert item1.sequence_id == item2.sequence_id

    # Once dequeued and completed, re-processing is prevented by router idempotency
    receipt = queue.process_next()
    assert receipt.status == DeliveryStatus.DELIVERED
    assert queue.get_state(msg.message_id) == QueueMessageState.COMPLETED


# ===========================================================================
# 8. Failure Handling and Error Details Preservation
# ===========================================================================

def test_8_failure_handling(queue):
    """Verify that delivery exceptions are recorded and preserved without discarding."""
    bad_handler = MagicMock(side_effect=ValueError("Invalid payload format"))
    queue.router.register_recipient_handler("CrashWorker", bad_handler)
    RoutingRules.SYSTEM_ROLES.add("CrashWorker")

    try:
        msg = AgentMessage(
            task_id="task_err_800",
            agent_name="Supervisor",
            sender="Supervisor",
            receiver="CrashWorker",
            stage="execution",
            status="waiting",
            message_type=MessageType.TASK_ASSIGNMENT,
            input_data="Crash payload",
            output_data=None,
        )

        item = queue.enqueue(msg, priority=MessagePriority.NORMAL)
        receipt = queue.process_next()

        assert receipt.status == DeliveryStatus.FAILED
        assert "Invalid payload format" in (receipt.error or "")
        assert queue.get_state(msg.message_id) == QueueMessageState.FAILED
        assert item.last_error is not None
    finally:
        RoutingRules.SYSTEM_ROLES.remove("CrashWorker")


# ===========================================================================
# 9. Retry Limits on Transient Failures
# ===========================================================================

def test_9_retry_limits_on_transient_failures(queue):
    """Verify that retryable transient failures are retried up to max_retries before marking FAILED."""
    attempts = [0]

    def always_fail_network(m):
        attempts[0] += 1
        raise ConnectionError("Network host unreachable")

    queue.router.register_recipient_handler("NetworkWorker", always_fail_network)
    RoutingRules.SYSTEM_ROLES.add("NetworkWorker")

    try:
        msg = AgentMessage(
            task_id="task_retry_900",
            agent_name="Supervisor",
            sender="Supervisor",
            receiver="NetworkWorker",
            stage="execution",
            status="waiting",
            message_type=MessageType.TASK_ASSIGNMENT,
            input_data="Network test",
            output_data=None,
        )

        item = queue.enqueue(msg, priority=MessagePriority.NORMAL)

        # Drain executes until terminal state (all attempts made)
        receipts = queue.drain()
        assert len(receipts) == 1
        assert receipts[0].status == DeliveryStatus.FAILED
        assert receipts[0].attempts == 3
        assert queue.get_state(msg.message_id) == QueueMessageState.FAILED
    finally:
        RoutingRules.SYSTEM_ROLES.remove("NetworkWorker")


# ===========================================================================
# 10. Context and Correlation ID Preservation
# ===========================================================================

def test_10_context_preservation(queue):
    """Verify that task_id, correlation_id, and delivery receipts are stored in TaskContext."""
    context = TaskContext(workflow_id="wf_ctx_1000", user_request="Context test")

    msg = AgentMessage(
        task_id="task_ctx_1000",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Planner",
        stage="planning",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Context payload",
        output_data=None,
        correlation_id="root_workflow_1000",
    )

    queue.enqueue(msg, priority=MessagePriority.NORMAL)
    receipt = queue.process_next(context=context)

    assert receipt.status == DeliveryStatus.DELIVERED
    assert receipt.correlation_id == "root_workflow_1000"
    assert len(context.messages) == 1
    assert context.messages[0].correlation_id == "root_workflow_1000"


# ===========================================================================
# 11. Anti-Starvation Aging
# ===========================================================================

def test_11_lower_priority_starvation_prevention(queue):
    """Verify that older LOW priority messages receive priority boosts to prevent starvation."""
    m_low = AgentMessage(task_id="t_low", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="low_item", output_data=None)
    item_low = queue.enqueue(m_low, priority=MessagePriority.LOW)

    # Simulate passage of time beyond starvation threshold (0.1s in fixture)
    time.sleep(0.15)

    # Enqueue a fresh NORMAL message
    m_norm = AgentMessage(task_id="t_norm", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="norm_item", output_data=None)
    queue.enqueue(m_norm, priority=MessagePriority.NORMAL)

    # Dequeue applies anti-starvation: aged LOW message was promoted and comes first or equal
    first = queue.dequeue()
    assert first is not None
    # Because m_low waited > threshold, its priority was boosted from 3 to 2, and by FIFO it was queued earlier!
    assert first.message.input_data == "low_item"


# ===========================================================================
# 12. Compatibility with Existing Synchronous Operations
# ===========================================================================

def test_12_synchronous_compatibility(queue):
    """Verify that sync drain() operates seamlessly with existing synchronous workflows."""
    for i in range(3):
        msg = AgentMessage(
            task_id=f"t_sync_{i}",
            agent_name="Supervisor",
            sender="Supervisor",
            receiver="Executor",
            stage="execution",
            status="waiting",
            message_type=MessageType.TASK_ASSIGNMENT,
            input_data=f"Step {i}",
            output_data=None,
        )
        queue.enqueue(msg, priority=MessagePriority.HIGH)

    receipts = queue.drain()
    assert len(receipts) == 3
    for r in receipts:
        assert r.status == DeliveryStatus.DELIVERED
    assert queue.pending_count() == 0


# ===========================================================================
# 13. End-to-End Multi-Agent Workflow Integration
# (Supervisor -> MessageQueue -> Researcher -> Planner -> Supervisor -> Executor -> Supervisor)
# ===========================================================================

def test_13_full_workflow_integration(populated_worker_registry):
    """Demonstrate end-to-end queued workflow:
    Supervisor -> MessageQueue -> Researcher
    Researcher -> MessageQueue -> Planner
    Planner -> MessageQueue -> Supervisor
    Supervisor -> MessageQueue -> Executor
    Executor -> MessageQueue -> Supervisor
    """
    mock_researcher = populated_worker_registry.get_worker("Researcher")
    mock_researcher.research.return_value = {
        "task_id": "test_queue_e2e",
        "status": "completed",
        "findings": "Verified queue latency under 5ms.",
    }

    mock_planner = populated_worker_registry.get_worker("Planner")
    mock_planner.plan.return_value = {
        "task_id": "test_queue_e2e",
        "status": "completed",
        "tasks": [{"id": 1, "description": "Execute prioritized batch", "authorized": True}],
    }

    mock_executor = populated_worker_registry.get_worker("Executor")
    mock_executor.execute.return_value = {
        "task_id": "test_queue_e2e_task_1",
        "status": "completed",
        "result": "Batch execution completed successfully.",
    }

    mock_supervisor = MagicMock(spec=SupervisorAgent)
    mock_supervisor.name = "SupervisorAgent"
    mock_supervisor.max_delegation_steps = 6

    # Supervised decisions
    mock_supervisor.decide.side_effect = [
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Researcher",
            capability_required="research",
            instruction="Research queue benchmark metrics",
        ),
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Planner",
            capability_required="planning",
            instruction="Plan queue deployment",
        ),
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Executor",
            capability_required="execution",
            instruction="Run execution tasks",
        ),
        SupervisorDecision(
            decision_type=DecisionType.COMPLETE,
            final_response="Queued multi-agent operations finished with verified success.",
        ),
    ]

    # Initialize router and queue
    router = MessageRouter(worker_registry=populated_worker_registry, max_retries=2, backoff_factor=0.0)
    queue = MessageQueue(router=router, max_retries=2)
    router.attach_queue(queue)

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
        supervisor=mock_supervisor,
        worker_registry=populated_worker_registry,
        message_router=router,
        message_queue=queue,
    )

    context = orchestrator.run_supervised("Run queued multi-agent benchmark")

    assert context.status == WorkflowStatus.COMPLETED
    assert context.final_response is not None

    # Verify that messages passed through MessageRouter and queue
    history = router._delivery_receipts
    assert len(history) >= 3

    receivers = [r.receiver for r in history if r.sender in ("Supervisor", "SupervisorAgent")]
    assert "Researcher" in receivers
    assert "Planner" in receivers
    assert "Executor" in receivers
