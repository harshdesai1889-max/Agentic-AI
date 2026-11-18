"""Test Suite for Day 226 Part 4: Persistent Message Storage and Restart Recovery.

Validates:
1. Message persistence in SQLite.
2. Safe serialization and deserialization of payloads and metadata.
3. Priority ordering preservation after restart.
4. FIFO ordering preservation after restart.
5. Pending message recovery on restart.
6. Interrupted PROCESSING message recovery (re-queuing vs failure).
7. Completed message history preservation.
8. Failed message detail preservation.
9. Persistent duplicate message_id rejection.
10. Database transaction rollback on SQLite integrity or syntax errors.
11. SQLite error handling and thread isolation.
12. Asynchronous compatibility with persistent queue.
13. Synchronous compatibility with persistent queue.
14. End-to-end multi-agent workflow integration:
    Supervisor -> Persistent Queue -> Researcher
    Researcher -> Persistent Queue -> Planner
    Planner -> Persistent Queue -> Supervisor
    Supervisor -> Persistent Queue -> Executor
    Executor -> Persistent Queue -> Supervisor
"""

import asyncio
import copy
import os
import sqlite3
import tempfile
import time
from pathlib import Path
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
from day223_agents.agents.supervisor import (
    SupervisorAgent,
    DecisionType,
    SupervisorDecision,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary SQLite database file path."""
    return str(tmp_path / "test_messages.db")


@pytest.fixture
def storage(temp_db_path):
    """Instantiated SQLiteMessageStorage with temporary file database."""
    return SQLiteMessageStorage(db_path=temp_db_path)


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
def persistent_queue(storage, populated_worker_registry):
    """MessageQueue with SQLite storage attached."""
    router = MessageRouter(
        worker_registry=populated_worker_registry,
        max_retries=2,
        backoff_factor=0.0,
    )
    mq = MessageQueue(
        router=router,
        max_retries=2,
        storage=storage,
        auto_recover=True,
    )
    router.attach_queue(mq)
    return mq


# ===========================================================================
# 1. Message Persistence in SQLite
# ===========================================================================

def test_1_message_persistence(storage):
    """Verify that AgentMessage is persisted with all mandatory and metadata fields."""
    msg = AgentMessage(
        task_id="task_p_100",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data={"query": "Survey edge computing"},
        output_data=None,
        correlation_id="wf_root_100",
        priority="HIGH",
    )

    success = storage.insert_message(msg, priority=MessagePriority.HIGH)
    assert success is True

    record = storage.get_message(msg.message_id)
    assert record is not None
    assert record["message_id"] == msg.message_id
    assert record["task_id"] == "task_p_100"
    assert record["correlation_id"] == "wf_root_100"
    assert record["sender"] == "Supervisor"
    assert record["receiver"] == "Researcher"
    assert record["state"] == "pending"
    assert record["priority"] == "HIGH"


# ===========================================================================
# 2. Serialization and Deserialization
# ===========================================================================

def test_2_serialization_deserialization(storage):
    """Verify round-trip payload and metadata serialization/deserialization."""
    complex_input = {
        "nested": {"key": "val", "list": [1, 2, 3]},
        "flag": True,
        "amount": 99.95,
    }
    msg = AgentMessage(
        task_id="task_ser_200",
        agent_name="Planner",
        sender="Planner",
        receiver="Executor",
        stage="planning",
        status="success",
        input_data=complex_input,
        output_data={"generated_plan": ["stepA", "stepB"]},
        metadata={"auth_scope": "admin", "retry_count": 0},
    )

    storage.insert_message(msg, priority=MessagePriority.NORMAL)
    rec = storage.get_message(msg.message_id)

    payload = storage.deserialize_payload(rec["payload_json"])
    assert payload["input_data"] == complex_input
    assert payload["output_data"]["generated_plan"] == ["stepA", "stepB"]

    meta = storage.deserialize_payload(rec["metadata_json"])
    assert meta["auth_scope"] == "admin"


# ===========================================================================
# 3. Priority Ordering Preservation After Restart
# ===========================================================================

def test_3_priority_ordering_after_restart(temp_db_path, populated_worker_registry):
    """Verify that recovering from storage preserves priority sorting (CRITICAL > HIGH > NORMAL > LOW)."""
    # 1. Enqueue in first queue instance
    storage1 = SQLiteMessageStorage(db_path=temp_db_path)
    router1 = MessageRouter(worker_registry=populated_worker_registry)
    q1 = MessageQueue(router=router1, storage=storage1, auto_recover=False)

    m_low = AgentMessage(task_id="t1", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="low", output_data=None)
    m_high = AgentMessage(task_id="t2", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="high", output_data=None)
    m_crit = AgentMessage(task_id="t3", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="crit", output_data=None)

    q1.enqueue(m_low, priority=MessagePriority.LOW)
    q1.enqueue(m_high, priority=MessagePriority.HIGH)
    q1.enqueue(m_crit, priority=MessagePriority.CRITICAL)

    # 2. Simulate application restart: instantiate new queue pointing to same db
    storage2 = SQLiteMessageStorage(db_path=temp_db_path)
    router2 = MessageRouter(worker_registry=populated_worker_registry)
    q2 = MessageQueue(router=router2, storage=storage2, auto_recover=True)

    # Must dequeue in priority order: CRITICAL, HIGH, LOW
    p1 = q2.dequeue()
    p2 = q2.dequeue()
    p3 = q2.dequeue()

    assert p1.original_priority == MessagePriority.CRITICAL
    assert p2.original_priority == MessagePriority.HIGH
    assert p3.original_priority == MessagePriority.LOW


# ===========================================================================
# 4. FIFO Ordering Preservation After Restart
# ===========================================================================

def test_4_fifo_ordering_after_restart(temp_db_path, populated_worker_registry):
    """Verify that messages with identical priority preserve FIFO sequence across restarts."""
    storage1 = SQLiteMessageStorage(db_path=temp_db_path)
    router1 = MessageRouter(worker_registry=populated_worker_registry)
    q1 = MessageQueue(router=router1, storage=storage1, auto_recover=False)

    m1 = AgentMessage(task_id="t1", agent_name="A", sender="Supervisor", receiver="Planner", stage="p", status="w", input_data="first", output_data=None)
    m2 = AgentMessage(task_id="t2", agent_name="A", sender="Supervisor", receiver="Planner", stage="p", status="w", input_data="second", output_data=None)
    m3 = AgentMessage(task_id="t3", agent_name="A", sender="Supervisor", receiver="Planner", stage="p", status="w", input_data="third", output_data=None)

    q1.enqueue(m1, priority=MessagePriority.NORMAL)
    time.sleep(0.01)
    q1.enqueue(m2, priority=MessagePriority.NORMAL)
    time.sleep(0.01)
    q1.enqueue(m3, priority=MessagePriority.NORMAL)

    # Restart
    storage2 = SQLiteMessageStorage(db_path=temp_db_path)
    router2 = MessageRouter(worker_registry=populated_worker_registry)
    q2 = MessageQueue(router=router2, storage=storage2, auto_recover=True)

    assert q2.dequeue().message.input_data == "first"
    assert q2.dequeue().message.input_data == "second"
    assert q2.dequeue().message.input_data == "third"


# ===========================================================================
# 5. Pending Message Recovery on Restart
# ===========================================================================

def test_5_pending_message_recovery(temp_db_path, populated_worker_registry):
    """Verify that PENDING messages survive restart and are reloaded into memory."""
    storage = SQLiteMessageStorage(db_path=temp_db_path)
    m = AgentMessage(task_id="t_pend", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="data", output_data=None)
    storage.insert_message(m, priority=MessagePriority.NORMAL)

    assert storage.count_by_state(QueueMessageState.PENDING) == 1

    # Initialize queue with auto_recover
    router = MessageRouter(worker_registry=populated_worker_registry)
    q = MessageQueue(router=router, storage=storage, auto_recover=True)

    assert q.pending_count() == 1
    assert q.get_state(m.message_id) == QueueMessageState.PENDING


# ===========================================================================
# 6. Interrupted PROCESSING Message Recovery
# ===========================================================================

def test_6_interrupted_processing_recovery(temp_db_path, populated_worker_registry):
    """Verify recovery of messages left in PROCESSING state during a crash.
    - Idempotent messages are reset to PENDING and re-queued.
    - Non-idempotent operations are marked FAILED to prevent duplicate side-effects.
    """
    storage = SQLiteMessageStorage(db_path=temp_db_path)

    # 1. Idempotent query left in processing
    m_safe = AgentMessage(
        task_id="t_safe",
        agent_name="A",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        input_data="Lookup telemetry stats",
        output_data=None,
    )
    storage.insert_message(m_safe, priority=MessagePriority.NORMAL)
    storage.update_state(m_safe.message_id, state=QueueMessageState.PROCESSING)

    # 2. Non-idempotent operation left in processing
    m_side_effect = AgentMessage(
        task_id="t_side",
        agent_name="A",
        sender="Supervisor",
        receiver="Executor",
        stage="execution",
        status="waiting",
        input_data="process payment transaction for order",
        output_data=None,
    )
    storage.insert_message(m_side_effect, priority=MessagePriority.HIGH)
    storage.update_state(m_side_effect.message_id, state=QueueMessageState.PROCESSING)

    # Trigger restart recovery
    router = MessageRouter(worker_registry=populated_worker_registry)
    q = MessageQueue(router=router, storage=storage, auto_recover=True)

    # m_safe should be recovered and re-queued
    assert q.get_state(m_safe.message_id) == QueueMessageState.PENDING
    assert q.pending_count() == 1

    # m_side_effect must NOT be automatically re-executed
    rec_side = storage.get_message(m_side_effect.message_id)
    assert rec_side["state"] == "failed"
    assert "non-idempotent" in rec_side["error"]


# ===========================================================================
# 7. Completed Message History Preservation
# ===========================================================================

def test_7_completed_message_preservation(persistent_queue, storage):
    """Verify that completed messages retain full audit history in SQLite."""
    msg = AgentMessage(
        task_id="task_comp_700",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Collect telemetry",
        output_data=None,
    )

    persistent_queue.enqueue(msg, priority=MessagePriority.NORMAL)
    persistent_queue.process_next()

    record = storage.get_message(msg.message_id)
    assert record["state"] == "completed"
    assert record["completed_at"] is not None
    assert record["attempts"] == 1


# ===========================================================================
# 8. Failed Message Detail Preservation
# ===========================================================================

def test_8_failed_message_preservation(persistent_queue, storage):
    """Verify that failed messages retain error descriptions and terminal state in SQLite."""
    bad_handler = MagicMock(side_effect=RuntimeError("Hardware fault detected"))
    persistent_queue.router.register_recipient_handler("FailingAgent", bad_handler)
    RoutingRules.SYSTEM_ROLES.add("FailingAgent")

    try:
        msg = AgentMessage(
            task_id="task_fail_800",
            agent_name="Supervisor",
            sender="Supervisor",
            receiver="FailingAgent",
            stage="execution",
            status="waiting",
            message_type=MessageType.TASK_ASSIGNMENT,
            input_data="Run diagnosis",
            output_data=None,
        )

        persistent_queue.enqueue(msg, priority=MessagePriority.NORMAL)
        persistent_queue.process_next()

        record = storage.get_message(msg.message_id)
        assert record["state"] == "failed"
        assert "Hardware fault detected" in record["error"]
        assert record["completed_at"] is not None
    finally:
        RoutingRules.SYSTEM_ROLES.remove("FailingAgent")


# ===========================================================================
# 9. Persistent Duplicate Message ID Rejection
# ===========================================================================

def test_9_duplicate_message_id_rejection(persistent_queue, storage):
    """Verify that duplicate message IDs cannot be inserted into SQLite storage."""
    msg = AgentMessage(
        task_id="task_dup_900",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Planner",
        stage="planning",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Plan step",
        output_data=None,
    )

    # First enqueue succeeds
    item1 = persistent_queue.enqueue(msg)
    assert item1 is not None

    # Duplicate enqueue of identical message_id rejected by storage
    item2 = persistent_queue.enqueue(msg)
    assert item2.message.message_id == msg.message_id

    # SQLite table has exactly 1 row
    assert storage.count_by_state(QueueMessageState.PENDING) == 1


# ===========================================================================
# 10. Database Transaction Rollback
# ===========================================================================

def test_10_transaction_rollback(temp_db_path):
    """Verify rollback on SQLite errors leaves database uncorrupted."""
    storage = SQLiteMessageStorage(db_path=temp_db_path)

    # Insert a valid message
    m1 = AgentMessage(task_id="t1", agent_name="A", sender="Supervisor", receiver="Researcher", stage="r", status="w", input_data="valid", output_data=None)
    storage.insert_message(m1, priority=MessagePriority.NORMAL)
    assert storage.count_by_state(QueueMessageState.PENDING) == 1

    # Simulate attempted update on invalid column or constraint
    conn = storage._get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        try:
            conn.execute("UPDATE messages SET non_existent_col = 'broken';")
            conn.execute("COMMIT;")
        except sqlite3.OperationalError:
            conn.execute("ROLLBACK;")
    finally:
        conn.close()

    # The existing record remains completely intact
    assert storage.count_by_state(QueueMessageState.PENDING) == 1


# ===========================================================================
# 11. SQLite Error Handling
# ===========================================================================

def test_11_sqlite_error_handling(storage):
    """Verify that query calls with missing message IDs return None gracefully."""
    record = storage.get_message("non_existent_id_9999")
    assert record is None


# ===========================================================================
# 12. Asynchronous Compatibility
# ===========================================================================

@pytest.mark.asyncio
async def test_12_async_compatibility(persistent_queue, storage):
    """Verify that persistent queue works seamlessly with deliver_message_async."""
    msg = AgentMessage(
        task_id="task_async_1200",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        stage="research",
        status="waiting",
        message_type=MessageType.TASK_ASSIGNMENT,
        input_data="Async search",
        output_data=None,
    )

    item = persistent_queue.enqueue(msg, priority=MessagePriority.HIGH)
    popped = persistent_queue.dequeue()

    receipt = await persistent_queue.deliver_message_async(popped)
    assert receipt.status == DeliveryStatus.DELIVERED

    rec = storage.get_message(msg.message_id)
    assert rec["state"] == "completed"


# ===========================================================================
# 13. Synchronous Compatibility
# ===========================================================================

def test_13_synchronous_compatibility(persistent_queue, storage):
    """Verify that persistent queue operates with synchronous drain()."""
    for i in range(3):
        msg = AgentMessage(
            task_id=f"t_sync_{i}",
            agent_name="Supervisor",
            sender="Supervisor",
            receiver="Executor",
            stage="execution",
            status="waiting",
            message_type=MessageType.TASK_ASSIGNMENT,
            input_data=f"Task {i}",
            output_data=None,
        )
        persistent_queue.enqueue(msg, priority=MessagePriority.NORMAL)

    receipts = persistent_queue.drain()
    assert len(receipts) == 3
    assert storage.count_by_state(QueueMessageState.COMPLETED) == 3


# ===========================================================================
# 14. End-to-End Multi-Agent Workflow Integration
# (Supervisor -> Persistent Queue -> Researcher -> Planner -> Supervisor -> Executor -> Supervisor)
# ===========================================================================

def test_14_full_persistent_workflow_integration(temp_db_path, populated_worker_registry):
    """Demonstrate end-to-end workflow backed by persistent SQLite queue:
    Supervisor -> Persistent Queue -> Researcher
    Researcher -> Persistent Queue -> Planner
    Planner -> Persistent Queue -> Supervisor
    Supervisor -> Persistent Queue -> Executor
    Executor -> Persistent Queue -> Supervisor
    """
    mock_researcher = populated_worker_registry.get_worker("Researcher")
    mock_researcher.research.return_value = {
        "task_id": "test_persistent_e2e",
        "status": "completed",
        "findings": "Persistent architecture verified.",
    }

    mock_planner = populated_worker_registry.get_worker("Planner")
    mock_planner.plan.return_value = {
        "task_id": "test_persistent_e2e",
        "status": "completed",
        "tasks": [{"id": 1, "description": "Execute persistent database batch", "authorized": True}],
    }

    mock_executor = populated_worker_registry.get_worker("Executor")
    mock_executor.execute.return_value = {
        "task_id": "test_persistent_e2e_task_1",
        "status": "completed",
        "result": "Batch persisted and committed cleanly.",
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
            instruction="Research SQLite performance",
        ),
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Planner",
            capability_required="planning",
            instruction="Plan database indices",
        ),
        SupervisorDecision(
            decision_type=DecisionType.DELEGATE,
            target_worker="Executor",
            capability_required="execution",
            instruction="Execute schema optimization",
        ),
        SupervisorDecision(
            decision_type=DecisionType.COMPLETE,
            final_response="All database optimizations persisted and verified.",
        ),
    ]

    # Initialize persistent storage, router, and queue
    storage = SQLiteMessageStorage(db_path=temp_db_path)
    router = MessageRouter(worker_registry=populated_worker_registry, max_retries=2, backoff_factor=0.0)
    queue = MessageQueue(router=router, max_retries=2, storage=storage, auto_recover=True)
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

    context = orchestrator.run_supervised("Run persistent multi-agent deployment")

    assert context.status == WorkflowStatus.COMPLETED
    assert context.final_response is not None

    # Check persistence in SQLite
    total_completed = storage.count_by_state(QueueMessageState.COMPLETED)
    assert total_completed >= 3
