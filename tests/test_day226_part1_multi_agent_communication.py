"""Test Suite for Day 226 Part 1: Multi-Agent Communication Architecture.

Validates:
1. Valid Supervisor-to-Worker communication and delegation contracts.
2. Valid Worker-to-Worker handoff and stage transition contracts.
3. Invalid sender or receiver detection using WorkerRegistry.
4. Missing required message fields rejection.
5. Incorrect or corrupted task ID and correlation ID tracking.
6. Malformed payload and error payload communication preservation.
7. Message correlation querying and audit retrieval from TaskContext.
8. Context preservation, immutability, and state defense against external mutations.
9. Communication error handling in workflow pipelines.
10. Backward compatibility with existing AgentMessage signatures and usage patterns.
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
from day223_agents.core.task_manager import TaskManager, TaskSpecification, TaskStatus
from day223_agents.core.failure_recovery import WorkerRegistry
from day223_agents.agents.supervisor import (
    SupervisorAgent,
    DecisionType,
    SupervisorDecision,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def worker_registry():
    """Populated registry matching the system's standard worker roster."""
    registry = WorkerRegistry()
    mock_researcher = MagicMock()
    mock_researcher.name = "Researcher"
    mock_planner = MagicMock()
    mock_planner.name = "Planner"
    mock_executor = MagicMock()
    mock_executor.name = "Executor"

    registry.register(mock_researcher, capabilities=["research", "web_search", "document_retrieval"], priority=10, name="Researcher")
    registry.register(mock_planner, capabilities=["planning", "task_decomposition", "dependency_analysis"], priority=10, name="Planner")
    registry.register(mock_executor, capabilities=["execution", "code_generation", "tool_execution"], priority=10, name="Executor")
    return registry


# ===========================================================================
# 1. Valid Supervisor-to-Worker Communication
# ===========================================================================

def test_1_valid_supervisor_to_worker_communication(worker_registry):
    """Verify that Supervisor-to-Worker delegation produces a valid, validated TASK_ASSIGNMENT message."""
    msg = AgentMessage(
        task_id="task_s2w_001",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        message_type=MessageType.TASK_ASSIGNMENT,
        stage="research",
        status="waiting",
        input_data={"query": "Evaluate distributed multi-agent consensus algorithms"},
        output_data=None,
        correlation_id="task_s2w_001",
    )

    # Validate against registry
    valid, err = msg.validate(worker_registry)
    assert valid is True
    assert err is None

    # Assert standard contracts
    assert msg.sender == "Supervisor"
    assert msg.receiver == "Researcher"
    assert msg.message_type == MessageType.TASK_ASSIGNMENT.value
    assert msg.correlation_id == "task_s2w_001"
    assert msg.message_id.startswith("msg_")
    assert msg.timestamp is not None

    # Verify serialization
    data = msg.to_dict()
    assert data["sender"] == "Supervisor"
    assert data["receiver"] == "Researcher"
    assert data["message_type"] == "task_assignment"


# ===========================================================================
# 2. Valid Worker-to-Worker Handoff
# ===========================================================================

def test_2_valid_worker_to_worker_handoff(worker_registry):
    """Verify that Worker-to-Worker handoff creates a validated STAGE_HANDOFF message."""
    handoff_msg = AgentMessage(
        task_id="task_w2w_002",
        agent_name="Researcher",
        sender="Researcher",
        receiver="Planner",
        message_type=MessageType.STAGE_HANDOFF,
        stage="planning",
        status="success",
        input_data={"raw_findings": "Consensus requires leader election."},
        output_data={"research_summary": "Leader election mechanisms analyzed."},
        correlation_id="task_w2w_002",
    )

    valid, err = handoff_msg.validate(worker_registry)
    assert valid is True
    assert err is None
    assert handoff_msg.sender == "Researcher"
    assert handoff_msg.receiver == "Planner"
    assert handoff_msg.message_type == MessageType.STAGE_HANDOFF.value
    assert handoff_msg.status == "success"


# ===========================================================================
# 3. Invalid Sender or Receiver Validation
# ===========================================================================

def test_3_invalid_sender_or_receiver_validation(worker_registry):
    """Verify that validate() detects and rejects unregistered senders or receivers."""
    # Unknown receiver
    msg_bad_receiver = AgentMessage(
        task_id="task_inv_003",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="NonExistentWorker",
        stage="execution",
        status="waiting",
        input_data={},
        output_data={},
    )
    valid, err = msg_bad_receiver.validate(worker_registry)
    assert valid is False
    assert "receiver 'NonExistentWorker' is not a registered worker" in err

    # Unknown sender
    msg_bad_sender = AgentMessage(
        task_id="task_inv_004",
        agent_name="RogueAgent",
        sender="RogueAgent",
        receiver="Executor",
        stage="execution",
        status="waiting",
        input_data={},
        output_data={},
    )
    valid, err = msg_bad_sender.validate(worker_registry)
    assert valid is False
    assert "sender 'RogueAgent' is not a registered worker" in err


# ===========================================================================
# 4. Missing Required Message Fields
# ===========================================================================

def test_4_missing_required_message_fields(worker_registry):
    """Verify that validate() rejects messages with missing message_id, task_id, or sender."""
    # Missing message_id
    msg_no_id = AgentMessage(
        task_id="task_req_005",
        agent_name="Planner",
        sender="Planner",
        stage="planning",
        status="success",
        input_data={},
        output_data={},
    )
    msg_no_id.message_id = ""
    valid, err = msg_no_id.validate(worker_registry)
    assert valid is False
    assert "message_id cannot be empty" in err

    # Missing task_id
    msg_no_task = AgentMessage(
        task_id="",
        agent_name="Planner",
        sender="Planner",
        stage="planning",
        status="success",
        input_data={},
        output_data={},
    )
    valid, err = msg_no_task.validate(worker_registry)
    assert valid is False
    assert "task_id cannot be empty" in err

    # Missing sender
    msg_no_sender = AgentMessage(
        task_id="task_req_006",
        agent_name="",
        sender="",
        stage="planning",
        status="success",
        input_data={},
        output_data={},
    )
    # Even if default sets "Agent", explicitly blanking it triggers validation
    msg_no_sender.sender = ""
    valid, err = msg_no_sender.validate(worker_registry)
    assert valid is False
    assert "sender cannot be empty" in err


# ===========================================================================
# 5. Incorrect or Corrupted Task ID and Correlation ID Tracking
# ===========================================================================

def test_5_incorrect_task_id_and_correlation():
    """Verify correlation ID tracking and separation across nested or related tasks."""
    parent_task_id = "parent_workflow_100"
    subtask_id = "child_subtask_101"

    # Message associated with subtask but correlated back to parent
    msg = AgentMessage(
        task_id=subtask_id,
        agent_name="Executor",
        sender="Executor",
        receiver="Supervisor",
        stage="execution",
        status="success",
        input_data={"command": "compile"},
        output_data={"result": "build passed"},
        correlation_id=parent_task_id,
    )

    assert msg.task_id == "child_subtask_101"
    assert msg.correlation_id == "parent_workflow_100"

    # Serialized and deserialized correctly
    d = msg.to_dict()
    reconstructed = AgentMessage.from_dict(d)
    assert reconstructed.task_id == subtask_id
    assert reconstructed.correlation_id == parent_task_id


# ===========================================================================
# 6. Malformed Payload and Error Payload Communication
# ===========================================================================

def test_6_malformed_payload_and_error_payload():
    """Verify that error alerts and failure states are captured cleanly with structured error details."""
    error_msg = AgentMessage(
        task_id="task_err_007",
        agent_name="Executor",
        sender="Executor",
        receiver="Supervisor",
        message_type=MessageType.ERROR_ALERT,
        stage="execution",
        status="failed",
        input_data={"code": "undefined_symbol()"},
        output_data=None,
        error="NameError: name 'undefined_symbol' is not defined",
        metadata={"traceback_depth": 3, "retryable": False},
    )

    assert error_msg.status == "failed"
    assert error_msg.message_type == "error_alert"
    assert error_msg.error is not None
    assert "NameError" in error_msg.error
    assert error_msg.metadata.get("retryable") is False

    # Ensure round-trip serialization retains error fields
    data = error_msg.to_dict()
    assert data["status"] == "failed"
    assert data["error"] == error_msg.error
    restored = AgentMessage.from_dict(data)
    assert restored.error == error_msg.error
    assert restored.status == "failed"


# ===========================================================================
# 7. Message Correlation Querying and Audit Trail in TaskContext
# ===========================================================================

def test_7_message_correlation_querying_in_context():
    """Verify that TaskContext allows querying messages by correlation_id, sender, and receiver."""
    context = TaskContext(workflow_id="wf_query_001", user_request="Analyze cluster logs")

    m1 = AgentMessage(
        task_id="task_1",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Researcher",
        correlation_id="root_workflow",
        stage="research",
        status="waiting",
        input_data="query",
        output_data=None,
    )
    m2 = AgentMessage(
        task_id="task_2",
        agent_name="Researcher",
        sender="Researcher",
        receiver="Planner",
        correlation_id="root_workflow",
        stage="research",
        status="success",
        input_data="query",
        output_data="logs extracted",
    )
    m3 = AgentMessage(
        task_id="unrelated_task",
        agent_name="Supervisor",
        sender="Supervisor",
        receiver="Executor",
        correlation_id="other_workflow",
        stage="execution",
        status="waiting",
        input_data="run",
        output_data=None,
    )

    context.add_message(m1)
    context.add_message(m2)
    context.add_message(m3)

    # Query by correlation ID
    root_messages = context.get_messages_by_correlation("root_workflow")
    assert len(root_messages) == 2
    assert root_messages[0].sender == "Supervisor"
    assert root_messages[1].sender == "Researcher"

    # Query by sender
    supervisor_msgs = context.get_messages_by_sender("Supervisor")
    assert len(supervisor_msgs) == 2

    # Query by receiver
    planner_msgs = context.get_messages_by_receiver("Planner")
    assert len(planner_msgs) == 1
    assert planner_msgs[0].output_data == "logs extracted"


# ===========================================================================
# 8. Context Preservation and Immutability Protection
# ===========================================================================

def test_8_context_preservation_and_immutability():
    """Verify that external mutations after add_message do not corrupt stored messages."""
    context = TaskContext(workflow_id="wf_imm_001", user_request="Test immutability")

    mutable_metadata = {"step_id": 1, "flags": ["initial"]}
    mutable_input = {"params": ["a", "b"]}

    msg = AgentMessage(
        task_id="task_imm_008",
        agent_name="Planner",
        sender="Planner",
        receiver="Executor",
        stage="planning",
        status="success",
        input_data=mutable_input,
        output_data={"plan": "step 1"},
        metadata=mutable_metadata,
    )

    # Add message into context
    context.add_message(msg)

    # Mutate external sources and message object post-addition
    mutable_metadata["flags"].append("TAMPERED")
    mutable_metadata["step_id"] = 999
    msg.status = "MUTATED_EXTERNALLY"

    # Verify that stored message in context remained isolated and intact
    stored = context.messages[0]
    assert stored.status == "success"
    assert stored.metadata["step_id"] == 1
    assert "TAMPERED" not in stored.metadata["flags"]


# ===========================================================================
# 9. Communication Error Handling in Workflow
# ===========================================================================

def test_9_communication_error_handling_in_workflow():
    """Verify that agent execution exceptions are captured as structured failed AgentMessages."""
    mock_researcher = MagicMock()
    mock_researcher.name = "Researcher"
    mock_researcher.research.side_effect = RuntimeError("Worker network connection severed")

    mock_planner = MagicMock()
    mock_planner.name = "Planner"
    mock_executor = MagicMock()
    mock_executor.name = "Executor"

    orchestrator = WorkflowOrchestrator(
        researcher=mock_researcher,
        planner=mock_planner,
        executor=mock_executor,
    )

    context = orchestrator.run("Simulate network drop")

    # The workflow should transition to FAILED status
    assert context.status == WorkflowStatus.FAILED
    assert len(context.messages) >= 1

    # The failing message must be recorded with failed status and error details
    failed_msg = context.messages[-1]
    assert failed_msg.status == "failed"
    assert failed_msg.error is not None
    assert "Worker network connection severed" in failed_msg.error
    assert failed_msg.stage == "research"


# ===========================================================================
# 10. Backward Compatibility with Existing AgentMessage Signatures
# ===========================================================================

def test_10_backward_compatibility():
    """Verify that legacy AgentMessage positional/keyword arguments remain 100% functional."""
    # Legacy 6-argument construction
    legacy_msg = AgentMessage(
        task_id="legacy_task_010",
        agent_name="Planner",
        stage="planning",
        status="success",
        input_data={"req": "legacy format"},
        output_data={"result": "works seamlessly"},
    )

    # Check auto-populated defaults
    assert legacy_msg.sender == "Planner"
    assert legacy_msg.receiver == ""
    assert legacy_msg.correlation_id == "legacy_task_010"
    assert legacy_msg.message_type == MessageType.STAGE_HANDOFF.value
    assert legacy_msg.message_id.startswith("msg_")
    assert legacy_msg.timestamp is not None
    assert legacy_msg.metadata == {}

    # Dict round-trip
    d = legacy_msg.to_dict()
    assert d["task_id"] == "legacy_task_010"
    assert d["agent_name"] == "Planner"
    assert d["sender"] == "Planner"

    restored = AgentMessage.from_dict(d)
    assert restored.task_id == legacy_msg.task_id
    assert restored.sender == legacy_msg.sender
    assert restored.status == "success"
