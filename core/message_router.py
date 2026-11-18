"""
core/message_router.py

Agent Message Router and Controlled Delivery Subsystem for Day 226 Part 2.

Implements:
1. DeliveryStatus: PENDING, DELIVERED, FAILED
2. DeliveryReceipt: Tracks per-message delivery state, timestamp, error, attempts, and latency
3. RoutingRules: Validates routing constraints for TASK_ASSIGNMENT, STAGE_HANDOFF,
   RESULT_RESPONSE / TASK_RESULT, ERROR_ALERT, STATUS_UPDATE, etc.
4. MessageRouter: Central message router responsible for delivering AgentMessage objects to
   registered worker and system recipients with:
   - WorkerRegistry recipient validation
   - Tracking PENDING, DELIVERED, FAILED states
   - Correlation and Task ID preservation
   - Duplicate delivery prevention (idempotency per message_id)
   - Integrated failure classification and controlled retry (respecting retry limits,
     never retrying authorization failures, never duplicating non-idempotent operations)
   - Delivery history and audit trail preservation without discarding messages
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.orchestrator import AgentMessage, MessageType, TaskContext
from core.failure_recovery import (
    WorkerRegistry,
    FailureClassifier,
    FailureType,
    is_non_idempotent_operation,
)
from utils.logging_config import get_logger

logger = get_logger()


# ===========================================================================
# 1. Delivery Status and Receipt
# ===========================================================================

class DeliveryStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass
class DeliveryReceipt:
    """Receipt recording the delivery outcome of an AgentMessage."""
    message_id: str
    task_id: str
    sender: str
    receiver: str
    message_type: str
    status: DeliveryStatus
    correlation_id: Optional[str] = None
    delivery_timestamp: Optional[str] = None
    attempts: int = 1
    error: Optional[str] = None
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "task_id": self.task_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "status": self.status.value if isinstance(self.status, DeliveryStatus) else str(self.status),
            "correlation_id": self.correlation_id,
            "delivery_timestamp": self.delivery_timestamp,
            "attempts": self.attempts,
            "error": self.error,
            "latency_seconds": self.latency_seconds,
            "metadata": self.metadata,
        }


# ===========================================================================
# 2. Routing Rules
# ===========================================================================

class RoutingRules:
    """Enforces routing policies based on MessageType and sender/receiver roles."""

    SYSTEM_ROLES = {
        "Supervisor",
        "SupervisorAgent",
        "WorkflowOrchestrator",
        "Orchestrator",
        "ApprovalGate",
        "Synthesizer",
        "User",
    }

    @classmethod
    def validate_route(
        cls,
        message: AgentMessage,
        worker_registry: Optional[WorkerRegistry] = None,
        recipient_handlers: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validate whether a message adheres to routing rules and identity constraints."""
        # 1. Message integrity
        valid, err = message.validate(worker_registry=None)
        if not valid:
            return False, err

        msg_type = (
            message.message_type.value
            if isinstance(message.message_type, MessageType)
            else str(message.message_type)
        )

        sender = message.sender
        receiver = message.receiver

        # 2. Specific MessageType Routing Rules
        if msg_type == MessageType.TASK_ASSIGNMENT.value:
            # Task assignment should typically originate from a supervisor/orchestrator or delegator
            if not receiver:
                return False, "TASK_ASSIGNMENT requires a non-empty recipient."

        elif msg_type == MessageType.STAGE_HANDOFF.value:
            # Stage handoff must have both sender and receiver specified
            if not receiver:
                return False, "STAGE_HANDOFF requires a target recipient stage or agent."

        elif msg_type in (MessageType.RESULT_RESPONSE.value, MessageType.TASK_RESULT.value):
            # Result response should typically be directed to supervisor or requesting agent
            if not receiver:
                return False, "RESULT_RESPONSE requires a target recipient."

        elif msg_type == MessageType.ERROR_ALERT.value:
            # Error alert should be directed to supervisor/orchestrator
            if not receiver:
                return False, "ERROR_ALERT requires a designated receiver."

        elif msg_type == MessageType.STATUS_UPDATE.value:
            # Status update must have receiver
            if not receiver:
                return False, "STATUS_UPDATE requires a designated receiver."

        # 3. Recipient and Sender existence check
        # A recipient or sender is valid if it is a system role, has a registered handler, or is in worker_registry
        handlers = recipient_handlers or {}
        if worker_registry is not None:
            if receiver and receiver not in cls.SYSTEM_ROLES and receiver not in handlers:
                if not worker_registry.has_worker(receiver):
                    return False, f"Routing rejected: Recipient '{receiver}' is not registered in WorkerRegistry."
            if sender and sender not in cls.SYSTEM_ROLES and sender not in handlers:
                if not worker_registry.has_worker(sender):
                    return False, f"Routing rejected: Sender '{sender}' is not registered in WorkerRegistry."

        return True, None


# ===========================================================================
# 3. Message Router & Delivery System
# ===========================================================================

class MessageRouter:
    """Centralized message router and controlled delivery engine.

    Responsibilities:
    - Route AgentMessage objects between Supervisor and Workers, and between Workers.
    - Validate recipients using WorkerRegistry and prevent unauthorized or unintended deliveries.
    - Track PENDING, DELIVERED, and FAILED states with precise timestamps and latencies.
    - Prevent duplicate delivery of already-delivered messages (idempotent delivery).
    - Coordinate with FailureClassifier to retry only eligible failures while respecting
      retry limits, never retrying authorization failures, and preventing duplicate
      execution of non-idempotent operations.
    - Preserve full message history and delivery audit logs without discarding records.
    """

    def __init__(
        self,
        worker_registry: Optional[WorkerRegistry] = None,
        max_retries: int = 2,
        backoff_factor: float = 0.05,
    ):
        self.worker_registry = worker_registry
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Delivery tracking
        self._delivered_message_ids: Set[str] = set()
        self._delivery_receipts: List[DeliveryReceipt] = []
        self._message_history: List[AgentMessage] = []
        self._recipient_handlers: Dict[str, Callable[[AgentMessage], Any]] = {}
        self._queue: Optional[Any] = None

    def attach_queue(self, queue: Any) -> None:
        """Attach a MessageQueue for asynchronous or queued routing."""
        self._queue = queue

    def get_attached_queue(self) -> Optional[Any]:
        """Return the attached MessageQueue if any."""
        return self._queue

    # -----------------------------------------------------------------------
    # Handler Registration
    # -----------------------------------------------------------------------

    def register_recipient_handler(self, recipient_name: str, handler: Callable[[AgentMessage], Any]) -> None:
        """Register a message receiver callback for a specific agent name or role."""
        self._recipient_handlers[recipient_name] = handler
        logger.debug(f"[MessageRouter] Registered handler for recipient: {recipient_name}")

    def unregister_recipient_handler(self, recipient_name: str) -> None:
        """Remove a recipient callback handler."""
        self._recipient_handlers.pop(recipient_name, None)

    # -----------------------------------------------------------------------
    # Routing and Delivery
    # -----------------------------------------------------------------------

    def route(
        self,
        message: AgentMessage,
        context: Optional[TaskContext] = None,
        recipient_override: Optional[Any] = None,
        use_queue: bool = True,
    ) -> DeliveryReceipt:
        """Route and deliver an AgentMessage with validation, retry, and state tracking.

        If use_queue is True and a queue is attached, enqueues the message and drains/returns
        the resulting DeliveryReceipt.
        """
        if use_queue and self._queue is not None:
            item = self._queue.enqueue(message)
            receipt = self._queue.process_next(context=context)
            if receipt is not None:
                return receipt
        start_time = time.perf_counter()

        # Enforce deep copy of message to preserve historical immutability
        msg_copy = copy.deepcopy(message)

        # 0. Multi-Agent Security Validation (Day 230)
        try:
            from core.security import get_active_security_manager, SecurityDecision
            sec_mgr = get_active_security_manager()
            if sec_mgr is not None and getattr(sec_mgr, "multi_agent", None) is not None:
                expected_tid = context.task_id if context else None
                sec_dec, sec_reason = sec_mgr.multi_agent.validate_message(msg_copy, expected_task_id=expected_tid)
                if sec_dec == SecurityDecision.DENY:
                    logger.warning(f"[MessageRouter] Security rejected message {msg_copy.message_id}: {sec_reason}")
                    receipt = DeliveryReceipt(
                        message_id=msg_copy.message_id,
                        task_id=msg_copy.task_id,
                        sender=msg_copy.sender,
                        receiver=msg_copy.receiver,
                        message_type=str(msg_copy.message_type),
                        status=DeliveryStatus.FAILED,
                        correlation_id=msg_copy.correlation_id,
                        delivery_timestamp=datetime.now(timezone.utc).isoformat(),
                        attempts=1,
                        error=f"Security check failed: {sec_reason}",
                        latency_seconds=time.perf_counter() - start_time,
                    )
                    self._record_message(msg_copy, receipt, context)
                    return receipt
        except Exception as sec_err:
            logger.debug(f"[MessageRouter] Security check skipped or failed: {sec_err}")

        # 1. Routing Rule & Integrity Validation
        is_valid_route, route_err = RoutingRules.validate_route(
            message=msg_copy,
            worker_registry=self.worker_registry,
            recipient_handlers=self._recipient_handlers,
        )
        if not is_valid_route:
            logger.warning(f"[MessageRouter] Message {msg_copy.message_id} failed route validation: {route_err}")
            receipt = DeliveryReceipt(
                message_id=msg_copy.message_id,
                task_id=msg_copy.task_id,
                sender=msg_copy.sender,
                receiver=msg_copy.receiver,
                message_type=str(msg_copy.message_type),
                status=DeliveryStatus.FAILED,
                correlation_id=msg_copy.correlation_id,
                delivery_timestamp=datetime.now(timezone.utc).isoformat(),
                attempts=1,
                error=route_err,
                latency_seconds=time.perf_counter() - start_time,
            )
            self._record_message(msg_copy, receipt, context)
            return receipt

        # 2. Duplicate Delivery Protection (Idempotent delivery)
        if msg_copy.message_id in self._delivered_message_ids:
            logger.warning(
                f"[MessageRouter] Duplicate delivery rejected: message {msg_copy.message_id} was already delivered."
            )
            receipt = DeliveryReceipt(
                message_id=msg_copy.message_id,
                task_id=msg_copy.task_id,
                sender=msg_copy.sender,
                receiver=msg_copy.receiver,
                message_type=str(msg_copy.message_type),
                status=DeliveryStatus.DELIVERED,
                correlation_id=msg_copy.correlation_id,
                delivery_timestamp=datetime.now(timezone.utc).isoformat(),
                attempts=0,
                error=None,
                latency_seconds=time.perf_counter() - start_time,
                metadata={"duplicate_suppressed": True},
            )
            return receipt

        # 3. Resolve Target Recipient Object or Handler
        recipient_target = recipient_override
        if recipient_target is None and msg_copy.receiver in self._recipient_handlers:
            recipient_target = self._recipient_handlers[msg_copy.receiver]
        elif recipient_target is None and self.worker_registry is not None:
            recipient_target = self.worker_registry.get_worker(msg_copy.receiver)

        # 4. Controlled Delivery Execution with Eligible Retry
        receipt = self._deliver_with_retry(
            message=msg_copy,
            recipient=recipient_target,
            start_time=start_time,
        )

        # 5. History & Audit Logging
        self._record_message(msg_copy, receipt, context)
        return receipt

    def _deliver_with_retry(
        self,
        message: AgentMessage,
        recipient: Any,
        start_time: float,
    ) -> DeliveryReceipt:
        """Execute message delivery with controlled failure recovery and idempotency checks."""
        # Non-idempotent operation check
        is_non_idempotent = False
        payload_desc = ""
        if isinstance(message.input_data, str):
            payload_desc = message.input_data
        elif isinstance(message.input_data, dict):
            payload_desc = str(message.input_data.get("description", "") or message.input_data.get("query", ""))
        is_non_idempotent = is_non_idempotent_operation(payload_desc)

        max_attempts = 1 + (0 if is_non_idempotent else max(0, self.max_retries))
        attempts_performed = 0
        last_error: Optional[str] = None
        delivery_succeeded = False

        for attempt in range(1, max_attempts + 1):
            attempts_performed += 1

            if attempt > 1:
                if self.backoff_factor > 0:
                    time.sleep(self.backoff_factor * (2 ** (attempt - 2)))
                logger.info(
                    f"[MessageRouter] Retrying delivery of message {message.message_id} to '{message.receiver}' (attempt {attempt}/{max_attempts})"
                )

            try:
                self._dispatch_to_recipient(message, recipient)
                delivery_succeeded = True
                break
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    f"[MessageRouter] Delivery exception for message {message.message_id} to '{message.receiver}': {last_error}"
                )

                # Failure classification
                failure = FailureClassifier.classify(
                    error=exc,
                    stage=message.stage or "delivery",
                    agent_name=message.receiver,
                    task_id=message.task_id,
                )

                # Never retry authorization failures
                if failure.failure_type == FailureType.AUTHORIZATION:
                    logger.warning(
                        f"[MessageRouter] Authorization failure for message {message.message_id}; aborting retries."
                    )
                    break

                # Respect failure classification retry eligibility
                if not failure.is_retryable:
                    logger.warning(
                        f"[MessageRouter] Non-retryable {failure.failure_type.value} failure for message {message.message_id}; aborting retries."
                    )
                    break

                # Never retry non-idempotent operations
                if is_non_idempotent:
                    logger.warning(
                        f"[MessageRouter] Message {message.message_id} is non-idempotent; stopping after first failure."
                    )
                    break

        latency = time.perf_counter() - start_time
        now_ts = datetime.now(timezone.utc).isoformat()

        if delivery_succeeded:
            self._delivered_message_ids.add(message.message_id)
            return DeliveryReceipt(
                message_id=message.message_id,
                task_id=message.task_id,
                sender=message.sender,
                receiver=message.receiver,
                message_type=str(message.message_type),
                status=DeliveryStatus.DELIVERED,
                correlation_id=message.correlation_id,
                delivery_timestamp=now_ts,
                attempts=attempts_performed,
                error=None,
                latency_seconds=latency,
            )
        else:
            return DeliveryReceipt(
                message_id=message.message_id,
                task_id=message.task_id,
                sender=message.sender,
                receiver=message.receiver,
                message_type=str(message.message_type),
                status=DeliveryStatus.FAILED,
                correlation_id=message.correlation_id,
                delivery_timestamp=now_ts,
                attempts=attempts_performed,
                error=last_error or "Delivery failed",
                latency_seconds=latency,
            )

    def _dispatch_to_recipient(self, message: AgentMessage, recipient: Any) -> Any:
        """Dispatch message directly to callable handler or agent mailbox/interface."""
        if recipient is None:
            # If no active invocation target is bound, deliver to mailbox / passive record
            # (Standard for passive system roles or simulated message routing)
            return True

        if callable(recipient):
            # Recipient callback handler
            return recipient(message)

        # If recipient is an agent instance
        if hasattr(recipient, "receive_message") and callable(recipient.receive_message):
            return recipient.receive_message(message)
        elif hasattr(recipient, "handle_message") and callable(recipient.handle_message):
            return recipient.handle_message(message)

        return True

    def _record_message(
        self,
        message: AgentMessage,
        receipt: DeliveryReceipt,
        context: Optional[TaskContext],
    ) -> None:
        """Store message and receipt in router history and update context if provided."""
        self._message_history.append(message)
        self._delivery_receipts.append(receipt)

        if context is not None:
            # Record receipt metadata on message
            message.metadata["delivery_receipt"] = receipt.to_dict()
            context.add_message(message)

    # -----------------------------------------------------------------------
    # Query and Audit Helpers
    # -----------------------------------------------------------------------

    def get_receipt(self, message_id: str) -> Optional[DeliveryReceipt]:
        """Find the delivery receipt for a specific message ID."""
        for r in self._delivery_receipts:
            if r.message_id == message_id:
                return r
        return None

    def get_receipts_by_task(self, task_id: str) -> List[DeliveryReceipt]:
        """Find all delivery receipts associated with a task ID."""
        return [r for r in self._delivery_receipts if r.task_id == task_id]

    def get_receipts_by_status(self, status: DeliveryStatus) -> List[DeliveryReceipt]:
        """Find all delivery receipts with a given DeliveryStatus."""
        target = status.value if isinstance(status, DeliveryStatus) else str(status)
        return [r for r in self._delivery_receipts if (r.status.value if isinstance(r.status, DeliveryStatus) else str(r.status)) == target]

    def get_messages_for_receiver(self, receiver: str) -> List[AgentMessage]:
        """Retrieve all routed messages addressed to a given receiver."""
        return [m for m in self._message_history if m.receiver == receiver]

    def get_messages_by_correlation(self, correlation_id: str) -> List[AgentMessage]:
        """Retrieve all routed messages linked to a correlation ID."""
        return [m for m in self._message_history if m.correlation_id == correlation_id]

    def clear_history(self) -> None:
        """Reset internal history and delivered message caches."""
        self._delivered_message_ids.clear()
        self._delivery_receipts.clear()
        self._message_history.clear()
