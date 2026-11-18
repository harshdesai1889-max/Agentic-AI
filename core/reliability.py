"""
core/reliability.py

Message Reliability, Acknowledgements, Dead-Letter Queues, and Delivery Guarantees for Day 226 Part 5.

Implements:
1. AckStatus: ACKNOWLEDGED, REJECTED, ACK_TIMEOUT, UNACKNOWLEDGED
2. DeliveryGuarantee: AT_MOST_ONCE, AT_LEAST_ONCE
3. MessageAck: Structured acknowledgement record tracking confirmation, reason, and latency.
4. DLQEntry: Container for permanently failed or retry-exhausted messages in the DeadLetterQueue.
5. ReliabilityMetrics: Tracking total, delivered, acknowledged, failed, retried, and DLQ counts,
   plus delivery latency averages and processing failure counts.
6. DeadLetterQueue: Persistent DLQ storing exhausted messages with authorized replay mechanisms,
   strict non-idempotent operation defenses, and rejection explanations.
7. ReliabilityCoordinator: Coordinates acknowledgements, delivery guarantees, DLQ routing,
   idempotency checks, and metric aggregation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.orchestrator import AgentMessage, MessageType, TaskContext
from core.message_router import DeliveryReceipt, DeliveryStatus
from core.failure_recovery import (
    FailureClassifier,
    FailureType,
    is_non_idempotent_operation,
)
from utils.logging_config import get_logger

logger = get_logger()


# ===========================================================================
# 1. Enums & Core Models
# ===========================================================================

class AckStatus(str, Enum):
    """Acknowledgement state of a delivered message."""
    UNACKNOWLEDGED = "unacknowledged"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    ACK_TIMEOUT = "ack_timeout"


class DeliveryGuarantee(str, Enum):
    """Delivery policy modes for multi-agent messaging."""
    AT_MOST_ONCE = "at_most_once"    # Fast-fire: zero automatic retries, delivery failure drops or fails directly
    AT_LEAST_ONCE = "at_least_once"  # Guaranteed delivery: retries transient errors until ACK or DLQ threshold


@dataclass
class MessageAck:
    """Structured acknowledgement outcome returned by a message recipient."""
    message_id: str
    task_id: str
    recipient: str
    status: AckStatus
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "task_id": self.task_id,
            "recipient": self.recipient,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


@dataclass
class DLQEntry:
    """Entry stored in DeadLetterQueue representing an unresolvable message."""
    dlq_id: str
    message: AgentMessage
    reason: str
    retry_count: int
    entered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    replayed: bool = False
    replay_count: int = 0
    replay_history: List[Dict[str, Any]] = field(default_factory=list)
    failure_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dlq_id": self.dlq_id,
            "message_id": self.message.message_id,
            "task_id": self.message.task_id,
            "correlation_id": self.message.correlation_id,
            "sender": self.message.sender,
            "receiver": self.message.receiver,
            "message": self.message.to_dict(),
            "reason": self.reason,
            "retry_count": self.retry_count,
            "entered_at": self.entered_at,
            "replayed": self.replayed,
            "replay_count": self.replay_count,
            "replay_history": self.replay_history,
            "failure_details": self.failure_details,
        }


# ===========================================================================
# 2. Reliability Metrics Tracker
# ===========================================================================

class ReliabilityMetrics:
    """Thread-safe telemetry tracker for multi-agent message delivery and reliability."""

    def __init__(self):
        self._lock = threading.RLock()
        self.total_messages: int = 0
        self.delivered_messages: int = 0
        self.acknowledged_messages: int = 0
        self.failed_messages: int = 0
        self.retried_messages: int = 0
        self.dead_letter_messages: int = 0
        self.processing_failures: int = 0
        self._latencies: List[float] = []

    def record_message(self) -> None:
        with self._lock:
            self.total_messages += 1

    def record_delivery(self, latency_seconds: float) -> None:
        with self._lock:
            self.delivered_messages += 1
            if latency_seconds >= 0:
                self._latencies.append(latency_seconds)

    def record_ack(self, status: AckStatus) -> None:
        with self._lock:
            if status == AckStatus.ACKNOWLEDGED:
                self.acknowledged_messages += 1
            elif status in (AckStatus.REJECTED, AckStatus.ACK_TIMEOUT):
                self.processing_failures += 1

    def record_retry(self) -> None:
        with self._lock:
            self.retried_messages += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed_messages += 1

    def record_dlq(self) -> None:
        with self._lock:
            self.dead_letter_messages += 1

    def record_processing_failure(self) -> None:
        with self._lock:
            self.processing_failures += 1

    @property
    def average_delivery_latency(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_messages": self.total_messages,
                "delivered_messages": self.delivered_messages,
                "acknowledged_messages": self.acknowledged_messages,
                "failed_messages": self.failed_messages,
                "retried_messages": self.retried_messages,
                "dead_letter_messages": self.dead_letter_messages,
                "processing_failures": self.processing_failures,
                "average_delivery_latency_seconds": round(self.average_delivery_latency, 6),
            }


# ===========================================================================
# 3. Persistent Dead-Letter Queue (DLQ)
# ===========================================================================

class DeadLetterQueue:
    """Persistent DLQ managing unresolvable, exhausted, or rejected messages."""

    def __init__(self, storage: Optional[Any] = None):
        self.storage = storage
        self._lock = threading.RLock()
        self._entries: Dict[str, DLQEntry] = {}
        if self.storage is not None:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Hydrate DLQ entries from persistent SQLite storage on startup."""
        if hasattr(self.storage, "get_all_dlq_entries"):
            entries = self.storage.get_all_dlq_entries()
            for e in entries:
                self._entries[e.dlq_id] = e

    def push(
        self,
        message: AgentMessage,
        reason: str,
        retry_count: int,
        failure_details: Optional[Dict[str, Any]] = None,
    ) -> DLQEntry:
        """Push a message into DLQ with full metadata and audit trail."""
        with self._lock:
            dlq_id = f"dlq_{message.message_id}"
            entry = DLQEntry(
                dlq_id=dlq_id,
                message=copy.deepcopy(message),
                reason=reason,
                retry_count=retry_count,
                failure_details=failure_details or {},
            )
            self._entries[dlq_id] = entry

            if self.storage is not None and hasattr(self.storage, "insert_dlq_entry"):
                self.storage.insert_dlq_entry(entry)

            logger.warning(
                f"[DeadLetterQueue] Message '{message.message_id}' moved to DLQ. Reason: {reason}"
            )
            return entry

    def get(self, dlq_id: str) -> Optional[DLQEntry]:
        """Inspect a specific DLQ entry."""
        with self._lock:
            return self._entries.get(dlq_id)

    def list_entries(self) -> List[DLQEntry]:
        """Query all active DLQ records."""
        with self._lock:
            return list(self._entries.values())

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def replay(
        self,
        dlq_id: str,
        authorized: bool = False,
        authorization_token: Optional[str] = None,
        target_queue: Optional[Any] = None,
    ) -> Tuple[bool, str, Optional[AgentMessage]]:
        """Safely replay an eligible message from the DLQ.

        Enforces:
        1. Explicit authorization requirement.
        2. Non-idempotent operation defense (refuses replay of payments, drops, emails).
        3. Message re-validation before dispatch.
        4. Preservation of original failure and replay audit history.
        """
        with self._lock:
            entry = self._entries.get(dlq_id)
            if not entry:
                return False, f"DLQ entry '{dlq_id}' not found.", None

            # 1. Authorization Gate
            if not authorized or not authorization_token:
                logger.warning(f"[DeadLetterQueue] Replay rejected for {dlq_id}: Unauthorized.")
                return False, "Replay rejected: Explicit authorization and token required.", None

            msg = copy.deepcopy(entry.message)

            # 2. Idempotency Check: Protect against repeated external side-effects
            desc = ""
            if isinstance(msg.input_data, str):
                desc = msg.input_data
            elif isinstance(msg.input_data, dict):
                desc = str(msg.input_data.get("description", "") or msg.input_data.get("query", ""))

            if is_non_idempotent_operation(desc):
                reason = (
                    "Replay rejected: Message represents a non-idempotent operation "
                    "(e.g. payment, external deletion, email) and cannot be safely re-executed."
                )
                logger.warning(f"[DeadLetterQueue] {reason} (DLQ: {dlq_id})")
                return False, reason, None

            # 3. Authorization Failure Pattern Defense
            if "authorization" in str(entry.reason).lower() or "401" in str(entry.reason) or "403" in str(entry.reason):
                return False, "Replay rejected: Security or authorization failures cannot be replayed automatically.", None

            # 4. Record Replay Audit
            now_ts = datetime.now(timezone.utc).isoformat()
            replay_record = {
                "timestamp": now_ts,
                "authorized_by": authorization_token,
                "attempt_number": entry.replay_count + 1,
            }
            entry.replay_history.append(replay_record)
            entry.replayed = True
            entry.replay_count += 1

            if self.storage is not None and hasattr(self.storage, "update_dlq_replay"):
                self.storage.update_dlq_replay(dlq_id, entry.replay_count, now_ts)

            # 5. Dispatch to Queue if provided
            if target_queue is not None:
                # Retain original message identity and correlation
                target_queue.enqueue(msg, priority=msg.priority)

            logger.info(f"[DeadLetterQueue] Replay authorized and queued for {dlq_id}")
            return True, "Replay successfully authorized and enqueued.", msg


# ===========================================================================
# 4. Reliability Coordinator
# ===========================================================================

class ReliabilityCoordinator:
    """Central engine orchestrating acknowledgements, delivery guarantees, DLQ, and idempotency."""

    def __init__(
        self,
        default_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE,
        dlq: Optional[DeadLetterQueue] = None,
        metrics: Optional[ReliabilityMetrics] = None,
        ack_timeout_seconds: float = 5.0,
    ):
        self.delivery_guarantee = default_guarantee
        self.dlq = dlq or DeadLetterQueue()
        self.metrics = metrics or ReliabilityMetrics()
        self.ack_timeout_seconds = ack_timeout_seconds

        self._lock = threading.RLock()
        self._completed_idempotency_keys: Set[str] = set()
        self._uncertain_execution_keys: Set[str] = set()
        self._acknowledgements: Dict[str, MessageAck] = {}

    # -----------------------------------------------------------------------
    # Idempotency & Side-Effect Protection
    # -----------------------------------------------------------------------

    def compute_idempotency_key(self, message: AgentMessage) -> str:
        """Derive a stable idempotency key for an operation."""
        # Use explicit correlation or fallback to task_id + message_id
        return f"idem_{message.task_id}_{message.message_id}"

    def is_operation_completed(self, message: AgentMessage) -> bool:
        """Check whether an operation has already executed successfully."""
        with self._lock:
            key = self.compute_idempotency_key(message)
            return key in self._completed_idempotency_keys

    def mark_operation_completed(self, message: AgentMessage) -> None:
        """Record an operation as completed to prevent duplicate side effects."""
        with self._lock:
            key = self.compute_idempotency_key(message)
            self._completed_idempotency_keys.add(key)
            self._uncertain_execution_keys.discard(key)

    def mark_execution_uncertain(self, message: AgentMessage) -> None:
        """Flag an operation as uncertain after an ungraceful crash or timeout."""
        with self._lock:
            key = self.compute_idempotency_key(message)
            self._uncertain_execution_keys.add(key)

    def is_execution_uncertain(self, message: AgentMessage) -> bool:
        """Check whether an operation's completion is uncertain and requires reconciliation."""
        with self._lock:
            key = self.compute_idempotency_key(message)
            return key in self._uncertain_execution_keys

    # -----------------------------------------------------------------------
    # Acknowledgement Management
    # -----------------------------------------------------------------------

    def record_acknowledgement(
        self,
        message: AgentMessage,
        recipient: str,
        status: AckStatus,
        reason: Optional[str] = None,
    ) -> MessageAck:
        """Process and store a delivery/processing acknowledgement from a worker."""
        with self._lock:
            ack = MessageAck(
                message_id=message.message_id,
                task_id=message.task_id,
                recipient=recipient,
                status=status,
                reason=reason,
                correlation_id=message.correlation_id,
            )
            self._acknowledgements[message.message_id] = ack
            self.metrics.record_ack(status)

            if status == AckStatus.ACKNOWLEDGED:
                self.mark_operation_completed(message)

            logger.info(
                f"[ReliabilityCoordinator] ACK received for {message.message_id}: status={status.value}"
            )
            return ack

    def get_acknowledgement(self, message_id: str) -> Optional[MessageAck]:
        """Fetch the acknowledgement status for a message ID."""
        with self._lock:
            return self._acknowledgements.get(message_id)

    # -----------------------------------------------------------------------
    # DLQ Escalation
    # -----------------------------------------------------------------------

    def route_to_dlq(
        self,
        message: AgentMessage,
        reason: str,
        retry_count: int,
        failure_details: Optional[Dict[str, Any]] = None,
    ) -> DLQEntry:
        """Move permanently failed or unresolvable message into DLQ and update metrics."""
        self.metrics.record_dlq()
        self.metrics.record_failure()
        return self.dlq.push(
            message=message,
            reason=reason,
            retry_count=retry_count,
            failure_details=failure_details,
        )

    def get_monitor(
        self,
        queue: Optional[Any] = None,
        storage: Optional[Any] = None,
        recovery_history: Optional[Any] = None,
    ) -> ReliabilityMonitor:
        """Obtain a consolidated ReliabilityMonitor instance for this coordinator."""
        return ReliabilityMonitor(
            coordinator=self,
            queue=queue,
            storage=storage or self.dlq.storage,
            recovery_history=recovery_history,
        )


# ===========================================================================
# 5. Consolidated Reliability Monitor (Day 226 Part 6)
# ===========================================================================

@dataclass
class RecoveryStatistics:
    """Audit summary of failure recovery attempts and outcomes."""
    total_retries: int = 0
    recovered_count: int = 0
    reassignments_count: int = 0
    active_recoveries: int = 0
    failed_recoveries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_retries": self.total_retries,
            "recovered_count": self.recovered_count,
            "reassignments_count": self.reassignments_count,
            "active_recoveries": self.active_recoveries,
            "failed_recoveries": self.failed_recoveries,
        }


class ReliabilityMonitor:
    """Consolidated Python monitoring and telemetry interface for the multi-agent messaging system.

    Provides a clean, unified programmatic inspection interface across all 10 core dimensions:
    1. Total messages
    2. Pending messages
    3. Delivered messages
    4. Acknowledged messages
    5. Failed messages
    6. Retried messages
    7. Dead-letter messages
    8. Average delivery latency
    9. Processing failures
    10. Recovery statistics
    """

    def __init__(
        self,
        coordinator: ReliabilityCoordinator,
        queue: Optional[Any] = None,
        storage: Optional[Any] = None,
        recovery_history: Optional[Any] = None,
    ):
        self.coordinator = coordinator
        self.queue = queue
        self.storage = storage or (coordinator.dlq.storage if coordinator and coordinator.dlq else None)
        self.recovery_history = recovery_history
        self._recovery_stats = RecoveryStatistics()

    @property
    def total_messages(self) -> int:
        return self.coordinator.metrics.total_messages

    @property
    def pending_messages(self) -> int:
        if self.queue is not None and hasattr(self.queue, "pending_count"):
            return self.queue.pending_count()
        if self.storage is not None and hasattr(self.storage, "count_by_state"):
            from core.message_queue import QueueMessageState
            return self.storage.count_by_state(QueueMessageState.PENDING)
        return 0

    @property
    def delivered_messages(self) -> int:
        return self.coordinator.metrics.delivered_messages

    @property
    def acknowledged_messages(self) -> int:
        return self.coordinator.metrics.acknowledged_messages

    @property
    def failed_messages(self) -> int:
        return self.coordinator.metrics.failed_messages

    @property
    def retried_messages(self) -> int:
        return self.coordinator.metrics.retried_messages

    @property
    def dead_letter_messages(self) -> int:
        return self.coordinator.dlq.size()

    @property
    def average_delivery_latency(self) -> float:
        return self.coordinator.metrics.average_delivery_latency

    @property
    def average_delivery_latency_seconds(self) -> float:
        return self.coordinator.metrics.average_delivery_latency

    @property
    def processing_failures(self) -> int:
        return self.coordinator.metrics.processing_failures

    @property
    def recovery_statistics(self) -> Dict[str, Any]:
        if self.recovery_history is not None:
            return {
                "total_retries": getattr(self.recovery_history, "total_retries", 0),
                "recovered_count": 1 if getattr(self.recovery_history, "recovered", False) else 0,
                "reassignments_count": len(getattr(self.recovery_history, "reassignments", [])),
                "active_recoveries": 0,
                "failed_recoveries": 1 if getattr(self.recovery_history, "final_status", "") == "failed" else 0,
            }
        return self._recovery_stats.to_dict()

    def record_recovery(
        self,
        recovered: bool = True,
        reassignments: int = 0,
        retries: int = 0,
    ) -> None:
        """Record manual or observed recovery telemetry."""
        self._recovery_stats.total_retries += retries
        self._recovery_stats.reassignments_count += reassignments
        if recovered:
            self._recovery_stats.recovered_count += 1
        else:
            self._recovery_stats.failed_recoveries += 1

    def inspect(self) -> Dict[str, Any]:
        """Consolidated inspection dictionary reporting all 10 core telemetry dimensions."""
        return {
            "total_messages": self.total_messages,
            "pending_messages": self.pending_messages,
            "delivered_messages": self.delivered_messages,
            "acknowledged_messages": self.acknowledged_messages,
            "failed_messages": self.failed_messages,
            "retried_messages": self.retried_messages,
            "dead_letter_messages": self.dead_letter_messages,
            "average_delivery_latency": round(self.average_delivery_latency, 6),
            "average_delivery_latency_seconds": round(self.average_delivery_latency, 6),
            "processing_failures": self.processing_failures,
            "recovery_statistics": self.recovery_statistics,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Alias for inspect()."""
        return self.inspect()

