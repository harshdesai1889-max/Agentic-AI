"""
core/message_queue.py

Message Queues, Prioritization and Asynchronous Processing Subsystem for Day 226 Part 3.

Implements:
1. MessagePriority: CRITICAL (0), HIGH (1), NORMAL (2), LOW (3)
2. QueueMessageState: PENDING, PROCESSING, COMPLETED, FAILED
3. QueueItem: Priority-heap compatible envelope tracking sequence, priority, retry counters,
   and aging to prevent lower-priority starvation.
4. MessageQueue: Centralized thread-safe and asyncio-compatible priority message queue.
   - Accepts AgentMessage objects from MessageRouter
   - Tracks message states: PENDING, PROCESSING, COMPLETED, FAILED
   - Priority ordering with FIFO tie-breaking
   - Aging mechanism to prevent starvation of LOW/NORMAL messages
   - Concurrency safeguards & duplicate-processing prevention (mutex / lock per message)
   - Asynchronous dispatch via asyncio with compatibility for synchronous callers
   - Integration with FailureClassifier, FailureRecovery, and MessageRouter
   - Non-idempotent operation safeguards (never retrying side-effects)
   - Preserves complete message history and audit trail
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import heapq
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.orchestrator import AgentMessage, MessageType, TaskContext
from core.message_router import MessageRouter, DeliveryReceipt, DeliveryStatus
from core.failure_recovery import (
    WorkerRegistry,
    FailureClassifier,
    FailureType,
    is_non_idempotent_operation,
)
from utils.logging_config import get_logger

logger = get_logger()


# ===========================================================================
# 1. Message Priorities and Queue States
# ===========================================================================

class MessagePriority(str, Enum):
    """Priority classifications for queued agent messages."""
    CRITICAL = "CRITICAL"  # System halts, security blocks, urgent human intervention
    HIGH = "HIGH"          # Approval responses, critical error recovery
    NORMAL = "NORMAL"      # Standard task delegations and handoffs
    LOW = "LOW"            # Background sync, telemetry, non-blocking status updates

    @property
    def level(self) -> int:
        """Numeric rank for priority ordering: lower value = higher priority."""
        ranks = {
            MessagePriority.CRITICAL: 0,
            MessagePriority.HIGH: 1,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 3,
        }
        return ranks.get(self, 2)

    @classmethod
    def from_value(cls, val: Any) -> MessagePriority:
        """Parse priority from string or enum cleanly."""
        if isinstance(val, MessagePriority):
            return val
        s = str(val or "").strip().upper()
        for member in cls:
            if member.value == s:
                return member
        return cls.NORMAL


class QueueMessageState(str, Enum):
    """Lifecycle states of a message inside the MessageQueue."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ===========================================================================
# 2. Queue Item Envelope (Heap Ordering & Aging)
# ===========================================================================

@dataclass(order=True)
class QueueItem:
    """Internal heap-ordered container for prioritized queue processing.

    Comparison tuple: (effective_priority, sequence_id).
    Lower effective_priority processes first; equal priorities follow FIFO sequence_id.
    """
    effective_priority: int = field(compare=True)
    sequence_id: int = field(compare=True)
    message: AgentMessage = field(compare=False)
    original_priority: MessagePriority = field(compare=False)
    state: QueueMessageState = field(default=QueueMessageState.PENDING, compare=False)
    enqueued_at: float = field(default_factory=time.time, compare=False)
    attempts: int = field(default=0, compare=False)
    last_error: Optional[str] = field(default=None, compare=False)
    completed_at: Optional[str] = field(default=None, compare=False)
    receipt: Optional[DeliveryReceipt] = field(default=None, compare=False)


# ===========================================================================
# 3. Centralized MessageQueue
# ===========================================================================

class MessageQueue:
    """Thread-safe, priority-aware, asynchronous message queue for multi-agent systems.

    Key Features:
    - Priority-based scheduling (CRITICAL > HIGH > NORMAL > LOW)
    - Strict FIFO tie-breaking for equal priority messages
    - Anti-starvation aging: Boosts priority of aging items to guarantee progress
    - Duplicate-processing protection: Prevents concurrent execution of the same message_id
    - Non-idempotent operation safeguards: Disallows automated retries on external side-effects
    - Seamless async (`process_all_async`, `deliver_message_async`) and sync (`process_next`, `drain`) execution
    - Integrated with MessageRouter for recipient validation, delivery, and audit tracking
    """

    def __init__(
        self,
        router: Optional[MessageRouter] = None,
        max_retries: int = 2,
        starvation_threshold_seconds: float = 5.0,
        concurrency_limit: int = 4,
        storage: Optional[Any] = None,
        auto_recover: bool = True,
        reliability_coordinator: Optional[Any] = None,
    ):
        self.router = router or MessageRouter()
        self.max_retries = max_retries
        self.starvation_threshold_seconds = starvation_threshold_seconds
        self.concurrency_limit = concurrency_limit
        self.storage = storage

        # Setup ReliabilityCoordinator (Day 226 Part 5)
        if reliability_coordinator is not None:
            self.reliability = reliability_coordinator
        else:
            from core.reliability import ReliabilityCoordinator, DeadLetterQueue
            dlq = DeadLetterQueue(storage=self.storage)
            self.reliability = ReliabilityCoordinator(dlq=dlq)

        # Concurrency & state structures
        self._lock = threading.RLock()
        self._heap: List[QueueItem] = []
        self._sequence_counter: int = 0
        self._items_by_id: Dict[str, QueueItem] = {}
        self._processing_ids: Set[str] = set()
        self._completed_ids: Set[str] = set()
        self._failed_ids: Set[str] = set()
        self._history: List[QueueItem] = []

        # Auto-recover on restart if storage is attached
        if self.storage is not None and auto_recover:
            self.recover_from_storage()

    def recover_from_storage(self) -> Tuple[int, int]:
        """Scan SQLite storage to recover pending and interrupted processing messages."""
        if self.storage is None:
            return 0, 0

        with self._lock:
            recovered_items, failed_ids = self.storage.recover_on_restart(max_retries=self.max_retries)
            for item in recovered_items:
                self._sequence_counter += 1
                item.sequence_id = self._sequence_counter
                heapq.heappush(self._heap, item)
                self._items_by_id[item.message.message_id] = item

            for fid in failed_ids:
                self._failed_ids.add(fid)

            logger.info(
                f"[MessageQueue] Recovered {len(recovered_items)} items from persistent storage; "
                f"{len(failed_ids)} unrecoverable/non-idempotent items marked FAILED."
            )
            return len(recovered_items), len(failed_ids)

    # -----------------------------------------------------------------------
    # Insertion and Retrieval
    # -----------------------------------------------------------------------

    def enqueue(
        self,
        message: AgentMessage,
        priority: Optional[Union[MessagePriority, str]] = None,
    ) -> QueueItem:
        """Enqueue an AgentMessage with validated priority ordering and thread-safety."""
        if hasattr(self, "reliability") and self.reliability is not None:
            self.reliability.metrics.record_message()
        with self._lock:
            # 1. Resolve Priority
            p_val = priority or getattr(message, "priority", MessagePriority.NORMAL)
            norm_priority = MessagePriority.from_value(p_val)
            message.priority = norm_priority.value

            # 2. Check for duplicate active message_id
            msg_id = message.message_id
            if msg_id in self._items_by_id:
                existing = self._items_by_id[msg_id]
                if existing.state in (QueueMessageState.PENDING, QueueMessageState.PROCESSING):
                    logger.warning(
                        f"[MessageQueue] Duplicate enqueue suppressed for active message '{msg_id}'."
                    )
                    return existing

            # 3. Create envelope and push to heap
            self._sequence_counter += 1
            item = QueueItem(
                effective_priority=norm_priority.level,
                sequence_id=self._sequence_counter,
                message=copy.deepcopy(message),
                original_priority=norm_priority,
                state=QueueMessageState.PENDING,
                enqueued_at=time.time(),
            )

            # Persist to SQLite if storage backend is present
            if self.storage is not None:
                persisted = self.storage.insert_message(
                    message=item.message,
                    priority=norm_priority,
                    state=QueueMessageState.PENDING,
                    enqueued_at=item.enqueued_at,
                )
                if not persisted:
                    # Message already exists in storage (persistent duplicate)
                    logger.warning(
                        f"[MessageQueue] Duplicate message_id '{msg_id}' rejected by persistent storage."
                    )
                    existing_db = self.storage.get_message(msg_id)
                    if existing_db:
                        item.state = QueueMessageState(existing_db["state"])
                    return item

            heapq.heappush(self._heap, item)
            self._items_by_id[msg_id] = item
            logger.info(
                f"[MessageQueue] Enqueued message {msg_id} [Priority={norm_priority.value}, Seq={item.sequence_id}] "
                f"from '{message.sender}' to '{message.receiver}'"
            )
            return item

    def dequeue(self) -> Optional[QueueItem]:
        """Safely retrieve the highest-priority pending message with starvation protection."""
        with self._lock:
            if not self._heap:
                return None

            # Apply aging check to prevent starvation of low priority items
            self._apply_anti_starvation()

            # Retrieve next pending item not already in processing
            temp_popped = []
            selected_item: Optional[QueueItem] = None

            while self._heap:
                item = heapq.heappop(self._heap)
                # Ensure message is strictly in PENDING state and not currently being processed
                if item.state == QueueMessageState.PENDING and item.message.message_id not in self._processing_ids:
                    item.state = QueueMessageState.PROCESSING
                    self._processing_ids.add(item.message.message_id)

                    # Update persistent state
                    if self.storage is not None:
                        self.storage.update_state(
                            message_id=item.message.message_id,
                            state=QueueMessageState.PROCESSING,
                        )

                    selected_item = item
                    break
                else:
                    temp_popped.append(item)

            # Reinsert bypassed items
            for it in temp_popped:
                heapq.heappush(self._heap, it)

            return selected_item

    def _apply_anti_starvation(self) -> None:
        """Boost effective priority of items waiting beyond starvation_threshold_seconds."""
        now = time.time()
        needs_reheap = False
        for item in self._heap:
            if item.state == QueueMessageState.PENDING:
                waited = now - item.enqueued_at
                if waited >= self.starvation_threshold_seconds and item.effective_priority > 0:
                    # Promote priority by 1 level
                    item.effective_priority -= 1
                    logger.info(
                        f"[MessageQueue] Anti-starvation boost: message {item.message.message_id} "
                        f"promoted to effective rank {item.effective_priority} (waited {waited:.1f}s)"
                    )
                    needs_reheap = True

        if needs_reheap:
            heapq.heapify(self._heap)

    # -----------------------------------------------------------------------
    # Message Inspection and History
    # -----------------------------------------------------------------------

    def get_state(self, message_id: str) -> Optional[QueueMessageState]:
        """Query the current state of a message in the queue."""
        with self._lock:
            item = self._items_by_id.get(message_id)
            return item.state if item else None

    def get_item(self, message_id: str) -> Optional[QueueItem]:
        """Get the QueueItem record by message ID."""
        with self._lock:
            return self._items_by_id.get(message_id)

    def pending_count(self) -> int:
        """Number of items currently waiting in PENDING state."""
        with self._lock:
            return sum(1 for it in self._heap if it.state == QueueMessageState.PENDING)

    def size(self) -> int:
        """Number of pending items in queue (alias for pending_count)."""
        return self.pending_count()

    def qsize(self) -> int:
        """Standard queue size interface (alias for pending_count)."""
        return self.pending_count()

    def __len__(self) -> int:
        return self.pending_count()

    def processing_count(self) -> int:
        """Number of items currently in PROCESSING state."""
        with self._lock:
            return len(self._processing_ids)

    def get_history(self) -> List[QueueItem]:
        """Return a snapshot of completed, failed, and historical queue items."""
        with self._lock:
            return list(self._history)

    # -----------------------------------------------------------------------
    # Synchronous Execution / Processing
    # -----------------------------------------------------------------------

    def process_next(self, context: Optional[TaskContext] = None) -> Optional[DeliveryReceipt]:
        """Synchronously pop and deliver the next highest-priority message."""
        item = self.dequeue()
        if not item:
            return None
        return self._execute_delivery(item, context=context)

    def drain(self, context: Optional[TaskContext] = None, max_messages: Optional[int] = None) -> List[DeliveryReceipt]:
        """Synchronously process all queued messages until empty or limit reached."""
        receipts = []
        count = 0
        while True:
            if max_messages is not None and count >= max_messages:
                break
            receipt = self.process_next(context=context)
            if receipt is None:
                break
            receipts.append(receipt)
            count += 1
        return receipts

    # -----------------------------------------------------------------------
    # Asynchronous Execution Engine
    # -----------------------------------------------------------------------

    async def deliver_message_async(
        self,
        item: QueueItem,
        context: Optional[TaskContext] = None,
    ) -> DeliveryReceipt:
        """Deliver a single queue item asynchronously in a background thread executor."""
        loop = asyncio.get_running_loop()
        receipt = await loop.run_in_executor(None, self._execute_delivery, item, context)
        return receipt

    async def process_all_async(
        self,
        context: Optional[TaskContext] = None,
        max_messages: Optional[int] = None,
    ) -> List[DeliveryReceipt]:
        """Asynchronously process messages concurrently up to concurrency_limit."""
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        tasks = []

        async def worker():
            nonlocal tasks
            while True:
                # Thread-safe dequeue
                item = self.dequeue()
                if item is None:
                    break

                async with semaphore:
                    receipt = await self.deliver_message_async(item, context=context)
                    tasks.append(receipt)

        # Launch concurrent worker loops matching concurrency_limit
        workers = [asyncio.create_task(worker()) for _ in range(self.concurrency_limit)]
        await asyncio.gather(*workers)
        return tasks

    # -----------------------------------------------------------------------
    # Delivery Execution with Controlled Failure Handling & Retries
    # -----------------------------------------------------------------------

    def _execute_delivery(
        self,
        item: QueueItem,
        context: Optional[TaskContext] = None,
    ) -> DeliveryReceipt:
        """Execute message delivery via MessageRouter with retry and failure handling."""
        msg = item.message
        msg_id = msg.message_id
        item.attempts += 1

        # Check non-idempotent operation constraints
        desc = ""
        if isinstance(msg.input_data, str):
            desc = msg.input_data
        elif isinstance(msg.input_data, dict):
            desc = str(msg.input_data.get("description", "") or msg.input_data.get("query", ""))
        is_non_idempotent = is_non_idempotent_operation(desc)

        try:
            # Route message via MessageRouter directly (bypassing re-enqueue)
            from core.reliability import DeliveryGuarantee, AckStatus
            orig_retries = self.router.max_retries
            try:
                if hasattr(self, "reliability") and self.reliability is not None:
                    if self.reliability.delivery_guarantee == DeliveryGuarantee.AT_MOST_ONCE:
                        self.router.max_retries = 0
                receipt = self.router.route(msg, context=context, use_queue=False)
            finally:
                self.router.max_retries = orig_retries

            item.receipt = receipt

            if receipt.status == DeliveryStatus.DELIVERED:
                with self._lock:
                    item.state = QueueMessageState.COMPLETED
                    item.completed_at = datetime.now(timezone.utc).isoformat()
                    self._processing_ids.discard(msg_id)
                    self._completed_ids.add(msg_id)
                    self._history.append(item)

                    # Persist state update
                    if self.storage is not None:
                        self.storage.update_state(
                            message_id=msg_id,
                            state=QueueMessageState.COMPLETED,
                            attempts=item.attempts,
                            completed_at=item.completed_at,
                        )

                # Record delivery telemetry and successful acknowledgement
                if hasattr(self, "reliability") and self.reliability is not None:
                    from core.reliability import AckStatus
                    self.reliability.metrics.record_delivery(receipt.latency_seconds)
                    self.reliability.record_acknowledgement(
                        message=msg,
                        recipient=msg.receiver,
                        status=AckStatus.ACKNOWLEDGED,
                        reason="Delivered and processed successfully",
                    )
                    if self.storage is not None and hasattr(self.storage, "update_acknowledgement"):
                        self.storage.update_acknowledgement(
                            message_id=msg_id,
                            ack_status=AckStatus.ACKNOWLEDGED.value,
                            reason="Delivered and processed successfully",
                        )

                logger.info(f"[MessageQueue] Successfully processed message {msg_id} -> COMPLETED")
                return receipt
            else:
                # Delivery failed inside router
                return self._handle_delivery_failure(
                    item=item,
                    error_message=receipt.error or "Router delivery failure",
                    is_non_idempotent=is_non_idempotent,
                    receipt=receipt,
                )

        except Exception as exc:
            err_str = str(exc)
            logger.error(f"[MessageQueue] Unhandled delivery exception for {msg_id}: {err_str}")
            return self._handle_delivery_failure(
                item=item,
                error_message=err_str,
                is_non_idempotent=is_non_idempotent,
                receipt=None,
            )

    def _handle_delivery_failure(
        self,
        item: QueueItem,
        error_message: str,
        is_non_idempotent: bool,
        receipt: Optional[DeliveryReceipt] = None,
    ) -> DeliveryReceipt:
        """Evaluate failure classification and re-enqueue if eligible for retry."""
        msg = item.message
        msg_id = msg.message_id
        item.last_error = error_message
        if receipt is not None and receipt.attempts > item.attempts:
            item.attempts = receipt.attempts

        failure = FailureClassifier.classify(
            error=error_message,
            stage=msg.stage or "delivery",
            agent_name=msg.receiver,
            task_id=msg.task_id,
        )

        can_retry = (
            failure.is_retryable
            and not is_non_idempotent
            and failure.failure_type != FailureType.AUTHORIZATION
            and item.attempts <= self.max_retries
        )

        # Check delivery guarantee policy
        from core.reliability import DeliveryGuarantee, AckStatus
        if hasattr(self, "reliability") and self.reliability is not None:
            if self.reliability.delivery_guarantee == DeliveryGuarantee.AT_MOST_ONCE:
                # Under at-most-once delivery guarantee, retries are disabled
                can_retry = False

        with self._lock:
            self._processing_ids.discard(msg_id)

            if can_retry:
                # Re-queue for next attempt
                item.state = QueueMessageState.PENDING
                heapq.heappush(self._heap, item)

                if self.storage is not None:
                    self.storage.update_state(
                        message_id=msg_id,
                        state=QueueMessageState.PENDING,
                        attempts=item.attempts,
                        error=error_message,
                    )

                if hasattr(self, "reliability") and self.reliability is not None:
                    self.reliability.metrics.record_retry()

                logger.info(
                    f"[MessageQueue] Re-queued message {msg_id} for retry ({item.attempts}/{self.max_retries + 1}). "
                    f"Reason: {error_message}"
                )
            else:
                # Terminal failure -> move to Dead-Letter Queue (DLQ)
                item.state = QueueMessageState.FAILED
                item.completed_at = datetime.now(timezone.utc).isoformat()
                self._failed_ids.add(msg_id)
                self._history.append(item)

                if self.storage is not None:
                    self.storage.update_state(
                        message_id=msg_id,
                        state=QueueMessageState.FAILED,
                        attempts=item.attempts,
                        error=error_message,
                        completed_at=item.completed_at,
                    )

                # Route to DLQ & update reliability metrics
                if hasattr(self, "reliability") and self.reliability is not None:
                    self.reliability.route_to_dlq(
                        message=msg,
                        reason=error_message,
                        retry_count=item.attempts,
                        failure_details={
                            "failure_type": failure.failure_type.value,
                            "is_retryable": failure.is_retryable,
                            "error": error_message,
                        },
                    )
                    self.reliability.record_acknowledgement(
                        message=msg,
                        recipient=msg.receiver,
                        status=AckStatus.REJECTED,
                        reason=error_message,
                    )
                    if self.storage is not None and hasattr(self.storage, "update_acknowledgement"):
                        self.storage.update_acknowledgement(
                            message_id=msg_id,
                            ack_status=AckStatus.REJECTED.value,
                            reason=error_message,
                        )

                logger.warning(
                    f"[MessageQueue] Message {msg_id} marked as FAILED (terminal) -> Escalated to DLQ. Reason: {error_message}"
                )

        if receipt is not None:
            return receipt

        return DeliveryReceipt(
            message_id=msg_id,
            task_id=msg.task_id,
            sender=msg.sender,
            receiver=msg.receiver,
            message_type=str(msg.message_type),
            status=DeliveryStatus.FAILED,
            correlation_id=msg.correlation_id,
            delivery_timestamp=datetime.now(timezone.utc).isoformat(),
            attempts=item.attempts,
            error=error_message,
        )

    # -----------------------------------------------------------------------
    # Queue Management
    # -----------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all pending items and internal state."""
        with self._lock:
            self._heap.clear()
            self._items_by_id.clear()
            self._processing_ids.clear()
            self._completed_ids.clear()
            self._failed_ids.clear()
            self._history.clear()
