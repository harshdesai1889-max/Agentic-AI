"""
core/hitl.py

Human-in-the-Loop (HITL) Subsystem for Day 227 (Parts 1, 2, and 3).

Implements:
1. InteractionType, InteractionStatus, ApprovalDecision enums.
2. HumanInteractionRequest & ApprovalRecord structured contracts.
3. HumanInteractionManager (Part 1):
   - Reusable interaction request lifecycle (create, query, resolve, cancel, expire).
   - Interaction history and context attachment.
   - Persistent SQLite backing and restart hydration.
   - Integration with message router and queue.
4. HumanApprovalManager (Part 2):
   - Decision gates: APPROVED, REJECTED, MODIFICATION_REQUESTED, EXPIRED, CANCELLED.
   - Action-bound, task-bound authorization records with cryptographic tokens.
   - Execution boundary enforcement preventing unauthorized or expired executions.
   - Protection against approval reuse across different tasks.
5. HumanFeedbackManager (Part 3):
   - Clarification, information, feedback, and plan modification requests.
   - Safe workflow pause and persistence preserving completed work.
   - Strict response validation preventing cross-task leakage and duplicate processing.
   - Safe workflow resumption continuing from the exact paused stage.
6. HumanInTheLoopEngine:
   - Unified coordinator combining interaction, approval, and feedback subsystems.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.orchestrator import AgentMessage, MessageType, TaskContext, WorkflowStatus
from core.message_router import MessageRouter, DeliveryReceipt, DeliveryStatus
from core.persistent_storage import SQLiteMessageStorage
from utils.logging_config import get_logger

logger = get_logger()


def _record_hitl_lifecycle_event(
    event_category: str,
    message: str,
    observability: Optional[Any] = None,
    task_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    agent_name: str = "hitl_engine",
    log_level: str = "INFO",
) -> None:
    """Record a structured log and discrete execution event for HITL lifecycle transitions."""
    obs = observability
    if obs is None:
        try:
            from core.observability import get_active_observability_manager
            obs = get_active_observability_manager()
        except Exception:
            obs = None
    if not obs:
        return

    safe_payload = dict(payload or {})
    for k in list(safe_payload.keys()):
        if any(sec in k.lower() for sec in ("token", "secret", "password", "key", "auth")):
            safe_payload[k] = "***REDACTED***"

    try:
        from core.observability import LogLevel
        level = getattr(LogLevel, log_level.upper(), LogLevel.INFO)
        obs.log(
            event_type=event_category,
            message=message,
            log_level=level,
            task_id=task_id,
            workflow_id=workflow_id,
            agent_name=agent_name,
            attributes=safe_payload,
        )
        obs.record_event(
            event_type=event_category,
            payload=safe_payload,
            agent_name=agent_name,
        )
    except Exception as exc:
        logger.debug(f"[HITL Observability] Failed to record event {event_category}: {exc}")


# ===========================================================================
# 1. Enums
# ===========================================================================

class InteractionType(str, Enum):
    """Semantic type of human intervention request."""
    APPROVAL_REQUEST = "approval_request"
    CLARIFICATION_REQUEST = "clarification_request"
    REVIEW_REQUEST = "review_request"
    FEEDBACK_REQUEST = "feedback_request"
    INFORMATION_REQUEST = "information_request"
    PLAN_MODIFICATION_REQUEST = "plan_modification_request"


class InteractionStatus(str, Enum):
    """Lifecycle status of a human interaction request."""
    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalDecision(str, Enum):
    """Supported human decisions for approval gates."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFICATION_REQUESTED = "MODIFICATION_REQUESTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


# ===========================================================================
# 2. Structured HITL Contracts
# ===========================================================================

@dataclass
class HumanInteractionRequest:
    """Structured contract defining a request for human intervention."""
    interaction_id: str
    task_id: str
    correlation_id: str
    requesting_agent: str
    interaction_type: Union[InteractionType, str]
    question_or_request: str
    relevant_context: Dict[str, Any] = field(default_factory=dict)
    available_decisions: List[str] = field(default_factory=list)
    status: Union[InteractionStatus, str] = InteractionStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    decision: Optional[str] = None
    human_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_pending(self) -> bool:
        """Check if request is awaiting a human decision."""
        status_val = self.status.value if isinstance(self.status, InteractionStatus) else str(self.status)
        return status_val == InteractionStatus.PENDING.value

    def resolve(
        self,
        decision: str,
        human_response: Optional[str] = None,
        resolved_at: Optional[str] = None,
    ) -> None:
        """Mark request as resolved with human decision and feedback."""
        self.status = InteractionStatus.RESOLVED
        self.decision = str(decision)
        self.human_response = human_response
        self.resolved_at = resolved_at or datetime.now(timezone.utc).isoformat()

    def cancel(self, reason: str = "") -> None:
        """Cancel the pending request."""
        self.status = InteractionStatus.CANCELLED
        self.resolved_at = datetime.now(timezone.utc).isoformat()
        if reason:
            self.metadata["cancellation_reason"] = reason

    def expire(self) -> None:
        """Mark the pending request as expired."""
        self.status = InteractionStatus.EXPIRED
        self.resolved_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to a serializable dictionary."""
        return {
            "interaction_id": self.interaction_id,
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
            "requesting_agent": self.requesting_agent,
            "interaction_type": self.interaction_type.value if isinstance(self.interaction_type, InteractionType) else str(self.interaction_type),
            "question_or_request": self.question_or_request,
            "relevant_context": self.relevant_context,
            "available_decisions": self.available_decisions,
            "status": self.status.value if isinstance(self.status, InteractionStatus) else str(self.status),
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "decision": self.decision,
            "human_response": self.human_response,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HumanInteractionRequest:
        """Reconstruct request from dictionary representation."""
        raw_type = data.get("interaction_type", "approval_request")
        try:
            itype = InteractionType(str(raw_type))
        except ValueError:
            itype = str(raw_type)

        raw_status = data.get("status", "pending")
        try:
            istatus = InteractionStatus(str(raw_status))
        except ValueError:
            istatus = str(raw_status)

        return cls(
            interaction_id=data["interaction_id"],
            task_id=data["task_id"],
            correlation_id=data.get("correlation_id", data["task_id"]),
            requesting_agent=data["requesting_agent"],
            interaction_type=itype,
            question_or_request=data["question_or_request"],
            relevant_context=data.get("relevant_context", {}),
            available_decisions=data.get("available_decisions", []),
            status=istatus,
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            resolved_at=data.get("resolved_at"),
            decision=data.get("decision"),
            human_response=data.get("human_response"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ApprovalRecord:
    """Action-bound approval record enforcing execution gates."""
    approval_id: str
    interaction_id: str
    task_id: str
    correlation_id: str
    requested_action: str
    decision: Union[ApprovalDecision, str]
    human_feedback: Optional[str] = None
    decision_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approval_status: str = "APPROVED"
    expires_at: Optional[str] = None
    token: str = field(default_factory=lambda: f"tok_appr_{uuid.uuid4().hex[:12]}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(
        self,
        task_id: Optional[str] = None,
        action: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validate whether approval record authorizes execution."""
        if getattr(self, "approval_status", "") == "CONSUMED":
            return False, "Approval token has already been consumed."

        dec_val = self.decision.value if isinstance(self.decision, ApprovalDecision) else str(self.decision)
        if dec_val != ApprovalDecision.APPROVED.value:
            return False, f"Action not approved: decision is '{dec_val}'"

        # Check task bound
        if task_id is not None and str(self.task_id) != str(task_id):
            return False, f"Task ID mismatch: approval is bound to task '{self.task_id}', not '{task_id}'"

        # Check action bound
        if action is not None:
            norm_req = str(self.requested_action).strip().lower()
            norm_act = str(action).strip().lower()
            if norm_act not in norm_req and norm_req not in norm_act:
                return False, f"Action mismatch: approval authorizes '{self.requested_action}', not '{action}'"

        # Check token bound
        if token is not None and str(self.token) != str(token):
            return False, "Approval token mismatch."

        # Check expiration
        if self.expires_at is not None:
            now_iso = datetime.now(timezone.utc).isoformat()
            if now_iso > self.expires_at:
                return False, f"Approval expired at '{self.expires_at}'"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert approval record to serializable dictionary."""
        return {
            "approval_id": self.approval_id,
            "interaction_id": self.interaction_id,
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
            "requested_action": self.requested_action,
            "decision": self.decision.value if isinstance(self.decision, ApprovalDecision) else str(self.decision),
            "human_feedback": self.human_feedback,
            "decision_timestamp": self.decision_timestamp,
            "approval_status": self.approval_status,
            "expires_at": self.expires_at,
            "token": self.token,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApprovalRecord:
        """Reconstruct approval record from dictionary."""
        raw_dec = data.get("decision", "APPROVED")
        try:
            dec = ApprovalDecision(str(raw_dec))
        except ValueError:
            dec = str(raw_dec)

        return cls(
            approval_id=data["approval_id"],
            interaction_id=data["interaction_id"],
            task_id=data["task_id"],
            correlation_id=data.get("correlation_id", data["task_id"]),
            requested_action=data["requested_action"],
            decision=dec,
            human_feedback=data.get("human_feedback"),
            decision_timestamp=data.get("decision_timestamp", datetime.now(timezone.utc).isoformat()),
            approval_status=data.get("approval_status", str(dec)),
            expires_at=data.get("expires_at"),
            token=data.get("token", f"tok_appr_{uuid.uuid4().hex[:12]}"),
            metadata=data.get("metadata", {}),
        )


# ===========================================================================
# 3. Part 1: HumanInteractionManager
# ===========================================================================

class HumanInteractionManager:
    """Central manager coordinating all human intervention requests.

    Responsibilities:
    - Lifecycle management of interaction requests (APPROVAL, CLARIFICATION, REVIEW, FEEDBACK).
    - Preserves correlation and task IDs.
    - Maintains comprehensive interaction history.
    - Attaches interaction events to TaskContext audit log.
    - Persists records to SQLiteMessageStorage with restart recovery.
    - Integrates with MessageRouter for dispatching interaction alerts.
    """

    def __init__(
        self,
        storage: Optional[SQLiteMessageStorage] = None,
        router: Optional[MessageRouter] = None,
        observability: Optional[Any] = None,
    ):
        self.storage = storage
        self.router = router
        self.observability = observability
        self._lock = threading.RLock()
        self._interactions: Dict[str, HumanInteractionRequest] = {}
        if self.storage is not None:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Hydrate interaction requests from SQLite storage on startup."""
        if hasattr(self.storage, "list_interactions"):
            rows = self.storage.list_interactions()
            for r in rows:
                req = HumanInteractionRequest.from_dict(r)
                self._interactions[req.interaction_id] = req
            logger.info(f"[HumanInteractionManager] Hydrated {len(rows)} interactions from storage.")

    def create_interaction(
        self,
        task_id: str,
        correlation_id: str,
        requesting_agent: str,
        interaction_type: Union[InteractionType, str],
        question_or_request: str,
        relevant_context: Optional[Dict[str, Any]] = None,
        available_decisions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
    ) -> HumanInteractionRequest:
        """Create and register a new structured human interaction request."""
        with self._lock:
            interaction_id = f"hitl_{uuid.uuid4().hex[:8]}"
            itype = (
                interaction_type
                if isinstance(interaction_type, InteractionType)
                else InteractionType(str(interaction_type))
            )

            if available_decisions is None:
                if itype == InteractionType.APPROVAL_REQUEST:
                    available_decisions = ["APPROVED", "REJECTED", "MODIFICATION_REQUESTED"]
                elif itype == InteractionType.CLARIFICATION_REQUEST:
                    available_decisions = ["PROVIDE_INFO", "CANCEL"]
                else:
                    available_decisions = ["SUBMIT", "CANCEL"]

            request = HumanInteractionRequest(
                interaction_id=interaction_id,
                task_id=task_id,
                correlation_id=correlation_id or task_id,
                requesting_agent=requesting_agent,
                interaction_type=itype,
                question_or_request=question_or_request,
                relevant_context=relevant_context or {},
                available_decisions=available_decisions,
                status=InteractionStatus.PENDING,
                metadata=metadata or {},
            )

            self._interactions[interaction_id] = request

            # Persist to SQLite
            if self.storage is not None and hasattr(self.storage, "save_interaction"):
                self.storage.save_interaction(request.to_dict())

            # Attach to TaskContext if provided
            if context is not None:
                self.attach_to_context(request, context)

            # Notify via MessageRouter if attached
            if self.router is not None:
                alert_msg = AgentMessage(
                    task_id=task_id,
                    agent_name=requesting_agent,
                    sender=requesting_agent,
                    receiver="User",
                    stage="human_intervention",
                    status="waiting",
                    message_type=MessageType.CLARIFICATION_REQUEST if itype == InteractionType.CLARIFICATION_REQUEST else MessageType.APPROVAL_REQUEST,
                    input_data=question_or_request,
                    output_data=None,
                    correlation_id=correlation_id,
                    metadata={"interaction_id": interaction_id, "interaction_type": itype.value},
                )
                self.router.route(alert_msg, context=context)

            logger.info(
                f"[HumanInteractionManager] Created {itype.value} {interaction_id} for task '{task_id}' by '{requesting_agent}'"
            )
            return request

    def get_interaction(self, interaction_id: str) -> Optional[HumanInteractionRequest]:
        """Fetch an interaction by interaction_id."""
        with self._lock:
            return self._interactions.get(interaction_id)

    def list_interactions(
        self,
        task_id: Optional[str] = None,
        status: Optional[Union[InteractionStatus, str]] = None,
    ) -> List[HumanInteractionRequest]:
        """List interaction history with optional filtering."""
        with self._lock:
            status_val = status.value if isinstance(status, InteractionStatus) else (str(status) if status else None)
            results = list(self._interactions.values())
            if task_id:
                results = [it for it in results if str(it.task_id) == str(task_id)]
            if status_val:
                results = [
                    it for it in results
                    if (it.status.value if isinstance(it.status, InteractionStatus) else str(it.status)) == status_val
                ]
            return results

    def get_pending_interactions(self, task_id: Optional[str] = None) -> List[HumanInteractionRequest]:
        """Get all unresolved interaction requests."""
        return self.list_interactions(task_id=task_id, status=InteractionStatus.PENDING)

    def resolve_interaction(
        self,
        interaction_id: str,
        decision: str,
        human_response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
    ) -> HumanInteractionRequest:
        """Resolve a pending interaction with the human decision."""
        with self._lock:
            req = self._interactions.get(interaction_id)
            if not req:
                raise ValueError(f"Interaction '{interaction_id}' not found.")

            if not req.is_pending():
                raise ValueError(
                    f"Interaction '{interaction_id}' is already resolved with status '{req.status}'."
                )

            req.resolve(decision=decision, human_response=human_response)
            if metadata:
                req.metadata.update(metadata)

            # Persist update to SQLite
            if self.storage is not None and hasattr(self.storage, "update_interaction"):
                self.storage.update_interaction(
                    interaction_id=interaction_id,
                    status=InteractionStatus.RESOLVED.value,
                    decision=decision,
                    human_response=human_response,
                    resolved_at=req.resolved_at,
                    metadata=req.metadata,
                )

            # Update context if provided
            if context is not None:
                resp_msg = AgentMessage(
                    task_id=req.task_id,
                    agent_name="User",
                    sender="User",
                    receiver=req.requesting_agent,
                    stage="human_intervention",
                    status="success",
                    message_type=MessageType.APPROVAL_DECISION if req.interaction_type == InteractionType.APPROVAL_REQUEST else MessageType.STATUS_UPDATE,
                    input_data=decision,
                    output_data=human_response,
                    correlation_id=req.correlation_id,
                    metadata={"interaction_id": interaction_id, "decision": decision},
                )
                context.add_message(resp_msg)

            logger.info(
                f"[HumanInteractionManager] Resolved interaction {interaction_id}: decision='{decision}'"
            )
            return req

    def cancel_interaction(self, interaction_id: str, reason: str = "") -> HumanInteractionRequest:
        """Cancel an interaction request."""
        with self._lock:
            req = self._interactions.get(interaction_id)
            if not req:
                raise ValueError(f"Interaction '{interaction_id}' not found.")
            req.cancel(reason=reason)
            if self.storage is not None and hasattr(self.storage, "update_interaction"):
                self.storage.update_interaction(
                    interaction_id=interaction_id,
                    status=InteractionStatus.CANCELLED.value,
                    resolved_at=req.resolved_at,
                    metadata=req.metadata,
                )
            if req.interaction_type == InteractionType.APPROVAL_REQUEST:
                _record_hitl_lifecycle_event(
                    event_category="human_approval_cancelled",
                    message=f"Approval request '{interaction_id}' cancelled: {reason}",
                    observability=self.observability,
                    task_id=req.task_id,
                    payload={"interaction_id": interaction_id, "reason": reason},
                )
            return req

    def expire_interaction(self, interaction_id: str) -> HumanInteractionRequest:
        """Mark an interaction request as expired."""
        with self._lock:
            req = self._interactions.get(interaction_id)
            if not req:
                raise ValueError(f"Interaction '{interaction_id}' not found.")
            req.expire()
            if self.storage is not None and hasattr(self.storage, "update_interaction"):
                self.storage.update_interaction(
                    interaction_id=interaction_id,
                    status=InteractionStatus.EXPIRED.value,
                    resolved_at=req.resolved_at,
                )
            if req.interaction_type == InteractionType.APPROVAL_REQUEST:
                _record_hitl_lifecycle_event(
                    event_category="human_approval_expired",
                    message=f"Approval request '{interaction_id}' expired",
                    observability=self.observability,
                    task_id=req.task_id,
                    payload={"interaction_id": interaction_id},
                    log_level="WARNING",
                )
            return req

    def attach_to_context(
        self,
        request: HumanInteractionRequest,
        context: TaskContext,
    ) -> None:
        """Attach interaction metadata and audit message to TaskContext."""
        if "human_interactions" not in context.metadata:
            context.metadata["human_interactions"] = []
        context.metadata["human_interactions"].append(request.to_dict())

        msg = AgentMessage(
            task_id=request.task_id,
            agent_name=request.requesting_agent,
            sender=request.requesting_agent,
            receiver="User",
            stage="human_intervention",
            status="waiting",
            message_type=MessageType.CLARIFICATION_REQUEST if request.interaction_type == InteractionType.CLARIFICATION_REQUEST else MessageType.APPROVAL_REQUEST,
            input_data=request.question_or_request,
            output_data=None,
            correlation_id=request.correlation_id,
            metadata={"interaction_id": request.interaction_id, "interaction_type": str(request.interaction_type)},
        )
        context.add_message(msg)


# ===========================================================================
# 4. Part 2: HumanApprovalManager
# ===========================================================================

class HumanApprovalManager:
    """Manages human approval decisions and enforces execution gates.

    Responsibilities:
    - Coordinates approval requests for sensitive / consequential operations.
    - Records APPROVED, REJECTED, MODIFICATION_REQUESTED, EXPIRED, CANCELLED states.
    - Generates and verifies task-bound, action-bound approval tokens.
    - Enforces verification at the execution boundary (ExecutorAgent / AgentEngine).
    - Prevents approval reuse across different tasks or past expiration windows.
    - Rejects execution attempts when human approval is missing or bypass was attempted.
    """

    def __init__(
        self,
        interaction_manager: Optional[HumanInteractionManager] = None,
        storage: Optional[SQLiteMessageStorage] = None,
        default_ttl_seconds: Optional[float] = None,
        observability: Optional[Any] = None,
    ):
        self.storage = storage
        self.interaction_manager = interaction_manager or HumanInteractionManager(storage=storage, observability=observability)
        self.default_ttl_seconds = default_ttl_seconds
        self.observability = observability
        self._lock = threading.RLock()
        self._approvals: Dict[str, ApprovalRecord] = {}
        self._approvals_by_token: Dict[str, ApprovalRecord] = {}
        self._approvals_by_task: Dict[str, List[ApprovalRecord]] = {}
        if self.storage is not None:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Hydrate approvals from SQLite storage."""
        if hasattr(self.storage, "list_approvals"):
            rows = self.storage.list_approvals()
            for r in rows:
                appr = ApprovalRecord.from_dict(r)
                self._approvals[appr.approval_id] = appr
                self._approvals_by_token[appr.token] = appr
                if appr.task_id not in self._approvals_by_task:
                    self._approvals_by_task[appr.task_id] = []
                self._approvals_by_task[appr.task_id].append(appr)
            logger.info(f"[HumanApprovalManager] Hydrated {len(rows)} approvals from storage.")

    def request_approval(
        self,
        task_id: str,
        correlation_id: str,
        requesting_agent: str,
        requested_action: str,
        relevant_context: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
        context: Optional[TaskContext] = None,
    ) -> HumanInteractionRequest:
        """Create a structured approval request for a sensitive action."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        expires_at = None
        if ttl is not None and ttl > 0:
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()

        meta = {"requested_action": requested_action}
        if expires_at:
            meta["expires_at"] = expires_at

        req = self.interaction_manager.create_interaction(
            task_id=task_id,
            correlation_id=correlation_id,
            requesting_agent=requesting_agent,
            interaction_type=InteractionType.APPROVAL_REQUEST,
            question_or_request=f"Approval requested for action: {requested_action}",
            relevant_context=relevant_context or {"action": requested_action},
            available_decisions=["APPROVED", "REJECTED", "MODIFICATION_REQUESTED"],
            metadata=meta,
            context=context,
        )

        _record_hitl_lifecycle_event(
            event_category="human_approval_requested",
            message=f"Approval requested for action '{requested_action}' on task '{task_id}'.",
            observability=self.observability,
            task_id=task_id,
            payload={
                "requested_action": requested_action,
                "requesting_agent": requesting_agent,
                "interaction_id": req.interaction_id,
            },
        )
        return req

    def approve(
        self,
        interaction_id: str,
        feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
    ) -> ApprovalRecord:
        """Record human approval and issue an action-bound approval token."""
        with self._lock:
            interaction = self.interaction_manager.get_interaction(interaction_id)
            if not interaction:
                raise ValueError(f"Interaction '{interaction_id}' not found.")

            # Expiration check
            expires_at = interaction.metadata.get("expires_at")
            if expires_at:
                now_iso = datetime.now(timezone.utc).isoformat()
                if now_iso > expires_at:
                    self.interaction_manager.expire_interaction(interaction_id)
                    raise TimeoutError(f"Approval request '{interaction_id}' has expired (expired at {expires_at}).")

            self.interaction_manager.resolve_interaction(
                interaction_id=interaction_id,
                decision=ApprovalDecision.APPROVED.value,
                human_response=feedback,
                context=context,
            )

            approval_id = f"appr_{uuid.uuid4().hex[:8]}"
            token = f"tok_appr_{uuid.uuid4().hex[:12]}"
            record = ApprovalRecord(
                approval_id=approval_id,
                interaction_id=interaction_id,
                task_id=interaction.task_id,
                correlation_id=interaction.correlation_id,
                requested_action=interaction.metadata.get("requested_action", interaction.question_or_request),
                decision=ApprovalDecision.APPROVED,
                human_feedback=feedback,
                approval_status="APPROVED",
                expires_at=expires_at,
                token=token,
                metadata=metadata or {},
            )

            self._approvals[approval_id] = record
            self._approvals_by_token[token] = record
            if record.task_id not in self._approvals_by_task:
                self._approvals_by_task[record.task_id] = []
            self._approvals_by_task[record.task_id].append(record)

            if self.storage is not None and hasattr(self.storage, "save_approval"):
                self.storage.save_approval(record.to_dict())

            _record_hitl_lifecycle_event(
                event_category="human_approval_granted",
                message=f"Action approved for task '{record.task_id}' (token redacted).",
                observability=self.observability,
                task_id=record.task_id,
                payload={
                    "approval_id": record.approval_id,
                    "requested_action": record.requested_action,
                    "approval_token": "***REDACTED***",
                },
            )

            logger.info(
                f"[HumanApprovalManager] Action approved for task '{record.task_id}' (token={token})"
            )
            return record

    def reject(
        self,
        interaction_id: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
    ) -> ApprovalRecord:
        """Record human rejection of an action."""
        with self._lock:
            interaction = self.interaction_manager.get_interaction(interaction_id)
            if not interaction:
                raise ValueError(f"Interaction '{interaction_id}' not found.")

            self.interaction_manager.resolve_interaction(
                interaction_id=interaction_id,
                decision=ApprovalDecision.REJECTED.value,
                human_response=reason,
                context=context,
            )

            approval_id = f"appr_{uuid.uuid4().hex[:8]}"
            record = ApprovalRecord(
                approval_id=approval_id,
                interaction_id=interaction_id,
                task_id=interaction.task_id,
                correlation_id=interaction.correlation_id,
                requested_action=interaction.metadata.get("requested_action", interaction.question_or_request),
                decision=ApprovalDecision.REJECTED,
                human_feedback=reason,
                approval_status="REJECTED",
                metadata=metadata or {},
            )

            self._approvals[approval_id] = record
            if record.task_id not in self._approvals_by_task:
                self._approvals_by_task[record.task_id] = []
            self._approvals_by_task[record.task_id].append(record)

            if self.storage is not None and hasattr(self.storage, "save_approval"):
                self.storage.save_approval(record.to_dict())

            _record_hitl_lifecycle_event(
                event_category="human_approval_rejected",
                message=f"Action rejected for task '{record.task_id}': reason='{reason}'.",
                observability=self.observability,
                task_id=record.task_id,
                payload={
                    "approval_id": record.approval_id,
                    "reason": reason,
                },
                log_level="WARNING",
            )

            logger.warning(
                f"[HumanApprovalManager] Action rejected for task '{record.task_id}': reason='{reason}'"
            )
            return record

    def cancel_approval(self, interaction_id: str, reason: str = "") -> HumanInteractionRequest:
        """Explicitly cancel a pending approval request."""
        return self.interaction_manager.cancel_interaction(interaction_id=interaction_id, reason=reason)

    def expire_approval(self, interaction_id: str) -> HumanInteractionRequest:
        """Explicitly mark a pending approval request as expired."""
        return self.interaction_manager.expire_interaction(interaction_id=interaction_id)

    def request_modification(
        self,
        interaction_id: str,
        instructions: str,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
    ) -> ApprovalRecord:
        """Record a human request for modifications before approval."""
        with self._lock:
            interaction = self.interaction_manager.get_interaction(interaction_id)
            if not interaction:
                raise ValueError(f"Interaction '{interaction_id}' not found.")

            self.interaction_manager.resolve_interaction(
                interaction_id=interaction_id,
                decision=ApprovalDecision.MODIFICATION_REQUESTED.value,
                human_response=instructions,
                context=context,
            )

            approval_id = f"appr_{uuid.uuid4().hex[:8]}"
            record = ApprovalRecord(
                approval_id=approval_id,
                interaction_id=interaction_id,
                task_id=interaction.task_id,
                correlation_id=interaction.correlation_id,
                requested_action=interaction.metadata.get("requested_action", interaction.question_or_request),
                decision=ApprovalDecision.MODIFICATION_REQUESTED,
                human_feedback=instructions,
                approval_status="MODIFICATION_REQUESTED",
                metadata=metadata or {},
            )

            self._approvals[approval_id] = record
            if record.task_id not in self._approvals_by_task:
                self._approvals_by_task[record.task_id] = []
            self._approvals_by_task[record.task_id].append(record)

            if self.storage is not None and hasattr(self.storage, "save_approval"):
                self.storage.save_approval(record.to_dict())

            logger.info(
                f"[HumanApprovalManager] Modification requested for task '{record.task_id}': instructions='{instructions}'"
            )
            return record

    def get_approval(self, approval_id: str) -> Optional[ApprovalRecord]:
        """Fetch approval record by ID."""
        with self._lock:
            return self._approvals.get(approval_id)

    def get_approval_by_token(self, token: str) -> Optional[ApprovalRecord]:
        """Fetch approval record by cryptographic token."""
        with self._lock:
            return self._approvals_by_token.get(token)

    def get_approval_for_task(self, task_id: str) -> Optional[ApprovalRecord]:
        """Fetch the latest approval record for a specific task."""
        with self._lock:
            records = self._approvals_by_task.get(str(task_id), [])
            return records[-1] if records else None

    def is_approved(
        self,
        task_id: str,
        action: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Verify approval at the actual execution boundary.

        Enforces:
        - Must have a matching approval record.
        - Approval must be in APPROVED state.
        - Approval must not be expired.
        - Approval must match the exact task_id (prevents task reuse).
        - If token is supplied, must match token.
        - If action is supplied, must match requested action.
        """
        with self._lock:
            record: Optional[ApprovalRecord] = None
            if token:
                record = self.get_approval_by_token(token)
                if not record:
                    return False, f"Invalid approval token: '{token}' not recognized."
            else:
                record = self.get_approval_for_task(task_id)

            if not record:
                return False, f"Missing human approval: No approval record exists for task '{task_id}'."

            # Verify action and task binding
            valid, err = record.is_valid(task_id=task_id, action=action, token=token)
            if not valid:
                return False, f"Approval gate check failed: {err}"

            return True, None

    def consume_token(
        self,
        token: str,
        task_id: Optional[str] = None,
        action: Optional[str] = None,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Atomically validate and consume a single-use human approval token.

        Guarantees single-use semantics under concurrent workers.
        Enforces token, task, action, workflow, agent, and tool bindings.
        Only one worker can successfully transition an approval record from APPROVED to CONSUMED.
        """
        with self._lock:
            record = self._approvals_by_token.get(token)
            # Cross-process lookup: if not in in-memory cache, query persistent storage
            if not record and self.storage is not None and hasattr(self.storage, "get_approval_by_token"):
                try:
                    db_appr = self.storage.get_approval_by_token(token)
                    if db_appr:
                        record = ApprovalRecord.from_dict(db_appr)
                        self._approvals_by_token[token] = record
                except Exception as exc:
                    logger.debug(f"[HumanApprovalManager] DB lookup error: {exc}")

            if not record:
                return False, f"Invalid approval token: '{token}' not recognized."

            if getattr(record, "approval_status", "") == "CONSUMED":
                return False, f"Approval token '{token[:12]}...' has already been consumed."

            # Verify action, task binding and expiration
            valid, err = record.is_valid(task_id=task_id, action=action, token=token)
            if not valid:
                return False, err

            # Check metadata bindings if present
            meta = getattr(record, "metadata", {}) or {}
            if meta.get("workflow_id") and workflow_id and str(meta["workflow_id"]) != str(workflow_id):
                return False, f"Workflow mismatch: Token bound to workflow '{meta['workflow_id']}', not '{workflow_id}'."
            if meta.get("agent_name") and agent_name and str(meta["agent_name"]) != str(agent_name):
                return False, f"Agent mismatch: Token bound to agent '{meta['agent_name']}', not '{agent_name}'."
            if meta.get("tool_name") and tool_name and str(meta["tool_name"]).lower() != str(tool_name).lower():
                return False, f"Tool mismatch: Token bound to tool '{meta['tool_name']}', not '{tool_name}'."

            # Multi-process atomic transition in persistent storage if available
            if self.storage is not None and hasattr(self.storage, "atomic_consume_approval"):
                c_ok, c_err = self.storage.atomic_consume_approval(
                    token=token,
                    consumed_by=agent_name or "ExecutorAgent",
                    task_id=task_id,
                    action=action,
                )
                if not c_ok:
                    record.approval_status = "CONSUMED"
                    return False, c_err

            # Atomic state transition in memory
            record.approval_status = "CONSUMED"
            if not hasattr(record, "metadata") or not isinstance(record.metadata, dict):
                record.metadata = {}
            record.metadata["consumed_at"] = datetime.now(timezone.utc).isoformat()
            record.metadata["consumed_by"] = agent_name or "ExecutorAgent"

            # Persist update in storage if atomic_consume_approval was not used
            if self.storage is not None and hasattr(self.storage, "save_approval") and not hasattr(self.storage, "atomic_consume_approval"):
                try:
                    self.storage.save_approval(record.to_dict())
                except Exception as exc:
                    logger.warning(f"[HumanApprovalManager] Could not persist consumed approval status: {exc}")

            _record_hitl_lifecycle_event(
                event_category="approval_token_consumed",
                message=f"Approval token '{token[:10]}...' consumed for task '{task_id}'.",
                observability=self.observability,
                task_id=task_id or record.task_id,
                payload={"token": "***REDACTED***", "task_id": task_id, "action": action},
            )

            logger.info(f"[HumanApprovalManager] Token '{token[:10]}...' successfully consumed.")
            return True, None


# ===========================================================================
# 5. Part 3: HumanFeedbackManager
# ===========================================================================

class HumanFeedbackManager:
    """Manages human clarification, missing information, and workflow feedback.

    Responsibilities:
    - Allows agents (Researcher, Planner, Supervisor) to request human clarification.
    - Supports CLARIFICATION, INFORMATION, FEEDBACK, and PLAN_MODIFICATION requests.
    - Pauses the workflow safely, persisting completed work without duplication.
    - Validates incoming human feedback, rejecting empty, duplicate, or misdirected responses.
    - Attaches feedback to TaskContext and resumes execution from the exact paused stage.
    - Prevents resumption of cancelled workflows.
    - Allows the Supervisor to reconsider previous delegation decisions with user feedback.
    """

    def __init__(
        self,
        interaction_manager: Optional[HumanInteractionManager] = None,
        storage: Optional[SQLiteMessageStorage] = None,
        observability: Optional[Any] = None,
    ):
        self.storage = storage
        self.interaction_manager = interaction_manager or HumanInteractionManager(storage=storage, observability=observability)
        self.observability = observability
        self._lock = threading.RLock()
        self._processed_feedback_ids: Set[str] = set()

    def request_clarification(
        self,
        task_id: str,
        correlation_id: str,
        requesting_agent: str,
        question: str,
        relevant_context: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
        request_type: InteractionType = InteractionType.CLARIFICATION_REQUEST,
    ) -> HumanInteractionRequest:
        """Pause the workflow and request clarification from the human."""
        with self._lock:
            # 1. Create clarification interaction request
            interaction = self.interaction_manager.create_interaction(
                task_id=task_id,
                correlation_id=correlation_id,
                requesting_agent=requesting_agent,
                interaction_type=request_type,
                question_or_request=question,
                relevant_context=relevant_context or {},
                available_decisions=["PROVIDE_INFO", "CANCEL"],
                context=context,
            )

            # 2. Pause the workflow and persist completed work
            if context is not None:
                context.status = WorkflowStatus.PAUSED
                context.metadata["clarification_pending"] = True
                context.metadata["pending_interaction_id"] = interaction.interaction_id
                context.metadata["clarification_question"] = question

                # Persist workflow state snapshot to SQLite
                if self.storage is not None and hasattr(self.storage, "save_workflow_state"):
                    self.storage.save_workflow_state(
                        workflow_id=context.workflow_id,
                        task_id=context.task_id,
                        status=WorkflowStatus.PAUSED.value,
                        current_stage=context.metadata.get("current_stage", "clarification"),
                        context_data=context.to_dict(),
                    )

            _record_hitl_lifecycle_event(
                event_category="human_clarification_requested",
                message=f"Clarification requested for task '{task_id}': '{question[:60]}...'",
                observability=self.observability,
                task_id=task_id,
                workflow_id=getattr(context, "workflow_id", None),
                payload={"question": question, "requesting_agent": requesting_agent, "interaction_id": interaction.interaction_id},
            )
            _record_hitl_lifecycle_event(
                event_category="workflow_paused",
                message=f"Workflow '{task_id}' paused awaiting clarification.",
                observability=self.observability,
                task_id=task_id,
                workflow_id=getattr(context, "workflow_id", None),
                payload={"reason": "clarification_pending", "interaction_id": interaction.interaction_id},
            )

            logger.info(
                f"[HumanFeedbackManager] Workflow '{task_id}' paused for clarification: '{question[:60]}...'"
            )
            return interaction

    def submit_feedback(
        self,
        interaction_id: str,
        response_text: str,
        task_id: Optional[str] = None,
        context: Optional[TaskContext] = None,
    ) -> Tuple[bool, str, Optional[HumanInteractionRequest]]:
        """Submit human feedback/clarification for a pending interaction."""
        with self._lock:
            # 1. Validate response content
            if not response_text or not str(response_text).strip():
                return False, "Validation rejected: Feedback cannot be empty.", None

            cleaned_resp = str(response_text).strip()

            # 2. Retrieve interaction
            interaction = self.interaction_manager.get_interaction(interaction_id)
            if not interaction:
                return False, f"Interaction '{interaction_id}' not found.", None

            # 3. Prevent duplicate response processing
            if interaction_id in self._processed_feedback_ids or not interaction.is_pending():
                return False, f"Duplicate response rejected: Interaction '{interaction_id}' has already been processed.", None

            # 4. Prevent response attachment to wrong task
            if task_id is not None and str(interaction.task_id) != str(task_id):
                return False, f"Task mismatch: Response is directed to task '{task_id}', but interaction is bound to '{interaction.task_id}'.", None

            # 5. Resolve interaction
            self.interaction_manager.resolve_interaction(
                interaction_id=interaction_id,
                decision="PROVIDED",
                human_response=cleaned_resp,
                context=context,
            )
            self._processed_feedback_ids.add(interaction_id)

            # 6. Update TaskContext if provided
            if context is not None:
                if "human_feedback" not in context.metadata:
                    context.metadata["human_feedback"] = []
                context.metadata["human_feedback"].append({
                    "interaction_id": interaction_id,
                    "question": interaction.question_or_request,
                    "response": cleaned_resp,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                context.metadata["clarification_pending"] = False

            _record_hitl_lifecycle_event(
                event_category="human_clarification_received",
                message=f"Feedback accepted for interaction {interaction_id} on task '{interaction.task_id}'.",
                observability=self.observability,
                task_id=interaction.task_id,
                payload={"interaction_id": interaction_id, "response_preview": cleaned_resp[:100]},
            )

            logger.info(
                f"[HumanFeedbackManager] Feedback accepted for interaction {interaction_id} on task '{interaction.task_id}'"
            )
            return True, "Feedback accepted successfully.", interaction

    def resume_workflow(
        self,
        workflow_id_or_context: Union[str, TaskContext],
        orchestrator: Any,
        feedback_response: Optional[str] = None,
    ) -> TaskContext:
        """Resume a paused workflow after human feedback has been provided.

        Preserves all completed work (research findings, plan, execution steps)
        and continues execution without repeating completed stages.
        """
        with self._lock:
            context: TaskContext
            if isinstance(workflow_id_or_context, TaskContext):
                context = workflow_id_or_context
            else:
                # Load context from SQLite
                if self.storage is None or not hasattr(self.storage, "get_workflow_state"):
                    raise ValueError("Cannot resume by workflow ID without persistent storage.")
                row = self.storage.get_workflow_state(str(workflow_id_or_context))
                if not row:
                    raise ValueError(f"Workflow state for '{workflow_id_or_context}' not found.")
                ctx_data = row["context"]
                context = TaskContext(
                    workflow_id=row["workflow_id"],
                    task_id=row["task_id"],
                    user_request=ctx_data.get("user_request", ""),
                    status=WorkflowStatus(row["status"]),
                    research_findings=ctx_data.get("research_findings"),
                    execution_plan=ctx_data.get("execution_plan"),
                    execution_results=ctx_data.get("execution_results", []),
                    final_response=ctx_data.get("final_response"),
                    errors=ctx_data.get("errors", []),
                    metadata=ctx_data.get("metadata", {}),
                )

            # Safety check: Cannot resume cancelled workflow
            if context.status in (WorkflowStatus.STOPPED, WorkflowStatus.FAILED):
                err = f"Cannot resume workflow '{context.workflow_id}': status is {context.status.value}"
                logger.error(err)
                raise RuntimeError(err)

            # Attach feedback if supplied directly
            if feedback_response:
                context.metadata["latest_feedback"] = feedback_response
                context.user_request = f"{context.user_request}\n[User Clarification]: {feedback_response}"

            context.status = WorkflowStatus.INITIALIZED
            context.metadata["clarification_pending"] = False

            _record_hitl_lifecycle_event(
                event_category="workflow_resumed",
                message=f"Resuming workflow '{context.workflow_id}' from preserved checkpoint.",
                observability=self.observability,
                task_id=context.task_id,
                workflow_id=context.workflow_id,
            )

            logger.info(
                f"[HumanFeedbackManager] Resuming workflow '{context.workflow_id}' from preserved checkpoint."
            )

            # Run continuation through orchestrator
            if hasattr(orchestrator, "run_communication_workflow"):
                resumed_ctx = orchestrator.run_communication_workflow(
                    user_request=context.user_request,
                    task_id=context.workflow_id,
                )
                return resumed_ctx
            elif hasattr(orchestrator, "run"):
                resumed_ctx = orchestrator.run(
                    user_request=context.user_request,
                    task_id=context.workflow_id,
                )
                return resumed_ctx

            return context


# ===========================================================================
# 6. Sensitive Data Redaction Helper
# ===========================================================================

REDACTION_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|secret|token|password|auth|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?"), r"\1=[REDACTED]"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{15,}"), "Bearer [REDACTED]"),
]


def redact_sensitive_data(val: Any) -> Any:
    """Recursively scrub secrets, API keys, and sensitive tokens from audit logs."""
    if isinstance(val, str):
        res = val
        for pat, repl in REDACTION_PATTERNS:
            res = pat.sub(repl, res)
        return res
    elif isinstance(val, dict):
        cleaned = {}
        for k, v in val.items():
            if any(s in str(k).lower() for s in ("key", "secret", "token", "password", "auth", "credential")):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = redact_sensitive_data(v)
        return cleaned
    elif isinstance(val, list):
        return [redact_sensitive_data(item) for item in val]
    return val


# ===========================================================================
# 7. Part 4: Risk-Based Human Intervention
# ===========================================================================

class RiskLevel(str, Enum):
    """Configurable action risk classifications."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def score_rank(self) -> int:
        ranks = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3, RiskLevel.CRITICAL: 4}
        return ranks.get(self, 1)

    @property
    def score_threshold(self) -> float:
        thresholds = {RiskLevel.LOW: 0.1, RiskLevel.MEDIUM: 0.4, RiskLevel.HIGH: 0.7, RiskLevel.CRITICAL: 0.9}
        return thresholds.get(self, 0.1)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, RiskLevel):
            return self.score_rank < other.score_rank
        return False

    def __le__(self, other: Any) -> bool:
        if isinstance(other, RiskLevel):
            return self.score_rank <= other.score_rank
        return False

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, RiskLevel):
            return self.score_rank >= other.score_rank
        return False

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, RiskLevel):
            return self.score_rank > other.score_rank
        return False


@dataclass
class RiskPolicyConfig:
    """Configurable centralized policy for risk evaluation and required controls."""
    allow_auto_approval_for_low: bool = True
    medium_requires_validation: bool = True
    high_requires_human_approval: bool = True
    critical_requires_stricter_controls: bool = True
    critical_ttl_seconds: float = 300.0   # 5 minute TTL for critical actions
    high_ttl_seconds: float = 1800.0       # 30 minute TTL for high risk actions
    custom_rules: Dict[str, Union[RiskLevel, str]] = field(default_factory=dict)


@dataclass
class RiskAssessmentResult:
    """Structured contract describing an action's risk evaluation."""
    assessment_id: str
    task_id: str
    action: str
    risk_level: Union[RiskLevel, str]
    risk_score: float
    identified_risks: List[str] = field(default_factory=list)
    required_controls: List[str] = field(default_factory=list)
    approval_required: bool = False
    assessment_reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "task_id": self.task_id,
            "action": self.action,
            "risk_level": self.risk_level.value if isinstance(self.risk_level, RiskLevel) else str(self.risk_level),
            "risk_score": self.risk_score,
            "identified_risks": self.identified_risks,
            "required_controls": self.required_controls,
            "approval_required": self.approval_required,
            "assessment_reason": self.assessment_reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RiskAssessmentResult:
        raw_level = data.get("risk_level", "LOW")
        try:
            rlevel = RiskLevel(str(raw_level))
        except ValueError:
            rlevel = str(raw_level)

        return cls(
            assessment_id=data["assessment_id"],
            task_id=data["task_id"],
            action=data["action"],
            risk_level=rlevel,
            risk_score=float(data.get("risk_score", 0.0)),
            identified_risks=data.get("identified_risks", []),
            required_controls=data.get("required_controls", []),
            approval_required=bool(data.get("approval_required", False)),
            assessment_reason=data.get("assessment_reason", ""),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


class RiskAssessmentEngine:
    """Evaluates agent actions, classifies risk, and determines required human controls.

    Evaluates:
    - Action type (read, write, delete, permissions, financial)
    - Potential external side effects (email, payment, external API)
    - Reversibility (read=reversible, drop=irreversible)
    - Financial or operational impact
    - Data sensitivity (PII, credentials)
    - Configurable centralized RiskPolicyConfig
    - Enforces: Never allow low-risk classification to bypass mandatory authorization
    - Reassesses modified actions before execution
    """

    CRITICAL_PATTERNS = [
        r"\b(drop\s+database|drop\s+table|drop\s+schema)\b",
        r"\b(truncate\s+table|destroy\s+cluster|wipe\s+disk|format\s+drive)\b",
        r"\b(transfer\s+funds?|process\s+payment|billing|credit\s+card)\b",
        r"\b(grant\s+admin|grant\s+sudo|grant\s+superuser|sudo\s+rm)\b",
    ]

    HIGH_PATTERNS = [
        r"\b(delete|remove|destroy|purge|erase)\b",
        r"\b(modify\s+config|change\s+configuration|overwrite\s+settings)\b",
        r"\b(send\s+email|send\s+message|send\s+notification|post\s+tweet)\b",
        r"\b(chmod|chown|grant\s+permission|revoke\s+permission)\b",
    ]

    MEDIUM_PATTERNS = [
        r"\b(write|create|update|insert|modify|restart|upload)\b",
        r"\b(backup|archive|export|import)\b",
    ]

    LOW_PATTERNS = [
        r"\b(read|select|calculate|search|get|summarize|inspect|verify|query)\b",
    ]

    def __init__(
        self,
        policy: Optional[RiskPolicyConfig] = None,
        storage: Optional[SQLiteMessageStorage] = None,
        audit_trail: Optional[Any] = None,
    ):
        self.policy = policy or RiskPolicyConfig()
        self.storage = storage
        self.audit_trail = audit_trail
        self._lock = threading.RLock()
        self._assessments: Dict[str, RiskAssessmentResult] = {}
        if self.storage is not None:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Hydrate recent risk assessments from SQLite storage."""
        if hasattr(self.storage, "list_risk_assessments"):
            rows = self.storage.list_risk_assessments()
            for r in rows:
                res = RiskAssessmentResult.from_dict(r)
                self._assessments[res.assessment_id] = res

    def evaluate_action(
        self,
        task_id: str,
        action: Union[str, Dict[str, Any]],
        context: Optional[TaskContext] = None,
        explicit_requires_approval: bool = False,
    ) -> RiskAssessmentResult:
        """Evaluate an action and determine its risk level and required controls."""
        with self._lock:
            from agents.executor import is_sensitive_operation

            action_desc = ""
            action_meta: Dict[str, Any] = {}
            if isinstance(action, dict):
                action_desc = str(action.get("description", "")).strip()
                action_meta = action
                if action.get("requires_approval") is True:
                    explicit_requires_approval = True
            else:
                action_desc = str(action).strip()

            assessment_id = f"risk_{uuid.uuid4().hex[:8]}"
            norm_action = action_desc.lower()

            identified_risks: List[str] = []
            required_controls: List[str] = []
            score = 0.0
            risk_level = RiskLevel.LOW
            reason_parts: List[str] = []

            # 1. Custom policy rule checks (explicit overrides)
            custom_matched = False
            for pattern, custom_level in self.policy.custom_rules.items():
                if re.search(pattern, norm_action, re.IGNORECASE):
                    c_lvl = custom_level if isinstance(custom_level, RiskLevel) else RiskLevel(str(custom_level))
                    risk_level = c_lvl
                    score = max(score, c_lvl.score_threshold)
                    identified_risks.append(f"Matched custom rule '{pattern}'")
                    reason_parts.append(f"Custom policy rule matched '{pattern}'")
                    custom_matched = True
                    break

            # 2. Pattern checks (if not explicitly overridden by custom rule)
            if not custom_matched:
                for pat in self.CRITICAL_PATTERNS:
                    if re.search(pat, norm_action, re.IGNORECASE):
                        risk_level = RiskLevel.CRITICAL
                        score = max(score, 0.95)
                        identified_risks.append(f"Critical pattern: {pat}")
                        reason_parts.append("Contains irreversible, high-impact, or administrative operations.")
                        break

            # 3. High pattern checks (if not critical and not custom matched)
            if not custom_matched and risk_level < RiskLevel.CRITICAL:
                for pat in self.HIGH_PATTERNS:
                    if re.search(pat, norm_action, re.IGNORECASE):
                        risk_level = RiskLevel.HIGH
                        score = max(score, 0.75)
                        identified_risks.append(f"High risk pattern: {pat}")
                        reason_parts.append("Contains potentially destructive or external-facing side effects.")
                        break

            # 4. Medium pattern checks (if low and not custom matched)
            if not custom_matched and risk_level == RiskLevel.LOW:
                for pat in self.MEDIUM_PATTERNS:
                    if re.search(pat, norm_action, re.IGNORECASE):
                        risk_level = RiskLevel.MEDIUM
                        score = max(score, 0.45)
                        identified_risks.append(f"Medium risk pattern: {pat}")
                        reason_parts.append("Contains state mutation or modification operations.")
                        break

            # 5. Low pattern checks
            if risk_level == RiskLevel.LOW and not identified_risks:
                score = 0.10
                reason_parts.append("Read-only, analytical, or purely informative operation.")

            # 6. Sensitive operation override
            if is_sensitive_operation(action_desc) and risk_level < RiskLevel.HIGH:
                risk_level = RiskLevel.HIGH
                score = max(score, 0.75)
                identified_risks.append("Matches system-wide sensitive operation boundary")
                reason_parts.append("Identified by system sensitive patterns.")

            # 7. Determine Approval & Required Controls based on RiskPolicyConfig
            approval_required = False
            if risk_level == RiskLevel.CRITICAL:
                approval_required = True
                required_controls.extend([
                    "explicit_human_authorization",
                    "stricter_approval_controls",
                    f"ttl_limited_to_{int(self.policy.critical_ttl_seconds)}s",
                    "audit_logging",
                ])
            elif risk_level == RiskLevel.HIGH:
                approval_required = self.policy.high_requires_human_approval
                required_controls.extend([
                    "explicit_human_approval",
                    f"ttl_limited_to_{int(self.policy.high_ttl_seconds)}s",
                    "audit_logging",
                ])
            elif risk_level == RiskLevel.MEDIUM:
                if self.policy.medium_requires_validation:
                    required_controls.append("parameter_validation")
                required_controls.append("audit_logging")
            else:
                if self.policy.allow_auto_approval_for_low:
                    required_controls.append("automated_execution")

            # 8. MANDATORY OVERRIDE: Never allow a low-risk classification to bypass mandatory authorization
            if explicit_requires_approval:
                approval_required = True
                if "mandatory_authorization" not in required_controls:
                    required_controls.append("mandatory_authorization")
                reason_parts.append("Mandatory authorization enforced by task requirements.")

            assessment = RiskAssessmentResult(
                assessment_id=assessment_id,
                task_id=task_id,
                action=action_desc,
                risk_level=risk_level,
                risk_score=score,
                identified_risks=identified_risks,
                required_controls=required_controls,
                approval_required=approval_required,
                assessment_reason="; ".join(reason_parts) if reason_parts else "Standard policy evaluation.",
                metadata=action_meta,
            )

            self._assessments[assessment_id] = assessment

            # Persist to SQLite
            if self.storage is not None and hasattr(self.storage, "save_risk_assessment"):
                self.storage.save_risk_assessment(assessment.to_dict())

            # Log to audit trail
            if self.audit_trail is not None and hasattr(self.audit_trail, "record_event"):
                self.audit_trail.record_event(
                    task_id=task_id,
                    event_type=AuditEventType.RISK_ASSESSMENT,
                    description=f"Action assessed as {risk_level.value} (score={score:.2f}, approval_required={approval_required})",
                    agent_name="RiskAssessmentEngine",
                    payload=assessment.to_dict(),
                    context=context,
                )

            logger.info(
                f"[RiskAssessmentEngine] Task '{task_id}' action '{action_desc[:50]}' assessed as {risk_level.value} (score={score:.2f}, approval_required={approval_required})"
            )
            return assessment

    def reassess_if_modified(
        self,
        previous_assessment: RiskAssessmentResult,
        current_action: Union[str, Dict[str, Any]],
        task_id: Optional[str] = None,
        context: Optional[TaskContext] = None,
    ) -> Tuple[bool, RiskAssessmentResult]:
        """Check if action was modified since previous assessment, and reassess if so.

        Returns:
            Tuple[was_modified: bool, assessment: RiskAssessmentResult]
        """
        curr_desc = (
            str(current_action.get("description", "")).strip()
            if isinstance(current_action, dict)
            else str(current_action).strip()
        )
        prev_desc = previous_assessment.action.strip()

        # If normalized action differs, force reassessment
        if curr_desc.lower() != prev_desc.lower():
            logger.warning(
                f"[RiskAssessmentEngine] Action modification detected! Previous: '{prev_desc}', New: '{curr_desc}'. Invalidating prior approval."
            )
            new_assessment = self.evaluate_action(
                task_id=task_id or previous_assessment.task_id,
                action=current_action,
                context=context,
            )
            new_assessment.metadata["modified_from"] = prev_desc
            return True, new_assessment

        return False, previous_assessment

    def get_assessment(self, assessment_id: str) -> Optional[RiskAssessmentResult]:
        """Retrieve risk assessment by ID."""
        with self._lock:
            return self._assessments.get(assessment_id)

    def list_assessments(
        self,
        task_id: Optional[str] = None,
        risk_level: Optional[Union[RiskLevel, str]] = None,
    ) -> List[RiskAssessmentResult]:
        """Query risk assessments with filtering."""
        with self._lock:
            lvl_str = risk_level.value if isinstance(risk_level, RiskLevel) else (str(risk_level) if risk_level else None)
            res = list(self._assessments.values())
            if task_id:
                res = [r for r in res if str(r.task_id) == str(task_id)]
            if lvl_str:
                res = [
                    r for r in res
                    if (r.risk_level.value if isinstance(r.risk_level, RiskLevel) else str(r.risk_level)) == lvl_str
                ]
            return res


# ===========================================================================
# 8. Part 5: Human Review System
# ===========================================================================

class ReviewOutcome(str, Enum):
    """Supported human review outcomes."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    NEEDS_REVISION = "NEEDS_REVISION"
    ESCALATED = "ESCALATED"


@dataclass
class HumanReviewRecord:
    """Structured record of human review for research, plans, decisions, or results."""
    review_id: str
    task_id: str
    correlation_id: str
    target_type: str  # "research", "plan", "decision", "execution", "response"
    target_data: Dict[str, Any]
    reviewer: str
    outcome: Union[ReviewOutcome, str]
    feedback: Optional[str] = None
    review_round: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
            "target_type": self.target_type,
            "target_data": self.target_data,
            "reviewer": self.reviewer,
            "outcome": self.outcome.value if isinstance(self.outcome, ReviewOutcome) else str(self.outcome),
            "feedback": self.feedback,
            "review_round": self.review_round,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HumanReviewRecord:
        raw_out = data.get("outcome", "ACCEPTED")
        try:
            out = ReviewOutcome(str(raw_out))
        except ValueError:
            out = str(raw_out)

        return cls(
            review_id=data["review_id"],
            task_id=data["task_id"],
            correlation_id=data.get("correlation_id", data["task_id"]),
            target_type=data.get("target_type", "plan"),
            target_data=data.get("target_data", {}),
            reviewer=data.get("reviewer", "HumanReviewer"),
            outcome=out,
            feedback=data.get("feedback"),
            review_round=int(data.get("review_round", 1)),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


class HumanReviewManager:
    """Manages human review of research findings, execution plans, agent decisions, and tool results."""

    def __init__(
        self,
        storage: Optional[SQLiteMessageStorage] = None,
        audit_trail: Optional[Any] = None,
        authorized_reviewers: Optional[Set[str]] = None,
        observability: Optional[Any] = None,
    ):
        self.storage = storage
        self.audit_trail = audit_trail
        self.authorized_reviewers = authorized_reviewers
        self.observability = observability
        self._lock = threading.RLock()
        self._reviews: Dict[str, HumanReviewRecord] = {}
        self._rounds_by_target: Dict[str, int] = {}
        if self.storage is not None:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Hydrate review records from SQLite storage."""
        if hasattr(self.storage, "list_reviews"):
            rows = self.storage.list_reviews()
            for r in rows:
                rev = HumanReviewRecord.from_dict(r)
                self._reviews[rev.review_id] = rev
                target_key = f"{rev.task_id}_{rev.target_type}"
                self._rounds_by_target[target_key] = max(
                    self._rounds_by_target.get(target_key, 0), rev.review_round
                )

    def set_authorized_reviewers(self, reviewers: Set[str]) -> None:
        """Configure authorized reviewer pool."""
        with self._lock:
            self.authorized_reviewers = set(reviewers)

    def submit_review(
        self,
        task_id: str,
        target_type: str,
        target_data: Dict[str, Any],
        reviewer: str,
        outcome: Union[ReviewOutcome, str],
        feedback: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
    ) -> HumanReviewRecord:
        """Submit a formal human review for an agent artifact."""
        with self._lock:
            # 1. Authorize reviewer
            if not reviewer or not str(reviewer).strip():
                raise ValueError("Reviewer identity cannot be empty.")
            clean_reviewer = str(reviewer).strip()

            if self.authorized_reviewers is not None and clean_reviewer not in self.authorized_reviewers:
                err = f"Unauthorized reviewer '{clean_reviewer}'. Not in authorized review pool."
                logger.warning(f"[HumanReviewManager] {err}")
                raise PermissionError(err)

            # 2. Track review rounds
            target_key = f"{task_id}_{target_type}"
            current_round = self._rounds_by_target.get(target_key, 0) + 1
            self._rounds_by_target[target_key] = current_round

            out = outcome if isinstance(outcome, ReviewOutcome) else ReviewOutcome(str(outcome))
            review_id = f"rev_{uuid.uuid4().hex[:8]}"

            record = HumanReviewRecord(
                review_id=review_id,
                task_id=task_id,
                correlation_id=correlation_id or task_id,
                target_type=target_type,
                target_data=target_data or {},
                reviewer=clean_reviewer,
                outcome=out,
                feedback=feedback,
                review_round=current_round,
                metadata=metadata or {},
            )

            self._reviews[review_id] = record

            # Persist to SQLite
            if self.storage is not None and hasattr(self.storage, "save_review"):
                self.storage.save_review(record.to_dict())

            # Update context if provided
            if context is not None:
                if "human_reviews" not in context.metadata:
                    context.metadata["human_reviews"] = []
                context.metadata["human_reviews"].append(record.to_dict())

            # Log to audit trail
            if self.audit_trail is not None and hasattr(self.audit_trail, "record_event"):
                self.audit_trail.record_event(
                    task_id=task_id,
                    event_type=AuditEventType.REVIEW_OUTCOME,
                    description=f"{target_type.title()} review by {clean_reviewer}: {out.value} (Round {current_round})",
                    agent_name=clean_reviewer,
                    payload={"review_id": review_id, "outcome": out.value, "feedback": feedback},
                    context=context,
                )

            _record_hitl_lifecycle_event(
                event_category="human_review_completed",
                message=f"Human review completed for {target_type} on task '{task_id}' with outcome '{out.value}'.",
                observability=self.observability,
                task_id=task_id,
                payload={
                    "review_id": review_id,
                    "target_type": target_type,
                    "reviewer": clean_reviewer,
                    "outcome": out.value,
                    "review_round": current_round,
                },
            )

            logger.info(
                f"[HumanReviewManager] Review {review_id} for '{task_id}' ({target_type}) by '{clean_reviewer}': outcome={out.value} (Round {current_round})"
            )
            return record

    def get_review(self, review_id: str) -> Optional[HumanReviewRecord]:
        """Fetch review by ID."""
        with self._lock:
            return self._reviews.get(review_id)

    def list_reviews(
        self,
        task_id: Optional[str] = None,
        outcome: Optional[Union[ReviewOutcome, str]] = None,
        target_type: Optional[str] = None,
    ) -> List[HumanReviewRecord]:
        """Query review history with filters."""
        with self._lock:
            out_str = outcome.value if isinstance(outcome, ReviewOutcome) else (str(outcome) if outcome else None)
            res = list(self._reviews.values())
            if task_id:
                res = [r for r in res if str(r.task_id) == str(task_id)]
            if out_str:
                res = [
                    r for r in res
                    if (r.outcome.value if isinstance(r.outcome, ReviewOutcome) else str(r.outcome)) == out_str
                ]
            if target_type:
                res = [r for r in res if str(r.target_type) == str(target_type)]
            return res

    def get_latest_review(
        self,
        task_id: str,
        target_type: Optional[str] = None,
    ) -> Optional[HumanReviewRecord]:
        """Get the most recent review for a task and optional target type."""
        revs = self.list_reviews(task_id=task_id, target_type=target_type)
        return revs[-1] if revs else None


# ===========================================================================
# 9. Part 5: Unified Audit Trail
# ===========================================================================

class AuditEventType(str, Enum):
    """Categorization of audit events."""
    AGENT_ACTION = "agent_action"
    RISK_ASSESSMENT = "risk_assessment"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_DECISION = "approval_decision"
    APPROVAL_REJECTION = "approval_rejection"
    CLARIFICATION_REQUEST = "clarification_request"
    HUMAN_FEEDBACK = "human_feedback"
    EXECUTION_ATTEMPT = "execution_attempt"
    REVIEW_OUTCOME = "review_outcome"
    WORKFLOW_RESUMPTION = "workflow_resumption"
    WORKFLOW_PAUSE = "workflow_pause"
    POLICY_PROPOSAL = "policy_proposal"
    POLICY_APPLIED = "policy_applied"
    ERROR_ALERT = "error_alert"


@dataclass
class AuditEvent:
    """Immutable audit trail record capturing agent and human events."""
    event_uuid: str
    task_id: str
    correlation_id: Optional[str]
    event_type: Union[AuditEventType, str]
    agent_name: Optional[str]
    description: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_uuid": self.event_uuid,
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
            "event_type": self.event_type.value if isinstance(self.event_type, AuditEventType) else str(self.event_type),
            "agent_name": self.agent_name,
            "description": self.description,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditEvent:
        raw_type = data.get("event_type", "agent_action")
        try:
            etype = AuditEventType(str(raw_type))
        except ValueError:
            etype = str(raw_type)

        return cls(
            event_id=data.get("event_id"),
            event_uuid=data["event_uuid"],
            task_id=data["task_id"],
            correlation_id=data.get("correlation_id"),
            event_type=etype,
            agent_name=data.get("agent_name"),
            description=data.get("description", ""),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


class AuditTrail:
    """Unified append-only audit trail persisting ordered events to SQLite with secret redaction."""

    def __init__(self, storage: Optional[SQLiteMessageStorage] = None):
        self.storage = storage
        self._lock = threading.RLock()
        self._events: List[AuditEvent] = []
        if self.storage is not None:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Hydrate recent audit events from storage."""
        if hasattr(self.storage, "list_audit_events"):
            rows = self.storage.list_audit_events(limit=500)
            for r in rows:
                self._events.append(AuditEvent.from_dict(r))

    def record_event(
        self,
        task_id: str,
        event_type: Union[AuditEventType, str],
        description: str,
        agent_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
    ) -> AuditEvent:
        """Append an event to the audit trail with automatic secret scrubbing."""
        with self._lock:
            event_uuid = f"aud_{uuid.uuid4().hex[:12]}"
            clean_payload = redact_sensitive_data(payload or {})
            clean_desc = redact_sensitive_data(description)
            etype = event_type if isinstance(event_type, AuditEventType) else AuditEventType(str(event_type))

            event = AuditEvent(
                event_uuid=event_uuid,
                task_id=task_id,
                correlation_id=correlation_id or task_id,
                event_type=etype,
                agent_name=agent_name,
                description=clean_desc,
                payload=clean_payload,
            )

            self._events.append(event)

            # Persist to SQLite
            if self.storage is not None and hasattr(self.storage, "save_audit_event"):
                self.storage.save_audit_event(event.to_dict())

            # Update context if provided
            if context is not None:
                if "audit_trail" not in context.metadata:
                    context.metadata["audit_trail"] = []
                context.metadata["audit_trail"].append(event.to_dict())

            return event

    def query(
        self,
        task_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_type: Optional[Union[AuditEventType, str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Filter audit events with contextual parameters."""
        with self._lock:
            # Query from storage if available for freshness
            if self.storage is not None and hasattr(self.storage, "list_audit_events"):
                rows = self.storage.list_audit_events(
                    task_id=task_id,
                    agent_name=agent_name,
                    event_type=event_type.value if isinstance(event_type, AuditEventType) else event_type,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                )
                return [AuditEvent.from_dict(r) for r in rows]

            # In-memory fallback
            etype_str = event_type.value if isinstance(event_type, AuditEventType) else (str(event_type) if event_type else None)
            res = list(self._events)
            if task_id:
                res = [e for e in res if str(e.task_id) == str(task_id)]
            if agent_name:
                res = [e for e in res if str(e.agent_name) == str(agent_name)]
            if etype_str:
                res = [
                    e for e in res
                    if (e.event_type.value if isinstance(e.event_type, AuditEventType) else str(e.event_type)) == etype_str
                ]
            if start_time:
                res = [e for e in res if e.timestamp >= start_time]
            if end_time:
                res = [e for e in res if e.timestamp <= end_time]
            return res[:limit]


# ===========================================================================
# 10. Part 5: Feedback Intelligence & Advisory Proposals
# ===========================================================================

@dataclass
class FeedbackSummary:
    """Aggregated analytical summary of accumulated reviews and human feedback."""
    analysis_id: str
    summary_title: str
    total_reviews: int
    total_feedback_items: int
    recurring_issue_categories: List[str]
    agent_failure_rates: Dict[str, float]
    observed_patterns: List[str]
    actionable_recommendations: List[str]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "summary_title": self.summary_title,
            "total_reviews": self.total_reviews,
            "total_feedback_items": self.total_feedback_items,
            "recurring_issue_categories": self.recurring_issue_categories,
            "agent_failure_rates": self.agent_failure_rates,
            "observed_patterns": self.observed_patterns,
            "actionable_recommendations": self.actionable_recommendations,
            "generated_at": self.generated_at,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeedbackSummary:
        return cls(
            analysis_id=data["analysis_id"],
            summary_title=data["summary_title"],
            total_reviews=int(data.get("total_reviews", 0)),
            total_feedback_items=int(data.get("total_feedback_items", 0)),
            recurring_issue_categories=data.get("recurring_issue_categories", []),
            agent_failure_rates=data.get("agent_failure_rates", {}),
            observed_patterns=data.get("observed_patterns", []),
            actionable_recommendations=data.get("actionable_recommendations", []),
            generated_at=data.get("generated_at", datetime.now(timezone.utc).isoformat()),
            time_window_start=data.get("time_window_start"),
            time_window_end=data.get("time_window_end"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PolicyChangeProposal:
    """Structured advisory proposal for improving agent instructions or workflow policy."""
    proposal_id: str
    analysis_id: Optional[str]
    proposed_change: str
    target_policy: str
    rationale: str
    status: str = "PROPOSED"  # "PROPOSED", "APPROVED", "REJECTED", "APPLIED"
    authorized_by: Optional[str] = None
    authorized_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "analysis_id": self.analysis_id,
            "proposed_change": self.proposed_change,
            "target_policy": self.target_policy,
            "rationale": self.rationale,
            "status": self.status,
            "authorized_by": self.authorized_by,
            "authorized_at": self.authorized_at,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyChangeProposal:
        return cls(
            proposal_id=data["proposal_id"],
            analysis_id=data.get("analysis_id"),
            proposed_change=data["proposed_change"],
            target_policy=data["target_policy"],
            rationale=data["rationale"],
            status=data.get("status", "PROPOSED"),
            authorized_by=data.get("authorized_by"),
            authorized_at=data.get("authorized_at"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


class FeedbackAnalysisEngine:
    """Analyzes accumulated human reviews and clarification feedback to identify weaknesses.

    Advisory-Only Safety Rule:
    Does NOT automatically rewrite prompts or alter agent behaviors.
    Produces structured summaries, distinguishes observed evidence from inferred recommendations,
    and requires explicit human authorization before applying policy-change proposals.
    """

    def __init__(
        self,
        review_manager: Optional[HumanReviewManager] = None,
        interaction_manager: Optional[HumanInteractionManager] = None,
        storage: Optional[SQLiteMessageStorage] = None,
        audit_trail: Optional[AuditTrail] = None,
    ):
        self.review_manager = review_manager
        self.interaction_manager = interaction_manager
        self.storage = storage
        self.audit_trail = audit_trail
        self._lock = threading.RLock()
        self._summaries: Dict[str, FeedbackSummary] = {}
        self._proposals: Dict[str, PolicyChangeProposal] = {}
        if self.storage is not None:
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Hydrate past analyses and proposals."""
        if hasattr(self.storage, "list_feedback_analyses"):
            for r in self.storage.list_feedback_analyses():
                s = FeedbackSummary.from_dict(r)
                self._summaries[s.analysis_id] = s
        if hasattr(self.storage, "list_policy_proposals"):
            for p in self.storage.list_policy_proposals():
                prop = PolicyChangeProposal.from_dict(p)
                self._proposals[prop.proposal_id] = prop

    def generate_summary(
        self,
        time_window_start: Optional[str] = None,
        time_window_end: Optional[str] = None,
        title: Optional[str] = None,
    ) -> FeedbackSummary:
        """Analyze accumulated feedback and generate advisory intelligence report."""
        with self._lock:
            analysis_id = f"fba_{uuid.uuid4().hex[:8]}"

            reviews: List[HumanReviewRecord] = []
            if self.review_manager is not None:
                reviews = self.review_manager.list_reviews()
            elif self.storage is not None and hasattr(self.storage, "list_reviews"):
                reviews = [HumanReviewRecord.from_dict(r) for r in self.storage.list_reviews()]

            interactions: List[HumanInteractionRequest] = []
            if self.interaction_manager is not None:
                interactions = self.interaction_manager.list_interactions(status=InteractionStatus.RESOLVED)
            elif self.storage is not None and hasattr(self.storage, "list_interactions"):
                interactions = [HumanInteractionRequest.from_dict(r) for r in self.storage.list_interactions(status="resolved")]

            # Time filtering
            if time_window_start:
                reviews = [r for r in reviews if r.timestamp >= time_window_start]
                interactions = [i for i in interactions if (i.resolved_at or i.created_at) >= time_window_start]
            if time_window_end:
                reviews = [r for r in reviews if r.timestamp <= time_window_end]
                interactions = [i for i in interactions if (i.resolved_at or i.created_at) <= time_window_end]

            total_revs = len(reviews)
            total_feedback = len(interactions)

            # Issue category extraction
            categories_count: Dict[str, int] = {}
            patterns: List[str] = []
            recommendations: List[str] = []
            target_outcomes: Dict[str, Dict[str, int]] = {}

            for rev in reviews:
                tt = rev.target_type
                if tt not in target_outcomes:
                    target_outcomes[tt] = {"ACCEPTED": 0, "REJECTED": 0, "NEEDS_REVISION": 0, "ESCALATED": 0}
                out_str = rev.outcome.value if isinstance(rev.outcome, ReviewOutcome) else str(rev.outcome)
                target_outcomes[tt][out_str] = target_outcomes[tt].get(out_str, 0) + 1

                fb_text = (rev.feedback or "").lower()
                if "missing" in fb_text or "lacks" in fb_text:
                    categories_count["missing_information"] = categories_count.get("missing_information", 0) + 1
                if "destructive" in fb_text or "dangerous" in fb_text or "unauthorized" in fb_text:
                    categories_count["excessive_risk"] = categories_count.get("excessive_risk", 0) + 1
                if "parameter" in fb_text or "invalid" in fb_text:
                    categories_count["parameter_validation"] = categories_count.get("parameter_validation", 0) + 1
                if "format" in fb_text or "syntax" in fb_text:
                    categories_count["formatting_issues"] = categories_count.get("formatting_issues", 0) + 1

            # Clarification analysis
            clarification_count = len([i for i in interactions if i.interaction_type == InteractionType.CLARIFICATION_REQUEST])
            if clarification_count > 0:
                categories_count["clarification_requirements"] = clarification_count

            # Failure rate calculation
            failure_rates: Dict[str, float] = {}
            for tt, counts in target_outcomes.items():
                total_for_target = sum(counts.values())
                unfavorable = counts.get("REJECTED", 0) + counts.get("NEEDS_REVISION", 0)
                failure_rates[tt] = round(unfavorable / total_for_target, 2) if total_for_target > 0 else 0.0

            # 1. Distinguish OBSERVED EVIDENCE
            for tt, counts in target_outcomes.items():
                if counts.get("REJECTED", 0) > 0 or counts.get("NEEDS_REVISION", 0) > 0:
                    patterns.append(
                        f"Target '{tt}' had {counts.get('REJECTED', 0)} rejections and {counts.get('NEEDS_REVISION', 0)} revision requests across {sum(counts.values())} reviews."
                    )
            if clarification_count > 0:
                patterns.append(f"Recorded {clarification_count} human clarification requests across analyzed workflows.")

            if not patterns:
                patterns.append("All analyzed reviews and feedback items were favorable without critical recurring patterns.")

            # 2. Distinguish INFERRED RECOMMENDATIONS (Advisory only)
            if categories_count.get("missing_information", 0) > 0:
                recommendations.append(
                    "Advisory: Strengthen Researcher initial query decomposition to solicit missing parameters upfront."
                )
            if categories_count.get("excessive_risk", 0) > 0:
                recommendations.append(
                    "Advisory: Enhance Planner verification to mandate non-destructive alternatives or dry-run steps."
                )
            if categories_count.get("parameter_validation", 0) > 0:
                recommendations.append(
                    "Advisory: Implement pre-execution parameter schema checks in ExecutorAgent boundary."
                )
            if not recommendations:
                recommendations.append("Advisory: Current agent workflows adhere to quality benchmarks; continue monitoring.")

            summary = FeedbackSummary(
                analysis_id=analysis_id,
                summary_title=title or f"HITL Feedback Intelligence Analysis ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})",
                total_reviews=total_revs,
                total_feedback_items=total_feedback,
                recurring_issue_categories=list(categories_count.keys()),
                agent_failure_rates=failure_rates,
                observed_patterns=patterns,
                actionable_recommendations=recommendations,
                time_window_start=time_window_start,
                time_window_end=time_window_end,
            )

            self._summaries[analysis_id] = summary

            if self.storage is not None and hasattr(self.storage, "save_feedback_analysis"):
                self.storage.save_feedback_analysis(summary.to_dict())

            logger.info(
                f"[FeedbackAnalysisEngine] Generated analysis {analysis_id}: {total_revs} reviews, {total_feedback} feedback items analyzed."
            )
            return summary

    def create_proposal(
        self,
        proposed_change: str,
        target_policy: str,
        rationale: str,
        analysis_id: Optional[str] = None,
    ) -> PolicyChangeProposal:
        """Create an advisory policy-change proposal (requires human authorization to apply)."""
        with self._lock:
            proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
            proposal = PolicyChangeProposal(
                proposal_id=proposal_id,
                analysis_id=analysis_id,
                proposed_change=proposed_change,
                target_policy=target_policy,
                rationale=rationale,
                status="PROPOSED",
            )
            self._proposals[proposal_id] = proposal

            if self.storage is not None and hasattr(self.storage, "save_policy_proposal"):
                self.storage.save_policy_proposal(proposal.to_dict())

            if self.audit_trail is not None and hasattr(self.audit_trail, "record_event"):
                self.audit_trail.record_event(
                    task_id=analysis_id or "system",
                    event_type=AuditEventType.POLICY_PROPOSAL,
                    description=f"New policy change proposed: {proposed_change[:60]}",
                    agent_name="FeedbackAnalysisEngine",
                    payload=proposal.to_dict(),
                )

            logger.info(f"[FeedbackAnalysisEngine] Created policy proposal {proposal_id}: {proposed_change[:60]}")
            return proposal

    def apply_proposal(
        self,
        proposal_id: str,
        authorized_by: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[PolicyChangeProposal]]:
        """Apply a proposed policy change, strictly enforcing human authorization."""
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if not prop and self.storage is not None and hasattr(self.storage, "get_policy_proposal"):
                data = self.storage.get_policy_proposal(proposal_id)
                if data:
                    prop = PolicyChangeProposal.from_dict(data)
                    self._proposals[proposal_id] = prop

            if not prop:
                return False, f"Proposal '{proposal_id}' not found.", None

            # Enforce human authorization check
            if not authorized_by or not str(authorized_by).strip():
                err = "Unauthorized: Human authorization signature is strictly required before applying policy changes."
                logger.warning(f"[FeedbackAnalysisEngine] {err}")
                return False, err, prop

            clean_auth = str(authorized_by).strip()
            prop.status = "APPLIED"
            prop.authorized_by = clean_auth
            prop.authorized_at = datetime.now(timezone.utc).isoformat()

            if self.storage is not None and hasattr(self.storage, "update_policy_proposal"):
                self.storage.update_policy_proposal(
                    proposal_id=proposal_id,
                    status="APPLIED",
                    authorized_by=clean_auth,
                    authorized_at=prop.authorized_at,
                )

            if self.audit_trail is not None and hasattr(self.audit_trail, "record_event"):
                self.audit_trail.record_event(
                    task_id=prop.analysis_id or "system",
                    event_type=AuditEventType.POLICY_APPLIED,
                    description=f"Policy change '{proposal_id}' applied with human authorization from '{clean_auth}'",
                    agent_name=clean_auth,
                    payload=prop.to_dict(),
                )

            logger.info(f"[FeedbackAnalysisEngine] Policy proposal {proposal_id} APPLIED by '{clean_auth}'")
            return True, "Policy proposal successfully authorized and applied.", prop

    def list_proposals(self, status: Optional[str] = None) -> List[PolicyChangeProposal]:
        """List policy proposals with optional status filter."""
        with self._lock:
            props = list(self._proposals.values())
            if status:
                props = [p for p in props if p.status == status]
            return props


# ===========================================================================
# 11. Part 6: Unified Production Human-in-the-Loop Engine
# ===========================================================================

class HumanInTheLoopEngine:
    """Unified Production Engine coordinating all 7 HITL Subsystems (Day 227 Parts 1–6):

    Subsystems:
    1. interactions: HumanInteractionManager (Lifecycle, requests, history)
    2. approvals: HumanApprovalManager (Decision gates, action-bound tokens, execution boundary)
    3. feedback: HumanFeedbackManager (Clarification, pausing, resuming without repetition)
    4. risk: RiskAssessmentEngine (Part 4: Configurable risk classification, policies, reassessment)
    5. reviews: HumanReviewManager (Part 5: Multi-round reviews of findings, plans, decisions, executions)
    6. audit: AuditTrail (Part 5: Append-only persistent event trail with secret redaction)
    7. intelligence: FeedbackAnalysisEngine (Part 5: Advisory pattern discovery & policy proposals)
    """

    def __init__(
        self,
        storage: Optional[SQLiteMessageStorage] = None,
        router: Optional[MessageRouter] = None,
        default_ttl_seconds: Optional[float] = None,
        risk_policy: Optional[RiskPolicyConfig] = None,
        authorized_reviewers: Optional[Set[str]] = None,
        observability: Optional[Any] = None,
    ):
        self.storage = storage
        self.router = router
        self.default_ttl_seconds = default_ttl_seconds
        self.observability = observability

        # 1. Audit Trail (initialized first so other subsystems can record events)
        self.audit = AuditTrail(storage=storage)

        # 2. Part 1: Interaction Manager
        self.interactions = HumanInteractionManager(storage=storage, router=router, observability=observability)

        # 3. Part 2: Approval Manager
        self.approvals = HumanApprovalManager(
            interaction_manager=self.interactions,
            storage=storage,
            default_ttl_seconds=default_ttl_seconds,
            observability=observability,
        )

        # 4. Part 3: Feedback Manager
        self.feedback = HumanFeedbackManager(
            interaction_manager=self.interactions,
            storage=storage,
            observability=observability,
        )

        # 5. Part 4: Risk Assessment Engine
        self.risk = RiskAssessmentEngine(
            policy=risk_policy,
            storage=storage,
            audit_trail=self.audit,
        )

        # 6. Part 5: Human Review Manager
        self.reviews = HumanReviewManager(
            storage=storage,
            audit_trail=self.audit,
            authorized_reviewers=authorized_reviewers,
            observability=observability,
        )

        # 7. Part 5: Feedback Analysis Engine
        self.intelligence = FeedbackAnalysisEngine(
            review_manager=self.reviews,
            interaction_manager=self.interactions,
            storage=storage,
            audit_trail=self.audit,
        )

    def set_observability(self, observability: Any) -> None:
        """Bind or update active observability manager across all child subsystems."""
        self.observability = observability
        self.interactions.observability = observability
        self.approvals.observability = observability
        self.feedback.observability = observability
        self.reviews.observability = observability

        # 7. Part 5: Feedback Analysis Engine
        self.intelligence = FeedbackAnalysisEngine(
            review_manager=self.reviews,
            interaction_manager=self.interactions,
            storage=storage,
            audit_trail=self.audit,
        )

    # -----------------------------------------------------------------------
    # Clean Python Interface Helpers
    # -----------------------------------------------------------------------

    def assess_action(
        self,
        task_id: str,
        action: Union[str, Dict[str, Any]],
        context: Optional[TaskContext] = None,
        explicit_requires_approval: bool = False,
    ) -> RiskAssessmentResult:
        """Evaluate an action's risk profile and required human controls."""
        return self.risk.evaluate_action(
            task_id=task_id,
            action=action,
            context=context,
            explicit_requires_approval=explicit_requires_approval,
        )

    def submit_review(
        self,
        task_id: str,
        target_type: str,
        target_data: Dict[str, Any],
        reviewer: str,
        outcome: Union[ReviewOutcome, str],
        feedback: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[TaskContext] = None,
    ) -> HumanReviewRecord:
        """Submit a formal review for a workflow artifact."""
        return self.reviews.submit_review(
            task_id=task_id,
            target_type=target_type,
            target_data=target_data,
            reviewer=reviewer,
            outcome=outcome,
            feedback=feedback,
            correlation_id=correlation_id,
            metadata=metadata,
            context=context,
        )

    def query_audit_trail(
        self,
        task_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_type: Optional[Union[AuditEventType, str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query the audit trail with contextual filters."""
        return self.audit.query(
            task_id=task_id,
            agent_name=agent_name,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    def analyze_feedback(
        self,
        time_window_start: Optional[str] = None,
        time_window_end: Optional[str] = None,
        title: Optional[str] = None,
    ) -> FeedbackSummary:
        """Generate structured advisory feedback summary report."""
        return self.intelligence.generate_summary(
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            title=title,
        )

    def propose_policy_change(
        self,
        proposed_change: str,
        target_policy: str,
        rationale: str,
        analysis_id: Optional[str] = None,
    ) -> PolicyChangeProposal:
        """Create an advisory policy proposal."""
        return self.intelligence.create_proposal(
            proposed_change=proposed_change,
            target_policy=target_policy,
            rationale=rationale,
            analysis_id=analysis_id,
        )

    def apply_policy_change(
        self,
        proposal_id: str,
        authorized_by: str,
    ) -> Tuple[bool, str, Optional[PolicyChangeProposal]]:
        """Authorize and apply a policy change proposal."""
        return self.intelligence.apply_proposal(
            proposal_id=proposal_id,
            authorized_by=authorized_by,
        )
