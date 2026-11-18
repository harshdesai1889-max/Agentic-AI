"""
core/failure_recovery.py

Failure Recovery and Reassignment Subsystem for Day 225 Part 5.

Provides:
1. Structured Failure Classification (TRANSIENT, VALIDATION, EXECUTION, AUTHORIZATION, PERMANENT)
2. Controlled Retry with configurable limits, backoff, and attempt audit history
3. Worker Reassignment via capability-based WorkerRegistry
4. State & Idempotency Preservation preventing duplicate side-effects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from utils.logging_config import get_logger

logger = get_logger()


# ===========================================================================
# 1. Failure Classification Enum & Details
# ===========================================================================

class FailureType(str, Enum):
    """Classification of failure modes for targeted recovery strategies."""
    TRANSIENT = "transient"          # Temporary network, timeout, or rate-limit failures
    VALIDATION = "validation"        # Agent output schema or contract mismatch
    EXECUTION = "execution"          # Operational tool execution or runtime errors
    AUTHORIZATION = "authorization"  # Missing permissions, sensitive boundary blocks, approval rejection
    PERMANENT = "permanent"          # Fatal, non-recoverable, or unresolvable logic errors


@dataclass
class FailureDetails:
    """Structured failure representation containing audit and classification data."""
    failure_type: FailureType
    message: str
    stage: str
    agent_name: str
    task_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_retryable: bool = False
    raw_error: Optional[Any] = None
    error_code: Optional[Union[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert failure details to a serializable dictionary."""
        return {
            "failure_type": self.failure_type.value,
            "message": self.message,
            "stage": self.stage,
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "is_retryable": self.is_retryable,
            "error_code": self.error_code,
            "metadata": self.metadata,
        }


# ===========================================================================
# 2. Attempt & Reassignment Audit Records
# ===========================================================================

@dataclass
class AttemptRecord:
    """Audit record for a single execution or retry attempt."""
    task_id: str
    stage: str
    agent_name: str
    attempt_number: int
    status: str  # "success" | "failed"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    failure_type: Optional[FailureType] = None
    error_message: Optional[str] = None
    backoff_delay_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert attempt record to a serializable dictionary."""
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "agent_name": self.agent_name,
            "attempt_number": self.attempt_number,
            "status": self.status,
            "timestamp": self.timestamp,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "error_message": self.error_message,
            "backoff_delay_seconds": self.backoff_delay_seconds,
            "metadata": self.metadata,
        }


@dataclass
class ReassignmentRecord:
    """Audit record capturing agent reassignment from an original worker to a replacement."""
    task_id: str
    stage: str
    original_worker: str
    replacement_worker: Optional[str]
    reason: str
    status: str  # "reassigned" | "failed_no_worker"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert reassignment record to a serializable dictionary."""
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "original_worker": self.original_worker,
            "replacement_worker": self.replacement_worker,
            "reason": self.reason,
            "status": self.status,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryHistory:
    """Container tracking complete failure recovery, retry, and reassignment lifecycle."""
    task_id: str
    attempts: List[AttemptRecord] = field(default_factory=list)
    reassignments: List[ReassignmentRecord] = field(default_factory=list)
    total_retries: int = 0
    recovered: bool = False
    final_status: str = "initialized"

    def add_attempt(self, record: AttemptRecord) -> None:
        """Add an attempt record and increment total_retries if applicable."""
        self.attempts.append(record)
        if record.attempt_number > 1:
            self.total_retries += 1

    def add_reassignment(self, record: ReassignmentRecord) -> None:
        """Add a reassignment record."""
        self.reassignments.append(record)

    def to_dict(self) -> Dict[str, Any]:
        """Convert recovery history to a serializable dictionary."""
        return {
            "task_id": self.task_id,
            "total_attempts": len(self.attempts),
            "total_retries": self.total_retries,
            "recovered": self.recovered,
            "final_status": self.final_status,
            "attempts": [a.to_dict() for a in self.attempts],
            "reassignments": [r.to_dict() for r in self.reassignments],
        }


# ===========================================================================
# 3. Failure Classifier
# ===========================================================================

class FailureClassifier:
    """Classifies runtime errors, exceptions, and contract rejections into structured failure types."""

    TRANSIENT_EXCEPTIONS = (
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        ConnectionRefusedError,
    )

    TRANSIENT_PATTERNS = [
        r"\b(timeout|timed\s*out)\b",
        r"\b(rate\s*limit|too\s+many\s+requests|429)\b",
        r"\b(service\s*unavailable|503|502|504|bad\s*gateway|gateway\s*timeout)\b",
        r"\b(connection\s*(reset|refused|error|lost|aborted))\b",
        r"\b(high\s+demand|spikes\s+in\s+demand|overloaded)\b",
        r"\b(temporary|transient)\b",
    ]

    AUTHORIZATION_PATTERNS = [
        r"\b(unauthorized|authorization|forbidden|401|403)\b",
        r"\b(permission\s*denied|missing\s*permission)\b",
        r"\b(approval\s*checkpoint|approval\s*(rejected|denied|was\s*not\s*approved))\b",
        r"\b(sensitive\s*operation\s*blocked|requires\s*explicit\s*authorization)\b",
        r"\b(unauthorized\s*sensitive\s*operation)\b",
    ]

    VALIDATION_PATTERNS = [
        r"\b(validation\s*(error|failed))\b",
        r"\b(malformed|invalid\s*(output|format|schema|json|plan))\b",
        r"\b(missing\s*required\s*field|tasks\s*list\s*must\s*be)\b",
        r"\b(task\s*id\s*mismatch)\b",
        r"\b(plan\s*must\s*be\s*a\s*json\s*object)\b",
    ]

    PERMANENT_PATTERNS = [
        r"\b(404|not\s*found|model\s*is\s*no\s*longer\s*available)\b",
        r"\b(invalid\s*argument|unsupported\s*model|unsupported\s*action)\b",
        r"\b(permanent|unrecoverable|fatal)\b",
    ]

    @classmethod
    def classify(
        cls,
        error: Any,
        stage: str = "execution",
        agent_name: str = "Agent",
        task_id: str = "",
        explicit_type: Optional[FailureType] = None,
    ) -> FailureDetails:
        """Classify an error into a structured FailureDetails object."""
        if explicit_type is not None:
            is_retryable = explicit_type in (
                FailureType.TRANSIENT,
                FailureType.VALIDATION,
            )
            return FailureDetails(
                failure_type=explicit_type,
                message=str(error),
                stage=stage,
                agent_name=agent_name,
                task_id=task_id,
                is_retryable=is_retryable,
                raw_error=error,
            )

        # 1. Structured dictionary or object check
        if isinstance(error, dict):
            if "failure_type" in error:
                try:
                    ft = FailureType(str(error["failure_type"]))
                    is_retryable = ft in (
                        FailureType.TRANSIENT,
                        FailureType.VALIDATION,
                    )
                    return FailureDetails(
                        failure_type=ft,
                        message=error.get("message") or error.get("error") or str(error),
                        stage=stage,
                        agent_name=agent_name,
                        task_id=task_id,
                        is_retryable=is_retryable,
                        error_code=error.get("error_code") or error.get("code"),
                        raw_error=error,
                    )
                except ValueError:
                    pass

            code = error.get("code") or error.get("status_code")
            if code in (429, 500, 502, 503, 504) or str(code) in ("429", "500", "502", "503", "504"):
                return FailureDetails(
                    failure_type=FailureType.TRANSIENT,
                    message=error.get("message") or error.get("error") or str(error),
                    stage=stage,
                    agent_name=agent_name,
                    task_id=task_id,
                    is_retryable=True,
                    error_code=code,
                    raw_error=error,
                )
            if code in (401, 403) or str(code) in ("401", "403"):
                return FailureDetails(
                    failure_type=FailureType.AUTHORIZATION,
                    message=error.get("message") or error.get("error") or str(error),
                    stage=stage,
                    agent_name=agent_name,
                    task_id=task_id,
                    is_retryable=False,
                    error_code=code,
                    raw_error=error,
                )
            if code == 404 or str(code) == "404":
                return FailureDetails(
                    failure_type=FailureType.PERMANENT,
                    message=error.get("message") or error.get("error") or str(error),
                    stage=stage,
                    agent_name=agent_name,
                    task_id=task_id,
                    is_retryable=False,
                    error_code=code,
                    raw_error=error,
                )

        # 2. Python Exception Instances
        if isinstance(error, cls.TRANSIENT_EXCEPTIONS):
            return FailureDetails(
                failure_type=FailureType.TRANSIENT,
                message=str(error),
                stage=stage,
                agent_name=agent_name,
                task_id=task_id,
                is_retryable=True,
                raw_error=error,
            )

        if isinstance(error, PermissionError):
            return FailureDetails(
                failure_type=FailureType.AUTHORIZATION,
                message=str(error),
                stage=stage,
                agent_name=agent_name,
                task_id=task_id,
                is_retryable=False,
                raw_error=error,
            )

        err_str = str(error).strip()

        # 3. Authorization patterns (Priority: never retry unauthorized operations)
        for pat in cls.AUTHORIZATION_PATTERNS:
            if re.search(pat, err_str, re.IGNORECASE):
                return FailureDetails(
                    failure_type=FailureType.AUTHORIZATION,
                    message=err_str,
                    stage=stage,
                    agent_name=agent_name,
                    task_id=task_id,
                    is_retryable=False,
                    raw_error=error,
                )

        # 4. Permanent patterns (Explicit no-retry)
        for pat in cls.PERMANENT_PATTERNS:
            if re.search(pat, err_str, re.IGNORECASE):
                return FailureDetails(
                    failure_type=FailureType.PERMANENT,
                    message=err_str,
                    stage=stage,
                    agent_name=agent_name,
                    task_id=task_id,
                    is_retryable=False,
                    raw_error=error,
                )

        # 5. Transient patterns
        for pat in cls.TRANSIENT_PATTERNS:
            if re.search(pat, err_str, re.IGNORECASE):
                return FailureDetails(
                    failure_type=FailureType.TRANSIENT,
                    message=err_str,
                    stage=stage,
                    agent_name=agent_name,
                    task_id=task_id,
                    is_retryable=True,
                    raw_error=error,
                )

        # 6. Validation patterns
        for pat in cls.VALIDATION_PATTERNS:
            if re.search(pat, err_str, re.IGNORECASE):
                return FailureDetails(
                    failure_type=FailureType.VALIDATION,
                    message=err_str,
                    stage=stage,
                    agent_name=agent_name,
                    task_id=task_id,
                    is_retryable=True,
                    raw_error=error,
                )

        # 7. Default to Execution failure (not retryable automatically)
        return FailureDetails(
            failure_type=FailureType.EXECUTION,
            message=err_str,
            stage=stage,
            agent_name=agent_name,
            task_id=task_id,
            is_retryable=False,
            raw_error=error,
        )


# ===========================================================================
# 4. Worker Registry
# ===========================================================================

class WorkerRegistry:
    """Central registry of specialized workers and their verified capabilities."""

    def __init__(self):
        self._workers: Dict[str, Any] = {}
        self._capabilities: Dict[str, Set[str]] = {}
        self._priorities: Dict[str, int] = {}

    def register(
        self,
        worker: Any,
        capabilities: List[str],
        priority: int = 10,
        name: Optional[str] = None,
    ) -> None:
        """Register a worker with its associated capabilities and selection priority."""
        worker_name = name or getattr(worker, "name", worker.__class__.__name__)
        self._workers[worker_name] = worker
        self._capabilities[worker_name] = set(c.lower() for c in capabilities)
        self._priorities[worker_name] = priority
        logger.info(
            f"[WorkerRegistry] Registered worker '{worker_name}' with capabilities: {capabilities} (priority={priority})"
        )

    def get_worker(self, name: str) -> Optional[Any]:
        """Retrieve a registered worker by name."""
        return self._workers.get(name)

    def has_worker(self, name: str) -> bool:
        """Check whether a worker with the given name is registered."""
        return name in self._workers

    def get_workers_for_capability(self, capability: str) -> List[Tuple[str, Any, int]]:
        """Find all registered workers possessing a specific capability, sorted by priority."""
        target = capability.lower()
        matches = []
        for name, worker in self._workers.items():
            caps = self._capabilities.get(name, set())
            if target in caps or any(target in c for c in caps):
                matches.append((name, worker, self._priorities.get(name, 10)))
        # Sort by priority descending, then name ascending
        matches.sort(key=lambda x: (-x[2], x[0]))
        return matches

    def get_best_worker_for_capability(
        self,
        capability: str,
        exclude_names: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Find the highest-priority worker matching a capability."""
        excludes = set(exclude_names or [])
        matches = [m for m in self.get_workers_for_capability(capability) if m[0] not in excludes]
        if matches:
            return matches[0][0], matches[0][1]
        return None

    def is_capable(self, worker: Any, capability_or_stage: str) -> bool:
        """Check whether a worker has the requested capability or stage qualification."""
        worker_name = getattr(worker, "name", worker.__class__.__name__)
        caps = self._capabilities.get(worker_name, set())
        target = capability_or_stage.lower()
        return target in caps or any(target in c for c in caps)

    def get_replacement_worker(
        self,
        current_worker: Any,
        stage: str,
        exclude_names: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Find an alternative registered worker capable of fulfilling the specified stage."""
        curr_name = (
            getattr(current_worker, "name", current_worker.__class__.__name__)
            if current_worker
            else ""
        )
        excludes = set(exclude_names or [])
        if curr_name:
            excludes.add(curr_name)

        target_stage = stage.lower()

        eligible: List[Tuple[int, str, Any]] = []
        for name, worker in self._workers.items():
            if name in excludes:
                continue
            caps = self._capabilities.get(name, set())
            if target_stage in caps or any(target_stage in c for c in caps):
                priority = self._priorities.get(name, 10)
                eligible.append((priority, name, worker))

        if not eligible:
            logger.warning(
                f"[WorkerRegistry] No capable replacement worker found for stage '{stage}' (excluded: {excludes})"
            )
            return None

        # Sort by priority descending (higher number = higher priority), then by name alphabetically
        eligible.sort(key=lambda x: (-x[0], x[1]))
        chosen_worker = eligible[0][2]
        logger.info(
            f"[WorkerRegistry] Reassigning stage '{stage}' from '{curr_name}' to replacement '{eligible[0][1]}'"
        )
        return chosen_worker

    def list_workers(self) -> List[Dict[str, Any]]:
        """List all registered workers and their metadata."""
        return [
            {
                "name": name,
                "capabilities": sorted(list(self._capabilities[name])),
                "priority": self._priorities[name],
            }
            for name in self._workers
        ]


# ===========================================================================
# 5. Idempotency Helper
# ===========================================================================

# Sensitive action patterns that must never be repeated automatically without authorization
NON_IDEMPOTENT_PATTERNS = [
    r"\b(send\s+email|send\s+message|send\s+notification|post\s+tweet)\b",
    r"\b(transfer\s+funds?|process\s+payment|billing|charge\s+credit\s+card)\b",
    r"\b(delete|drop|truncate|remove\s+database|destroy\s+table|purge|wipe)\b",
    r"\b(modify\s+production|deploy\s+to\s+prod|apply\s+migration)\b",
]


def is_non_idempotent_operation(task_description: str) -> bool:
    """Detect whether a task description involves external side-effects that are unsafe to retry."""
    if not task_description or not isinstance(task_description, str):
        return False
    return any(re.search(pat, task_description, re.IGNORECASE) for pat in NON_IDEMPOTENT_PATTERNS)
