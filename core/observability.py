"""
core/observability.py

Comprehensive Agent Observability & Tracing Framework for Day 229 (Parts 1 to 6).

Implements:
1. Part 1: Agent Observability Fundamentals
   - SpanStatus and SpanKind enums.
   - TraceContext, SpanRecord, TraceRecord, AgentExecutionEvent, ObservabilityConfig.
   - Centralized, non-interfering trace and span lifecycle management.
2. Part 2: Distributed Agent Tracing
   - Context propagation across async message queues, worker handoffs, and provider calls.
   - Hierarchical parent-child span tracking.
   - Trace querying by trace_id, workflow_id, task_id, and agent_name.
   - Missing-context resilience without crashing active workflows.
3. Part 3: Structured Logging and Event Tracking
   - StructuredLogRecord with standard log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).
   - Standard event categories across agent, LLM, tool, message, HITL, and evaluation events.
   - Deep recursive secret redaction (API keys, passwords, JWTs, Bearer tokens, .env contents).
   - Fault-isolated logging execution.
4. Part 4: Metrics Collection and Performance Monitoring
   - AgentMetricsCollector tracking agent latency (avg, P50, P95, P99), success/failure rates, retries.
   - LLM metrics (provider, model, token counts, latency). Honest missing-token handling without fabrication.
   - Tool metrics (invocation counts, duration, success rates).
   - Workflow metrics and time-window filtering.
5. Part 5: Failure Diagnosis and Trace Analysis
   - FailureCategory classification attributing agent, tool, provider, network, HITL, or evaluation failures.
   - Bottleneck detection identifying slowest operations, excessive retries, and high-latency stages.
   - Deterministic root-cause analysis correlating with Day 228 evaluations and HITL events.
6. Part 6: Unified Observability Framework
   - AgentObservabilityManager coordinating tracing, logging, metrics, failure analysis, and SQLite persistence.
   - Configurable retention cleanup policy.
   - Instrumented provider wrapper and workflow hooks.
"""

from __future__ import annotations

import copy
import contextvars
import json
import math
import re
import threading
import time
import uuid
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

from core.persistent_storage import SQLiteMessageStorage
from core.provider_interface import LLMMessage, LLMProvider, LLMResponse
from utils.logging_config import get_logger

logger = get_logger()

# ContextVar maintaining active trace context across async tasks and coroutines
_current_trace_context: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar(
    "_current_trace_context", default=None
)

# Global active observability manager instance for cross-cutting instrumentation
_active_observability_manager: Optional[Any] = None


def get_active_observability_manager() -> Optional[Any]:
    """Return the currently registered active AgentObservabilityManager, if any."""
    return _active_observability_manager


def set_active_observability_manager(manager: Optional[Any]) -> None:
    """Set the globally accessible AgentObservabilityManager instance."""
    global _active_observability_manager
    _active_observability_manager = manager


# ===========================================================================
# 1. PART 1: AGENT OBSERVABILITY FUNDAMENTALS & DATA CONTRACTS
# ===========================================================================

class SpanStatus(str, Enum):
    """Execution status of an observability span or trace."""
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


class SpanKind(str, Enum):
    """Semantic category of an execution span."""
    WORKFLOW = "workflow"
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    MESSAGE = "message"
    EVALUATION = "evaluation"
    APPROVAL = "approval"
    RECOVERY = "recovery"


@dataclass
class TraceContext:
    """Propagatable context identifying distributed trace lineage."""
    trace_id: str
    workflow_id: str
    task_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    current_span_id: Optional[str] = None
    sampled: bool = True
    baggage: Dict[str, str] = field(default_factory=dict)

    def create_child_context(self, span_id: str, new_task_id: Optional[str] = None) -> "TraceContext":
        """Generate a child context setting parent_span_id to the current span."""
        return TraceContext(
            trace_id=self.trace_id,
            workflow_id=self.workflow_id,
            task_id=new_task_id or self.task_id,
            parent_span_id=self.current_span_id,
            current_span_id=span_id,
            sampled=self.sampled,
            baggage=dict(self.baggage),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization over messages/queues."""
        return {
            "trace_id": self.trace_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "parent_span_id": self.parent_span_id,
            "current_span_id": self.current_span_id,
            "sampled": self.sampled,
            "baggage": dict(self.baggage),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TraceContext":
        """Reconstruct TraceContext from a dictionary safely."""
        if not data or not isinstance(data, dict):
            # Fall back to detached trace
            tid = f"tr_{uuid.uuid4().hex[:12]}"
            wid = f"wf_{uuid.uuid4().hex[:8]}"
            return cls(trace_id=tid, workflow_id=wid)

        return cls(
            trace_id=str(data.get("trace_id") or f"tr_{uuid.uuid4().hex[:12]}"),
            workflow_id=str(data.get("workflow_id") or f"wf_{uuid.uuid4().hex[:8]}"),
            task_id=data.get("task_id"),
            parent_span_id=data.get("parent_span_id"),
            current_span_id=data.get("current_span_id"),
            sampled=bool(data.get("sampled", True)),
            baggage=dict(data.get("baggage", {})),
        )

    def inject(self, carrier: Dict[str, Any]) -> None:
        """Inject trace context into a carrier dictionary (e.g. message.metadata)."""
        if isinstance(carrier, dict):
            carrier["trace_context"] = self.to_dict()
            carrier["trace_id"] = self.trace_id

    @classmethod
    def extract(cls, carrier: Optional[Dict[str, Any]]) -> Optional["TraceContext"]:
        """Extract trace context from a carrier dictionary."""
        if not carrier or not isinstance(carrier, dict):
            return None


        data = carrier.get("trace_context")
        if data and isinstance(data, dict):
            return cls.from_dict(data)
        if "trace_id" in carrier:
            return cls(
                trace_id=carrier["trace_id"],
                workflow_id=carrier.get("workflow_id", carrier.get("task_id", f"wf_{carrier['trace_id']}")),
                task_id=carrier.get("task_id"),
                parent_span_id=carrier.get("parent_span_id"),
                current_span_id=carrier.get("span_id"),
            )
        return None


def inject_context(context: Optional[TraceContext], carrier: Dict[str, Any]) -> None:
    """Module-level helper to inject TraceContext into carrier."""
    if context is not None:
        context.inject(carrier)


def extract_context(carrier: Optional[Dict[str, Any]]) -> Optional[TraceContext]:
    """Module-level helper to extract TraceContext from carrier."""
    return TraceContext.extract(carrier)


@dataclass
class SpanRecord:
    """Individual unit of instrumented work within a distributed trace."""
    span_id: str
    trace_id: str
    workflow_id: str
    agent_name: str
    operation_name: str
    parent_span_id: Optional[str] = None
    task_id: Optional[str] = None
    span_kind: SpanKind = SpanKind.AGENT
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.STARTED
    attributes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(
        self,
        attributes: Optional[Dict[str, Any]] = None,
        status: SpanStatus = SpanStatus.COMPLETED,
    ) -> None:
        """Mark the span successfully completed and compute duration."""
        self.end_time = datetime.now(timezone.utc).isoformat()
        try:
            t0 = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(self.end_time.replace("Z", "+00:00"))
            self.duration_ms = round((t1 - t0).total_seconds() * 1000.0, 2)
        except Exception:
            self.duration_ms = 0.0
        self.status = status
        if attributes:
            self.attributes.update(attributes)

    def fail(
        self,
        error: Union[Exception, str, Dict[str, Any]],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark the span failed, capture error diagnostic info, and compute duration."""
        self.end_time = datetime.now(timezone.utc).isoformat()
        try:
            t0 = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(self.end_time.replace("Z", "+00:00"))
            self.duration_ms = round((t1 - t0).total_seconds() * 1000.0, 2)
        except Exception:
            self.duration_ms = 0.0
        self.status = SpanStatus.FAILED

        if isinstance(error, Exception):
            self.error = {"type": type(error).__name__, "message": str(error)}
        elif isinstance(error, str):
            self.error = {"type": "ExecutionError", "message": error}
        elif isinstance(error, dict):
            self.error = dict(error)
        else:
            self.error = {"type": "UnknownError", "message": str(error)}

        if attributes:
            self.attributes.update(attributes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert span record to a serializable dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "operation_name": self.operation_name,
            "span_kind": self.span_kind.value if isinstance(self.span_kind, SpanKind) else str(self.span_kind),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value if isinstance(self.status, SpanStatus) else str(self.status),
            "attributes": self.attributes,
            "error": self.error or {},
            "metadata": self.metadata,
        }

    def __getitem__(self, key: str) -> Any:
        val = getattr(self, key)
        if isinstance(val, (SpanStatus, LogLevel, SpanKind)):
            return val.value
        return val


@dataclass
class TraceRecord:
    """Root trace representing an entire multi-agent workflow invocation."""
    trace_id: str
    workflow_id: str
    task_id: Optional[str] = None
    root_span_id: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.STARTED
    agent_sequence: List[str] = field(default_factory=list)
    spans: List[SpanRecord] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_agent_transition(self, agent_name: str) -> None:
        """Record an agent execution in the transition sequence."""
        norm_name = agent_name.strip()
        if not self.agent_sequence or self.agent_sequence[-1] != norm_name:
            self.agent_sequence.append(norm_name)

    def complete(self, status: SpanStatus = SpanStatus.COMPLETED) -> None:
        """Mark trace completed and calculate total end-to-end duration."""
        self.end_time = datetime.now(timezone.utc).isoformat()
        try:
            t0 = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(self.end_time.replace("Z", "+00:00"))
            self.duration_ms = round((t1 - t0).total_seconds() * 1000.0, 2)
        except Exception:
            self.duration_ms = 0.0
        self.status = status

    def fail(self, error: Union[Exception, str, Dict[str, Any]]) -> None:
        """Mark trace failed with diagnostic error details."""
        self.end_time = datetime.now(timezone.utc).isoformat()
        try:
            t0 = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(self.end_time.replace("Z", "+00:00"))
            self.duration_ms = round((t1 - t0).total_seconds() * 1000.0, 2)
        except Exception:
            self.duration_ms = 0.0
        self.status = SpanStatus.FAILED

        if isinstance(error, Exception):
            self.error = {"type": type(error).__name__, "message": str(error)}
        elif isinstance(error, str):
            self.error = {"type": "WorkflowError", "message": error}
        elif isinstance(error, dict):
            self.error = dict(error)
        else:
            self.error = {"type": "UnknownError", "message": str(error)}

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace record to a serializable dictionary."""
        return {
            "trace_id": self.trace_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "root_span_id": self.root_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value if isinstance(self.status, SpanStatus) else str(self.status),
            "agent_sequence": self.agent_sequence,
            "error": self.error or {},
            "attributes": self.attributes,
            "metadata": self.metadata,
            "span_count": len(self.spans),
        }

    def __getitem__(self, key: str) -> Any:
        val = getattr(self, key)
        if isinstance(val, (SpanStatus, LogLevel, SpanKind)):
            return val.value
        return val


@dataclass
class AgentExecutionEvent:
    """Discrete event emitted during an agent's execution lifecycle."""
    event_id: str
    trace_id: str
    span_id: Optional[str]
    event_type: str
    agent_name: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "event_type": self.event_type,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


@dataclass
class ObservabilityConfig:
    """Settings controlling observability sampling, retention, and collection."""
    enabled: bool = True
    sampling_rate: float = 1.0  # 1.0 = sample all traces
    collect_events: bool = True
    collect_logs: bool = True
    collect_metrics: bool = True
    redact_sensitive_data: bool = True
    retention_days: int = 30
    max_spans_per_trace: int = 1000
    service_name: str = "ab_smart_ai"
    min_log_level: Any = "DEBUG"


@dataclass
class ObservabilitySummary:
    """High-level summary of observability health and metrics."""
    total_traces: int
    active_traces: int
    completed_traces: int
    failed_traces: int
    total_spans: int
    total_logs: int
    total_metrics: int
    p95_workflow_latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_traces": self.total_traces,
            "active_traces": self.active_traces,
            "completed_traces": self.completed_traces,
            "failed_traces": self.failed_traces,
            "total_spans": self.total_spans,
            "total_logs": self.total_logs,
            "total_metrics": self.total_metrics,
            "p95_workflow_latency_ms": self.p95_workflow_latency_ms,
            "timestamp": self.timestamp,
        }


# ===========================================================================
# 2. PART 3: STRUCTURED LOGGING & SECRET REDACTION
# ===========================================================================

class LogLevel(str, Enum):
    """Severity levels for structured observability logs."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventCategory(str, Enum):
    """Standardized event taxonomy across the multi-agent system."""
    WORKFLOW_STARTED = "WORKFLOW_STARTED"
    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
    WORKFLOW_FAILED = "WORKFLOW_FAILED"
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"
    LLM_REQUEST_STARTED = "LLM_REQUEST_STARTED"
    LLM_REQUEST_COMPLETED = "LLM_REQUEST_COMPLETED"
    LLM_REQUEST_FAILED = "LLM_REQUEST_FAILED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    TOOL_FAILED = "TOOL_FAILED"
    MESSAGE_SENT = "MESSAGE_SENT"
    MESSAGE_RECEIVED = "MESSAGE_RECEIVED"
    RETRY_STARTED = "RETRY_STARTED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"
    HUMAN_APPROVAL_REQUESTED = "HUMAN_APPROVAL_REQUESTED"
    HUMAN_APPROVAL_GRANTED = "HUMAN_APPROVAL_GRANTED"
    HUMAN_APPROVAL_REJECTED = "HUMAN_APPROVAL_REJECTED"
    HUMAN_APPROVAL_EXPIRED = "HUMAN_APPROVAL_EXPIRED"
    HUMAN_APPROVAL_CANCELLED = "HUMAN_APPROVAL_CANCELLED"
    HUMAN_APPROVAL_COMPLETED = "HUMAN_APPROVAL_COMPLETED"
    WORKFLOW_PAUSED = "WORKFLOW_PAUSED"
    WORKFLOW_RESUMED = "WORKFLOW_RESUMED"
    HUMAN_CLARIFICATION_REQUESTED = "HUMAN_CLARIFICATION_REQUESTED"
    HUMAN_CLARIFICATION_RECEIVED = "HUMAN_CLARIFICATION_RECEIVED"
    HUMAN_REVIEW_COMPLETED = "HUMAN_REVIEW_COMPLETED"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"


# Regular expressions scrubbing secrets, credentials, and tokens from traces and logs
OBSERVABILITY_REDACTION_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|secret[a-z0-9_-]*|token|password|auth|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{6,})['\"]?"), r"\1=[REDACTED]"),
    (re.compile(r"sk-[a-zA-Z0-9_\-]{8,}"), "[REDACTED]"),
    (re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"ey[A-Za-z0-9_-]{8,}\.ey[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "[REDACTED]"),
    (re.compile(r"(?i)(OPENROUTER_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY|OPENAI_API_KEY)\s*=\s*[^\s]+"), r"\1=[REDACTED]"),
]


def redact_observability_data(data: Any) -> Any:
    """Recursively scrub sensitive keys and regex pattern matches from observability data."""
    if data is None:
        return None



    if isinstance(data, str):
        res = data
        for pat, repl in OBSERVABILITY_REDACTION_PATTERNS:
            res = pat.sub(repl, res)
        return res

    elif isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            k_str = str(k).lower()
            if any(s in k_str for s in ("api_key", "secret", "token", "password", "auth", "credential", "private", "access_key", "bearer")) or k_str in ("key", "api-key", "apikey"):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = redact_observability_data(v)
        return cleaned

    elif isinstance(data, (list, tuple)):
        return [redact_observability_data(item) for item in data]

    return data


def instrument_tool_execution(
    tool_func: Callable,
    tool_name: str,
    tool_args: Dict[str, Any],
    observability: Optional[Any] = None,
    agent_name: str = "ExecutorAgent",
) -> Any:
    """Execute tool_func wrapped in an observability tool span."""
    obs = observability or get_active_observability_manager()
    if obs is None:
        return tool_func(**tool_args)

    span = obs.start_span(
        operation_name=f"tool_{tool_name}",
        agent_name=agent_name,
        span_kind=SpanKind.TOOL,
        attributes={"tool_name": tool_name, "tool_args": redact_observability_data(tool_args)},
    )
    start_time = time.perf_counter()
    try:
        res = tool_func(**tool_args)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        try:
            if hasattr(obs, "metrics") and hasattr(obs.metrics, "record_tool_call"):
                obs.metrics.record_tool_call(tool_name=tool_name, duration_ms=duration_ms, success=True)
            elif hasattr(obs, "record_metric"):
                obs.record_metric(
                    metric_name="tool_execution_duration_ms",
                    metric_type="histogram",
                    value=duration_ms,
                    tool_name=tool_name,
                    agent_name=agent_name,
                )
        except Exception:
            pass
        obs.end_span(span.span_id, status=SpanStatus.COMPLETED)
        return res
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        try:
            if hasattr(obs, "metrics") and hasattr(obs.metrics, "record_tool_call"):
                obs.metrics.record_tool_call(tool_name=tool_name, duration_ms=duration_ms, success=False, error_type=type(exc).__name__)
        except Exception:
            pass
        obs.end_span(span.span_id, status=SpanStatus.FAILED, error=exc)
        raise


@dataclass
class StructuredLogRecord:
    """Structured, queryable log entry correlated with distributed traces."""
    log_id: str
    timestamp: str
    log_level: LogLevel
    event_type: str
    trace_id: Optional[str]
    span_id: Optional[str]
    workflow_id: Optional[str]
    task_id: Optional[str]
    agent_name: Optional[str]
    operation_name: Optional[str]
    message: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "log_level": self.log_level.value if isinstance(self.log_level, LogLevel) else str(self.log_level),
            "event_type": self.event_type,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "operation_name": self.operation_name,
            "message": self.message,
            "attributes": self.attributes,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    def __getitem__(self, key: str) -> Any:
        val = getattr(self, key)
        if isinstance(val, (SpanStatus, LogLevel, SpanKind)):
            return val.value
        return val


# ===========================================================================
# 3. PART 4: METRICS COLLECTION & PERFORMANCE MONITORING
# ===========================================================================

def calculate_percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Calculate nearest-rank percentile (e.g. 50.0, 95.0, 99.0) with empty safety."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = max(0, min(n - 1, math.ceil((percentile / 100.0) * n) - 1))
    return round(float(sorted_values[k]), 2)


class AgentMetricsCollector:
    """Aggregates and tracks operational metrics across agents, LLMs, tools, and workflows."""

    def __init__(self, storage: Optional[SQLiteMessageStorage] = None):
        self.storage = storage
        self._lock = threading.RLock()
        self._in_memory_metrics: List[Dict[str, Any]] = []

    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = "gauge",
        agent_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tool_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an arbitrary numeric metric."""
        metric_data = {
            "metric_id": f"m_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric_name": metric_name,
            "metric_type": metric_type,
            "value": float(value),
            "agent_name": agent_name.strip().lower() if agent_name else None,
            "workflow_id": workflow_id,
            "task_id": task_id,
            "provider": provider,
            "model": model,
            "tool_name": tool_name,
            "attributes": dict(attributes or {}),
        }

        with self._lock:
            self._in_memory_metrics.append(metric_data)

        if self.storage:
            try:
                self.storage.save_observability_metric(metric_data)
            except Exception as exc:
                logger.warning(f"[MetricsCollector] Could not persist metric: {exc}")

        return metric_data

    def record_agent_execution(
        self,
        agent_name: str,
        duration_ms: float,
        success: bool,
        partial_success: bool = False,
        retry_count: int = 0,
        recovered: bool = False,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> None:
        """Record an agent execution event for latency and reliability monitoring."""
        self.record_metric(
            metric_name="agent_execution_latency_ms",
            value=duration_ms,
            metric_type="latency",
            agent_name=agent_name,
            workflow_id=workflow_id,
            task_id=task_id,
            attributes={
                "success": bool(success),
                "partial_success": bool(partial_success),
                "retry_count": int(retry_count),
                "recovered": bool(recovered),
                "task_type": task_type,
            },
        )

    def record_llm_call(
        self,
        provider_name: str,
        model_name: str,
        latency_ms: float,
        success: bool,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Record an LLM call with latency, tokens (without fabrication), and status."""
        attrs = {
            "success": bool(success),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
        }
        self.record_metric(
            metric_name="llm_request_latency_ms",
            value=latency_ms,
            metric_type="latency",
            provider=provider_name,
            model=model_name,
            workflow_id=workflow_id,
            task_id=task_id,
            attributes=attrs,
        )

    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> None:
        """Record tool invocation duration and status."""
        self.record_metric(
            metric_name="tool_execution_duration_ms",
            value=duration_ms,
            metric_type="latency",
            tool_name=tool_name,
            workflow_id=workflow_id,
            task_id=task_id,
            attributes={"success": bool(success), "error_type": error_type},
        )

    def record_workflow_execution(
        self,
        workflow_id: str,
        duration_ms: float,
        success: bool,
        agent_transitions: int = 0,
        human_interventions: int = 0,
        approval_rejections: int = 0,
        evaluation_failures: int = 0,
    ) -> None:
        """Record complete workflow latency and operational counters."""
        self.record_metric(
            metric_name="workflow_execution_latency_ms",
            value=duration_ms,
            metric_type="latency",
            workflow_id=workflow_id,
            attributes={
                "success": bool(success),
                "agent_transitions": int(agent_transitions),
                "human_interventions": int(human_interventions),
                "approval_rejections": int(approval_rejections),
                "evaluation_failures": int(evaluation_failures),
            },
        )

    def get_agent_metrics(
        self,
        agent_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregated latency and reliability metrics for an agent."""
        norm_agent = agent_name.strip().lower() if agent_name else None
        if self.storage:
            rows = self.storage.list_observability_metrics(
                metric_name="agent_execution_latency_ms",
                agent_name=norm_agent,
                workflow_id=workflow_id,
                start_time=start_time,
                end_time=end_time,
            )
            if task_type:
                rows = [r for r in rows if r.get("attributes", {}).get("task_type") == task_type]
        else:
            with self._lock:
                rows = [
                    m for m in self._in_memory_metrics
                    if m["metric_name"] == "agent_execution_latency_ms"
                    and (not norm_agent or m.get("agent_name") == norm_agent)
                    and (not workflow_id or m.get("workflow_id") == workflow_id)
                    and (not task_type or m.get("attributes", {}).get("task_type") == task_type)
                    and (not start_time or m.get("timestamp", "") >= start_time)
                    and (not end_time or m.get("timestamp", "") <= end_time)
                ]

        total = len(rows)
        if total == 0:
            return {
                "agent_name": agent_name or "all",
                "execution_count": 0,
                "success_count": 0,
                "partial_success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "retry_count": 0,
                "recovery_count": 0,
            }

        latencies = sorted([float(r["value"]) for r in rows])
        success_count = sum(1 for r in rows if r.get("attributes", {}).get("success") and not r.get("attributes", {}).get("partial_success"))
        partial_count = sum(1 for r in rows if r.get("attributes", {}).get("partial_success"))
        failure_count = sum(1 for r in rows if not r.get("attributes", {}).get("success") and not r.get("attributes", {}).get("partial_success"))
        retries = sum(int(r.get("attributes", {}).get("retry_count", 0)) for r in rows)
        recoveries = sum(1 for r in rows if r.get("attributes", {}).get("recovered"))

        effective_success = success_count + partial_count

        return {
            "agent_name": agent_name or "all",
            "execution_count": total,
            "success_count": success_count,
            "partial_success_count": partial_count,
            "failure_count": failure_count,
            "success_rate": round(effective_success / total, 4),
            "failure_rate": round(failure_count / total, 4),
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p50_latency_ms": calculate_percentile(latencies, 50.0),
            "p95_latency_ms": calculate_percentile(latencies, 95.0),
            "p99_latency_ms": calculate_percentile(latencies, 99.0),
            "retry_count": retries,
            "recovery_count": recoveries,
        }

    def get_llm_metrics(
        self,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregated metrics for LLM calls with honest token reporting."""
        if self.storage:
            rows = self.storage.list_observability_metrics(
                metric_name="llm_request_latency_ms",
                provider=provider_name,
                start_time=start_time,
                end_time=end_time,
            )
            if model_name:
                rows = [r for r in rows if r.get("model") == model_name]
        else:
            with self._lock:
                rows = [
                    m for m in self._in_memory_metrics
                    if m["metric_name"] == "llm_request_latency_ms"
                    and (not provider_name or m.get("provider") == provider_name)
                    and (not model_name or m.get("model") == model_name)
                ]

        total = len(rows)
        if total == 0:
            return {
                "provider": provider_name or "all",
                "model": model_name or "all",
                "request_count": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "total_input_tokens": None,
                "total_output_tokens": None,
                "total_tokens": None,
                "estimated_cost": None,
            }

        latencies = sorted([float(r["value"]) for r in rows])
        success_count = sum(1 for r in rows if r["attributes"].get("success"))

        # Aggregate tokens only where present (honest reporting without fabrication)
        input_tokens_list = [r["attributes"]["input_tokens"] for r in rows if r["attributes"].get("input_tokens") is not None]
        output_tokens_list = [r["attributes"]["output_tokens"] for r in rows if r["attributes"].get("output_tokens") is not None]
        total_tokens_list = [r["attributes"]["total_tokens"] for r in rows if r["attributes"].get("total_tokens") is not None]
        cost_list = [r["attributes"]["cost"] for r in rows if r["attributes"].get("cost") is not None]

        return {
            "provider": provider_name or "all",
            "model": model_name or "all",
            "request_count": total,
            "successful_requests": success_count,
            "failed_requests": total - success_count,
            "success_rate": round(success_count / total, 4),
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p95_latency_ms": calculate_percentile(latencies, 95.0),
            "total_input_tokens": sum(input_tokens_list) if input_tokens_list else None,
            "total_output_tokens": sum(output_tokens_list) if output_tokens_list else None,
            "total_tokens": sum(total_tokens_list) if total_tokens_list else None,
            "estimated_cost": round(sum(cost_list), 6) if cost_list else None,
        }

    def get_tool_metrics(
        self,
        tool_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregated metrics for tool executions."""
        if self.storage:
            rows = self.storage.list_observability_metrics(
                metric_name="tool_execution_duration_ms",
                start_time=start_time,
                end_time=end_time,
            )
            if tool_name:
                rows = [r for r in rows if r.get("tool_name") == tool_name]
        else:
            with self._lock:
                rows = [
                    m for m in self._in_memory_metrics
                    if m["metric_name"] == "tool_execution_duration_ms"
                    and (not tool_name or m.get("tool_name") == tool_name)
                ]

        total = len(rows)
        if total == 0:
            return {
                "tool_name": tool_name or "all",
                "invocation_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "avg_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
                "error_breakdown": {},
            }

        durations = sorted([float(r["value"]) for r in rows])
        success_count = sum(1 for r in rows if r["attributes"].get("success"))
        error_breakdown: Dict[str, int] = {}
        for r in rows:
            attrs = r.get("attributes", {})
            if not attrs.get("success"):
                err = attrs.get("error_type") or "UnknownError"
                error_breakdown[err] = error_breakdown.get(err, 0) + 1

        return {
            "tool_name": tool_name or "all",
            "invocation_count": total,
            "success_count": success_count,
            "failure_count": total - success_count,
            "success_rate": round(success_count / total, 4),
            "failure_rate": round((total - success_count) / total, 4),
            "avg_duration_ms": round(sum(durations) / total, 2),
            "p95_duration_ms": calculate_percentile(durations, 95.0),
            "error_breakdown": error_breakdown,
        }

    def get_workflow_metrics(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate end-to-end workflow performance metrics."""
        if self.storage:
            rows = self.storage.list_observability_metrics(
                metric_name="workflow_execution_latency_ms",
                start_time=start_time,
                end_time=end_time,
            )
        else:
            with self._lock:
                rows = [
                    m for m in self._in_memory_metrics
                    if m["metric_name"] == "workflow_execution_latency_ms"
                ]

        total = len(rows)
        if total == 0:
            return {
                "total_workflows": 0,
                "completion_rate": 0.0,
                "failure_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "total_agent_transitions": 0,
                "total_human_interventions": 0,
                "total_approval_rejections": 0,
                "total_evaluation_failures": 0,
            }

        latencies = sorted([float(r["value"]) for r in rows])
        success_count = sum(1 for r in rows if r["attributes"].get("success"))
        transitions = sum(int(r["attributes"].get("agent_transitions", 0)) for r in rows)
        human_ints = sum(int(r["attributes"].get("human_interventions", 0)) for r in rows)
        rejections = sum(int(r["attributes"].get("approval_rejections", 0)) for r in rows)
        eval_fails = sum(int(r["attributes"].get("evaluation_failures", 0)) for r in rows)

        return {
            "total_workflows": total,
            "completion_rate": round(success_count / total, 4),
            "failure_rate": round((total - success_count) / total, 4),
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p95_latency_ms": calculate_percentile(latencies, 95.0),
            "total_agent_transitions": transitions,
            "total_human_interventions": human_ints,
            "total_approval_rejections": rejections,
            "total_evaluation_failures": eval_fails,
        }

    def compare_agents(self) -> List[Dict[str, Any]]:
        """Rank and compare performance across all observed agents."""
        if self.storage:
            all_m = self.storage.list_observability_metrics(metric_name="agent_execution_latency_ms")
        else:
            with self._lock:
                all_m = [m for m in self._in_memory_metrics if m["metric_name"] == "agent_execution_latency_ms"]

        agents = sorted(list(set(r["agent_name"] for r in all_m if r.get("agent_name"))))
        results = [self.get_agent_metrics(agent_name=a) for a in agents]
        # Rank by success_rate descending, then P95 latency ascending
        results.sort(key=lambda s: (-s["success_rate"], s["p95_latency_ms"]))
        return results


# ===========================================================================
# 4. PART 5: FAILURE DIAGNOSIS & TRACE ANALYSIS
# ===========================================================================

class FailureCategory(str, Enum):
    """Categorical classification of workflow and span failures."""
    AGENT_FAILURE = "agent_failure"
    TOOL_FAILURE = "tool_failure"
    PROVIDER_FAILURE = "provider_failure"
    NETWORK_OR_TIMEOUT_FAILURE = "network_or_timeout_failure"
    MESSAGE_DELIVERY_FAILURE = "message_delivery_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    HUMAN_REJECTION = "human_rejection"
    EVALUATION_FAILURE = "evaluation_failure"
    OBSERVABILITY_INFRASTRUCTURE_FAILURE = "observability_infrastructure_failure"


@dataclass
class TraceAnalysisReport:
    """Structured diagnostic summary of a trace's execution, bottlenecks, and root cause."""
    trace_id: str
    workflow_id: str
    status: str
    total_duration_ms: float
    agent_sequence: List[str]
    failed_spans: List[Dict[str, Any]]
    root_cause: Optional[Dict[str, Any]]
    failure_category: Optional[FailureCategory]
    retry_count: int
    recovery_succeeded: bool
    slowest_operations: List[Dict[str, Any]]
    evaluation_records: List[Dict[str, Any]]
    hitl_records: List[Dict[str, Any]]
    bottlenecks: List[Dict[str, Any]]
    deterministic_explanation: str

    @property
    def primary_failure_category(self) -> Optional[FailureCategory]:
        return self.failure_category

    @property
    def root_cause_span_id(self) -> Optional[str]:
        return self.root_cause.get("span_id") if self.root_cause else None

    @property
    def root_cause_error(self) -> str:
        if not self.root_cause:
            return ""
        err = self.root_cause.get("error")
        if isinstance(err, dict):
            return err.get("message", str(err))
        return str(err or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
            "agent_sequence": self.agent_sequence,
            "failed_spans": self.failed_spans,
            "root_cause": self.root_cause,
            "failure_category": self.failure_category.value if isinstance(self.failure_category, FailureCategory) else str(self.failure_category) if self.failure_category else None,
            "retry_count": self.retry_count,
            "recovery_succeeded": self.recovery_succeeded,
            "slowest_operations": self.slowest_operations,
            "evaluation_records": self.evaluation_records,
            "hitl_records": self.hitl_records,
            "bottlenecks": self.bottlenecks,
            "deterministic_explanation": self.deterministic_explanation,
        }


class TraceAnalysisEngine:
    """Analyzes trace spans, identifies bottlenecks, and performs root-cause failure attribution."""

    def __init__(self, storage: Optional[SQLiteMessageStorage] = None):
        self.storage = storage

    @staticmethod
    def classify_failure(
        span_or_error: Union[Dict[str, Any], str],
        operation_name: Optional[str] = None,
        span_kind: Optional[str] = None,
    ) -> FailureCategory:
        """Deterministically attribute the failure category of a failed span or error."""
        if isinstance(span_or_error, dict):
            err_val = span_or_error.get("error", {})
            if isinstance(err_val, dict):
                err_msg = str(err_val.get("message", "")).lower()
                err_type = str(err_val.get("type", "")).lower()
            else:
                err_msg = str(err_val).lower()
                err_type = ""
            op = str(span_or_error.get("operation_name", "")).lower()
            kind = str(span_or_error.get("span_kind", "")).lower()
            attrs = span_or_error.get("attributes", {})
        else:
            err_msg = str(span_or_error).lower()
            err_type = ""
            op = str(operation_name or "").lower()
            kind = str(span_kind or "").lower()
            attrs = {}

        # 1. Human Rejection
        if attrs.get("human_rejected") or "rejected by human" in err_msg or "human approval rejected" in err_msg or ("reject" in err_msg and ("human" in err_msg or "approval" in op)):
            return FailureCategory.HUMAN_REJECTION

        # 2. Authorization Failure
        if "unauthorized" in err_msg or "authorization" in err_msg or "permission denied" in err_msg or "insufficient" in err_msg:
            return FailureCategory.AUTHORIZATION_FAILURE

        # 3. Network / Timeout Failure
        if "timeout" in err_msg or "timed out" in err_msg or "connectionerror" in err_type or "network" in err_msg:
            return FailureCategory.NETWORK_OR_TIMEOUT_FAILURE

        # 4. Provider Failure
        if kind == "llm" or "provider" in err_msg or "api key" in err_msg or "rate limit" in err_msg or "ratelimit" in err_msg or "openai" in err_msg or "openrouter" in err_msg or "anthropic" in err_msg:
            return FailureCategory.PROVIDER_FAILURE

        # 5. Tool Failure
        if kind == "tool" or "tool" in err_type or "tool" in op or "tool execution error" in err_msg:
            return FailureCategory.TOOL_FAILURE

        # 6. Evaluation Failure
        if kind == "evaluation" or "evaluation" in err_msg or "evaluation failed" in err_msg or "criteria" in err_msg or "validation schema" in err_msg:
            return FailureCategory.EVALUATION_FAILURE

        # 7. Message Delivery Failure
        if kind == "message" or "message" in op or "delivery failed" in err_msg or "undeliverable" in err_msg or "not found in registry" in err_msg or "route" in op:
            return FailureCategory.MESSAGE_DELIVERY_FAILURE

        # 8. Default to Agent Failure
        return FailureCategory.AGENT_FAILURE

    def analyze_trace(
        self,
        trace: Dict[str, Any],
        spans: List[Dict[str, Any]],
        evaluation_records: Optional[List[Dict[str, Any]]] = None,
        hitl_records: Optional[List[Dict[str, Any]]] = None,
    ) -> TraceAnalysisReport:
        """Synthesize a diagnostic report analyzing execution flow, failures, and bottlenecks."""
        trace_id = trace.get("trace_id", "")
        workflow_id = trace.get("workflow_id", "")
        status = trace.get("status", "COMPLETED")
        total_duration = float(trace.get("duration_ms") or 0.0)
        agent_seq = list(trace.get("agent_sequence", []))

        # Identify failed spans
        failed_spans = [s for s in spans if s.get("status") in ("FAILED", "TIMED_OUT")]
        primary_category: Optional[FailureCategory] = None
        root_cause = None

        if failed_spans:
            # First failed span is the primary causal trigger
            first_fail = failed_spans[0]
            primary_category = self.classify_failure(first_fail)
            root_cause = {
                "span_id": first_fail.get("span_id"),
                "agent_name": first_fail.get("agent_name"),
                "operation_name": first_fail.get("operation_name"),
                "failure_category": primary_category.value,
                "error": first_fail.get("error"),
            }

        # Analyze bottlenecks (slowest operations)
        sorted_by_duration = sorted(
            [s for s in spans if s.get("duration_ms") is not None],
            key=lambda s: float(s["duration_ms"]),
            reverse=True,
        )
        slowest_ops = [
            {
                "span_id": s["span_id"],
                "operation_name": s["operation_name"],
                "agent_name": s["agent_name"],
                "duration_ms": s["duration_ms"],
                "span_kind": s.get("span_kind"),
            }
            for s in sorted_by_duration[:5]
        ]

        # Bottlenecks: flag operations that took > 40% of total duration or > 5000ms
        bottlenecks = []
        for s in slowest_ops:
            if total_duration > 0 and (s["duration_ms"] / total_duration >= 0.40):
                bottlenecks.append({
                    "type": "high_relative_duration",
                    "operation": s["operation_name"],
                    "duration_ms": s["duration_ms"],
                    "fraction_of_total": round(s["duration_ms"] / total_duration, 2),
                })
            elif s["duration_ms"] >= 5000.0:
                bottlenecks.append({
                    "type": "absolute_high_latency",
                    "operation": s["operation_name"],
                    "duration_ms": s["duration_ms"],
                })

        # Count retries across spans
        retries = sum(int(s.get("attributes", {}).get("retry_count", 0)) for s in spans)
        recovered = any(bool(s.get("attributes", {}).get("recovered")) for s in spans)

        # Deterministic explanation
        if primary_category and root_cause:
            err_obj = root_cause.get('error')
            if isinstance(err_obj, dict):
                err_detail = err_obj.get('message', 'Unknown error')
            else:
                err_detail = str(err_obj or 'Unknown error')
            explanation = (
                f"Workflow '{workflow_id}' failed due to {primary_category.value} in "
                f"'{root_cause['agent_name']}' during '{root_cause['operation_name']}': "
                f"{err_detail}."
            )
        elif status == "COMPLETED":
            explanation = (
                f"Workflow '{workflow_id}' completed successfully in {total_duration}ms across "
                f"{len(spans)} spans and {len(agent_seq)} agent transitions."
            )
        else:
            explanation = f"Workflow '{workflow_id}' ended with status '{status}'."

        return TraceAnalysisReport(
            trace_id=trace_id,
            workflow_id=workflow_id,
            status=status,
            total_duration_ms=total_duration,
            agent_sequence=agent_seq,
            failed_spans=failed_spans,
            root_cause=root_cause,
            failure_category=primary_category,
            retry_count=retries,
            recovery_succeeded=recovered,
            slowest_operations=slowest_ops,
            evaluation_records=list(evaluation_records or []),
            hitl_records=list(hitl_records or []),
            bottlenecks=bottlenecks,
            deterministic_explanation=explanation,
        )

    def detect_system_bottlenecks(
        self,
        spans: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Aggregate bottlenecks across multiple traces."""
        if not spans:
            return {"slowest_operations": [], "slowest_agents": [], "slowest_tools": []}

        # Slowest operations
        valid_spans = [s for s in spans if s.get("duration_ms") is not None]
        slowest_ops = sorted(valid_spans, key=lambda s: float(s["duration_ms"]), reverse=True)[:top_k]

        # Agent durations
        agent_durations: Dict[str, List[float]] = {}
        tool_durations: Dict[str, List[float]] = {}

        for s in valid_spans:
            agent = s.get("agent_name")
            if agent:
                agent_durations.setdefault(agent, []).append(float(s["duration_ms"]))
            if s.get("span_kind") == "tool":
                tool_name = s.get("operation_name")
                if tool_name:
                    tool_durations.setdefault(tool_name, []).append(float(s["duration_ms"]))

        agent_rank = [
            {"agent_name": a, "avg_duration_ms": round(sum(d) / len(d), 2), "p95_duration_ms": calculate_percentile(sorted(d), 95.0)}
            for a, d in agent_durations.items()
        ]
        agent_rank.sort(key=lambda x: x["p95_duration_ms"], reverse=True)

        tool_rank = [
            {"tool_name": t, "avg_duration_ms": round(sum(d) / len(d), 2), "p95_duration_ms": calculate_percentile(sorted(d), 95.0)}
            for t, d in tool_durations.items()
        ]
        tool_rank.sort(key=lambda x: x["p95_duration_ms"], reverse=True)

        return {
            "slowest_operations": [
                {"operation": s["operation_name"], "agent": s["agent_name"], "duration_ms": s["duration_ms"]}
                for s in slowest_ops
            ],
            "slowest_agents": agent_rank[:top_k],
            "slowest_tools": tool_rank[:top_k],
        }


# ===========================================================================
# 5. PART 6: UNIFIED AGENT OBSERVABILITY MANAGER
# ===========================================================================

class AgentObservabilityManager:
    """Central coordinator for distributed tracing, structured logging, and metrics."""

    def __init__(
        self,
        storage: Optional[SQLiteMessageStorage] = None,
        config: Optional[ObservabilityConfig] = None,
    ):
        self.storage = storage
        self.config = config or ObservabilityConfig()
        self.metrics = AgentMetricsCollector(storage=storage)
        self.analysis_engine = TraceAnalysisEngine(storage=storage)

        self._lock = threading.RLock()
        self._active_traces: Dict[str, TraceRecord] = {}
        self._active_spans: Dict[str, SpanRecord] = {}
        self._in_memory_logs: List[StructuredLogRecord] = []
        self._in_memory_events: List[AgentExecutionEvent] = []

        # Thread-local storage maintaining the ambient trace context
        self._context_storage = threading.local()

    # -----------------------------------------------------------------------
    # Context Management
    # -----------------------------------------------------------------------

    def get_current_context(self) -> Optional[TraceContext]:
        """Retrieve the ambient trace context for the current thread/task."""
        ctx = _current_trace_context.get()
        if ctx is not None:
            return ctx
        return getattr(self._context_storage, "current_context", None)

    def set_current_context(self, context: Optional[TraceContext]) -> None:
        """Set the ambient trace context for the current thread/task."""
        _current_trace_context.set(context)
        self._context_storage.current_context = context

    # -----------------------------------------------------------------------
    # Trace & Span Lifecycle
    # -----------------------------------------------------------------------

    def start_trace(
        self,
        workflow_id: str,
        task_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceRecord:
        """Initiate a root distributed trace for a workflow."""
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        root_span_id = f"span_{uuid.uuid4().hex[:8]}"

        trace = TraceRecord(
            trace_id=trace_id,
            workflow_id=workflow_id,
            task_id=task_id,
            root_span_id=root_span_id,
            start_time=datetime.now(timezone.utc).isoformat(),
            status=SpanStatus.STARTED,
            attributes=dict(attributes or {}),
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._active_traces[trace_id] = trace

        # Set ambient context
        ctx = TraceContext(
            trace_id=trace_id,
            workflow_id=workflow_id,
            task_id=task_id,
            current_span_id=root_span_id,
        )
        self.set_current_context(ctx)

        # Log event safely
        self.log(
            event_type=EventCategory.WORKFLOW_STARTED.value,
            message=f"Workflow '{workflow_id}' initiated with trace '{trace_id}'.",
            log_level=LogLevel.INFO,
            trace_id=trace_id,
            workflow_id=workflow_id,
            task_id=task_id,
            attributes=attributes,
        )

        if self.storage and self.config.enabled:
            try:
                self.storage.save_observability_trace(trace.to_dict())
            except Exception as exc:
                logger.warning(f"[ObservabilityManager] Could not persist trace: {exc}")

        return trace

    def start_span(
        self,
        operation_name: str,
        agent_name: str,
        span_kind: Union[SpanKind, str] = SpanKind.AGENT,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpanRecord:
        """Create and start a child execution span."""
        span_id = f"span_{uuid.uuid4().hex[:8]}"
        ambient_ctx = self.get_current_context()

        # Derive trace linkage
        if ambient_ctx:
            trace_id = ambient_ctx.trace_id
            workflow_id = ambient_ctx.workflow_id
            task_id = ambient_ctx.task_id
            parent_id = parent_span_id or ambient_ctx.current_span_id
        else:
            trace_id = f"tr_{uuid.uuid4().hex[:12]}"
            workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
            task_id = None
            parent_id = parent_span_id

        kind_enum = span_kind if isinstance(span_kind, SpanKind) else SpanKind(str(span_kind).lower())

        span = SpanRecord(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_id,
            workflow_id=workflow_id,
            task_id=task_id,
            agent_name=agent_name,
            operation_name=operation_name,
            span_kind=kind_enum,
            start_time=datetime.now(timezone.utc).isoformat(),
            status=SpanStatus.STARTED,
            attributes=dict(attributes or {}),
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._active_spans[span_id] = span
            if trace_id in self._active_traces:
                self._active_traces[trace_id].spans.append(span)
                self._active_traces[trace_id].add_agent_transition(agent_name)

        # Update ambient context
        new_ctx = TraceContext(
            trace_id=trace_id,
            workflow_id=workflow_id,
            task_id=task_id,
            parent_span_id=parent_id,
            current_span_id=span_id,
        )
        self.set_current_context(new_ctx)

        if self.storage and self.config.enabled:
            try:
                self.storage.save_observability_span(span.to_dict())
            except Exception as exc:
                logger.warning(f"[ObservabilityManager] Could not persist span: {exc}")

        return span

    def end_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.COMPLETED,
        attributes: Optional[Dict[str, Any]] = None,
        error: Optional[Any] = None,
    ) -> Optional[SpanRecord]:
        """Conclude an execution span and update metrics."""
        with self._lock:
            span = self._active_spans.get(span_id)
            if not span:
                return None



            if error:
                span.fail(error=error, attributes=attributes)
            else:
                span.complete(attributes=attributes, status=status)

            # Record duration metric
            if span.duration_ms is not None:
                if span.span_kind == SpanKind.AGENT:
                    self.metrics.record_agent_execution(
                        agent_name=span.agent_name,
                        duration_ms=span.duration_ms,
                        success=(span.status == SpanStatus.COMPLETED),
                        workflow_id=span.workflow_id,
                        task_id=span.task_id,
                    )
                elif span.span_kind == SpanKind.TOOL:
                    self.metrics.record_tool_call(
                        tool_name=span.operation_name,
                        duration_ms=span.duration_ms,
                        success=(span.status == SpanStatus.COMPLETED),
                        workflow_id=span.workflow_id,
                        task_id=span.task_id,
                    )

            if self.storage and self.config.enabled:
                try:
                    self.storage.save_observability_span(span.to_dict())
                except Exception as exc:
                    logger.warning(f"[ObservabilityManager] Could not persist completed span: {exc}")

            return span

    def end_trace(
        self,
        trace_id: str,
        status: SpanStatus = SpanStatus.COMPLETED,
        error: Optional[Any] = None,
    ) -> Optional[TraceRecord]:
        """Conclude a root trace and record workflow metrics."""
        with self._lock:
            trace = self._active_traces.get(trace_id)
            if not trace:
                return None



            if error:
                trace.fail(error=error)
            else:
                trace.complete(status=status)

            # Record workflow metric
            if trace.duration_ms is not None:
                self.metrics.record_workflow_execution(
                    workflow_id=trace.workflow_id,
                    duration_ms=trace.duration_ms,
                    success=(trace.status == SpanStatus.COMPLETED),
                    agent_transitions=len(trace.agent_sequence),
                )

            # Log completion event
            ev_type = EventCategory.WORKFLOW_COMPLETED.value if trace.status == SpanStatus.COMPLETED else EventCategory.WORKFLOW_FAILED.value
            self.log(
                event_type=ev_type,
                message=f"Workflow '{trace.workflow_id}' completed with status '{trace.status.value}'.",
                log_level=LogLevel.INFO if trace.status == SpanStatus.COMPLETED else LogLevel.ERROR,
                trace_id=trace.trace_id,
                workflow_id=trace.workflow_id,
                task_id=trace.task_id,
            )

            if self.storage and self.config.enabled:
                try:
                    self.storage.save_observability_trace(trace.to_dict())
                except Exception as exc:
                    logger.warning(f"[ObservabilityManager] Could not persist ended trace: {exc}")

            # Clear ambient context if it matches this trace
            curr = self.get_current_context()
            if curr and curr.trace_id == trace_id:
                self.set_current_context(None)

            return trace

    # -----------------------------------------------------------------------
    # Python Context Managers
    # -----------------------------------------------------------------------

    @asynccontextmanager
    async def async_trace(
        self,
        workflow_id: str,
        task_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        trace_record = self.start_trace(workflow_id=workflow_id, task_id=task_id, attributes=attributes)
        try:
            yield trace_record
            if trace_record.status == SpanStatus.STARTED:
                self.end_trace(trace_record.trace_id, status=SpanStatus.COMPLETED)
        except Exception as exc:
            self.end_trace(trace_record.trace_id, status=SpanStatus.FAILED, error=exc)
            raise

    @asynccontextmanager
    async def async_span(
        self,
        operation_name: str,
        agent_name: str,
        span_kind: Union[SpanKind, str] = SpanKind.AGENT,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        prev_ctx = self.get_current_context()
        span_rec = self.start_span(
            operation_name=operation_name,
            agent_name=agent_name,
            span_kind=span_kind,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )
        try:
            yield span_rec
            if span_rec.status == SpanStatus.STARTED:
                self.end_span(span_rec.span_id, status=SpanStatus.COMPLETED)
        except Exception as exc:
            self.end_span(span_rec.span_id, status=SpanStatus.FAILED, error=exc)
            raise
        finally:
            if prev_ctx:
                self.set_current_context(prev_ctx)

    @contextmanager
    def trace(
        self,
        workflow_id: str,
        task_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[TraceRecord, None, None]:
        """Context manager scoping an entire workflow execution trace."""
        trace = self.start_trace(workflow_id=workflow_id, task_id=task_id, attributes=attributes)
        try:
            yield trace
            if trace.status == SpanStatus.STARTED:
                self.end_trace(trace.trace_id, status=SpanStatus.COMPLETED)
        except Exception as exc:
            self.end_trace(trace.trace_id, status=SpanStatus.FAILED, error=exc)
            raise

    @contextmanager
    def span(
        self,
        operation_name: str,
        agent_name: str,
        span_kind: Union[SpanKind, str] = SpanKind.AGENT,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanRecord, None, None]:
        """Context manager scoping a child execution span."""
        span_rec = self.start_span(
            operation_name=operation_name,
            agent_name=agent_name,
            span_kind=span_kind,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )
        try:
            yield span_rec
            if span_rec.status == SpanStatus.STARTED:
                self.end_span(span_rec.span_id, status=SpanStatus.COMPLETED)
        except Exception as exc:
            self.end_span(span_rec.span_id, status=SpanStatus.FAILED, error=exc)
            raise

    # -----------------------------------------------------------------------
    # Structured Logging & Event Emission
    # -----------------------------------------------------------------------

    def log(
        self,
        event_type: str,
        message: str,
        log_level: Union[LogLevel, str] = LogLevel.INFO,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        error: Optional[Any] = None,
    ) -> StructuredLogRecord:
        """Record a structured log entry with automatic secret scrubbing and fault isolation."""
        try:
            # Derive trace context if not supplied
            ambient = self.get_current_context()
            tid = trace_id or (ambient.trace_id if ambient else None)
            sid = span_id or (ambient.current_span_id if ambient else None)
            wid = workflow_id or (ambient.workflow_id if ambient else None)
            tkid = task_id or (ambient.task_id if ambient else None)

            level_enum = log_level if isinstance(log_level, LogLevel) else LogLevel(str(log_level).upper())

            err_type = type(error).__name__ if isinstance(error, Exception) else None
            err_msg = str(error) if error else None

            # Apply sensitive data scrubbing
            cleaned_message = redact_observability_data(message) if self.config.redact_sensitive_data else message
            cleaned_attrs = redact_observability_data(attributes or {}) if self.config.redact_sensitive_data else dict(attributes or {})

            record = StructuredLogRecord(
                log_id=f"log_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                log_level=level_enum,
                event_type=event_type,
                trace_id=tid,
                span_id=sid,
                workflow_id=wid,
                task_id=tkid,
                agent_name=agent_name,
                operation_name=operation_name,
                message=cleaned_message,
                attributes=cleaned_attrs,
                error_type=err_type,
                error_message=err_msg,
            )

            with self._lock:
                self._in_memory_logs.append(record)

            if self.storage and self.config.enabled and self.config.collect_logs:
                self.storage.save_observability_log(record.to_dict())

            return record
        except Exception as exc:
            # Fault isolation: Logging failure must NEVER break the agent execution loop
            logger.warning(f"[ObservabilityManager] Internal logging failure safely suppressed: {exc}")
            return StructuredLogRecord(
                log_id="log_fallback",
                timestamp=datetime.now(timezone.utc).isoformat(),
                log_level=LogLevel.ERROR,
                event_type="OBSERVABILITY_LOG_ERROR",
                trace_id=None,
                span_id=None,
                workflow_id=None,
                task_id=None,
                agent_name=None,
                operation_name=None,
                message=str(message),
            )

    def record_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        agent_name: Optional[str] = None,
        span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> AgentExecutionEvent:
        """Emit a discrete execution event."""
        ambient = self.get_current_context()
        tid = trace_id or (ambient.trace_id if ambient else f"tr_{uuid.uuid4().hex[:12]}")
        sid = span_id or (ambient.current_span_id if ambient else None)

        cleaned_payload = redact_observability_data(payload or {}) if self.config.redact_sensitive_data else dict(payload or {})

        event = AgentExecutionEvent(
            event_id=f"ev_{uuid.uuid4().hex[:8]}",
            trace_id=tid,
            span_id=sid,
            event_type=event_type,
            agent_name=agent_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=cleaned_payload,
        )

        with self._lock:
            self._in_memory_events.append(event)

        if self.storage and self.config.enabled and self.config.collect_events:
            try:
                self.storage.save_observability_event(event.to_dict())
            except Exception as exc:
                logger.warning(f"[ObservabilityManager] Could not persist event: {exc}")

        return event

    # -----------------------------------------------------------------------
    # Query & Inspection APIs
    # -----------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> Optional[TraceRecord]:
        """Fetch a trace record from active memory or SQLite."""
        with self._lock:
            if trace_id in self._active_traces:
                return self._active_traces[trace_id]

        if self.storage:
            row = self.storage.get_observability_trace(trace_id)
            if row:
                spans_rows = self.storage.list_observability_spans(trace_id=trace_id)
                spans = [
                    SpanRecord(
                        span_id=s["span_id"],
                        trace_id=s["trace_id"],
                        parent_span_id=s["parent_span_id"],
                        workflow_id=s["workflow_id"],
                        task_id=s["task_id"],
                        agent_name=s["agent_name"],
                        operation_name=s["operation_name"],
                        span_kind=SpanKind(s.get("span_kind", "agent")),
                        start_time=s["start_time"],
                        end_time=s["end_time"],
                        duration_ms=s["duration_ms"],
                        status=SpanStatus(s.get("status", "STARTED")),
                        attributes=s["attributes"],
                        error=s["error"],
                        metadata=s["metadata"],
                    )
                    for s in spans_rows
                ]
                return TraceRecord(
                    trace_id=row["trace_id"],
                    workflow_id=row["workflow_id"],
                    task_id=row["task_id"],
                    root_span_id=row["root_span_id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    duration_ms=row["duration_ms"],
                    status=SpanStatus(row.get("status", "STARTED")),
                    agent_sequence=row["agent_sequence"],
                    spans=spans,
                    error=row["error"],
                    attributes=row["attributes"],
                    metadata=row["metadata"],
                )
        return None



    def get_workflow_trace(self, workflow_id: str) -> Optional[TraceRecord]:
        """Fetch the most recent trace for a workflow."""
        with self._lock:
            for t in self._active_traces.values():
                if t.workflow_id == workflow_id:
                    return t

        if self.storage:
            traces = self.storage.list_observability_traces(workflow_id=workflow_id, limit=1)
            if traces:
                return self.get_trace(traces[0]["trace_id"])
        return None



    def get_spans_by_trace(self, trace_id: str) -> List[SpanRecord]:
        """Retrieve all spans associated with a trace."""
        trace = self.get_trace(trace_id)
        if trace:
            return trace.spans
        if self.storage:
            rows = self.storage.list_observability_spans(trace_id=trace_id)
            return [
                SpanRecord(
                    span_id=s["span_id"],
                    trace_id=s["trace_id"],
                    parent_span_id=s["parent_span_id"],
                    workflow_id=s["workflow_id"],
                    task_id=s["task_id"],
                    agent_name=s["agent_name"],
                    operation_name=s["operation_name"],
                    span_kind=SpanKind(s.get("span_kind", "agent")),
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                    duration_ms=s["duration_ms"],
                    status=SpanStatus(s.get("status", "STARTED")),
                    attributes=s["attributes"],
                    error=s["error"],
                    metadata=s["metadata"],
                )
                for s in rows
            ]
        return []

    def get_failed_spans(self, trace_id: str) -> List[SpanRecord]:
        """Return all failed or timed-out spans for a trace."""
        all_spans = self.get_spans_by_trace(trace_id)
        return [s for s in all_spans if s.status in (SpanStatus.FAILED, SpanStatus.TIMED_OUT)]

    def query_logs(
        self,
        trace_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        log_level: Optional[Union[LogLevel, str]] = None,
        event_type: Optional[str] = None,
        agent_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 200,
    ) -> List[StructuredLogRecord]:
        """Query structured logs matching given filters."""
        level_str = log_level.value if isinstance(log_level, LogLevel) else (str(log_level).upper() if log_level else None)
        if self.storage:
            rows = self.storage.list_observability_logs(
                trace_id=trace_id,
                workflow_id=workflow_id,
                log_level=level_str,
                event_type=event_type,
                agent_name=agent_name,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
            return [
                StructuredLogRecord(
                    log_id=r["log_id"],
                    timestamp=r["timestamp"],
                    log_level=LogLevel(r["log_level"]),
                    event_type=r["event_type"],
                    trace_id=r["trace_id"],
                    span_id=r["span_id"],
                    workflow_id=r["workflow_id"],
                    task_id=r["task_id"],
                    agent_name=r["agent_name"],
                    operation_name=r["operation_name"],
                    message=r["message"],
                    attributes=r["attributes"],
                    error_type=r["error_type"],
                    error_message=r["error_message"],
                )
                for r in rows
            ]
        else:
            with self._lock:
                res = [
                    l for l in self._in_memory_logs
                    if (not trace_id or l.trace_id == trace_id)
                    and (not workflow_id or l.workflow_id == workflow_id)
                    and (not level_str or l.log_level.value == level_str)
                    and (not event_type or l.event_type == event_type)
                    and (not agent_name or l.agent_name == agent_name)
                ]
                return res[:limit]

    def list_logs(
        self,
        trace_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        log_level: Optional[Union[LogLevel, str]] = None,
        event_type: Optional[str] = None,
        agent_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List structured logs matching filters, returned as dictionary records."""
        records = self.query_logs(
            trace_id=trace_id,
            workflow_id=workflow_id,
            log_level=log_level,
            event_type=event_type,
            agent_name=agent_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]

    def list_spans(
        self,
        trace_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List spans matching filters, returned as dictionary records."""
        if self.storage:
            return self.storage.list_observability_spans(
                trace_id=trace_id,
                agent_name=agent_name,
                operation_name=operation_name,
                status=status,
            )
        else:
            with self._lock:
                spans = [s.to_dict() for s in self._active_spans.values()]
                if trace_id:
                    spans = [s for s in spans if s.get("trace_id") == trace_id]
                if agent_name:
                    spans = [s for s in spans if s.get("agent_name") == agent_name]
                if operation_name:
                    spans = [s for s in spans if s.get("operation_name") == operation_name]
                if status:
                    spans = [s for s in spans if s.get("status") == status]
                return spans

    def list_traces(
        self,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List traces matching filters, returned as dictionary records."""
        if self.storage:
            return self.storage.list_observability_traces(
                workflow_id=workflow_id,
                task_id=task_id,
                status=status,
                limit=limit,
            )
        else:
            with self._lock:
                traces = [t.to_dict() for t in self._active_traces.values()]
                if workflow_id:
                    traces = [t for t in traces if t.get("workflow_id") == workflow_id]
                if task_id:
                    traces = [t for t in traces if t.get("task_id") == task_id]
                if status:
                    traces = [t for t in traces if t.get("status") == status]
                return traces[:limit]

    def analyze_trace(self, trace_id: str) -> Optional[TraceAnalysisReport]:
        """Generate a complete diagnostic failure and bottleneck report for a trace."""
        trace = self.get_trace(trace_id)
        if not trace:
            return None



        spans = [s.to_dict() for s in trace.spans]

        # Gather correlated evaluation records if storage is available
        evals: List[Dict[str, Any]] = []
        if self.storage and hasattr(self.storage, "list_agent_evaluations"):
            try:
                evals = self.storage.list_agent_evaluations(task_id=trace.task_id or trace.workflow_id)
            except Exception:
                pass

        return self.analysis_engine.analyze_trace(
            trace=trace.to_dict(),
            spans=spans,
            evaluation_records=evals,
        )

    def detect_bottlenecks(self, top_k: int = 5) -> Dict[str, Any]:
        """Aggregate system bottlenecks across all observed spans."""
        if self.storage:
            spans = self.storage.list_observability_spans()
        else:
            with self._lock:
                spans = [s.to_dict() for s in self._active_spans.values()]
        return self.analysis_engine.detect_system_bottlenecks(spans=spans, top_k=top_k)

    def get_summary(self) -> ObservabilitySummary:
        """Produce an overview summary of observability health."""
        if self.storage:
            traces = self.storage.list_observability_traces(limit=1000)
            spans = self.storage.list_observability_spans()
            logs = self.storage.list_observability_logs(limit=1000)
            metrics = self.storage.list_observability_metrics()
        else:
            with self._lock:
                traces = [t.to_dict() for t in self._active_traces.values()]
                spans = [s.to_dict() for s in self._active_spans.values()]
                logs = [l.to_dict() for l in self._in_memory_logs]
                metrics = list(self.metrics._in_memory_metrics)

        total_t = len(traces)
        active_t = sum(1 for t in traces if t.get("status") == "STARTED")
        comp_t = sum(1 for t in traces if t.get("status") == "COMPLETED")
        fail_t = sum(1 for t in traces if t.get("status") in ("FAILED", "TIMED_OUT"))

        latencies = sorted([float(t["duration_ms"]) for t in traces if t.get("duration_ms") is not None])
        p95 = calculate_percentile(latencies, 95.0)

        return ObservabilitySummary(
            total_traces=total_t,
            active_traces=active_t,
            completed_traces=comp_t,
            failed_traces=fail_t,
            total_spans=len(spans),
            total_logs=len(logs),
            total_metrics=len(metrics),
            p95_workflow_latency_ms=p95,
        )

    def enforce_retention(self, retention_days: Optional[int] = None) -> Dict[str, int]:
        """Enforce retention policy by purging records older than retention threshold."""
        days = retention_days if retention_days is not None else self.config.retention_days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        if self.storage:
            return self.storage.cleanup_observability_records(older_than_iso=cutoff)
        return {"purged_in_memory": 0}

    # -----------------------------------------------------------------------
    # Provider & Tool Instrumentation Wrappers
    # -----------------------------------------------------------------------

    def wrap_provider(self, provider: LLMProvider) -> LLMProvider:
        """Wrap an existing LLMProvider instance with transparent observability instrumentation."""
        return ObservedLLMProvider(underlying_provider=provider, observability=self)


class ObservedLLMProvider(LLMProvider):
    """Decorator wrapping an LLMProvider to record spans and metrics on every call."""

    def __init__(self, underlying_provider: LLMProvider, observability: AgentObservabilityManager):
        self.underlying_provider = underlying_provider
        self.observability = observability
        self.provider_name = getattr(underlying_provider, "provider_name", "llm_provider")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.underlying_provider, name)

    def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Execute model generation while capturing spans, latency, and tokens."""
        span_name = f"llm_generate:{model}"
        span = self.observability.start_span(
            operation_name=span_name,
            agent_name="llm_client",
            span_kind=SpanKind.LLM,
            attributes={
                "provider": self.provider_name,
                "model": model,
                "message_count": len(messages),
                "has_tools": bool(tools),
            },
        )

        self.observability.log(
            event_type=EventCategory.LLM_REQUEST_STARTED.value,
            message=f"LLM request started for model '{model}' ({len(messages)} messages).",
            log_level=LogLevel.DEBUG,
            span_id=span.span_id,
        )

        try:
            response: LLMResponse = self.underlying_provider.generate(
                messages=messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
            )

            # Extract usage metrics if honestly reported
            usage = getattr(response, "usage", None) or {}
            in_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            out_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            total_tokens = usage.get("total_tokens") or ((in_tokens + out_tokens) if in_tokens and out_tokens else None)
            cost = usage.get("estimated_cost") or usage.get("cost")

            if response.error:
                self.observability.end_span(
                    span.span_id,
                    status=SpanStatus.FAILED,
                    error=response.error,
                    attributes={"raw_response": str(response.raw_response)[:100]},
                )
                self.observability.metrics.record_llm_call(
                    provider_name=self.provider_name,
                    model_name=model,
                    latency_ms=span.duration_ms or 0.0,
                    success=False,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                )
                self.observability.log(
                    event_type=EventCategory.LLM_REQUEST_FAILED.value,
                    message=f"LLM request failed: {response.error}",
                    log_level=LogLevel.ERROR,
                    span_id=span.span_id,
                    error=response.error,
                )
            else:
                self.observability.end_span(
                    span.span_id,
                    status=SpanStatus.COMPLETED,
                    attributes={
                        "model": model,
                        "has_tool_calls": len(response.tool_calls) > 0,
                        "tool_calls_count": len(response.tool_calls),
                        "input_tokens": in_tokens,
                        "output_tokens": out_tokens,
                        "total_tokens": total_tokens,
                    },
                )
                self.observability.metrics.record_llm_call(
                    provider_name=self.provider_name,
                    model_name=model,
                    latency_ms=span.duration_ms or 0.0,
                    success=True,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    total_tokens=total_tokens,
                    cost=cost,
                )
                self.observability.log(
                    event_type=EventCategory.LLM_REQUEST_COMPLETED.value,
                    message=f"LLM request completed in {span.duration_ms}ms.",
                    log_level=LogLevel.DEBUG,
                    span_id=span.span_id,
                )

            return response
        except Exception as exc:
            span.fail(error=exc)
            self.observability.metrics.record_llm_call(
                provider_name=self.provider_name,
                model_name=model,
                latency_ms=span.duration_ms or 0.0,
                success=False,
            )
            self.observability.log(
                event_type=EventCategory.LLM_REQUEST_FAILED.value,
                message=f"LLM request exception: {str(exc)}",
                log_level=LogLevel.ERROR,
                span_id=span.span_id,
                error=exc,
            )
            raise
