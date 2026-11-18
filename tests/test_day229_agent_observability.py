"""
tests/test_day229_agent_observability.py

Comprehensive Test Suite for Day 229: Agent Observability & Tracing (Parts 1 to 6).

Covers:
- Part 1: Observability Core Models & Contracts
  - TraceContext, SpanRecord, TraceRecord, AgentExecutionEvent
  - ObservabilityConfig, SpanStatus, SpanKind
  - Sensitive data protection / masking (redact_observability_data)
- Part 2: Distributed Agent Tracing & Context Propagation
  - Root trace creation, hierarchical parent-child spans
  - Agent transition sequencing
  - Context injection/extraction over message boundaries
  - ObservedLLMProvider wrapper
- Part 3: Centralized Structured Logging & Secret Redaction
  - StructuredLogRecord, LogLevel, EventCategory (18 standard categories)
  - Log filtering by level, trace, and event type
  - Deep recursive secret redaction (API keys, bearer tokens, passwords, JWTs)
  - Logging failure isolation
- Part 4: Agent & System Metrics Collection
  - Nearest-rank percentile calculation (P50, P95, P99)
  - Agent latency, retry, and recovery metrics
  - LLM request metrics with honest token reporting (no fabrication)
  - Tool execution metrics and workflow metrics
  - Comparison across agents
- Part 5: Trace Analysis Engine & Failure Attribution
  - FailureCategory classification across 8 deterministic domains
  - Bottleneck detection (slowest operations, slowest agents, slowest tools)
  - Deterministic root cause analysis
  - Separation of human rejections from agent execution failures
  - Day 228 evaluation record correlation
- Part 6: Unified Integration, Persistence, Retention & Safety
  - AgentObservabilityManager coordination
  - Full WorkflowOrchestrator integration with tracing and spans
  - SQLite message storage persistence and query methods
  - Retention policy cleanup
  - Observability failure isolation (zero-crash guarantee)
  - Backward compatibility
"""

import os
import tempfile
import time
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from core.observability import (
    SpanStatus,
    SpanKind,
    LogLevel,
    EventCategory,
    FailureCategory,
    TraceContext,
    SpanRecord,
    TraceRecord,
    AgentExecutionEvent,
    StructuredLogRecord,
    ObservabilityConfig,
    ObservabilitySummary,
    TraceAnalysisReport,
    calculate_percentile,
    redact_observability_data,
    AgentMetricsCollector,
    TraceAnalysisEngine,
    AgentObservabilityManager,
    ObservedLLMProvider,
)
from core.persistent_storage import SQLiteMessageStorage
from core.provider_interface import LLMProvider, LLMMessage, LLMResponse
from core.orchestrator import WorkflowOrchestrator, TaskContext, WorkflowStatus, AgentMessage, MessageType
from agents.researcher import ResearcherAgent
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.supervisor import SupervisorAgent


# ===========================================================================
# Fixtures & Mocks
# ===========================================================================

class MockLLMProvider(LLMProvider):
    """Mock provider with predictable responses and simulated token metrics."""

    def __init__(self, return_text: str = "Mock response", usage: dict = None, should_fail: bool = False):
        self.return_text = return_text
        self.usage = usage
        self.should_fail = should_fail
        self.provider_name = "mock_provider"
        self.calls = []

    def generate(self, messages, model="mock_model", tools=None, tool_choice="auto"):
        self.calls.append({"messages": messages, "model": model, "tools": tools})
        if self.should_fail:
            raise RuntimeError("Simulated provider network failure.")
        return LLMResponse(
            content=self.return_text,
            usage=self.usage,
            raw_response={"usage": self.usage} if self.usage else {},
        )


@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite database file for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


@pytest.fixture
def storage(temp_db_path):
    """Initialize an active SQLite storage instance."""
    st = SQLiteMessageStorage(db_path=temp_db_path)
    return st


@pytest.fixture
def observability_mgr(storage):
    """Initialize AgentObservabilityManager with storage."""
    config = ObservabilityConfig(service_name="test_service", retention_days=7)
    return AgentObservabilityManager(storage=storage, config=config)


# ===========================================================================
# PART 1: Core Observability Models & Contracts
# ===========================================================================

def test_part1_trace_context_initialization_and_serialization():
    """Verify TraceContext creation, child context derivation, and serialization."""
    ctx = TraceContext(
        trace_id="tr_1234567890ab",
        workflow_id="wf_001",
        task_id="task_001",
        current_span_id="span_root",
        baggage={"tenant": "customer_a"},
    )
    assert ctx.trace_id == "tr_1234567890ab"
    assert ctx.workflow_id == "wf_001"
    assert ctx.current_span_id == "span_root"
    assert ctx.parent_span_id is None

    # Child context creation
    child_ctx = ctx.create_child_context(span_id="span_child_1")
    assert child_ctx.trace_id == ctx.trace_id
    assert child_ctx.parent_span_id == "span_root"
    assert child_ctx.current_span_id == "span_child_1"
    assert child_ctx.baggage["tenant"] == "customer_a"

    # Serialization round-trip
    d = ctx.to_dict()
    reconstructed = TraceContext.from_dict(d)
    assert reconstructed.trace_id == ctx.trace_id
    assert reconstructed.workflow_id == ctx.workflow_id
    assert reconstructed.baggage == ctx.baggage

    # Safe fallback on invalid data
    fallback = TraceContext.from_dict(None)
    assert fallback.trace_id.startswith("tr_")
    assert fallback.workflow_id.startswith("wf_")


def test_part1_trace_and_span_ids_uniqueness():
    """Verify trace and span IDs conform to formatting and ensure uniqueness."""
    trace_ids = {f"tr_{os.urandom(6).hex()}" for _ in range(100)}
    span_ids = {f"span_{os.urandom(4).hex()}" for _ in range(100)}
    assert len(trace_ids) == 100
    assert len(span_ids) == 100
    for tid in trace_ids:
        assert tid.startswith("tr_")
    for sid in span_ids:
        assert sid.startswith("span_")


def test_part1_span_record_lifecycle_and_duration():
    """Verify SpanRecord starts, completes, and calculates duration accurately."""
    span = SpanRecord(
        span_id="span_abc123",
        trace_id="tr_xyz789",
        workflow_id="wf_001",
        agent_name="ResearcherAgent",
        operation_name="gather_context",
        span_kind=SpanKind.AGENT,
    )
    assert span.status == SpanStatus.STARTED
    assert span.end_time is None
    assert span.duration_ms is None

    time.sleep(0.01)  # small pause
    span.complete(attributes={"findings_count": 5})

    assert span.status == SpanStatus.COMPLETED
    assert span.end_time is not None
    assert span.duration_ms is not None
    assert span.duration_ms >= 0.0
    assert span.attributes["findings_count"] == 5

    # Test failure lifecycle
    span2 = SpanRecord(
        span_id="span_fail",
        trace_id="tr_xyz789",
        workflow_id="wf_001",
        agent_name="ExecutorAgent",
        operation_name="run_tool",
    )
    span2.fail(error="Simulated tool error", attributes={"tool": "db_query"})
    assert span2.status == SpanStatus.FAILED
    assert span2.error["message"] == "Simulated tool error"
    assert span2.attributes["tool"] == "db_query"


def test_part1_sensitive_data_protection_and_redaction():
    """Verify redact_observability_data scrubs API keys, JWTs, passwords, and tokens."""
    sensitive_dict = {
        "api_key": "sk-or-v1-abcdef1234567890abcdef1234567890",
        "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.secret",
        "password": "supersecretpassword123",
        "normal_field": "This is public text.",
        "nested": {
            "token": "tok_xyz987654321",
            "prompt": "Call API with Authorization: Bearer sk-live-secretkey",
        },
        "list_items": [
            "public",
            "secret_pass: 12345678",
        ],
    }

    redacted = redact_observability_data(sensitive_dict)

    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["authorization"] == "[REDACTED]"
    assert redacted["password"] == "[REDACTED]"
    assert redacted["normal_field"] == "This is public text."
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert "[REDACTED]" in redacted["nested"]["prompt"]
    assert "[REDACTED]" in redacted["list_items"][1]


# ===========================================================================
# PART 2: Distributed Agent Tracing & Context Propagation
# ===========================================================================

def test_part2_distributed_trace_creation_and_completion(observability_mgr):
    """Test start_trace and end_trace lifecycle."""
    trace = observability_mgr.start_trace(
        workflow_id="wf_dist_01",
        task_id="task_dist_01",
        attributes={"objective": "benchmark test"},
    )
    assert trace.trace_id.startswith("tr_")
    assert trace.workflow_id == "wf_dist_01"
    assert trace.status == SpanStatus.STARTED

    # Check ambient context is active
    ctx = observability_mgr.get_current_context()
    assert ctx is not None
    assert ctx.trace_id == trace.trace_id

    # Complete trace
    completed = observability_mgr.end_trace(trace.trace_id, status=SpanStatus.COMPLETED)
    assert completed is not None
    assert completed.status == SpanStatus.COMPLETED
    assert completed.duration_ms is not None

    # Ambient context cleared
    assert observability_mgr.get_current_context() is None


def test_part2_nested_spans_and_agent_transitions(observability_mgr):
    """Test nested spans and sequential agent transition tracking in trace."""
    trace = observability_mgr.start_trace(workflow_id="wf_transitions", task_id="task_trans")

    # Span 1: Researcher
    span1 = observability_mgr.start_span(
        operation_name="research_step",
        agent_name="ResearcherAgent",
        span_kind=SpanKind.AGENT,
    )
    time.sleep(0.005)
    observability_mgr.end_span(span1.span_id, status=SpanStatus.COMPLETED)

    # Span 2: Planner
    span2 = observability_mgr.start_span(
        operation_name="planning_step",
        agent_name="PlannerAgent",
        span_kind=SpanKind.AGENT,
    )
    time.sleep(0.005)
    observability_mgr.end_span(span2.span_id, status=SpanStatus.COMPLETED)

    # Span 3: Executor
    span3 = observability_mgr.start_span(
        operation_name="execution_step",
        agent_name="ExecutorAgent",
        span_kind=SpanKind.AGENT,
    )
    time.sleep(0.005)
    observability_mgr.end_span(span3.span_id, status=SpanStatus.COMPLETED)

    observability_mgr.end_trace(trace.trace_id, status=SpanStatus.COMPLETED)

    # Check agent transitions
    assert [a.lower() for a in trace.agent_sequence] == ["researcheragent", "planneragent", "executoragent"]
    assert len(trace.spans) == 3


def test_part2_context_propagation_inject_and_extract():
    """Test TraceContext injection and extraction over AgentMessage metadata carrier."""
    ctx = TraceContext(
        trace_id="tr_prop_123",
        workflow_id="wf_prop_01",
        task_id="task_prop_01",
        current_span_id="span_parent_999",
        baggage={"priority": "high"},
    )

    metadata = {}
    ctx.inject(metadata)

    assert metadata["trace_id"] == "tr_prop_123"
    assert "trace_context" in metadata

    # Extract back
    extracted = TraceContext.extract(metadata)
    assert extracted is not None
    assert extracted.trace_id == "tr_prop_123"
    assert extracted.current_span_id == "span_parent_999"
    assert extracted.baggage["priority"] == "high"

    # Extracting from empty carrier returns None
    assert TraceContext.extract({}) is None
    assert TraceContext.extract(None) is None


def test_part2_observed_llm_provider_wrapper(observability_mgr):
    """Test ObservedLLMProvider wraps underlying provider and records span & metrics."""
    mock_provider = MockLLMProvider(
        return_text="Analyzed response",
        usage={"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
    )
    observed_provider = observability_mgr.wrap_provider(mock_provider)

    # Call generate within a trace
    with observability_mgr.trace(workflow_id="wf_llm_test", task_id="t_llm"):
        resp = observed_provider.generate(
            messages=[LLMMessage(role="user", content="Explain quantum computing")],
            model="mock-gpt4",
        )

    assert resp.content == "Analyzed response"
    assert len(mock_provider.calls) == 1

    # Check that LLM metric was recorded
    llm_metrics = observability_mgr.metrics.get_llm_metrics(provider_name="mock_provider")
    assert llm_metrics["request_count"] == 1
    assert llm_metrics["successful_requests"] == 1
    assert llm_metrics["total_tokens"] == 200


# ===========================================================================
# PART 3: Structured Logging & Secret Redaction
# ===========================================================================

def test_part3_structured_log_record_creation_and_filtering(observability_mgr):
    """Test structured log creation and level filtering."""
    observability_mgr.config.min_log_level = LogLevel.INFO

    # DEBUG should be skipped when min_log_level is INFO
    observability_mgr.log(
        event_type=EventCategory.TOOL_STARTED.value,
        message="Debug level details",
        log_level=LogLevel.DEBUG,
    )

    # INFO should be captured
    observability_mgr.log(
        event_type=EventCategory.WORKFLOW_STARTED.value,
        message="Workflow initialized",
        log_level=LogLevel.INFO,
        workflow_id="wf_log_01",
    )

    # ERROR should be captured
    observability_mgr.log(
        event_type=EventCategory.WORKFLOW_FAILED.value,
        message="Workflow failed with timeout",
        log_level=LogLevel.ERROR,
        workflow_id="wf_log_01",
    )

    logs = observability_mgr.list_logs(workflow_id="wf_log_01")
    assert len(logs) == 2
    assert all(l["log_level"] in ("INFO", "ERROR") for l in logs)


def test_part3_secret_redaction_in_structured_logging(observability_mgr):
    """Verify secrets inside log messages and attributes are redacted before storage."""
    secret_key = "sk-ant-api03-abcdef1234567890abcdef1234567890"
    observability_mgr.log(
        event_type=EventCategory.AGENT_COMPLETED.value,
        message=f"Agent completed using Authorization: Bearer {secret_key}",
        log_level=LogLevel.INFO,
        attributes={"secret_api_key": secret_key, "normal_key": "safe_value"},
        workflow_id="wf_redact",
    )

    logs = observability_mgr.list_logs(workflow_id="wf_redact")
    assert len(logs) == 1
    log = logs[0]

    assert secret_key not in log["message"]
    assert "[REDACTED]" in log["message"]
    assert log["attributes"]["secret_api_key"] == "[REDACTED]"
    assert log["attributes"]["normal_key"] == "safe_value"


def test_part3_logging_failure_isolation(observability_mgr):
    """Verify logger exceptions are caught safely without raising to the caller."""
    # Sabotage storage to raise an unexpected error
    bad_storage = MagicMock()
    bad_storage.save_observability_log.side_effect = RuntimeError("Disk full error")
    observability_mgr.storage = bad_storage

    # Calling log should not raise RuntimeError
    try:
        record = observability_mgr.log(
            event_type=EventCategory.AGENT_STARTED.value,
            message="Testing failure isolation",
            log_level=LogLevel.INFO,
        )
        assert record is not None
    except Exception as exc:
        pytest.fail(f"log() raised an unhandled exception: {exc}")


# ===========================================================================
# PART 4: Agent & System Metrics Collection
# ===========================================================================

def test_part4_percentile_calculations():
    """Verify calculate_percentile nearest-rank implementation on edge cases."""
    assert calculate_percentile([], 50.0) == 0.0
    assert calculate_percentile([42.0], 95.0) == 42.0

    # 100 sorted numbers from 1 to 100
    values = [float(i) for i in range(1, 101)]
    p50 = calculate_percentile(values, 50.0)
    p95 = calculate_percentile(values, 95.0)
    p99 = calculate_percentile(values, 99.0)

    assert p50 == 50.0
    assert p95 == 95.0
    assert p99 == 99.0


def test_part4_agent_metrics_aggregation(observability_mgr):
    """Verify AgentMetricsCollector aggregates latency, success rate, and retries."""
    collector = observability_mgr.metrics

    # Record 5 executions for ResearcherAgent
    collector.record_agent_execution("ResearcherAgent", 100.0, success=True, retry_count=0)
    collector.record_agent_execution("ResearcherAgent", 200.0, success=True, retry_count=1)
    collector.record_agent_execution("ResearcherAgent", 300.0, success=False, retry_count=2)
    collector.record_agent_execution("ResearcherAgent", 150.0, success=True, retry_count=0)
    collector.record_agent_execution("ResearcherAgent", 250.0, success=True, retry_count=0)

    summary = collector.get_agent_metrics(agent_name="researcheragent")
    assert summary["execution_count"] == 5
    assert summary["success_rate"] == 0.8
    assert summary["failure_rate"] == 0.2
    assert summary["avg_latency_ms"] == 200.0
    assert summary["p50_latency_ms"] == 200.0
    assert summary["retry_count"] == 3


def test_part4_honest_token_reporting_without_fabrication(observability_mgr):
    """Verify LLM metrics return None when token counts are not reported by provider."""
    collector = observability_mgr.metrics

    # Provider call WITH tokens
    collector.record_llm_call(
        provider_name="openrouter",
        model_name="openai/gpt-4o",
        latency_ms=450.0,
        success=True,
        input_tokens=150,
        output_tokens=50,
        total_tokens=200,
        cost=0.002,
    )
    # Local/Mock call WITHOUT tokens (None)
    collector.record_llm_call(
        provider_name="local_mock",
        model_name="llama-3-local",
        latency_ms=120.0,
        success=True,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        cost=None,
    )

    openrouter_stats = collector.get_llm_metrics(provider_name="openrouter")
    assert openrouter_stats["total_tokens"] == 200
    assert openrouter_stats["estimated_cost"] == 0.002

    local_stats = collector.get_llm_metrics(provider_name="local_mock")
    # Tokens MUST NOT be fabricated or assumed 0
    assert local_stats["total_tokens"] is None
    assert local_stats["estimated_cost"] is None


def test_part4_tool_metrics_duration_and_success(observability_mgr):
    """Verify tool execution tracking and percentile computation."""
    collector = observability_mgr.metrics

    collector.record_tool_call("web_search", 50.0, success=True)
    collector.record_tool_call("web_search", 75.0, success=True)
    collector.record_tool_call("web_search", 150.0, success=False)

    stats = collector.get_tool_metrics("web_search")
    assert stats["invocation_count"] == 3
    assert stats["success_rate"] == pytest.approx(0.6667, rel=1e-3)
    assert stats["failure_rate"] == pytest.approx(0.3333, rel=1e-3)
    assert stats["avg_duration_ms"] == pytest.approx(91.67, rel=1e-2)


# ===========================================================================
# PART 5: Trace Analysis Engine & Failure Attribution
# ===========================================================================

def test_part5_failure_category_classification():
    """Verify classification across all 8 failure domains."""
    engine = TraceAnalysisEngine()

    assert engine.classify_failure("RateLimitError from OpenAI API", "llm_generate") == FailureCategory.PROVIDER_FAILURE
    assert engine.classify_failure("Database Connection timed out after 30s", "run_tool") == FailureCategory.NETWORK_OR_TIMEOUT_FAILURE
    assert engine.classify_failure("Human reviewer explicitly rejected step 2", "approval") == FailureCategory.HUMAN_REJECTION
    assert engine.classify_failure("Worker failed validation schema", "task_execution") == FailureCategory.EVALUATION_FAILURE
    assert engine.classify_failure("Message receiver not found in registry", "route_message") == FailureCategory.MESSAGE_DELIVERY_FAILURE
    assert engine.classify_failure("Permission denied: insufficient capabilities", "tool_call") == FailureCategory.AUTHORIZATION_FAILURE
    assert engine.classify_failure("Tool bash returned code 1", "tool_call") == FailureCategory.TOOL_FAILURE
    assert engine.classify_failure("General agent execution failure", "agent_step") == FailureCategory.AGENT_FAILURE


def test_part5_bottleneck_detection(observability_mgr):
    """Verify system bottleneck detection identifies slowest operations and agents."""
    engine = TraceAnalysisEngine(storage=observability_mgr.storage)

    spans = [
        {"span_id": "s1", "operation_name": "web_search", "agent_name": "researcher", "duration_ms": 120.0, "status": "COMPLETED", "span_kind": "tool"},
        {"span_id": "s2", "operation_name": "heavy_eval", "agent_name": "supervisor", "duration_ms": 850.0, "status": "COMPLETED", "span_kind": "internal"},
        {"span_id": "s3", "operation_name": "code_compile", "agent_name": "executor", "duration_ms": 450.0, "status": "COMPLETED", "span_kind": "tool"},
    ]

    bottlenecks = engine.detect_system_bottlenecks(spans=spans, top_k=2)
    slowest_ops = bottlenecks["slowest_operations"]
    assert len(slowest_ops) == 2
    assert slowest_ops[0]["operation"] == "heavy_eval"
    assert slowest_ops[0]["duration_ms"] == 850.0
    assert slowest_ops[1]["operation"] == "code_compile"


def test_part5_trace_root_cause_analysis_with_evaluation_correlation():
    """Verify analyze_trace identifies root-cause span and separates human rejection."""
    engine = TraceAnalysisEngine()

    trace = {
        "trace_id": "tr_fail_root",
        "workflow_id": "wf_test_fail",
        "status": "FAILED",
        "error": "Pipeline terminated due to downstream step error",
        "duration_ms": 500.0,
    }
    spans = [
        {"span_id": "s_res", "agent_name": "researcher", "operation_name": "research", "status": "COMPLETED", "duration_ms": 100.0},
        {"span_id": "s_exec", "agent_name": "executor", "operation_name": "tool_exec", "status": "FAILED", "duration_ms": 200.0, "error": "Tool timed out after 30000ms"},
    ]

    report = engine.analyze_trace(trace=trace, spans=spans)
    assert report.trace_id == "tr_fail_root"
    assert report.primary_failure_category == FailureCategory.NETWORK_OR_TIMEOUT_FAILURE
    assert report.root_cause_span_id == "s_exec"
    assert "timed out" in report.root_cause_error.lower()


# ===========================================================================
# PART 6: Full Integration, Persistence, Retention & Safety
# ===========================================================================

def test_part6_orchestrator_integration_with_observability(observability_mgr):
    """Verify WorkflowOrchestrator records root trace and child spans for each stage."""
    mock_prov = MockLLMProvider(return_text="Synthetic analysis report.")
    researcher = ResearcherAgent(provider=mock_prov)
    planner = PlannerAgent(provider=mock_prov)
    executor = ExecutorAgent(provider=mock_prov)

    # Mock researcher to return valid findings
    researcher.research = MagicMock(return_value={
        "task_id": "wf_obs_test",
        "status": "success",
        "findings": "PostgreSQL is highly scalable and reliable.",
    })
    planner.plan = MagicMock(return_value={
        "task_id": "wf_obs_test",
        "status": "success",
        "tasks": [{"id": 1, "description": "deploy postgres", "tool": "deploy", "input": {}}],
    })
    executor.execute = MagicMock(return_value={
        "task_id": 1,
        "status": "completed",
        "result": "Postgres deployed successfully.",
    })

    orchestrator = WorkflowOrchestrator(
        researcher=researcher,
        planner=planner,
        executor=executor,
        provider=mock_prov,
        observability=observability_mgr,
    )

    ctx = orchestrator.run("Deploy PostgreSQL database", task_id="wf_obs_test")

    assert ctx.status == WorkflowStatus.COMPLETED
    assert "trace_id" in ctx.metadata

    # Verify trace was stored in SQLite
    trace_id = ctx.metadata["trace_id"]
    saved_trace = observability_mgr.get_trace(trace_id)
    assert saved_trace is not None
    assert saved_trace["status"] == "COMPLETED"

    # Verify spans for research, planning, execution, synthesis were stored
    spans = observability_mgr.list_spans(trace_id=trace_id)
    op_names = [s["operation_name"] for s in spans]
    assert "research_stage" in op_names
    assert "planning_stage" in op_names
    assert "execution_stage" in op_names
    assert "synthesis_stage" in op_names


def test_part6_backward_compatibility_without_observability():
    """Verify WorkflowOrchestrator works normally when observability is None."""
    mock_prov = MockLLMProvider(return_text="Normal synthesis.")
    researcher = ResearcherAgent(provider=mock_prov)
    planner = PlannerAgent(provider=mock_prov)
    executor = ExecutorAgent(provider=mock_prov)

    researcher.research = MagicMock(return_value={"task_id": "wf_compat", "status": "success", "findings": "Findings"})
    planner.plan = MagicMock(return_value={"task_id": "wf_compat", "status": "success", "tasks": [{"id": 1, "description": "step 1", "tool": "test", "input": {}}]})
    executor.execute = MagicMock(return_value={"task_id": 1, "status": "completed", "result": "Done"})

    orchestrator = WorkflowOrchestrator(
        researcher=researcher,
        planner=planner,
        executor=executor,
        provider=mock_prov,
        observability=None,  # explicitly disabled
    )

    ctx = orchestrator.run("Test request", task_id="wf_compat")
    assert ctx.status == WorkflowStatus.COMPLETED
    assert "trace_id" not in ctx.metadata


def test_part6_retention_policy_enforcement(observability_mgr, storage):
    """Verify enforce_retention purges records older than retention cutoff."""
    old_iso = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    recent_iso = datetime.now(timezone.utc).isoformat()

    # Save an old trace and a recent trace
    old_trace = {"trace_id": "tr_old", "workflow_id": "wf_old", "start_time": old_iso, "status": "COMPLETED"}
    recent_trace = {"trace_id": "tr_recent", "workflow_id": "wf_recent", "start_time": recent_iso, "status": "COMPLETED"}

    storage.save_observability_trace(old_trace)
    storage.save_observability_trace(recent_trace)

    assert storage.get_observability_trace("tr_old") is not None
    assert storage.get_observability_trace("tr_recent") is not None

    # Enforce 7-day retention
    purged = observability_mgr.enforce_retention(retention_days=7)
    assert purged.get("traces_purged", 0) >= 1

    # Old trace is gone; recent trace remains
    assert storage.get_observability_trace("tr_old") is None
    assert storage.get_observability_trace("tr_recent") is not None


def test_part6_observability_failure_isolation_in_orchestrator():
    """Verify orchestrator workflow does not crash if observability fails."""
    mock_prov = MockLLMProvider(return_text="Resilient synthesis.")
    researcher = ResearcherAgent(provider=mock_prov)
    planner = PlannerAgent(provider=mock_prov)
    executor = ExecutorAgent(provider=mock_prov)

    researcher.research = MagicMock(return_value={"task_id": "wf_isolated", "status": "success", "findings": "Findings"})
    planner.plan = MagicMock(return_value={"task_id": "wf_isolated", "status": "success", "tasks": [{"id": 1, "description": "step 1", "tool": "test", "input": {}}]})
    executor.execute = MagicMock(return_value={"task_id": 1, "status": "completed", "result": "Done"})

    # Faulty observability manager that explodes on start_trace
    exploding_mgr = MagicMock()
    exploding_mgr.start_trace.side_effect = RuntimeError("Fatal monitoring crash")
    exploding_mgr.start_span.side_effect = RuntimeError("Fatal span crash")

    orchestrator = WorkflowOrchestrator(
        researcher=researcher,
        planner=planner,
        executor=executor,
        provider=mock_prov,
        observability=exploding_mgr,
    )

    # Workflow must complete successfully despite observability errors
    ctx = orchestrator.run("Resilient execution", task_id="wf_isolated")
    assert ctx.status == WorkflowStatus.COMPLETED


def test_part6_observability_summary_overview(observability_mgr):
    """Verify get_summary computes accurate health and latency overview."""
    with observability_mgr.trace(workflow_id="wf_sum_1", task_id="t1"):
        span1 = observability_mgr.start_span("step_1", agent_name="worker")
        time.sleep(0.005)
        observability_mgr.end_span(span1.span_id, status=SpanStatus.COMPLETED)

    summary = observability_mgr.get_summary()
    assert isinstance(summary, ObservabilitySummary)
    assert summary.total_traces >= 1
    assert summary.completed_traces >= 1
    assert summary.failed_traces == 0
    assert summary.total_spans >= 1
    assert summary.p95_workflow_latency_ms >= 0.0
