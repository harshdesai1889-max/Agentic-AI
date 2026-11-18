"""
tests/test_day229_observability_remediation.py

Comprehensive Verification & Hardening Suite for Day 229: Observability Remediation.
Organized into 5 rigorous test groups:
- Group 1: Asynchronous Trace Context Propagation (ContextVar, isolation, nested spans, restoration)
- Group 2: Real Provider and Tool Instrumentation (ObservedLLMProvider, execute_tool, MCP tools, no duplicate execution, redaction)
- Group 3: Complete HITL Lifecycle Observability (10 lifecycle events, pause/resume, token redaction, audit preservation)
- Group 4: Metrics Completeness and Accuracy (Agent latencies P50/P95/P99, honest tokens, tool breakdown, workflow metrics)
- Group 5: Retention, Trace Integrity and Data Consistency (Active trace preservation, cascading cleanup, dry-run, rollback)
"""

import asyncio
import os
import sqlite3
import tempfile
import time
import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from core.observability import (
    AgentObservabilityManager,
    EventCategory,
    LogLevel,
    ObservedLLMProvider,
    ObservabilityConfig,
    SpanKind,
    SpanStatus,
    TraceContext,
    _current_trace_context,
    get_active_observability_manager,
    set_active_observability_manager,
    inject_context,
    extract_context,
    instrument_tool_execution,
    redact_observability_data,
)
from core.persistent_storage import SQLiteMessageStorage
from core.provider_interface import LLMMessage, LLMProvider, LLMResponse
from tools.executor import execute_tool
from mcp_servers.basic_tools_server import _exec_mcp_tool
from tools.basic_tools import calculator, get_datetime, word_counter
from core.hitl import (
    HumanInTheLoopEngine,
    HumanApprovalManager,
    HumanInteractionManager,
    HumanFeedbackManager,
    HumanReviewManager,
    InteractionType,
    ApprovalDecision,
    ReviewOutcome,
)
from core.orchestrator import TaskContext, WorkflowStatus


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.fixture
def storage(temp_db_path):
    """Provide a clean SQLiteMessageStorage instance."""
    st = SQLiteMessageStorage(db_path=temp_db_path)
    return st


@pytest.fixture
def obs_manager(storage):
    """Provide an AgentObservabilityManager instance backed by storage."""
    mgr = AgentObservabilityManager(storage=storage)
    set_active_observability_manager(mgr)
    yield mgr
    set_active_observability_manager(None)
    _current_trace_context.set(None)


class MockLLMProvider(LLMProvider):
    """Mock LLM provider supporting token tracking and attribute delegation."""
    def __init__(self, provider_name: str = "mock_provider", report_tokens: bool = True):
        self.provider_name = provider_name
        self.report_tokens = report_tokens
        self.call_count = 0
        self.custom_attr = "verified_attr_123"

    def generate(
        self,
        messages: List[LLMMessage],
        model: str,
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        self.call_count += 1
        usage = None
        if self.report_tokens:
            usage = {
                "prompt_tokens": 120,
                "completion_tokens": 45,
                "total_tokens": 165,
                "estimated_cost": 0.00045,
            }
        return LLMResponse(
            content=f"Mock response from {model} for call {self.call_count}",
            tool_calls=[],
            raw_response={"status": "ok"},
            usage=usage,
        )


# ===========================================================================
# GROUP 1: ASYNCHRONOUS TRACE CONTEXT PROPAGATION
# ===========================================================================

@pytest.mark.asyncio
async def test_remediation_group1_contextvar_async_task_isolation(obs_manager):
    """Verify ContextVar isolates ambient trace context across concurrent asyncio tasks."""
    results = {}

    async def worker(task_name: str, delay: float):
        trace = obs_manager.start_trace(workflow_id=f"wf_{task_name}", task_id=task_name)
        await asyncio.sleep(delay)
        ctx = obs_manager.get_current_context()
        assert ctx is not None
        assert ctx.workflow_id == f"wf_{task_name}"
        assert ctx.task_id == task_name
        obs_manager.end_trace(trace.trace_id, status=SpanStatus.COMPLETED)
        results[task_name] = ctx.trace_id

    # Run 3 concurrent async workers with different delays
    await asyncio.gather(
        worker("task_alpha", 0.05),
        worker("task_beta", 0.02),
        worker("task_gamma", 0.03),
    )

    assert len(results) == 3
    # Ensure all trace IDs are distinct (no cross-task contamination)
    assert len(set(results.values())) == 3


@pytest.mark.asyncio
async def test_remediation_group1_child_span_references_and_restoration(obs_manager):
    """Verify child span references parent span and restores outer context on exit."""
    async with obs_manager.async_trace(workflow_id="wf_parent_child") as trace:
        root_ctx = obs_manager.get_current_context()
        assert root_ctx is not None
        assert root_ctx.current_span_id == trace.root_span_id

        async with obs_manager.async_span("operation_outer", agent_name="agent_a") as outer_span:
            outer_ctx = obs_manager.get_current_context()
            assert outer_ctx.current_span_id == outer_span.span_id
            assert outer_span.parent_span_id == trace.root_span_id

            async with obs_manager.async_span("operation_inner", agent_name="agent_b") as inner_span:
                inner_ctx = obs_manager.get_current_context()
                assert inner_ctx.current_span_id == inner_span.span_id
                assert inner_span.parent_span_id == outer_span.span_id

            # After inner span finishes, context pops back to outer span
            restored_outer_ctx = obs_manager.get_current_context()
            assert restored_outer_ctx.current_span_id == outer_span.span_id

        # After outer span finishes, context pops back to root trace
        restored_root_ctx = obs_manager.get_current_context()
        assert restored_root_ctx.current_span_id == trace.root_span_id


@pytest.mark.asyncio
async def test_remediation_group1_nested_async_spans_with_exception_safety(obs_manager):
    """Verify nested spans restore context even when an inner span raises an exception."""
    async with obs_manager.async_trace(workflow_id="wf_exception_test") as trace:
        async with obs_manager.async_span("outer_safe_op", agent_name="safe_agent") as outer_span:
            with pytest.raises(ZeroDivisionError):
                async with obs_manager.async_span("inner_failing_op", agent_name="failing_agent") as inner_span:
                    _ = 1 / 0

            # Outer span context must be reliably restored despite the exception
            curr_ctx = obs_manager.get_current_context()
            assert curr_ctx is not None
            assert curr_ctx.current_span_id == outer_span.span_id

    # Verify span statuses in storage
    spans = obs_manager.storage.list_observability_spans(trace_id=trace.trace_id)
    failing_spans = [s for s in spans if s["operation_name"] == "inner_failing_op"]
    assert len(failing_spans) == 1
    assert failing_spans[0]["status"] == "FAILED"


def test_remediation_group1_queue_and_message_trace_propagation(obs_manager):
    """Verify trace context injection and extraction across queue message carriers."""
    trace = obs_manager.start_trace(workflow_id="wf_msg_queue", task_id="task_queue_1")
    span = obs_manager.start_span("enqueue_job", agent_name="dispatcher")

    ctx = obs_manager.get_current_context()
    assert ctx is not None

    # Simulate message metadata carrier
    carrier: Dict[str, Any] = {"payload": "process_order"}
    inject_context(ctx, carrier)

    assert "trace_context" in carrier
    assert carrier["trace_id"] == trace.trace_id

    # Simulate deserialization at receiver end
    extracted = extract_context(carrier)
    assert extracted is not None
    assert extracted.trace_id == trace.trace_id
    assert extracted.workflow_id == "wf_msg_queue"
    assert extracted.task_id == "task_queue_1"
    assert extracted.current_span_id == span.span_id

    obs_manager.end_span(span.span_id)
    obs_manager.end_trace(trace.trace_id)


# ===========================================================================
# GROUP 2: REAL PROVIDER AND TOOL INSTRUMENTATION
# ===========================================================================

def test_remediation_group2_observed_llm_provider_delegation_and_generation(obs_manager):
    """Verify ObservedLLMProvider transparent delegation, single invocation, and metrics."""
    mock = MockLLMProvider(provider_name="test_openrouter", report_tokens=True)
    wrapped = obs_manager.wrap_provider(mock)

    # 1. Verify attribute transparent delegation
    assert wrapped.custom_attr == "verified_attr_123"
    assert wrapped.provider_name == "test_openrouter"

    # 2. Call generate through wrapper
    trace = obs_manager.start_trace(workflow_id="wf_provider_test")
    messages = [LLMMessage(role="user", content="Hello LLM")]
    resp = wrapped.generate(messages=messages, model="openai/gpt-4o")

    assert "Mock response from openai/gpt-4o" in resp.content
    assert mock.call_count == 1  # Exactly ONE execution (no duplicates)

    obs_manager.end_trace(trace.trace_id)

    # 3. Verify LLM span was captured in storage
    spans = obs_manager.storage.list_observability_spans(trace_id=trace.trace_id)
    llm_spans = [s for s in spans if s["span_kind"] == "llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["attributes"]["model"] == "openai/gpt-4o"
    assert llm_spans[0]["attributes"]["total_tokens"] == 165

    # 4. Verify LLM metrics
    metrics = obs_manager.metrics.get_llm_metrics(provider_name="test_openrouter")
    assert metrics["request_count"] == 1
    assert metrics["total_tokens"] == 165


def test_remediation_group2_tool_execution_boundary_instrumentation(obs_manager):
    """Verify tools/executor.py boundary instruments tool calls with single execution."""
    trace = obs_manager.start_trace(workflow_id="wf_tool_boundary")

    # 1. Calculator tool call
    calc_call = {
        "function": {
            "name": "calculator",
            "arguments": '{"expression": "40 + 2"}',
        }
    }
    calc_res = execute_tool(calc_call, observability=obs_manager)
    assert calc_res == {"result": 42}

    # 2. Word counter tool call
    word_call = {
        "function": {
            "name": "word_counter",
            "arguments": '{"text": "Observability and tracing"}',
        }
    }
    word_res = execute_tool(word_call, observability=obs_manager)
    assert word_res["word_count"] == 3

    obs_manager.end_trace(trace.trace_id)

    # 3. Verify TOOL spans in storage
    spans = obs_manager.storage.list_observability_spans(trace_id=trace.trace_id)
    tool_spans = [s for s in spans if s["span_kind"] == "tool"]
    assert len(tool_spans) == 2

    # 4. Verify tool metrics
    calc_metrics = obs_manager.metrics.get_tool_metrics(tool_name="calculator")
    assert calc_metrics["invocation_count"] == 1
    assert calc_metrics["success_rate"] == 1.0


def test_remediation_group2_mcp_tool_execution_instrumentation(obs_manager):
    """Verify MCP tools are instrumented with TOOL spans."""
    trace = obs_manager.start_trace(workflow_id="wf_mcp_tools")

    res = _exec_mcp_tool(calculator, "calculator", {"expression": "100 / 4"})
    assert res == {"result": 25.0}

    dt_res = _exec_mcp_tool(get_datetime, "get_datetime", {})
    assert "datetime" in dt_res

    obs_manager.end_trace(trace.trace_id)

    spans = obs_manager.storage.list_observability_spans(trace_id=trace.trace_id)
    mcp_spans = [s for s in spans if s["agent_name"] == "mcp_server"]
    assert len(mcp_spans) == 2


def test_remediation_group2_secret_sanitization_in_tool_and_provider_attributes(obs_manager):
    """Verify secrets and API keys are scrubbed before being recorded in span attributes."""
    dirty_args = {
        "api_key": "sk-proj-supersecretkey1234567890",
        "authorization": "Bearer ya29.a0AfH6SMCxyz123456789",
        "normal_field": "public_data",
    }
    clean = redact_observability_data(dirty_args)
    assert clean["api_key"] == "[REDACTED]"
    assert clean["authorization"] == "[REDACTED]"
    assert clean["normal_field"] == "public_data"
    assert "sk-proj" not in str(clean)
    assert "ya29" not in str(clean)


# ===========================================================================
# GROUP 3: COMPLETE HITL LIFECYCLE OBSERVABILITY
# ===========================================================================

def test_remediation_group3_approval_lifecycle_events_and_token_redaction(obs_manager, storage):
    """Verify all 5 approval lifecycle events are traced with approval token redaction."""
    hitl = HumanInTheLoopEngine(storage=storage, observability=obs_manager)
    trace = obs_manager.start_trace(workflow_id="wf_hitl_approval", task_id="task_hitl_1")

    # 1. HUMAN_APPROVAL_REQUESTED
    req = hitl.approvals.request_approval(
        task_id="task_hitl_1",
        correlation_id="corr_1",
        requesting_agent="ExecutorAgent",
        requested_action="rm -rf /tmp/test",
    )
    assert req.interaction_id is not None

    # 2. HUMAN_APPROVAL_GRANTED
    appr = hitl.approvals.approve(
        interaction_id=req.interaction_id,
        feedback="Proceed with caution",
    )
    assert appr.token.startswith("tok_appr_")

    # 3. Create another request and REJECT it -> HUMAN_APPROVAL_REJECTED
    req2 = hitl.approvals.request_approval(
        task_id="task_hitl_1",
        correlation_id="corr_2",
        requesting_agent="ExecutorAgent",
        requested_action="drop_table_prod",
    )
    hitl.approvals.reject(
        interaction_id=req2.interaction_id,
        reason="Action too dangerous",
    )

    # 4. CANCEL an approval -> HUMAN_APPROVAL_CANCELLED
    req3 = hitl.approvals.request_approval(
        task_id="task_hitl_1",
        correlation_id="corr_3",
        requesting_agent="ExecutorAgent",
        requested_action="restart_cluster",
    )
    hitl.approvals.cancel_approval(
        interaction_id=req3.interaction_id,
        reason="Operator cancelled request",
    )

    # 5. EXPIRE an approval -> HUMAN_APPROVAL_EXPIRED
    req4 = hitl.approvals.request_approval(
        task_id="task_hitl_1",
        correlation_id="corr_4",
        requesting_agent="ExecutorAgent",
        requested_action="wipe_cache",
    )
    hitl.approvals.expire_approval(interaction_id=req4.interaction_id)

    obs_manager.end_trace(trace.trace_id)

    # Inspect events saved in SQLite
    events = storage.list_observability_events()
    event_types = [e["event_type"] for e in events]

    assert "human_approval_requested" in event_types
    assert "human_approval_granted" in event_types
    assert "human_approval_rejected" in event_types
    assert "human_approval_cancelled" in event_types
    assert "human_approval_expired" in event_types

    # Verify raw token is NEVER logged in cleartext
    for ev in events:
        payload_str = str(ev.get("payload", {}))
        assert appr.token not in payload_str


def test_remediation_group3_feedback_clarification_and_pause_resume_events(obs_manager, storage):
    """Verify clarification and pause/resume lifecycle events are emitted."""
    hitl = HumanInTheLoopEngine(storage=storage, observability=obs_manager)
    trace = obs_manager.start_trace(workflow_id="wf_clarify", task_id="task_clarify_1")

    context = TaskContext(workflow_id="wf_clarify", task_id="task_clarify_1", user_request="Build pipeline")

    # 1. HUMAN_CLARIFICATION_REQUESTED & WORKFLOW_PAUSED
    req = hitl.feedback.request_clarification(
        task_id="task_clarify_1",
        correlation_id="corr_clarify",
        requesting_agent="PlannerAgent",
        question="Which database backend should we configure?",
        context=context,
    )
    assert context.status == WorkflowStatus.PAUSED

    # 2. HUMAN_CLARIFICATION_RECEIVED
    success, msg, _ = hitl.feedback.submit_feedback(
        interaction_id=req.interaction_id,
        response_text="Use PostgreSQL 16 with TimescaleDB.",
        task_id="task_clarify_1",
        context=context,
    )
    assert success is True

    # 3. WORKFLOW_RESUMED
    class DummyOrchestrator:
        def run(self, user_request, task_id):
            return context

    resumed_ctx = hitl.feedback.resume_workflow(
        workflow_id_or_context=context,
        orchestrator=DummyOrchestrator(),
        feedback_response="PostgreSQL 16",
    )
    assert resumed_ctx.status == WorkflowStatus.INITIALIZED

    obs_manager.end_trace(trace.trace_id)

    events = storage.list_observability_events()
    ev_types = [e["event_type"] for e in events]
    assert "human_clarification_requested" in ev_types
    assert "workflow_paused" in ev_types
    assert "human_clarification_received" in ev_types
    assert "workflow_resumed" in ev_types


def test_remediation_group3_review_lifecycle_and_audit_preservation(obs_manager, storage):
    """Verify HUMAN_REVIEW_COMPLETED event is recorded and audit trail persists."""
    hitl = HumanInTheLoopEngine(storage=storage, observability=obs_manager)
    trace = obs_manager.start_trace(workflow_id="wf_review", task_id="task_rev_1")

    rev = hitl.reviews.submit_review(
        task_id="task_rev_1",
        target_type="plan",
        target_data={"steps": ["research", "execute"]},
        reviewer="senior_architect",
        outcome=ReviewOutcome.ACCEPTED,
        feedback="Looks solid and compliant.",
    )
    assert rev.review_id is not None

    obs_manager.end_trace(trace.trace_id)

    # 1. Verify observability event
    events = storage.list_observability_events()
    ev_types = [e["event_type"] for e in events]
    assert "human_review_completed" in ev_types

    # 2. Verify audit trail integrity
    audit_events = hitl.audit.query(task_id="task_rev_1")
    assert len(audit_events) >= 1
    assert any("senior_architect" in a.description for a in audit_events)


# ===========================================================================
# GROUP 4: METRICS COMPLETENESS AND ACCURACY
# ===========================================================================

def test_remediation_group4_agent_metrics_accuracy_and_percentiles(obs_manager):
    """Verify agent metrics calculation: success/failure counts, rates, and P50/P95/P99 latencies."""
    # Record 10 agent executions with known latencies: 10, 20, 30, ..., 100
    for i in range(1, 11):
        obs_manager.metrics.record_agent_execution(
            agent_name="ResearcherAgent",
            duration_ms=float(i * 10),
            success=(i <= 8),  # 8 successes, 2 failures
            task_type="research",
        )

    metrics = obs_manager.metrics.get_agent_metrics(agent_name="ResearcherAgent")
    assert metrics["execution_count"] == 10
    assert metrics["success_count"] == 8
    assert metrics["failure_count"] == 2
    assert metrics["success_rate"] == 0.8
    assert metrics["failure_rate"] == 0.2
    assert metrics["avg_latency_ms"] == 55.0
    assert metrics["p50_latency_ms"] == 50.0
    assert metrics["p95_latency_ms"] == 100.0
    assert metrics["p99_latency_ms"] == 100.0


def test_remediation_group4_provider_metrics_honest_token_tracking(obs_manager):
    """Verify honest token usage tracking (no fabricated numbers when absent)."""
    # 1. Provider reporting tokens
    obs_manager.metrics.record_llm_call(
        provider_name="gemini_provider",
        model_name="gemini-2.0-flash",
        latency_ms=120.0,
        success=True,
        input_tokens=250,
        output_tokens=75,
        total_tokens=325,
        cost=0.000325,
    )

    # 2. Provider with missing tokens (e.g. streaming or unmetered endpoint)
    obs_manager.metrics.record_llm_call(
        provider_name="local_mock",
        model_name="llama-3-8b",
        latency_ms=85.0,
        success=True,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        cost=None,
    )

    gemini_metrics = obs_manager.metrics.get_llm_metrics(provider_name="gemini_provider")
    assert gemini_metrics["total_tokens"] == 325
    assert gemini_metrics["estimated_cost"] == 0.000325

    local_metrics = obs_manager.metrics.get_llm_metrics(provider_name="local_mock")
    # MUST be None, not fabricated 0 or estimated fake value
    assert local_metrics["total_tokens"] is None
    assert local_metrics["estimated_cost"] is None


def test_remediation_group4_tool_and_workflow_metrics_with_filtering(obs_manager):
    """Verify tool and workflow metrics calculation, error breakdown, and filtering."""
    # Record 5 tool calls
    obs_manager.metrics.record_tool_call(tool_name="calculator", duration_ms=10.0, success=True)
    obs_manager.metrics.record_tool_call(tool_name="calculator", duration_ms=15.0, success=True)
    obs_manager.metrics.record_tool_call(tool_name="calculator", duration_ms=30.0, success=False, error_type="ZeroDivisionError")

    tool_metrics = obs_manager.metrics.get_tool_metrics(tool_name="calculator")
    assert tool_metrics["invocation_count"] == 3
    assert tool_metrics["success_count"] == 2
    assert tool_metrics["failure_count"] == 1
    assert tool_metrics["error_breakdown"]["ZeroDivisionError"] == 1

    # Record 2 workflows
    obs_manager.metrics.record_workflow_execution(
        workflow_id="wf_prod_1",
        duration_ms=450.0,
        success=True,
        agent_transitions=3,
        human_interventions=1,
    )
    obs_manager.metrics.record_workflow_execution(
        workflow_id="wf_prod_2",
        duration_ms=200.0,
        success=False,
        agent_transitions=1,
        approval_rejections=1,
    )

    wf_metrics = obs_manager.metrics.get_workflow_metrics()
    assert wf_metrics["total_workflows"] == 2
    assert wf_metrics["completion_rate"] == 0.5
    assert wf_metrics["total_human_interventions"] == 1
    assert wf_metrics["total_approval_rejections"] == 1


# ===========================================================================
# GROUP 5: RETENTION, TRACE INTEGRITY AND DATA CONSISTENCY
# ===========================================================================

def test_remediation_group5_active_trace_preservation_during_cleanup(obs_manager, storage):
    """Verify active traces (status == 'STARTED') and their child records are NEVER deleted."""
    old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    # 1. Insert an ACTIVE trace started 60 days ago (long-running workflow)
    active_trace = {
        "trace_id": "tr_active_longrunning",
        "workflow_id": "wf_active_99",
        "task_id": "task_active_99",
        "status": "STARTED",
        "start_time": old_time,
        "root_span_id": "span_root_active",
        "agent_sequence": [],
        "attributes": {},
        "metadata": {},
    }
    storage.save_observability_trace(active_trace)

    active_span = {
        "span_id": "span_active_child",
        "trace_id": "tr_active_longrunning",
        "parent_span_id": "span_root_active",
        "workflow_id": "wf_active_99",
        "task_id": "task_active_99",
        "agent_name": "SupervisorAgent",
        "operation_name": "monitoring",
        "span_kind": "agent",
        "start_time": old_time,
        "status": "STARTED",
        "attributes": {},
        "metadata": {},
    }
    storage.save_observability_span(active_span)

    # 2. Insert a COMPLETED trace started 60 days ago
    completed_trace = {
        "trace_id": "tr_completed_old",
        "workflow_id": "wf_completed_old",
        "task_id": "task_completed_old",
        "status": "COMPLETED",
        "start_time": old_time,
        "end_time": old_time,
        "duration_ms": 500.0,
        "root_span_id": "span_root_old",
        "agent_sequence": [],
        "attributes": {},
        "metadata": {},
    }
    storage.save_observability_trace(completed_trace)

    completed_span = {
        "span_id": "span_completed_child",
        "trace_id": "tr_completed_old",
        "parent_span_id": "span_root_old",
        "workflow_id": "wf_completed_old",
        "task_id": "task_completed_old",
        "agent_name": "ExecutorAgent",
        "operation_name": "execute",
        "span_kind": "agent",
        "start_time": old_time,
        "status": "COMPLETED",
        "attributes": {},
        "metadata": {},
    }
    storage.save_observability_span(completed_span)

    # Enforce retention policy with 30-day cutoff
    cleanup_stats = storage.cleanup_observability_records(older_than_iso=recent_cutoff, dry_run=False)

    assert cleanup_stats["traces"] == 1
    assert cleanup_stats["spans"] == 1

    # Verify ACTIVE trace and its span are STILL present
    remaining_traces = storage.list_observability_traces()
    remaining_trace_ids = [t["trace_id"] for t in remaining_traces]
    assert "tr_active_longrunning" in remaining_trace_ids
    assert "tr_completed_old" not in remaining_trace_ids

    remaining_spans = storage.list_observability_spans()
    remaining_span_ids = [s["span_id"] for s in remaining_spans]
    assert "span_active_child" in remaining_span_ids
    assert "span_completed_child" not in remaining_span_ids


def test_remediation_group5_cascading_cleanup_of_eligible_traces(obs_manager, storage):
    """Verify cascading deletion removes child spans, events, and logs of purged traces."""
    old_time = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    trace_id = "tr_cascade_test"
    storage.save_observability_trace({
        "trace_id": trace_id,
        "workflow_id": "wf_cascade",
        "task_id": "task_cascade",
        "status": "FAILED",
        "start_time": old_time,
        "root_span_id": "span_root_c",
        "agent_sequence": [],
        "attributes": {},
        "metadata": {},
    })

    storage.save_observability_span({
        "span_id": "span_child_c",
        "trace_id": trace_id,
        "workflow_id": "wf_cascade",
        "agent_name": "ResearcherAgent",
        "operation_name": "research",
        "start_time": old_time,
        "status": "FAILED",
        "attributes": {},
        "metadata": {},
    })

    storage.save_observability_event({
        "event_id": "ev_child_c",
        "trace_id": trace_id,
        "event_type": "agent_failed",
        "agent_name": "ResearcherAgent",
        "timestamp": old_time,
        "payload": {"error": "test"},
    })

    storage.save_observability_log({
        "log_id": "log_child_c",
        "timestamp": old_time,
        "log_level": "ERROR",
        "event_type": "agent_failed",
        "trace_id": trace_id,
        "workflow_id": "wf_cascade",
        "message": "Cascade log test",
        "attributes": {},
    })

    stats = storage.cleanup_observability_records(older_than_iso=cutoff, dry_run=False)

    assert stats["traces"] == 1
    assert stats["spans"] == 1
    assert stats["events"] == 1
    assert stats["logs"] == 1

    # Confirm no orphaned records remain
    assert len(storage.list_observability_spans(trace_id=trace_id)) == 0
    assert len(storage.list_observability_events(trace_id=trace_id)) == 0
    assert len(storage.list_observability_logs(trace_id=trace_id)) == 0


def test_remediation_group5_retention_dry_run_mode(obs_manager, storage):
    """Verify dry_run=True returns purge preview counts without modifying database."""
    old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    storage.save_observability_trace({
        "trace_id": "tr_dryrun_test",
        "workflow_id": "wf_dryrun",
        "status": "COMPLETED",
        "start_time": old_time,
        "root_span_id": "span_dry",
        "agent_sequence": [],
        "attributes": {},
        "metadata": {},
    })

    # Dry run preview
    preview = storage.cleanup_observability_records(older_than_iso=cutoff, dry_run=True)
    assert preview["dry_run"] is True
    assert preview["traces"] == 1

    # Verify trace is still in database
    traces = storage.list_observability_traces()
    assert any(t["trace_id"] == "tr_dryrun_test" for t in traces)


def test_remediation_group5_transactional_rollback_on_failure(obs_manager, storage):
    """Verify retention cleanup aborts transaction and rolls back when an error occurs."""
    old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    storage.save_observability_trace({
        "trace_id": "tr_rollback_test",
        "workflow_id": "wf_rollback",
        "status": "COMPLETED",
        "start_time": old_time,
        "root_span_id": "span_rb",
        "agent_sequence": [],
        "attributes": {},
        "metadata": {},
    })

    # Tamper with database table to force error during cleanup
    with storage._lock:
        conn = storage._get_connection()
        conn.execute("DROP TABLE observability_metrics;")
        conn.close()

    # Attempt cleanup; should fail and rollback
    with pytest.raises(Exception):
        storage.cleanup_observability_records(older_than_iso=cutoff, dry_run=False)

    # Reconnect and recreate table to inspect state
    with storage._lock:
        conn = storage._get_connection()
        storage._init_db()
        # Verify trace was NOT committed as deleted
        row = conn.execute("SELECT * FROM observability_traces WHERE trace_id = 'tr_rollback_test';").fetchone()
        conn.close()
        assert row is not None
