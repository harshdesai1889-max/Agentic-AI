"""
core/persistent_storage.py

SQLite Persistent Storage and Restart Recovery Subsystem for Day 226 Part 4.

Implements:
1. SQLiteMessageStorage:
   - Built on Python's built-in `sqlite3` module.
   - Complete schema storing: message_id (PK), task_id, correlation_id, sender, receiver,
     message_type, priority, payload_json, state, attempts, error, enqueued_at,
     updated_at, completed_at, and metadata_json.
   - Robust serialization & deserialization to/from AgentMessage and QueueItem.
   - Transactional safety: explicit commits and rollbacks on SQLite errors.
   - Unique message_id constraint enforcing persistent deduplication.
   - Atomic state transitions: PENDING -> PROCESSING -> COMPLETED / FAILED.
   - Restart recovery scanner restoring pending and interrupted processing messages.
   - Non-idempotent operation defense during recovery.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.orchestrator import AgentMessage, MessageType
from core.message_queue import MessagePriority, QueueMessageState, QueueItem
from core.failure_recovery import (
    FailureClassifier,
    FailureType,
    is_non_idempotent_operation,
)
from utils.logging_config import get_logger

logger = get_logger()


# ===========================================================================
# 1. Database Schema
# ===========================================================================

CREATE_MESSAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    correlation_id TEXT,
    sender TEXT NOT NULL,
    receiver TEXT NOT NULL,
    message_type TEXT NOT NULL,
    priority TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    enqueued_at REAL NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    metadata_json TEXT NOT NULL,
    ack_status TEXT NOT NULL DEFAULT 'unacknowledged',
    ack_reason TEXT,
    ack_timestamp TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_state_prio ON messages(state, priority, enqueued_at);
CREATE INDEX IF NOT EXISTS idx_messages_task_id ON messages(task_id);
CREATE INDEX IF NOT EXISTS idx_messages_correlation ON messages(correlation_id);
CREATE INDEX IF NOT EXISTS idx_messages_ack ON messages(ack_status);

CREATE TABLE IF NOT EXISTS dead_letter_queue (
    dlq_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    correlation_id TEXT,
    sender TEXT NOT NULL,
    receiver TEXT NOT NULL,
    reason TEXT NOT NULL,
    retry_count INTEGER NOT NULL,
    entered_at TEXT NOT NULL,
    replayed INTEGER NOT NULL DEFAULT 0,
    replay_count INTEGER NOT NULL DEFAULT 0,
    replay_history_json TEXT NOT NULL DEFAULT '[]',
    failure_details_json TEXT NOT NULL DEFAULT '{}',
    message_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dlq_message_id ON dead_letter_queue(message_id);

-- Day 227 HITL Tables: Human Interactions, Approvals, and Workflow State Preservation
CREATE TABLE IF NOT EXISTS human_interactions (
    interaction_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    correlation_id TEXT,
    requesting_agent TEXT NOT NULL,
    interaction_type TEXT NOT NULL,
    question_or_request TEXT NOT NULL,
    relevant_context_json TEXT NOT NULL DEFAULT '{}',
    available_decisions_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    decision TEXT,
    human_response TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_interactions_task_id ON human_interactions(task_id);
CREATE INDEX IF NOT EXISTS idx_interactions_status ON human_interactions(status);
CREATE INDEX IF NOT EXISTS idx_interactions_corr ON human_interactions(correlation_id);

CREATE TABLE IF NOT EXISTS human_approvals (
    approval_id TEXT PRIMARY KEY,
    interaction_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    correlation_id TEXT,
    requested_action TEXT NOT NULL,
    decision TEXT NOT NULL,
    human_feedback TEXT,
    decision_timestamp TEXT NOT NULL,
    approval_status TEXT NOT NULL,
    expires_at TEXT,
    token TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (interaction_id) REFERENCES human_interactions (interaction_id)
);

CREATE INDEX IF NOT EXISTS idx_approvals_task_id ON human_approvals(task_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON human_approvals(approval_status);
CREATE INDEX IF NOT EXISTS idx_approvals_token ON human_approvals(token);

CREATE TABLE IF NOT EXISTS workflow_states (
    workflow_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    context_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workflow_task_id ON workflow_states(task_id);

-- Day 227 Parts 4-6: Risk Assessments, Human Reviews, Audit Events, Feedback Analyses, Policy Proposals
CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    action TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    risk_score REAL NOT NULL,
    identified_risks_json TEXT NOT NULL DEFAULT '[]',
    required_controls_json TEXT NOT NULL DEFAULT '[]',
    approval_required INTEGER NOT NULL DEFAULT 0,
    assessment_reason TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_risk_task_id ON risk_assessments(task_id);
CREATE INDEX IF NOT EXISTS idx_risk_level ON risk_assessments(risk_level);

CREATE TABLE IF NOT EXISTS human_reviews (
    review_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    correlation_id TEXT,
    target_type TEXT NOT NULL,
    target_data_json TEXT NOT NULL DEFAULT '{}',
    reviewer TEXT NOT NULL,
    outcome TEXT NOT NULL,
    feedback TEXT,
    review_round INTEGER NOT NULL DEFAULT 1,
    timestamp TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_reviews_task_id ON human_reviews(task_id);
CREATE INDEX IF NOT EXISTS idx_reviews_outcome ON human_reviews(outcome);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uuid TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    correlation_id TEXT,
    event_type TEXT NOT NULL,
    agent_name TEXT,
    description TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_task_id ON audit_events(task_id);
CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_events(agent_name);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);

CREATE TABLE IF NOT EXISTS feedback_analyses (
    analysis_id TEXT PRIMARY KEY,
    summary_title TEXT NOT NULL,
    time_window_start TEXT,
    time_window_end TEXT,
    total_reviews INTEGER NOT NULL DEFAULT 0,
    total_feedback_items INTEGER NOT NULL DEFAULT 0,
    recurring_issue_categories_json TEXT NOT NULL DEFAULT '[]',
    agent_failure_rates_json TEXT NOT NULL DEFAULT '{}',
    observed_patterns_json TEXT NOT NULL DEFAULT '[]',
    actionable_recommendations_json TEXT NOT NULL DEFAULT '[]',
    generated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS policy_change_proposals (
    proposal_id TEXT PRIMARY KEY,
    analysis_id TEXT,
    proposed_change TEXT NOT NULL,
    target_policy TEXT NOT NULL,
    rationale TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PROPOSED',
    authorized_by TEXT,
    authorized_at TEXT,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_proposals_status ON policy_change_proposals(status);

-- Day 228 Tables: Agent Evaluations, Task Outcomes, LLM Judge Evaluations, Retrieval Evaluations, Performance Metrics, Revisions
CREATE TABLE IF NOT EXISTS agent_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    evaluation_type TEXT NOT NULL,
    overall_score REAL NOT NULL,
    metric_scores_json TEXT NOT NULL DEFAULT '{}',
    passed INTEGER NOT NULL DEFAULT 0,
    failures_json TEXT NOT NULL DEFAULT '[]',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    evaluation_reason TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_evals_task_id ON agent_evaluations(task_id);
CREATE INDEX IF NOT EXISTS idx_evals_agent_name ON agent_evaluations(agent_name);
CREATE INDEX IF NOT EXISTS idx_evals_passed ON agent_evaluations(passed);

CREATE TABLE IF NOT EXISTS task_outcomes (
    outcome_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    classification TEXT NOT NULL,
    passed_criteria_json TEXT NOT NULL DEFAULT '[]',
    failed_criteria_json TEXT NOT NULL DEFAULT '[]',
    numeric_checks_json TEXT NOT NULL DEFAULT '{}',
    missing_fields_json TEXT NOT NULL DEFAULT '[]',
    evidence_verified INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_task_outcomes_task_id ON task_outcomes(task_id);
CREATE INDEX IF NOT EXISTS idx_task_outcomes_classification ON task_outcomes(classification);

CREATE TABLE IF NOT EXISTS llm_judge_evaluations (
    judge_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    target_stage TEXT NOT NULL,
    accuracy REAL NOT NULL,
    completeness REAL NOT NULL,
    relevance REAL NOT NULL,
    instruction_adherence REAL NOT NULL,
    evidence_support REAL NOT NULL,
    overall_score REAL NOT NULL,
    identified_weaknesses_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL,
    reasoning_summary TEXT NOT NULL,
    evaluator_model TEXT NOT NULL,
    evaluator_provider TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'COMPLETED',
    timestamp TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_judge_task_id ON llm_judge_evaluations(task_id);
CREATE INDEX IF NOT EXISTS idx_judge_target_agent ON llm_judge_evaluations(target_agent);

CREATE TABLE IF NOT EXISTS retrieval_evaluations (
    retrieval_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    query TEXT NOT NULL,
    k INTEGER NOT NULL,
    precision_at_k REAL NOT NULL,
    recall_at_k REAL NOT NULL,
    mrr REAL NOT NULL,
    ndcg REAL NOT NULL,
    context_relevance REAL NOT NULL,
    context_coverage REAL NOT NULL,
    faithfulness_score REAL NOT NULL,
    grounded_status TEXT NOT NULL,
    unsupported_claims_json TEXT NOT NULL DEFAULT '[]',
    timestamp TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_retrieval_task_id ON retrieval_evaluations(task_id);

CREATE TABLE IF NOT EXISTS agent_performance_metrics (
    metric_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    workflow_id TEXT,
    execution_time_ms REAL NOT NULL,
    outcome TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    recovery_succeeded INTEGER NOT NULL DEFAULT 0,
    tool_failure INTEGER NOT NULL DEFAULT 0,
    human_intervention INTEGER NOT NULL DEFAULT 0,
    human_rejected INTEGER NOT NULL DEFAULT 0,
    evaluation_failed INTEGER NOT NULL DEFAULT 0,
    failure_category TEXT,
    timestamp TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_perf_agent_name ON agent_performance_metrics(agent_name);
CREATE INDEX IF NOT EXISTS idx_perf_task_type ON agent_performance_metrics(task_type);
CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON agent_performance_metrics(timestamp);

CREATE TABLE IF NOT EXISTS evaluation_revisions (
    revision_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    failing_agent TEXT NOT NULL,
    failing_stage TEXT NOT NULL,
    feedback TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    preserved_outputs_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'REQUESTED',
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_revisions_task_id ON evaluation_revisions(task_id);

-- Day 229 Observability Tables: Traces, Spans, Events, Logs, Metrics
CREATE TABLE IF NOT EXISTS observability_traces (
    trace_id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    task_id TEXT,
    status TEXT NOT NULL DEFAULT 'STARTED',
    start_time TEXT NOT NULL,
    end_time TEXT,
    duration_ms REAL,
    root_span_id TEXT,
    agent_sequence_json TEXT NOT NULL DEFAULT '[]',
    error_json TEXT NOT NULL DEFAULT '{}',
    attributes_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_traces_workflow_id ON observability_traces(workflow_id);
CREATE INDEX IF NOT EXISTS idx_traces_task_id ON observability_traces(task_id);
CREATE INDEX IF NOT EXISTS idx_traces_status ON observability_traces(status);
CREATE INDEX IF NOT EXISTS idx_traces_start_time ON observability_traces(start_time);

CREATE TABLE IF NOT EXISTS observability_spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    workflow_id TEXT NOT NULL,
    task_id TEXT,
    agent_name TEXT NOT NULL,
    operation_name TEXT NOT NULL,
    span_kind TEXT NOT NULL DEFAULT 'agent',
    start_time TEXT NOT NULL,
    end_time TEXT,
    duration_ms REAL,
    status TEXT NOT NULL DEFAULT 'STARTED',
    attributes_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON observability_spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent_span_id ON observability_spans(parent_span_id);
CREATE INDEX IF NOT EXISTS idx_spans_agent_name ON observability_spans(agent_name);
CREATE INDEX IF NOT EXISTS idx_spans_operation ON observability_spans(operation_name);
CREATE INDEX IF NOT EXISTS idx_spans_status ON observability_spans(status);

CREATE TABLE IF NOT EXISTS observability_events (
    event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    span_id TEXT,
    event_type TEXT NOT NULL,
    agent_name TEXT,
    timestamp TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_obs_events_trace_id ON observability_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_obs_events_type ON observability_events(event_type);
CREATE INDEX IF NOT EXISTS idx_obs_events_timestamp ON observability_events(timestamp);

CREATE TABLE IF NOT EXISTS observability_logs (
    log_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    log_level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    trace_id TEXT,
    span_id TEXT,
    workflow_id TEXT,
    task_id TEXT,
    agent_name TEXT,
    operation_name TEXT,
    message TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    error_type TEXT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_logs_trace_id ON observability_logs(trace_id);
CREATE INDEX IF NOT EXISTS idx_obs_logs_workflow_id ON observability_logs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_obs_logs_level ON observability_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_obs_logs_event_type ON observability_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_obs_logs_timestamp ON observability_logs(timestamp);

CREATE TABLE IF NOT EXISTS observability_metrics (
    metric_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    value REAL NOT NULL,
    agent_name TEXT,
    workflow_id TEXT,
    task_id TEXT,
    provider TEXT,
    model TEXT,
    tool_name TEXT,
    attributes_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_obs_metrics_name ON observability_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_obs_metrics_agent ON observability_metrics(agent_name);
CREATE INDEX IF NOT EXISTS idx_obs_metrics_provider ON observability_metrics(provider);
CREATE INDEX IF NOT EXISTS idx_obs_metrics_timestamp ON observability_metrics(timestamp);

-- Day 230 Security Events Table with Verifiable Hash Chain
CREATE TABLE IF NOT EXISTS security_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    trace_id TEXT,
    workflow_id TEXT,
    task_id TEXT,
    agent TEXT,
    threat_category TEXT NOT NULL,
    severity TEXT NOT NULL,
    security_decision TEXT NOT NULL,
    sanitized_explanation TEXT,
    relevant_resource TEXT,
    findings_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    sequence_number INTEGER,
    previous_event_hash TEXT,
    event_hash TEXT,
    hash_version TEXT DEFAULT 'v1'
);
CREATE INDEX IF NOT EXISTS idx_sec_events_threat ON security_events(threat_category);
CREATE INDEX IF NOT EXISTS idx_sec_events_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_sec_events_decision ON security_events(security_decision);
CREATE INDEX IF NOT EXISTS idx_sec_events_workflow ON security_events(workflow_id);
CREATE INDEX IF NOT EXISTS idx_sec_events_task ON security_events(task_id);
CREATE INDEX IF NOT EXISTS idx_sec_events_timestamp ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_sec_events_seq ON security_events(sequence_number);
"""


# ===========================================================================
# 2. SQLite Persistent Message Storage
# ===========================================================================

class SQLiteMessageStorage:
    """Thread-safe SQLite storage engine for AgentMessage persistence and restart recovery."""

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with timeout and foreign key support."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode disabled via explicit transactions
        )
        conn.row_factory = sqlite3.Row
        # Enable write-ahead logging (WAL) for file databases to allow concurrent readers & writer
        if self.db_path != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
            except sqlite3.Error as exc:
                logger.warning(f"[SQLiteStorage] Could not set WAL mode: {exc}")
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite tables and indexes safely."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript(CREATE_MESSAGES_TABLE_SQL)
                # Backward-compatible schema migration for security_events hash chain
                try:
                    cols = [r["name"] for r in conn.execute("PRAGMA table_info(security_events)").fetchall()]
                    if cols and "sequence_number" not in cols:
                        conn.execute("ALTER TABLE security_events ADD COLUMN sequence_number INTEGER;")
                        conn.execute("ALTER TABLE security_events ADD COLUMN previous_event_hash TEXT;")
                        conn.execute("ALTER TABLE security_events ADD COLUMN event_hash TEXT;")
                        conn.execute("ALTER TABLE security_events ADD COLUMN hash_version TEXT DEFAULT 'v1';")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_sec_events_seq ON security_events(sequence_number);")
                except Exception as mig_exc:
                    logger.debug(f"[SQLiteStorage] Migration note for security_events: {mig_exc}")
                logger.info(f"[SQLiteStorage] Initialized message database at: {self.db_path}")
            except Exception as exc:
                logger.error(f"[SQLiteStorage] Failed to initialize database: {exc}")
                raise
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Serialization / Deserialization Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def serialize_payload(data: Any) -> str:
        """Serialize payload (input_data/output_data) safely to JSON."""
        try:
            return json.dumps(data)
        except (TypeError, ValueError):
            return json.dumps(str(data))

    @staticmethod
    def deserialize_payload(json_str: str) -> Any:
        """Deserialize JSON string to Python structure."""
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except Exception:
            return json_str

    def _row_to_queue_item(self, row: sqlite3.Row, sequence_id: int) -> QueueItem:
        """Reconstruct a QueueItem and underlying AgentMessage from a SQLite row."""
        payload = self.deserialize_payload(row["payload_json"])
        meta = self.deserialize_payload(row["metadata_json"]) or {}

        # Rebuild input/output
        input_data = payload.get("input_data") if isinstance(payload, dict) else payload
        output_data = payload.get("output_data") if isinstance(payload, dict) else None

        agent_msg = AgentMessage(
            message_id=row["message_id"],
            task_id=row["task_id"],
            agent_name=row["sender"],
            sender=row["sender"],
            receiver=row["receiver"],
            stage=meta.get("stage", "execution"),
            status=meta.get("agent_status", "waiting"),
            message_type=row["message_type"],
            correlation_id=row["correlation_id"],
            input_data=input_data,
            output_data=output_data,
            error=row["error"],
            metadata=meta,
            priority=row["priority"],
        )

        priority_enum = MessagePriority.from_value(row["priority"])
        state_enum = QueueMessageState(row["state"])

        return QueueItem(
            effective_priority=priority_enum.level,
            sequence_id=sequence_id,
            message=agent_msg,
            original_priority=priority_enum,
            state=state_enum,
            enqueued_at=row["enqueued_at"],
            attempts=row["attempts"],
            last_error=row["error"],
            completed_at=row["completed_at"],
        )

    # -----------------------------------------------------------------------
    # CRUD Operations
    # -----------------------------------------------------------------------

    def insert_message(
        self,
        message: AgentMessage,
        priority: MessagePriority,
        state: QueueMessageState = QueueMessageState.PENDING,
        enqueued_at: Optional[float] = None,
    ) -> bool:
        """Insert a new message into SQLite storage. Fails cleanly if message_id already exists."""
        now_ts = datetime.now(timezone.utc).isoformat()
        t_enqueued = enqueued_at or datetime.now(timezone.utc).timestamp()

        payload = {
            "input_data": message.input_data,
            "output_data": message.output_data,
        }
        meta = dict(message.metadata)
        meta["stage"] = message.stage
        meta["agent_status"] = message.status

        sql = """
        INSERT INTO messages (
            message_id, task_id, correlation_id, sender, receiver,
            message_type, priority, payload_json, state, attempts,
            error, enqueued_at, updated_at, completed_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    sql,
                    (
                        message.message_id,
                        message.task_id,
                        message.correlation_id,
                        message.sender,
                        message.receiver,
                        str(message.message_type),
                        priority.value,
                        self.serialize_payload(payload),
                        state.value,
                        0,
                        message.error,
                        t_enqueued,
                        now_ts,
                        None,
                        self.serialize_payload(meta),
                    ),
                )
                conn.execute("COMMIT;")
                logger.debug(f"[SQLiteStorage] Inserted message '{message.message_id}' (state={state.value})")
                return True
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK;")
                logger.warning(
                    f"[SQLiteStorage] Duplicate message insertion rejected: {message.message_id} ({exc})"
                )
                return False
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to insert message {message.message_id}: {exc}")
                raise
            finally:
                conn.close()

    def update_state(
        self,
        message_id: str,
        state: QueueMessageState,
        attempts: Optional[int] = None,
        error: Optional[str] = None,
        completed_at: Optional[str] = None,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update the state, attempt count, error, and metadata of a stored message atomically."""
        now_ts = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                # Fetch existing metadata if update provided
                meta_json = None
                if metadata_update:
                    cur = conn.execute("SELECT metadata_json FROM messages WHERE message_id = ?", (message_id,))
                    row = cur.fetchone()
                    if row:
                        curr_meta = self.deserialize_payload(row["metadata_json"]) or {}
                        curr_meta.update(metadata_update)
                        meta_json = self.serialize_payload(curr_meta)

                updates = ["state = ?", "updated_at = ?"]
                params: List[Any] = [state.value, now_ts]

                if attempts is not None:
                    updates.append("attempts = ?")
                    params.append(attempts)

                if error is not None:
                    updates.append("error = ?")
                    params.append(error)

                if completed_at is not None:
                    updates.append("completed_at = ?")
                    params.append(completed_at)

                if meta_json is not None:
                    updates.append("metadata_json = ?")
                    params.append(meta_json)

                params.append(message_id)
                sql = f"UPDATE messages SET {', '.join(updates)} WHERE message_id = ?;"

                conn.execute(sql, tuple(params))
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to update state for {message_id}: {exc}")
                raise
            finally:
                conn.close()

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a raw message dictionary by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return dict(row)
            finally:
                conn.close()

    def get_messages_by_state(self, state: QueueMessageState) -> List[Dict[str, Any]]:
        """Query all messages in a specific state."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM messages WHERE state = ? ORDER BY enqueued_at ASC",
                    (state.value,),
                )
                return [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def count_by_state(self, state: QueueMessageState) -> int:
        """Count total messages matching a given state."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT COUNT(*) FROM messages WHERE state = ?", (state.value,))
                return cur.fetchone()[0]
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Restart Recovery Scanner
    # -----------------------------------------------------------------------

    def recover_on_restart(
        self,
        max_retries: int = 2,
    ) -> Tuple[List[QueueItem], List[str]]:
        """Scan SQLite database on startup to recover interrupted or pending messages.

        Returns:
            Tuple of:
            - List of recoverable QueueItem objects to load into the priority queue
            - List of permanently failed or unrecoverable message_ids
        """
        recoverable_items: List[QueueItem] = []
        permanently_failed: List[str] = []

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")

                # 1. Recover PENDING messages (ordered by priority, then enqueued_at)
                cur = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE state = 'pending'
                    ORDER BY enqueued_at ASC;
                    """
                )
                pending_rows = cur.fetchall()

                seq = 0
                for row in pending_rows:
                    seq += 1
                    item = self._row_to_queue_item(row, sequence_id=seq)
                    recoverable_items.append(item)

                # 2. Recover messages left in PROCESSING state during a crash
                cur = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE state = 'processing'
                    ORDER BY enqueued_at ASC;
                    """
                )
                processing_rows = cur.fetchall()

                for row in processing_rows:
                    seq += 1
                    msg_id = row["message_id"]
                    payload = self.deserialize_payload(row["payload_json"])
                    desc = ""
                    if isinstance(payload, dict):
                        inp = payload.get("input_data")
                        if isinstance(inp, str):
                            desc = inp
                        elif isinstance(inp, dict):
                            desc = str(inp.get("description", "") or inp.get("query", ""))

                    # Non-idempotent operation defense: NEVER automatically re-execute interrupted operations
                    is_non_idempotent = is_non_idempotent_operation(desc)
                    attempts = row["attempts"] + 1

                    if is_non_idempotent or attempts > max_retries + 1:
                        # Mark as permanently failed to avoid duplicate side-effects
                        reason = (
                            "Interrupted non-idempotent operation cannot be safely re-executed on restart."
                            if is_non_idempotent
                            else f"Restart recovery: attempt limit ({max_retries}) exceeded."
                        )
                        conn.execute(
                            """
                            UPDATE messages
                            SET state = 'failed', error = ?, updated_at = ?, completed_at = ?
                            WHERE message_id = ?;
                            """,
                            (reason, datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat(), msg_id),
                        )
                        permanently_failed.append(msg_id)
                        logger.warning(f"[SQLiteStorage] Crash recovery: Marked {msg_id} as FAILED: {reason}")
                    else:
                        # Revert state to PENDING and re-enqueue
                        conn.execute(
                            """
                            UPDATE messages
                            SET state = 'pending', attempts = ?, error = ?, updated_at = ?
                            WHERE message_id = ?;
                            """,
                            (
                                attempts,
                                "Recovered from ungraceful application restart.",
                                datetime.now(timezone.utc).isoformat(),
                                msg_id,
                            ),
                        )
                        # Re-fetch updated row to build QueueItem
                        item = self._row_to_queue_item(row, sequence_id=seq)
                        item.state = QueueMessageState.PENDING
                        item.attempts = attempts
                        item.last_error = "Recovered from ungraceful application restart."
                        recoverable_items.append(item)
                        logger.info(
                            f"[SQLiteStorage] Crash recovery: Re-queued interrupted message {msg_id} (attempt {attempts})"
                        )

                conn.execute("COMMIT;")
                logger.info(
                    f"[SQLiteStorage] Restart recovery complete: {len(recoverable_items)} items recovered, "
                    f"{len(permanently_failed)} items failed."
                )
                return recoverable_items, permanently_failed

            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Restart recovery failed: {exc}")
                raise
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Acknowledgement Storage Helpers (Day 226 Part 5)
    # -----------------------------------------------------------------------

    def update_acknowledgement(
        self,
        message_id: str,
        ack_status: str,
        reason: Optional[str] = None,
        ack_timestamp: Optional[str] = None,
    ) -> bool:
        """Persist recipient acknowledgement status for a message."""
        now_ts = ack_timestamp or datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    UPDATE messages
                    SET ack_status = ?, ack_reason = ?, ack_timestamp = ?, updated_at = ?
                    WHERE message_id = ?;
                    """,
                    (ack_status, reason, now_ts, now_ts, message_id),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to update ACK for {message_id}: {exc}")
                raise
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Dead-Letter Queue Storage Helpers (Day 226 Part 5)
    # -----------------------------------------------------------------------

    def insert_dlq_entry(self, entry: Any) -> bool:
        """Persist an unresolvable or exhausted message into dead_letter_queue table."""
        sql = """
        INSERT INTO dead_letter_queue (
            dlq_id, message_id, task_id, correlation_id, sender, receiver,
            reason, retry_count, entered_at, replayed, replay_count,
            replay_history_json, failure_details_json, message_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    sql,
                    (
                        entry.dlq_id,
                        entry.message.message_id,
                        entry.message.task_id,
                        entry.message.correlation_id,
                        entry.message.sender,
                        entry.message.receiver,
                        entry.reason,
                        entry.retry_count,
                        entry.entered_at,
                        1 if entry.replayed else 0,
                        entry.replay_count,
                        self.serialize_payload(entry.replay_history),
                        self.serialize_payload(entry.failure_details or {}),
                        self.serialize_payload(entry.message.to_dict()),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK;")
                return False
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to insert DLQ entry {entry.dlq_id}: {exc}")
                raise
            finally:
                conn.close()

    def update_dlq_replay(self, dlq_id: str, replay_count: int, timestamp: str) -> bool:
        """Update replay status and count for a DLQ entry."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    UPDATE dead_letter_queue
                    SET replayed = 1, replay_count = ?
                    WHERE dlq_id = ?;
                    """,
                    (replay_count, dlq_id),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to update DLQ replay for {dlq_id}: {exc}")
                raise
            finally:
                conn.close()

    def get_all_dlq_entries(self) -> List[Any]:
        """Load all DLQ entries from persistent storage."""
        from core.reliability import DLQEntry
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM dead_letter_queue ORDER BY entered_at ASC;")
                rows = cur.fetchall()
                results = []
                for r in rows:
                    msg_dict = self.deserialize_payload(r["message_json"])
                    msg = AgentMessage.from_dict(msg_dict) if msg_dict else None
                    if msg:
                        entry = DLQEntry(
                            dlq_id=r["dlq_id"],
                            message=msg,
                            reason=r["reason"],
                            retry_count=r["retry_count"],
                            entered_at=r["entered_at"],
                            replayed=bool(r["replayed"]),
                            replay_count=r["replay_count"],
                            replay_history=self.deserialize_payload(r["replay_history_json"]) or [],
                            failure_details=self.deserialize_payload(r["failure_details_json"]) or {},
                        )
                        results.append(entry)
                return results
            finally:
                conn.close()



    def save_task_outcome(self, data: Dict[str, Any]) -> bool:
        """Persist or update a task outcome evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO task_outcomes (
                        outcome_id, task_id, objective, classification, passed_criteria_json,
                        failed_criteria_json, numeric_checks_json, missing_fields_json,
                        evidence_verified, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(outcome_id) DO UPDATE SET
                        classification = excluded.classification,
                        passed_criteria_json = excluded.passed_criteria_json,
                        failed_criteria_json = excluded.failed_criteria_json,
                        numeric_checks_json = excluded.numeric_checks_json,
                        missing_fields_json = excluded.missing_fields_json,
                        evidence_verified = excluded.evidence_verified,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["outcome_id"],
                        data["task_id"],
                        data["objective"],
                        str(data["classification"]),
                        self.serialize_payload(data.get("passed_criteria", [])),
                        self.serialize_payload(data.get("failed_criteria", [])),
                        self.serialize_payload(data.get("numeric_checks", {})),
                        self.serialize_payload(data.get("missing_fields", [])),
                        1 if data.get("evidence_verified") else 0,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save task outcome {data.get('outcome_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_task_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task outcome evaluation by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM task_outcomes WHERE outcome_id = ?;", (outcome_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "outcome_id": row["outcome_id"],
                    "task_id": row["task_id"],
                    "objective": row["objective"],
                    "classification": row["classification"],
                    "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                    "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                    "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                    "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                    "evidence_verified": bool(row["evidence_verified"]),
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_task_outcomes(
        self,
        task_id: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query task outcome evaluations."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM task_outcomes WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if classification:
                    query += " AND classification = ?"
                    params.append(classification)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "outcome_id": row["outcome_id"],
                        "task_id": row["task_id"],
                        "objective": row["objective"],
                        "classification": row["classification"],
                        "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                        "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                        "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                        "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                        "evidence_verified": bool(row["evidence_verified"]),
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_llm_judge_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update an LLM Judge evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO llm_judge_evaluations (
                        judge_id, task_id, target_agent, target_stage, accuracy, completeness,
                        relevance, instruction_adherence, evidence_support, overall_score,
                        identified_weaknesses_json, confidence, reasoning_summary, evaluator_model,
                        evaluator_provider, status, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(judge_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        confidence = excluded.confidence,
                        reasoning_summary = excluded.reasoning_summary,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["judge_id"],
                        data["task_id"],
                        data["target_agent"],
                        data["target_stage"],
                        float(data.get("accuracy", 0.0)),
                        float(data.get("completeness", 0.0)),
                        float(data.get("relevance", 0.0)),
                        float(data.get("instruction_adherence", 0.0)),
                        float(data.get("evidence_support", 0.0)),
                        float(data.get("overall_score", 0.0)),
                        self.serialize_payload(data.get("identified_weaknesses", [])),
                        float(data.get("confidence", 0.0)),
                        data.get("reasoning_summary", ""),
                        data.get("evaluator_model", ""),
                        data.get("evaluator_provider", ""),
                        data.get("status", "COMPLETED"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save LLM judge evaluation {data.get('judge_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_llm_judge_evaluation(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve LLM judge evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM llm_judge_evaluations WHERE judge_id = ?;", (judge_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "judge_id": row["judge_id"],
                    "task_id": row["task_id"],
                    "target_agent": row["target_agent"],
                    "target_stage": row["target_stage"],
                    "accuracy": row["accuracy"],
                    "completeness": row["completeness"],
                    "relevance": row["relevance"],
                    "instruction_adherence": row["instruction_adherence"],
                    "evidence_support": row["evidence_support"],
                    "overall_score": row["overall_score"],
                    "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                    "confidence": row["confidence"],
                    "reasoning_summary": row["reasoning_summary"],
                    "evaluator_model": row["evaluator_model"],
                    "evaluator_provider": row["evaluator_provider"],
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_llm_judge_evaluations(
        self,
        task_id: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query LLM judge evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM llm_judge_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if target_agent:
                    query += " AND target_agent = ?"
                    params.append(target_agent)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "judge_id": row["judge_id"],
                        "task_id": row["task_id"],
                        "target_agent": row["target_agent"],
                        "target_stage": row["target_stage"],
                        "accuracy": row["accuracy"],
                        "completeness": row["completeness"],
                        "relevance": row["relevance"],
                        "instruction_adherence": row["instruction_adherence"],
                        "evidence_support": row["evidence_support"],
                        "overall_score": row["overall_score"],
                        "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                        "confidence": row["confidence"],
                        "reasoning_summary": row["reasoning_summary"],
                        "evaluator_model": row["evaluator_model"],
                        "evaluator_provider": row["evaluator_provider"],
                        "status": row["status"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_retrieval_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update a retrieval & RAG evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO retrieval_evaluations (
                        retrieval_id, task_id, query, k, precision_at_k, recall_at_k,
                        mrr, ndcg, context_relevance, context_coverage, faithfulness_score,
                        grounded_status, unsupported_claims_json, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(retrieval_id) DO UPDATE SET
                        precision_at_k = excluded.precision_at_k,
                        recall_at_k = excluded.recall_at_k,
                        mrr = excluded.mrr,
                        ndcg = excluded.ndcg,
                        faithfulness_score = excluded.faithfulness_score,
                        grounded_status = excluded.grounded_status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["retrieval_id"],
                        data["task_id"],
                        data["query"],
                        int(data.get("k", 5)),
                        float(data.get("precision_at_k", 0.0)),
                        float(data.get("recall_at_k", 0.0)),
                        float(data.get("mrr", 0.0)),
                        float(data.get("ndcg", 0.0)),
                        float(data.get("context_relevance", 0.0)),
                        float(data.get("context_coverage", 0.0)),
                        float(data.get("faithfulness_score", 0.0)),
                        str(data.get("grounded_status", "GROUNDED")),
                        self.serialize_payload(data.get("unsupported_claims", [])),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save retrieval evaluation {data.get('retrieval_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_retrieval_evaluation(self, retrieval_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve retrieval evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM retrieval_evaluations WHERE retrieval_id = ?;", (retrieval_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "retrieval_id": row["retrieval_id"],
                    "task_id": row["task_id"],
                    "query": row["query"],
                    "k": row["k"],
                    "precision_at_k": row["precision_at_k"],
                    "recall_at_k": row["recall_at_k"],
                    "mrr": row["mrr"],
                    "ndcg": row["ndcg"],
                    "context_relevance": row["context_relevance"],
                    "context_coverage": row["context_coverage"],
                    "faithfulness_score": row["faithfulness_score"],
                    "grounded_status": row["grounded_status"],
                    "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_retrieval_evaluations(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query retrieval evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM retrieval_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "retrieval_id": row["retrieval_id"],
                        "task_id": row["task_id"],
                        "query": row["query"],
                        "k": row["k"],
                        "precision_at_k": row["precision_at_k"],
                        "recall_at_k": row["recall_at_k"],
                        "mrr": row["mrr"],
                        "ndcg": row["ndcg"],
                        "context_relevance": row["context_relevance"],
                        "context_coverage": row["context_coverage"],
                        "faithfulness_score": row["faithfulness_score"],
                        "grounded_status": row["grounded_status"],
                        "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_performance_metric(self, data: Dict[str, Any]) -> bool:
        """Persist an agent performance metric record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                metric_id = data.get("metric_id") or f"pm_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO agent_performance_metrics (
                        metric_id, agent_name, task_type, workflow_id, execution_time_ms,
                        outcome, retry_count, recovery_succeeded, tool_failure, human_intervention,
                        human_rejected, evaluation_failed, failure_category, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_id) DO UPDATE SET
                        execution_time_ms = excluded.execution_time_ms,
                        outcome = excluded.outcome,
                        retry_count = excluded.retry_count,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        metric_id,
                        data["agent_name"],
                        data.get("task_type", "general"),
                        data.get("workflow_id"),
                        float(data.get("execution_time_ms", 0.0)),
                        str(data.get("outcome", "success")),
                        int(data.get("retry_count", 0)),
                        1 if data.get("recovery_succeeded") else 0,
                        1 if data.get("tool_failure") else 0,
                        1 if data.get("human_intervention") else 0,
                        1 if data.get("human_rejected") else 0,
                        1 if data.get("evaluation_failed") else 0,
                        data.get("failure_category"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save performance metric {data.get('metric_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_performance_metrics(
        self,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query performance metrics with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM agent_performance_metrics WHERE 1=1"
                params = []
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "metric_id": row["metric_id"],
                        "agent_name": row["agent_name"],
                        "task_type": row["task_type"],
                        "workflow_id": row["workflow_id"],
                        "execution_time_ms": row["execution_time_ms"],
                        "outcome": row["outcome"],
                        "retry_count": row["retry_count"],
                        "recovery_succeeded": bool(row["recovery_succeeded"]),
                        "tool_failure": bool(row["tool_failure"]),
                        "human_intervention": bool(row["human_intervention"]),
                        "human_rejected": bool(row["human_rejected"]),
                        "evaluation_failed": bool(row["evaluation_failed"]),
                        "failure_category": row["failure_category"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_evaluation_revision(self, data: Dict[str, Any]) -> bool:
        """Persist or update an evaluation revision request."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO evaluation_revisions (
                        revision_id, task_id, failing_agent, failing_stage, feedback,
                        attempt_number, max_attempts, preserved_outputs_json, status,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(revision_id) DO UPDATE SET
                        status = excluded.status,
                        completed_at = excluded.completed_at;
                    """,
                    (
                        data["revision_id"],
                        data["task_id"],
                        data["failing_agent"],
                        data["failing_stage"],
                        data["feedback"],
                        int(data.get("attempt_number", 1)),
                        int(data.get("max_attempts", 2)),
                        self.serialize_payload(data.get("preserved_outputs", {})),
                        data.get("status", "REQUESTED"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        data.get("completed_at"),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save evaluation revision {data.get('revision_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_evaluation_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation revision by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM evaluation_revisions WHERE revision_id = ?;", (revision_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "revision_id": row["revision_id"],
                    "task_id": row["task_id"],
                    "failing_agent": row["failing_agent"],
                    "failing_stage": row["failing_stage"],
                    "feedback": row["feedback"],
                    "attempt_number": row["attempt_number"],
                    "max_attempts": row["max_attempts"],
                    "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                }
            finally:
                conn.close()

    def list_evaluation_revisions(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query evaluation revisions."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM evaluation_revisions WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "revision_id": row["revision_id"],
                        "task_id": row["task_id"],
                        "failing_agent": row["failing_agent"],
                        "failing_stage": row["failing_stage"],
                        "feedback": row["feedback"],
                        "attempt_number": row["attempt_number"],
                        "max_attempts": row["max_attempts"],
                        "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "completed_at": row["completed_at"],
                    })
                return results
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Day 227: Human-in-the-Loop (HITL) Persistence
    # -----------------------------------------------------------------------

    def save_interaction(self, data: Dict[str, Any]) -> bool:
        """Persist or upsert a HumanInteractionRequest."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO human_interactions (
                        interaction_id, task_id, correlation_id, requesting_agent,
                        interaction_type, question_or_request, relevant_context_json,
                        available_decisions_json, status, decision, human_response,
                        created_at, resolved_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(interaction_id) DO UPDATE SET
                        status = excluded.status,
                        decision = excluded.decision,
                        human_response = excluded.human_response,
                        resolved_at = excluded.resolved_at,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["interaction_id"],
                        data["task_id"],
                        data.get("correlation_id"),
                        data["requesting_agent"],
                        data["interaction_type"],
                        data["question_or_request"],
                        self.serialize_payload(data.get("relevant_context", {})),
                        self.serialize_payload(data.get("available_decisions", [])),
                        data.get("status", "pending"),
                        data.get("decision"),
                        data.get("human_response"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        data.get("resolved_at"),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save human interaction {data.get('interaction_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific human interaction by its ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM human_interactions WHERE interaction_id = ?;",
                    (interaction_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "interaction_id": row["interaction_id"],
                    "task_id": row["task_id"],
                    "correlation_id": row["correlation_id"],
                    "requesting_agent": row["requesting_agent"],
                    "interaction_type": row["interaction_type"],
                    "question_or_request": row["question_or_request"],
                    "relevant_context": self.deserialize_payload(row["relevant_context_json"]) or {},
                    "available_decisions": self.deserialize_payload(row["available_decisions_json"]) or [],
                    "status": row["status"],
                    "decision": row["decision"],
                    "human_response": row["human_response"],
                    "created_at": row["created_at"],
                    "resolved_at": row["resolved_at"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_interactions(
        self,
        task_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query interactions with optional task_id and status filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM human_interactions WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if status:
                    query += " AND status = ?"
                    params.append(status)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "interaction_id": row["interaction_id"],
                        "task_id": row["task_id"],
                        "correlation_id": row["correlation_id"],
                        "requesting_agent": row["requesting_agent"],
                        "interaction_type": row["interaction_type"],
                        "question_or_request": row["question_or_request"],
                        "relevant_context": self.deserialize_payload(row["relevant_context_json"]) or {},
                        "available_decisions": self.deserialize_payload(row["available_decisions_json"]) or [],
                        "status": row["status"],
                        "decision": row["decision"],
                        "human_response": row["human_response"],
                        "created_at": row["created_at"],
                        "resolved_at": row["resolved_at"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()



    def save_task_outcome(self, data: Dict[str, Any]) -> bool:
        """Persist or update a task outcome evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO task_outcomes (
                        outcome_id, task_id, objective, classification, passed_criteria_json,
                        failed_criteria_json, numeric_checks_json, missing_fields_json,
                        evidence_verified, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(outcome_id) DO UPDATE SET
                        classification = excluded.classification,
                        passed_criteria_json = excluded.passed_criteria_json,
                        failed_criteria_json = excluded.failed_criteria_json,
                        numeric_checks_json = excluded.numeric_checks_json,
                        missing_fields_json = excluded.missing_fields_json,
                        evidence_verified = excluded.evidence_verified,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["outcome_id"],
                        data["task_id"],
                        data["objective"],
                        str(data["classification"]),
                        self.serialize_payload(data.get("passed_criteria", [])),
                        self.serialize_payload(data.get("failed_criteria", [])),
                        self.serialize_payload(data.get("numeric_checks", {})),
                        self.serialize_payload(data.get("missing_fields", [])),
                        1 if data.get("evidence_verified") else 0,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save task outcome {data.get('outcome_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_task_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task outcome evaluation by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM task_outcomes WHERE outcome_id = ?;", (outcome_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "outcome_id": row["outcome_id"],
                    "task_id": row["task_id"],
                    "objective": row["objective"],
                    "classification": row["classification"],
                    "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                    "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                    "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                    "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                    "evidence_verified": bool(row["evidence_verified"]),
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_task_outcomes(
        self,
        task_id: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query task outcome evaluations."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM task_outcomes WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if classification:
                    query += " AND classification = ?"
                    params.append(classification)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "outcome_id": row["outcome_id"],
                        "task_id": row["task_id"],
                        "objective": row["objective"],
                        "classification": row["classification"],
                        "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                        "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                        "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                        "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                        "evidence_verified": bool(row["evidence_verified"]),
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_llm_judge_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update an LLM Judge evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO llm_judge_evaluations (
                        judge_id, task_id, target_agent, target_stage, accuracy, completeness,
                        relevance, instruction_adherence, evidence_support, overall_score,
                        identified_weaknesses_json, confidence, reasoning_summary, evaluator_model,
                        evaluator_provider, status, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(judge_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        confidence = excluded.confidence,
                        reasoning_summary = excluded.reasoning_summary,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["judge_id"],
                        data["task_id"],
                        data["target_agent"],
                        data["target_stage"],
                        float(data.get("accuracy", 0.0)),
                        float(data.get("completeness", 0.0)),
                        float(data.get("relevance", 0.0)),
                        float(data.get("instruction_adherence", 0.0)),
                        float(data.get("evidence_support", 0.0)),
                        float(data.get("overall_score", 0.0)),
                        self.serialize_payload(data.get("identified_weaknesses", [])),
                        float(data.get("confidence", 0.0)),
                        data.get("reasoning_summary", ""),
                        data.get("evaluator_model", ""),
                        data.get("evaluator_provider", ""),
                        data.get("status", "COMPLETED"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save LLM judge evaluation {data.get('judge_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_llm_judge_evaluation(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve LLM judge evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM llm_judge_evaluations WHERE judge_id = ?;", (judge_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "judge_id": row["judge_id"],
                    "task_id": row["task_id"],
                    "target_agent": row["target_agent"],
                    "target_stage": row["target_stage"],
                    "accuracy": row["accuracy"],
                    "completeness": row["completeness"],
                    "relevance": row["relevance"],
                    "instruction_adherence": row["instruction_adherence"],
                    "evidence_support": row["evidence_support"],
                    "overall_score": row["overall_score"],
                    "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                    "confidence": row["confidence"],
                    "reasoning_summary": row["reasoning_summary"],
                    "evaluator_model": row["evaluator_model"],
                    "evaluator_provider": row["evaluator_provider"],
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_llm_judge_evaluations(
        self,
        task_id: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query LLM judge evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM llm_judge_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if target_agent:
                    query += " AND target_agent = ?"
                    params.append(target_agent)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "judge_id": row["judge_id"],
                        "task_id": row["task_id"],
                        "target_agent": row["target_agent"],
                        "target_stage": row["target_stage"],
                        "accuracy": row["accuracy"],
                        "completeness": row["completeness"],
                        "relevance": row["relevance"],
                        "instruction_adherence": row["instruction_adherence"],
                        "evidence_support": row["evidence_support"],
                        "overall_score": row["overall_score"],
                        "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                        "confidence": row["confidence"],
                        "reasoning_summary": row["reasoning_summary"],
                        "evaluator_model": row["evaluator_model"],
                        "evaluator_provider": row["evaluator_provider"],
                        "status": row["status"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_retrieval_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update a retrieval & RAG evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO retrieval_evaluations (
                        retrieval_id, task_id, query, k, precision_at_k, recall_at_k,
                        mrr, ndcg, context_relevance, context_coverage, faithfulness_score,
                        grounded_status, unsupported_claims_json, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(retrieval_id) DO UPDATE SET
                        precision_at_k = excluded.precision_at_k,
                        recall_at_k = excluded.recall_at_k,
                        mrr = excluded.mrr,
                        ndcg = excluded.ndcg,
                        faithfulness_score = excluded.faithfulness_score,
                        grounded_status = excluded.grounded_status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["retrieval_id"],
                        data["task_id"],
                        data["query"],
                        int(data.get("k", 5)),
                        float(data.get("precision_at_k", 0.0)),
                        float(data.get("recall_at_k", 0.0)),
                        float(data.get("mrr", 0.0)),
                        float(data.get("ndcg", 0.0)),
                        float(data.get("context_relevance", 0.0)),
                        float(data.get("context_coverage", 0.0)),
                        float(data.get("faithfulness_score", 0.0)),
                        str(data.get("grounded_status", "GROUNDED")),
                        self.serialize_payload(data.get("unsupported_claims", [])),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save retrieval evaluation {data.get('retrieval_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_retrieval_evaluation(self, retrieval_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve retrieval evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM retrieval_evaluations WHERE retrieval_id = ?;", (retrieval_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "retrieval_id": row["retrieval_id"],
                    "task_id": row["task_id"],
                    "query": row["query"],
                    "k": row["k"],
                    "precision_at_k": row["precision_at_k"],
                    "recall_at_k": row["recall_at_k"],
                    "mrr": row["mrr"],
                    "ndcg": row["ndcg"],
                    "context_relevance": row["context_relevance"],
                    "context_coverage": row["context_coverage"],
                    "faithfulness_score": row["faithfulness_score"],
                    "grounded_status": row["grounded_status"],
                    "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_retrieval_evaluations(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query retrieval evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM retrieval_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "retrieval_id": row["retrieval_id"],
                        "task_id": row["task_id"],
                        "query": row["query"],
                        "k": row["k"],
                        "precision_at_k": row["precision_at_k"],
                        "recall_at_k": row["recall_at_k"],
                        "mrr": row["mrr"],
                        "ndcg": row["ndcg"],
                        "context_relevance": row["context_relevance"],
                        "context_coverage": row["context_coverage"],
                        "faithfulness_score": row["faithfulness_score"],
                        "grounded_status": row["grounded_status"],
                        "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_performance_metric(self, data: Dict[str, Any]) -> bool:
        """Persist an agent performance metric record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                metric_id = data.get("metric_id") or f"pm_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO agent_performance_metrics (
                        metric_id, agent_name, task_type, workflow_id, execution_time_ms,
                        outcome, retry_count, recovery_succeeded, tool_failure, human_intervention,
                        human_rejected, evaluation_failed, failure_category, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_id) DO UPDATE SET
                        execution_time_ms = excluded.execution_time_ms,
                        outcome = excluded.outcome,
                        retry_count = excluded.retry_count,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        metric_id,
                        data["agent_name"],
                        data.get("task_type", "general"),
                        data.get("workflow_id"),
                        float(data.get("execution_time_ms", 0.0)),
                        str(data.get("outcome", "success")),
                        int(data.get("retry_count", 0)),
                        1 if data.get("recovery_succeeded") else 0,
                        1 if data.get("tool_failure") else 0,
                        1 if data.get("human_intervention") else 0,
                        1 if data.get("human_rejected") else 0,
                        1 if data.get("evaluation_failed") else 0,
                        data.get("failure_category"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save performance metric {data.get('metric_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_performance_metrics(
        self,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query performance metrics with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM agent_performance_metrics WHERE 1=1"
                params = []
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "metric_id": row["metric_id"],
                        "agent_name": row["agent_name"],
                        "task_type": row["task_type"],
                        "workflow_id": row["workflow_id"],
                        "execution_time_ms": row["execution_time_ms"],
                        "outcome": row["outcome"],
                        "retry_count": row["retry_count"],
                        "recovery_succeeded": bool(row["recovery_succeeded"]),
                        "tool_failure": bool(row["tool_failure"]),
                        "human_intervention": bool(row["human_intervention"]),
                        "human_rejected": bool(row["human_rejected"]),
                        "evaluation_failed": bool(row["evaluation_failed"]),
                        "failure_category": row["failure_category"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_evaluation_revision(self, data: Dict[str, Any]) -> bool:
        """Persist or update an evaluation revision request."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO evaluation_revisions (
                        revision_id, task_id, failing_agent, failing_stage, feedback,
                        attempt_number, max_attempts, preserved_outputs_json, status,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(revision_id) DO UPDATE SET
                        status = excluded.status,
                        completed_at = excluded.completed_at;
                    """,
                    (
                        data["revision_id"],
                        data["task_id"],
                        data["failing_agent"],
                        data["failing_stage"],
                        data["feedback"],
                        int(data.get("attempt_number", 1)),
                        int(data.get("max_attempts", 2)),
                        self.serialize_payload(data.get("preserved_outputs", {})),
                        data.get("status", "REQUESTED"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        data.get("completed_at"),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save evaluation revision {data.get('revision_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_evaluation_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation revision by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM evaluation_revisions WHERE revision_id = ?;", (revision_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "revision_id": row["revision_id"],
                    "task_id": row["task_id"],
                    "failing_agent": row["failing_agent"],
                    "failing_stage": row["failing_stage"],
                    "feedback": row["feedback"],
                    "attempt_number": row["attempt_number"],
                    "max_attempts": row["max_attempts"],
                    "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                }
            finally:
                conn.close()

    def list_evaluation_revisions(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query evaluation revisions."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM evaluation_revisions WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "revision_id": row["revision_id"],
                        "task_id": row["task_id"],
                        "failing_agent": row["failing_agent"],
                        "failing_stage": row["failing_stage"],
                        "feedback": row["feedback"],
                        "attempt_number": row["attempt_number"],
                        "max_attempts": row["max_attempts"],
                        "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "completed_at": row["completed_at"],
                    })
                return results
            finally:
                conn.close()

    def update_interaction(
        self,
        interaction_id: str,
        status: str,
        decision: Optional[str] = None,
        human_response: Optional[str] = None,
        resolved_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update the resolution status and response of an interaction."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                now_iso = resolved_at or datetime.now(timezone.utc).isoformat()
                meta_json = self.serialize_payload(metadata) if metadata is not None else None
                if meta_json:
                    conn.execute(
                        """
                        UPDATE human_interactions
                        SET status = ?, decision = ?, human_response = ?, resolved_at = ?, metadata_json = ?
                        WHERE interaction_id = ?;
                        """,
                        (status, decision, human_response, now_iso, meta_json, interaction_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE human_interactions
                        SET status = ?, decision = ?, human_response = ?, resolved_at = ?
                        WHERE interaction_id = ?;
                        """,
                        (status, decision, human_response, now_iso, interaction_id),
                    )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to update interaction {interaction_id}: {exc}")
                raise
            finally:
                conn.close()

    def save_approval(self, data: Dict[str, Any]) -> bool:
        """Persist or update an ApprovalRecord."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO human_approvals (
                        approval_id, interaction_id, task_id, correlation_id,
                        requested_action, decision, human_feedback, decision_timestamp,
                        approval_status, expires_at, token, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(approval_id) DO UPDATE SET
                        decision = excluded.decision,
                        human_feedback = excluded.human_feedback,
                        decision_timestamp = excluded.decision_timestamp,
                        approval_status = excluded.approval_status,
                        expires_at = excluded.expires_at,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["approval_id"],
                        data["interaction_id"],
                        data["task_id"],
                        data.get("correlation_id"),
                        str(data["requested_action"]),
                        data["decision"],
                        data.get("human_feedback"),
                        data.get("decision_timestamp") or datetime.now(timezone.utc).isoformat(),
                        data.get("approval_status", data["decision"]),
                        data.get("expires_at"),
                        data["token"],
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save approval record {data.get('approval_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an approval record by approval_id."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM human_approvals WHERE approval_id = ?;", (approval_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "approval_id": row["approval_id"],
                    "interaction_id": row["interaction_id"],
                    "task_id": row["task_id"],
                    "correlation_id": row["correlation_id"],
                    "requested_action": row["requested_action"],
                    "decision": row["decision"],
                    "human_feedback": row["human_feedback"],
                    "decision_timestamp": row["decision_timestamp"],
                    "approval_status": row["approval_status"],
                    "expires_at": row["expires_at"],
                    "token": row["token"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def get_approval_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Retrieve an approval record by token."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM human_approvals WHERE token = ?;", (token,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "approval_id": row["approval_id"],
                    "interaction_id": row["interaction_id"],
                    "task_id": row["task_id"],
                    "correlation_id": row["correlation_id"],
                    "requested_action": row["requested_action"],
                    "decision": row["decision"],
                    "human_feedback": row["human_feedback"],
                    "decision_timestamp": row["decision_timestamp"],
                    "approval_status": row["approval_status"],
                    "expires_at": row["expires_at"],
                    "token": row["token"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def atomic_consume_approval(
        self,
        token: str,
        consumed_by: str = "ExecutorAgent",
        task_id: Optional[str] = None,
        action: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Atomically transition approval_status to 'CONSUMED' in SQLite.
        
        Guarantees cross-process single-use semantics:
        Even if multiple operating system processes attempt to consume the same
        approval token simultaneously, SQLite transaction isolation ensures
        exactly one process succeeds and all others fail.
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                cur = conn.execute(
                    "SELECT approval_id, task_id, requested_action, approval_status, metadata_json, expires_at FROM human_approvals WHERE token = ?;",
                    (token,),
                )
                row = cur.fetchone()
                if not row:
                    conn.execute("ROLLBACK;")
                    return False, f"Approval token '{token}' not recognized in database."

                if row["approval_status"] == "CONSUMED":
                    conn.execute("ROLLBACK;")
                    return False, f"Approval token '{token[:12]}...' has already been consumed."

                # Verify task_id and action binding if provided
                if task_id and str(row["task_id"]) != str(task_id):
                    conn.execute("ROLLBACK;")
                    return False, f"Token task binding mismatch: expected '{task_id}', found '{row['task_id']}'."

                if action and str(row["requested_action"]) != str(action):
                    conn.execute("ROLLBACK;")
                    return False, f"Token action binding mismatch: expected '{action}', found '{row['requested_action']}'."

                # Check expiration
                exp = row["expires_at"]
                if exp and exp < now_ts:
                    conn.execute("ROLLBACK;")
                    return False, f"Approval token '{token[:12]}...' expired at '{exp}'."

                meta = self.deserialize_payload(row["metadata_json"]) or {}
                meta["consumed_at"] = now_ts
                meta["consumed_by"] = consumed_by
                meta_json = self.serialize_payload(meta)

                # Conditional update enforcing atomic single-use
                cur_upd = conn.execute(
                    """
                    UPDATE human_approvals
                    SET approval_status = 'CONSUMED', metadata_json = ?
                    WHERE token = ? AND approval_status != 'CONSUMED';
                    """,
                    (meta_json, token),
                )
                if cur_upd.rowcount != 1:
                    conn.execute("ROLLBACK;")
                    return False, f"Approval token '{token[:12]}...' was concurrently consumed."

                conn.execute("COMMIT;")
                logger.info(f"[SQLiteStorage] Atomically consumed approval token '{token[:10]}...' by '{consumed_by}'.")
                return True, None
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                logger.error(f"[SQLiteStorage] Error during atomic approval consumption: {exc}")
                return False, f"Database error during token consumption: {exc}"
            finally:
                conn.close()

    def get_approval_by_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an approval record by interaction_id."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM human_approvals WHERE interaction_id = ?;", (interaction_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "approval_id": row["approval_id"],
                    "interaction_id": row["interaction_id"],
                    "task_id": row["task_id"],
                    "correlation_id": row["correlation_id"],
                    "requested_action": row["requested_action"],
                    "decision": row["decision"],
                    "human_feedback": row["human_feedback"],
                    "decision_timestamp": row["decision_timestamp"],
                    "approval_status": row["approval_status"],
                    "expires_at": row["expires_at"],
                    "token": row["token"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_approvals(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all approvals optionally filtered by task_id."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM human_approvals"
                params = []
                if task_id:
                    query += " WHERE task_id = ?"
                    params.append(task_id)
                query += " ORDER BY decision_timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "approval_id": row["approval_id"],
                        "interaction_id": row["interaction_id"],
                        "task_id": row["task_id"],
                        "correlation_id": row["correlation_id"],
                        "requested_action": row["requested_action"],
                        "decision": row["decision"],
                        "human_feedback": row["human_feedback"],
                        "decision_timestamp": row["decision_timestamp"],
                        "approval_status": row["approval_status"],
                        "expires_at": row["expires_at"],
                        "token": row["token"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()



    def save_task_outcome(self, data: Dict[str, Any]) -> bool:
        """Persist or update a task outcome evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO task_outcomes (
                        outcome_id, task_id, objective, classification, passed_criteria_json,
                        failed_criteria_json, numeric_checks_json, missing_fields_json,
                        evidence_verified, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(outcome_id) DO UPDATE SET
                        classification = excluded.classification,
                        passed_criteria_json = excluded.passed_criteria_json,
                        failed_criteria_json = excluded.failed_criteria_json,
                        numeric_checks_json = excluded.numeric_checks_json,
                        missing_fields_json = excluded.missing_fields_json,
                        evidence_verified = excluded.evidence_verified,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["outcome_id"],
                        data["task_id"],
                        data["objective"],
                        str(data["classification"]),
                        self.serialize_payload(data.get("passed_criteria", [])),
                        self.serialize_payload(data.get("failed_criteria", [])),
                        self.serialize_payload(data.get("numeric_checks", {})),
                        self.serialize_payload(data.get("missing_fields", [])),
                        1 if data.get("evidence_verified") else 0,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save task outcome {data.get('outcome_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_task_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task outcome evaluation by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM task_outcomes WHERE outcome_id = ?;", (outcome_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "outcome_id": row["outcome_id"],
                    "task_id": row["task_id"],
                    "objective": row["objective"],
                    "classification": row["classification"],
                    "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                    "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                    "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                    "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                    "evidence_verified": bool(row["evidence_verified"]),
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_task_outcomes(
        self,
        task_id: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query task outcome evaluations."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM task_outcomes WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if classification:
                    query += " AND classification = ?"
                    params.append(classification)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "outcome_id": row["outcome_id"],
                        "task_id": row["task_id"],
                        "objective": row["objective"],
                        "classification": row["classification"],
                        "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                        "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                        "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                        "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                        "evidence_verified": bool(row["evidence_verified"]),
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_llm_judge_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update an LLM Judge evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO llm_judge_evaluations (
                        judge_id, task_id, target_agent, target_stage, accuracy, completeness,
                        relevance, instruction_adherence, evidence_support, overall_score,
                        identified_weaknesses_json, confidence, reasoning_summary, evaluator_model,
                        evaluator_provider, status, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(judge_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        confidence = excluded.confidence,
                        reasoning_summary = excluded.reasoning_summary,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["judge_id"],
                        data["task_id"],
                        data["target_agent"],
                        data["target_stage"],
                        float(data.get("accuracy", 0.0)),
                        float(data.get("completeness", 0.0)),
                        float(data.get("relevance", 0.0)),
                        float(data.get("instruction_adherence", 0.0)),
                        float(data.get("evidence_support", 0.0)),
                        float(data.get("overall_score", 0.0)),
                        self.serialize_payload(data.get("identified_weaknesses", [])),
                        float(data.get("confidence", 0.0)),
                        data.get("reasoning_summary", ""),
                        data.get("evaluator_model", ""),
                        data.get("evaluator_provider", ""),
                        data.get("status", "COMPLETED"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save LLM judge evaluation {data.get('judge_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_llm_judge_evaluation(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve LLM judge evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM llm_judge_evaluations WHERE judge_id = ?;", (judge_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "judge_id": row["judge_id"],
                    "task_id": row["task_id"],
                    "target_agent": row["target_agent"],
                    "target_stage": row["target_stage"],
                    "accuracy": row["accuracy"],
                    "completeness": row["completeness"],
                    "relevance": row["relevance"],
                    "instruction_adherence": row["instruction_adherence"],
                    "evidence_support": row["evidence_support"],
                    "overall_score": row["overall_score"],
                    "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                    "confidence": row["confidence"],
                    "reasoning_summary": row["reasoning_summary"],
                    "evaluator_model": row["evaluator_model"],
                    "evaluator_provider": row["evaluator_provider"],
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_llm_judge_evaluations(
        self,
        task_id: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query LLM judge evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM llm_judge_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if target_agent:
                    query += " AND target_agent = ?"
                    params.append(target_agent)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "judge_id": row["judge_id"],
                        "task_id": row["task_id"],
                        "target_agent": row["target_agent"],
                        "target_stage": row["target_stage"],
                        "accuracy": row["accuracy"],
                        "completeness": row["completeness"],
                        "relevance": row["relevance"],
                        "instruction_adherence": row["instruction_adherence"],
                        "evidence_support": row["evidence_support"],
                        "overall_score": row["overall_score"],
                        "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                        "confidence": row["confidence"],
                        "reasoning_summary": row["reasoning_summary"],
                        "evaluator_model": row["evaluator_model"],
                        "evaluator_provider": row["evaluator_provider"],
                        "status": row["status"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_retrieval_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update a retrieval & RAG evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO retrieval_evaluations (
                        retrieval_id, task_id, query, k, precision_at_k, recall_at_k,
                        mrr, ndcg, context_relevance, context_coverage, faithfulness_score,
                        grounded_status, unsupported_claims_json, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(retrieval_id) DO UPDATE SET
                        precision_at_k = excluded.precision_at_k,
                        recall_at_k = excluded.recall_at_k,
                        mrr = excluded.mrr,
                        ndcg = excluded.ndcg,
                        faithfulness_score = excluded.faithfulness_score,
                        grounded_status = excluded.grounded_status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["retrieval_id"],
                        data["task_id"],
                        data["query"],
                        int(data.get("k", 5)),
                        float(data.get("precision_at_k", 0.0)),
                        float(data.get("recall_at_k", 0.0)),
                        float(data.get("mrr", 0.0)),
                        float(data.get("ndcg", 0.0)),
                        float(data.get("context_relevance", 0.0)),
                        float(data.get("context_coverage", 0.0)),
                        float(data.get("faithfulness_score", 0.0)),
                        str(data.get("grounded_status", "GROUNDED")),
                        self.serialize_payload(data.get("unsupported_claims", [])),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save retrieval evaluation {data.get('retrieval_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_retrieval_evaluation(self, retrieval_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve retrieval evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM retrieval_evaluations WHERE retrieval_id = ?;", (retrieval_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "retrieval_id": row["retrieval_id"],
                    "task_id": row["task_id"],
                    "query": row["query"],
                    "k": row["k"],
                    "precision_at_k": row["precision_at_k"],
                    "recall_at_k": row["recall_at_k"],
                    "mrr": row["mrr"],
                    "ndcg": row["ndcg"],
                    "context_relevance": row["context_relevance"],
                    "context_coverage": row["context_coverage"],
                    "faithfulness_score": row["faithfulness_score"],
                    "grounded_status": row["grounded_status"],
                    "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_retrieval_evaluations(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query retrieval evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM retrieval_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "retrieval_id": row["retrieval_id"],
                        "task_id": row["task_id"],
                        "query": row["query"],
                        "k": row["k"],
                        "precision_at_k": row["precision_at_k"],
                        "recall_at_k": row["recall_at_k"],
                        "mrr": row["mrr"],
                        "ndcg": row["ndcg"],
                        "context_relevance": row["context_relevance"],
                        "context_coverage": row["context_coverage"],
                        "faithfulness_score": row["faithfulness_score"],
                        "grounded_status": row["grounded_status"],
                        "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_performance_metric(self, data: Dict[str, Any]) -> bool:
        """Persist an agent performance metric record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                metric_id = data.get("metric_id") or f"pm_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO agent_performance_metrics (
                        metric_id, agent_name, task_type, workflow_id, execution_time_ms,
                        outcome, retry_count, recovery_succeeded, tool_failure, human_intervention,
                        human_rejected, evaluation_failed, failure_category, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_id) DO UPDATE SET
                        execution_time_ms = excluded.execution_time_ms,
                        outcome = excluded.outcome,
                        retry_count = excluded.retry_count,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        metric_id,
                        data["agent_name"],
                        data.get("task_type", "general"),
                        data.get("workflow_id"),
                        float(data.get("execution_time_ms", 0.0)),
                        str(data.get("outcome", "success")),
                        int(data.get("retry_count", 0)),
                        1 if data.get("recovery_succeeded") else 0,
                        1 if data.get("tool_failure") else 0,
                        1 if data.get("human_intervention") else 0,
                        1 if data.get("human_rejected") else 0,
                        1 if data.get("evaluation_failed") else 0,
                        data.get("failure_category"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save performance metric {data.get('metric_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_performance_metrics(
        self,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query performance metrics with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM agent_performance_metrics WHERE 1=1"
                params = []
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "metric_id": row["metric_id"],
                        "agent_name": row["agent_name"],
                        "task_type": row["task_type"],
                        "workflow_id": row["workflow_id"],
                        "execution_time_ms": row["execution_time_ms"],
                        "outcome": row["outcome"],
                        "retry_count": row["retry_count"],
                        "recovery_succeeded": bool(row["recovery_succeeded"]),
                        "tool_failure": bool(row["tool_failure"]),
                        "human_intervention": bool(row["human_intervention"]),
                        "human_rejected": bool(row["human_rejected"]),
                        "evaluation_failed": bool(row["evaluation_failed"]),
                        "failure_category": row["failure_category"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_evaluation_revision(self, data: Dict[str, Any]) -> bool:
        """Persist or update an evaluation revision request."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO evaluation_revisions (
                        revision_id, task_id, failing_agent, failing_stage, feedback,
                        attempt_number, max_attempts, preserved_outputs_json, status,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(revision_id) DO UPDATE SET
                        status = excluded.status,
                        completed_at = excluded.completed_at;
                    """,
                    (
                        data["revision_id"],
                        data["task_id"],
                        data["failing_agent"],
                        data["failing_stage"],
                        data["feedback"],
                        int(data.get("attempt_number", 1)),
                        int(data.get("max_attempts", 2)),
                        self.serialize_payload(data.get("preserved_outputs", {})),
                        data.get("status", "REQUESTED"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        data.get("completed_at"),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save evaluation revision {data.get('revision_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_evaluation_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation revision by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM evaluation_revisions WHERE revision_id = ?;", (revision_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "revision_id": row["revision_id"],
                    "task_id": row["task_id"],
                    "failing_agent": row["failing_agent"],
                    "failing_stage": row["failing_stage"],
                    "feedback": row["feedback"],
                    "attempt_number": row["attempt_number"],
                    "max_attempts": row["max_attempts"],
                    "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                }
            finally:
                conn.close()

    def list_evaluation_revisions(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query evaluation revisions."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM evaluation_revisions WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "revision_id": row["revision_id"],
                        "task_id": row["task_id"],
                        "failing_agent": row["failing_agent"],
                        "failing_stage": row["failing_stage"],
                        "feedback": row["feedback"],
                        "attempt_number": row["attempt_number"],
                        "max_attempts": row["max_attempts"],
                        "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "completed_at": row["completed_at"],
                    })
                return results
            finally:
                conn.close()

    def save_workflow_state(
        self,
        workflow_id: str,
        task_id: str,
        status: str,
        current_stage: str,
        context_data: Any,
    ) -> bool:
        """Persist workflow pause/resume snapshot."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                now_iso = datetime.now(timezone.utc).isoformat()
                ctx_json = context_data if isinstance(context_data, str) else self.serialize_payload(context_data)
                conn.execute(
                    """
                    INSERT INTO workflow_states (
                        workflow_id, task_id, status, current_stage, context_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workflow_id) DO UPDATE SET
                        status = excluded.status,
                        current_stage = excluded.current_stage,
                        context_json = excluded.context_json,
                        updated_at = excluded.updated_at;
                    """,
                    (workflow_id, task_id, status, current_stage, ctx_json, now_iso, now_iso),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save workflow state for {workflow_id}: {exc}")
                raise
            finally:
                conn.close()

    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve persisted workflow pause/resume snapshot."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM workflow_states WHERE workflow_id = ?;", (workflow_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "workflow_id": row["workflow_id"],
                    "task_id": row["task_id"],
                    "status": row["status"],
                    "current_stage": row["current_stage"],
                    "context": self.deserialize_payload(row["context_json"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Day 227 Parts 4-6: Risk Assessments, Reviews, Audits & Intelligence
    # -----------------------------------------------------------------------

    def save_risk_assessment(self, data: Dict[str, Any]) -> bool:
        """Persist or update a RiskAssessmentResult."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO risk_assessments (
                        assessment_id, task_id, action, risk_level, risk_score,
                        identified_risks_json, required_controls_json, approval_required,
                        assessment_reason, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(assessment_id) DO UPDATE SET
                        risk_level = excluded.risk_level,
                        risk_score = excluded.risk_score,
                        identified_risks_json = excluded.identified_risks_json,
                        required_controls_json = excluded.required_controls_json,
                        approval_required = excluded.approval_required,
                        assessment_reason = excluded.assessment_reason,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["assessment_id"],
                        data["task_id"],
                        data["action"],
                        str(data["risk_level"]),
                        float(data["risk_score"]),
                        self.serialize_payload(data.get("identified_risks", [])),
                        self.serialize_payload(data.get("required_controls", [])),
                        1 if data.get("approval_required") else 0,
                        data.get("assessment_reason", ""),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save risk assessment {data.get('assessment_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_risk_assessment(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific risk assessment by its ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM risk_assessments WHERE assessment_id = ?;", (assessment_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "assessment_id": row["assessment_id"],
                    "task_id": row["task_id"],
                    "action": row["action"],
                    "risk_level": row["risk_level"],
                    "risk_score": row["risk_score"],
                    "identified_risks": self.deserialize_payload(row["identified_risks_json"]) or [],
                    "required_controls": self.deserialize_payload(row["required_controls_json"]) or [],
                    "approval_required": bool(row["approval_required"]),
                    "assessment_reason": row["assessment_reason"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_risk_assessments(
        self,
        task_id: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query risk assessments with optional task and risk level filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM risk_assessments WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if risk_level:
                    query += " AND risk_level = ?"
                    params.append(risk_level)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "assessment_id": row["assessment_id"],
                        "task_id": row["task_id"],
                        "action": row["action"],
                        "risk_level": row["risk_level"],
                        "risk_score": row["risk_score"],
                        "identified_risks": self.deserialize_payload(row["identified_risks_json"]) or [],
                        "required_controls": self.deserialize_payload(row["required_controls_json"]) or [],
                        "approval_required": bool(row["approval_required"]),
                        "assessment_reason": row["assessment_reason"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()



    def save_task_outcome(self, data: Dict[str, Any]) -> bool:
        """Persist or update a task outcome evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO task_outcomes (
                        outcome_id, task_id, objective, classification, passed_criteria_json,
                        failed_criteria_json, numeric_checks_json, missing_fields_json,
                        evidence_verified, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(outcome_id) DO UPDATE SET
                        classification = excluded.classification,
                        passed_criteria_json = excluded.passed_criteria_json,
                        failed_criteria_json = excluded.failed_criteria_json,
                        numeric_checks_json = excluded.numeric_checks_json,
                        missing_fields_json = excluded.missing_fields_json,
                        evidence_verified = excluded.evidence_verified,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["outcome_id"],
                        data["task_id"],
                        data["objective"],
                        str(data["classification"]),
                        self.serialize_payload(data.get("passed_criteria", [])),
                        self.serialize_payload(data.get("failed_criteria", [])),
                        self.serialize_payload(data.get("numeric_checks", {})),
                        self.serialize_payload(data.get("missing_fields", [])),
                        1 if data.get("evidence_verified") else 0,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save task outcome {data.get('outcome_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_task_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task outcome evaluation by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM task_outcomes WHERE outcome_id = ?;", (outcome_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "outcome_id": row["outcome_id"],
                    "task_id": row["task_id"],
                    "objective": row["objective"],
                    "classification": row["classification"],
                    "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                    "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                    "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                    "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                    "evidence_verified": bool(row["evidence_verified"]),
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_task_outcomes(
        self,
        task_id: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query task outcome evaluations."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM task_outcomes WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if classification:
                    query += " AND classification = ?"
                    params.append(classification)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "outcome_id": row["outcome_id"],
                        "task_id": row["task_id"],
                        "objective": row["objective"],
                        "classification": row["classification"],
                        "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                        "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                        "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                        "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                        "evidence_verified": bool(row["evidence_verified"]),
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_llm_judge_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update an LLM Judge evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO llm_judge_evaluations (
                        judge_id, task_id, target_agent, target_stage, accuracy, completeness,
                        relevance, instruction_adherence, evidence_support, overall_score,
                        identified_weaknesses_json, confidence, reasoning_summary, evaluator_model,
                        evaluator_provider, status, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(judge_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        confidence = excluded.confidence,
                        reasoning_summary = excluded.reasoning_summary,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["judge_id"],
                        data["task_id"],
                        data["target_agent"],
                        data["target_stage"],
                        float(data.get("accuracy", 0.0)),
                        float(data.get("completeness", 0.0)),
                        float(data.get("relevance", 0.0)),
                        float(data.get("instruction_adherence", 0.0)),
                        float(data.get("evidence_support", 0.0)),
                        float(data.get("overall_score", 0.0)),
                        self.serialize_payload(data.get("identified_weaknesses", [])),
                        float(data.get("confidence", 0.0)),
                        data.get("reasoning_summary", ""),
                        data.get("evaluator_model", ""),
                        data.get("evaluator_provider", ""),
                        data.get("status", "COMPLETED"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save LLM judge evaluation {data.get('judge_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_llm_judge_evaluation(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve LLM judge evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM llm_judge_evaluations WHERE judge_id = ?;", (judge_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "judge_id": row["judge_id"],
                    "task_id": row["task_id"],
                    "target_agent": row["target_agent"],
                    "target_stage": row["target_stage"],
                    "accuracy": row["accuracy"],
                    "completeness": row["completeness"],
                    "relevance": row["relevance"],
                    "instruction_adherence": row["instruction_adherence"],
                    "evidence_support": row["evidence_support"],
                    "overall_score": row["overall_score"],
                    "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                    "confidence": row["confidence"],
                    "reasoning_summary": row["reasoning_summary"],
                    "evaluator_model": row["evaluator_model"],
                    "evaluator_provider": row["evaluator_provider"],
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_llm_judge_evaluations(
        self,
        task_id: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query LLM judge evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM llm_judge_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if target_agent:
                    query += " AND target_agent = ?"
                    params.append(target_agent)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "judge_id": row["judge_id"],
                        "task_id": row["task_id"],
                        "target_agent": row["target_agent"],
                        "target_stage": row["target_stage"],
                        "accuracy": row["accuracy"],
                        "completeness": row["completeness"],
                        "relevance": row["relevance"],
                        "instruction_adherence": row["instruction_adherence"],
                        "evidence_support": row["evidence_support"],
                        "overall_score": row["overall_score"],
                        "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                        "confidence": row["confidence"],
                        "reasoning_summary": row["reasoning_summary"],
                        "evaluator_model": row["evaluator_model"],
                        "evaluator_provider": row["evaluator_provider"],
                        "status": row["status"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_retrieval_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update a retrieval & RAG evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO retrieval_evaluations (
                        retrieval_id, task_id, query, k, precision_at_k, recall_at_k,
                        mrr, ndcg, context_relevance, context_coverage, faithfulness_score,
                        grounded_status, unsupported_claims_json, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(retrieval_id) DO UPDATE SET
                        precision_at_k = excluded.precision_at_k,
                        recall_at_k = excluded.recall_at_k,
                        mrr = excluded.mrr,
                        ndcg = excluded.ndcg,
                        faithfulness_score = excluded.faithfulness_score,
                        grounded_status = excluded.grounded_status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["retrieval_id"],
                        data["task_id"],
                        data["query"],
                        int(data.get("k", 5)),
                        float(data.get("precision_at_k", 0.0)),
                        float(data.get("recall_at_k", 0.0)),
                        float(data.get("mrr", 0.0)),
                        float(data.get("ndcg", 0.0)),
                        float(data.get("context_relevance", 0.0)),
                        float(data.get("context_coverage", 0.0)),
                        float(data.get("faithfulness_score", 0.0)),
                        str(data.get("grounded_status", "GROUNDED")),
                        self.serialize_payload(data.get("unsupported_claims", [])),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save retrieval evaluation {data.get('retrieval_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_retrieval_evaluation(self, retrieval_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve retrieval evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM retrieval_evaluations WHERE retrieval_id = ?;", (retrieval_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "retrieval_id": row["retrieval_id"],
                    "task_id": row["task_id"],
                    "query": row["query"],
                    "k": row["k"],
                    "precision_at_k": row["precision_at_k"],
                    "recall_at_k": row["recall_at_k"],
                    "mrr": row["mrr"],
                    "ndcg": row["ndcg"],
                    "context_relevance": row["context_relevance"],
                    "context_coverage": row["context_coverage"],
                    "faithfulness_score": row["faithfulness_score"],
                    "grounded_status": row["grounded_status"],
                    "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_retrieval_evaluations(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query retrieval evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM retrieval_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "retrieval_id": row["retrieval_id"],
                        "task_id": row["task_id"],
                        "query": row["query"],
                        "k": row["k"],
                        "precision_at_k": row["precision_at_k"],
                        "recall_at_k": row["recall_at_k"],
                        "mrr": row["mrr"],
                        "ndcg": row["ndcg"],
                        "context_relevance": row["context_relevance"],
                        "context_coverage": row["context_coverage"],
                        "faithfulness_score": row["faithfulness_score"],
                        "grounded_status": row["grounded_status"],
                        "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_performance_metric(self, data: Dict[str, Any]) -> bool:
        """Persist an agent performance metric record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                metric_id = data.get("metric_id") or f"pm_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO agent_performance_metrics (
                        metric_id, agent_name, task_type, workflow_id, execution_time_ms,
                        outcome, retry_count, recovery_succeeded, tool_failure, human_intervention,
                        human_rejected, evaluation_failed, failure_category, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_id) DO UPDATE SET
                        execution_time_ms = excluded.execution_time_ms,
                        outcome = excluded.outcome,
                        retry_count = excluded.retry_count,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        metric_id,
                        data["agent_name"],
                        data.get("task_type", "general"),
                        data.get("workflow_id"),
                        float(data.get("execution_time_ms", 0.0)),
                        str(data.get("outcome", "success")),
                        int(data.get("retry_count", 0)),
                        1 if data.get("recovery_succeeded") else 0,
                        1 if data.get("tool_failure") else 0,
                        1 if data.get("human_intervention") else 0,
                        1 if data.get("human_rejected") else 0,
                        1 if data.get("evaluation_failed") else 0,
                        data.get("failure_category"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save performance metric {data.get('metric_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_performance_metrics(
        self,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query performance metrics with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM agent_performance_metrics WHERE 1=1"
                params = []
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "metric_id": row["metric_id"],
                        "agent_name": row["agent_name"],
                        "task_type": row["task_type"],
                        "workflow_id": row["workflow_id"],
                        "execution_time_ms": row["execution_time_ms"],
                        "outcome": row["outcome"],
                        "retry_count": row["retry_count"],
                        "recovery_succeeded": bool(row["recovery_succeeded"]),
                        "tool_failure": bool(row["tool_failure"]),
                        "human_intervention": bool(row["human_intervention"]),
                        "human_rejected": bool(row["human_rejected"]),
                        "evaluation_failed": bool(row["evaluation_failed"]),
                        "failure_category": row["failure_category"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_evaluation_revision(self, data: Dict[str, Any]) -> bool:
        """Persist or update an evaluation revision request."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO evaluation_revisions (
                        revision_id, task_id, failing_agent, failing_stage, feedback,
                        attempt_number, max_attempts, preserved_outputs_json, status,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(revision_id) DO UPDATE SET
                        status = excluded.status,
                        completed_at = excluded.completed_at;
                    """,
                    (
                        data["revision_id"],
                        data["task_id"],
                        data["failing_agent"],
                        data["failing_stage"],
                        data["feedback"],
                        int(data.get("attempt_number", 1)),
                        int(data.get("max_attempts", 2)),
                        self.serialize_payload(data.get("preserved_outputs", {})),
                        data.get("status", "REQUESTED"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        data.get("completed_at"),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save evaluation revision {data.get('revision_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_evaluation_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation revision by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM evaluation_revisions WHERE revision_id = ?;", (revision_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "revision_id": row["revision_id"],
                    "task_id": row["task_id"],
                    "failing_agent": row["failing_agent"],
                    "failing_stage": row["failing_stage"],
                    "feedback": row["feedback"],
                    "attempt_number": row["attempt_number"],
                    "max_attempts": row["max_attempts"],
                    "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                }
            finally:
                conn.close()

    def list_evaluation_revisions(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query evaluation revisions."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM evaluation_revisions WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "revision_id": row["revision_id"],
                        "task_id": row["task_id"],
                        "failing_agent": row["failing_agent"],
                        "failing_stage": row["failing_stage"],
                        "feedback": row["feedback"],
                        "attempt_number": row["attempt_number"],
                        "max_attempts": row["max_attempts"],
                        "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "completed_at": row["completed_at"],
                    })
                return results
            finally:
                conn.close()

    def save_review(self, data: Dict[str, Any]) -> bool:
        """Persist or update a HumanReviewRecord."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO human_reviews (
                        review_id, task_id, correlation_id, target_type, target_data_json,
                        reviewer, outcome, feedback, review_round, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(review_id) DO UPDATE SET
                        outcome = excluded.outcome,
                        feedback = excluded.feedback,
                        review_round = excluded.review_round,
                        timestamp = excluded.timestamp,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["review_id"],
                        data["task_id"],
                        data.get("correlation_id"),
                        data["target_type"],
                        self.serialize_payload(data.get("target_data", {})),
                        data["reviewer"],
                        str(data["outcome"]),
                        data.get("feedback"),
                        int(data.get("review_round", 1)),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save human review {data.get('review_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_review(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a human review record by its ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM human_reviews WHERE review_id = ?;", (review_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "review_id": row["review_id"],
                    "task_id": row["task_id"],
                    "correlation_id": row["correlation_id"],
                    "target_type": row["target_type"],
                    "target_data": self.deserialize_payload(row["target_data_json"]) or {},
                    "reviewer": row["reviewer"],
                    "outcome": row["outcome"],
                    "feedback": row["feedback"],
                    "review_round": row["review_round"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_reviews(
        self,
        task_id: Optional[str] = None,
        outcome: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query human reviews with optional task, outcome, and target_type filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM human_reviews WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if outcome:
                    query += " AND outcome = ?"
                    params.append(outcome)
                if target_type:
                    query += " AND target_type = ?"
                    params.append(target_type)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "review_id": row["review_id"],
                        "task_id": row["task_id"],
                        "correlation_id": row["correlation_id"],
                        "target_type": row["target_type"],
                        "target_data": self.deserialize_payload(row["target_data_json"]) or {},
                        "reviewer": row["reviewer"],
                        "outcome": row["outcome"],
                        "feedback": row["feedback"],
                        "review_round": row["review_round"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()



    def save_task_outcome(self, data: Dict[str, Any]) -> bool:
        """Persist or update a task outcome evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO task_outcomes (
                        outcome_id, task_id, objective, classification, passed_criteria_json,
                        failed_criteria_json, numeric_checks_json, missing_fields_json,
                        evidence_verified, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(outcome_id) DO UPDATE SET
                        classification = excluded.classification,
                        passed_criteria_json = excluded.passed_criteria_json,
                        failed_criteria_json = excluded.failed_criteria_json,
                        numeric_checks_json = excluded.numeric_checks_json,
                        missing_fields_json = excluded.missing_fields_json,
                        evidence_verified = excluded.evidence_verified,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["outcome_id"],
                        data["task_id"],
                        data["objective"],
                        str(data["classification"]),
                        self.serialize_payload(data.get("passed_criteria", [])),
                        self.serialize_payload(data.get("failed_criteria", [])),
                        self.serialize_payload(data.get("numeric_checks", {})),
                        self.serialize_payload(data.get("missing_fields", [])),
                        1 if data.get("evidence_verified") else 0,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save task outcome {data.get('outcome_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_task_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task outcome evaluation by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM task_outcomes WHERE outcome_id = ?;", (outcome_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "outcome_id": row["outcome_id"],
                    "task_id": row["task_id"],
                    "objective": row["objective"],
                    "classification": row["classification"],
                    "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                    "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                    "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                    "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                    "evidence_verified": bool(row["evidence_verified"]),
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_task_outcomes(
        self,
        task_id: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query task outcome evaluations."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM task_outcomes WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if classification:
                    query += " AND classification = ?"
                    params.append(classification)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "outcome_id": row["outcome_id"],
                        "task_id": row["task_id"],
                        "objective": row["objective"],
                        "classification": row["classification"],
                        "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                        "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                        "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                        "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                        "evidence_verified": bool(row["evidence_verified"]),
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_llm_judge_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update an LLM Judge evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO llm_judge_evaluations (
                        judge_id, task_id, target_agent, target_stage, accuracy, completeness,
                        relevance, instruction_adherence, evidence_support, overall_score,
                        identified_weaknesses_json, confidence, reasoning_summary, evaluator_model,
                        evaluator_provider, status, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(judge_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        confidence = excluded.confidence,
                        reasoning_summary = excluded.reasoning_summary,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["judge_id"],
                        data["task_id"],
                        data["target_agent"],
                        data["target_stage"],
                        float(data.get("accuracy", 0.0)),
                        float(data.get("completeness", 0.0)),
                        float(data.get("relevance", 0.0)),
                        float(data.get("instruction_adherence", 0.0)),
                        float(data.get("evidence_support", 0.0)),
                        float(data.get("overall_score", 0.0)),
                        self.serialize_payload(data.get("identified_weaknesses", [])),
                        float(data.get("confidence", 0.0)),
                        data.get("reasoning_summary", ""),
                        data.get("evaluator_model", ""),
                        data.get("evaluator_provider", ""),
                        data.get("status", "COMPLETED"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save LLM judge evaluation {data.get('judge_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_llm_judge_evaluation(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve LLM judge evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM llm_judge_evaluations WHERE judge_id = ?;", (judge_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "judge_id": row["judge_id"],
                    "task_id": row["task_id"],
                    "target_agent": row["target_agent"],
                    "target_stage": row["target_stage"],
                    "accuracy": row["accuracy"],
                    "completeness": row["completeness"],
                    "relevance": row["relevance"],
                    "instruction_adherence": row["instruction_adherence"],
                    "evidence_support": row["evidence_support"],
                    "overall_score": row["overall_score"],
                    "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                    "confidence": row["confidence"],
                    "reasoning_summary": row["reasoning_summary"],
                    "evaluator_model": row["evaluator_model"],
                    "evaluator_provider": row["evaluator_provider"],
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_llm_judge_evaluations(
        self,
        task_id: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query LLM judge evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM llm_judge_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if target_agent:
                    query += " AND target_agent = ?"
                    params.append(target_agent)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "judge_id": row["judge_id"],
                        "task_id": row["task_id"],
                        "target_agent": row["target_agent"],
                        "target_stage": row["target_stage"],
                        "accuracy": row["accuracy"],
                        "completeness": row["completeness"],
                        "relevance": row["relevance"],
                        "instruction_adherence": row["instruction_adherence"],
                        "evidence_support": row["evidence_support"],
                        "overall_score": row["overall_score"],
                        "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                        "confidence": row["confidence"],
                        "reasoning_summary": row["reasoning_summary"],
                        "evaluator_model": row["evaluator_model"],
                        "evaluator_provider": row["evaluator_provider"],
                        "status": row["status"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_retrieval_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update a retrieval & RAG evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO retrieval_evaluations (
                        retrieval_id, task_id, query, k, precision_at_k, recall_at_k,
                        mrr, ndcg, context_relevance, context_coverage, faithfulness_score,
                        grounded_status, unsupported_claims_json, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(retrieval_id) DO UPDATE SET
                        precision_at_k = excluded.precision_at_k,
                        recall_at_k = excluded.recall_at_k,
                        mrr = excluded.mrr,
                        ndcg = excluded.ndcg,
                        faithfulness_score = excluded.faithfulness_score,
                        grounded_status = excluded.grounded_status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["retrieval_id"],
                        data["task_id"],
                        data["query"],
                        int(data.get("k", 5)),
                        float(data.get("precision_at_k", 0.0)),
                        float(data.get("recall_at_k", 0.0)),
                        float(data.get("mrr", 0.0)),
                        float(data.get("ndcg", 0.0)),
                        float(data.get("context_relevance", 0.0)),
                        float(data.get("context_coverage", 0.0)),
                        float(data.get("faithfulness_score", 0.0)),
                        str(data.get("grounded_status", "GROUNDED")),
                        self.serialize_payload(data.get("unsupported_claims", [])),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save retrieval evaluation {data.get('retrieval_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_retrieval_evaluation(self, retrieval_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve retrieval evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM retrieval_evaluations WHERE retrieval_id = ?;", (retrieval_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "retrieval_id": row["retrieval_id"],
                    "task_id": row["task_id"],
                    "query": row["query"],
                    "k": row["k"],
                    "precision_at_k": row["precision_at_k"],
                    "recall_at_k": row["recall_at_k"],
                    "mrr": row["mrr"],
                    "ndcg": row["ndcg"],
                    "context_relevance": row["context_relevance"],
                    "context_coverage": row["context_coverage"],
                    "faithfulness_score": row["faithfulness_score"],
                    "grounded_status": row["grounded_status"],
                    "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_retrieval_evaluations(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query retrieval evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM retrieval_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "retrieval_id": row["retrieval_id"],
                        "task_id": row["task_id"],
                        "query": row["query"],
                        "k": row["k"],
                        "precision_at_k": row["precision_at_k"],
                        "recall_at_k": row["recall_at_k"],
                        "mrr": row["mrr"],
                        "ndcg": row["ndcg"],
                        "context_relevance": row["context_relevance"],
                        "context_coverage": row["context_coverage"],
                        "faithfulness_score": row["faithfulness_score"],
                        "grounded_status": row["grounded_status"],
                        "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_performance_metric(self, data: Dict[str, Any]) -> bool:
        """Persist an agent performance metric record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                metric_id = data.get("metric_id") or f"pm_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO agent_performance_metrics (
                        metric_id, agent_name, task_type, workflow_id, execution_time_ms,
                        outcome, retry_count, recovery_succeeded, tool_failure, human_intervention,
                        human_rejected, evaluation_failed, failure_category, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_id) DO UPDATE SET
                        execution_time_ms = excluded.execution_time_ms,
                        outcome = excluded.outcome,
                        retry_count = excluded.retry_count,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        metric_id,
                        data["agent_name"],
                        data.get("task_type", "general"),
                        data.get("workflow_id"),
                        float(data.get("execution_time_ms", 0.0)),
                        str(data.get("outcome", "success")),
                        int(data.get("retry_count", 0)),
                        1 if data.get("recovery_succeeded") else 0,
                        1 if data.get("tool_failure") else 0,
                        1 if data.get("human_intervention") else 0,
                        1 if data.get("human_rejected") else 0,
                        1 if data.get("evaluation_failed") else 0,
                        data.get("failure_category"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save performance metric {data.get('metric_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_performance_metrics(
        self,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query performance metrics with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM agent_performance_metrics WHERE 1=1"
                params = []
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "metric_id": row["metric_id"],
                        "agent_name": row["agent_name"],
                        "task_type": row["task_type"],
                        "workflow_id": row["workflow_id"],
                        "execution_time_ms": row["execution_time_ms"],
                        "outcome": row["outcome"],
                        "retry_count": row["retry_count"],
                        "recovery_succeeded": bool(row["recovery_succeeded"]),
                        "tool_failure": bool(row["tool_failure"]),
                        "human_intervention": bool(row["human_intervention"]),
                        "human_rejected": bool(row["human_rejected"]),
                        "evaluation_failed": bool(row["evaluation_failed"]),
                        "failure_category": row["failure_category"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_evaluation_revision(self, data: Dict[str, Any]) -> bool:
        """Persist or update an evaluation revision request."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO evaluation_revisions (
                        revision_id, task_id, failing_agent, failing_stage, feedback,
                        attempt_number, max_attempts, preserved_outputs_json, status,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(revision_id) DO UPDATE SET
                        status = excluded.status,
                        completed_at = excluded.completed_at;
                    """,
                    (
                        data["revision_id"],
                        data["task_id"],
                        data["failing_agent"],
                        data["failing_stage"],
                        data["feedback"],
                        int(data.get("attempt_number", 1)),
                        int(data.get("max_attempts", 2)),
                        self.serialize_payload(data.get("preserved_outputs", {})),
                        data.get("status", "REQUESTED"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        data.get("completed_at"),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save evaluation revision {data.get('revision_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_evaluation_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation revision by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM evaluation_revisions WHERE revision_id = ?;", (revision_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "revision_id": row["revision_id"],
                    "task_id": row["task_id"],
                    "failing_agent": row["failing_agent"],
                    "failing_stage": row["failing_stage"],
                    "feedback": row["feedback"],
                    "attempt_number": row["attempt_number"],
                    "max_attempts": row["max_attempts"],
                    "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                }
            finally:
                conn.close()

    def list_evaluation_revisions(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query evaluation revisions."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM evaluation_revisions WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "revision_id": row["revision_id"],
                        "task_id": row["task_id"],
                        "failing_agent": row["failing_agent"],
                        "failing_stage": row["failing_stage"],
                        "feedback": row["feedback"],
                        "attempt_number": row["attempt_number"],
                        "max_attempts": row["max_attempts"],
                        "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "completed_at": row["completed_at"],
                    })
                return results
            finally:
                conn.close()

    def save_audit_event(self, data: Dict[str, Any]) -> bool:
        """Append an event to the unified audit trail (append-only)."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                event_uuid = data.get("event_uuid") or f"aud_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO audit_events (
                        event_uuid, task_id, correlation_id, event_type,
                        agent_name, description, payload_json, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event_uuid,
                        data["task_id"],
                        data.get("correlation_id"),
                        str(data["event_type"]),
                        data.get("agent_name"),
                        data["description"],
                        self.serialize_payload(data.get("payload", {})),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save audit event: {exc}")
                raise
            finally:
                conn.close()

    def list_audit_events(
        self,
        task_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Filter audit trail events with temporal and contextual filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if event_type:
                    query += " AND event_type = ?"
                    params.append(str(event_type))
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY event_id ASC LIMIT ?;"
                params.append(limit)

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "event_id": row["event_id"],
                        "event_uuid": row["event_uuid"],
                        "task_id": row["task_id"],
                        "correlation_id": row["correlation_id"],
                        "event_type": row["event_type"],
                        "agent_name": row["agent_name"],
                        "description": row["description"],
                        "payload": self.deserialize_payload(row["payload_json"]) or {},
                        "timestamp": row["timestamp"],
                    })
                return results
            finally:
                conn.close()



    def save_task_outcome(self, data: Dict[str, Any]) -> bool:
        """Persist or update a task outcome evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO task_outcomes (
                        outcome_id, task_id, objective, classification, passed_criteria_json,
                        failed_criteria_json, numeric_checks_json, missing_fields_json,
                        evidence_verified, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(outcome_id) DO UPDATE SET
                        classification = excluded.classification,
                        passed_criteria_json = excluded.passed_criteria_json,
                        failed_criteria_json = excluded.failed_criteria_json,
                        numeric_checks_json = excluded.numeric_checks_json,
                        missing_fields_json = excluded.missing_fields_json,
                        evidence_verified = excluded.evidence_verified,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["outcome_id"],
                        data["task_id"],
                        data["objective"],
                        str(data["classification"]),
                        self.serialize_payload(data.get("passed_criteria", [])),
                        self.serialize_payload(data.get("failed_criteria", [])),
                        self.serialize_payload(data.get("numeric_checks", {})),
                        self.serialize_payload(data.get("missing_fields", [])),
                        1 if data.get("evidence_verified") else 0,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save task outcome {data.get('outcome_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_task_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task outcome evaluation by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM task_outcomes WHERE outcome_id = ?;", (outcome_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "outcome_id": row["outcome_id"],
                    "task_id": row["task_id"],
                    "objective": row["objective"],
                    "classification": row["classification"],
                    "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                    "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                    "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                    "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                    "evidence_verified": bool(row["evidence_verified"]),
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_task_outcomes(
        self,
        task_id: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query task outcome evaluations."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM task_outcomes WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if classification:
                    query += " AND classification = ?"
                    params.append(classification)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "outcome_id": row["outcome_id"],
                        "task_id": row["task_id"],
                        "objective": row["objective"],
                        "classification": row["classification"],
                        "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                        "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                        "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                        "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                        "evidence_verified": bool(row["evidence_verified"]),
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_llm_judge_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update an LLM Judge evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO llm_judge_evaluations (
                        judge_id, task_id, target_agent, target_stage, accuracy, completeness,
                        relevance, instruction_adherence, evidence_support, overall_score,
                        identified_weaknesses_json, confidence, reasoning_summary, evaluator_model,
                        evaluator_provider, status, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(judge_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        confidence = excluded.confidence,
                        reasoning_summary = excluded.reasoning_summary,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["judge_id"],
                        data["task_id"],
                        data["target_agent"],
                        data["target_stage"],
                        float(data.get("accuracy", 0.0)),
                        float(data.get("completeness", 0.0)),
                        float(data.get("relevance", 0.0)),
                        float(data.get("instruction_adherence", 0.0)),
                        float(data.get("evidence_support", 0.0)),
                        float(data.get("overall_score", 0.0)),
                        self.serialize_payload(data.get("identified_weaknesses", [])),
                        float(data.get("confidence", 0.0)),
                        data.get("reasoning_summary", ""),
                        data.get("evaluator_model", ""),
                        data.get("evaluator_provider", ""),
                        data.get("status", "COMPLETED"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save LLM judge evaluation {data.get('judge_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_llm_judge_evaluation(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve LLM judge evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM llm_judge_evaluations WHERE judge_id = ?;", (judge_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "judge_id": row["judge_id"],
                    "task_id": row["task_id"],
                    "target_agent": row["target_agent"],
                    "target_stage": row["target_stage"],
                    "accuracy": row["accuracy"],
                    "completeness": row["completeness"],
                    "relevance": row["relevance"],
                    "instruction_adherence": row["instruction_adherence"],
                    "evidence_support": row["evidence_support"],
                    "overall_score": row["overall_score"],
                    "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                    "confidence": row["confidence"],
                    "reasoning_summary": row["reasoning_summary"],
                    "evaluator_model": row["evaluator_model"],
                    "evaluator_provider": row["evaluator_provider"],
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_llm_judge_evaluations(
        self,
        task_id: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query LLM judge evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM llm_judge_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if target_agent:
                    query += " AND target_agent = ?"
                    params.append(target_agent)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "judge_id": row["judge_id"],
                        "task_id": row["task_id"],
                        "target_agent": row["target_agent"],
                        "target_stage": row["target_stage"],
                        "accuracy": row["accuracy"],
                        "completeness": row["completeness"],
                        "relevance": row["relevance"],
                        "instruction_adherence": row["instruction_adherence"],
                        "evidence_support": row["evidence_support"],
                        "overall_score": row["overall_score"],
                        "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                        "confidence": row["confidence"],
                        "reasoning_summary": row["reasoning_summary"],
                        "evaluator_model": row["evaluator_model"],
                        "evaluator_provider": row["evaluator_provider"],
                        "status": row["status"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_retrieval_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update a retrieval & RAG evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO retrieval_evaluations (
                        retrieval_id, task_id, query, k, precision_at_k, recall_at_k,
                        mrr, ndcg, context_relevance, context_coverage, faithfulness_score,
                        grounded_status, unsupported_claims_json, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(retrieval_id) DO UPDATE SET
                        precision_at_k = excluded.precision_at_k,
                        recall_at_k = excluded.recall_at_k,
                        mrr = excluded.mrr,
                        ndcg = excluded.ndcg,
                        faithfulness_score = excluded.faithfulness_score,
                        grounded_status = excluded.grounded_status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["retrieval_id"],
                        data["task_id"],
                        data["query"],
                        int(data.get("k", 5)),
                        float(data.get("precision_at_k", 0.0)),
                        float(data.get("recall_at_k", 0.0)),
                        float(data.get("mrr", 0.0)),
                        float(data.get("ndcg", 0.0)),
                        float(data.get("context_relevance", 0.0)),
                        float(data.get("context_coverage", 0.0)),
                        float(data.get("faithfulness_score", 0.0)),
                        str(data.get("grounded_status", "GROUNDED")),
                        self.serialize_payload(data.get("unsupported_claims", [])),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save retrieval evaluation {data.get('retrieval_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_retrieval_evaluation(self, retrieval_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve retrieval evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM retrieval_evaluations WHERE retrieval_id = ?;", (retrieval_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "retrieval_id": row["retrieval_id"],
                    "task_id": row["task_id"],
                    "query": row["query"],
                    "k": row["k"],
                    "precision_at_k": row["precision_at_k"],
                    "recall_at_k": row["recall_at_k"],
                    "mrr": row["mrr"],
                    "ndcg": row["ndcg"],
                    "context_relevance": row["context_relevance"],
                    "context_coverage": row["context_coverage"],
                    "faithfulness_score": row["faithfulness_score"],
                    "grounded_status": row["grounded_status"],
                    "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_retrieval_evaluations(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query retrieval evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM retrieval_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "retrieval_id": row["retrieval_id"],
                        "task_id": row["task_id"],
                        "query": row["query"],
                        "k": row["k"],
                        "precision_at_k": row["precision_at_k"],
                        "recall_at_k": row["recall_at_k"],
                        "mrr": row["mrr"],
                        "ndcg": row["ndcg"],
                        "context_relevance": row["context_relevance"],
                        "context_coverage": row["context_coverage"],
                        "faithfulness_score": row["faithfulness_score"],
                        "grounded_status": row["grounded_status"],
                        "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_performance_metric(self, data: Dict[str, Any]) -> bool:
        """Persist an agent performance metric record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                metric_id = data.get("metric_id") or f"pm_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO agent_performance_metrics (
                        metric_id, agent_name, task_type, workflow_id, execution_time_ms,
                        outcome, retry_count, recovery_succeeded, tool_failure, human_intervention,
                        human_rejected, evaluation_failed, failure_category, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_id) DO UPDATE SET
                        execution_time_ms = excluded.execution_time_ms,
                        outcome = excluded.outcome,
                        retry_count = excluded.retry_count,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        metric_id,
                        data["agent_name"],
                        data.get("task_type", "general"),
                        data.get("workflow_id"),
                        float(data.get("execution_time_ms", 0.0)),
                        str(data.get("outcome", "success")),
                        int(data.get("retry_count", 0)),
                        1 if data.get("recovery_succeeded") else 0,
                        1 if data.get("tool_failure") else 0,
                        1 if data.get("human_intervention") else 0,
                        1 if data.get("human_rejected") else 0,
                        1 if data.get("evaluation_failed") else 0,
                        data.get("failure_category"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save performance metric {data.get('metric_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_performance_metrics(
        self,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query performance metrics with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM agent_performance_metrics WHERE 1=1"
                params = []
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "metric_id": row["metric_id"],
                        "agent_name": row["agent_name"],
                        "task_type": row["task_type"],
                        "workflow_id": row["workflow_id"],
                        "execution_time_ms": row["execution_time_ms"],
                        "outcome": row["outcome"],
                        "retry_count": row["retry_count"],
                        "recovery_succeeded": bool(row["recovery_succeeded"]),
                        "tool_failure": bool(row["tool_failure"]),
                        "human_intervention": bool(row["human_intervention"]),
                        "human_rejected": bool(row["human_rejected"]),
                        "evaluation_failed": bool(row["evaluation_failed"]),
                        "failure_category": row["failure_category"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_evaluation_revision(self, data: Dict[str, Any]) -> bool:
        """Persist or update an evaluation revision request."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO evaluation_revisions (
                        revision_id, task_id, failing_agent, failing_stage, feedback,
                        attempt_number, max_attempts, preserved_outputs_json, status,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(revision_id) DO UPDATE SET
                        status = excluded.status,
                        completed_at = excluded.completed_at;
                    """,
                    (
                        data["revision_id"],
                        data["task_id"],
                        data["failing_agent"],
                        data["failing_stage"],
                        data["feedback"],
                        int(data.get("attempt_number", 1)),
                        int(data.get("max_attempts", 2)),
                        self.serialize_payload(data.get("preserved_outputs", {})),
                        data.get("status", "REQUESTED"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        data.get("completed_at"),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save evaluation revision {data.get('revision_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_evaluation_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation revision by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM evaluation_revisions WHERE revision_id = ?;", (revision_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "revision_id": row["revision_id"],
                    "task_id": row["task_id"],
                    "failing_agent": row["failing_agent"],
                    "failing_stage": row["failing_stage"],
                    "feedback": row["feedback"],
                    "attempt_number": row["attempt_number"],
                    "max_attempts": row["max_attempts"],
                    "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                }
            finally:
                conn.close()

    def list_evaluation_revisions(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query evaluation revisions."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM evaluation_revisions WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "revision_id": row["revision_id"],
                        "task_id": row["task_id"],
                        "failing_agent": row["failing_agent"],
                        "failing_stage": row["failing_stage"],
                        "feedback": row["feedback"],
                        "attempt_number": row["attempt_number"],
                        "max_attempts": row["max_attempts"],
                        "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "completed_at": row["completed_at"],
                    })
                return results
            finally:
                conn.close()

    def save_feedback_analysis(self, data: Dict[str, Any]) -> bool:
        """Persist a FeedbackSummary result."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO feedback_analyses (
                        analysis_id, summary_title, time_window_start, time_window_end,
                        total_reviews, total_feedback_items, recurring_issue_categories_json,
                        agent_failure_rates_json, observed_patterns_json, actionable_recommendations_json,
                        generated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(analysis_id) DO UPDATE SET
                        summary_title = excluded.summary_title,
                        total_reviews = excluded.total_reviews,
                        total_feedback_items = excluded.total_feedback_items,
                        recurring_issue_categories_json = excluded.recurring_issue_categories_json,
                        agent_failure_rates_json = excluded.agent_failure_rates_json,
                        observed_patterns_json = excluded.observed_patterns_json,
                        actionable_recommendations_json = excluded.actionable_recommendations_json,
                        generated_at = excluded.generated_at,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["analysis_id"],
                        data["summary_title"],
                        data.get("time_window_start"),
                        data.get("time_window_end"),
                        int(data.get("total_reviews", 0)),
                        int(data.get("total_feedback_items", 0)),
                        self.serialize_payload(data.get("recurring_issue_categories", [])),
                        self.serialize_payload(data.get("agent_failure_rates", {})),
                        self.serialize_payload(data.get("observed_patterns", [])),
                        self.serialize_payload(data.get("actionable_recommendations", [])),
                        data.get("generated_at") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save feedback analysis {data.get('analysis_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_feedback_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve feedback analysis summary by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM feedback_analyses WHERE analysis_id = ?;", (analysis_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "analysis_id": row["analysis_id"],
                    "summary_title": row["summary_title"],
                    "time_window_start": row["time_window_start"],
                    "time_window_end": row["time_window_end"],
                    "total_reviews": row["total_reviews"],
                    "total_feedback_items": row["total_feedback_items"],
                    "recurring_issue_categories": self.deserialize_payload(row["recurring_issue_categories_json"]) or [],
                    "agent_failure_rates": self.deserialize_payload(row["agent_failure_rates_json"]) or {},
                    "observed_patterns": self.deserialize_payload(row["observed_patterns_json"]) or [],
                    "actionable_recommendations": self.deserialize_payload(row["actionable_recommendations_json"]) or [],
                    "generated_at": row["generated_at"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_feedback_analyses(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent feedback analyses."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM feedback_analyses ORDER BY generated_at DESC LIMIT ?;",
                    (limit,),
                )
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "analysis_id": row["analysis_id"],
                        "summary_title": row["summary_title"],
                        "time_window_start": row["time_window_start"],
                        "time_window_end": row["time_window_end"],
                        "total_reviews": row["total_reviews"],
                        "total_feedback_items": row["total_feedback_items"],
                        "recurring_issue_categories": self.deserialize_payload(row["recurring_issue_categories_json"]) or [],
                        "agent_failure_rates": self.deserialize_payload(row["agent_failure_rates_json"]) or {},
                        "observed_patterns": self.deserialize_payload(row["observed_patterns_json"]) or [],
                        "actionable_recommendations": self.deserialize_payload(row["actionable_recommendations_json"]) or [],
                        "generated_at": row["generated_at"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()



    def save_task_outcome(self, data: Dict[str, Any]) -> bool:
        """Persist or update a task outcome evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO task_outcomes (
                        outcome_id, task_id, objective, classification, passed_criteria_json,
                        failed_criteria_json, numeric_checks_json, missing_fields_json,
                        evidence_verified, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(outcome_id) DO UPDATE SET
                        classification = excluded.classification,
                        passed_criteria_json = excluded.passed_criteria_json,
                        failed_criteria_json = excluded.failed_criteria_json,
                        numeric_checks_json = excluded.numeric_checks_json,
                        missing_fields_json = excluded.missing_fields_json,
                        evidence_verified = excluded.evidence_verified,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["outcome_id"],
                        data["task_id"],
                        data["objective"],
                        str(data["classification"]),
                        self.serialize_payload(data.get("passed_criteria", [])),
                        self.serialize_payload(data.get("failed_criteria", [])),
                        self.serialize_payload(data.get("numeric_checks", {})),
                        self.serialize_payload(data.get("missing_fields", [])),
                        1 if data.get("evidence_verified") else 0,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save task outcome {data.get('outcome_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_task_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task outcome evaluation by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM task_outcomes WHERE outcome_id = ?;", (outcome_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "outcome_id": row["outcome_id"],
                    "task_id": row["task_id"],
                    "objective": row["objective"],
                    "classification": row["classification"],
                    "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                    "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                    "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                    "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                    "evidence_verified": bool(row["evidence_verified"]),
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_task_outcomes(
        self,
        task_id: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query task outcome evaluations."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM task_outcomes WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if classification:
                    query += " AND classification = ?"
                    params.append(classification)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "outcome_id": row["outcome_id"],
                        "task_id": row["task_id"],
                        "objective": row["objective"],
                        "classification": row["classification"],
                        "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                        "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                        "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                        "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                        "evidence_verified": bool(row["evidence_verified"]),
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_llm_judge_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update an LLM Judge evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO llm_judge_evaluations (
                        judge_id, task_id, target_agent, target_stage, accuracy, completeness,
                        relevance, instruction_adherence, evidence_support, overall_score,
                        identified_weaknesses_json, confidence, reasoning_summary, evaluator_model,
                        evaluator_provider, status, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(judge_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        confidence = excluded.confidence,
                        reasoning_summary = excluded.reasoning_summary,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["judge_id"],
                        data["task_id"],
                        data["target_agent"],
                        data["target_stage"],
                        float(data.get("accuracy", 0.0)),
                        float(data.get("completeness", 0.0)),
                        float(data.get("relevance", 0.0)),
                        float(data.get("instruction_adherence", 0.0)),
                        float(data.get("evidence_support", 0.0)),
                        float(data.get("overall_score", 0.0)),
                        self.serialize_payload(data.get("identified_weaknesses", [])),
                        float(data.get("confidence", 0.0)),
                        data.get("reasoning_summary", ""),
                        data.get("evaluator_model", ""),
                        data.get("evaluator_provider", ""),
                        data.get("status", "COMPLETED"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save LLM judge evaluation {data.get('judge_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_llm_judge_evaluation(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve LLM judge evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM llm_judge_evaluations WHERE judge_id = ?;", (judge_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "judge_id": row["judge_id"],
                    "task_id": row["task_id"],
                    "target_agent": row["target_agent"],
                    "target_stage": row["target_stage"],
                    "accuracy": row["accuracy"],
                    "completeness": row["completeness"],
                    "relevance": row["relevance"],
                    "instruction_adherence": row["instruction_adherence"],
                    "evidence_support": row["evidence_support"],
                    "overall_score": row["overall_score"],
                    "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                    "confidence": row["confidence"],
                    "reasoning_summary": row["reasoning_summary"],
                    "evaluator_model": row["evaluator_model"],
                    "evaluator_provider": row["evaluator_provider"],
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_llm_judge_evaluations(
        self,
        task_id: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query LLM judge evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM llm_judge_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if target_agent:
                    query += " AND target_agent = ?"
                    params.append(target_agent)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "judge_id": row["judge_id"],
                        "task_id": row["task_id"],
                        "target_agent": row["target_agent"],
                        "target_stage": row["target_stage"],
                        "accuracy": row["accuracy"],
                        "completeness": row["completeness"],
                        "relevance": row["relevance"],
                        "instruction_adherence": row["instruction_adherence"],
                        "evidence_support": row["evidence_support"],
                        "overall_score": row["overall_score"],
                        "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                        "confidence": row["confidence"],
                        "reasoning_summary": row["reasoning_summary"],
                        "evaluator_model": row["evaluator_model"],
                        "evaluator_provider": row["evaluator_provider"],
                        "status": row["status"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_retrieval_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update a retrieval & RAG evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO retrieval_evaluations (
                        retrieval_id, task_id, query, k, precision_at_k, recall_at_k,
                        mrr, ndcg, context_relevance, context_coverage, faithfulness_score,
                        grounded_status, unsupported_claims_json, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(retrieval_id) DO UPDATE SET
                        precision_at_k = excluded.precision_at_k,
                        recall_at_k = excluded.recall_at_k,
                        mrr = excluded.mrr,
                        ndcg = excluded.ndcg,
                        faithfulness_score = excluded.faithfulness_score,
                        grounded_status = excluded.grounded_status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["retrieval_id"],
                        data["task_id"],
                        data["query"],
                        int(data.get("k", 5)),
                        float(data.get("precision_at_k", 0.0)),
                        float(data.get("recall_at_k", 0.0)),
                        float(data.get("mrr", 0.0)),
                        float(data.get("ndcg", 0.0)),
                        float(data.get("context_relevance", 0.0)),
                        float(data.get("context_coverage", 0.0)),
                        float(data.get("faithfulness_score", 0.0)),
                        str(data.get("grounded_status", "GROUNDED")),
                        self.serialize_payload(data.get("unsupported_claims", [])),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save retrieval evaluation {data.get('retrieval_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_retrieval_evaluation(self, retrieval_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve retrieval evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM retrieval_evaluations WHERE retrieval_id = ?;", (retrieval_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "retrieval_id": row["retrieval_id"],
                    "task_id": row["task_id"],
                    "query": row["query"],
                    "k": row["k"],
                    "precision_at_k": row["precision_at_k"],
                    "recall_at_k": row["recall_at_k"],
                    "mrr": row["mrr"],
                    "ndcg": row["ndcg"],
                    "context_relevance": row["context_relevance"],
                    "context_coverage": row["context_coverage"],
                    "faithfulness_score": row["faithfulness_score"],
                    "grounded_status": row["grounded_status"],
                    "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_retrieval_evaluations(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query retrieval evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM retrieval_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "retrieval_id": row["retrieval_id"],
                        "task_id": row["task_id"],
                        "query": row["query"],
                        "k": row["k"],
                        "precision_at_k": row["precision_at_k"],
                        "recall_at_k": row["recall_at_k"],
                        "mrr": row["mrr"],
                        "ndcg": row["ndcg"],
                        "context_relevance": row["context_relevance"],
                        "context_coverage": row["context_coverage"],
                        "faithfulness_score": row["faithfulness_score"],
                        "grounded_status": row["grounded_status"],
                        "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_performance_metric(self, data: Dict[str, Any]) -> bool:
        """Persist an agent performance metric record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                metric_id = data.get("metric_id") or f"pm_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO agent_performance_metrics (
                        metric_id, agent_name, task_type, workflow_id, execution_time_ms,
                        outcome, retry_count, recovery_succeeded, tool_failure, human_intervention,
                        human_rejected, evaluation_failed, failure_category, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_id) DO UPDATE SET
                        execution_time_ms = excluded.execution_time_ms,
                        outcome = excluded.outcome,
                        retry_count = excluded.retry_count,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        metric_id,
                        data["agent_name"],
                        data.get("task_type", "general"),
                        data.get("workflow_id"),
                        float(data.get("execution_time_ms", 0.0)),
                        str(data.get("outcome", "success")),
                        int(data.get("retry_count", 0)),
                        1 if data.get("recovery_succeeded") else 0,
                        1 if data.get("tool_failure") else 0,
                        1 if data.get("human_intervention") else 0,
                        1 if data.get("human_rejected") else 0,
                        1 if data.get("evaluation_failed") else 0,
                        data.get("failure_category"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save performance metric {data.get('metric_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_performance_metrics(
        self,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query performance metrics with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM agent_performance_metrics WHERE 1=1"
                params = []
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "metric_id": row["metric_id"],
                        "agent_name": row["agent_name"],
                        "task_type": row["task_type"],
                        "workflow_id": row["workflow_id"],
                        "execution_time_ms": row["execution_time_ms"],
                        "outcome": row["outcome"],
                        "retry_count": row["retry_count"],
                        "recovery_succeeded": bool(row["recovery_succeeded"]),
                        "tool_failure": bool(row["tool_failure"]),
                        "human_intervention": bool(row["human_intervention"]),
                        "human_rejected": bool(row["human_rejected"]),
                        "evaluation_failed": bool(row["evaluation_failed"]),
                        "failure_category": row["failure_category"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_evaluation_revision(self, data: Dict[str, Any]) -> bool:
        """Persist or update an evaluation revision request."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO evaluation_revisions (
                        revision_id, task_id, failing_agent, failing_stage, feedback,
                        attempt_number, max_attempts, preserved_outputs_json, status,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(revision_id) DO UPDATE SET
                        status = excluded.status,
                        completed_at = excluded.completed_at;
                    """,
                    (
                        data["revision_id"],
                        data["task_id"],
                        data["failing_agent"],
                        data["failing_stage"],
                        data["feedback"],
                        int(data.get("attempt_number", 1)),
                        int(data.get("max_attempts", 2)),
                        self.serialize_payload(data.get("preserved_outputs", {})),
                        data.get("status", "REQUESTED"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        data.get("completed_at"),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save evaluation revision {data.get('revision_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_evaluation_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation revision by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM evaluation_revisions WHERE revision_id = ?;", (revision_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "revision_id": row["revision_id"],
                    "task_id": row["task_id"],
                    "failing_agent": row["failing_agent"],
                    "failing_stage": row["failing_stage"],
                    "feedback": row["feedback"],
                    "attempt_number": row["attempt_number"],
                    "max_attempts": row["max_attempts"],
                    "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                }
            finally:
                conn.close()

    def list_evaluation_revisions(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query evaluation revisions."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM evaluation_revisions WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "revision_id": row["revision_id"],
                        "task_id": row["task_id"],
                        "failing_agent": row["failing_agent"],
                        "failing_stage": row["failing_stage"],
                        "feedback": row["feedback"],
                        "attempt_number": row["attempt_number"],
                        "max_attempts": row["max_attempts"],
                        "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "completed_at": row["completed_at"],
                    })
                return results
            finally:
                conn.close()

    def save_policy_proposal(self, data: Dict[str, Any]) -> bool:
        """Persist or update a PolicyChangeProposal."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO policy_change_proposals (
                        proposal_id, analysis_id, proposed_change, target_policy,
                        rationale, status, authorized_by, authorized_at, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(proposal_id) DO UPDATE SET
                        proposed_change = excluded.proposed_change,
                        status = excluded.status,
                        authorized_by = excluded.authorized_by,
                        authorized_at = excluded.authorized_at,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["proposal_id"],
                        data.get("analysis_id"),
                        data["proposed_change"],
                        data["target_policy"],
                        data["rationale"],
                        data.get("status", "PROPOSED"),
                        data.get("authorized_by"),
                        data.get("authorized_at"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save policy proposal {data.get('proposal_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_policy_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve policy proposal by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM policy_change_proposals WHERE proposal_id = ?;", (proposal_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "proposal_id": row["proposal_id"],
                    "analysis_id": row["analysis_id"],
                    "proposed_change": row["proposed_change"],
                    "target_policy": row["target_policy"],
                    "rationale": row["rationale"],
                    "status": row["status"],
                    "authorized_by": row["authorized_by"],
                    "authorized_at": row["authorized_at"],
                    "created_at": row["created_at"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def update_policy_proposal(
        self,
        proposal_id: str,
        status: str,
        authorized_by: Optional[str] = None,
        authorized_at: Optional[str] = None,
    ) -> bool:
        """Update proposal approval status and authorization signature."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                now_iso = authorized_at or (datetime.now(timezone.utc).isoformat() if authorized_by else None)
                conn.execute(
                    """
                    UPDATE policy_change_proposals
                    SET status = ?, authorized_by = ?, authorized_at = ?
                    WHERE proposal_id = ?;
                    """,
                    (status, authorized_by, now_iso, proposal_id),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to update policy proposal {proposal_id}: {exc}")
                raise
            finally:
                conn.close()

    def list_policy_proposals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query policy change proposals with optional status filter."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM policy_change_proposals WHERE 1=1"
                params = []
                if status:
                    query += " AND status = ?"
                    params.append(status)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "proposal_id": row["proposal_id"],
                        "analysis_id": row["analysis_id"],
                        "proposed_change": row["proposed_change"],
                        "target_policy": row["target_policy"],
                        "rationale": row["rationale"],
                        "status": row["status"],
                        "authorized_by": row["authorized_by"],
                        "authorized_at": row["authorized_at"],
                        "created_at": row["created_at"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Day 228: Agent Evaluation Framework Persistence
    # -----------------------------------------------------------------------

    def save_agent_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or upsert an Agent EvaluationResult."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO agent_evaluations (
                        evaluation_id, task_id, agent_name, evaluation_type, overall_score,
                        metric_scores_json, passed, failures_json, warnings_json, evidence_json,
                        evaluation_reason, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(evaluation_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        metric_scores_json = excluded.metric_scores_json,
                        passed = excluded.passed,
                        failures_json = excluded.failures_json,
                        warnings_json = excluded.warnings_json,
                        evidence_json = excluded.evidence_json,
                        evaluation_reason = excluded.evaluation_reason,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["evaluation_id"],
                        data["task_id"],
                        data.get("agent_name", ""),
                        data.get("evaluation_type", "task_outcome"),
                        float(data.get("overall_score", 0.0)),
                        self.serialize_payload(data.get("metric_scores", {})),
                        1 if data.get("passed") else 0,
                        self.serialize_payload(data.get("failures", [])),
                        self.serialize_payload(data.get("warnings", [])),
                        self.serialize_payload(data.get("evidence", {})),
                        data.get("evaluation_reason", ""),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save agent evaluation {data.get('evaluation_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_agent_evaluation(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve agent evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM agent_evaluations WHERE evaluation_id = ?;", (evaluation_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "evaluation_id": row["evaluation_id"],
                    "task_id": row["task_id"],
                    "agent_name": row["agent_name"],
                    "evaluation_type": row["evaluation_type"],
                    "overall_score": row["overall_score"],
                    "metric_scores": self.deserialize_payload(row["metric_scores_json"]) or {},
                    "passed": bool(row["passed"]),
                    "failures": self.deserialize_payload(row["failures_json"]) or [],
                    "warnings": self.deserialize_payload(row["warnings_json"]) or [],
                    "evidence": self.deserialize_payload(row["evidence_json"]) or {},
                    "evaluation_reason": row["evaluation_reason"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_agent_evaluations(
        self,
        task_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        passed: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query agent evaluations with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM agent_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if passed is not None:
                    query += " AND passed = ?"
                    params.append(1 if passed else 0)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "evaluation_id": row["evaluation_id"],
                        "task_id": row["task_id"],
                        "agent_name": row["agent_name"],
                        "evaluation_type": row["evaluation_type"],
                        "overall_score": row["overall_score"],
                        "metric_scores": self.deserialize_payload(row["metric_scores_json"]) or {},
                        "passed": bool(row["passed"]),
                        "failures": self.deserialize_payload(row["failures_json"]) or [],
                        "warnings": self.deserialize_payload(row["warnings_json"]) or [],
                        "evidence": self.deserialize_payload(row["evidence_json"]) or {},
                        "evaluation_reason": row["evaluation_reason"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_task_outcome(self, data: Dict[str, Any]) -> bool:
        """Persist or update a task outcome evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO task_outcomes (
                        outcome_id, task_id, objective, classification, passed_criteria_json,
                        failed_criteria_json, numeric_checks_json, missing_fields_json,
                        evidence_verified, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(outcome_id) DO UPDATE SET
                        classification = excluded.classification,
                        passed_criteria_json = excluded.passed_criteria_json,
                        failed_criteria_json = excluded.failed_criteria_json,
                        numeric_checks_json = excluded.numeric_checks_json,
                        missing_fields_json = excluded.missing_fields_json,
                        evidence_verified = excluded.evidence_verified,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["outcome_id"],
                        data["task_id"],
                        data["objective"],
                        str(data["classification"]),
                        self.serialize_payload(data.get("passed_criteria", [])),
                        self.serialize_payload(data.get("failed_criteria", [])),
                        self.serialize_payload(data.get("numeric_checks", {})),
                        self.serialize_payload(data.get("missing_fields", [])),
                        1 if data.get("evidence_verified") else 0,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save task outcome {data.get('outcome_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_task_outcome(self, outcome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task outcome evaluation by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM task_outcomes WHERE outcome_id = ?;", (outcome_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "outcome_id": row["outcome_id"],
                    "task_id": row["task_id"],
                    "objective": row["objective"],
                    "classification": row["classification"],
                    "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                    "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                    "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                    "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                    "evidence_verified": bool(row["evidence_verified"]),
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_task_outcomes(
        self,
        task_id: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query task outcome evaluations."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM task_outcomes WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if classification:
                    query += " AND classification = ?"
                    params.append(classification)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "outcome_id": row["outcome_id"],
                        "task_id": row["task_id"],
                        "objective": row["objective"],
                        "classification": row["classification"],
                        "passed_criteria": self.deserialize_payload(row["passed_criteria_json"]) or [],
                        "failed_criteria": self.deserialize_payload(row["failed_criteria_json"]) or [],
                        "numeric_checks": self.deserialize_payload(row["numeric_checks_json"]) or {},
                        "missing_fields": self.deserialize_payload(row["missing_fields_json"]) or [],
                        "evidence_verified": bool(row["evidence_verified"]),
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_llm_judge_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update an LLM Judge evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO llm_judge_evaluations (
                        judge_id, task_id, target_agent, target_stage, accuracy, completeness,
                        relevance, instruction_adherence, evidence_support, overall_score,
                        identified_weaknesses_json, confidence, reasoning_summary, evaluator_model,
                        evaluator_provider, status, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(judge_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        confidence = excluded.confidence,
                        reasoning_summary = excluded.reasoning_summary,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["judge_id"],
                        data["task_id"],
                        data["target_agent"],
                        data["target_stage"],
                        float(data.get("accuracy", 0.0)),
                        float(data.get("completeness", 0.0)),
                        float(data.get("relevance", 0.0)),
                        float(data.get("instruction_adherence", 0.0)),
                        float(data.get("evidence_support", 0.0)),
                        float(data.get("overall_score", 0.0)),
                        self.serialize_payload(data.get("identified_weaknesses", [])),
                        float(data.get("confidence", 0.0)),
                        data.get("reasoning_summary", ""),
                        data.get("evaluator_model", ""),
                        data.get("evaluator_provider", ""),
                        data.get("status", "COMPLETED"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save LLM judge evaluation {data.get('judge_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_llm_judge_evaluation(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve LLM judge evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM llm_judge_evaluations WHERE judge_id = ?;", (judge_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "judge_id": row["judge_id"],
                    "task_id": row["task_id"],
                    "target_agent": row["target_agent"],
                    "target_stage": row["target_stage"],
                    "accuracy": row["accuracy"],
                    "completeness": row["completeness"],
                    "relevance": row["relevance"],
                    "instruction_adherence": row["instruction_adherence"],
                    "evidence_support": row["evidence_support"],
                    "overall_score": row["overall_score"],
                    "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                    "confidence": row["confidence"],
                    "reasoning_summary": row["reasoning_summary"],
                    "evaluator_model": row["evaluator_model"],
                    "evaluator_provider": row["evaluator_provider"],
                    "status": row["status"],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_llm_judge_evaluations(
        self,
        task_id: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query LLM judge evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM llm_judge_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if target_agent:
                    query += " AND target_agent = ?"
                    params.append(target_agent)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "judge_id": row["judge_id"],
                        "task_id": row["task_id"],
                        "target_agent": row["target_agent"],
                        "target_stage": row["target_stage"],
                        "accuracy": row["accuracy"],
                        "completeness": row["completeness"],
                        "relevance": row["relevance"],
                        "instruction_adherence": row["instruction_adherence"],
                        "evidence_support": row["evidence_support"],
                        "overall_score": row["overall_score"],
                        "identified_weaknesses": self.deserialize_payload(row["identified_weaknesses_json"]) or [],
                        "confidence": row["confidence"],
                        "reasoning_summary": row["reasoning_summary"],
                        "evaluator_model": row["evaluator_model"],
                        "evaluator_provider": row["evaluator_provider"],
                        "status": row["status"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_retrieval_evaluation(self, data: Dict[str, Any]) -> bool:
        """Persist or update a retrieval & RAG evaluation."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO retrieval_evaluations (
                        retrieval_id, task_id, query, k, precision_at_k, recall_at_k,
                        mrr, ndcg, context_relevance, context_coverage, faithfulness_score,
                        grounded_status, unsupported_claims_json, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(retrieval_id) DO UPDATE SET
                        precision_at_k = excluded.precision_at_k,
                        recall_at_k = excluded.recall_at_k,
                        mrr = excluded.mrr,
                        ndcg = excluded.ndcg,
                        faithfulness_score = excluded.faithfulness_score,
                        grounded_status = excluded.grounded_status,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["retrieval_id"],
                        data["task_id"],
                        data["query"],
                        int(data.get("k", 5)),
                        float(data.get("precision_at_k", 0.0)),
                        float(data.get("recall_at_k", 0.0)),
                        float(data.get("mrr", 0.0)),
                        float(data.get("ndcg", 0.0)),
                        float(data.get("context_relevance", 0.0)),
                        float(data.get("context_coverage", 0.0)),
                        float(data.get("faithfulness_score", 0.0)),
                        str(data.get("grounded_status", "GROUNDED")),
                        self.serialize_payload(data.get("unsupported_claims", [])),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save retrieval evaluation {data.get('retrieval_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_retrieval_evaluation(self, retrieval_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve retrieval evaluation record by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM retrieval_evaluations WHERE retrieval_id = ?;", (retrieval_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "retrieval_id": row["retrieval_id"],
                    "task_id": row["task_id"],
                    "query": row["query"],
                    "k": row["k"],
                    "precision_at_k": row["precision_at_k"],
                    "recall_at_k": row["recall_at_k"],
                    "mrr": row["mrr"],
                    "ndcg": row["ndcg"],
                    "context_relevance": row["context_relevance"],
                    "context_coverage": row["context_coverage"],
                    "faithfulness_score": row["faithfulness_score"],
                    "grounded_status": row["grounded_status"],
                    "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                    "timestamp": row["timestamp"],
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_retrieval_evaluations(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query retrieval evaluation records."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM retrieval_evaluations WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "retrieval_id": row["retrieval_id"],
                        "task_id": row["task_id"],
                        "query": row["query"],
                        "k": row["k"],
                        "precision_at_k": row["precision_at_k"],
                        "recall_at_k": row["recall_at_k"],
                        "mrr": row["mrr"],
                        "ndcg": row["ndcg"],
                        "context_relevance": row["context_relevance"],
                        "context_coverage": row["context_coverage"],
                        "faithfulness_score": row["faithfulness_score"],
                        "grounded_status": row["grounded_status"],
                        "unsupported_claims": self.deserialize_payload(row["unsupported_claims_json"]) or [],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_performance_metric(self, data: Dict[str, Any]) -> bool:
        """Persist an agent performance metric record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                metric_id = data.get("metric_id") or f"pm_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO agent_performance_metrics (
                        metric_id, agent_name, task_type, workflow_id, execution_time_ms,
                        outcome, retry_count, recovery_succeeded, tool_failure, human_intervention,
                        human_rejected, evaluation_failed, failure_category, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_id) DO UPDATE SET
                        execution_time_ms = excluded.execution_time_ms,
                        outcome = excluded.outcome,
                        retry_count = excluded.retry_count,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        metric_id,
                        data["agent_name"],
                        data.get("task_type", "general"),
                        data.get("workflow_id"),
                        float(data.get("execution_time_ms", 0.0)),
                        str(data.get("outcome", "success")),
                        int(data.get("retry_count", 0)),
                        1 if data.get("recovery_succeeded") else 0,
                        1 if data.get("tool_failure") else 0,
                        1 if data.get("human_intervention") else 0,
                        1 if data.get("human_rejected") else 0,
                        1 if data.get("evaluation_failed") else 0,
                        data.get("failure_category"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save performance metric {data.get('metric_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_performance_metrics(
        self,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query performance metrics with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM agent_performance_metrics WHERE 1=1"
                params = []
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "metric_id": row["metric_id"],
                        "agent_name": row["agent_name"],
                        "task_type": row["task_type"],
                        "workflow_id": row["workflow_id"],
                        "execution_time_ms": row["execution_time_ms"],
                        "outcome": row["outcome"],
                        "retry_count": row["retry_count"],
                        "recovery_succeeded": bool(row["recovery_succeeded"]),
                        "tool_failure": bool(row["tool_failure"]),
                        "human_intervention": bool(row["human_intervention"]),
                        "human_rejected": bool(row["human_rejected"]),
                        "evaluation_failed": bool(row["evaluation_failed"]),
                        "failure_category": row["failure_category"],
                        "timestamp": row["timestamp"],
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_evaluation_revision(self, data: Dict[str, Any]) -> bool:
        """Persist or update an evaluation revision request."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO evaluation_revisions (
                        revision_id, task_id, failing_agent, failing_stage, feedback,
                        attempt_number, max_attempts, preserved_outputs_json, status,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(revision_id) DO UPDATE SET
                        status = excluded.status,
                        completed_at = excluded.completed_at;
                    """,
                    (
                        data["revision_id"],
                        data["task_id"],
                        data["failing_agent"],
                        data["failing_stage"],
                        data["feedback"],
                        int(data.get("attempt_number", 1)),
                        int(data.get("max_attempts", 2)),
                        self.serialize_payload(data.get("preserved_outputs", {})),
                        data.get("status", "REQUESTED"),
                        data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                        data.get("completed_at"),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save evaluation revision {data.get('revision_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_evaluation_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve evaluation revision by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM evaluation_revisions WHERE revision_id = ?;", (revision_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "revision_id": row["revision_id"],
                    "task_id": row["task_id"],
                    "failing_agent": row["failing_agent"],
                    "failing_stage": row["failing_stage"],
                    "feedback": row["feedback"],
                    "attempt_number": row["attempt_number"],
                    "max_attempts": row["max_attempts"],
                    "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "completed_at": row["completed_at"],
                }
            finally:
                conn.close()

    def list_evaluation_revisions(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query evaluation revisions."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM evaluation_revisions WHERE 1=1"
                params = []
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                query += " ORDER BY created_at ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "revision_id": row["revision_id"],
                        "task_id": row["task_id"],
                        "failing_agent": row["failing_agent"],
                        "failing_stage": row["failing_stage"],
                        "feedback": row["feedback"],
                        "attempt_number": row["attempt_number"],
                        "max_attempts": row["max_attempts"],
                        "preserved_outputs": self.deserialize_payload(row["preserved_outputs_json"]) or {},
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "completed_at": row["completed_at"],
                    })
                return results
            finally:
                conn.close()



    # -----------------------------------------------------------------------
    # Day 229: Observability & Tracing Storage Methods
    # -----------------------------------------------------------------------

    def save_observability_trace(self, data: Dict[str, Any]) -> bool:
        """Persist or update an end-to-end observability trace."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO observability_traces (
                        trace_id, workflow_id, task_id, status, start_time, end_time,
                        duration_ms, root_span_id, agent_sequence_json, error_json,
                        attributes_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trace_id) DO UPDATE SET
                        status = excluded.status,
                        end_time = excluded.end_time,
                        duration_ms = excluded.duration_ms,
                        agent_sequence_json = excluded.agent_sequence_json,
                        error_json = excluded.error_json,
                        attributes_json = excluded.attributes_json,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["trace_id"],
                        data["workflow_id"],
                        data.get("task_id"),
                        str(data.get("status", "STARTED")),
                        data.get("start_time") or datetime.now(timezone.utc).isoformat(),
                        data.get("end_time"),
                        float(data["duration_ms"]) if data.get("duration_ms") is not None else None,
                        data.get("root_span_id"),
                        self.serialize_payload(data.get("agent_sequence", [])),
                        self.serialize_payload(data.get("error") or {}),
                        self.serialize_payload(data.get("attributes", {})),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save trace {data.get('trace_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_observability_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an observability trace by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM observability_traces WHERE trace_id = ?;", (trace_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "trace_id": row["trace_id"],
                    "workflow_id": row["workflow_id"],
                    "task_id": row["task_id"],
                    "status": row["status"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "duration_ms": row["duration_ms"],
                    "root_span_id": row["root_span_id"],
                    "agent_sequence": self.deserialize_payload(row["agent_sequence_json"]) or [],
                    "error": self.deserialize_payload(row["error_json"]) or {},
                    "attributes": self.deserialize_payload(row["attributes_json"]) or {},
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_observability_traces(
        self,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query observability traces with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM observability_traces WHERE 1=1"
                params = []
                if workflow_id:
                    query += " AND workflow_id = ?"
                    params.append(workflow_id)
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if status:
                    query += " AND status = ?"
                    params.append(status)
                if start_time:
                    query += " AND start_time >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND start_time <= ?"
                    params.append(end_time)
                query += " ORDER BY start_time DESC LIMIT ?;"
                params.append(limit)

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "trace_id": row["trace_id"],
                        "workflow_id": row["workflow_id"],
                        "task_id": row["task_id"],
                        "status": row["status"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "duration_ms": row["duration_ms"],
                        "root_span_id": row["root_span_id"],
                        "agent_sequence": self.deserialize_payload(row["agent_sequence_json"]) or [],
                        "error": self.deserialize_payload(row["error_json"]) or {},
                        "attributes": self.deserialize_payload(row["attributes_json"]) or {},
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_observability_span(self, data: Dict[str, Any]) -> bool:
        """Persist or update an observability span."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                conn.execute(
                    """
                    INSERT INTO observability_spans (
                        span_id, trace_id, parent_span_id, workflow_id, task_id,
                        agent_name, operation_name, span_kind, start_time, end_time,
                        duration_ms, status, attributes_json, error_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(span_id) DO UPDATE SET
                        end_time = excluded.end_time,
                        duration_ms = excluded.duration_ms,
                        status = excluded.status,
                        attributes_json = excluded.attributes_json,
                        error_json = excluded.error_json,
                        metadata_json = excluded.metadata_json;
                    """,
                    (
                        data["span_id"],
                        data["trace_id"],
                        data.get("parent_span_id"),
                        data["workflow_id"],
                        data.get("task_id"),
                        data["agent_name"],
                        data["operation_name"],
                        str(data.get("span_kind", "agent")),
                        data.get("start_time") or datetime.now(timezone.utc).isoformat(),
                        data.get("end_time"),
                        float(data["duration_ms"]) if data.get("duration_ms") is not None else None,
                        str(data.get("status", "STARTED")),
                        self.serialize_payload(data.get("attributes", {})),
                        self.serialize_payload(data.get("error") or {}),
                        self.serialize_payload(data.get("metadata", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save span {data.get('span_id')}: {exc}")
                raise
            finally:
                conn.close()

    def get_observability_span(self, span_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an observability span by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT * FROM observability_spans WHERE span_id = ?;", (span_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "span_id": row["span_id"],
                    "trace_id": row["trace_id"],
                    "parent_span_id": row["parent_span_id"],
                    "workflow_id": row["workflow_id"],
                    "task_id": row["task_id"],
                    "agent_name": row["agent_name"],
                    "operation_name": row["operation_name"],
                    "span_kind": row["span_kind"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "duration_ms": row["duration_ms"],
                    "status": row["status"],
                    "attributes": self.deserialize_payload(row["attributes_json"]) or {},
                    "error": self.deserialize_payload(row["error_json"]) or {},
                    "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                }
            finally:
                conn.close()

    def list_observability_spans(
        self,
        trace_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        status: Optional[str] = None,
        operation_name: Optional[str] = None,
        span_kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query observability spans with filters."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM observability_spans WHERE 1=1"
                params = []
                if trace_id:
                    query += " AND trace_id = ?"
                    params.append(trace_id)
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if status:
                    query += " AND status = ?"
                    params.append(status)
                if operation_name:
                    query += " AND operation_name = ?"
                    params.append(operation_name)
                if span_kind:
                    query += " AND span_kind = ?"
                    params.append(span_kind)
                query += " ORDER BY start_time ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "span_id": row["span_id"],
                        "trace_id": row["trace_id"],
                        "parent_span_id": row["parent_span_id"],
                        "workflow_id": row["workflow_id"],
                        "task_id": row["task_id"],
                        "agent_name": row["agent_name"],
                        "operation_name": row["operation_name"],
                        "span_kind": row["span_kind"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "duration_ms": row["duration_ms"],
                        "status": row["status"],
                        "attributes": self.deserialize_payload(row["attributes_json"]) or {},
                        "error": self.deserialize_payload(row["error_json"]) or {},
                        "metadata": self.deserialize_payload(row["metadata_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_observability_event(self, data: Dict[str, Any]) -> bool:
        """Persist an observability event record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                event_id = data.get("event_id") or f"ev_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO observability_events (
                        event_id, trace_id, span_id, event_type, agent_name, timestamp, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO NOTHING;
                    """,
                    (
                        event_id,
                        data["trace_id"],
                        data.get("span_id"),
                        data["event_type"],
                        data.get("agent_name"),
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        self.serialize_payload(data.get("payload", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save event {data.get('event_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_observability_events(
        self,
        trace_id: Optional[str] = None,
        event_type: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query observability events."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM observability_events WHERE 1=1"
                params = []
                if trace_id:
                    query += " AND trace_id = ?"
                    params.append(trace_id)
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                if span_id:
                    query += " AND span_id = ?"
                    params.append(span_id)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "event_id": row["event_id"],
                        "trace_id": row["trace_id"],
                        "span_id": row["span_id"],
                        "event_type": row["event_type"],
                        "agent_name": row["agent_name"],
                        "timestamp": row["timestamp"],
                        "payload": self.deserialize_payload(row["payload_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def save_observability_log(self, data: Dict[str, Any]) -> bool:
        """Persist a structured observability log entry."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                log_id = data.get("log_id") or f"log_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO observability_logs (
                        log_id, timestamp, log_level, event_type, trace_id, span_id,
                        workflow_id, task_id, agent_name, operation_name, message,
                        attributes_json, error_type, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(log_id) DO NOTHING;
                    """,
                    (
                        log_id,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        str(data.get("log_level", "INFO")),
                        data.get("event_type", "GENERAL"),
                        data.get("trace_id"),
                        data.get("span_id"),
                        data.get("workflow_id"),
                        data.get("task_id"),
                        data.get("agent_name"),
                        data.get("operation_name"),
                        str(data.get("message", "")),
                        self.serialize_payload(data.get("attributes", {})),
                        data.get("error_type"),
                        data.get("error_message"),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save observability log {data.get('log_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_observability_logs(
        self,
        trace_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        log_level: Optional[str] = None,
        event_type: Optional[str] = None,
        agent_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Query structured observability logs."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM observability_logs WHERE 1=1"
                params = []
                if trace_id:
                    query += " AND trace_id = ?"
                    params.append(trace_id)
                if workflow_id:
                    query += " AND workflow_id = ?"
                    params.append(workflow_id)
                if log_level:
                    query += " AND log_level = ?"
                    params.append(log_level)
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC LIMIT ?;"
                params.append(limit)

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "log_id": row["log_id"],
                        "timestamp": row["timestamp"],
                        "log_level": row["log_level"],
                        "event_type": row["event_type"],
                        "trace_id": row["trace_id"],
                        "span_id": row["span_id"],
                        "workflow_id": row["workflow_id"],
                        "task_id": row["task_id"],
                        "agent_name": row["agent_name"],
                        "operation_name": row["operation_name"],
                        "message": row["message"],
                        "attributes": self.deserialize_payload(row["attributes_json"]) or {},
                        "error_type": row["error_type"],
                        "error_message": row["error_message"],
                    })
                return results
            finally:
                conn.close()

    def save_observability_metric(self, data: Dict[str, Any]) -> bool:
        """Persist a single observability metric data point."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                metric_id = data.get("metric_id") or f"m_{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO observability_metrics (
                        metric_id, timestamp, metric_name, metric_type, value,
                        agent_name, workflow_id, task_id, provider, model, tool_name, attributes_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_id) DO NOTHING;
                    """,
                    (
                        metric_id,
                        data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        data["metric_name"],
                        str(data.get("metric_type", "gauge")),
                        float(data["value"]),
                        data.get("agent_name"),
                        data.get("workflow_id"),
                        data.get("task_id"),
                        data.get("provider"),
                        data.get("model"),
                        data.get("tool_name"),
                        self.serialize_payload(data.get("attributes", {})),
                    ),
                )
                conn.execute("COMMIT;")
                return True
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Failed to save observability metric {data.get('metric_id')}: {exc}")
                raise
            finally:
                conn.close()

    def list_observability_metrics(
        self,
        metric_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        provider: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query observability metrics."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM observability_metrics WHERE 1=1"
                params = []
                if metric_name:
                    query += " AND metric_name = ?"
                    params.append(metric_name)
                if agent_name:
                    query += " AND agent_name = ?"
                    params.append(agent_name)
                if workflow_id:
                    query += " AND workflow_id = ?"
                    params.append(workflow_id)
                if provider:
                    query += " AND provider = ?"
                    params.append(provider)
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                query += " ORDER BY timestamp ASC;"

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "metric_id": row["metric_id"],
                        "timestamp": row["timestamp"],
                        "metric_name": row["metric_name"],
                        "metric_type": row["metric_type"],
                        "value": row["value"],
                        "agent_name": row["agent_name"],
                        "workflow_id": row["workflow_id"],
                        "task_id": row["task_id"],
                        "provider": row["provider"],
                        "model": row["model"],
                        "tool_name": row["tool_name"],
                        "attributes": self.deserialize_payload(row["attributes_json"]) or {},
                    })
                return results
            finally:
                conn.close()

    def cleanup_observability_records(self, older_than_iso: str, dry_run: bool = False) -> Dict[str, Any]:
        """Enforce retention policy by purging expired traces, spans, logs, and events.

        Active traces (status == 'STARTED') and their child spans/events/logs are strictly preserved.
        Supports dry_run to preview purge statistics without modifying database.
        Preserves audit trail, messages, and security events intact.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Subquery identifying eligible completed/failed/cancelled traces
                eligible_trace_filter = (
                    "trace_id IN (SELECT trace_id FROM observability_traces "
                    "WHERE start_time < ? AND status IN ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT')) "
                    "AND trace_id NOT IN (SELECT trace_id FROM observability_traces WHERE status = 'STARTED')"
                )

                if dry_run:
                    c_spans = conn.execute(
                        f"SELECT COUNT(*) FROM observability_spans WHERE {eligible_trace_filter};",
                        (older_than_iso,),
                    ).fetchone()[0]

                    c_events = conn.execute(
                        f"SELECT COUNT(*) FROM observability_events WHERE {eligible_trace_filter};",
                        (older_than_iso,),
                    ).fetchone()[0]

                    c_logs = conn.execute(
                        f"SELECT COUNT(*) FROM observability_logs WHERE (({eligible_trace_filter}) OR (trace_id IS NULL AND timestamp < ?)) "
                        "AND (trace_id IS NULL OR trace_id NOT IN (SELECT trace_id FROM observability_traces WHERE status = 'STARTED'));",
                        (older_than_iso, older_than_iso),
                    ).fetchone()[0]

                    c_metrics = conn.execute(
                        "SELECT COUNT(*) FROM observability_metrics WHERE timestamp < ? "
                        "AND (workflow_id IS NULL OR workflow_id NOT IN (SELECT workflow_id FROM observability_traces WHERE status = 'STARTED'));",
                        (older_than_iso,),
                    ).fetchone()[0]

                    c_traces = conn.execute(
                        "SELECT COUNT(*) FROM observability_traces WHERE start_time < ? AND status IN ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT');",
                        (older_than_iso,),
                    ).fetchone()[0]

                    return {
                        "traces": c_traces,
                        "spans": c_spans,
                        "events": c_events,
                        "logs": c_logs,
                        "metrics": c_metrics,
                        "purged_traces": c_traces,
                        "purged_spans": c_spans,
                        "purged_events": c_events,
                        "purged_logs": c_logs,
                        "purged_metrics": c_metrics,
                        "dry_run": True,
                    }

                conn.execute("BEGIN IMMEDIATE;")
                deleted_counts = {}

                # Delete old spans of eligible traces (never active traces)
                c = conn.execute(
                    f"DELETE FROM observability_spans WHERE {eligible_trace_filter};",
                    (older_than_iso,),
                )
                deleted_counts["spans"] = c.rowcount

                # Delete old events of eligible traces (never active traces)
                c = conn.execute(
                    f"DELETE FROM observability_events WHERE {eligible_trace_filter};",
                    (older_than_iso,),
                )
                deleted_counts["events"] = c.rowcount

                # Delete old logs (linked to eligible traces or standalone, never active traces)
                c = conn.execute(
                    f"DELETE FROM observability_logs WHERE (({eligible_trace_filter}) OR (trace_id IS NULL AND timestamp < ?)) "
                    "AND (trace_id IS NULL OR trace_id NOT IN (SELECT trace_id FROM observability_traces WHERE status = 'STARTED'));",
                    (older_than_iso, older_than_iso),
                )
                deleted_counts["logs"] = c.rowcount

                # Delete old metrics (not associated with active workflows)
                c = conn.execute(
                    "DELETE FROM observability_metrics WHERE timestamp < ? "
                    "AND (workflow_id IS NULL OR workflow_id NOT IN (SELECT workflow_id FROM observability_traces WHERE status = 'STARTED'));",
                    (older_than_iso,),
                )
                deleted_counts["metrics"] = c.rowcount

                # Delete old completed/failed traces (never STARTED)
                c = conn.execute(
                    "DELETE FROM observability_traces WHERE start_time < ? AND status IN ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT');",
                    (older_than_iso,),
                )
                deleted_counts["traces"] = c.rowcount
                deleted_counts["traces_purged"] = c.rowcount
                deleted_counts["spans_purged"] = deleted_counts["spans"]
                deleted_counts["logs_purged"] = deleted_counts["logs"]
                deleted_counts["metrics_purged"] = deleted_counts["metrics"]
                deleted_counts["events_purged"] = deleted_counts["events"]
                deleted_counts["dry_run"] = False

                conn.execute("COMMIT;")
                logger.info(f"[SQLiteStorage] Observability retention cleanup completed: {deleted_counts}")
                return deleted_counts
            except Exception as exc:
                conn.execute("ROLLBACK;")
                logger.error(f"[SQLiteStorage] Retention cleanup failed: {exc}")
                raise
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Day 230 Security Events Persistence
    # -----------------------------------------------------------------------

    def save_security_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a security event record to SQLite with atomic hash chaining."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                # Fetch last sequence and hash for cryptographic chaining
                row = conn.execute(
                    "SELECT sequence_number, event_hash FROM security_events WHERE sequence_number IS NOT NULL ORDER BY sequence_number DESC LIMIT 1;"
                ).fetchone()

                if row and row["sequence_number"] is not None:
                    next_seq = int(row["sequence_number"]) + 1
                    prev_hash = str(row["event_hash"])
                else:
                    next_seq = 1
                    prev_hash = "0" * 64

                event_id = str(event_data.get("event_id") or f"sec_{uuid.uuid4().hex[:8]}")
                ts = str(event_data.get("timestamp") or datetime.now(timezone.utc).isoformat())
                findings = event_data.get("findings", [])
                findings_json = json.dumps(findings) if isinstance(findings, (list, dict)) else str(findings)
                metadata = event_data.get("metadata", {})
                metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else str(metadata)
                threat_category = str(event_data.get("threat_category", "unknown"))
                severity = str(event_data.get("severity", "LOW"))
                decision = str(event_data.get("security_decision", "ALLOW"))
                agent_actor = event_data.get("agent") or event_data.get("actor_role")
                workflow_id = event_data.get("workflow_id")
                task_id = event_data.get("task_id")
                resource = event_data.get("relevant_resource") or event_data.get("target_resource")
                hash_version = "v1"

                # Deterministic canonical serialization for SHA-256 chain verification
                canonical_dict = {
                    "event_id": event_id,
                    "timestamp": ts,
                    "threat_category": threat_category,
                    "severity": severity,
                    "security_decision": decision,
                    "agent": agent_actor,
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "relevant_resource": resource,
                    "previous_event_hash": prev_hash,
                    "sequence_number": next_seq,
                    "hash_version": hash_version,
                }
                canonical_bytes = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
                event_hash = hashlib.sha256(canonical_bytes).hexdigest()

                conn.execute(
                    """
                    INSERT OR REPLACE INTO security_events (
                        event_id, timestamp, trace_id, workflow_id, task_id, agent,
                        threat_category, severity, security_decision,
                        sanitized_explanation, relevant_resource, findings_json, metadata_json,
                        sequence_number, previous_event_hash, event_hash, hash_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event_id,
                        ts,
                        event_data.get("trace_id"),
                        workflow_id,
                        task_id,
                        agent_actor,
                        threat_category,
                        severity,
                        decision,
                        event_data.get("sanitized_explanation") or event_data.get("description"),
                        resource,
                        findings_json,
                        metadata_json,
                        next_seq,
                        prev_hash,
                        event_hash,
                        hash_version,
                    ),
                )
                conn.commit()
                logger.debug(f"[SQLiteStorage] Saved hashed security event: {event_id} (seq={next_seq})")
                return {
                    "event_id": event_id,
                    "sequence_number": next_seq,
                    "previous_event_hash": prev_hash,
                    "event_hash": event_hash,
                    "hash_version": hash_version,
                }
            except Exception as exc:
                try:
                    conn.rollback()
                except Exception:
                    pass
                logger.error(f"[SQLiteStorage] Failed to save security event: {exc}")
                raise
            finally:
                conn.close()

    def get_security_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a security event by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM security_events WHERE event_id = ?;",
                    (event_id,),
                ).fetchone()
                if not row:
                    return None
                d = dict(row)
                d["findings"] = json.loads(d.pop("findings_json", "[]"))
                d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
                return d
            finally:
                conn.close()

    def list_security_events(
        self,
        threat_category: Optional[str] = None,
        severity: Optional[str] = None,
        decision: Optional[str] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query security events with optional filtering."""
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM security_events WHERE 1=1"
                params: List[Any] = []
                if threat_category:
                    query += " AND threat_category = ?"
                    params.append(threat_category)
                if severity:
                    query += " AND severity = ?"
                    params.append(severity)
                if decision:
                    query += " AND security_decision = ?"
                    params.append(decision)
                if workflow_id:
                    query += " AND workflow_id = ?"
                    params.append(workflow_id)
                if task_id:
                    query += " AND task_id = ?"
                    params.append(task_id)
                if agent:
                    query += " AND agent = ?"
                    params.append(agent)
                query += " ORDER BY timestamp DESC LIMIT ?;"
                params.append(limit)

                rows = conn.execute(query, tuple(params)).fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    d["findings"] = json.loads(d.pop("findings_json", "[]"))
                    d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
                    results.append(d)
                return results
            finally:
                conn.close()

    def verify_security_audit_chain(self) -> Dict[str, Any]:
        """Verify the cryptographic hash-chain integrity of the security_events audit table.

        Validates:
        1. Monotonic sequence continuity (1, 2, 3... with no gaps or missing records)
        2. Genesis root previous_event_hash ('0'*64)
        3. Cryptographic hash chaining (each previous_event_hash matches previous event's event_hash)
        4. Invariant content integrity (recomputed SHA-256 matches stored event_hash)
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM security_events WHERE sequence_number IS NOT NULL ORDER BY sequence_number ASC;"
                ).fetchall()

                if not rows:
                    return {
                        "is_valid": True,
                        "total_events": 0,
                        "tampered_event_id": None,
                        "sequence_number": None,
                        "error": None,
                    }

                expected_seq = 1
                expected_prev_hash = "0" * 64

                for r in rows:
                    row_seq = r["sequence_number"]
                    row_prev_hash = r["previous_event_hash"]
                    row_hash = r["event_hash"]
                    row_id = r["event_id"]

                    # 1. Check sequence number
                    if row_seq != expected_seq:
                        return {
                            "is_valid": False,
                            "total_events": len(rows),
                            "tampered_event_id": row_id,
                            "sequence_number": row_seq,
                            "error": f"Sequence break: Expected sequence {expected_seq}, found {row_seq} (missing or deleted event).",
                        }

                    # 2. Check previous event hash
                    if row_prev_hash != expected_prev_hash:
                        return {
                            "is_valid": False,
                            "total_events": len(rows),
                            "tampered_event_id": row_id,
                            "sequence_number": row_seq,
                            "error": f"Hash chain broken at sequence {row_seq}: Previous hash '{row_prev_hash}' does not match expected '{expected_prev_hash}'.",
                        }

                    # 3. Recalculate event hash
                    canonical_dict = {
                        "event_id": row_id,
                        "timestamp": r["timestamp"],
                        "threat_category": str(r["threat_category"]),
                        "severity": str(r["severity"]),
                        "security_decision": str(r["security_decision"]),
                        "agent": r["agent"],
                        "workflow_id": r["workflow_id"],
                        "task_id": r["task_id"],
                        "relevant_resource": r["relevant_resource"],
                        "previous_event_hash": row_prev_hash,
                        "sequence_number": row_seq,
                        "hash_version": r["hash_version"] or "v1",
                    }
                    canonical_bytes = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    computed_hash = hashlib.sha256(canonical_bytes).hexdigest()

                    if computed_hash != row_hash:
                        return {
                            "is_valid": False,
                            "total_events": len(rows),
                            "tampered_event_id": row_id,
                            "sequence_number": row_seq,
                            "error": f"Data tampering detected at event '{row_id}' (seq {row_seq}): Hash mismatch (computed={computed_hash}, stored={row_hash}).",
                        }

                    # Advance for next event
                    expected_seq += 1
                    expected_prev_hash = row_hash

                return {
                    "is_valid": True,
                    "total_events": len(rows),
                    "tampered_event_id": None,
                    "sequence_number": None,
                    "error": None,
                }
            finally:
                conn.close()


def verify_security_audit_chain(storage: Optional[Any] = None) -> Dict[str, Any]:
    """Module-level helper to verify audit chain on storage instance."""
    if storage is None:
        storage = SQLiteMessageStorage()
    if hasattr(storage, "verify_security_audit_chain"):
        return storage.verify_security_audit_chain()
    return {"is_valid": True, "total_events": 0, "error": None}
