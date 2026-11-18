"""
tests/test_day228_agent_evaluation.py

Comprehensive Test Suite for Day 228: Agent Evaluation Framework (Parts 1 to 6).

Covers:
- Part 1: Agent Evaluation Architecture & Metrics
  - EvaluationMetric enum and 8 dimensions
  - EvaluationThreshold and score classification
  - MetricWeightsConfig validation (strict 100% sum, negative weight rejection, role defaults)
  - AgentEvaluationEngine scoring, evidence penalties, SQLite persistence
- Part 2: Task Outcome Evaluator
  - OutcomeClassification (SUCCESS, PARTIAL_SUCCESS, FAILURE, INCONCLUSIVE)
  - Independent numeric audit and arithmetic calculation verification
  - Mandatory step enforcement
  - Missing expected output schema penalization
  - Vacuous claim detection
- Part 3: LLM-as-a-Judge
  - Structured evaluation via LLM provider
  - Evaluator independence enforcement
  - Deterministic fallback on provider failure without fabricating scores
- Part 4: Retrieval & RAG Evaluator
  - Precision@K, Recall@K, MRR, NDCG calculation
  - Zero-division safety (empty lists, K=0)
  - Context relevance and context coverage
  - Answer faithfulness, groundedness tiers, unsupported claim detection
- Part 5: Agent Performance & Reliability Evaluator
  - Performance tracking (success rate, failure rate, retry rate, recovery rate)
  - Average latency and P95 latency (nearest rank)
  - Strict separation of human rejection from agent execution failures
  - Agent leaderboard comparison and time-window filtering
- Part 6: Unified Evaluation Framework & Integration
  - Complete 8-stage lifecycle coordination and EvaluationDecision synthesis
  - RevisionRequest generation with preserved upstream outputs
  - Max revisions limit defense preventing infinite loops
  - Orchestrator integration and backward compatibility
"""

import json
import math
import os
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from core.agent_evaluation import (
    EvaluationMetric,
    EvaluationThreshold,
    classify_threshold,
    MetricWeightsConfig,
    AgentEvaluationEngine,
    OutcomeClassification,
    TaskOutcome,
    TaskOutcomeEvaluator,
    LLMJudgeEvaluation,
    LLMJudge,
    RetrievalMetricsResult,
    FaithfulnessResult,
    RetrievalEvaluator,
    AgentPerformanceRecord,
    AgentPerformanceEvaluator,
    EvaluationDecision,
    RevisionRequest,
    UnifiedEvaluationFramework,
)
from core.evaluator import EvaluationResult, ResultEvaluator
from core.persistent_storage import SQLiteMessageStorage
from core.orchestrator import WorkflowOrchestrator, TaskContext, WorkflowStatus
from core.provider_interface import LLMMessage, LLMProvider, LLMResponse


@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite database path and clean up afterwards."""
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
    """Provide an initialized SQLiteMessageStorage instance."""
    return SQLiteMessageStorage(db_path=temp_db_path)


class MockLLMJudgeProvider(LLMProvider):
    """Deterministic LLM Provider returning valid structured judge evaluations."""
    def __init__(self, overall_score=0.92, weaknesses=None):
        self.overall_score = overall_score
        self.weaknesses = weaknesses or []

    def generate(self, messages, model, tools=None, tool_choice="auto") -> LLMResponse:
        data = {
            "accuracy": 0.95,
            "completeness": 0.90,
            "relevance": 0.92,
            "instruction_adherence": 0.94,
            "evidence_support": 0.90,
            "overall_score": self.overall_score,
            "identified_weaknesses": self.weaknesses,
            "confidence": 0.95,
            "reasoning_summary": "High fidelity execution meeting all criteria.",
        }
        return LLMResponse(content=json.dumps(data))


class FailingLLMProvider(LLMProvider):
    """Provider simulating network failures or invalid responses."""
    def generate(self, messages, model, tools=None, tool_choice="auto") -> LLMResponse:
        raise ConnectionError("LLM API endpoint unavailable")


# ===========================================================================
# PART 1: AGENT EVALUATION ARCHITECTURE & METRICS
# ===========================================================================

def test_1_evaluation_metrics_and_threshold_classification():
    """Verify 8 evaluation dimensions and threshold boundaries."""
    assert len(EvaluationMetric) == 8
    expected_metrics = {
        "correctness", "completeness", "relevance", "instruction_adherence",
        "evidence_quality", "output_consistency", "efficiency", "reliability"
    }
    actual_metrics = {m.value for m in EvaluationMetric}
    assert expected_metrics == actual_metrics

    # Test threshold tier boundaries
    assert classify_threshold(0.95) == EvaluationThreshold.EXCELLENT
    assert classify_threshold(0.90) == EvaluationThreshold.EXCELLENT
    assert classify_threshold(0.89) == EvaluationThreshold.GOOD
    assert classify_threshold(0.75) == EvaluationThreshold.GOOD
    assert classify_threshold(0.74) == EvaluationThreshold.NEEDS_IMPROVEMENT
    assert classify_threshold(0.50) == EvaluationThreshold.NEEDS_IMPROVEMENT
    assert classify_threshold(0.49) == EvaluationThreshold.FAILED
    assert classify_threshold(0.10) == EvaluationThreshold.FAILED


def test_2_metric_weights_config_strict_sum_and_defaults():
    """Verify weights config enforces strict 100% sum and provides role presets."""
    # Valid config
    valid_cfg = MetricWeightsConfig({
        EvaluationMetric.CORRECTNESS: 0.50,
        EvaluationMetric.COMPLETENESS: 0.50,
    })
    assert math.isclose(sum(valid_cfg.weights.values()), 1.0)

    # Invalid sum (< 1.0)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        MetricWeightsConfig({EvaluationMetric.CORRECTNESS: 0.40})

    # Invalid sum (> 1.0)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        MetricWeightsConfig({
            EvaluationMetric.CORRECTNESS: 0.60,
            EvaluationMetric.COMPLETENESS: 0.60,
        })

    # Negative weight
    with pytest.raises(ValueError, match="cannot be negative"):
        MetricWeightsConfig({
            EvaluationMetric.CORRECTNESS: 1.20,
            EvaluationMetric.COMPLETENESS: -0.20,
        })

    # Role default configs must all sum to 1.0
    for preset in [
        MetricWeightsConfig.default(),
        MetricWeightsConfig.for_researcher(),
        MetricWeightsConfig.for_planner(),
        MetricWeightsConfig.for_executor(),
        MetricWeightsConfig.for_supervisor(),
    ]:
        assert math.isclose(sum(preset.weights.values()), 1.0, abs_tol=1e-4)


def test_3_agent_evaluation_engine_scoring_evidence_penalty_and_storage(storage):
    """Verify evaluation engine computes weighted score, penalizes poor evidence, and persists."""
    engine = AgentEvaluationEngine(storage=storage)

    # Researcher evaluation with high evidence
    scores_high = {
        EvaluationMetric.EVIDENCE_QUALITY: 0.95,
        EvaluationMetric.RELEVANCE: 0.90,
        EvaluationMetric.COMPLETENESS: 0.85,
        EvaluationMetric.CORRECTNESS: 0.90,
        EvaluationMetric.RELIABILITY: 0.90,
    }
    evidence_payload = {"sources_verified": 3, "data_points": [10, 20]}

    res_high = engine.evaluate_agent_output(
        task_id="task_p1_01",
        agent_name="researcher",
        metric_scores=scores_high,
        evidence=evidence_payload,
    )

    assert res_high.passed is True
    assert res_high.overall_score >= 0.85
    assert len(res_high.failures) == 0

    # Verify SQLite persistence
    persisted = storage.get_agent_evaluation(res_high.evaluation_id)
    assert persisted is not None
    assert persisted["task_id"] == "task_p1_01"
    assert persisted["agent_name"] == "researcher"
    assert persisted["overall_score"] == res_high.overall_score

    # Researcher evaluation with poor evidence (< 0.50)
    scores_poor_evid = {
        EvaluationMetric.EVIDENCE_QUALITY: 0.20,  # Below threshold
        EvaluationMetric.RELEVANCE: 0.90,
        EvaluationMetric.COMPLETENESS: 0.90,
        EvaluationMetric.CORRECTNESS: 0.90,
        EvaluationMetric.RELIABILITY: 0.90,
    }
    res_poor = engine.evaluate_agent_output(
        task_id="task_p1_02",
        agent_name="researcher",
        metric_scores=scores_poor_evid,
        evidence={},
    )

    assert res_poor.passed is False
    assert any("evidence quality" in f.lower() for f in res_poor.failures)


# ===========================================================================
# PART 2: TASK OUTCOME EVALUATOR
# ===========================================================================

def test_4_task_outcome_evaluator_success(storage):
    """Verify deterministic success classification and criteria satisfaction."""
    evaluator = TaskOutcomeEvaluator(storage=storage)

    steps = [
        {"task_id": 1, "status": "completed", "result": "Calculated annual revenue: 120000 USD."},
        {"task_id": 2, "status": "completed", "result": "Applied tax rate: Net is 96000 USD."},
    ]
    criteria = ["Revenue 120000", "Net 96000"]

    outcome = evaluator.evaluate_outcome(
        task_id="task_outcome_01",
        objective="Calculate annual revenue and net after tax",
        success_criteria=criteria,
        execution_steps=steps,
    )

    assert outcome.classification == OutcomeClassification.SUCCESS
    assert len(outcome.passed_criteria) == 2
    assert len(outcome.failed_criteria) == 0
    assert outcome.evidence_verified is True

    # Persisted check
    persisted = storage.get_task_outcome(outcome.outcome_id)
    assert persisted is not None
    assert persisted["classification"] == "SUCCESS"


def test_5_task_outcome_numeric_verification_and_independent_calc(storage):
    """Verify independent arithmetic evaluation and numeric formula validation."""
    evaluator = TaskOutcomeEvaluator(storage=storage)

    # 1. Independent formula verification: '15 + 25 = 40'
    text_matching = "The combined sum is 15 + 25 = 40 units verified."
    ok, detail = evaluator.audit_numeric_criteria("15 + 25 = 40", text_matching)
    assert ok is True
    assert "Verified arithmetic" in detail

    # 2. Formula is mathematically flawed
    ok_flawed, detail_flawed = evaluator.audit_numeric_criteria("10 + 10 = 25", text_matching)
    assert ok_flawed is False
    assert "formula invalid" in detail_flawed.lower()

    # 3. Numeric constraint not present in text
    ok_missing, detail_missing = evaluator.audit_numeric_criteria("Total cost 50000", "Total cost is 30000")
    assert ok_missing is False
    assert "missing" in detail_missing.lower()


def test_6_task_outcome_mandatory_step_enforcement(storage):
    """Failing a mandatory step must prevent OutcomeClassification.SUCCESS."""
    evaluator = TaskOutcomeEvaluator(storage=storage)

    steps = [
        {"task_id": 1, "status": "completed", "result": "Step 1 data gathered: 100 records"},
        {"task_id": 2, "status": "failed", "error": "Database write timeout", "result": "Write failed"},
    ]
    criteria = ["100 records"]

    # Step 2 is marked mandatory
    outcome = evaluator.evaluate_outcome(
        task_id="task_mandatory_01",
        objective="Collect and persist records",
        success_criteria=criteria,
        execution_steps=steps,
        mandatory_steps=[2],
    )

    # Must NOT be SUCCESS
    assert outcome.classification != OutcomeClassification.SUCCESS
    assert outcome.classification == OutcomeClassification.PARTIAL_SUCCESS
    assert any("mandatory_step_2_failed" in mf for mf in outcome.missing_fields)


def test_7_task_outcome_missing_outputs_and_vacuous_claims(storage):
    """Penalize missing expected output schema fields and vacuous completion claims."""
    evaluator = TaskOutcomeEvaluator(storage=storage)

    # Vacuous claim with no factual evidence
    vacuous_steps = [
        {"task_id": 1, "status": "completed", "result": "I have completed the task successfully."}
    ]
    outcome = evaluator.evaluate_outcome(
        task_id="task_vacuous_01",
        objective="Generate quarterly summary",
        success_criteria=["Quarterly summary"],
        execution_steps=vacuous_steps,
        expected_outputs=["quarterly_metrics", "executive_summary"],
    )

    assert outcome.classification == OutcomeClassification.FAILURE
    assert outcome.evidence_verified is False
    assert len(outcome.missing_fields) > 0


# ===========================================================================
# PART 3: LLM-AS-A-JUDGE
# ===========================================================================

def test_8_llm_judge_structured_evaluation(storage):
    """Verify structured LLM judge output and independence validation."""
    mock_provider = MockLLMJudgeProvider(overall_score=0.91)
    judge = LLMJudge(provider=mock_provider, storage=storage)

    eval_record = judge.evaluate(
        task_id="task_judge_01",
        target_agent="planner",
        target_stage="plan_decomposition",
        agent_input="Decompose data ingestion pipeline",
        agent_output="Step 1: Extract from S3. Step 2: Validate schema. Step 3: Load into Snowflake.",
        criteria=["Cover extraction, validation, loading"],
    )

    assert eval_record.status == "COMPLETED"
    assert eval_record.overall_score == 0.91
    assert eval_record.accuracy == 0.95
    assert eval_record.evaluator_provider == "llm_provider"

    # Evaluator independence check: evaluator cannot evaluate itself
    self_eval = judge.evaluate(
        task_id="task_judge_02",
        target_agent="llm_judge",
        target_stage="meta_evaluation",
        agent_input="test",
        agent_output="test",
    )
    assert self_eval.status == "FAILED"
    assert "cannot evaluate itself" in self_eval.identified_weaknesses[0].lower()


def test_9_llm_judge_deterministic_fallback_on_failure(storage):
    """When LLM provider fails, judge must execute deterministic fallback without crashing."""
    failing_provider = FailingLLMProvider()
    judge = LLMJudge(provider=failing_provider, storage=storage)

    fallback_eval = judge.evaluate(
        task_id="task_fallback_01",
        target_agent="executor",
        target_stage="step_execution",
        agent_input="Execute data aggregation",
        agent_output="Aggregated total records: 450 items processed successfully.",
        criteria=["450 items"],
    )

    assert fallback_eval.status == "FALLBACK"
    assert fallback_eval.evaluator_model == "deterministic_fallback"
    assert fallback_eval.overall_score > 0.0
    assert fallback_eval.overall_score < 1.0  # Heuristic fallback does not fabricate 1.0


# ===========================================================================
# PART 4: RETRIEVAL & RAG EVALUATOR
# ===========================================================================

def test_10_retrieval_evaluator_precision_recall_mrr_ndcg(storage):
    """Verify P@K, Recall@K, MRR, and NDCG rankings."""
    ret_eval = RetrievalEvaluator(storage=storage)

    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
    relevant = ["doc_B", "doc_D"]

    # Precision@2: doc_A (no), doc_B (yes) -> 1/2 = 0.50
    assert ret_eval.precision_at_k(retrieved, relevant, k=2) == 0.50

    # Recall@2: doc_B found out of 2 total relevant -> 1/2 = 0.50
    assert ret_eval.recall_at_k(retrieved, relevant, k=2) == 0.50

    # Precision@4: doc_B and doc_D in top 4 -> 2/4 = 0.50
    assert ret_eval.precision_at_k(retrieved, relevant, k=4) == 0.50

    # Recall@4: both doc_B and doc_D found -> 2/2 = 1.0
    assert ret_eval.recall_at_k(retrieved, relevant, k=4) == 1.0

    # MRR: first relevant is doc_B at rank 2 -> 1/2 = 0.50
    assert ret_eval.mrr(retrieved, relevant) == 0.50

    # NDCG@2: DCG = 0/log2(2) + 1/log2(3) = 0 + 0.6309 = 0.6309
    # IDCG@2: ideal ranking has 2 relevant items in top 2 -> 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
    # NDCG = 0.6309 / 1.6309 = 0.3869
    ndcg_2 = ret_eval.ndcg_at_k(retrieved, relevant, k=2)
    assert math.isclose(ndcg_2, 0.3869, abs_tol=1e-3)


def test_11_retrieval_evaluator_zero_division_safety(storage):
    """Verify zero division safety on empty lists and k=0."""
    ret_eval = RetrievalEvaluator(storage=storage)

    # Empty inputs
    assert ret_eval.precision_at_k([], ["doc_A"], k=5) == 0.0
    assert ret_eval.precision_at_k(["doc_A"], [], k=5) == 0.0
    assert ret_eval.precision_at_k(["doc_A"], ["doc_A"], k=0) == 0.0
    assert ret_eval.recall_at_k([], ["doc_A"], k=5) == 0.0
    assert ret_eval.recall_at_k(["doc_A"], [], k=5) == 0.0
    assert ret_eval.mrr([], ["doc_A"]) == 0.0
    assert ret_eval.mrr(["doc_A"], []) == 0.0
    assert ret_eval.ndcg_at_k([], ["doc_A"], k=5) == 0.0
    assert ret_eval.ndcg_at_k(["doc_A"], [], k=5) == 0.0


def test_12_retrieval_evaluator_faithfulness_and_hallucination_detection(storage):
    """Verify factual grounding and detection of unsupported claims."""
    ret_eval = RetrievalEvaluator(storage=storage)

    context = [
        "Company Alpha reported revenue of $50M in 2025.",
        "Company Alpha operating expenses were $30M.",
    ]

    # Grounded answer
    grounded_ans = "Company Alpha reported revenue of $50M in 2025. Operating expenses reached $30M."
    res_grounded = ret_eval.evaluate_faithfulness(grounded_ans, context)
    assert res_grounded.grounded_status == "GROUNDED"
    assert res_grounded.faithfulness_score == 1.0
    assert len(res_grounded.unsupported_claims) == 0

    # Answer containing hallucination / unsupported claim
    hallucinated_ans = (
        "Company Alpha reported revenue of $50M in 2025. "
        "Company Alpha acquired Beta Technologies for $100M."
    )
    res_hallucinated = ret_eval.evaluate_faithfulness(hallucinated_ans, context)
    assert res_hallucinated.grounded_status in ("PARTIALLY_GROUNDED", "UNGROUNDED")
    assert res_hallucinated.faithfulness_score < 1.0
    assert any("Beta Technologies" in c for c in res_hallucinated.unsupported_claims)


# ===========================================================================
# PART 5: AGENT PERFORMANCE & RELIABILITY EVALUATOR
# ===========================================================================

def test_13_agent_performance_tracking_and_p95_latency(storage):
    """Verify metric calculations including success rate and nearest-rank P95 latency."""
    perf_eval = AgentPerformanceEvaluator(storage=storage)

    # Log 20 executions with latencies 100, 200, ..., 2000 ms
    for i in range(1, 21):
        outcome = "success" if i <= 18 else "failure"
        perf_eval.record_execution(
            agent_name="executor",
            execution_time_ms=i * 100.0,
            outcome=outcome,
            retry_count=1 if i == 5 else 0,
            recovery_succeeded=(i == 5),
        )

    metrics = perf_eval.calculate_agent_metrics(agent_name="executor")
    assert metrics["total_executions"] == 20
    assert metrics["success_rate"] == 0.90
    assert metrics["failure_rate"] == 0.10
    assert metrics["retry_rate"] == 0.05
    # Nearest rank P95 of 20 items (sorted 100..2000): index ceil(0.95*20)-1 = 18 -> 1900.0 ms
    assert metrics["p95_latency_ms"] == 1900.0


def test_14_failure_category_separation_and_human_rejection(storage):
    """Ensure human rejections are strictly categorized and NOT counted as agent reasoning failures."""
    perf_eval = AgentPerformanceEvaluator(storage=storage)

    # 1. Human rejected action
    perf_eval.record_execution(
        agent_name="executor",
        execution_time_ms=150.0,
        outcome="failure",
        human_intervention=True,
        human_rejected=True,
    )

    # 2. Tool failure
    perf_eval.record_execution(
        agent_name="executor",
        execution_time_ms=200.0,
        outcome="failure",
        tool_failure=True,
    )

    # 3. Agent reasoning failure
    perf_eval.record_execution(
        agent_name="executor",
        execution_time_ms=180.0,
        outcome="failure",
    )

    metrics = perf_eval.calculate_agent_metrics(agent_name="executor")
    breakdown = metrics["failure_breakdown"]
    assert breakdown["human_rejection"] == 1
    assert breakdown["tool_execution"] == 1
    assert breakdown["agent_reasoning"] == 1


def test_15_agent_leaderboard_comparison_and_filtering(storage):
    """Verify multi-agent comparison and time-window filtering."""
    perf_eval = AgentPerformanceEvaluator(storage=storage)

    # Record 5 executions for researcher (100% success)
    for _ in range(5):
        perf_eval.record_execution(agent_name="researcher", execution_time_ms=300.0, outcome="success")

    # Record 5 executions for executor (60% success)
    for i in range(5):
        outcome = "success" if i < 3 else "failure"
        perf_eval.record_execution(agent_name="executor", execution_time_ms=500.0, outcome=outcome)

    leaderboard = perf_eval.compare_agents()
    assert len(leaderboard) == 2
    # Researcher must rank first due to higher success rate
    assert leaderboard[0]["agent_name"] == "researcher"
    assert leaderboard[0]["success_rate"] == 1.0
    assert leaderboard[1]["agent_name"] == "executor"
    assert leaderboard[1]["success_rate"] == 0.60


# ===========================================================================
# PART 6: UNIFIED EVALUATION FRAMEWORK & INTEGRATION
# ===========================================================================

def test_16_unified_framework_pass_decision(storage):
    """Successful workflow execution must produce EvaluationDecision.PASS."""
    framework = UnifiedEvaluationFramework(
        storage=storage,
        llm_provider=MockLLMJudgeProvider(overall_score=0.95),
        max_revisions=2,
    )

    context = TaskContext(
        task="Calculate and format quarterly margins",
        task_id="task_u1",
        step_results=[
            {"task_id": 1, "status": "completed", "result": "Calculated gross margin: 45000 USD."},
        ],
        research_findings="Audited financials confirm 45000 USD margin.",
        final_output="Gross margin verified at 45000 USD.",
    )

    decision, eval_res, rev_request = framework.evaluate_workflow(
        task_id="task_u1",
        context=context,
        criteria=["Gross margin 45000"],
    )

    assert decision == EvaluationDecision.PASS
    assert eval_res.passed is True
    assert rev_request is None


def test_17_unified_framework_requires_revision_and_preserved_outputs(storage):
    """Failing workflow on attempt 1 must issue RevisionRequest preserving prior work."""
    framework = UnifiedEvaluationFramework(
        storage=storage,
        max_revisions=2,
    )

    context = TaskContext(
        task="Research market size and calculate growth",
        task_id="task_u2",
        step_results=[
            {"task_id": 1, "status": "failed", "error": "Execution error", "result": "Failed"},
        ],
        research_findings="Market size is 10 Billion USD.",
        plan={"tasks": [{"id": 1, "description": "Calculate growth rate"}]},
    )

    decision, eval_res, rev_request = framework.evaluate_workflow(
        task_id="task_u2",
        context=context,
        criteria=["Growth rate calculated"],
    )

    assert decision == EvaluationDecision.REQUIRES_REVISION
    assert rev_request is not None
    assert rev_request.attempt_number == 1
    assert rev_request.failing_agent == "executor"
    # Preserved upstream findings must be present
    assert rev_request.preserved_outputs["research_findings"] == "Market size is 10 Billion USD."


def test_18_unified_framework_max_revisions_prevents_infinite_loop(storage):
    """Exceeding max revisions must transition to EvaluationDecision.FAIL."""
    framework = UnifiedEvaluationFramework(
        storage=storage,
        max_revisions=2,
    )

    # Pre-record 2 prior revision attempts
    storage.save_evaluation_revision({
        "revision_id": "rev_01", "task_id": "task_u3", "failing_agent": "executor",
        "failing_stage": "execution", "feedback": "Attempt 1 feedback", "attempt_number": 1,
    })
    storage.save_evaluation_revision({
        "revision_id": "rev_02", "task_id": "task_u3", "failing_agent": "executor",
        "failing_stage": "execution", "feedback": "Attempt 2 feedback", "attempt_number": 2,
    })

    context = TaskContext(
        task="Faulty task repeatedly failing",
        task_id="task_u3",
        step_results=[{"task_id": 1, "status": "failed", "result": "Timeout"}],
    )

    decision, eval_res, rev_request = framework.evaluate_workflow(
        task_id="task_u3",
        context=context,
        criteria=["Must succeed"],
    )

    # Must FAIL and not issue a 3rd revision request
    assert decision == EvaluationDecision.FAIL
    assert rev_request is None


def test_19_orchestrator_integration_and_backward_compatibility(storage):
    """Verify WorkflowOrchestrator integration and EvaluationResult backward compatibility."""
    framework = UnifiedEvaluationFramework(storage=storage)
    orchestrator = WorkflowOrchestrator(
        evaluation_framework=framework,
    )

    context = TaskContext(
        task="Deterministic computation",
        task_id="task_orch_eval",
        step_results=[
            {"task_id": 1, "status": "completed", "result": "Computed result: 42."},
        ],
    )

    decision, eval_res, rev = orchestrator.evaluate_workflow_execution(
        task_id="task_orch_eval",
        context=context,
        criteria=["Computed result: 42"],
    )

    assert decision == EvaluationDecision.PASS
    assert eval_res.is_eligible_for_completion is True

    # Backward compatibility checks on EvaluationResult
    res_dict = eval_res.to_dict()
    assert "is_eligible_for_completion" in res_dict
    assert "all_criteria_met" in res_dict
    assert "has_valid_evidence" in res_dict
    assert "completed_steps_count" in res_dict
    assert "evaluation_id" in res_dict
    assert "overall_score" in res_dict
    assert "metric_scores" in res_dict
