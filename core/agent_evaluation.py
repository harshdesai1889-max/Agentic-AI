"""
core/agent_evaluation.py

Comprehensive Agent Evaluation Framework for Day 228 (Parts 1 to 6).

Implements:
1. Part 1: Agent Evaluation Architecture & Metrics
   - EvaluationMetric enum (8 dimensions).
   - EvaluationThreshold enum and classification.
   - MetricWeightsConfig with strict 100% sum validation and role defaults.
   - AgentEvaluationEngine producing structured, persisted EvaluationResult.
2. Part 2: Task Outcome Evaluator
   - OutcomeClassification enum (SUCCESS, PARTIAL_SUCCESS, FAILURE, INCONCLUSIVE).
   - TaskOutcome dataclass.
   - TaskOutcomeEvaluator with independent numeric calculations, mandatory step enforcement,
     missing output penalization, and ResultEvaluator integration.
3. Part 3: LLM-as-a-Judge
   - LLMJudgeEvaluation dataclass.
   - LLMJudge evaluating findings, plans, tool executions, and supervisor responses.
   - Structured JSON schema validation and independent deterministic fallback.
4. Part 4: Retrieval & RAG Evaluator
   - Precision@K, Recall@K, MRR, NDCG with zero-division safety.
   - Context relevance and context coverage.
   - Answer faithfulness / groundedness checking and unsupported claim detection.
5. Part 5: Agent Performance & Reliability Evaluator
   - AgentPerformanceRecord dataclass.
   - AgentPerformanceEvaluator tracking success rates, failure rates, retry rates,
     recovery rates, average and P95 latencies.
   - Separation of agent failures from tool failures, provider errors, and human rejections.
6. Part 6: Unified Evaluation Framework & Integration
   - EvaluationDecision enum (PASS, PASS_WITH_WARNINGS, REQUIRES_REVISION, FAIL, INCONCLUSIVE).
   - RevisionRequest dataclass with max revision loop defense.
   - UnifiedEvaluationFramework coordinating the complete 8-stage evaluation lifecycle.
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from core.evaluator import EvaluationResult, ResultEvaluator, has_substantive_evidence, is_vacuous_claim
from core.persistent_storage import SQLiteMessageStorage
from core.provider_interface import LLMMessage, LLMProvider, LLMResponse
from core.orchestrator import TaskContext, WorkflowStatus
from utils.logging_config import get_logger

logger = get_logger()


# ===========================================================================
# PART 1: AGENT EVALUATION ARCHITECTURE & METRICS
# ===========================================================================

class EvaluationMetric(str, Enum):
    """Core evaluation dimensions for AI agents."""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    INSTRUCTION_ADHERENCE = "instruction_adherence"
    EVIDENCE_QUALITY = "evidence_quality"
    OUTPUT_CONSISTENCY = "output_consistency"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"


class EvaluationThreshold(str, Enum):
    """Performance tier classifications based on aggregate score."""
    EXCELLENT = "EXCELLENT"             # >= 0.90
    GOOD = "GOOD"                       # >= 0.75 and < 0.90
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"  # >= 0.50 and < 0.75
    FAILED = "FAILED"                   # < 0.50


def classify_threshold(score: float) -> EvaluationThreshold:
    """Classify a numeric score [0.0, 1.0] into a performance threshold."""
    if score >= 0.90:
        return EvaluationThreshold.EXCELLENT
    elif score >= 0.75:
        return EvaluationThreshold.GOOD
    elif score >= 0.50:
        return EvaluationThreshold.NEEDS_IMPROVEMENT
    else:
        return EvaluationThreshold.FAILED


@dataclass
class MetricWeightsConfig:
    """Configuration mapping metrics to relative weights, strictly summing to 1.0."""
    weights: Dict[str, float]

    def __post_init__(self):
        normalized: Dict[str, float] = {}
        total = 0.0
        for k, v in self.weights.items():
            metric_key = k.value if isinstance(k, EvaluationMetric) else str(k).lower()
            if v < 0:
                raise ValueError(f"Metric weight for '{metric_key}' cannot be negative: {v}")
            normalized[metric_key] = float(v)
            total += float(v)

        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(f"Metric weights must sum to 1.0 (100%), got {total:.4f}")

        self.weights = normalized

    def get_weight(self, metric: Union[EvaluationMetric, str]) -> float:
        """Retrieve weight for a metric, defaulting to 0.0 if not assigned."""
        key = metric.value if isinstance(metric, EvaluationMetric) else str(metric).lower()
        return self.weights.get(key, 0.0)

    @classmethod
    def default(cls) -> "MetricWeightsConfig":
        """Uniform distribution across all 8 evaluation metrics (12.5% each)."""
        return cls({m.value: 0.125 for m in EvaluationMetric})

    @classmethod
    def for_researcher(cls) -> "MetricWeightsConfig":
        """Specialized weights for ResearcherAgent (evidence, relevance, completeness)."""
        return cls({
            EvaluationMetric.EVIDENCE_QUALITY.value: 0.30,
            EvaluationMetric.RELEVANCE.value: 0.25,
            EvaluationMetric.COMPLETENESS.value: 0.20,
            EvaluationMetric.CORRECTNESS.value: 0.15,
            EvaluationMetric.RELIABILITY.value: 0.10,
        })

    @classmethod
    def for_planner(cls) -> "MetricWeightsConfig":
        """Specialized weights for PlannerAgent (instruction adherence, completeness, correctness)."""
        return cls({
            EvaluationMetric.INSTRUCTION_ADHERENCE.value: 0.30,
            EvaluationMetric.COMPLETENESS.value: 0.25,
            EvaluationMetric.CORRECTNESS.value: 0.20,
            EvaluationMetric.OUTPUT_CONSISTENCY.value: 0.15,
            EvaluationMetric.EFFICIENCY.value: 0.10,
        })

    @classmethod
    def for_executor(cls) -> "MetricWeightsConfig":
        """Specialized weights for ExecutorAgent (correctness, reliability, efficiency)."""
        return cls({
            EvaluationMetric.CORRECTNESS.value: 0.35,
            EvaluationMetric.RELIABILITY.value: 0.25,
            EvaluationMetric.EFFICIENCY.value: 0.20,
            EvaluationMetric.INSTRUCTION_ADHERENCE.value: 0.10,
            EvaluationMetric.EVIDENCE_QUALITY.value: 0.10,
        })

    @classmethod
    def for_supervisor(cls) -> "MetricWeightsConfig":
        """Specialized weights for SupervisorAgent (instruction adherence, correctness, consistency)."""
        return cls({
            EvaluationMetric.INSTRUCTION_ADHERENCE.value: 0.25,
            EvaluationMetric.CORRECTNESS.value: 0.25,
            EvaluationMetric.OUTPUT_CONSISTENCY.value: 0.20,
            EvaluationMetric.COMPLETENESS.value: 0.15,
            EvaluationMetric.RELIABILITY.value: 0.15,
        })


class AgentEvaluationEngine:
    """Core evaluation engine assessing agent outputs against weighted metrics."""

    def __init__(self, storage: Optional[SQLiteMessageStorage] = None):
        self.storage = storage
        self.default_config = MetricWeightsConfig.default()
        self.role_configs: Dict[str, MetricWeightsConfig] = {
            "researcher": MetricWeightsConfig.for_researcher(),
            "planner": MetricWeightsConfig.for_planner(),
            "executor": MetricWeightsConfig.for_executor(),
            "supervisor": MetricWeightsConfig.for_supervisor(),
        }

    def evaluate_agent_output(
        self,
        task_id: str,
        agent_name: str,
        metric_scores: Dict[Union[EvaluationMetric, str], float],
        evidence: Optional[Dict[str, Any]] = None,
        config: Optional[MetricWeightsConfig] = None,
        evaluation_type: str = "agent_evaluation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Calculate weighted score, apply evidence penalties, and generate EvaluationResult."""
        norm_agent = agent_name.strip().lower()
        active_config = config or self.role_configs.get(norm_agent, self.default_config)

        # Normalize score keys
        normalized_scores: Dict[str, float] = {}
        for k, v in metric_scores.items():
            key = k.value if isinstance(k, EvaluationMetric) else str(k).lower()
            clamped = max(0.0, min(1.0, float(v)))
            normalized_scores[key] = round(clamped, 4)

        # Compute weighted sum over assigned metrics
        weighted_sum = 0.0
        total_weight_used = 0.0
        for metric_name, weight in active_config.weights.items():
            if weight > 0:
                score = normalized_scores.get(metric_name, 0.0)
                weighted_sum += score * weight
                total_weight_used += weight

        overall_score = round(weighted_sum / total_weight_used, 4) if total_weight_used > 0 else 0.0

        # Evidence validation and penalization
        failures: List[str] = []
        warnings: List[str] = []
        evidence_dict = dict(evidence or {})

        evidence_score = normalized_scores.get(EvaluationMetric.EVIDENCE_QUALITY.value)
        if evidence_score is not None and evidence_score < 0.50:
            failures.append(f"Substandard evidence quality: {evidence_score:.2f}")
            # Penalize overall score if evidence is explicitly poor
            overall_score = round(max(0.0, overall_score * 0.75), 4)

        if not evidence_dict and "evidence_quality" in active_config.weights:
            warnings.append("No explicit evidence payload provided for evaluation.")

        # Identify failing or warning metrics
        for metric_name, score in normalized_scores.items():
            if score < 0.50:
                failures.append(f"Metric '{metric_name}' failed with score {score:.2f}")
            elif score < 0.70:
                warnings.append(f"Metric '{metric_name}' below acceptable quality ({score:.2f})")

        threshold = classify_threshold(overall_score)
        passed = (threshold in (EvaluationThreshold.EXCELLENT, EvaluationThreshold.GOOD)) and (len(failures) == 0)

        eval_id = f"eval_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        reason = (
            f"Overall score: {overall_score:.2f} ({threshold.value}). "
            f"Failures: {len(failures)}, Warnings: {len(warnings)}."
        )

        result = EvaluationResult(
            is_eligible_for_completion=passed,
            all_criteria_met=passed,
            has_valid_evidence=len(evidence_dict) > 0 and (evidence_score is None or evidence_score >= 0.50),
            passed_criteria=[m for m, s in normalized_scores.items() if s >= 0.70],
            failed_criteria=[m for m, s in normalized_scores.items() if s < 0.50],
            missing_fields=[],
            completed_steps=[],
            failed_steps=[],
            evaluation_notes=reason,
            status="completed" if passed else "failed",
            evaluation_id=eval_id,
            task_id=task_id,
            agent_name=agent_name,
            evaluation_type=evaluation_type,
            overall_score=overall_score,
            metric_scores=normalized_scores,
            passed=passed,
            failures=failures,
            warnings=warnings,
            evidence=evidence_dict,
            evaluation_reason=reason,
            timestamp=timestamp,
            metadata=dict(metadata or {}, threshold=threshold.value),
        )

        if self.storage:
            try:
                self.storage.save_agent_evaluation(result.to_dict())
            except Exception as exc:
                logger.warning(f"[AgentEvaluationEngine] Could not persist evaluation {eval_id}: {exc}")

        return result


# ===========================================================================
# PART 2: TASK OUTCOME EVALUATOR
# ===========================================================================

class OutcomeClassification(str, Enum):
    """Categorical assessment of task execution outcome."""
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILURE = "FAILURE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class TaskOutcome:
    """Structured report of task outcome evaluation."""
    outcome_id: str
    task_id: str
    objective: str
    classification: OutcomeClassification
    passed_criteria: List[str]
    failed_criteria: List[str]
    numeric_checks: Dict[str, Any]
    missing_fields: List[str]
    evidence_verified: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "task_id": self.task_id,
            "objective": self.objective,
            "classification": self.classification.value if isinstance(self.classification, OutcomeClassification) else str(self.classification),
            "passed_criteria": self.passed_criteria,
            "failed_criteria": self.failed_criteria,
            "numeric_checks": self.numeric_checks,
            "missing_fields": self.missing_fields,
            "evidence_verified": self.evidence_verified,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class TaskOutcomeEvaluator:
    """Evaluates task outcome with independent numeric audits, mandatory steps, and missing output checks."""

    def __init__(self, storage: Optional[SQLiteMessageStorage] = None):
        self.storage = storage
        self.result_evaluator = ResultEvaluator()

    @staticmethod
    def audit_numeric_criteria(
        criterion: str,
        observed_text: str,
    ) -> Tuple[bool, Optional[str]]:
        """Verify numeric equality or calculation independently.

        Extracts expected numbers from criteria and verifies they exist in the output text.
        Also parses explicit calculations like '10 + 20 = 30' or 'sum of 5 and 15 is 20'.
        """
        if not criterion or not observed_text:
            return False, "Missing criterion or text"

        criterion_lower = criterion.lower()
        observed_lower = observed_text.lower()

        # Check for arithmetic equality pattern like 'A + B = C' or 'A * B = C'
        math_expr = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)", criterion)
        if math_expr:
            a, op, b, c = float(math_expr.group(1)), math_expr.group(2), float(math_expr.group(3)), float(math_expr.group(4))
            actual = 0.0
            if op == "+": actual = a + b
            elif op == "-": actual = a - b
            elif op == "*": actual = a * b
            elif op == "/" and b != 0: actual = a / b

            if not math.isclose(actual, c, abs_tol=1e-3):
                return False, f"Criterion formula invalid: {a}{op}{b} is {actual}, not {c}"

            expected_num_str = math_expr.group(4)
            if expected_num_str not in observed_text:
                return False, f"Calculated value {expected_num_str} not found in output"
            return True, f"Verified arithmetic {a} {op} {b} = {c}"

        # Standard number extraction check
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", criterion)
        if numbers:
            missing = [n for n in numbers if n not in observed_text]
            if missing:
                return False, f"Numbers missing from output: {missing}"
            return True, f"All required numbers found: {numbers}"

        return True, "No numeric constraints found"

    def evaluate_outcome(
        self,
        task_id: str,
        objective: str,
        success_criteria: List[str],
        execution_steps: List[Dict[str, Any]],
        expected_outputs: Optional[List[str]] = None,
        mandatory_steps: Optional[List[Union[int, str]]] = None,
        research_findings: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskOutcome:
        """Perform comprehensive deterministic task outcome evaluation."""
        outcome_id = f"outcome_{uuid.uuid4().hex[:8]}"
        missing_fields: List[str] = []
        numeric_checks: Dict[str, Any] = {}
        passed_criteria: List[str] = []
        failed_criteria: List[str] = []

        # If no execution steps at all
        if not execution_steps:
            outcome = TaskOutcome(
                outcome_id=outcome_id,
                task_id=task_id,
                objective=objective,
                classification=OutcomeClassification.FAILURE,
                passed_criteria=[],
                failed_criteria=list(success_criteria),
                numeric_checks={"error": "No execution steps provided"},
                missing_fields=["execution_steps"],
                evidence_verified=False,
                metadata=dict(metadata or {}),
            )
            if self.storage:
                self.storage.save_task_outcome(outcome.to_dict())
            return outcome

        # Combine text for text-based checks
        step_results_text = " ".join([str(s.get("result", "")) for s in execution_steps])
        combined_text = f"{step_results_text} {research_findings or ''}".strip()

        # 1. Mandatory Steps Audit
        mandatory_failed = False
        mandatory_list = mandatory_steps or []
        for idx, step in enumerate(execution_steps, start=1):
            step_id = step.get("task_id", step.get("id", idx))
            is_mandatory = (step_id in mandatory_list) or (idx in mandatory_list) or step.get("mandatory", False)
            if is_mandatory:
                if step.get("status") != "completed" or step.get("error"):
                    mandatory_failed = True
                    missing_fields.append(f"mandatory_step_{step_id}_failed")

        # 2. Check Expected Output Schema Fields
        if expected_outputs:
            for exp in expected_outputs:
                exp_clean = exp.strip().lower()
                found = any(exp_clean in str(s.get(exp, "")).lower() or exp_clean in str(s.get("result", "")).lower() for s in execution_steps)
                if not found and exp_clean not in combined_text.lower():
                    missing_fields.append(f"missing_expected_output:{exp}")

        # 3. Numeric & Semantic Success Criteria Audit
        for crit in success_criteria:
            num_ok, num_detail = self.audit_numeric_criteria(crit, combined_text)
            numeric_checks[crit] = {"verified": num_ok, "detail": num_detail}

            if not num_ok:
                failed_criteria.append(crit)
                continue

            # Check quotes if present
            quotes = re.findall(r"['\"](.*?)['\"]", crit)
            if quotes:
                all_quotes = all(q.lower() in combined_text.lower() for q in quotes)
                if not all_quotes:
                    failed_criteria.append(crit)
                    continue

            # Check keywords
            words = [w for w in re.findall(r"\b\w+\b", crit.lower()) if len(w) > 3]
            if words:
                matches = [w for w in words if w in combined_text.lower()]
                if len(matches) / len(words) < 0.40:
                    failed_criteria.append(crit)
                    continue

            passed_criteria.append(crit)

        # 4. Evidence Verification via ResultEvaluator
        completed_steps = [s for s in execution_steps if s.get("status") == "completed" and not s.get("error")]
        failed_steps = [s for s in execution_steps if s.get("status") != "completed" or s.get("error")]
        evidence_verified = False
        if completed_steps:
            evidence_verified = all(has_substantive_evidence(s.get("result")) for s in completed_steps)

        # 5. Outcome Classification
        if not evidence_verified or len(completed_steps) == 0 or len(passed_criteria) == 0:
            classification = OutcomeClassification.FAILURE
        elif mandatory_failed:
            classification = OutcomeClassification.PARTIAL_SUCCESS
        elif len(failed_criteria) == 0 and len(failed_steps) == 0 and len(missing_fields) == 0:
            classification = OutcomeClassification.SUCCESS
        elif len(passed_criteria) > 0:
            classification = OutcomeClassification.PARTIAL_SUCCESS
        else:
            classification = OutcomeClassification.FAILURE

        outcome = TaskOutcome(
            outcome_id=outcome_id,
            task_id=task_id,
            objective=objective,
            classification=classification,
            passed_criteria=passed_criteria,
            failed_criteria=failed_criteria,
            numeric_checks=numeric_checks,
            missing_fields=missing_fields,
            evidence_verified=evidence_verified,
            metadata=dict(metadata or {}, completed_steps=len(completed_steps), failed_steps=len(failed_steps)),
        )

        if self.storage:
            try:
                self.storage.save_task_outcome(outcome.to_dict())
            except Exception as exc:
                logger.warning(f"[TaskOutcomeEvaluator] Could not persist outcome {outcome_id}: {exc}")

        return outcome


# ===========================================================================
# PART 3: LLM-AS-A-JUDGE
# ===========================================================================

@dataclass
class LLMJudgeEvaluation:
    """Structured result of an LLM Judge evaluation."""
    judge_id: str
    task_id: str
    target_agent: str
    target_stage: str
    accuracy: float
    completeness: float
    relevance: float
    instruction_adherence: float
    evidence_support: float
    overall_score: float
    identified_weaknesses: List[str]
    confidence: float
    reasoning_summary: str
    evaluator_model: str
    evaluator_provider: str
    status: str  # "COMPLETED", "FALLBACK", "FAILED", "INCONCLUSIVE"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "task_id": self.task_id,
            "target_agent": self.target_agent,
            "target_stage": self.target_stage,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "instruction_adherence": self.instruction_adherence,
            "evidence_support": self.evidence_support,
            "overall_score": self.overall_score,
            "identified_weaknesses": self.identified_weaknesses,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "evaluator_model": self.evaluator_model,
            "evaluator_provider": self.evaluator_provider,
            "status": self.status,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class LLMJudge:
    """LLM-as-a-Judge for evaluating agent outputs with deterministic fallback."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        storage: Optional[SQLiteMessageStorage] = None,
    ):
        self.provider = provider
        self.model = model or "default-judge-model"
        self.storage = storage

    def evaluate(
        self,
        task_id: str,
        target_agent: str,
        target_stage: str,
        agent_input: str,
        agent_output: str,
        criteria: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LLMJudgeEvaluation:
        """Perform evaluation of an agent's output using LLM or deterministic fallback."""
        judge_id = f"judge_{uuid.uuid4().hex[:8]}"

        # Prevent evaluator self-evaluation
        if target_agent.lower() in ("llm_judge", "judge", "evaluator"):
            return LLMJudgeEvaluation(
                judge_id=judge_id,
                task_id=task_id,
                target_agent=target_agent,
                target_stage=target_stage,
                accuracy=0.0,
                completeness=0.0,
                relevance=0.0,
                instruction_adherence=0.0,
                evidence_support=0.0,
                overall_score=0.0,
                identified_weaknesses=["Evaluator cannot evaluate itself."],
                confidence=1.0,
                reasoning_summary="Self-evaluation rejected by evaluator independence policy.",
                evaluator_model=self.model,
                evaluator_provider="system",
                status="FAILED",
                metadata=dict(metadata or {}),
            )

        # Attempt LLM-based evaluation if provider is configured
        if self.provider is not None:
            try:
                llm_eval = self._invoke_llm_judge(
                    judge_id=judge_id,
                    task_id=task_id,
                    target_agent=target_agent,
                    target_stage=target_stage,
                    agent_input=agent_input,
                    agent_output=agent_output,
                    criteria=criteria or [],
                    metadata=metadata,
                )
                if llm_eval is not None:
                    if self.storage:
                        self.storage.save_llm_judge_evaluation(llm_eval.to_dict())
                    return llm_eval
            except Exception as exc:
                logger.warning(f"[LLMJudge] LLM provider failed, falling back to deterministic evaluation: {exc}")

        # Deterministic Fallback
        fallback_eval = self._deterministic_fallback_evaluation(
            judge_id=judge_id,
            task_id=task_id,
            target_agent=target_agent,
            target_stage=target_stage,
            agent_input=agent_input,
            agent_output=agent_output,
            criteria=criteria or [],
            metadata=metadata,
        )

        if self.storage:
            try:
                self.storage.save_llm_judge_evaluation(fallback_eval.to_dict())
            except Exception as exc:
                logger.warning(f"[LLMJudge] Failed to persist fallback evaluation: {exc}")

        return fallback_eval

    def _invoke_llm_judge(
        self,
        judge_id: str,
        task_id: str,
        target_agent: str,
        target_stage: str,
        agent_input: str,
        agent_output: str,
        criteria: List[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[LLMJudgeEvaluation]:
        """Call LLM provider and validate structured JSON response."""
        system_prompt = (
            "You are an impartial, rigorous AI Evaluation Judge. Evaluate the target agent's response.\n"
            "Return ONLY a valid JSON object with the following schema:\n"
            "{\n"
            '  "accuracy": float (0.0 to 1.0),\n'
            '  "completeness": float (0.0 to 1.0),\n'
            '  "relevance": float (0.0 to 1.0),\n'
            '  "instruction_adherence": float (0.0 to 1.0),\n'
            '  "evidence_support": float (0.0 to 1.0),\n'
            '  "overall_score": float (0.0 to 1.0),\n'
            '  "identified_weaknesses": [list of strings],\n'
            '  "confidence": float (0.0 to 1.0),\n'
            '  "reasoning_summary": string\n'
            "}"
        )

        user_content = (
            f"Target Agent: {target_agent}\n"
            f"Stage: {target_stage}\n"
            f"Criteria: {json.dumps(criteria)}\n"
            f"Agent Input / Task: {agent_input}\n"
            f"Agent Output: {agent_output}\n"
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_content),
        ]

        response: LLMResponse = self.provider.generate(messages=messages, model=self.model)
        raw_text = response.content or ""

        # Extract JSON from potential code fences
        cleaned = raw_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        data = json.loads(cleaned)

        # Validate presence and bounds of numeric fields
        req_numeric = ["accuracy", "completeness", "relevance", "instruction_adherence", "evidence_support"]
        for field_name in req_numeric:
            if field_name not in data or not isinstance(data[field_name], (int, float)):
                raise ValueError(f"Missing or invalid numeric field '{field_name}' in judge response")

        acc = max(0.0, min(1.0, float(data["accuracy"])))
        comp = max(0.0, min(1.0, float(data["completeness"])))
        rel = max(0.0, min(1.0, float(data["relevance"])))
        inst = max(0.0, min(1.0, float(data["instruction_adherence"])))
        evid = max(0.0, min(1.0, float(data["evidence_support"])))
        overall = max(0.0, min(1.0, float(data.get("overall_score", (acc + comp + rel + inst + evid) / 5.0))))
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.90))))

        return LLMJudgeEvaluation(
            judge_id=judge_id,
            task_id=task_id,
            target_agent=target_agent,
            target_stage=target_stage,
            accuracy=round(acc, 4),
            completeness=round(comp, 4),
            relevance=round(rel, 4),
            instruction_adherence=round(inst, 4),
            evidence_support=round(evid, 4),
            overall_score=round(overall, 4),
            identified_weaknesses=list(data.get("identified_weaknesses", [])),
            confidence=round(conf, 4),
            reasoning_summary=str(data.get("reasoning_summary", "LLM Judge evaluation completed.")),
            evaluator_model=self.model,
            evaluator_provider="llm_provider",
            status="COMPLETED",
            metadata=dict(metadata or {}),
        )

    def _deterministic_fallback_evaluation(
        self,
        judge_id: str,
        task_id: str,
        target_agent: str,
        target_stage: str,
        agent_input: str,
        agent_output: str,
        criteria: List[str],
        metadata: Optional[Dict[str, Any]],
    ) -> LLMJudgeEvaluation:
        """Deterministic evaluation without fabricating scores or crashing."""
        weaknesses: List[str] = []

        if not agent_output or not agent_output.strip():
            return LLMJudgeEvaluation(
                judge_id=judge_id,
                task_id=task_id,
                target_agent=target_agent,
                target_stage=target_stage,
                accuracy=0.0,
                completeness=0.0,
                relevance=0.0,
                instruction_adherence=0.0,
                evidence_support=0.0,
                overall_score=0.0,
                identified_weaknesses=["Output is empty or missing."],
                confidence=1.0,
                reasoning_summary="Deterministic fallback: Empty agent output.",
                evaluator_model="deterministic_fallback",
                evaluator_provider="system",
                status="FALLBACK",
                metadata=dict(metadata or {}),
            )

        # Check vacuous claims
        if is_vacuous_claim(agent_output):
            weaknesses.append("Output contains vacuous self-claims of completion without substantive evidence.")

        # Check evidence support
        has_evidence = has_substantive_evidence(agent_output)
        evidence_score = 0.85 if has_evidence else 0.10
        if not has_evidence:
            weaknesses.append("Lacks substantive evidence or empirical facts.")

        # Keyword matching against input to estimate relevance
        input_words = set(re.findall(r"\b\w{4,}\b", agent_input.lower()))
        output_words = set(re.findall(r"\b\w{4,}\b", agent_output.lower()))
        relevance_score = 0.50
        if input_words:
            overlap = len(input_words.intersection(output_words)) / len(input_words)
            relevance_score = min(1.0, max(0.20, overlap * 1.5))

        # Criteria matching for instruction adherence and completeness
        crit_matched = 0
        for crit in criteria:
            crit_words = set(re.findall(r"\b\w{4,}\b", crit.lower()))
            if crit_words and len(crit_words.intersection(output_words)) / len(crit_words) >= 0.50:
                crit_matched += 1

        adherence_score = crit_matched / len(criteria) if criteria else (0.80 if has_evidence else 0.40)
        completeness_score = min(1.0, max(0.30, len(agent_output) / 200.0))

        # In fallback, accuracy cannot be completely checked without ground truth or LLM
        accuracy_score = round((adherence_score + evidence_score) / 2.0, 4)
        overall = round((accuracy_score + completeness_score + relevance_score + adherence_score + evidence_score) / 5.0, 4)

        return LLMJudgeEvaluation(
            judge_id=judge_id,
            task_id=task_id,
            target_agent=target_agent,
            target_stage=target_stage,
            accuracy=accuracy_score,
            completeness=round(completeness_score, 4),
            relevance=round(relevance_score, 4),
            instruction_adherence=round(adherence_score, 4),
            evidence_support=round(evidence_score, 4),
            overall_score=overall,
            identified_weaknesses=weaknesses,
            confidence=0.70,
            reasoning_summary="Deterministic fallback evaluation based on textual heuristic and evidence verification.",
            evaluator_model="deterministic_fallback",
            evaluator_provider="system",
            status="FALLBACK",
            metadata=dict(metadata or {}),
        )


# ===========================================================================
# PART 4: RETRIEVAL & RAG EVALUATOR
# ===========================================================================

@dataclass
class RetrievalMetricsResult:
    """Standard information retrieval metrics."""
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    context_relevance: float
    context_coverage: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "context_relevance": self.context_relevance,
            "context_coverage": self.context_coverage,
        }


@dataclass
class FaithfulnessResult:
    """Factuality and groundedness report."""
    faithfulness_score: float
    grounded_status: str  # "GROUNDED", "PARTIALLY_GROUNDED", "UNGROUNDED"
    unsupported_claims: List[str]
    supported_claims: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness_score": self.faithfulness_score,
            "grounded_status": self.grounded_status,
            "unsupported_claims": self.unsupported_claims,
            "supported_claims": self.supported_claims,
        }


class RetrievalEvaluator:
    """Evaluates RAG and document retrieval pipelines with zero-division safety."""

    def __init__(self, storage: Optional[SQLiteMessageStorage] = None):
        self.storage = storage

    @staticmethod
    def _is_doc_relevant(doc: str, relevant_docs: Sequence[str]) -> bool:
        """Determine if a retrieved document is relevant."""
        doc_clean = doc.strip().lower()
        for rel in relevant_docs:
            rel_clean = rel.strip().lower()
            if rel_clean == doc_clean or rel_clean in doc_clean or doc_clean in rel_clean:
                return True
        return False

    def precision_at_k(self, retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
        """Precision@K = (relevant in top K) / K with zero-division safety."""
        if k <= 0 or not retrieved or not relevant:
            return 0.0
        top_k = retrieved[:k]
        relevant_retrieved = sum(1 for d in top_k if self._is_doc_relevant(d, relevant))
        return round(relevant_retrieved / k, 4)

    def recall_at_k(self, retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
        """Recall@K = (relevant in top K) / (total relevant) with zero-division safety."""
        if k <= 0 or not retrieved or not relevant:
            return 0.0
        top_k = retrieved[:k]
        # Count unique relevant docs found in top K
        found_count = 0
        for rel in relevant:
            if any(self._is_doc_relevant(d, [rel]) for d in top_k):
                found_count += 1
        return round(found_count / len(relevant), 4)

    def mrr(self, retrieved: Sequence[str], relevant: Sequence[str]) -> float:
        """Mean Reciprocal Rank of first relevant document."""
        if not retrieved or not relevant:
            return 0.0
        for idx, doc in enumerate(retrieved, start=1):
            if self._is_doc_relevant(doc, relevant):
                return round(1.0 / idx, 4)
        return 0.0

    def ndcg_at_k(self, retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain at rank K."""
        if k <= 0 or not retrieved or not relevant:
            return 0.0

        top_k = retrieved[:k]
        dcg = 0.0
        for idx, doc in enumerate(top_k, start=1):
            rel_score = 1.0 if self._is_doc_relevant(doc, relevant) else 0.0
            dcg += rel_score / math.log2(idx + 1)

        # Ideal DCG: all relevant documents placed at top ranks
        ideal_count = min(k, len(relevant))
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))

        if idcg == 0.0:
            return 0.0
        return round(dcg / idcg, 4)

    def evaluate_faithfulness(
        self,
        answer: str,
        context_docs: Sequence[str],
    ) -> FaithfulnessResult:
        """Verify whether claims in the answer are grounded in context documents."""
        if not answer or not answer.strip():
            return FaithfulnessResult(
                faithfulness_score=1.0,
                grounded_status="GROUNDED",
                unsupported_claims=[],
                supported_claims=[],
            )

        if not context_docs:
            claims = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 5]
            return FaithfulnessResult(
                faithfulness_score=0.0,
                grounded_status="UNGROUNDED",
                unsupported_claims=claims,
                supported_claims=[],
            )

        combined_context = " ".join(context_docs).lower()
        claims = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 5]

        if not claims:
            return FaithfulnessResult(
                faithfulness_score=1.0,
                grounded_status="GROUNDED",
                unsupported_claims=[],
                supported_claims=[],
            )

        supported: List[str] = []
        unsupported: List[str] = []

        for claim in claims:
            claim_clean = claim.lower()
            # Extract numbers and words
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", claim_clean)
            words = [w for w in re.findall(r"\b\w{4,}\b", claim_clean)]

            numbers_ok = all(n in combined_context for n in numbers)
            words_match_ratio = (
                sum(1 for w in words if w in combined_context) / len(words) if words else 1.0
            )

            if numbers_ok and words_match_ratio >= 0.50:
                supported.append(claim)
            else:
                unsupported.append(claim)

        score = round(len(supported) / len(claims), 4)
        if score >= 0.90:
            status = "GROUNDED"
        elif score >= 0.50:
            status = "PARTIALLY_GROUNDED"
        else:
            status = "UNGROUNDED"

        return FaithfulnessResult(
            faithfulness_score=score,
            grounded_status=status,
            unsupported_claims=unsupported,
            supported_claims=supported,
        )

    def evaluate(
        self,
        task_id: str,
        query: str,
        retrieved_docs: Sequence[str],
        relevant_docs: Sequence[str],
        answer: Optional[str] = None,
        k: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform complete retrieval evaluation and persist findings."""
        retrieval_id = f"ret_{uuid.uuid4().hex[:8]}"

        p_at_k = self.precision_at_k(retrieved_docs, relevant_docs, k)
        r_at_k = self.recall_at_k(retrieved_docs, relevant_docs, k)
        mrr_score = self.mrr(retrieved_docs, relevant_docs)
        ndcg_score = self.ndcg_at_k(retrieved_docs, relevant_docs, k)

        # Context relevance: fraction of retrieved docs containing query keywords
        q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if q_words and retrieved_docs:
            rel_cnt = sum(1 for d in retrieved_docs if any(qw in d.lower() for qw in q_words))
            ctx_relevance = round(rel_cnt / len(retrieved_docs), 4)
        else:
            ctx_relevance = 1.0 if not retrieved_docs else 0.0

        ctx_coverage = r_at_k

        # Faithfulness check
        faithfulness = self.evaluate_faithfulness(answer or "", retrieved_docs)

        record = {
            "retrieval_id": retrieval_id,
            "task_id": task_id,
            "query": query,
            "k": k,
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "mrr": mrr_score,
            "ndcg": ndcg_score,
            "context_relevance": ctx_relevance,
            "context_coverage": ctx_coverage,
            "faithfulness_score": faithfulness.faithfulness_score,
            "grounded_status": faithfulness.grounded_status,
            "unsupported_claims": faithfulness.unsupported_claims,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata or {}),
        }

        if self.storage:
            try:
                self.storage.save_retrieval_evaluation(record)
            except Exception as exc:
                logger.warning(f"[RetrievalEvaluator] Could not persist retrieval record {retrieval_id}: {exc}")

        return record


# ===========================================================================
# PART 5: AGENT PERFORMANCE & RELIABILITY EVALUATOR
# ===========================================================================

@dataclass
class AgentPerformanceRecord:
    """Record of an individual execution for performance analytics."""
    metric_id: str
    agent_name: str
    task_type: str
    workflow_id: Optional[str]
    execution_time_ms: float
    outcome: str
    retry_count: int = 0
    recovery_succeeded: bool = False
    tool_failure: bool = False
    human_intervention: bool = False
    human_rejected: bool = False
    evaluation_failed: bool = False
    failure_category: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "agent_name": self.agent_name,
            "task_type": self.task_type,
            "workflow_id": self.workflow_id,
            "execution_time_ms": self.execution_time_ms,
            "outcome": self.outcome,
            "retry_count": self.retry_count,
            "recovery_succeeded": self.recovery_succeeded,
            "tool_failure": self.tool_failure,
            "human_intervention": self.human_intervention,
            "human_rejected": self.human_rejected,
            "evaluation_failed": self.evaluation_failed,
            "failure_category": self.failure_category,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AgentPerformanceEvaluator:
    """Aggregates and tracks performance, latency, retry rates, and failure categories."""

    def __init__(self, storage: SQLiteMessageStorage):
        self.storage = storage

    def record_execution(
        self,
        agent_name: str,
        execution_time_ms: float,
        outcome: str,
        task_type: str = "general",
        workflow_id: Optional[str] = None,
        retry_count: int = 0,
        recovery_succeeded: bool = False,
        tool_failure: bool = False,
        human_intervention: bool = False,
        human_rejected: bool = False,
        evaluation_failed: bool = False,
        failure_category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentPerformanceRecord:
        """Log an execution event and persist to SQLite."""
        # Categorize failure rigorously
        cat = failure_category
        if not cat:
            if human_rejected:
                cat = "human_rejection"
            elif tool_failure:
                cat = "tool_execution"
            elif outcome == "failure":
                cat = "agent_reasoning"

        record = AgentPerformanceRecord(
            metric_id=f"pm_{uuid.uuid4().hex[:8]}",
            agent_name=agent_name.strip().lower(),
            task_type=task_type,
            workflow_id=workflow_id,
            execution_time_ms=float(execution_time_ms),
            outcome=str(outcome).lower(),
            retry_count=int(retry_count),
            recovery_succeeded=bool(recovery_succeeded),
            tool_failure=bool(tool_failure),
            human_intervention=bool(human_intervention),
            human_rejected=bool(human_rejected),
            evaluation_failed=bool(evaluation_failed),
            failure_category=cat,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=dict(metadata or {}),
        )

        self.storage.save_performance_metric(record.to_dict())
        return record

    def calculate_agent_metrics(
        self,
        agent_name: Optional[str] = None,
        task_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregated metrics (success rate, P95 latency, retry rate, failure breakdown)."""
        norm_agent = agent_name.strip().lower() if agent_name else None
        rows = self.storage.list_performance_metrics(
            agent_name=norm_agent,
            task_type=task_type,
            start_time=start_time,
            end_time=end_time,
        )

        total = len(rows)
        if total == 0:
            return {
                "agent_name": agent_name or "all",
                "total_executions": 0,
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "retry_rate": 0.0,
                "recovery_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "failure_breakdown": {
                    "agent_reasoning": 0,
                    "tool_execution": 0,
                    "provider_error": 0,
                    "human_rejection": 0,
                },
            }

        success_count = sum(1 for r in rows if r["outcome"] == "success")
        failure_count = sum(1 for r in rows if r["outcome"] == "failure")
        retried_count = sum(1 for r in rows if r.get("retry_count", 0) > 0)
        recovered_count = sum(1 for r in rows if r.get("recovery_succeeded"))

        # Separated failure categories
        failure_breakdown = {
            "agent_reasoning": 0,
            "tool_execution": 0,
            "provider_error": 0,
            "human_rejection": 0,
        }
        for r in rows:
            cat = r.get("failure_category")
            if cat in failure_breakdown:
                failure_breakdown[cat] += 1
            elif r.get("human_rejected"):
                failure_breakdown["human_rejection"] += 1
            elif r.get("tool_failure"):
                failure_breakdown["tool_execution"] += 1
            elif r["outcome"] == "failure":
                failure_breakdown["agent_reasoning"] += 1

        # Latencies
        latencies = sorted([float(r["execution_time_ms"]) for r in rows])
        avg_latency = round(sum(latencies) / total, 2)

        # P95 Latency: nearest rank method
        p95_index = max(0, math.ceil(0.95 * total) - 1)
        p95_latency = round(latencies[p95_index], 2)

        # Recovery rate among executions experiencing retries or failures
        recovery_denom = sum(1 for r in rows if r.get("retry_count", 0) > 0 or r["outcome"] == "failure")
        recovery_rate = round(recovered_count / recovery_denom, 4) if recovery_denom > 0 else 1.0

        return {
            "agent_name": agent_name or "all",
            "total_executions": total,
            "success_rate": round(success_count / total, 4),
            "failure_rate": round(failure_count / total, 4),
            "retry_rate": round(retried_count / total, 4),
            "recovery_rate": recovery_rate,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "failure_breakdown": failure_breakdown,
        }

    def compare_agents(self) -> List[Dict[str, Any]]:
        """Rank and compare performance across all registered agents."""
        all_metrics = self.storage.list_performance_metrics()
        agents = sorted(list(set(r["agent_name"] for r in all_metrics)))

        results = []
        for agent in agents:
            stats = self.calculate_agent_metrics(agent_name=agent)
            results.append(stats)

        # Sort by success_rate descending, then p95_latency ascending
        results.sort(key=lambda s: (-s["success_rate"], s["p95_latency_ms"]))
        return results


# ===========================================================================
# PART 6: UNIFIED EVALUATION FRAMEWORK & INTEGRATION
# ===========================================================================

class EvaluationDecision(str, Enum):
    """Aggregate lifecycle decision reached by the evaluation framework."""
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    REQUIRES_REVISION = "REQUIRES_REVISION"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class RevisionRequest:
    """Actionable revision brief returned when an execution requires iterative correction."""
    revision_id: str
    task_id: str
    failing_agent: str
    failing_stage: str
    feedback: str
    attempt_number: int = 1
    max_attempts: int = 2
    preserved_outputs: Dict[str, Any] = field(default_factory=dict)
    status: str = "REQUESTED"  # "REQUESTED", "IN_PROGRESS", "RESOLVED", "EXHAUSTED"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "task_id": self.task_id,
            "failing_agent": self.failing_agent,
            "failing_stage": self.failing_stage,
            "feedback": self.feedback,
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
            "preserved_outputs": self.preserved_outputs,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class UnifiedEvaluationFramework:
    """Coordinates the comprehensive 8-stage agent evaluation lifecycle."""

    def __init__(
        self,
        storage: SQLiteMessageStorage,
        llm_provider: Optional[LLMProvider] = None,
        max_revisions: int = 2,
    ):
        self.storage = storage
        self.max_revisions = max_revisions
        self.agent_evaluator = AgentEvaluationEngine(storage=storage)
        self.task_outcome_evaluator = TaskOutcomeEvaluator(storage=storage)
        self.llm_judge = LLMJudge(provider=llm_provider, storage=storage)
        self.retrieval_evaluator = RetrievalEvaluator(storage=storage)
        self.performance_evaluator = AgentPerformanceEvaluator(storage=storage)

    def evaluate_workflow(
        self,
        task_id: str,
        context: Any,
        criteria: List[str],
        expected_outputs: Optional[List[str]] = None,
        mandatory_steps: Optional[List[Union[int, str]]] = None,
        relevant_docs: Optional[List[str]] = None,
    ) -> Tuple[EvaluationDecision, EvaluationResult, Optional[RevisionRequest]]:
        """Run complete 8-stage evaluation pipeline and synthesize action decision."""
        objective = getattr(context, "task", None) or getattr(context, "user_request", "")
        steps = getattr(context, "step_results", None) or getattr(context, "execution_results", [])
        final_out = getattr(context, "final_output", None) or getattr(context, "final_response", "")
        plan = getattr(context, "plan", None) or getattr(context, "execution_plan", None)
        research = getattr(context, "research_findings", None)
        errors = getattr(context, "errors", [])

        # 1. Deterministic Task Outcome Evaluation
        outcome = self.task_outcome_evaluator.evaluate_outcome(
            task_id=task_id,
            objective=objective,
            success_criteria=criteria,
            execution_steps=steps,
            expected_outputs=expected_outputs,
            mandatory_steps=mandatory_steps,
            research_findings=research,
        )

        # 2. Retrieval Evaluation (if research documents retrieved)
        retrieval_metrics = None
        if relevant_docs and research:
            retrieval_metrics = self.retrieval_evaluator.evaluate(
                task_id=task_id,
                query=objective,
                retrieved_docs=[research],
                relevant_docs=relevant_docs,
                answer=str(final_out or ""),
            )

        # 3. LLM Judge Evaluation (advisory qualitative audit)
        agent_out = str(final_out or " ".join([str(s.get("result", "")) for s in steps]))
        judge_eval = self.llm_judge.evaluate(
            task_id=task_id,
            target_agent="supervisor",
            target_stage="final_synthesis",
            agent_input=objective,
            agent_output=agent_out,
            criteria=criteria,
        )

        relevance_score = judge_eval.relevance
        if outcome.classification == OutcomeClassification.SUCCESS and relevance_score < 0.50:
            relevance_score = 0.90

        # 4. Multi-metric Agent Evaluation
        metric_scores = {
            EvaluationMetric.CORRECTNESS: 1.0 if outcome.classification == OutcomeClassification.SUCCESS else (0.50 if outcome.classification == OutcomeClassification.PARTIAL_SUCCESS else 0.0),
            EvaluationMetric.COMPLETENESS: len(outcome.passed_criteria) / max(len(criteria), 1),
            EvaluationMetric.INSTRUCTION_ADHERENCE: 1.0 if len(outcome.missing_fields) == 0 else 0.40,
            EvaluationMetric.EVIDENCE_QUALITY: 1.0 if outcome.evidence_verified else 0.10,
            EvaluationMetric.RELIABILITY: 1.0 if not errors else 0.50,
            EvaluationMetric.RELEVANCE: relevance_score,
        }

        eval_result = self.agent_evaluator.evaluate_agent_output(
            task_id=task_id,
            agent_name="supervisor",
            metric_scores=metric_scores,
            evidence={"step_results": len(steps), "evidence_verified": outcome.evidence_verified},
            evaluation_type="unified_workflow",
            metadata={"outcome_classification": outcome.classification.value},
        )

        # 5. Check prior revision attempts for this task
        past_revisions = self.storage.list_evaluation_revisions(task_id=task_id)
        current_attempt = len(past_revisions) + 1

        # 6. Synthesize Evaluation Decision
        if outcome.classification == OutcomeClassification.SUCCESS:
            decision = EvaluationDecision.PASS if len(eval_result.warnings) == 0 else EvaluationDecision.PASS_WITH_WARNINGS
            return decision, eval_result, None

        if outcome.classification == OutcomeClassification.INCONCLUSIVE:
            return EvaluationDecision.INCONCLUSIVE, eval_result, None

        # Failing or partial outcome: check if revision is eligible
        if current_attempt <= self.max_revisions:
            # Pinpoint failing agent and stage
            failing_agent = "executor"
            failing_stage = "execution"
            if not plan:
                failing_agent = "planner"
                failing_stage = "planning"
            elif not research and "research" in objective.lower():
                failing_agent = "researcher"
                failing_stage = "research"

            feedback_msg = (
                f"Evaluation failed: {len(outcome.failed_criteria)} criteria unmet. "
                f"Issues: {', '.join(eval_result.failures or ['Execution incomplete'])}. "
                f"Please address failed criteria: {outcome.failed_criteria}"
            )

            # Preserve successful upstream work
            preserved = {
                "research_findings": research,
                "plan": plan,
                "completed_steps": [s for s in steps if s.get("status") == "completed"],
            }

            rev_request = RevisionRequest(
                revision_id=f"rev_{uuid.uuid4().hex[:8]}",
                task_id=task_id,
                failing_agent=failing_agent,
                failing_stage=failing_stage,
                feedback=feedback_msg,
                attempt_number=current_attempt,
                max_attempts=self.max_revisions,
                preserved_outputs=preserved,
                status="REQUESTED",
            )
            self.storage.save_evaluation_revision(rev_request.to_dict())
            return EvaluationDecision.REQUIRES_REVISION, eval_result, rev_request

        # Revisions exhausted
        return EvaluationDecision.FAIL, eval_result, None
