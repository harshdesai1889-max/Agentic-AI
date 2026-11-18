"""
core/evaluator.py

Result Evaluation and Worker Verification Engine for Day 225 Part 4.

Provides deterministic validation of multi-agent execution results:
1. Completion Verification: Confirms whether all required steps actually executed.
2. Required Output Fields: Checks presence of essential output schema fields.
3. Task Requirement & Criteria Matching: Deterministic numerical, quoted token, and constraint audits.
4. Actual Evidence Verification: Ensures outputs contain factual evidence rather than vacuous self-claims.
5. Step Failure Detection: Flags any failed execution step and prevents false completion.
6. Completion Eligibility: Evaluates whether final results qualify for COMPLETED status.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

from utils.logging_config import get_logger

logger = get_logger()

# Regex patterns matching vacuous self-assessment claims without factual evidence
VACUOUS_CLAIM_PATTERNS = [
    r"^\s*(i\s+(have\s+)?(completed|finished|done|executed)\s*(the\s+task|this)?(\s+successfully)?)\s*[.!]?\s*$",
    r"^\s*(task\s+(is\s+)?completed(\s+successfully)?)\s*[.!]?\s*$",
    r"^\s*(done|success|ok|finished|completed)\s*[.!]?\s*$",
    r"^\s*(all\s+tasks?\s+(are\s+)?(done|completed))\s*[.!]?\s*$",
    r"^\s*(i\s+have\s+done\s+this(\s+task)?)\s*[.!]?\s*$",
    r"^\s*(execution\s+(is\s+)?complete(\s+successfully)?)\s*[.!]?\s*$",
]

# Patterns indicating error or unhandled exception output
ERROR_PATTERNS = [
    r"^agent execution error:",
    r"^traceback \(most recent call last\):",
    r"^[a-zA-Z0-9_]+error:\s+",
    r"^execution failed:\s+",
]


def is_vacuous_claim(text: str) -> bool:
    """Check if the text is merely an unsubstantiated self-claim of task completion."""
    if not text or not isinstance(text, str):
        return True
    cleaned = text.strip()
    if not cleaned:
        return True
    for pattern in VACUOUS_CLAIM_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return True
    return False


def is_error_output(text: str) -> bool:
    """Check if the text represents an error message or exception traceback."""
    if not text or not isinstance(text, str):
        return False
    cleaned = text.strip()
    for pattern in ERROR_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return True
    return False


def has_substantive_evidence(result: Any) -> bool:
    """Determine whether the step result contains substantive evidence.

    Rejects:
    - Empty, None, or whitespace-only results.
    - Error messages and unhandled exceptions.
    - Vacuous claims of completion (e.g. 'Task completed successfully.').

    Accepts:
    - Factual strings with concrete data, numbers, calculations, or structured text.
    - Dictionaries or lists containing substantive records.
    """
    if result is None:
        return False

    if isinstance(result, (dict, list)):
        return len(result) > 0

    if not isinstance(result, str):
        result = str(result)

    cleaned = result.strip()
    if not cleaned:
        return False

    if is_error_output(cleaned):
        return False

    if is_vacuous_claim(cleaned):
        return False

    # Check for minimal content length and alphanumeric characters
    return len(cleaned) >= 2 and any(c.isalnum() for c in cleaned)


# ===========================================================================
# Evaluation Result Dataclass
# ===========================================================================

@dataclass
class EvaluationResult:
    """Comprehensive evaluation report of task execution against requirements."""
    # Day 225 fields (keep position for backward compatibility)
    is_eligible_for_completion: bool = True
    all_criteria_met: bool = True
    has_valid_evidence: bool = True
    passed_criteria: List[str] = field(default_factory=list)
    failed_criteria: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    completed_steps: List[Dict[str, Any]] = field(default_factory=list)
    failed_steps: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_notes: str = ""
    status: str = "completed"  # "completed" | "partially_completed" | "failed" | "rejected"

    # Day 228 fields
    evaluation_id: str = field(default_factory=lambda: f"eval_{uuid.uuid4().hex[:8]}")
    task_id: str = ""
    agent_name: str = ""
    evaluation_type: str = "general"
    overall_score: float = 0.0
    metric_scores: Dict[str, float] = field(default_factory=dict)
    passed: bool = True
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    evaluation_reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Synchronize backward compatible status flags
        if not self.passed or not self.is_eligible_for_completion or self.status == "failed":
            if not self.is_eligible_for_completion:
                self.passed = False
            if not self.passed:
                self.is_eligible_for_completion = False
        if not self.overall_score and self.passed_criteria:
            total = len(self.passed_criteria) + len(self.failed_criteria)
            if total > 0:
                self.overall_score = round(len(self.passed_criteria) / total, 4)
        elif self.overall_score > 0 and not self.passed_criteria:
            self.passed = self.overall_score >= 0.70

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to a serializable dictionary."""
        return {
            "is_eligible_for_completion": self.is_eligible_for_completion,
            "all_criteria_met": self.all_criteria_met,
            "has_valid_evidence": self.has_valid_evidence,
            "passed_criteria": self.passed_criteria,
            "failed_criteria": self.failed_criteria,
            "missing_fields": self.missing_fields,
            "completed_steps_count": len(self.completed_steps),
            "failed_steps_count": len(self.failed_steps),
            "evaluation_notes": self.evaluation_notes,
            "status": self.status,
            # Day 228 keys
            "evaluation_id": self.evaluation_id,
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "evaluation_type": self.evaluation_type,
            "overall_score": self.overall_score,
            "metric_scores": self.metric_scores,
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "evidence": self.evidence,
            "evaluation_reason": self.evaluation_reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ===========================================================================
# Result Evaluator Class
# ===========================================================================

class ResultEvaluator:
    """Evaluates multi-agent execution results deterministically.

    Enforces the 6 core checks required by Day 225 Part 4:
    1. Completion: checks if planned steps were executed.
    2. Required Fields: verifies presence of required output schema fields.
    3. Requirement Satisfaction: audits numerical, quoted, and semantic criteria.
    4. Evidence Verification: rejects vacuous self-claims; verifies factual evidence.
    5. Step Failure Audit: catches any step with status != 'completed' or error set.
    6. Completion Eligibility: computes final task eligibility (COMPLETED / PARTIAL / FAILED).
    """

    REQUIRED_STEP_FIELDS: Set[str] = {"task_id", "status", "result"}

    def evaluate_step(
        self,
        step: Dict[str, Any],
        expected_task_id: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """Audit an individual execution step result.

        Returns a dictionary with validation status, missing fields,
        and evidence assessment.
        """
        issues: List[str] = []
        missing: List[str] = []

        if not isinstance(step, dict):
            return {
                "valid": False,
                "missing_fields": ["step_dict"],
                "has_evidence": False,
                "issues": ["Step output is not a dictionary"],
            }

        # Check required step fields
        for field_name in self.REQUIRED_STEP_FIELDS:
            if field_name not in step or step.get(field_name) is None:
                missing.append(field_name)
                issues.append(f"Missing required step field: '{field_name}'")

        # Check task ID consistency if expected
        if expected_task_id is not None:
            actual_id = step.get("task_id")
            if str(actual_id) != str(expected_task_id):
                issues.append(
                    f"Task ID mismatch: expected '{expected_task_id}', got '{actual_id}'"
                )

        # Check step status
        status = step.get("status")
        if status != "completed":
            issues.append(f"Step status is '{status}', not 'completed'")

        # Check error field
        if step.get("error"):
            issues.append(f"Step contains error: {step.get('error')}")

        # Check evidence
        evidence_ok = has_substantive_evidence(step.get("result"))
        if not evidence_ok:
            if is_vacuous_claim(str(step.get("result", ""))):
                issues.append("Result contains only vacuous self-claim without factual evidence")
            else:
                issues.append("Result lacks substantive evidence")

        is_valid = (
            len(missing) == 0
            and len(issues) == 0
            and status == "completed"
            and evidence_ok
            and not step.get("error")
        )

        return {
            "valid": is_valid,
            "missing_fields": missing,
            "has_evidence": evidence_ok,
            "issues": issues,
        }

    def evaluate(
        self,
        task_id: str,
        objective: str,
        success_criteria: List[str],
        execution_steps: List[Dict[str, Any]],
        research_findings: Optional[str] = None,
        expected_output: Optional[str] = None,
        errors: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Perform a complete deterministic evaluation of task execution.

        Args:
            task_id: The unique task/workflow identifier.
            objective: The original task objective.
            success_criteria: List of explicit criteria strings.
            execution_steps: List of step result dictionaries.
            research_findings: Optional research summary string.
            expected_output: Optional specification of expected output fields.
            errors: Optional list of recorded errors.

        Returns:
            EvaluationResult detailing the audit findings.
        """
        all_errors = list(errors or [])
        completed_steps: List[Dict[str, Any]] = []
        failed_steps: List[Dict[str, Any]] = []
        missing_fields: List[str] = []
        passed_criteria: List[str] = []
        failed_criteria: List[str] = []

        # -------------------------------------------------------------------
        # Check 1 & 5: Evaluate each execution step
        # -------------------------------------------------------------------
        step_evidence_statuses: List[bool] = []

        for idx, step in enumerate(execution_steps, start=1):
            expected_id = step.get("id") or step.get("task_id") or idx
            step_eval = self.evaluate_step(step, expected_task_id=expected_id)

            if step_eval["missing_fields"]:
                for mf in step_eval["missing_fields"]:
                    missing_fields.append(f"step_{idx}.{mf}")

            step_evidence_statuses.append(step_eval["has_evidence"])

            if step_eval["valid"]:
                completed_steps.append(step)
            else:
                failed_steps.append(step)
                for issue in step_eval["issues"]:
                    if issue not in all_errors:
                        all_errors.append(f"Step {idx}: {issue}")

        # Check if any step was executed at all
        if not execution_steps:
            missing_fields.append("execution_steps")
            all_errors.append("No execution steps were performed.")

        # -------------------------------------------------------------------
        # Check 2: Verify Expected Output Requirements
        # -------------------------------------------------------------------
        if expected_output and isinstance(expected_output, str) and expected_output.strip():
            # Check if expected output specifies required tokens or fields
            expected_tokens = [
                tok.strip().lower() for tok in re.split(r"[,;|\n]", expected_output) if tok.strip()
            ]
            all_output_text = " ".join([
                str(s.get("result", "")) for s in completed_steps
            ]) + " " + str(research_findings or "")

            for token in expected_tokens:
                if token and token not in all_output_text.lower():
                    missing_fields.append(f"expected_output:{token}")

        # -------------------------------------------------------------------
        # Check 4: Evidence Assessment
        # -------------------------------------------------------------------
        has_evidence = False
        if completed_steps:
            has_evidence = all(has_substantive_evidence(s.get("result")) for s in completed_steps)

        # -------------------------------------------------------------------
        # Check 3: Deterministic Success Criteria Matching
        # -------------------------------------------------------------------
        valid_completed_text = " ".join([
            str(s.get("result", "")) for s in completed_steps
        ]) + " " + str(research_findings or "")

        for criterion in success_criteria:
            criterion_clean = criterion.strip().lower()

            # Rule 3A: Step completion / error-free criteria
            if any(phrase in criterion_clean for phrase in ["complete all", "all tasks", "without errors", "all steps"]):
                if failed_steps or not completed_steps or all_errors:
                    failed_criteria.append(criterion)
                    continue

            # Rule 3B: Deterministic Numerical Check
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", criterion)
            if numbers:
                all_numbers_found = all(num in valid_completed_text for num in numbers)
                if all_numbers_found and not failed_steps and completed_steps:
                    passed_criteria.append(criterion)
                else:
                    failed_criteria.append(criterion)
                continue

            # Rule 3C: Deterministic Quoted String Check
            quotes = re.findall(r"['\"](.*?)['\"]", criterion)
            if quotes:
                all_quotes_found = all(q.lower() in valid_completed_text.lower() for q in quotes)
                if all_quotes_found and not failed_steps and completed_steps:
                    passed_criteria.append(criterion)
                else:
                    failed_criteria.append(criterion)
                continue

            # Rule 3D: Semantic / Keyword Matching
            words = [w for w in re.findall(r"\b\w+\b", criterion_clean) if len(w) > 3]
            matched_words = [w for w in words if w in valid_completed_text.lower()]

            if words and (len(matched_words) / len(words) >= 0.5) and not failed_steps and completed_steps:
                passed_criteria.append(criterion)
            elif not failed_steps and completed_steps and not all_errors:
                passed_criteria.append(criterion)
            else:
                failed_criteria.append(criterion)

        # -------------------------------------------------------------------
        # Check 6: Completion Eligibility
        # -------------------------------------------------------------------
        all_criteria_ok = (len(failed_criteria) == 0) and (len(passed_criteria) > 0)
        no_failed_steps = (len(failed_steps) == 0) and (len(completed_steps) > 0)
        no_missing_fields = (len(missing_fields) == 0)

        is_eligible = (
            all_criteria_ok
            and no_failed_steps
            and has_evidence
            and no_missing_fields
        )

        # Status determination
        if is_eligible:
            final_status = "completed"
        elif len(completed_steps) > 0 and has_evidence:
            final_status = "partially_completed"
        else:
            final_status = "failed"

        notes = (
            f"Evaluation: {len(passed_criteria)}/{len(success_criteria)} criteria met. "
            f"Completed steps: {len(completed_steps)}, Failed steps: {len(failed_steps)}, "
            f"Missing fields: {len(missing_fields)}, Evidence valid: {has_evidence}."
        )

        calc_score = round(len(passed_criteria) / max(len(success_criteria), 1), 4) if passed_criteria else (1.0 if is_eligible else 0.0)

        return EvaluationResult(
            is_eligible_for_completion=is_eligible,
            all_criteria_met=all_criteria_ok,
            has_valid_evidence=has_evidence,
            passed_criteria=passed_criteria,
            failed_criteria=failed_criteria,
            missing_fields=missing_fields,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            evaluation_notes=notes,
            status=final_status,
            task_id=task_id,
            evaluation_type="task_outcome",
            overall_score=calc_score,
            passed=is_eligible,
            failures=list(failed_criteria) + [f"step_failed: {s.get('task')}" for s in failed_steps],
            warnings=[f"missing: {mf}" for mf in missing_fields],
            evidence={"completed_steps": len(completed_steps), "failed_steps": len(failed_steps)},
            evaluation_reason=notes,
        )
