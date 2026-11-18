---
name: task_planning
description: >-
  Provides structured methodology for decomposing complex user goals into atomic,
  dependency-ordered task items, risk classification, and execution sequencing for PlannerAgent.
version: 1.0.0
---

# Task Planning Skill

This skill guides the decomposition of multi-step, complex objectives into atomic,
executable, and verifiable subtasks with explicit dependencies and risk gating.

---

## When to Use This Skill

Activate this skill whenever:
- A user request involves more than a single atomic tool call.
- Tasks require sequential coordination across Researcher, Planner, and Executor agents.
- Sensitive or mutating actions must be isolated and flagged for Human-in-the-Loop approval.

---

## Execution Workflow

1. **Goal Decomposition**:
   - Break the overarching objective into distinct, numbered subtasks.
   - Ensure each subtask has a clear single responsibility and measurable completion criteria.

2. **Dependency Mapping**:
   - Identify sequential constraints (e.g., Task 2 depends on Task 1 output).
   - Flag parallelizable tasks to optimize multi-agent execution latency.

3. **Risk Classification & Gating**:
   - Classify each step: Low Risk (read-only/compute) vs. High Risk (file write, deletion, external API mutation).
   - Bind high-risk steps to mandatory HITL approval requirements.

4. **Structured Plan Generation**:
   - Produce a standardized plan containing:
     - `tasks`: list of `{id, description, assigned_agent, dependencies, risk_level, requires_approval}`
     - `success_criteria`: clear verification metrics for the final supervisor review.

---

## Input & Output Expectations

- **Input**: User task description, available worker capabilities, security constraints.
- **Output**: JSON or Markdown plan with dependency order and risk classifications.
