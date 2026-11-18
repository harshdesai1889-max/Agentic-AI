---
name: python-code-review
description: >-
  Performs a structured Python code review for the day223_agents project.
  Checks readability, error handling, security, type hints, code organisation,
  and testing standards against the project's established conventions.
  Use when the user asks for a code review, quality check, or audit of any
  Python file or module in this codebase.
---

# Python Code Review Skill

This skill guides a thorough, structured review of Python source files in the
`day223_agents` project. It enforces the project's established conventions and
produces an actionable, prioritised report.

---

## When to Use This Skill

Activate this skill whenever:

- The user asks for a "code review", "quality check", "audit", or "feedback" on
  a Python file or module.
- A new module is added (e.g. a new provider, tool, agent, or MCP server).
- A refactoring is proposed and its quality must be validated.
- A pull request or commit needs quality gates applied before merging.
- The user asks "is this code ready?" or "what can I improve?".

Do **not** use this skill to modify code. It is a read-and-report skill only.

---

## Reference Documents

Before executing the review, read:

- [`references/coding_standards.md`](references/coding_standards.md) — full
  project coding standards (readability, error handling, security, type hints,
  organisation, testing).

---

## Execution Workflow

Follow these steps in order. Do not skip any step.

### Step 1 — Identify the Target

1. Confirm the file(s) to review with the user if ambiguous.
2. Read the full content of each target file.
3. Note the module's role in the architecture:
   - `core/` — provider abstraction, agent engine, model routing
   - `providers/` — LLM provider adapters (OpenRouter, Gemini)
   - `tools/` — tool implementations, schemas, registry, executor
   - `agents/` — planner, executor, researcher orchestration
   - `mcp_servers/` — MCP server definitions
   - `config/` — settings and model configuration
   - `utils/` — shared utilities (logging, JSON, prompt loading)
   - `tests/` — pytest test modules

### Step 2 — Read the Coding Standards

Read `references/coding_standards.md` in full before scoring anything.

### Step 3 — Apply Each Review Dimension

For each dimension below, examine the code and record:
- **Findings** — specific lines or patterns that pass or fail.
- **Severity** — `critical`, `major`, `minor`, or `suggestion`.
- **Recommendation** — the exact change required, with example if helpful.

#### Dimension 1: Readability
- Module-level docstring present and accurate.
- All public functions and classes have docstrings (Google style).
- Variable and function names are descriptive and follow `snake_case`.
- Lines do not exceed 100 characters.
- Logical sections are separated by blank lines or comment headers.
- No commented-out dead code.

#### Dimension 2: Error Handling
- All external calls (API, I/O, parsing) are wrapped in `try/except`.
- Exceptions are caught at the correct granularity — never bare `except:`.
- Error messages are specific and actionable.
- Errors are propagated or returned in a consistent format (e.g. `{"error": "..."}` for tools).
- No silent swallowing of exceptions (`except Exception: pass`).

#### Dimension 3: Security
- No hardcoded secrets, API keys, or tokens.
- All environment variables are read through `config/settings.py`.
- No use of `eval()`, `exec()`, or `__import__()` on untrusted input.
- File path operations use safe path handling (no traversal).
- User-supplied strings are never executed as code (verify AST evaluator in `tools/basic_tools.py` patterns if relevant).

#### Dimension 4: Type Hints
- All public function signatures have type annotations for parameters and return values.
- Complex types use `typing` module imports (`Optional`, `List`, `Dict`, `Any`, etc.).
- Dataclasses use typed fields.
- No `Any` used where a more specific type is known.

#### Dimension 5: Code Organisation
- Single Responsibility Principle: each function does one thing.
- No business logic in `__init__.py` files.
- No circular imports.
- Module imports are ordered: stdlib → third-party → local (PEP 8).
- Constants are defined at module level in UPPER_SNAKE_CASE.
- Helper / private functions are prefixed with `_`.

#### Dimension 6: Testing Standards
- Every public function has at least one corresponding test.
- Tests use `pytest` conventions (no `unittest.TestCase` unless necessary).
- Tests do not make real API calls — all external calls are mocked.
- Test names follow `test_<what>_<condition>` pattern.
- Edge cases are covered: empty input, `None`, type errors, boundary values.
- Fixtures are used for repeated setup rather than copy-paste.

### Step 4 — Score Each Dimension

Rate each dimension on a 1–5 scale:

| Score | Meaning |
|-------|---------|
| 5 | Excellent — meets all standards |
| 4 | Good — minor issues only |
| 3 | Acceptable — some gaps, non-blocking |
| 2 | Needs improvement — significant gaps |
| 1 | Critical — blocking issues present |

### Step 5 — Produce the Report

Output a structured report in the format below. Do not truncate findings.

---

## Expected Output Format

```
# Python Code Review: <filename>

**Reviewed by:** python-code-review skill  
**Date:** <ISO date>  
**Module role:** <e.g. "LLM provider adapter for OpenRouter">

---

## Summary

Overall score: <average>/5  
Critical issues: <count>  
Major issues: <count>  
Minor issues: <count>  
Suggestions: <count>

<One paragraph of overall assessment.>

---

## Dimension Scores

| Dimension        | Score | Notes                   |
|-----------------|-------|-------------------------|
| Readability      | X/5   | ...                     |
| Error Handling   | X/5   | ...                     |
| Security         | X/5   | ...                     |
| Type Hints       | X/5   | ...                     |
| Code Organisation| X/5   | ...                     |
| Testing Standards| X/5   | ...                     |

---

## Findings

### 🔴 Critical

<finding id="C1">
**File:** `<filename>` **Line(s):** <N>  
**Issue:** <what is wrong>  
**Recommendation:** <exact fix>
</finding>

### 🟠 Major

<finding id="M1">
...
</finding>

### 🟡 Minor

<finding id="m1">
...
</finding>

### 💡 Suggestions

<finding id="S1">
...
</finding>

---

## Action Items (Prioritised)

1. [C1] <short description>
2. [M1] <short description>
...

---

## What is Done Well

- <positive finding 1>
- <positive finding 2>
```

---

## Safety Requirements

- **Never modify** any source file during a review. This skill is read-only.
- **Never expose** `.env` contents, API keys, or secrets in the report.
- **Never fabricate** findings. Only report what is observed in the actual code.
- **Never run** the code as part of the review. Static analysis only.
- If a finding requires running tests, instruct the user to run
  `python -m pytest -q` themselves and report results.
- Do not modify `main1.py`, `.env`, or existing model IDs under any
  circumstances, even if they appear in scope.

---

## Integration Notes

This skill is **not yet integrated** into the agent engine or MCP server.
It is a standalone read-and-report skill intended for direct human-in-the-loop
use. Future integration options include:

- Exposing `python-code-review` as an MCP tool via `mcp_servers/`.
- Triggering the skill automatically on `git diff` before commits.
- Connecting it to the `AgentEngine` as a specialised review provider.
