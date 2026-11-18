# Coding Standards — day223_agents

This document defines the Python coding standards for the `day223_agents`
project. All code in this repository — existing and new — must conform to these
standards. The `python-code-review` skill uses this document as its authority
when evaluating any module.

---

## 1. Python Readability

### 1.1 Docstrings

Every public module, class, and function **must** have a docstring.

**Format:** Google style.

```python
def create_plan(objective: str, model: str = MODEL2) -> dict:
    """Generate a structured task plan from a complex objective.

    Args:
        objective: The complex goal or request.
        model: Model identifier (defaults to MODEL2).

    Returns:
        A dict with a validated 'tasks' list, or empty tasks and an
        error message on failure.
    """
```

- Module-level docstrings describe the module's purpose, not its contents.
- One-line docstrings are acceptable for trivial getters.
- Docstrings must stay accurate when the function signature changes.

### 1.2 Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Module | `snake_case` | `openrouter_provider.py` |
| Class | `PascalCase` | `OpenRouterProvider` |
| Function / method | `snake_case` | `generate()`, `execute_tool()` |
| Constant | `UPPER_SNAKE_CASE` | `MODEL1`, `MAX_TOOL_ROUNDS` |
| Private helper | `_leading_underscore` | `_evaluate_node()` |
| Variable | `snake_case`, descriptive | `raw_arguments`, `tool_call_id` |

Avoid single-letter variables except in short comprehensions or standard
mathematical notation (`i`, `x`, `n` are acceptable).

### 1.3 Line Length

Maximum **100 characters** per line. Use implicit string concatenation or
backslash continuation for longer strings, not raw truncation.

```python
# Good
description = (
    "Evaluate a mathematical expression using a secure AST-based evaluator. "
    "Supports +, -, *, /, ** (power), and parentheses."
)

# Bad
description = "Evaluate a mathematical expression using a secure AST-based evaluator. Supports +, -, *, /, ** (power), and parentheses."
```

### 1.4 Blank Lines

- Two blank lines between top-level definitions (classes, functions).
- One blank line between methods inside a class.
- Use blank lines within functions to separate logical phases.

### 1.5 Comments

- Comments explain **why**, not **what** (the code shows what).
- Section headers use `# ---` dividers for large modules.
- Never commit commented-out code — use version control instead.

---

## 2. Error Handling

### 2.1 Wrap All External Calls

Any call that can raise an exception due to network, I/O, or external state
**must** be wrapped in `try/except`.

```python
# Good
try:
    response = client.chat.completions.create(...)
except Exception as exc:
    logger.error(f"LLM call failed: {str(exc)}")
    return {"error": f"LLM call failed: {str(exc)}"}

# Bad — unhandled exception propagates unchecked
response = client.chat.completions.create(...)
```

### 2.2 Catch at the Right Granularity

- Catch the **most specific exception** you can handle.
- Use broad `except Exception` only as a last-resort safety net at module
  boundaries (agent engine, executor entry points).
- **Never** use bare `except:` — it catches `SystemExit` and `KeyboardInterrupt`.

```python
# Good
except (ValueError, SyntaxError) as exc:
    return {"error": f"Invalid expression: {str(exc)}"}
except ZeroDivisionError:
    return {"error": "Division by zero"}
except Exception:
    return {"error": "Calculation failed"}

# Bad
except:
    pass
```

### 2.3 Consistent Error Return Format

Tools must return a dict with an `"error"` key on failure:

```python
{"error": "Division by zero"}
```

Providers must set `LLMResponse.error` and return a valid `LLMResponse` —
never raise exceptions to callers.

Agent functions that cannot recover must return a human-readable string:

```python
"Agent execution error: Provider returned None response."
```

### 2.4 Log Before Returning Errors

Use the project logger before returning error values:

```python
logger.error(f"Planner execution failed: {str(exc)}")
return {"tasks": [], "error": f"Planner failed: {str(exc)}"}
```

---

## 3. Security

### 3.1 No Hardcoded Credentials

API keys, tokens, and secrets **must never** appear in source code.

```python
# Good — read through the central config
from config.settings import OPENROUTER_API_KEY

# Bad — hardcoded secret
client = OpenAI(api_key="sk-...")
```

### 3.2 Centralised Environment Variable Access

All environment variables are read exclusively in `config/settings.py`.
No other module may call `os.getenv()` for credentials.

### 3.3 No Dynamic Code Execution

Never use `eval()`, `exec()`, `compile()` with user input, or
`__import__()` with user-controlled strings.

The `tools/basic_tools.py` calculator uses a whitelist AST evaluator — this
pattern is the **approved** exception and must be maintained exactly.

Approved operators (do not expand without security review):
- Binary: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Unary: `+`, `-`
- Constants: numeric literals only

### 3.4 Safe Path Handling

Never construct file paths from user input using string concatenation. Use
`pathlib.Path` or `os.path.join` and validate that the result stays within
the expected root.

### 3.5 Input Validation

Validate types and boundaries before processing:

```python
if not isinstance(expression, str) or not expression.strip():
    return {"error": "Invalid mathematical expression"}

if len(expression) > MAX_EXPRESSION_LENGTH:
    return {"error": f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)"}
```

---

## 4. Type Hints

### 4.1 All Public Signatures Must Be Annotated

```python
# Good
def execute_tool(tool_call: Any) -> Dict[str, Any]:

# Bad
def execute_tool(tool_call):
```

### 4.2 Use `typing` Imports for Complex Types

```python
from typing import Any, Dict, List, Optional, Union
```

Prefer built-in generics (`list[str]`, `dict[str, Any]`) in Python 3.9+.
This project targets Python 3.11; both styles are acceptable, but be consistent
within a module.

### 4.3 `Optional` for Nullable Parameters

```python
def run_agent(
    user_task: str,
    model: str = MODEL1,
    provider: Optional[LLMProvider] = None,
) -> str:
```

### 4.4 Dataclass Fields

All `@dataclass` fields must have type annotations:

```python
@dataclass
class LLMMessage:
    role: str
    content: Optional[str] = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
```

### 4.5 Avoid Overuse of `Any`

Use `Any` only when the type is genuinely unknown at the interface boundary
(e.g. `raw_response: Any` in `LLMResponse`). Never use `Any` to avoid thinking
about the type.

---

## 5. Code Organisation

### 5.1 Single Responsibility

Each module has one clearly stated purpose. Each function does one thing.

| Module | Single responsibility |
|--------|----------------------|
| `config/settings.py` | Load and validate environment variables |
| `core/provider_interface.py` | Define the provider abstraction only |
| `core/agent_engine.py` | Orchestrate the agent loop only |
| `tools/basic_tools.py` | Implement tool business logic only |
| `tools/executor.py` | Dispatch and execute tool calls only |
| `mcp_servers/basic_tools_server.py` | Define MCP tool wrappers only |

### 5.2 Import Order (PEP 8)

```python
# 1. Standard library
import json
import os
from typing import Any, Optional

# 2. Third-party
from openai import OpenAI

# 3. Local / project
from config.settings import OPENROUTER_API_KEY
from core.provider_interface import LLMProvider, LLMResponse
```

Blank line between each group.

### 5.3 Constants at Module Level

```python
MAX_TOOL_ROUNDS = 5
MAX_EXPRESSION_LENGTH = 200
MODEL1 = "nvidia/nemotron-3.5-lightning:free"
MODEL2 = "nvidia/nemotron-3-ultra-550b-a55b:free"
```

### 5.4 Private Helpers

Name internal helpers with a leading underscore. They must not appear in
`__all__` and should not be imported by other modules.

```python
def _evaluate_node(node: ast.AST) -> Union[int, float]:
    ...
```

### 5.5 `__init__.py` Files

Package `__init__.py` files must be minimal:
- May re-export key public symbols for convenience.
- Must not contain business logic.
- May be empty.

### 5.6 No Circular Imports

The dependency graph must be a DAG:

```
config/ → (nothing internal)
core/provider_interface → (nothing internal)
core/llm → config/
providers/ → core/provider_interface, config/
tools/ → (nothing internal)
core/agent_engine → providers/, tools/, config/, utils/
agents/ → core/, config/, utils/
mcp_servers/ → tools/
```

---

## 6. Testing Standards

### 6.1 Test File Location and Naming

- All tests live in `tests/`.
- Test files are named `test_<module>.py`.
- Test functions follow `test_<function>_<condition>()`.

```
tests/test_basic_tools.py         ← tests for tools/basic_tools.py
tests/test_tool_executor.py       ← tests for tools/executor.py
tests/test_openrouter_provider.py ← tests for providers/openrouter_provider.py
```

### 6.2 No Real API Calls in Tests

All tests are offline. Use `unittest.mock.MagicMock` or `unittest.mock.patch`
to substitute external dependencies.

```python
from unittest.mock import MagicMock, patch

def test_provider_calls_api(mock_client):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello"
    mock_client.chat.completions.create.return_value = mock_response
    ...
```

### 6.3 Coverage Requirements

Every public function needs tests for:

| Case | Example |
|------|---------|
| Happy path | Valid input returns expected output |
| Empty / `None` input | Function handles gracefully |
| Type error | Wrong type returns error, not exception |
| Boundary value | Maximum length, zero, etc. |
| External failure | Mocked API throws exception |

### 6.4 Test Independence

Each test must be fully independent:
- No shared mutable state between tests.
- Use `pytest` fixtures for repeated setup.
- Tests must pass in any order.

### 6.5 Assertions

Prefer specific assertions:

```python
# Good
assert result == {"result": 4}
assert "error" in result
assert result["word_count"] == 3

# Avoid
assert result  # too vague
```

### 6.6 Running Tests

```bash
# Full suite
python -m pytest -q

# Single file
python -m pytest tests/test_basic_tools.py -v

# With coverage (if pytest-cov is installed)
python -m pytest --cov=. --cov-report=term-missing -q
```

The full suite must pass with **0 failures** before any change is considered
complete.

---

## Appendix: Quick Checklist

Use before submitting any module for review:

- [ ] Module docstring present
- [ ] All public functions have Google-style docstrings
- [ ] All public function signatures have type hints
- [ ] No hardcoded credentials or secrets
- [ ] All external calls wrapped in `try/except`
- [ ] Error returns use consistent format
- [ ] Imports ordered: stdlib → third-party → local
- [ ] Constants at module level in `UPPER_SNAKE_CASE`
- [ ] Private helpers prefixed with `_`
- [ ] At least one test per public function
- [ ] No real API calls in tests (all mocked)
- [ ] `python -m pytest -q` passes with 0 failures
