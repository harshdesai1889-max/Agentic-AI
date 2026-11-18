import ast
import operator
from datetime import datetime
from typing import Any, Dict, Union

# Maximum allowed expression length to prevent DoS
MAX_EXPRESSION_LENGTH = 200

# Maximum AST nodes to prevent overly complex/nested expressions
MAX_AST_NODES = 50

# Exponentiation safety boundaries
MAX_POWER_BASE = 10000
MAX_POWER_EXPONENT = 100
MAX_RESULT_MAGNITUDE = 1e100

# Allowed binary operators
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_SAFE_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST node in a restricted environment."""
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_UNARY_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _evaluate_node(node.operand)
        return _SAFE_UNARY_OPERATORS[op_type](operand)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        # Enforce safety constraints on exponentiation
        if op_type is ast.Pow:
            if abs(left) > MAX_POWER_BASE:
                raise ValueError(f"Base exceeds maximum allowed value ({MAX_POWER_BASE})")
            if abs(right) > MAX_POWER_EXPONENT:
                raise ValueError(f"Exponent exceeds maximum allowed value ({MAX_POWER_EXPONENT})")
            if left == 0 and right < 0:
                raise ZeroDivisionError("Zero cannot be raised to a negative power")

        # Check for division/modulo by zero
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division or modulo by zero")

        result = _SAFE_OPERATORS[op_type](left, right)

        if isinstance(result, (int, float)) and abs(result) > MAX_RESULT_MAGNITUDE:
            raise OverflowError("Result exceeds maximum allowed magnitude")

        return result

    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def calculator(expression: str) -> Dict[str, Any]:
    """Calculate mathematical expressions using a secure AST evaluator.

    Args:
        expression: A mathematical expression string (e.g. '2 + 2', '15 * (3 + 4)').

    Returns:
        Dict with 'result' on success, or 'error' on failure.
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "Invalid mathematical expression"}

    expression = expression.strip()

    if len(expression) > MAX_EXPRESSION_LENGTH:
        return {"error": f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)"}

    try:
        parsed = ast.parse(expression, mode="eval")

        # Count total nodes to enforce complexity limits
        node_count = sum(1 for _ in ast.walk(parsed))
        if node_count > MAX_AST_NODES:
            return {"error": "Expression too complex"}

        computed = _evaluate_node(parsed)

        # Format integer results cleanly if equivalent
        if isinstance(computed, float) and computed.is_integer() and abs(computed) < 1e15:
            computed = int(computed)

        return {"result": computed}

    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except OverflowError:
        return {"error": "Numeric overflow in calculation"}
    except (ValueError, SyntaxError) as exc:
        return {"error": f"Invalid mathematical expression: {str(exc)}"}
    except Exception:
        return {"error": "Calculation failed"}


def get_datetime() -> Dict[str, str]:
    """Get the current date and time in ISO 8601 format."""
    return {
        "datetime": datetime.now().isoformat()
    }


def word_counter(text: str) -> Dict[str, Any]:
    """Count words and characters in the provided text.

    Args:
        text: The text to analyse.

    Returns:
        Dict containing 'word_count' and 'character_count'.
    """
    if not isinstance(text, str):
        return {"error": "Input must be a string"}

    words = text.split()
    return {
        "word_count": len(words),
        "character_count": len(text),
    }
