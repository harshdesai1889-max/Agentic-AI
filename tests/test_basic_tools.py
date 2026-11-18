from datetime import datetime
from tools.basic_tools import calculator, get_datetime, word_counter


def test_calculator_addition():
    """Test standard addition and decimal operations."""
    res = calculator("2 + 2")
    assert res == {"result": 4}

    res2 = calculator("10.5 + 4.5")
    assert res2 == {"result": 15}


def test_calculator_multiplication():
    """Test multiplication and parenthesized arithmetic."""
    res = calculator("6 * 7")
    assert res == {"result": 42}

    res_complex = calculator("3 * (4 + 5)")
    assert res_complex == {"result": 27}


def test_calculator_division_and_power():
    """Test division, modulo, and bounded exponentiation."""
    res_div = calculator("20 / 4")
    assert res_div == {"result": 5}

    res_pow = calculator("2 ** 8")
    assert res_pow == {"result": 256}

    res_zero_div = calculator("10 / 0")
    assert "error" in res_zero_div
    assert res_zero_div["error"] == "Division by zero"


def test_calculator_invalid_expressions():
    """Test syntax errors, empty input, and malformed strings."""
    assert "error" in calculator("")
    assert "error" in calculator("   ")
    assert "error" in calculator("2 + * 3")
    assert "error" in calculator("2 +")
    assert "error" in calculator("abc + 123")


def test_calculator_malicious_expression_rejection():
    """Verify that arbitrary code execution, imports, and introspection are blocked."""
    assert "error" in calculator("__import__('os').system('dir')")
    assert "error" in calculator("open('.env').read()")
    assert "error" in calculator("eval('1 + 1')")
    assert "error" in calculator("[x for x in (1, 2)]")
    assert "error" in calculator("().__class__.__bases__")
    assert "error" in calculator("lambda: 42")


def test_calculator_exponent_limits():
    """Verify protection against exponential algorithmic complexity/DoS."""
    # Base too large
    res_large_base = calculator("20000 ** 2")
    assert "error" in res_large_base
    assert "Base exceeds maximum" in res_large_base["error"]

    # Exponent too large
    res_large_exp = calculator("2 ** 500")
    assert "error" in res_large_exp
    assert "Exponent exceeds maximum" in res_large_exp["error"]


def test_get_datetime():
    """Verify ISO datetime format returned by get_datetime."""
    res = get_datetime()
    assert "datetime" in res
    # Ensure it parses as a valid ISO timestamp
    parsed = datetime.fromisoformat(res["datetime"])
    assert parsed.year >= 2024


def test_word_counter():
    """Verify word and character counts."""
    res = word_counter("Hello world from Nemotron agent")
    assert res["word_count"] == 5
    assert res["character_count"] == 31

    empty_res = word_counter("")
    assert empty_res["word_count"] == 0
    assert empty_res["character_count"] == 0
