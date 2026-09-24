import pytest

from calculator import CalculatorError, calculate


@pytest.mark.parametrize("expression, expected", [
    ("1 + 2 * 3", 7),
    ("(1 + 2) * 3", 9),
    ("2 ** 10", 1024),
    ("10 % 3", 1),
    ("-5 + abs(-2)", -3),
    ("sqrt(16)", 4.0),
    ("round(3.14159, 2)", 3.14),
    ("mean([1, 2, 3, 4])", 2.5),
    ("mean(2, 4)", 3.0),
    ("pct_change(100, 125)", 25.0),
    ("pct_change(200, 150)", -25.0),
])
def test_valid_expressions(expression, expected):
    assert calculate(expression) == pytest.approx(expected)


def test_pct_change_required_case():
    assert calculate("pct_change(100,125)") == 25


def test_rmse():
    assert calculate("rmse([1, 2, 3], [1, 2, 3])") == 0
    assert calculate("rmse([0, 0], [3, 4])") == pytest.approx((12.5) ** 0.5)


@pytest.mark.parametrize("attack", [
    "__import__('os')",
    "__import__('os').system('dir')",
    "open('secret.txt')",
    "x + 1",
    "(1).real",
    "'a' * 3",
    "[x for x in (1, 2)]",
    "lambda: 1",
    "sqrt(x=4)",
    "True + 1",
    "9 ** 99999",
])
def test_rejects_unsafe_or_unsupported(attack):
    with pytest.raises(CalculatorError):
        calculate(attack)


@pytest.mark.parametrize("bad", ["", "   ", "1 +", "1 / 0", "pct_change(0, 5)", "rmse([1], [1, 2])", "[1, 2]", "1" * 201])
def test_rejects_invalid_input(bad):
    with pytest.raises(CalculatorError):
        calculate(bad)
