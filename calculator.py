"""Safe calculator tool.

We parse the expression into a Python AST (abstract syntax tree) and only
evaluate the node types we explicitly allow. There is no eval(), so input like
"__import__('os')" can never run code.
"""
import ast
import math
import operator

MAX_LENGTH = 200
MAX_EXPONENT = 1000


class CalculatorError(ValueError):
    pass


def _mean(*values):
    nums = _flatten(values)
    if not nums:
        raise CalculatorError("mean() needs at least one number")
    return sum(nums) / len(nums)


def _rmse(actual, predicted):
    if not isinstance(actual, list) or not isinstance(predicted, list):
        raise CalculatorError("rmse() needs two lists, e.g. rmse([1, 2], [1, 3])")
    if len(actual) != len(predicted) or not actual:
        raise CalculatorError("rmse() lists must be non-empty and the same length")
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def _pct_change(old, new):
    if old == 0:
        raise CalculatorError("pct_change() is undefined when the old value is 0")
    return (new - old) / abs(old) * 100


def _round(x, ndigits=0):
    return round(x, int(ndigits))


def _flatten(values):
    out = []
    for v in values:
        out.extend(v if isinstance(v, list) else [v])
    return out


FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": _round,
    "mean": _mean,
    "rmse": _rmse,
    "pct_change": _pct_change,
}

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalculatorError("Only numbers are allowed")

    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError("Exponent too large")
        return BIN_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_eval(node.operand))

    if isinstance(node, ast.List):
        return [_eval(item) for item in node.elts]

    if isinstance(node, ast.Call):
        # Only plain names from the whitelist: no attributes (os.system), no keywords.
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise CalculatorError(f"Function not allowed. Allowed: {', '.join(FUNCTIONS)}")
        if node.keywords:
            raise CalculatorError("Keyword arguments are not allowed")
        return FUNCTIONS[node.func.id](*[_eval(arg) for arg in node.args])

    raise CalculatorError(f"Unsupported syntax: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Evaluate a math expression safely. Raises CalculatorError on bad input."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalculatorError("Expression is empty")
    if len(expression) > MAX_LENGTH:
        raise CalculatorError(f"Expression longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise CalculatorError("Invalid expression syntax") from None
    try:
        result = _eval(tree)
    except CalculatorError:
        raise
    except (ZeroDivisionError, OverflowError, ValueError, TypeError) as e:
        raise CalculatorError(f"Math error: {e}") from None
    if isinstance(result, list):
        raise CalculatorError("The result must be a single number, not a list")
    return result
