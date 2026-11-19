"""Induce expression lowering rules from observation, then compile to CIL."""

import ast
from dataclasses import dataclass
from itertools import product
import math
import operator

from cil import compile_cil


PRIMITIVES = {"add": operator.add, "mul": operator.mul}
OPERATORS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.USub: "neg"}
ARITY = {"+": 2, "-": 2, "*": 2, "neg": 1}
TERM_LIMIT = 5


@dataclass(frozen=True)
class Compiled:
    cil: str
    bytecode: bytes


def apply(tree, operands):
    if isinstance(tree, tuple):
        op, left, right = tree
        return PRIMITIVES[op](apply(left, operands), apply(right, operands))
    return operands[tree] if isinstance(tree, str) else tree


def rules_from(examples, audit, arity):
    """Use the smallest consistent hypothesis, then verify it once on separate evidence."""
    def check(rows):
        if not isinstance(rows, (tuple, list)) or not 3 <= len(rows) <= 64:
            raise ValueError("Supply 3 to 64 observations per split")
        if any(not isinstance(row, (tuple, list)) or len(row) != arity + 1
               or any(type(value) not in (int, float) or
               not -1e6 <= value <= 1e6 for value in row) for row in rows):
            raise ValueError("Expected finite operand/result rows")
        if len({tuple(row[:-1]) for row in rows}) < 3:
            raise ValueError("Supply at least three distinct inputs")

    check(examples)
    check(audit)
    if {tuple(r[:-1]) for r in examples} & {tuple(r[:-1]) for r in audit}:
        raise ValueError("Teaching and verification inputs must be separate")
    pools = {1: [*"ab"[:arity], -1., 0., 1.]}
    candidate_count = 0
    for size in range(1, TERM_LIMIT + 1, 2):
        if size > 1:
            pool = pools[size] = []
            for left_size in range(1, size - 1, 2):
                for op, left, right in product(PRIMITIVES, pools[left_size],
                                               pools[size - 1 - left_size]):
                    pool.append((op, left, right))
        for tree in pools[size]:
            candidate_count += 1
            if all(apply(tree, dict(zip("ab", row[:-1]))) == row[-1] for row in examples):
                if not all(apply(tree, dict(zip("ab", row[:-1]))) == row[-1] for row in audit):
                    raise ValueError("The selected rule failed verification")
                return tree, candidate_count
    raise ValueError("No supported lowering rule fits the examples")


def render(tree):
    if isinstance(tree, tuple):
        op, left, right = tree
        return f"({op} {render(left)} {render(right)})"
    return str(tree)


def substitute(tree, operands):
    if isinstance(tree, tuple):
        op, left, right = tree
        return op, substitute(left, operands), substitute(right, operands)
    return operands[tree] if isinstance(tree, str) else tree


def fold(tree):
    if not isinstance(tree, tuple):
        return tree
    op, left, right = tree
    a, b = fold(left), fold(right)
    if isinstance(a, float) and isinstance(b, float):
        value = PRIMITIVES[op](a, b)
        if not math.isfinite(value):
            raise ValueError("Constant expression is nonfinite")
        return value
    return op, a, b


def check_ir(tree):
    pending = [(tree, 0)]
    remaining = 1024
    while pending:
        node, depth = pending.pop()
        remaining -= 1
        if remaining < 0 or depth > 96:
            raise ValueError("Lowered expression exceeds the CIL size or depth limit")
        if isinstance(node, tuple):
            pending.extend((child, depth + 1) for child in node[1:])


class CompilerModel:
    """A small lowering model with explicitly taught and verified operator rules."""

    def __init__(self):
        self.rules = {}
        self.trials = {}

    def teach(self, op, examples, audit):
        if not isinstance(op, str) or op not in ARITY:
            raise ValueError("Unsupported operator name")
        tree, tried = rules_from(examples, audit, ARITY[op])
        self.rules[op], self.trials[op] = tree, tried
        return render(tree)

    def compile(self, source, *, optimize=True):
        if not isinstance(source, str) or not 1 <= len(source.strip()) <= 512:
            raise ValueError("Enter one expression up to 512 characters")
        try:
            tree = ast.parse(source.strip(), mode="eval").body
        except (SyntaxError, RecursionError) as error:
            raise ValueError("Invalid source expression") from error

        def lower(expr):
            if isinstance(expr, ast.Name) and expr.id == "x":
                return "x"
            if isinstance(expr, ast.Constant) and type(expr.value) in (float, int):
                if not -1e100 <= expr.value <= 1e100:
                    raise ValueError("Numeric literal is out of range")
                return float(expr.value)
            if isinstance(expr, ast.BinOp):
                op = OPERATORS.get(type(expr.op))
                values = (lower(expr.left), lower(expr.right))
            elif isinstance(expr, ast.UnaryOp):
                op = OPERATORS.get(type(expr.op))
                values = (lower(expr.operand),)
            else:
                raise ValueError("Only numeric literals, x, and taught arithmetic are supported")
            if op not in self.rules:
                raise ValueError("No verified rule for that operator")
            return substitute(self.rules[op], dict(zip("ab", values)))

        cil = lower(tree)
        check_ir(cil)
        source_cil = f"(program {render(fold(cil) if optimize else cil)})"
        return Compiled(source_cil, compile_cil(source_cil))
