"""Text conversation around the original NNM checkpoint and CIL tools."""

import ast
import json
import math
import operator
from pathlib import Path
import re

import numpy as np

from cil import compile_cil, execute
from compiler_model import CompilerModel, render
from event_language import answer_event
from nnm49 import NeuralCore


CHECKPOINT = Path(__file__).with_name("core.npz")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
AT_INPUT = rf"\s*(?:@|\bat\s+|\bfor\s+)\s*(?:x\s*=\s*)?({NUMBER})"
PATTERNS = (
    ("predict", re.compile(
        rf"(?:what (?:does|would) (?:the (?:model|nnm)|it) predict (?:at|for) |"
        rf"(?:please )?(?:predict|evaluate)(?: (?:at|for))? )?(?:x\s*=\s*)?({NUMBER})", re.I
    )),
    ("compare", re.compile(rf"compare ({NUMBER})(?:\s*(?:,| and )\s*)({NUMBER})", re.I)),
    ("repeat", re.compile(
        rf"(?:run|repeat|apply(?: the model)?) (\d{{1,3}}) times? (?:from|at|to) ({NUMBER})", re.I
    )),
    ("again", re.compile(rf"(?:try|run|evaluate) (?:that|it){AT_INPUT}", re.I)),
    ("run", re.compile(rf"(?:(?:run|evaluate)\s*:?\s*|what is )(.+?){AT_INPUT}", re.I | re.S)),
    ("compile", re.compile(r"(?:please )?compile\s*:?\s+(.+)", re.I | re.S)),
)
FIT = re.compile(r"(?:fit|find|create|build) (?:an? |the )?(?:simpler? )?"
                 r"(?:equation|formula)(?: (?:to|for|that fits))?\s*:?\s*(\[.+)",
                 re.I | re.S)
HELP = (
    "Try “What would the model predict for 0.5?”, “Run x*x + 2 at 3”, "
    "or “Try that at 5”. Use “Help” for equation fitting and event examples."
)


def make_compiler():
    """Teach the original compiler experiment using independent audit examples."""
    compiler = CompilerModel()
    fit_pairs = [(2, 3), (-5, 1), (0, -2), (-2, -4)]
    audit_pairs = [(6, -3), (4, 4), (-8, -7), (3, 0)]
    for name, fn in (("+", operator.add), ("-", operator.sub),
                     ("*", operator.mul), ("neg", operator.neg)):
        arity = 1 if name == "neg" else 2
        fit = [(*row[:arity], fn(*row[:arity])) for row in fit_pairs]
        audit = [(*row[:arity], fn(*row[:arity])) for row in audit_pairs]
        compiler.teach(name, fit, audit)
    return compiler


def input_number(value):
    if type(value) not in (int, float, str):
        raise ValueError("Enter a finite numeric input.")
    try:
        value = float(value)
    except (ValueError, OverflowError):
        raise ValueError("Enter a finite numeric input.") from None
    if not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("Enter a finite numeric input with magnitude at most 1e100.")
    return value


def number_text(value):
    return f"{value:.12g}"


def data_rows(text, minimum):
    try:
        rows = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError, RecursionError):
        raise ValueError("Enter pairs like [(0,1), (1,3), (2,5)].") from None
    if not isinstance(rows, (list, tuple)) or not minimum <= len(rows) <= 32:
        raise ValueError(f"Provide {minimum} to 32 pairs of x and y values per set.")
    if any(not isinstance(row, (list, tuple)) or len(row) != 2 or
           any(type(v) not in (int, float) or not -1e6 <= v <= 1e6 for v in row)
           for row in rows):
        raise ValueError("Data must be finite numeric pairs between -1e6 and 1e6.")
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Use distinct x values within each set.")
    return np.asarray(rows, dtype=float)


def coefficients(rows, degree):
    if degree == 0:
        return [rows[:, 1].mean()]
    fitted = np.polynomial.Polynomial.fit(rows[:, 0], rows[:, 1], degree)
    return fitted.convert().coef


def expression_from(coefficients):
    """Emit a compact polynomial using only the compiler's taught operators."""
    terms = []
    tolerance = max(1.0, max(abs(value) for value in coefficients)) * 1e-12
    for power, value in enumerate(coefficients):
        if abs(value) <= tolerance:
            continue
        if power == 0:
            terms.append(number_text(value))
        else:
            term = "*".join(["x"] * power)
            terms.append(term if value == 1 else f"{number_text(value)}*{term}")
    return " + ".join(terms) or "0"


def fit_equation(source, compiler):
    fit_text, separator, test_text = source.partition(";")
    fit = data_rows(fit_text, 3)
    holdout = None
    if separator:
        split = re.fullmatch(r"\s*(?:test on|holdout)\s*(.+)", test_text, re.I | re.S)
        if not split:
            raise ValueError("After the fit pairs, use “; test on [(x,y), ...]”.")
        holdout = data_rows(split[1], 1)
        if set(fit[:, 0]) & set(holdout[:, 0]):
            raise ValueError("Fit and holdout sets must use different x values.")

    scores = {}
    for degree in range(min(2, len(fit) - 2) + 1):
        errors = []
        for row in range(len(fit)):
            x, y = fit[row]
            weight = coefficients(np.delete(fit, row, axis=0), degree)
            errors.append((np.polynomial.polynomial.polyval(x, weight) - y) ** 2)
        scores[degree] = float(np.mean(errors))
    # Keep held-out values outside both coefficient fitting and degree selection.
    threshold = min(scores.values()) + 1e-12 * max(1.0, float(np.var(fit[:, 1])))
    chosen = next(degree for degree, score in scores.items() if score <= threshold)
    expression = expression_from(coefficients(fit, chosen))
    program = compiler.compile(expression)
    details = [f"y = {expression}", f"CIL: {program.cil}",
               "Fit-set leave-one-out MSE: " + ", ".join(
                   f"degree {degree} = {number_text(score)}" for degree, score in scores.items()
               )]
    if holdout is not None:
        errors = []
        for x, y in holdout:
            prediction = execute(program.bytecode, float(x), None)
            errors.append((prediction - y) ** 2)
            details.append(f"x={number_text(x)}: predicted {number_text(prediction)}, "
                           f"observed {number_text(y)}")
        details.append(f"Holdout MSE: {number_text(float(np.mean(errors)))}")
    return expression, program, "\n".join(details)


class ResearchChat:
    def __init__(self, *, checkpoint=CHECKPOINT, language=None):
        self.model = NeuralCore.load(checkpoint)
        self.compiler = make_compiler()
        self.neural_program = compile_cil("(program (learn x))")
        self.language = language
        self.reset()

    def reset(self):
        self.last_program = None
        self.event_question = None
        self.turns = []

    def reply(self, text):
        text = text.strip()
        if not 1 <= len(text) <= 2_000:
            return "Enter a message up to 2,000 characters."
        try:
            answer = self._reply(text)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            answer = f"Couldn’t complete that request. {error}"
        self.turns.extend(({"role": "user", "content": text},
                           {"role": "assistant", "content": answer[:2_000]}))
        self.turns = self.turns[-8:]
        return answer

    def run(self, expression, x):
        x = input_number(x)
        program = self.compiler.compile(expression)
        result = execute(program.bytecode, x, self.model)
        self.last_program = (expression.strip(), program)
        return f"{expression.strip()} at x={number_text(x)} → {number_text(result)}\nCIL: {program.cil}"

    def _apply(self, action, args):
        if action == "compile":
            program = self.compiler.compile(args[0])
            self.last_program = (args[0], program)
            return f"CIL: {program.cil}\nBytecode ({len(program.bytecode)} bytes): {program.bytecode.hex()}"
        if action == "run":
            return self.run(*args)
        if action == "again":
            if self.last_program is None:
                return "Compile or run an expression first."
            x = input_number(args[0])
            expression, program = self.last_program
            result = execute(program.bytecode, x, self.model)
            return f"{expression} at x={number_text(x)} → {number_text(result)}"
        if action == "repeat":
            count = int(args[0])
            if not 1 <= count <= 20:
                raise ValueError("Choose between 1 and 20 runs.")
            inputs, outputs = [input_number(args[1])], []
            for index in range(count):
                x = inputs[-1]
                y = execute(self.neural_program, x, self.model)
                outputs.append(f"{index+1}. {number_text(x)} → {number_text(y)}")
                inputs.append(y)
        else:
            inputs = [input_number(value) for value in args]
            outputs = [f"x={number_text(x)} → {number_text(execute(self.neural_program, x, self.model))}"
                       for x in inputs]
        label = "NNM-49 prediction:" if action == "predict" else "NNM-49 predictions:"
        if any(abs(x) > 1 for x in inputs):
            label += "\nInput outside the original training interval [-1, 1]."
        return label + "\n" + "\n".join(outputs)

    def _reply(self, text):
        query = text.replace("−", "-").rstrip("?").strip()
        if query.casefold() in {"help", "what can you do", "how do i use this"}:
            return (HELP + "\n\nFit an equation to [(0,1), (1,3), (2,5), (3,7)]; "
                    "test on [(4,9), (5,11)]\n"
                    "Maya was painting from 2:00 to 2:30. At 2:12, was she painting?")
        if query.casefold() in {"build a compiler", "show compiler rules"}:
            return "Verified arithmetic lowerings:\n" + "\n".join(
                f"{op} → {render(tree)}" for op, tree in self.compiler.rules.items()
            )
        if match := FIT.fullmatch(query):
            expression, program, details = fit_equation(match[1], self.compiler)
            self.last_program = (expression, program)
            return details
        for action, pattern in PATTERNS:
            if match := pattern.fullmatch(query):
                return self._apply(action, match.groups())

        result = answer_event(text)
        if result["disposition"] == "unsupported" and self.event_question:
            if re.match(r"^\s*(?:at\s+)?\d{1,2}(?::\d{2}|\s*[ap]m)", text, re.I):
                result = answer_event(self.event_question, follow_up=text)
        if result["disposition"] != "unsupported":
            if result["disposition"] == "answered":
                self.event_question = None
            elif result["disposition"] == "clarify" and text.endswith("?"):
                self.event_question = text
            return result["answer"]
        if self.language is not None:
            return self._language_reply(text)
        return HELP

    def _language_reply(self, text):
        response = self.language([*self.turns, {"role": "user", "content": text}])
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, RecursionError):
            return "Local language model\n" + response
        if isinstance(data, dict):
            action = data.get("action")
            expression = data.get("expression")
            if action == "compile" and isinstance(expression, str):
                return self._apply(action, (expression,))
            if action == "run" and isinstance(expression, str):
                return self.run(expression, data.get("x"))
            if action == "predict":
                return self._apply(action, (data.get("x"),))
            if action == "reply" and isinstance(data.get("text"), str):
                return "Local language model\n" + data["text"]
        return "The local language model returned an unrecognized action. " + HELP
