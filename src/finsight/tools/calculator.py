from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Sequence


class CalculationError(ValueError):
    pass


@dataclass(frozen=True)
class Calculation:
    value: float | str
    program: str
    steps: tuple[str, ...]


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)%?"
_TOKEN_RE = re.compile(rf"\s*([A-Za-z_][A-Za-z_0-9]*|#\d+|{_NUMBER}|[,()])")


def _constant(name: str) -> float:
    if not name.startswith("const_"):
        raise CalculationError(f"Unknown value: {name}")
    value = name.removeprefix("const_").replace("m", "-")
    try:
        return float(value)
    except ValueError as exc:
        raise CalculationError(f"Invalid constant: {name}") from exc


def _divide(left: float, right: float) -> float:
    if right == 0:
        raise CalculationError("Division by zero")
    return left / right


_OPERATIONS: dict[str, Callable[[list[float]], float | str]] = {
    "add": lambda values: values[0] + values[1],
    "subtract": lambda values: values[0] - values[1],
    "multiply": lambda values: values[0] * values[1],
    "divide": lambda values: _divide(values[0], values[1]),
    "exp": lambda values: math.pow(values[0], values[1]),
    "greater": lambda values: "yes" if values[0] > values[1] else "no",
    "table_average": lambda values: mean(values),
    "average": lambda values: mean(values),
    "table_sum": lambda values: sum(values),
    "sum": lambda values: sum(values),
    "table_max": lambda values: max(values),
    "max": lambda values: max(values),
    "table_min": lambda values: min(values),
    "min": lambda values: min(values),
}


class _Parser:
    def __init__(self, program: str, references: list[float | str] | None = None):
        matches = list(_TOKEN_RE.finditer(program))
        consumed = "".join(match.group(0) for match in matches).strip()
        if re.sub(r"\s+", "", consumed) != re.sub(r"\s+", "", program):
            raise CalculationError("Program contains unsupported characters")
        self.tokens = [match.group(1) for match in matches]
        self.position = 0
        self.steps: list[str] = []
        self.references = references or []

    def parse(self) -> float | str:
        value = self._expression()
        if self.position != len(self.tokens):
            raise CalculationError("Unexpected trailing tokens")
        return value

    def _take(self) -> str:
        if self.position >= len(self.tokens):
            raise CalculationError("Unexpected end of program")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def _expression(self) -> float | str:
        token = self._take()
        if re.fullmatch(rf"{_NUMBER}", token) and token.endswith("%"):
            return float(token[:-1]) / 100
        if re.fullmatch(rf"{_NUMBER}", token):
            return float(token)
        if token.startswith("#"):
            index = int(token[1:])
            try:
                return self.references[index]
            except IndexError as exc:
                raise CalculationError(f"Unknown step reference: {token}") from exc
        if token.startswith("const_"):
            return _constant(token)
        operation = _OPERATIONS.get(token)
        if operation is None:
            raise CalculationError(f"Operation is not allowed: {token}")
        if self._take() != "(":
            raise CalculationError("Expected '('")
        values = [self._expression()]
        while self.position < len(self.tokens) and self.tokens[self.position] == ",":
            self.position += 1
            values.append(self._expression())
        if self._take() != ")":
            raise CalculationError("Expected ')'")
        if token in {"add", "subtract", "multiply", "divide", "exp", "greater"} and len(values) != 2:
            raise CalculationError(f"{token} expects exactly two arguments")
        if any(isinstance(value, str) for value in values):
            raise CalculationError(f"{token} requires numeric arguments")
        result = operation(values)  # type: ignore[arg-type]
        self.steps.append(f"{token}({', '.join(map(str, values))}) = {result}")
        return result


def _split_steps(program: str) -> list[str]:
    steps: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(program):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise CalculationError("Unbalanced parentheses")
        elif character == "," and depth == 0:
            step = program[start:index].strip()
            if step:
                steps.append(step)
            start = index + 1
    if depth != 0:
        raise CalculationError("Unbalanced parentheses")
    final_step = program[start:].strip()
    if final_step:
        steps.append(final_step)
    return steps


def _parse_table_number(text: str) -> float:
    normalized = text.replace(",", "").replace("$", "").strip()
    normalized = normalized.split("(")[0].strip()
    if normalized.endswith("%"):
        return float(normalized[:-1]) / 100
    return float(normalized)


def _execute_table_step(
    step: str,
    table: Sequence[Sequence[str]],
) -> tuple[float, str] | None:
    match = re.fullmatch(r"(table_(?:average|sum|min|max))\((.*),\s*[^,()]+\)", step.strip())
    if not match:
        return None
    operation, row_label = match.groups()
    # Match the official evaluator: duplicate row labels are overwritten by the last row.
    rows_by_label = {row[0].strip(): row for row in table if row}
    row = rows_by_label.get(row_label.strip())
    if row is None:
        raise CalculationError(f"Table row not found: {row_label}")
    try:
        values = [_parse_table_number(value) for value in row[1:]]
    except ValueError as exc:
        raise CalculationError(f"Table row is not numeric: {row_label}") from exc
    if not values:
        raise CalculationError(f"Table row has no values: {row_label}")
    if operation == "table_average":
        result = mean(values)
    elif operation == "table_sum":
        result = sum(values)
    elif operation == "table_min":
        result = min(values)
    else:
        result = max(values)
    return result, f"{operation}({row_label}) = {result}"


def execute_program(
    program: str,
    *,
    table: Sequence[Sequence[str]] | None = None,
) -> Calculation:
    """Execute a whitelisted FinQA-style arithmetic DSL, never arbitrary Python."""
    references: list[float | str] = []
    trace: list[str] = []
    steps = _split_steps(program.strip())
    if not steps:
        raise CalculationError("Program is empty")
    for step in steps:
        table_result = _execute_table_step(step, table or ())
        if table_result is not None:
            value, table_trace = table_result
            references.append(value)
            trace.append(table_trace)
            continue
        parser = _Parser(step, references)
        value = parser.parse()
        references.append(value)
        trace.extend(parser.steps)
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculationError("Result is not finite")
    return Calculation(value=value, program=program, steps=tuple(trace))
