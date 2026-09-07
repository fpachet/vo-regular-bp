"""Stable fallback arithmetic for unusually small/large constrained masses."""

from __future__ import annotations

import math

NEG_INF = -math.inf


def log_mass(value: float) -> float:
    return math.log(value) if value > 0 else NEG_INF


def mass_from_log(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def log_sum(values) -> float:
    values = tuple(values)
    maximum = max(values, default=NEG_INF)
    if maximum == NEG_INF:
        return NEG_INF
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def relative_weights(log_weights) -> tuple[float, ...]:
    maximum = max(log_weights, default=NEG_INF)
    if maximum == NEG_INF:
        return tuple(0.0 for _ in log_weights)
    return tuple(math.exp(value - maximum) for value in log_weights)


def log_backward(start, memo, terminal, successors):
    """Evaluate an acyclic dependency graph without Python recursion.

    ``terminal(key)`` returns a log mass or None; successors yield
    (child_key, log_edge_weight). Only pending rows are retained here.
    """
    stack = [(start, None)]
    while stack:
        key, row = stack.pop()
        if key in memo:
            continue
        if row is None:
            value = terminal(key)
            if value is not None:
                memo[key] = value
                continue
            row = tuple(successors(key))
            stack.append((key, row))
            for child, _weight in reversed(row):
                if child not in memo:
                    stack.append((child, None))
        else:
            memo[key] = log_sum(weight + memo[child] for child, weight in row)
    return memo[start]
