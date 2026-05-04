"""Exact BP for fixed-horizon positional constraints.

This module is the no-DFA specialization of product BP.  Positional constraints
are represented as time-indexed symbol masks, so the recursion only ranges over
context states:

    beta[t, s] = sum_y P(y | s) beta[t + 1, canon(s, y))

No approximation or pruning is introduced.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import random
from typing import Protocol

from .context import Context, ContextGraph, Edge, Symbol, _as_context
from .product_bp import _sample_order

PositionConstraint = Callable[[Symbol], bool] | Iterable[Symbol]
PositionConstraints = Mapping[int, PositionConstraint]
AllowedForbiddenSymbols = Mapping[int, Iterable[Symbol]]


class ContextModel(Protocol):
    start_state: Context

    def outgoing(self, context: Iterable[Symbol] | Context) -> tuple[Edge, ...]:
        ...


class LazyBackoffContextModel:
    """Backoff continuation model with outgoing edges compiled on demand.

    The stochastic semantics match ``ContextGraph.from_backoff_sequences``:
    each context uses a normalized weighted mixture of its own continuation
    distribution and lower-order suffix distributions.  The difference is only
    operational: edges are materialized when reached by BP.
    """

    def __init__(
        self,
        counts: Mapping[Context, Counter[Symbol]],
        *,
        max_order: int,
        backoff_weight: float,
        start_state: Iterable[Symbol] | Context = (),
        track_order_weights: bool = False,
    ) -> None:
        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        if not 0.0 <= backoff_weight <= 1.0:
            raise ValueError("backoff_weight must be in [0, 1]")

        self.counts = dict(counts)
        self.max_order = int(max_order)
        self.backoff_weight = float(backoff_weight)
        self.start_state = _as_context(start_state)
        self.track_order_weights = bool(track_order_weights)
        contexts = set(self.counts)
        contexts.add(self.start_state)
        self.contexts = frozenset(contexts)
        self.alphabet = frozenset(symbol for counter in self.counts.values() for symbol in counter)
        self.distributions = {
            context: tuple((symbol, float(count) / total) for symbol, count in counter.items())
            for context, counter in self.counts.items()
            for total in (float(sum(counter.values())),)
            if total > 0.0
        }
        self._outgoing_cache: dict[Context, tuple[Edge, ...]] = {}

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[Sequence[Symbol]],
        *,
        max_order: int,
        backoff_weight: float,
        start_state: Iterable[Symbol] | Context = (),
        track_order_weights: bool = False,
    ) -> "LazyBackoffContextModel":
        if max_order < 0:
            raise ValueError("max_order must be non-negative")

        counts: dict[Context, Counter[Symbol]] = defaultdict(Counter)
        for sequence in sequences:
            tokens = tuple(sequence)
            for index, symbol in enumerate(tokens):
                order_limit = min(max_order, index)
                for order in range(order_limit + 1):
                    context = tokens[index - order : index] if order else ()
                    counts[context][symbol] += 1
        return cls(
            counts,
            max_order=max_order,
            backoff_weight=backoff_weight,
            start_state=start_state,
            track_order_weights=track_order_weights,
        )

    def outgoing(self, context: Iterable[Symbol] | Context) -> tuple[Edge, ...]:
        state = _as_context(context)
        cached = self._outgoing_cache.get(state)
        if cached is not None:
            return cached

        if self.track_order_weights:
            edges = self._outgoing_with_order_weights(state)
            self._outgoing_cache[state] = edges
            return edges

        scores: dict[Symbol, float] = {}
        context_order = len(state)
        for order in range(context_order, -1, -1):
            suffix = state[-order:] if order else ()
            distribution = self.distributions.get(suffix)
            if not distribution:
                continue
            weight = self.backoff_weight ** (context_order - order)
            for symbol, probability in distribution:
                contribution = weight * probability
                scores[symbol] = scores.get(symbol, 0.0) + contribution

        total_score = float(sum(scores.values()))
        if total_score <= 0.0:
            edges: tuple[Edge, ...] = ()
        else:
            edges = tuple(
                Edge(
                    symbol,
                    score / total_score,
                    ContextGraph._canon(state, symbol, self.contexts, self.max_order),
                )
                for symbol, score in scores.items()
                if score > 0.0
            )
        self._outgoing_cache[state] = edges
        return edges

    def _outgoing_with_order_weights(self, state: Context) -> tuple[Edge, ...]:
        scores: dict[Symbol, float] = {}
        order_scores: dict[Symbol, Counter[int]] = defaultdict(Counter)
        context_order = len(state)
        for order in range(context_order, -1, -1):
            suffix = state[-order:] if order else ()
            distribution = self.distributions.get(suffix)
            if not distribution:
                continue
            weight = self.backoff_weight ** (context_order - order)
            for symbol, probability in distribution:
                contribution = weight * probability
                scores[symbol] = scores.get(symbol, 0.0) + contribution
                order_scores[symbol][order] += contribution

        total_score = float(sum(scores.values()))
        if total_score <= 0.0:
            return ()
        return tuple(
            Edge(
                symbol,
                score / total_score,
                ContextGraph._canon(state, symbol, self.contexts, self.max_order),
                _normalized_order_weights(order_scores[symbol], score),
            )
            for symbol, score in scores.items()
            if score > 0.0
        )

    def prefix_context(self, prefix: Sequence[Symbol]) -> Context:
        for order in range(min(self.max_order, len(prefix)), -1, -1):
            suffix = tuple(prefix[-order:]) if order else ()
            if suffix in self.contexts:
                return suffix
        return self.start_state

    @property
    def materialized_edge_count(self) -> int:
        return sum(len(edges) for edges in self._outgoing_cache.values())


@dataclass
class PositionalBPResult:
    model: ContextModel
    length: int
    start_context: Context
    constraints: PositionConstraints
    betas: list[dict[Context, float]]
    allowed_edge_counts: list[dict[Context, int]]
    _transition_weight_cache: dict[tuple[int, Context], tuple[tuple[Edge, float], ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def partition_function(self) -> float:
        return self.betas[0].get(self.start_context, 0.0)

    @property
    def unique_state_count(self) -> int:
        return len(set().union(*self.betas)) if self.betas else 0

    @property
    def time_indexed_state_count(self) -> int:
        return sum(len(layer) for layer in self.betas)

    @property
    def edge_count(self) -> int:
        return sum(sum(layer.values()) for layer in self.allowed_edge_counts)

    def transition_weights(self, time: int, context: Iterable[Symbol] | Context) -> tuple[tuple[Edge, float], ...]:
        if time < 0 or time >= self.length:
            raise ValueError("time is outside the BP horizon")
        state = _as_context(context)
        cache_key = (time, state)
        cached = self._transition_weight_cache.get(cache_key)
        if cached is not None:
            return cached

        beta_next = self.betas[time + 1]
        constraint = self.constraints.get(time)
        weighted = []
        for edge in self.model.outgoing(state):
            if not _allows(constraint, edge.symbol):
                continue
            weight = edge.probability * beta_next.get(edge.next_state, 0.0)
            if weight > 0.0:
                weighted.append((edge, weight))
        result = tuple(weighted)
        self._transition_weight_cache[cache_key] = result
        return result

    def _sample_edge(self, time: int, context: Context, generator: random.Random) -> Edge:
        weighted = self.transition_weights(time, context)
        total = sum(weight for _, weight in weighted)
        if total <= 0.0:
            raise RuntimeError("BP table contains no positive continuation for a reachable state")
        threshold = generator.random() * total
        cumulative = 0.0
        chosen = weighted[-1][0]
        for edge, weight in weighted:
            cumulative += weight
            if threshold <= cumulative:
                return edge
        return chosen

    def sample(self, *, rng: random.Random | int | None = None) -> tuple[Symbol, ...]:
        generator = _coerce_rng(rng)
        if self.partition_function <= 0.0:
            raise ValueError("cannot sample because the constrained partition function is zero")

        context = self.start_context
        output: list[Symbol] = []
        for time in range(self.length):
            edge = self._sample_edge(time, context, generator)
            output.append(edge.symbol)
            context = edge.next_state
        return tuple(output)

    def sample_many(self, count: int, *, rng: random.Random | int | None = None) -> list[tuple[Symbol, ...]]:
        generator = _coerce_rng(rng)
        return [self.sample(rng=generator) for _ in range(count)]

    def sample_with_orders(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> tuple[tuple[Symbol, ...], tuple[int, ...]]:
        generator = _coerce_rng(rng)
        if self.partition_function <= 0.0:
            raise ValueError("cannot sample because the constrained partition function is zero")

        context = self.start_context
        output: list[Symbol] = []
        orders: list[int] = []
        for time in range(self.length):
            edge = self._sample_edge(time, context, generator)
            output.append(edge.symbol)
            orders.append(_sample_order(edge.order_weights or ((len(context), 1.0),), generator))
            context = edge.next_state
        return tuple(output), tuple(orders)

    def conditional_probability(self, sequence: Sequence[Symbol]) -> float:
        if len(sequence) != self.length:
            raise ValueError("sequence length must match the BP horizon")
        if self.partition_function <= 0.0:
            return 0.0

        context = self.start_context
        probability = 1.0
        for time, symbol in enumerate(sequence):
            beta_now = self.betas[time].get(context, 0.0)
            if beta_now <= 0.0:
                return 0.0
            edge = next(
                (edge for edge, _ in self.transition_weights(time, context) if edge.symbol == symbol),
                None,
            )
            if edge is None:
                return 0.0
            weight = edge.probability * self.betas[time + 1].get(edge.next_state, 0.0)
            if weight <= 0.0:
                return 0.0
            probability *= weight / beta_now
            context = edge.next_state
        return probability


def run_positional_bp(
    model: ContextModel,
    *,
    length: int,
    start_context: Iterable[Symbol] | Context | None = None,
    constraints: PositionConstraints | None = None,
) -> PositionalBPResult:
    """Run exact backward DP with time-indexed positional constraints."""

    if length < 0:
        raise ValueError("length must be non-negative")
    context0 = model.start_state if start_context is None else _as_context(start_context)
    position_constraints = dict(constraints or {})
    _validate_constraint_positions(position_constraints, length)

    betas: list[dict[Context, float]] = [dict() for _ in range(length + 1)]
    allowed_edge_counts: list[dict[Context, int]] = [dict() for _ in range(length)]

    def beta(time_index: int, context: Context) -> float:
        cached = betas[time_index].get(context)
        if cached is not None:
            return cached
        if time_index == length:
            betas[time_index][context] = 1.0
            return 1.0

        constraint = position_constraints.get(time_index)
        total = 0.0
        allowed_count = 0
        for edge in model.outgoing(context):
            if not _allows(constraint, edge.symbol):
                continue
            allowed_count += 1
            total += edge.probability * beta(time_index + 1, edge.next_state)

        allowed_edge_counts[time_index][context] = allowed_count
        betas[time_index][context] = total
        return total

    beta(0, context0)
    return PositionalBPResult(
        model=model,
        length=length,
        start_context=context0,
        constraints=position_constraints,
        betas=betas,
        allowed_edge_counts=allowed_edge_counts,
    )


def _normalized_order_weights(
    order_scores: Counter[int],
    symbol_score: float,
) -> tuple[tuple[int, float], ...]:
    if symbol_score <= 0.0:
        return ()
    return tuple(
        sorted(
            ((order, contribution / symbol_score) for order, contribution in order_scores.items()),
            reverse=True,
        )
    )


def _allows(constraint: PositionConstraint | None, symbol: Symbol) -> bool:
    if constraint is None:
        return True
    if callable(constraint):
        return bool(constraint(symbol))
    return symbol in constraint


def _validate_constraint_positions(constraints: Mapping[int, PositionConstraint], length: int) -> None:
    for position in constraints:
        if position < 0 or position >= length:
            raise IndexError(f"constraint position {position} is outside length {length}")


def _coerce_rng(rng: random.Random | int | None) -> random.Random:
    if isinstance(rng, random.Random):
        return rng
    return random.Random(rng)
