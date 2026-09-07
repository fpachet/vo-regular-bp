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
from ._numerics import NEG_INF, log_mass, log_sum, mass_from_log, relative_weights

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

    _log_betas: list[dict[Context, float]] | None = field(default=None, repr=False)

    @property
    def log_partition_function(self) -> float:
        if self._log_betas is not None:
            return self._log_betas[0].get(self.start_context, NEG_INF)
        return log_mass(self.partition_function)

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

        constraint = self.constraints.get(time)
        if self._log_betas is not None:
            edges = tuple(e for e in self.model.outgoing(state) if _allows(constraint, e.symbol))
            logs = tuple(
                log_mass(e.probability) + self._log_betas[time + 1].get(e.next_state, NEG_INF)
                for e in edges
            )
            result = tuple((e, w) for e, w in zip(edges, relative_weights(logs)) if w > 0)
            self._transition_weight_cache[cache_key] = result
            return result
        beta_next = self.betas[time + 1]
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
        if self.log_partition_function == NEG_INF:
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
        if self.log_partition_function == NEG_INF:
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
        if self.log_partition_function == NEG_INF:
            return 0.0

        if self._log_betas is not None:
            return mass_from_log(self.log_conditional_probability(sequence))
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

    def log_conditional_probability(self, sequence: Sequence[Symbol]) -> float:
        if len(sequence) != self.length:
            raise ValueError("sequence length must match the BP horizon")
        if self.log_partition_function == NEG_INF:
            return NEG_INF
        context = self.start_context
        value = 0.0
        for time, symbol in enumerate(sequence):
            if not _allows(self.constraints.get(time), symbol):
                return NEG_INF
            edge = next((e for e in self.model.outgoing(context) if e.symbol == symbol), None)
            if edge is None or edge.probability <= 0:
                return NEG_INF
            value += log_mass(edge.probability)
            context = edge.next_state
        return min(0.0, value - self.log_partition_function)


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

    layers = [{context0}]
    for time_index in range(length):
        next_layer = set()
        constraint = position_constraints.get(time_index)
        for context in layers[-1]:
            edges = tuple(e for e in model.outgoing(context) if _allows(constraint, e.symbol))
            allowed_edge_counts[time_index][context] = len(edges)
            next_layer.update(e.next_state for e in edges)
        layers.append(next_layer)
    betas[length] = {context: 1.0 for context in layers[length]}
    needs_log = False
    for time_index in range(length - 1, -1, -1):
        constraint = position_constraints.get(time_index)
        for context in layers[time_index]:
            value = sum(
                e.probability * betas[time_index + 1].get(e.next_state, 0.0)
                for e in model.outgoing(context)
                if _allows(constraint, e.symbol)
            )
            betas[time_index][context] = value
            if 0 < value < 1e-200:
                needs_log = True
            elif value == 0 and any(
                e.probability > 0 and betas[time_index + 1].get(e.next_state, 0) > 0
                for e in model.outgoing(context)
                if _allows(constraint, e.symbol)
            ):
                needs_log = True
    log_betas = None
    if needs_log:
        log_betas = [{} for _ in range(length + 1)]
        log_betas[length] = {context: 0.0 for context in layers[length]}
        for time_index in range(length - 1, -1, -1):
            constraint = position_constraints.get(time_index)
            for context in layers[time_index]:
                value = log_sum(
                    log_mass(e.probability) + log_betas[time_index + 1].get(e.next_state, NEG_INF)
                    for e in model.outgoing(context)
                    if _allows(constraint, e.symbol)
                )
                log_betas[time_index][context] = value
                betas[time_index][context] = mass_from_log(value)
    return PositionalBPResult(
        model=model,
        length=length,
        start_context=context0,
        constraints=position_constraints,
        betas=betas,
        _log_betas=log_betas,
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
