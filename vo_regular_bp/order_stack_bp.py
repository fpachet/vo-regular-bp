"""Continuator-style BP over a stack of fixed-order context graphs.

This module intentionally has different semantics from the explicit backoff
mixture graph in :mod:`vo_regular_bp.context`.  It runs backward messages on
one graph per order and samples with an order-selection policy.  That makes it
useful for comparisons with classic Continuator-style generation without
muddying the single-graph exact BP implementation.
"""

from __future__ import annotations

import bisect
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
import time
from typing import Hashable, Protocol, TypeAlias

from .acceptors import (
    DFA,
    has_custom_transition_weights,
    transition_weight as regular_transition_weight,
)
from .context import Context, Symbol, _as_context
from .minimization import minimize_fixed_order_graph, minimize_fixed_order_graphs
from .positional_bp import (
    AllowedForbiddenSymbols,
    PositionConstraint,
    PositionConstraints,
    _coerce_rng,
)


@dataclass(frozen=True)
class StackEdge:
    src: int
    dst: int
    symbol: Symbol
    probability: float
    order: int


_RegularTransition: TypeAlias = tuple[StackEdge, Hashable, float]


_TRANSITION_CACHE_MISSING = object()
_TRANSITION_REJECTED = object()


@dataclass(frozen=True)
class _PaddedMelodyConstraintSpec:
    component_acceptors: tuple[DFA, ...]
    duration_index: int
    final_note_index: int | None
    min_note_index: int | None
    target: int
    min_notes: int
    pad_symbol: Symbol
    symbol_costs: Mapping[Symbol, int]
    symbol_is_note: Mapping[Symbol, bool]
    additive_duration: bool
    cost_note_options: tuple[tuple[int, bool], ...]

    @property
    def requires_final_note(self) -> bool:
        return self.final_note_index is not None

    def parts(self, acceptor_state: Hashable) -> tuple[Hashable, ...] | None:
        if len(self.component_acceptors) == 1:
            return (acceptor_state,)
        if not isinstance(acceptor_state, tuple):
            return None
        if len(acceptor_state) != len(self.component_acceptors):
            return None
        return acceptor_state

    def compose(self, parts: Sequence[Hashable]) -> Hashable:
        if len(self.component_acceptors) == 1:
            return parts[0]
        return tuple(parts)


@dataclass(frozen=True)
class _PaddedMelodyState:
    parts: tuple[Hashable, ...]
    total: int
    ended: bool
    final_note: bool
    note_count: int


@dataclass(frozen=True)
class DurationViewQuotientOrderStats:
    """Exact duration-view quotient diagnostics for one fixed-order graph."""

    order: int
    states: int
    classes: int
    edges: int
    projected_edges: int
    ignored_edges: int
    quotient_edges: int
    category_count: int
    max_class_size: int
    refinement_rounds: int

    @property
    def state_reduction(self) -> float:
        return _ratio(self.states, self.classes)

    @property
    def projected_edge_reduction(self) -> float:
        return _ratio(self.projected_edges, self.quotient_edges)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "order": self.order,
            "states": self.states,
            "classes": self.classes,
            "edges": self.edges,
            "projected_edges": self.projected_edges,
            "ignored_edges": self.ignored_edges,
            "quotient_edges": self.quotient_edges,
            "category_count": self.category_count,
            "max_class_size": self.max_class_size,
            "refinement_rounds": self.refinement_rounds,
            "state_reduction": self.state_reduction,
            "projected_edge_reduction": self.projected_edge_reduction,
        }


@dataclass(frozen=True)
class DurationViewQuotientDiagnostics:
    """Exact quotient-compression diagnostics for additive duration views."""

    orders: tuple[DurationViewQuotientOrderStats, ...]
    seconds: float

    @property
    def states(self) -> int:
        return sum(stats.states for stats in self.orders)

    @property
    def classes(self) -> int:
        return sum(stats.classes for stats in self.orders)

    @property
    def edges(self) -> int:
        return sum(stats.edges for stats in self.orders)

    @property
    def projected_edges(self) -> int:
        return sum(stats.projected_edges for stats in self.orders)

    @property
    def ignored_edges(self) -> int:
        return sum(stats.ignored_edges for stats in self.orders)

    @property
    def quotient_edges(self) -> int:
        return sum(stats.quotient_edges for stats in self.orders)

    @property
    def max_class_size(self) -> int:
        return max((stats.max_class_size for stats in self.orders), default=0)

    @property
    def refinement_rounds(self) -> int:
        return max((stats.refinement_rounds for stats in self.orders), default=0)

    @property
    def state_reduction(self) -> float:
        return _ratio(self.states, self.classes)

    @property
    def projected_edge_reduction(self) -> float:
        return _ratio(self.projected_edges, self.quotient_edges)

    def as_dict(self) -> dict[str, object]:
        return {
            "states": self.states,
            "classes": self.classes,
            "edges": self.edges,
            "projected_edges": self.projected_edges,
            "ignored_edges": self.ignored_edges,
            "quotient_edges": self.quotient_edges,
            "max_class_size": self.max_class_size,
            "refinement_rounds": self.refinement_rounds,
            "state_reduction": self.state_reduction,
            "projected_edge_reduction": self.projected_edge_reduction,
            "seconds": self.seconds,
            "orders": tuple(stats.as_dict() for stats in self.orders),
        }


@dataclass(frozen=True)
class OrderCandidateSet:
    order: int
    graph: "FixedOrderContextGraph"
    state: int
    edges: tuple[StackEdge, ...]
    weights: tuple[float, ...]
    cumulative_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.weights):
            raise ValueError("edges and weights must have the same length")
        if self.cumulative_weights is None:
            object.__setattr__(self, "cumulative_weights", _cumulative_weights(self.weights))
        elif len(self.cumulative_weights) != len(self.weights):
            raise ValueError("cumulative_weights and weights must have the same length")


@dataclass(frozen=True)
class PolicyDecision:
    policy_name: str
    candidate_orders: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    selected_order: int | None = None
    selected_symbol: Symbol | None = None
    skipped_orders: tuple[int, ...] = ()
    skipped_symbol: Symbol | None = None
    accepted_singleton: bool = False
    suppressed_skipped_symbol: bool = False


@dataclass(frozen=True)
class CandidateChoice:
    candidate_set: OrderCandidateSet
    edge: StackEdge
    decision: PolicyDecision


class OrderPolicy(Protocol):
    def choose(
        self,
        candidate_sets: list[OrderCandidateSet],
        rng: random.Random,
    ) -> CandidateChoice | None:
        ...


@dataclass(frozen=True)
class LongestFeasiblePolicy:
    policy_name: str = "longest_feasible"

    def choose(
        self,
        candidate_sets: list[OrderCandidateSet],
        rng: random.Random,
    ) -> CandidateChoice | None:
        for candidate_set in candidate_sets:
            edge = _sample_from_weighted_edges(
                candidate_set.edges,
                candidate_set.weights,
                candidate_set.cumulative_weights,
                rng,
            )
            return CandidateChoice(
                candidate_set,
                edge,
                PolicyDecision(
                    policy_name=self.policy_name,
                    candidate_orders=tuple(candidate.order for candidate in candidate_sets),
                    candidate_counts=tuple(len(candidate.edges) for candidate in candidate_sets),
                    selected_order=candidate_set.order,
                    selected_symbol=edge.symbol,
                ),
            )
        return None


@dataclass(frozen=True)
class SingletonAvoidingBackoffPolicy:
    """Classic Continuator-style singleton avoidance."""

    acceptance_probability: float | None = None
    min_singleton_order: int = 2
    suppress_skipped_symbol: bool = True
    policy_name: str = "singleton_avoiding_backoff"

    def __post_init__(self) -> None:
        if self.acceptance_probability is not None:
            if self.acceptance_probability < 0.0 or self.acceptance_probability > 1.0:
                raise ValueError("acceptance_probability must be between 0 and 1")
        if self.min_singleton_order < 1:
            raise ValueError("min_singleton_order must be at least 1")

    def singleton_acceptance_probability(self, order: int) -> float:
        if self.acceptance_probability is not None:
            return self.acceptance_probability
        return 1.0 / (order + 1)

    def choose(
        self,
        candidate_sets: list[OrderCandidateSet],
        rng: random.Random,
    ) -> CandidateChoice | None:
        skipped_symbol = None
        skipped_orders: list[int] = []
        suppressed = False
        candidate_orders = tuple(candidate.order for candidate in candidate_sets)
        candidate_counts = tuple(len(candidate.edges) for candidate in candidate_sets)

        for candidate_set in candidate_sets:
            edges = candidate_set.edges
            weights = candidate_set.weights
            cumulative_weights = candidate_set.cumulative_weights

            if (
                self.suppress_skipped_symbol
                and skipped_symbol is not None
                and candidate_set.order >= self.min_singleton_order
            ):
                filtered = [
                    (edge, weight)
                    for edge, weight in zip(edges, weights)
                    if edge.symbol != skipped_symbol
                ]
                if not filtered:
                    suppressed = True
                    continue
                if len(filtered) < len(edges):
                    suppressed = True
                edges = tuple(edge for edge, _ in filtered)
                weights = tuple(weight for _, weight in filtered)
                cumulative_weights = None

            if len(edges) == 1 and candidate_set.order >= self.min_singleton_order:
                if rng.random() > self.singleton_acceptance_probability(candidate_set.order):
                    skipped_symbol = edges[0].symbol
                    skipped_orders.append(candidate_set.order)
                    continue
                skipped_symbol = None

            edge = _sample_from_weighted_edges(edges, weights, cumulative_weights, rng)
            return CandidateChoice(
                candidate_set,
                edge,
                PolicyDecision(
                    policy_name=self.policy_name,
                    candidate_orders=candidate_orders,
                    candidate_counts=candidate_counts,
                    selected_order=candidate_set.order,
                    selected_symbol=edge.symbol,
                    skipped_orders=tuple(skipped_orders),
                    skipped_symbol=skipped_symbol,
                    accepted_singleton=len(edges) == 1 and candidate_set.order >= self.min_singleton_order,
                    suppressed_skipped_symbol=suppressed,
                ),
            )
        return None


@dataclass(frozen=True)
class OrderSampleStep:
    position: int
    symbol: Symbol
    order: int
    context: Context
    policy: str
    candidate_orders: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    skipped_orders: tuple[int, ...] = ()
    skipped_symbol: Symbol | None = None
    accepted_singleton: bool = False
    suppressed_skipped_symbol: bool = False


class OrderStackModel:
    """Continuation counts for orders 1..K."""

    def __init__(
        self,
        counts: Mapping[Context, Counter[Symbol]],
        *,
        max_order: int,
        forbidden_symbols: Iterable[Symbol] = (),
    ) -> None:
        if max_order < 1:
            raise ValueError("max_order must be at least 1")
        self.counts = dict(counts)
        self.max_order = int(max_order)
        self.forbidden_symbols = frozenset(forbidden_symbols)
        self.alphabet = frozenset(symbol for counter in self.counts.values() for symbol in counter)
        self._graph_cache: dict[int, FixedOrderContextGraph] = {}
        self._minimized_graph_cache: dict[int, FixedOrderContextGraph] = {}

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[Sequence[Symbol]],
        *,
        max_order: int,
        start_symbol: Symbol | None = None,
        end_symbol: Symbol | None = None,
    ) -> "OrderStackModel":
        counts: dict[Context, Counter[Symbol]] = {}
        forbidden_symbols = {symbol for symbol in (start_symbol, end_symbol) if symbol is not None}
        for sequence in sequences:
            material = tuple(sequence)
            tokens = material
            if start_symbol is not None:
                tokens = (start_symbol,) + tokens
            if end_symbol is not None:
                tokens = tokens + (end_symbol,)
            for index in range(1, len(tokens)):
                symbol = tokens[index]
                order_limit = min(max_order, index)
                for order in range(1, order_limit + 1):
                    context = tokens[index - order : index]
                    counts.setdefault(context, Counter())[symbol] += 1
            if end_symbol is not None:
                counts.setdefault((end_symbol,), Counter())[end_symbol] += 1
        return cls(counts, max_order=max_order, forbidden_symbols=forbidden_symbols)

    def longest_available_suffix(
        self,
        context: Iterable[Symbol] | Context,
        *,
        max_order: int | None = None,
    ) -> Context | None:
        state = _as_context(context)
        order_limit = min(self.max_order if max_order is None else max_order, len(state))
        for order in range(order_limit, 0, -1):
            suffix = state[-order:]
            if suffix in self.counts:
                return suffix
        return None

    def continuation_distribution_with_order(
        self,
        context: Iterable[Symbol] | Context,
        *,
        max_order: int | None = None,
    ) -> tuple[tuple[tuple[Symbol, float], ...], int | None]:
        suffix = self.longest_available_suffix(context, max_order=max_order)
        if suffix is None:
            return (), None
        counts = self.counts[suffix]
        total = float(sum(counts.values()))
        if total <= 0.0:
            return (), None
        return tuple((symbol, float(count) / total) for symbol, count in counts.items()), len(suffix)

    def iter_contexts(self, order: int) -> Iterable[Context]:
        for context in self.counts:
            if len(context) <= order:
                yield context

    def compile_graph(self, order: int) -> "FixedOrderContextGraph":
        cached = self._graph_cache.get(order)
        if cached is not None:
            return cached
        graph = FixedOrderContextGraph.from_model(self, order=order)
        self._graph_cache[order] = graph
        return graph

    def compile_minimized_graph(self, order: int) -> "FixedOrderContextGraph":
        cached = self._minimized_graph_cache.get(order)
        if cached is not None:
            return cached
        graph = minimize_fixed_order_graph(self.compile_graph(order))
        self._minimized_graph_cache[order] = graph
        return graph


class FixedOrderContextGraph:
    def __init__(self, order: int) -> None:
        self.order = int(order)
        self.contexts: list[Context] = []
        self.context_to_id: dict[Context, int] = {}
        self.outgoing: list[list[StackEdge]] = []
        self.state_aliases: tuple[tuple[Context, ...], ...] = ()
        self.quotient_stats: object | None = None

    @classmethod
    def from_model(cls, model: OrderStackModel, *, order: int) -> "FixedOrderContextGraph":
        if order < 1 or order > model.max_order:
            raise ValueError(f"order must be between 1 and {model.max_order}")

        graph = cls(order)
        queue: deque[Context] = deque()

        def add_context(context: Iterable[Symbol] | Context) -> int:
            state = graph.truncate_context(context)
            found = graph.context_to_id.get(state)
            if found is not None:
                return found
            context_id = len(graph.contexts)
            graph.context_to_id[state] = context_id
            graph.contexts.append(state)
            graph.outgoing.append([])
            queue.append(state)
            return context_id

        iter_contexts = getattr(model, "iter_contexts", None)
        if iter_contexts is None:
            for context in model.counts:
                if len(context) <= order:
                    add_context(context)
        else:
            for context in iter_contexts(order):
                add_context(context)

        while queue:
            context = queue.popleft()
            src = graph.context_to_id[context]
            distribution, effective_order = model.continuation_distribution_with_order(
                context,
                max_order=order,
            )
            for symbol, probability in distribution:
                dst_context = graph.next_context(context, symbol)
                dst = add_context(dst_context)
                graph.outgoing[src].append(
                    StackEdge(
                        src=src,
                        dst=dst,
                        symbol=symbol,
                        probability=float(probability),
                        order=effective_order or 0,
                    )
                )

        return graph

    @classmethod
    def from_model_contexts(
        cls,
        model: OrderStackModel,
        *,
        order: int,
        contexts: Iterable[Context],
        expandable_contexts: Iterable[Context] | None = None,
    ) -> "FixedOrderContextGraph":
        """Build a graph over a caller-supplied finite context set."""

        if order < 1 or order > model.max_order:
            raise ValueError(f"order must be between 1 and {model.max_order}")

        graph = cls(order)

        def add_context(context: Iterable[Symbol] | Context) -> int:
            state = graph.truncate_context(context)
            found = graph.context_to_id.get(state)
            if found is not None:
                return found
            context_id = len(graph.contexts)
            graph.context_to_id[state] = context_id
            graph.contexts.append(state)
            graph.outgoing.append([])
            return context_id

        def sort_key(context: Context) -> tuple[int, str]:
            return (len(context), repr(context))

        context_set = {graph.truncate_context(context) for context in contexts}
        expandable_set = (
            context_set
            if expandable_contexts is None
            else {graph.truncate_context(context) for context in expandable_contexts}
        )
        for context in sorted(context_set, key=sort_key):
            add_context(context)

        for context in sorted(expandable_set, key=sort_key):
            src = add_context(context)
            distribution, effective_order = model.continuation_distribution_with_order(
                context,
                max_order=order,
            )
            for symbol, probability in distribution:
                dst_context = graph.next_context(context, symbol)
                dst = add_context(dst_context)
                graph.outgoing[src].append(
                    StackEdge(
                        src=src,
                        dst=dst,
                        symbol=symbol,
                        probability=float(probability),
                        order=effective_order or 0,
                    )
                )

        return graph

    def truncate_context(self, context: Iterable[Symbol] | Context) -> Context:
        state = _as_context(context)
        if len(state) <= self.order:
            return state
        return state[-self.order:]

    def next_context(self, context: Iterable[Symbol] | Context, symbol: Symbol) -> Context:
        return self.truncate_context(_as_context(context) + (symbol,))

    def state_id(self, context: Iterable[Symbol] | Context) -> int | None:
        return self.context_to_id.get(self.truncate_context(context))

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.outgoing)


@dataclass
class OrderStackBPResult:
    model: OrderStackModel
    length: int
    prefix: Context
    constraints: PositionConstraints
    allowed_forbidden_symbols: Mapping[int, frozenset[Symbol]]
    graphs: dict[int, FixedOrderContextGraph]
    backwards: dict[int, list[list[float]]]
    policy: OrderPolicy
    _candidate_set_cache: dict[tuple[int, Context], tuple[OrderCandidateSet, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def context_state_count(self) -> int:
        return sum(len(graph.contexts) for graph in self.graphs.values())

    @property
    def context_edge_count(self) -> int:
        return sum(graph.edge_count for graph in self.graphs.values())

    @property
    def bp_edge_relaxation_upper_bound(self) -> int:
        return self.length * self.context_edge_count

    @property
    def success_mass(self) -> float:
        return 1.0 if self._candidate_sets(0, self.prefix) else 0.0

    def start_order_masses(self) -> tuple[tuple[int, float], ...]:
        masses: list[tuple[int, float]] = []
        max_order = min(self.model.max_order, len(self.prefix))
        for order in range(1, max_order + 1):
            graph = self.graphs[order]
            state = graph.state_id(tuple(self.prefix[-order:]))
            mass = self.backwards[order][0][state] if state is not None else 0.0
            masses.append((order, mass))
        return tuple(masses)

    def sample_with_trace(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> tuple[tuple[Symbol, ...], tuple[OrderSampleStep, ...]]:
        generator = _coerce_rng(rng)
        history = list(self.prefix)
        output: list[Symbol] = []
        trace: list[OrderSampleStep] = []

        for position in range(self.length):
            candidate_sets = self._candidate_sets(position, history)
            choice = self.policy.choose(candidate_sets, generator)
            if choice is None:
                raise ValueError("No context path satisfies the constraints at any order.")
            edge = choice.edge
            trace_context = tuple(history[-choice.candidate_set.order :])
            output.append(edge.symbol)
            history.append(edge.symbol)
            trace.append(_sample_step(position, choice, context=trace_context))

        return tuple(output), tuple(trace)

    def sample_with_orders(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> tuple[tuple[Symbol, ...], tuple[int, ...]]:
        generator = _coerce_rng(rng)
        history = list(self.prefix)
        output: list[Symbol] = []
        orders: list[int] = []

        for position in range(self.length):
            candidate_sets = self._candidate_sets(position, history)
            choice = self.policy.choose(candidate_sets, generator)
            if choice is None:
                raise ValueError("No context path satisfies the constraints at any order.")
            edge = choice.edge
            output.append(edge.symbol)
            orders.append(edge.order)
            history.append(edge.symbol)

        return tuple(output), tuple(orders)

    def sample(self, *, rng: random.Random | int | None = None) -> tuple[Symbol, ...]:
        return self.sample_with_orders(rng=rng)[0]

    def sample_many_with_orders(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[tuple[tuple[Symbol, ...], tuple[int, ...]]]:
        generator = _coerce_rng(rng)
        return [self.sample_with_orders(rng=generator) for _ in range(count)]

    def _candidate_sets(self, position: int, history: Sequence[Symbol]) -> list[OrderCandidateSet]:
        history_key = tuple(history[-self.model.max_order :])
        cache_key = (position, history_key)
        cached = self._candidate_set_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        candidate_sets: list[OrderCandidateSet] = []
        max_order = min(self.model.max_order, len(history))
        constraint = self.constraints.get(position)
        allowed_forbidden = self.allowed_forbidden_symbols.get(position, frozenset())
        for order in range(max_order, 0, -1):
            graph = self.graphs[order]
            context = tuple(history[-order:])
            state = graph.state_id(context)
            if state is None:
                continue
            candidates = []
            weights = []
            beta_next = self.backwards[order][position + 1]
            if constraint is None:
                for edge in graph.outgoing[state]:
                    if _is_forbidden(edge.symbol, self.model.forbidden_symbols, allowed_forbidden):
                        continue
                    weight = edge.probability * beta_next[edge.dst]
                    if weight <= 0.0:
                        continue
                    candidates.append(edge)
                    weights.append(weight)
            elif callable(constraint):
                for edge in graph.outgoing[state]:
                    if _is_forbidden(edge.symbol, self.model.forbidden_symbols, allowed_forbidden):
                        continue
                    if not constraint(edge.symbol):
                        continue
                    weight = edge.probability * beta_next[edge.dst]
                    if weight <= 0.0:
                        continue
                    candidates.append(edge)
                    weights.append(weight)
            else:
                for edge in graph.outgoing[state]:
                    if _is_forbidden(edge.symbol, self.model.forbidden_symbols, allowed_forbidden):
                        continue
                    if edge.symbol not in constraint:
                        continue
                    weight = edge.probability * beta_next[edge.dst]
                    if weight <= 0.0:
                        continue
                    candidates.append(edge)
                    weights.append(weight)
            if candidates:
                candidate_sets.append(
                    OrderCandidateSet(
                        order=order,
                        graph=graph,
                        state=state,
                        edges=tuple(candidates),
                        weights=tuple(weights),
                    )
                )
        self._candidate_set_cache[cache_key] = tuple(candidate_sets)
        return candidate_sets


@dataclass(frozen=True)
class OrderStackBPPlan:
    """Prefix-independent positional order-stack BP preparation."""

    model: OrderStackModel
    length: int
    constraints: PositionConstraints
    allowed_forbidden_symbols: Mapping[int, frozenset[Symbol]]
    graphs: dict[int, FixedOrderContextGraph]
    backwards: dict[int, list[list[float]]]
    policy: OrderPolicy

    def for_prefix(self, prefix: Sequence[Symbol]) -> OrderStackBPResult:
        if not prefix:
            raise ValueError("order-stack BP requires a non-empty prefix")
        return OrderStackBPResult(
            model=self.model,
            length=self.length,
            prefix=tuple(prefix),
            constraints=self.constraints,
            allowed_forbidden_symbols=self.allowed_forbidden_symbols,
            graphs=self.graphs,
            backwards=self.backwards,
            policy=self.policy,
        )


def prepare_order_stack_bp(
    model: OrderStackModel,
    *,
    length: int,
    constraints: PositionConstraints | None = None,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    policy: OrderPolicy | None = None,
    minimize_source_graphs: bool = False,
) -> OrderStackBPPlan:
    """Prepare prefix-independent positional order-stack backward messages."""

    if length < 0:
        raise ValueError("length must be non-negative")
    position_constraints = dict(constraints or {})
    _validate_constraint_positions(position_constraints, length)
    allowed_forbidden = _normalize_allowed_forbidden_symbols(
        allowed_forbidden_symbols,
        length,
    )
    active_policy = policy or SingletonAvoidingBackoffPolicy()

    graphs = _compile_materialized_source_graphs(
        model,
        minimize_source_graphs=minimize_source_graphs,
    )
    backwards = {
        order: _backward_messages(
            graph,
            length=length,
            constraints=position_constraints,
            forbidden_symbols=model.forbidden_symbols,
            allowed_forbidden_symbols=allowed_forbidden,
        )
        for order, graph in graphs.items()
    }
    return OrderStackBPPlan(
        model=model,
        length=length,
        constraints=position_constraints,
        allowed_forbidden_symbols=allowed_forbidden,
        graphs=graphs,
        backwards=backwards,
        policy=active_policy,
    )


def run_order_stack_bp(
    model: OrderStackModel,
    *,
    length: int,
    prefix: Sequence[Symbol],
    constraints: PositionConstraints | None = None,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    policy: OrderPolicy | None = None,
    minimize_source_graphs: bool = False,
) -> OrderStackBPResult:
    if not prefix:
        raise ValueError("order-stack BP requires a non-empty prefix")
    return prepare_order_stack_bp(
        model,
        length=length,
        constraints=constraints,
        allowed_forbidden_symbols=allowed_forbidden_symbols,
        policy=policy,
        minimize_source_graphs=minimize_source_graphs,
    ).for_prefix(prefix)


def _backward_messages(
    graph: FixedOrderContextGraph,
    *,
    length: int,
    constraints: Mapping[int, PositionConstraint],
    forbidden_symbols: frozenset[Symbol],
    allowed_forbidden_symbols: Mapping[int, frozenset[Symbol]],
) -> list[list[float]]:
    backward = [[0.0 for _ in graph.contexts] for _ in range(length + 1)]
    for state in range(len(graph.contexts)):
        backward[length][state] = 1.0

    for position in range(length - 1, -1, -1):
        constraint = constraints.get(position)
        allowed_forbidden = allowed_forbidden_symbols.get(position, frozenset())
        beta_next = backward[position + 1]
        beta = backward[position]
        if constraint is None:
            for state, edges in enumerate(graph.outgoing):
                total = 0.0
                for edge in edges:
                    if _is_forbidden(edge.symbol, forbidden_symbols, allowed_forbidden):
                        continue
                    total += edge.probability * beta_next[edge.dst]
                beta[state] = total
        elif callable(constraint):
            for state, edges in enumerate(graph.outgoing):
                total = 0.0
                for edge in edges:
                    if _is_forbidden(edge.symbol, forbidden_symbols, allowed_forbidden):
                        continue
                    if constraint(edge.symbol):
                        total += edge.probability * beta_next[edge.dst]
                beta[state] = total
        else:
            for state, edges in enumerate(graph.outgoing):
                total = 0.0
                for edge in edges:
                    if _is_forbidden(edge.symbol, forbidden_symbols, allowed_forbidden):
                        continue
                    if edge.symbol in constraint:
                        total += edge.probability * beta_next[edge.dst]
                beta[state] = total
    return backward


def _sample_step(
    position: int,
    choice: CandidateChoice,
    *,
    context: Context | None = None,
) -> OrderSampleStep:
    decision = choice.decision
    graph = choice.candidate_set.graph
    trace_context = graph.contexts[choice.candidate_set.state] if context is None else context
    return OrderSampleStep(
        position=position,
        symbol=choice.edge.symbol,
        order=choice.edge.order,
        context=trace_context,
        policy=decision.policy_name,
        candidate_orders=decision.candidate_orders,
        candidate_counts=decision.candidate_counts,
        skipped_orders=decision.skipped_orders,
        skipped_symbol=decision.skipped_symbol,
        accepted_singleton=decision.accepted_singleton,
        suppressed_skipped_symbol=decision.suppressed_skipped_symbol,
    )


def _validate_constraint_positions(
    constraints: Mapping[int, PositionConstraint],
    length: int,
) -> None:
    for position in constraints:
        if position < 0 or position >= length:
            raise IndexError(f"constraint position {position} is outside length {length}")


def _normalize_allowed_forbidden_symbols(
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None,
    length: int,
) -> dict[int, frozenset[Symbol]]:
    allowed: dict[int, frozenset[Symbol]] = {}
    for position, symbols in (allowed_forbidden_symbols or {}).items():
        if position < 0 or position >= length:
            raise IndexError(
                f"allowed forbidden-symbol position {position} is outside length {length}"
            )
        symbol_set = frozenset(symbols)
        if symbol_set:
            allowed[int(position)] = symbol_set
    return allowed


def _is_forbidden(
    symbol: Symbol,
    forbidden_symbols: frozenset[Symbol],
    allowed_forbidden: frozenset[Symbol],
) -> bool:
    return symbol in forbidden_symbols and symbol not in allowed_forbidden


@dataclass
class _RegularBackwardCache:
    graph: FixedOrderContextGraph
    acceptor: DFA
    length: int
    constraints: PositionConstraints = field(default_factory=dict)
    forbidden_symbols: frozenset[Symbol] = field(default_factory=frozenset)
    allowed_forbidden_symbols: Mapping[int, frozenset[Symbol]] = field(default_factory=dict)
    padded_melody_spec: _PaddedMelodyConstraintSpec | None = None
    memo: dict[tuple[int, int, Hashable], float] = field(default_factory=dict)
    transition_rows: dict[tuple[int, Hashable], tuple[_RegularTransition, ...]] = field(default_factory=dict)
    acceptor_symbol_transitions: dict[
        Hashable,
        dict[Symbol, tuple[Hashable, float] | object],
    ] = field(default_factory=dict)
    padded_state_parse_cache: dict[Hashable, _PaddedMelodyState | None] = field(default_factory=dict)
    padded_feasibility_memo: dict[tuple[int, int, bool, bool, int], bool] = field(default_factory=dict)
    padded_options_by_total: dict[int, tuple[tuple[int, bool], ...]] = field(default_factory=dict)
    padded_additive_suffix_tables: tuple[dict[int, tuple[int, int]], ...] | None = None
    padded_edges_by_category: dict[int, dict[tuple[int, bool], tuple[StackEdge, ...]]] = field(default_factory=dict)
    padded_allowed_categories_cache: dict[tuple[int, int, int], tuple[tuple[int, bool], ...]] = field(default_factory=dict)
    expanded_edge_total: int = 0
    beta_cache_hits: int = 0
    beta_cache_misses: int = 0
    transition_row_cache_hits: int = 0
    transition_row_cache_misses: int = 0
    acceptor_symbol_transition_cache_hits: int = 0
    acceptor_symbol_transition_cache_misses: int = 0
    accepted_transition_total: int = 0
    dense_transition_by_symbol: Mapping[Symbol, tuple[int, ...]] | None = field(default=None, init=False)
    weighted_transitions: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        dense = getattr(self.acceptor, "dense_transition_by_symbol", None)
        if dense is not None:
            self.dense_transition_by_symbol = dense
        self.weighted_transitions = has_custom_transition_weights(self.acceptor)

    def beta(self, time: int, state: int, acceptor_state: Hashable) -> float:
        if self.padded_melody_spec is not None:
            return self._padded_melody_beta(time, state, acceptor_state)

        key = (time, state, acceptor_state)
        cached = self.memo.get(key)
        if cached is not None:
            self.beta_cache_hits += 1
            return cached

        self.beta_cache_misses += 1
        if time == self.length:
            value = 1.0 if self.acceptor.is_accepting(acceptor_state) else 0.0
            self.memo[key] = value
            return value

        total = 0.0
        edge_count = 0
        constraint = self.constraints.get(time)
        allowed_forbidden = self.allowed_forbidden_symbols.get(time, frozenset())
        row = self.accepted_transitions(state, acceptor_state)
        next_time = time + 1
        memo = self.memo
        memo_get = memo.get
        beta = self.beta
        beta_cache_hits = 0
        forbidden_symbols = self.forbidden_symbols
        if constraint is None:
            for edge, next_acceptor_state, transition_weight in row:
                if _is_forbidden(edge.symbol, forbidden_symbols, allowed_forbidden):
                    continue
                edge_count += 1
                next_beta = memo_get((next_time, edge.dst, next_acceptor_state))
                if next_beta is None:
                    next_beta = beta(next_time, edge.dst, next_acceptor_state)
                else:
                    beta_cache_hits += 1
                total += (
                    edge.probability
                    * transition_weight
                    * next_beta
                )
        elif callable(constraint):
            for edge, next_acceptor_state, transition_weight in row:
                if _is_forbidden(edge.symbol, forbidden_symbols, allowed_forbidden):
                    continue
                if not constraint(edge.symbol):
                    continue
                edge_count += 1
                next_beta = memo_get((next_time, edge.dst, next_acceptor_state))
                if next_beta is None:
                    next_beta = beta(next_time, edge.dst, next_acceptor_state)
                else:
                    beta_cache_hits += 1
                total += (
                    edge.probability
                    * transition_weight
                    * next_beta
                )
        else:
            for edge, next_acceptor_state, transition_weight in row:
                if _is_forbidden(edge.symbol, forbidden_symbols, allowed_forbidden):
                    continue
                if edge.symbol not in constraint:
                    continue
                edge_count += 1
                next_beta = memo_get((next_time, edge.dst, next_acceptor_state))
                if next_beta is None:
                    next_beta = beta(next_time, edge.dst, next_acceptor_state)
                else:
                    beta_cache_hits += 1
                total += (
                    edge.probability
                    * transition_weight
                    * next_beta
                )
        self.beta_cache_hits += beta_cache_hits
        self.expanded_edge_total += edge_count
        self.memo[key] = total
        return total

    def accepted_transitions(
        self,
        state: int,
        acceptor_state: Hashable,
    ) -> tuple[_RegularTransition, ...]:
        if self.padded_melody_spec is not None:
            return self._padded_melody_accepted_transitions(state, acceptor_state)

        key = (state, acceptor_state)
        cached = self.transition_rows.get(key)
        if cached is not None:
            self.transition_row_cache_hits += 1
            return cached

        self.transition_row_cache_misses += 1
        dense_transition_by_symbol = self.dense_transition_by_symbol
        row: list[_RegularTransition] = []
        if dense_transition_by_symbol is not None and isinstance(acceptor_state, int):
            for edge in self.graph.outgoing[state]:
                transitions = dense_transition_by_symbol.get(edge.symbol)
                if transitions is None:
                    continue
                next_state = transitions[acceptor_state]
                if next_state < 0:
                    continue
                dfa_weight = 1.0
                if self.weighted_transitions:
                    dfa_weight = regular_transition_weight(
                        self.acceptor,
                        acceptor_state,
                        edge.symbol,
                    )
                    if dfa_weight <= 0.0:
                        continue
                row.append((edge, next_state, dfa_weight))
        else:
            symbol_transitions = self.acceptor_symbol_transitions.get(acceptor_state)
            if symbol_transitions is None:
                symbol_transitions = {}
                self.acceptor_symbol_transitions[acceptor_state] = symbol_transitions
            symbol_get = symbol_transitions.get
            missing = _TRANSITION_CACHE_MISSING
            rejected = _TRANSITION_REJECTED
            transition_func = getattr(self.acceptor, "_transition_func", None)
            acceptor_next_state = self.acceptor.next_state
            if (
                transition_func is not None
                and getattr(type(self.acceptor), "next_state", None) is DFA.next_state
            ):
                acceptor_next_state = transition_func
            weighted_transitions = self.weighted_transitions
            transition_weight_func = getattr(self.acceptor, "_transition_weight_func", None)
            transition_weight_map = getattr(self.acceptor, "_transition_weights", None)
            can_use_direct_weight = (
                getattr(type(self.acceptor), "transition_weight", None)
                is DFA.transition_weight
            )
            row_append = row.append
            symbol_hits = 0
            symbol_misses = 0
            for edge in self.graph.outgoing[state]:
                symbol = edge.symbol
                cached_symbol_transition = symbol_get(symbol, missing)
                if cached_symbol_transition is missing:
                    symbol_misses += 1
                    next_acceptor_state = acceptor_next_state(acceptor_state, symbol)
                    if next_acceptor_state is None:
                        symbol_transitions[symbol] = rejected
                        continue
                    dfa_weight = 1.0
                    if weighted_transitions:
                        if can_use_direct_weight and transition_weight_func is not None:
                            dfa_weight = float(transition_weight_func(acceptor_state, symbol))
                            if not math.isfinite(dfa_weight) or dfa_weight < 0.0:
                                raise ValueError(
                                    f"transition weight for state {acceptor_state!r} "
                                    f"and symbol {symbol!r} must be a finite "
                                    f"nonnegative number, got {dfa_weight!r}"
                                )
                        elif can_use_direct_weight and transition_weight_map:
                            dfa_weight = float(
                                transition_weight_map.get(acceptor_state, {}).get(
                                    symbol,
                                    1.0,
                                )
                            )
                            if not math.isfinite(dfa_weight) or dfa_weight < 0.0:
                                raise ValueError(
                                    f"transition weight for state {acceptor_state!r} "
                                    f"and symbol {symbol!r} must be a finite "
                                    f"nonnegative number, got {dfa_weight!r}"
                                )
                        else:
                            dfa_weight = regular_transition_weight(
                                self.acceptor,
                                acceptor_state,
                                symbol,
                            )
                        if dfa_weight <= 0.0:
                            symbol_transitions[symbol] = rejected
                            continue
                    symbol_transitions[symbol] = (next_acceptor_state, dfa_weight)
                else:
                    symbol_hits += 1
                    if cached_symbol_transition is rejected:
                        continue
                    next_acceptor_state, dfa_weight = cached_symbol_transition  # type: ignore[misc]
                row_append((edge, next_acceptor_state, dfa_weight))
            self.acceptor_symbol_transition_cache_hits += symbol_hits
            self.acceptor_symbol_transition_cache_misses += symbol_misses

        result = tuple(row)
        self.accepted_transition_total += len(result)
        self.transition_rows[key] = result
        return result

    def next_acceptor_state(self, acceptor_state: Hashable, symbol: Symbol) -> Hashable | None:
        if self.padded_melody_spec is not None:
            return self._padded_melody_next_acceptor_state(acceptor_state, symbol)

        dense_transition_by_symbol = self.dense_transition_by_symbol
        if dense_transition_by_symbol is not None and isinstance(acceptor_state, int):
            transitions = dense_transition_by_symbol.get(symbol)
            if transitions is None:
                return None
            next_state = transitions[acceptor_state]
            if next_state < 0:
                return None
            return next_state
        return self.acceptor.next_state(acceptor_state, symbol)

    def _padded_melody_beta(
        self,
        time: int,
        state: int,
        acceptor_state: Hashable,
    ) -> float:
        parsed = self._padded_melody_state(acceptor_state)
        code = self._padded_melody_state_code(parsed)
        key = (time, state, code)
        cached = self.memo.get(key)
        if cached is not None:
            self.beta_cache_hits += 1
            return cached

        self.beta_cache_misses += 1
        remaining = self.length - time
        if parsed is None or not self._padded_melody_suffix_feasible_state(remaining, parsed):
            self.memo[key] = 0.0
            return 0.0

        if time == self.length:
            value = 1.0 if self.acceptor.is_accepting(acceptor_state) else 0.0
            self.memo[key] = value
            return value

        spec = self.padded_melody_spec
        if spec is not None and (parsed.ended or parsed.total == spec.target):
            value = self._padded_melody_forced_pad_suffix(time, state, acceptor_state)
            self.memo[key] = value
            return value

        total = 0.0
        edge_count = 0
        constraint = self.constraints.get(time)
        allowed_forbidden = self.allowed_forbidden_symbols.get(time, frozenset())
        next_time = time + 1
        next_remaining = self.length - next_time
        memo = self.memo
        memo_get = memo.get
        beta = self.beta
        beta_cache_hits = 0
        forbidden_symbols = self.forbidden_symbols
        if spec is not None and spec.additive_duration:
            for cost, is_note in self._padded_melody_allowed_next_categories(
                next_remaining,
                parsed.total,
                parsed.note_count,
            ):
                next_total = parsed.total + cost
                next_count = min(spec.min_notes, parsed.note_count + (1 if is_note else 0))
                next_acceptor_state = self._padded_melody_compose_state(
                    parsed,
                    total=next_total,
                    ended=False,
                    final_note=is_note,
                    note_count=next_count,
                )
                next_code = self._padded_melody_state_code(
                    self._padded_melody_state(next_acceptor_state)
                )
                for edge in self._padded_melody_edges_by_category(state).get(
                    (cost, is_note),
                    (),
                ):
                    symbol = edge.symbol
                    if _is_forbidden(symbol, forbidden_symbols, allowed_forbidden):
                        continue
                    if constraint is not None:
                        if callable(constraint):
                            if not constraint(symbol):
                                continue
                        elif symbol not in constraint:
                            continue
                    edge_count += 1
                    next_beta = memo_get((next_time, edge.dst, next_code))
                    if next_beta is None:
                        next_beta = beta(next_time, edge.dst, next_acceptor_state)
                    else:
                        beta_cache_hits += 1
                    total += edge.probability * next_beta
        else:
            row = self.accepted_transitions(state, acceptor_state)
            for edge, next_acceptor_state, transition_weight in row:
                symbol = edge.symbol
                if _is_forbidden(symbol, forbidden_symbols, allowed_forbidden):
                    continue
                if constraint is not None:
                    if callable(constraint):
                        if not constraint(symbol):
                            continue
                    elif symbol not in constraint:
                        continue
                next_parsed = self._padded_melody_state(next_acceptor_state)
                if next_parsed is None or not self._padded_melody_suffix_feasible_state(
                    next_remaining,
                    next_parsed,
                ):
                    continue
                edge_count += 1
                next_code = self._padded_melody_state_code(next_parsed)
                next_beta = memo_get((next_time, edge.dst, next_code))
                if next_beta is None:
                    next_beta = beta(next_time, edge.dst, next_acceptor_state)
                else:
                    beta_cache_hits += 1
                total += edge.probability * transition_weight * next_beta
        self.beta_cache_hits += beta_cache_hits
        self.expanded_edge_total += edge_count
        self.memo[key] = total
        return total

    def _padded_melody_accepted_transitions(
        self,
        state: int,
        acceptor_state: Hashable,
    ) -> tuple[_RegularTransition, ...]:
        parsed = self._padded_melody_state(acceptor_state)
        key = (state, self._padded_melody_state_code(parsed))
        cached = self.transition_rows.get(key)
        if cached is not None:
            self.transition_row_cache_hits += 1
            return cached

        self.transition_row_cache_misses += 1
        row: list[_RegularTransition] = []
        row_append = row.append
        for edge in self.graph.outgoing[state]:
            next_acceptor_state = self._padded_melody_next_acceptor_state(
                acceptor_state,
                edge.symbol,
            )
            if next_acceptor_state is None:
                continue
            row_append((edge, next_acceptor_state, 1.0))

        result = tuple(row)
        self.accepted_transition_total += len(result)
        self.transition_rows[key] = result
        return result

    def _padded_melody_state_code(
        self,
        state: _PaddedMelodyState | None,
    ) -> int:
        if state is None:
            return -1
        spec = self.padded_melody_spec
        min_notes = 0 if spec is None else spec.min_notes
        note_count = min(min_notes, state.note_count)
        return (
            ((((state.total * 2) + int(state.ended)) * 2 + int(state.final_note))
            * (min_notes + 1))
            + note_count
        )

    def _padded_melody_compose_state(
        self,
        current: _PaddedMelodyState,
        *,
        total: int,
        ended: bool,
        final_note: bool,
        note_count: int,
    ) -> Hashable:
        spec = self.padded_melody_spec
        if spec is None:
            return current.parts
        parts = list(current.parts)
        parts[spec.duration_index] = (total, ended)
        if spec.final_note_index is not None:
            parts[spec.final_note_index] = (final_note, ended)
        if spec.min_note_index is not None:
            parts[spec.min_note_index] = (min(spec.min_notes, note_count), ended)
        return spec.compose(parts)

    def _padded_melody_edges_by_category(
        self,
        state: int,
    ) -> dict[tuple[int, bool], tuple[StackEdge, ...]]:
        cached = self.padded_edges_by_category.get(state)
        if cached is not None:
            return cached
        spec = self.padded_melody_spec
        grouped: dict[tuple[int, bool], list[StackEdge]] = {}
        if spec is not None:
            for edge in self.graph.outgoing[state]:
                if edge.symbol == spec.pad_symbol:
                    continue
                cost = spec.symbol_costs.get(edge.symbol)
                is_note = spec.symbol_is_note.get(edge.symbol)
                if cost is None or is_note is None or cost <= 0:
                    continue
                grouped.setdefault((cost, bool(is_note)), []).append(edge)
        result = {category: tuple(edges) for category, edges in grouped.items()}
        self.padded_edges_by_category[state] = result
        return result

    def _padded_melody_allowed_next_categories(
        self,
        next_remaining: int,
        total: int,
        note_count: int,
    ) -> tuple[tuple[int, bool], ...]:
        spec = self.padded_melody_spec
        if spec is None:
            return ()
        key = (next_remaining, total, min(spec.min_notes, note_count))
        cached = self.padded_allowed_categories_cache.get(key)
        if cached is not None:
            return cached
        categories: list[tuple[int, bool]] = []
        for cost, is_note in spec.cost_note_options:
            next_total = total + cost
            if next_total > spec.target:
                continue
            next_count = min(spec.min_notes, note_count + (1 if is_note else 0))
            if self._padded_melody_suffix_feasible(
                next_remaining,
                next_total,
                False,
                is_note,
                next_count,
            ):
                categories.append((cost, is_note))
        result = tuple(categories)
        self.padded_allowed_categories_cache[key] = result
        return result

    def _padded_melody_next_acceptor_state(
        self,
        acceptor_state: Hashable,
        symbol: Symbol,
    ) -> Hashable | None:
        symbol_transitions = self.acceptor_symbol_transitions.get(acceptor_state)
        if symbol_transitions is None:
            symbol_transitions = {}
            self.acceptor_symbol_transitions[acceptor_state] = symbol_transitions
        cached_symbol_transition = symbol_transitions.get(symbol, _TRANSITION_CACHE_MISSING)
        if cached_symbol_transition is not _TRANSITION_CACHE_MISSING:
            self.acceptor_symbol_transition_cache_hits += 1
            if cached_symbol_transition is _TRANSITION_REJECTED:
                return None
            return cached_symbol_transition  # type: ignore[return-value]
        self.acceptor_symbol_transition_cache_misses += 1

        spec = self.padded_melody_spec
        if spec is None:
            symbol_transitions[symbol] = _TRANSITION_REJECTED
            return None
        parsed = self._padded_melody_state(acceptor_state)
        if parsed is None:
            symbol_transitions[symbol] = _TRANSITION_REJECTED
            return None
        if parsed.ended and symbol != spec.pad_symbol:
            symbol_transitions[symbol] = _TRANSITION_REJECTED
            return None

        parts = list(parsed.parts)
        duration_state = parsed.parts[spec.duration_index]
        duration_acceptor = spec.component_acceptors[spec.duration_index]
        if symbol == spec.pad_symbol:
            if parsed.total != spec.target:
                symbol_transitions[symbol] = _TRANSITION_REJECTED
                return None
            duration_next = duration_acceptor.next_state(duration_state, symbol)
            if duration_next is None:
                symbol_transitions[symbol] = _TRANSITION_REJECTED
                return None
            parts[spec.duration_index] = duration_next
            if spec.final_note_index is not None:
                parts[spec.final_note_index] = (parsed.final_note, True)
            if spec.min_note_index is not None:
                parts[spec.min_note_index] = (parsed.note_count, True)
            next_state = spec.compose(parts)
            symbol_transitions[symbol] = next_state
            return next_state

        if parsed.total == spec.target:
            symbol_transitions[symbol] = _TRANSITION_REJECTED
            return None
        if symbol not in spec.symbol_costs or symbol not in spec.symbol_is_note:
            symbol_transitions[symbol] = _TRANSITION_REJECTED
            return None
        duration_next = duration_acceptor.next_state(duration_state, symbol)
        if duration_next is None:
            symbol_transitions[symbol] = _TRANSITION_REJECTED
            return None
        if not (
            isinstance(duration_next, tuple)
            and len(duration_next) == 2
            and isinstance(duration_next[0], int)
            and isinstance(duration_next[1], bool)
        ):
            symbol_transitions[symbol] = _TRANSITION_REJECTED
            return None
        parts[spec.duration_index] = duration_next

        is_note = bool(spec.symbol_is_note[symbol])
        if spec.final_note_index is not None:
            parts[spec.final_note_index] = (is_note, False)
        if spec.min_note_index is not None:
            parts[spec.min_note_index] = (
                min(spec.min_notes, parsed.note_count + (1 if is_note else 0)),
                False,
            )
        next_state = spec.compose(parts)
        symbol_transitions[symbol] = next_state
        return next_state

    def _padded_melody_state(
        self,
        acceptor_state: Hashable,
    ) -> _PaddedMelodyState | None:
        spec = self.padded_melody_spec
        if spec is None:
            return None
        if acceptor_state in self.padded_state_parse_cache:
            return self.padded_state_parse_cache[acceptor_state]
        parts = spec.parts(acceptor_state)
        if parts is None:
            self.padded_state_parse_cache[acceptor_state] = None
            return None

        duration = parts[spec.duration_index]
        if not (
            isinstance(duration, tuple)
            and len(duration) == 2
            and isinstance(duration[0], int)
            and isinstance(duration[1], bool)
        ):
            self.padded_state_parse_cache[acceptor_state] = None
            return None
        total = int(duration[0])
        ended = bool(duration[1])

        final_note = False
        if spec.final_note_index is not None:
            final_state = parts[spec.final_note_index]
            if not (
                isinstance(final_state, tuple)
                and len(final_state) == 2
                and isinstance(final_state[0], bool)
                and isinstance(final_state[1], bool)
            ):
                self.padded_state_parse_cache[acceptor_state] = None
                return None
            final_note = bool(final_state[0])
            if bool(final_state[1]) != ended:
                self.padded_state_parse_cache[acceptor_state] = None
                return None

        note_count = 0
        if spec.min_note_index is not None:
            min_state = parts[spec.min_note_index]
            if not (
                isinstance(min_state, tuple)
                and len(min_state) == 2
                and isinstance(min_state[0], int)
                and isinstance(min_state[1], bool)
            ):
                self.padded_state_parse_cache[acceptor_state] = None
                return None
            note_count = int(min_state[0])
            if bool(min_state[1]) != ended:
                self.padded_state_parse_cache[acceptor_state] = None
                return None

        parsed = _PaddedMelodyState(
            parts=parts,
            total=total,
            ended=ended,
            final_note=final_note,
            note_count=note_count,
        )
        self.padded_state_parse_cache[acceptor_state] = parsed
        return parsed

    def _padded_melody_suffix_feasible_state(
        self,
        remaining: int,
        state: _PaddedMelodyState,
    ) -> bool:
        return self._padded_melody_suffix_feasible(
            remaining,
            state.total,
            state.ended,
            state.final_note,
            state.note_count,
        )

    def _padded_melody_suffix_feasible(
        self,
        remaining: int,
        total: int,
        ended: bool,
        final_note: bool,
        note_count: int,
    ) -> bool:
        spec = self.padded_melody_spec
        if spec is None:
            return False
        if total > spec.target or remaining < 0:
            return False
        capped_count = min(spec.min_notes, note_count)
        if spec.additive_duration:
            return self._padded_melody_additive_suffix_feasible(
                remaining,
                total,
                ended,
                final_note,
                capped_count,
            )
        key = (remaining, total, ended, final_note, capped_count)
        cached = self.padded_feasibility_memo.get(key)
        if cached is not None:
            return cached

        if remaining == 0:
            value = self._padded_melody_accepting_values(
                total,
                final_note,
                capped_count,
            )
            self.padded_feasibility_memo[key] = value
            return value

        if ended or total == spec.target:
            value = self._padded_melody_accepting_values(
                total,
                final_note,
                capped_count,
            )
            self.padded_feasibility_memo[key] = value
            return value

        value = False
        for next_total, is_note in self._padded_melody_duration_options(total):
            next_count = min(spec.min_notes, capped_count + (1 if is_note else 0))
            if self._padded_melody_suffix_feasible(
                remaining - 1,
                next_total,
                False,
                is_note,
                next_count,
            ):
                value = True
                break
        self.padded_feasibility_memo[key] = value
        return value

    def _padded_melody_additive_suffix_feasible(
        self,
        remaining: int,
        total: int,
        ended: bool,
        final_note: bool,
        note_count: int,
    ) -> bool:
        spec = self.padded_melody_spec
        if spec is None:
            return False
        if remaining == 0:
            return self._padded_melody_accepting_values(total, final_note, note_count)
        if ended or total == spec.target:
            return self._padded_melody_accepting_values(total, final_note, note_count)
        need = spec.target - total
        if need <= 0:
            return False
        tables = self._padded_melody_additive_suffix_tables()
        if remaining >= len(tables):
            return False
        masks = tables[remaining].get(need)
        if masks is None:
            return False
        required_gain = max(0, spec.min_notes - note_count)
        if required_gain <= 0:
            usable_mask = masks[1] if spec.requires_final_note else (masks[0] | masks[1])
            return usable_mask != 0
        usable_mask = masks[1] if spec.requires_final_note else (masks[0] | masks[1])
        return bool(usable_mask & ~((1 << required_gain) - 1))

    def _padded_melody_additive_suffix_tables(
        self,
    ) -> tuple[dict[int, tuple[int, int]], ...]:
        cached = self.padded_additive_suffix_tables
        if cached is not None:
            return cached
        spec = self.padded_melody_spec
        if spec is None:
            self.padded_additive_suffix_tables = ({},)
            return self.padded_additive_suffix_tables

        tables: list[dict[int, tuple[int, int]]] = [{} for _ in range(self.length + 1)]
        exact: set[tuple[int, int, bool]] = {(0, 0, False)}
        cumulative: dict[int, tuple[int, int]] = {}
        options = spec.cost_note_options
        for remaining in range(1, self.length + 1):
            next_exact: set[tuple[int, int, bool]] = set()
            for cost_so_far, note_gain, _last_is_note in exact:
                for cost, is_note in options:
                    next_cost = cost_so_far + cost
                    if next_cost > spec.target:
                        continue
                    next_gain = min(spec.min_notes, note_gain + (1 if is_note else 0))
                    next_exact.add((next_cost, next_gain, is_note))
            for next_cost, next_gain, last_is_note in next_exact:
                false_mask, true_mask = cumulative.get(next_cost, (0, 0))
                if last_is_note:
                    true_mask |= 1 << next_gain
                else:
                    false_mask |= 1 << next_gain
                cumulative[next_cost] = (false_mask, true_mask)
            tables[remaining] = dict(cumulative)
            exact = next_exact
        self.padded_additive_suffix_tables = tuple(tables)
        return self.padded_additive_suffix_tables

    def _padded_melody_accepting_values(
        self,
        total: int,
        final_note: bool,
        note_count: int,
    ) -> bool:
        spec = self.padded_melody_spec
        if spec is None:
            return False
        if total != spec.target:
            return False
        if note_count < spec.min_notes:
            return False
        if spec.requires_final_note and not final_note:
            return False
        return True

    def _padded_melody_duration_options(
        self,
        total: int,
    ) -> tuple[tuple[int, bool], ...]:
        cached = self.padded_options_by_total.get(total)
        if cached is not None:
            return cached
        spec = self.padded_melody_spec
        if spec is None:
            return ()
        duration_acceptor = spec.component_acceptors[spec.duration_index]
        duration_state = (total, False)
        options: set[tuple[int, bool]] = set()
        for symbol, _cost in spec.symbol_costs.items():
            if symbol == spec.pad_symbol:
                continue
            is_note = spec.symbol_is_note.get(symbol)
            if is_note is None:
                continue
            next_state = duration_acceptor.next_state(duration_state, symbol)
            if not (
                isinstance(next_state, tuple)
                and len(next_state) == 2
                and isinstance(next_state[0], int)
                and isinstance(next_state[1], bool)
            ):
                continue
            next_total = int(next_state[0])
            if next_total <= total or next_total > spec.target or bool(next_state[1]):
                continue
            options.add((next_total, bool(is_note)))
        result = tuple(sorted(options))
        self.padded_options_by_total[total] = result
        return result

    def _padded_melody_forced_pad_suffix(
        self,
        time: int,
        state: int,
        acceptor_state: Hashable,
    ) -> float:
        spec = self.padded_melody_spec
        if spec is None:
            return 0.0
        current_time = time
        current_state = state
        current_acceptor_state = acceptor_state
        mass = 1.0
        while current_time < self.length:
            if not self._padded_melody_symbol_allowed(current_time, spec.pad_symbol):
                return 0.0
            transition = self._padded_melody_pad_transition(
                current_state,
                current_acceptor_state,
            )
            if transition is None:
                return 0.0
            edge, next_acceptor_state = transition
            mass *= edge.probability
            if mass <= 0.0:
                return 0.0
            current_state = edge.dst
            current_acceptor_state = next_acceptor_state
            current_time += 1
        return mass if self.acceptor.is_accepting(current_acceptor_state) else 0.0

    def _padded_melody_pad_transition(
        self,
        state: int,
        acceptor_state: Hashable,
    ) -> tuple[StackEdge, Hashable] | None:
        spec = self.padded_melody_spec
        if spec is None:
            return None
        for edge in self.graph.outgoing[state]:
            if edge.symbol != spec.pad_symbol:
                continue
            next_acceptor_state = self._padded_melody_next_acceptor_state(
                acceptor_state,
                edge.symbol,
            )
            if next_acceptor_state is None:
                return None
            return edge, next_acceptor_state
        return None

    def _padded_melody_symbol_allowed(self, time: int, symbol: Symbol) -> bool:
        if _is_forbidden(
            symbol,
            self.forbidden_symbols,
            self.allowed_forbidden_symbols.get(time, frozenset()),
        ):
            return False
        constraint = self.constraints.get(time)
        if constraint is None:
            return True
        if callable(constraint):
            return bool(constraint(symbol))
        return symbol in constraint

    @property
    def time_indexed_product_state_count(self) -> int:
        return len(self.memo)

    @property
    def unique_product_state_count(self) -> int:
        return len({(state, acceptor_state) for _time, state, acceptor_state in self.memo})

    @property
    def product_edge_count(self) -> int:
        return self.expanded_edge_total

    @property
    def transition_row_count(self) -> int:
        return len(self.transition_rows)

    @property
    def beta_state_expansions(self) -> int:
        return self.beta_cache_misses


@dataclass
class RegularOrderStackBPResult:
    """Constrained order-stack BP with a deterministic regular acceptor.

    This result is a generation policy, not exact conditioning of one fixed
    stochastic source.  At each step it asks each order graph for edges with
    positive future mass under the acceptor, lets the policy select an order,
    and normalizes over the selected order's feasible outgoing edge weights.
    """

    model: OrderStackModel
    length: int
    prefix: Context
    acceptor: DFA
    start_acceptor_state: Hashable
    graphs: dict[int, FixedOrderContextGraph]
    backwards: dict[int, _RegularBackwardCache]
    policy: OrderPolicy
    constraints: PositionConstraints = field(default_factory=dict)
    allowed_forbidden_symbols: Mapping[int, frozenset[Symbol]] = field(default_factory=dict)
    _candidate_set_cache: dict[tuple[int, Context, Hashable], tuple[OrderCandidateSet, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _duration_view_quotient_diagnostics: DurationViewQuotientDiagnostics | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _duration_view_quotient_diagnostics_computed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    @property
    def context_state_count(self) -> int:
        return sum(len(graph.contexts) for graph in self.graphs.values())

    @property
    def context_edge_count(self) -> int:
        return sum(graph.edge_count for graph in self.graphs.values())

    @property
    def product_state_count(self) -> int:
        return sum(cache.unique_product_state_count for cache in self.backwards.values())

    @property
    def time_indexed_product_state_count(self) -> int:
        return sum(cache.time_indexed_product_state_count for cache in self.backwards.values())

    @property
    def product_edge_count(self) -> int:
        return sum(cache.product_edge_count for cache in self.backwards.values())

    @property
    def regular_transition_row_count(self) -> int:
        return sum(cache.transition_row_count for cache in self.backwards.values())

    @property
    def regular_transition_row_cache_hits(self) -> int:
        return sum(cache.transition_row_cache_hits for cache in self.backwards.values())

    @property
    def regular_transition_row_cache_misses(self) -> int:
        return sum(cache.transition_row_cache_misses for cache in self.backwards.values())

    @property
    def regular_accepted_transition_count(self) -> int:
        return sum(cache.accepted_transition_total for cache in self.backwards.values())

    @property
    def regular_beta_cache_hits(self) -> int:
        return sum(cache.beta_cache_hits for cache in self.backwards.values())

    @property
    def regular_beta_cache_misses(self) -> int:
        return sum(cache.beta_cache_misses for cache in self.backwards.values())

    @property
    def regular_beta_state_expansions(self) -> int:
        return sum(cache.beta_state_expansions for cache in self.backwards.values())

    @property
    def regular_acceptor_symbol_transition_cache_hits(self) -> int:
        return sum(
            cache.acceptor_symbol_transition_cache_hits
            for cache in self.backwards.values()
        )

    @property
    def regular_acceptor_symbol_transition_cache_misses(self) -> int:
        return sum(
            cache.acceptor_symbol_transition_cache_misses
            for cache in self.backwards.values()
        )

    @property
    def duration_view_quotient_diagnostics(
        self,
    ) -> DurationViewQuotientDiagnostics | None:
        if self._duration_view_quotient_diagnostics_computed:
            return self._duration_view_quotient_diagnostics
        spec = None
        for cache in self.backwards.values():
            if cache.padded_melody_spec is not None:
                spec = cache.padded_melody_spec
                break
        if spec is None:
            spec = _padded_melody_constraint_spec(self.acceptor, self.graphs)
        if spec is not None:
            self._duration_view_quotient_diagnostics = (
                _padded_melody_duration_view_quotient_diagnostics(
                    self.graphs,
                    spec,
                )
            )
        self._duration_view_quotient_diagnostics_computed = True
        return self._duration_view_quotient_diagnostics

    @property
    def success_mass(self) -> float:
        return 1.0 if self._candidate_sets(0, self.prefix, self.start_acceptor_state) else 0.0

    def start_order_masses(self) -> tuple[tuple[int, float], ...]:
        masses: list[tuple[int, float]] = []
        max_order = min(self.model.max_order, len(self.prefix))
        for order in range(1, max_order + 1):
            graph = self.graphs[order]
            state = graph.state_id(tuple(self.prefix[-order:]))
            mass = (
                self.backwards[order].beta(0, state, self.start_acceptor_state)
                if state is not None
                else 0.0
            )
            masses.append((order, mass))
        return tuple(masses)

    def sample_with_trace(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> tuple[tuple[Symbol, ...], tuple[OrderSampleStep, ...]]:
        generator = _coerce_rng(rng)
        history = list(self.prefix)
        output: list[Symbol] = []
        trace: list[OrderSampleStep] = []
        acceptor_state = self.start_acceptor_state

        for position in range(self.length):
            candidate_sets = self._candidate_sets(position, history, acceptor_state)
            choice = self.policy.choose(candidate_sets, generator)
            if choice is None:
                raise ValueError("No order has positive constrained future mass.")
            edge = choice.edge
            cache = self.backwards[choice.candidate_set.order]
            next_acceptor_state = cache.next_acceptor_state(acceptor_state, edge.symbol)
            if next_acceptor_state is None:
                raise RuntimeError("selected edge is not accepted by the DFA")
            trace_context = tuple(history[-choice.candidate_set.order :])
            output.append(edge.symbol)
            history.append(edge.symbol)
            acceptor_state = next_acceptor_state
            trace.append(_sample_step(position, choice, context=trace_context))

        if not self.acceptor.is_accepting(acceptor_state):
            raise RuntimeError("order-stack policy ended in a non-accepting DFA state")
        return tuple(output), tuple(trace)

    def sample_with_orders(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> tuple[tuple[Symbol, ...], tuple[int, ...]]:
        generator = _coerce_rng(rng)
        history = list(self.prefix)
        output: list[Symbol] = []
        orders: list[int] = []
        acceptor_state = self.start_acceptor_state

        for position in range(self.length):
            candidate_sets = self._candidate_sets(position, history, acceptor_state)
            choice = self.policy.choose(candidate_sets, generator)
            if choice is None:
                raise ValueError("No order has positive constrained future mass.")
            edge = choice.edge
            cache = self.backwards[choice.candidate_set.order]
            next_acceptor_state = cache.next_acceptor_state(acceptor_state, edge.symbol)
            if next_acceptor_state is None:
                raise RuntimeError("selected edge is not accepted by the DFA")
            output.append(edge.symbol)
            orders.append(edge.order)
            history.append(edge.symbol)
            acceptor_state = next_acceptor_state

        if not self.acceptor.is_accepting(acceptor_state):
            raise RuntimeError("order-stack policy ended in a non-accepting DFA state")
        return tuple(output), tuple(orders)

    def sample(self, *, rng: random.Random | int | None = None) -> tuple[Symbol, ...]:
        return self.sample_with_orders(rng=rng)[0]

    def sample_many_with_orders(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[tuple[tuple[Symbol, ...], tuple[int, ...]]]:
        generator = _coerce_rng(rng)
        return [self.sample_with_orders(rng=generator) for _ in range(count)]

    def _candidate_sets(
        self,
        position: int,
        history: Sequence[Symbol],
        acceptor_state: Hashable,
    ) -> list[OrderCandidateSet]:
        history_key = tuple(history[-self.model.max_order :])
        cache_key = (position, history_key, acceptor_state)
        cached = self._candidate_set_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        candidate_sets: list[OrderCandidateSet] = []
        max_order = min(self.model.max_order, len(history))
        constraint = self.constraints.get(position)
        allowed_forbidden = self.allowed_forbidden_symbols.get(position, frozenset())
        for order in range(max_order, 0, -1):
            graph = self.graphs[order]
            context = tuple(history[-order:])
            state = graph.state_id(context)
            if state is None:
                continue
            candidates: list[StackEdge] = []
            weights: list[float] = []
            cache = self.backwards[order]
            row = cache.accepted_transitions(state, acceptor_state)
            if constraint is None:
                for edge, next_acceptor_state, transition_weight in row:
                    if _is_forbidden(edge.symbol, self.model.forbidden_symbols, allowed_forbidden):
                        continue
                    weight = (
                        edge.probability
                        * transition_weight
                        * cache.beta(
                            position + 1,
                            edge.dst,
                            next_acceptor_state,
                        )
                    )
                    if weight <= 0.0:
                        continue
                    candidates.append(edge)
                    weights.append(weight)
            elif callable(constraint):
                for edge, next_acceptor_state, transition_weight in row:
                    if _is_forbidden(edge.symbol, self.model.forbidden_symbols, allowed_forbidden):
                        continue
                    if not constraint(edge.symbol):
                        continue
                    weight = (
                        edge.probability
                        * transition_weight
                        * cache.beta(
                            position + 1,
                            edge.dst,
                            next_acceptor_state,
                        )
                    )
                    if weight <= 0.0:
                        continue
                    candidates.append(edge)
                    weights.append(weight)
            else:
                for edge, next_acceptor_state, transition_weight in row:
                    if _is_forbidden(edge.symbol, self.model.forbidden_symbols, allowed_forbidden):
                        continue
                    if edge.symbol not in constraint:
                        continue
                    weight = (
                        edge.probability
                        * transition_weight
                        * cache.beta(
                            position + 1,
                            edge.dst,
                            next_acceptor_state,
                        )
                    )
                    if weight <= 0.0:
                        continue
                    candidates.append(edge)
                    weights.append(weight)
            if candidates:
                candidate_sets.append(
                    OrderCandidateSet(
                        order=order,
                        graph=graph,
                        state=state,
                        edges=tuple(candidates),
                        weights=tuple(weights),
                    )
                )
        self._candidate_set_cache[cache_key] = tuple(candidate_sets)
        return candidate_sets


@dataclass(frozen=True)
class RegularOrderStackBPPlan:
    """Prefix-independent regular order-stack BP preparation.

    The regular beta memo tables are shared by every prefix-bound result created
    from this plan. They are intentionally lazy: binding a prefix warms the
    messages reachable from that prefix, and later prefixes reuse any overlapping
    product states.
    """

    model: OrderStackModel
    length: int
    acceptor: DFA
    start_acceptor_state: Hashable
    graphs: dict[int, FixedOrderContextGraph]
    backwards: dict[int, _RegularBackwardCache]
    policy: OrderPolicy
    constraints: PositionConstraints = field(default_factory=dict)
    allowed_forbidden_symbols: Mapping[int, frozenset[Symbol]] = field(default_factory=dict)

    def for_prefix(
        self,
        prefix: Sequence[Symbol],
        *,
        start_acceptor_state: Hashable | None = None,
    ) -> RegularOrderStackBPResult:
        if not prefix:
            raise ValueError("order-stack BP requires a non-empty prefix")
        acceptor0 = (
            self.start_acceptor_state
            if start_acceptor_state is None
            else start_acceptor_state
        )
        result = RegularOrderStackBPResult(
            model=self.model,
            length=self.length,
            prefix=tuple(prefix),
            acceptor=self.acceptor,
            start_acceptor_state=acceptor0,
            graphs=self.graphs,
            backwards=self.backwards,
            policy=self.policy,
            constraints=self.constraints,
            allowed_forbidden_symbols=self.allowed_forbidden_symbols,
        )
        result.start_order_masses()
        return result


def _sample_from_weighted_edges(
    edges: tuple[StackEdge, ...],
    weights: tuple[float, ...],
    cumulative_weights: tuple[float, ...] | None,
    rng: random.Random,
) -> StackEdge:
    if not edges:
        raise ValueError("cannot sample from an empty edge set")
    if cumulative_weights is None:
        cumulative_weights = _cumulative_weights(weights)
    total = cumulative_weights[-1] if cumulative_weights else 0.0
    if total <= 0.0:
        raise ValueError("cannot sample from non-positive edge weights")
    index = bisect.bisect_left(cumulative_weights, rng.random() * total)
    if index >= len(edges):
        index = len(edges) - 1
    return edges[index]


def _cumulative_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = 0.0
    cumulative: list[float] = []
    for weight in weights:
        total += weight
        cumulative.append(total)
    return tuple(cumulative)


def prepare_order_stack_dfa_bp(
    model: OrderStackModel,
    acceptor: DFA,
    *,
    length: int,
    start_acceptor_state: Hashable | None = None,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    policy: OrderPolicy | None = None,
    minimize_source_graphs: bool = False,
) -> RegularOrderStackBPPlan:
    return prepare_order_stack_masked_dfa_bp(
        model,
        acceptor,
        length=length,
        constraints=None,
        start_acceptor_state=start_acceptor_state,
        allowed_forbidden_symbols=allowed_forbidden_symbols,
        policy=policy,
        minimize_source_graphs=minimize_source_graphs,
    )


def prepare_order_stack_masked_dfa_bp(
    model: OrderStackModel,
    acceptor: DFA,
    *,
    length: int,
    constraints: PositionConstraints | None = None,
    start_acceptor_state: Hashable | None = None,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    policy: OrderPolicy | None = None,
    minimize_source_graphs: bool = False,
) -> RegularOrderStackBPPlan:
    """Prepare reusable regular order-stack backward caches without a prefix."""

    if length < 0:
        raise ValueError("length must be non-negative")
    position_constraints = dict(constraints or {})
    _validate_constraint_positions(position_constraints, length)
    allowed_forbidden = _normalize_allowed_forbidden_symbols(
        allowed_forbidden_symbols,
        length,
    )
    active_policy = policy or LongestFeasiblePolicy()
    acceptor0 = acceptor.start_state if start_acceptor_state is None else start_acceptor_state
    graphs = _compile_graphs_for_prefixless_plan(
        model,
        length=length,
        minimize_source_graphs=minimize_source_graphs,
    )
    padded_melody_spec = _padded_melody_constraint_spec(acceptor, graphs)
    backwards = {
        order: _RegularBackwardCache(
            graph,
            acceptor,
            length,
            constraints=position_constraints,
            forbidden_symbols=model.forbidden_symbols,
            allowed_forbidden_symbols=allowed_forbidden,
            padded_melody_spec=padded_melody_spec,
        )
        for order, graph in graphs.items()
    }
    return RegularOrderStackBPPlan(
        model=model,
        length=length,
        acceptor=acceptor,
        start_acceptor_state=acceptor0,
        graphs=graphs,
        backwards=backwards,
        policy=active_policy,
        constraints=position_constraints,
        allowed_forbidden_symbols=allowed_forbidden,
    )


def _compile_graphs_for_prefixless_plan(
    model: OrderStackModel,
    *,
    length: int,
    minimize_source_graphs: bool = False,
) -> dict[int, FixedOrderContextGraph]:
    compile_graphs = getattr(model, "compile_graphs_for_plan", None)
    if compile_graphs is not None:
        graphs = compile_graphs(length=length)
        if minimize_source_graphs:
            graphs = minimize_fixed_order_graphs(graphs)
        return graphs
    if getattr(model, "compile_graphs_for_prefix", None) is not None:
        raise ValueError(
            "prefix-independent order-stack plans require prefix-independent graph "
            "compilation; implement compile_graphs_for_plan on the model"
        )
    graphs = _compile_materialized_source_graphs(
        model,
        minimize_source_graphs=minimize_source_graphs,
    )
    return graphs


def _compile_materialized_source_graphs(
    model: OrderStackModel,
    *,
    minimize_source_graphs: bool = False,
) -> dict[int, FixedOrderContextGraph]:
    if minimize_source_graphs:
        compile_minimized = getattr(model, "compile_minimized_graph", None)
        if compile_minimized is not None:
            return {
                order: compile_minimized(order)
                for order in range(1, model.max_order + 1)
            }
    graphs = {order: model.compile_graph(order) for order in range(1, model.max_order + 1)}
    if minimize_source_graphs:
        graphs = minimize_fixed_order_graphs(graphs)
    return graphs


def _padded_melody_constraint_spec(
    acceptor: DFA,
    graphs: Mapping[int, FixedOrderContextGraph],
) -> _PaddedMelodyConstraintSpec | None:
    components = getattr(acceptor, "component_acceptors", None)
    if components is None:
        if getattr(acceptor, "name", None) != "padded_melody_duration_total":
            return None
        components = (acceptor,)
    components = tuple(components)
    if not components:
        return None
    if any(has_custom_transition_weights(component) for component in components):
        return None

    known_names = {
        "padded_melody_duration_total",
        "final_real_note",
        "min_real_note_count",
    }
    names = tuple(str(getattr(component, "name", "")) for component in components)
    if any(name not in known_names for name in names):
        return None
    if names.count("padded_melody_duration_total") != 1:
        return None
    if names.count("final_real_note") > 1 or names.count("min_real_note_count") > 1:
        return None

    duration_index = names.index("padded_melody_duration_total")
    final_note_index = names.index("final_real_note") if "final_real_note" in names else None
    min_note_index = names.index("min_real_note_count") if "min_real_note_count" in names else None
    duration_acceptor = components[duration_index]
    target = _padded_melody_duration_target(duration_acceptor)
    if target is None:
        return None

    min_notes = 0
    if min_note_index is not None:
        inferred_min_notes = _padded_melody_min_notes(components[min_note_index])
        if inferred_min_notes is None:
            return None
        min_notes = inferred_min_notes

    symbols = _graph_edge_symbols(graphs)
    pad_symbol = _padded_melody_pad_symbol(duration_acceptor, symbols, target)
    if pad_symbol is None:
        return None

    symbol_costs = _padded_melody_symbol_costs(
        duration_acceptor,
        symbols,
        target,
        pad_symbol,
    )
    symbol_is_note = _padded_melody_symbol_note_flags(
        components,
        final_note_index,
        min_note_index,
        symbols,
        pad_symbol,
    )
    if not symbol_costs or not symbol_is_note:
        return None
    additive_duration = _padded_melody_has_additive_duration(
        duration_acceptor,
        symbol_costs,
        target,
        pad_symbol,
    )
    cost_note_options = tuple(
        sorted(
            {
                (cost, bool(symbol_is_note[symbol]))
                for symbol, cost in symbol_costs.items()
                if symbol != pad_symbol and cost > 0 and symbol in symbol_is_note
            }
        )
    )

    return _PaddedMelodyConstraintSpec(
        component_acceptors=components,
        duration_index=duration_index,
        final_note_index=final_note_index,
        min_note_index=min_note_index,
        target=target,
        min_notes=min_notes,
        pad_symbol=pad_symbol,
        symbol_costs=symbol_costs,
        symbol_is_note=symbol_is_note,
        additive_duration=additive_duration,
        cost_note_options=cost_note_options,
    )


def padded_melody_duration_view_quotient_diagnostics(
    graphs: Mapping[int, FixedOrderContextGraph],
    acceptor: DFA,
) -> DurationViewQuotientDiagnostics | None:
    """Measure exact duration-view compression for LSDB-style melody meters.

    The diagnostic projects each concrete symbol edge to its additive meter
    view: duration cost, note/rest flag, and PAD status. It then computes the
    coarsest exact quotient whose projected transition masses agree by
    destination class. This is the quotient a future beta backend could use
    without changing additive-meter future masses.
    """

    spec = _padded_melody_constraint_spec(acceptor, graphs)
    if spec is None:
        return None
    return _padded_melody_duration_view_quotient_diagnostics(graphs, spec)


def _padded_melody_duration_view_quotient_diagnostics(
    graphs: Mapping[int, FixedOrderContextGraph],
    spec: _PaddedMelodyConstraintSpec,
) -> DurationViewQuotientDiagnostics:
    started = time.perf_counter()
    order_stats = tuple(
        _padded_melody_duration_view_quotient_order_stats(graph, spec)
        for _order, graph in sorted(graphs.items())
    )
    return DurationViewQuotientDiagnostics(
        orders=order_stats,
        seconds=time.perf_counter() - started,
    )


def _padded_melody_duration_view_quotient_order_stats(
    graph: FixedOrderContextGraph,
    spec: _PaddedMelodyConstraintSpec,
) -> DurationViewQuotientOrderStats:
    state_count = len(graph.contexts)
    classes = [0 for _state in graph.contexts]
    rounds = 0
    projected_edge_count = 0
    ignored_edge_count = 0
    category_set: set[tuple[int, bool, bool]] = set()

    while True:
        signatures: list[tuple[tuple[tuple[int, bool, bool], int, tuple[int, int]], ...]] = []
        projected_edge_count = 0
        ignored_edge_count = 0
        category_set.clear()
        for edges in graph.outgoing:
            buckets: dict[tuple[tuple[int, bool, bool], int], list[float]] = {}
            for edge in edges:
                category = _padded_melody_duration_view_category(edge.symbol, spec)
                if category is None:
                    ignored_edge_count += 1
                    continue
                projected_edge_count += 1
                category_set.add(category)
                buckets.setdefault((category, classes[edge.dst]), []).append(
                    edge.probability,
                )
            signatures.append(
                tuple(
                    sorted(
                        (
                            category,
                            dst_class,
                            _float_key(math.fsum(probabilities)),
                        )
                        for (category, dst_class), probabilities in buckets.items()
                    )
                )
            )
        new_classes, counts = _classes_from_signatures(signatures)
        rounds += 1
        if new_classes == classes:
            representatives = _representative_states(new_classes)
            quotient_edges = sum(len(signatures[state]) for state in representatives)
            return DurationViewQuotientOrderStats(
                order=graph.order,
                states=state_count,
                classes=len(counts),
                edges=graph.edge_count,
                projected_edges=projected_edge_count,
                ignored_edges=ignored_edge_count,
                quotient_edges=quotient_edges,
                category_count=len(category_set),
                max_class_size=max(counts or [0]),
                refinement_rounds=rounds,
            )
        classes = new_classes


def _padded_melody_duration_view_category(
    symbol: Symbol,
    spec: _PaddedMelodyConstraintSpec,
) -> tuple[int, bool, bool] | None:
    if symbol == spec.pad_symbol:
        return (0, False, True)
    cost = spec.symbol_costs.get(symbol)
    is_note = spec.symbol_is_note.get(symbol)
    if cost is None or is_note is None or cost <= 0:
        return None
    return (int(cost), bool(is_note), False)


def _graph_edge_symbols(graphs: Mapping[int, FixedOrderContextGraph]) -> tuple[Symbol, ...]:
    symbols = {
        edge.symbol
        for graph in graphs.values()
        for edges in graph.outgoing
        for edge in edges
    }
    return tuple(symbols)


def _padded_melody_duration_target(acceptor: DFA) -> int | None:
    states = getattr(acceptor, "states", None)
    if states is None:
        return None
    totals = [
        state[0]
        for state in states
        if (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[0], int)
            and isinstance(state[1], bool)
        )
    ]
    if not totals:
        return None
    target = max(totals)
    if target < 0:
        return None
    return int(target)


def _padded_melody_min_notes(acceptor: DFA) -> int | None:
    states = getattr(acceptor, "states", None)
    if states is None:
        return None
    counts = [
        state[0]
        for state in states
        if (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[0], int)
            and isinstance(state[1], bool)
        )
    ]
    if not counts:
        return None
    return int(max(counts))


def _padded_melody_pad_symbol(
    duration_acceptor: DFA,
    symbols: Iterable[Symbol],
    target: int,
) -> Symbol | None:
    candidates: list[Symbol] = []
    target_state = (target, False)
    for symbol in symbols:
        try:
            next_state = duration_acceptor.next_state(target_state, symbol)
        except (TypeError, ValueError, KeyError):
            continue
        if next_state == (target, True):
            candidates.append(symbol)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _padded_melody_symbol_costs(
    duration_acceptor: DFA,
    symbols: Iterable[Symbol],
    target: int,
    pad_symbol: Symbol,
) -> dict[Symbol, int]:
    costs: dict[Symbol, int] = {pad_symbol: 0}
    start_state = (0, False)
    for symbol in symbols:
        if symbol == pad_symbol:
            continue
        try:
            next_state = duration_acceptor.next_state(start_state, symbol)
        except (TypeError, ValueError, KeyError):
            continue
        if not (
            isinstance(next_state, tuple)
            and len(next_state) == 2
            and isinstance(next_state[0], int)
            and isinstance(next_state[1], bool)
        ):
            continue
        cost = int(next_state[0])
        if cost <= 0 or cost > target or bool(next_state[1]):
            continue
        costs[symbol] = cost
    return costs


def _padded_melody_has_additive_duration(
    duration_acceptor: DFA,
    symbol_costs: Mapping[Symbol, int],
    target: int,
    pad_symbol: Symbol,
) -> bool:
    representative_by_cost: dict[int, Symbol] = {}
    for symbol, cost in symbol_costs.items():
        if symbol == pad_symbol or cost <= 0:
            continue
        representative_by_cost.setdefault(cost, symbol)
    for total in range(target + 1):
        state = (total, False)
        for cost, symbol in representative_by_cost.items():
            expected = (total + cost, False) if total + cost <= target else None
            try:
                next_state = duration_acceptor.next_state(state, symbol)
            except (TypeError, ValueError, KeyError):
                next_state = None
            if next_state != expected:
                return False
    return True


def _padded_melody_symbol_note_flags(
    components: Sequence[DFA],
    final_note_index: int | None,
    min_note_index: int | None,
    symbols: Iterable[Symbol],
    pad_symbol: Symbol,
) -> dict[Symbol, bool]:
    flags: dict[Symbol, bool] = {}
    if final_note_index is not None:
        final_acceptor = components[final_note_index]
        for symbol in symbols:
            if symbol == pad_symbol:
                continue
            try:
                next_state = final_acceptor.next_state((False, False), symbol)
            except (TypeError, ValueError, KeyError):
                continue
            if not (
                isinstance(next_state, tuple)
                and len(next_state) == 2
                and isinstance(next_state[0], bool)
                and isinstance(next_state[1], bool)
            ):
                continue
            if bool(next_state[1]):
                continue
            flags[symbol] = bool(next_state[0])
        return flags

    if min_note_index is not None:
        min_acceptor = components[min_note_index]
        for symbol in symbols:
            if symbol == pad_symbol:
                continue
            try:
                next_state = min_acceptor.next_state((0, False), symbol)
            except (TypeError, ValueError, KeyError):
                continue
            if not (
                isinstance(next_state, tuple)
                and len(next_state) == 2
                and isinstance(next_state[0], int)
                and isinstance(next_state[1], bool)
            ):
                continue
            if bool(next_state[1]):
                continue
            flags[symbol] = int(next_state[0]) > 0
        return flags

    return {symbol: False for symbol in symbols if symbol != pad_symbol}


def _classes_from_signatures(
    signatures: Sequence[Hashable],
) -> tuple[list[int], list[int]]:
    class_ids: dict[Hashable, int] = {}
    classes: list[int] = []
    counts: list[int] = []
    for signature in signatures:
        class_id = class_ids.get(signature)
        if class_id is None:
            class_id = len(class_ids)
            class_ids[signature] = class_id
            counts.append(0)
        classes.append(class_id)
        counts[class_id] += 1
    return classes, counts


def _representative_states(classes: Sequence[int]) -> list[int]:
    representatives: dict[int, int] = {}
    for state, class_id in enumerate(classes):
        representatives.setdefault(class_id, state)
    return [representatives[class_id] for class_id in range(len(representatives))]


def _float_key(value: float) -> tuple[int, int]:
    return float(value).as_integer_ratio()


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("inf") if numerator else 1.0
    return float(numerator) / float(denominator)


def run_order_stack_dfa_bp(
    model: OrderStackModel,
    acceptor: DFA,
    *,
    length: int,
    prefix: Sequence[Symbol],
    start_acceptor_state: Hashable | None = None,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    policy: OrderPolicy | None = None,
    minimize_source_graphs: bool = False,
) -> RegularOrderStackBPResult:
    return _run_order_stack_regular_bp(
        model,
        acceptor,
        length=length,
        prefix=prefix,
        start_acceptor_state=start_acceptor_state,
        constraints=None,
        allowed_forbidden_symbols=allowed_forbidden_symbols,
        policy=policy,
        minimize_source_graphs=minimize_source_graphs,
    )


def run_order_stack_masked_dfa_bp(
    model: OrderStackModel,
    acceptor: DFA,
    *,
    length: int,
    prefix: Sequence[Symbol],
    constraints: PositionConstraints | None = None,
    start_acceptor_state: Hashable | None = None,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None = None,
    policy: OrderPolicy | None = None,
    minimize_source_graphs: bool = False,
) -> RegularOrderStackBPResult:
    """Run regular order-stack BP with additional time-indexed symbol masks."""

    return _run_order_stack_regular_bp(
        model,
        acceptor,
        length=length,
        prefix=prefix,
        start_acceptor_state=start_acceptor_state,
        constraints=constraints,
        allowed_forbidden_symbols=allowed_forbidden_symbols,
        policy=policy,
        minimize_source_graphs=minimize_source_graphs,
    )


def _run_order_stack_regular_bp(
    model: OrderStackModel,
    acceptor: DFA,
    *,
    length: int,
    prefix: Sequence[Symbol],
    start_acceptor_state: Hashable | None,
    constraints: PositionConstraints | None,
    allowed_forbidden_symbols: AllowedForbiddenSymbols | None,
    policy: OrderPolicy | None,
    minimize_source_graphs: bool = False,
) -> RegularOrderStackBPResult:
    if length < 0:
        raise ValueError("length must be non-negative")
    if not prefix:
        raise ValueError("order-stack BP requires a non-empty prefix")
    position_constraints = dict(constraints or {})
    _validate_constraint_positions(position_constraints, length)
    allowed_forbidden = _normalize_allowed_forbidden_symbols(
        allowed_forbidden_symbols,
        length,
    )
    active_policy = policy or LongestFeasiblePolicy()
    acceptor0 = acceptor.start_state if start_acceptor_state is None else start_acceptor_state
    compile_graphs = getattr(model, "compile_graphs_for_prefix", None)
    if compile_graphs is None:
        graphs = _compile_materialized_source_graphs(
            model,
            minimize_source_graphs=minimize_source_graphs,
        )
    else:
        graphs = compile_graphs(prefix=prefix, length=length)
    if minimize_source_graphs and compile_graphs is not None:
        graphs = minimize_fixed_order_graphs(graphs)
    backwards = {
        order: _RegularBackwardCache(
            graph,
            acceptor,
            length,
            constraints=position_constraints,
            forbidden_symbols=model.forbidden_symbols,
            allowed_forbidden_symbols=allowed_forbidden,
        )
        for order, graph in graphs.items()
    }
    result = RegularOrderStackBPResult(
        model=model,
        length=length,
        prefix=tuple(prefix),
        acceptor=acceptor,
        start_acceptor_state=acceptor0,
        graphs=graphs,
        backwards=backwards,
        policy=active_policy,
        constraints=position_constraints,
        allowed_forbidden_symbols=allowed_forbidden,
    )
    result.start_order_masses()
    return result
