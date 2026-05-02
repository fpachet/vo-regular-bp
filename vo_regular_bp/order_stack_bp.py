"""Continuator-style BP over a stack of fixed-order context graphs.

This module intentionally has different semantics from the explicit backoff
mixture graph in :mod:`vo_regular_bp.context`.  It runs backward messages on
one graph per order and samples with an order-selection policy.  That makes it
useful for comparisons with classic Continuator-style generation without
muddying the single-graph exact BP implementation.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Protocol

from .context import Context, Symbol, _as_context
from .positional_bp import PositionConstraint, PositionConstraints, _allows, _coerce_rng


@dataclass(frozen=True)
class StackEdge:
    src: int
    dst: int
    symbol: Symbol
    probability: float
    order: int


@dataclass(frozen=True)
class OrderCandidateSet:
    order: int
    graph: "FixedOrderContextGraph"
    state: int
    edges: tuple[StackEdge, ...]
    weights: tuple[float, ...]


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
            edge = rng.choices(candidate_set.edges, weights=candidate_set.weights, k=1)[0]
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

            if len(edges) == 1 and candidate_set.order >= self.min_singleton_order:
                if rng.random() > self.singleton_acceptance_probability(candidate_set.order):
                    skipped_symbol = edges[0].symbol
                    skipped_orders.append(candidate_set.order)
                    continue
                skipped_symbol = None

            edge = rng.choices(edges, weights=weights, k=1)[0]
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

    def compile_graph(self, order: int) -> "FixedOrderContextGraph":
        cached = self._graph_cache.get(order)
        if cached is not None:
            return cached
        graph = FixedOrderContextGraph.from_model(self, order=order)
        self._graph_cache[order] = graph
        return graph


class FixedOrderContextGraph:
    def __init__(self, order: int) -> None:
        self.order = int(order)
        self.contexts: list[Context] = []
        self.context_to_id: dict[Context, int] = {}
        self.outgoing: list[list[StackEdge]] = []

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

        for context in model.counts:
            if len(context) <= order:
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
    graphs: dict[int, FixedOrderContextGraph]
    backwards: dict[int, list[list[float]]]
    policy: OrderPolicy

    @property
    def context_state_count(self) -> int:
        return sum(len(graph.contexts) for graph in self.graphs.values())

    @property
    def context_edge_count(self) -> int:
        return sum(graph.edge_count for graph in self.graphs.values())

    @property
    def bp_edge_relaxation_upper_bound(self) -> int:
        return self.length * self.context_edge_count

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
            output.append(edge.symbol)
            history.append(edge.symbol)
            trace.append(_sample_step(position, choice))

        return tuple(output), tuple(trace)

    def sample_with_orders(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> tuple[tuple[Symbol, ...], tuple[int, ...]]:
        sequence, trace = self.sample_with_trace(rng=rng)
        return sequence, tuple(step.order for step in trace)

    def sample(self, *, rng: random.Random | int | None = None) -> tuple[Symbol, ...]:
        return self.sample_with_trace(rng=rng)[0]

    def sample_many_with_orders(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[tuple[tuple[Symbol, ...], tuple[int, ...]]]:
        generator = _coerce_rng(rng)
        return [self.sample_with_orders(rng=generator) for _ in range(count)]

    def _candidate_sets(self, position: int, history: Sequence[Symbol]) -> list[OrderCandidateSet]:
        candidate_sets = []
        max_order = min(self.model.max_order, len(history))
        constraint = self.constraints.get(position)
        for order in range(max_order, 0, -1):
            graph = self.graphs[order]
            context = tuple(history[-order:])
            state = graph.state_id(context)
            if state is None:
                continue
            candidates = []
            weights = []
            beta_next = self.backwards[order][position + 1]
            for edge in graph.outgoing[state]:
                if edge.symbol in self.model.forbidden_symbols:
                    continue
                if not _allows(constraint, edge.symbol):
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
        return candidate_sets


def run_order_stack_bp(
    model: OrderStackModel,
    *,
    length: int,
    prefix: Sequence[Symbol],
    constraints: PositionConstraints | None = None,
    policy: OrderPolicy | None = None,
) -> OrderStackBPResult:
    if length < 0:
        raise ValueError("length must be non-negative")
    if not prefix:
        raise ValueError("order-stack BP requires a non-empty prefix")
    position_constraints = dict(constraints or {})
    _validate_constraint_positions(position_constraints, length)
    active_policy = policy or SingletonAvoidingBackoffPolicy()

    graphs = {order: model.compile_graph(order) for order in range(1, model.max_order + 1)}
    backwards = {
        order: _backward_messages(
            graph,
            length=length,
            constraints=position_constraints,
            forbidden_symbols=model.forbidden_symbols,
        )
        for order, graph in graphs.items()
    }
    return OrderStackBPResult(
        model=model,
        length=length,
        prefix=tuple(prefix),
        constraints=position_constraints,
        graphs=graphs,
        backwards=backwards,
        policy=active_policy,
    )


def _backward_messages(
    graph: FixedOrderContextGraph,
    *,
    length: int,
    constraints: Mapping[int, PositionConstraint],
    forbidden_symbols: frozenset[Symbol],
) -> list[list[float]]:
    backward = [[0.0 for _ in graph.contexts] for _ in range(length + 1)]
    for state in range(len(graph.contexts)):
        backward[length][state] = 1.0

    for position in range(length - 1, -1, -1):
        constraint = constraints.get(position)
        beta_next = backward[position + 1]
        beta = backward[position]
        for state, edges in enumerate(graph.outgoing):
            total = 0.0
            for edge in edges:
                if edge.symbol in forbidden_symbols:
                    continue
                if _allows(constraint, edge.symbol):
                    total += edge.probability * beta_next[edge.dst]
            beta[state] = total
    return backward


def _sample_step(position: int, choice: CandidateChoice) -> OrderSampleStep:
    decision = choice.decision
    graph = choice.candidate_set.graph
    context = graph.contexts[choice.candidate_set.state]
    return OrderSampleStep(
        position=position,
        symbol=choice.edge.symbol,
        order=choice.edge.order,
        context=context,
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
