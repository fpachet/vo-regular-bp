"""Backward dynamic programming on reachable context-acceptor products."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Hashable, Iterable, Sequence

from .acceptors import DFA, transition_weight as regular_transition_weight
from .context import Context, ContextGraph, Symbol, _as_context

ProductState = tuple[Context, Hashable]


@dataclass(frozen=True)
class ProductEdge:
    symbol: Symbol
    probability: float
    transition_weight: float
    next_state: ProductState
    order_weights: tuple[tuple[int, float], ...] = ()


@dataclass
class ProductBPResult:
    graph: ContextGraph
    acceptor: DFA
    length: int
    start_state: ProductState
    layers: list[set[ProductState]]
    edges: list[dict[ProductState, tuple[ProductEdge, ...]]]
    betas: list[dict[ProductState, float]]
    _transition_weight_cache: dict[tuple[int, ProductState], tuple[tuple[ProductEdge, float], ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def partition_function(self) -> float:
        return self.betas[0].get(self.start_state, 0.0)

    @property
    def unique_product_state_count(self) -> int:
        return len(set().union(*self.layers)) if self.layers else 0

    @property
    def time_indexed_product_state_count(self) -> int:
        return sum(len(layer) for layer in self.layers)

    @property
    def product_edge_count(self) -> int:
        return sum(len(edges) for layer_edges in self.edges for edges in layer_edges.values())

    @property
    def reachable_acceptor_state_count(self) -> int:
        return len({state for layer in self.layers for _, state in layer})

    def transition_weights(self, time: int, state: ProductState) -> tuple[tuple[ProductEdge, float], ...]:
        """Return outgoing product edges and their conditional sampling weights."""

        if time < 0 or time >= self.length:
            raise ValueError("time is outside the BP horizon")
        cache_key = (time, state)
        if cache_key in self._transition_weight_cache:
            return self._transition_weight_cache[cache_key]

        beta_next = self.betas[time + 1]
        weighted = []
        for edge in self.edges[time].get(state, ()):
            weight = (
                edge.probability
                * edge.transition_weight
                * beta_next.get(edge.next_state, 0.0)
            )
            if weight > 0.0:
                weighted.append((edge, weight))
        result = tuple(weighted)
        self._transition_weight_cache[cache_key] = result
        return result

    def _sample_edge(self, time: int, state: ProductState, generator: random.Random) -> ProductEdge:
        weighted = self.transition_weights(time, state)
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
        """Draw one exact sample from the constrained distribution."""

        generator = _coerce_rng(rng)
        if self.partition_function <= 0.0:
            raise ValueError("cannot sample because the constrained partition function is zero")

        state = self.start_state
        output: list[Symbol] = []
        for time in range(self.length):
            chosen = self._sample_edge(time, state, generator)
            output.append(chosen.symbol)
            state = chosen.next_state
        return tuple(output)

    def sample_many(self, count: int, *, rng: random.Random | int | None = None) -> list[tuple[Symbol, ...]]:
        generator = _coerce_rng(rng)
        return [self.sample(rng=generator) for _ in range(count)]

    def sample_with_orders(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> tuple[tuple[Symbol, ...], tuple[int, ...]]:
        """Draw one exact sample and a latent order used at each event.

        For explicit backoff edges, the order is sampled from the posterior
        contribution of each suffix order to the selected symbol. For ordinary
        MLE edges without order metadata, the current context length is used.
        """

        generator = _coerce_rng(rng)
        if self.partition_function <= 0.0:
            raise ValueError("cannot sample because the constrained partition function is zero")

        state = self.start_state
        output: list[Symbol] = []
        orders: list[int] = []
        for time in range(self.length):
            chosen = self._sample_edge(time, state, generator)
            output.append(chosen.symbol)
            orders.append(_sample_order(chosen.order_weights, generator))
            state = chosen.next_state
        return tuple(output), tuple(orders)

    def sample_many_with_orders(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[tuple[tuple[Symbol, ...], tuple[int, ...]]]:
        generator = _coerce_rng(rng)
        return [self.sample_with_orders(rng=generator) for _ in range(count)]

    def conditional_probability(self, sequence: Sequence[Symbol]) -> float:
        """Probability of ``sequence`` under the constrained BP distribution."""

        if len(sequence) != self.length:
            raise ValueError("sequence length must match the BP horizon")
        if self.partition_function <= 0.0:
            return 0.0

        state = self.start_state
        probability = 1.0
        for time, symbol in enumerate(sequence):
            beta_now = self.betas[time].get(state, 0.0)
            if beta_now <= 0.0:
                return 0.0
            edge = next(
                (
                    edge
                    for edge in self.edges[time].get(state, ())
                    if edge.symbol == symbol
                ),
                None,
            )
            if edge is None:
                return 0.0
            weight = (
                edge.probability
                * edge.transition_weight
                * self.betas[time + 1].get(edge.next_state, 0.0)
            )
            if weight <= 0.0:
                return 0.0
            probability *= weight / beta_now
            state = edge.next_state
        return probability


def run_bp(
    graph: ContextGraph,
    acceptor: DFA,
    *,
    length: int,
    start_context: Iterable[Symbol] | Context | None = None,
    start_acceptor_state: Hashable | None = None,
) -> ProductBPResult:
    """Run exact backward DP on the reachable product for a fixed horizon."""

    if length < 0:
        raise ValueError("length must be non-negative")

    context0 = graph.start_state if start_context is None else _as_context(start_context)
    acceptor0 = acceptor.start_state if start_acceptor_state is None else start_acceptor_state
    start = (context0, acceptor0)

    layers: list[set[ProductState]] = [set([start])]
    edges_by_time: list[dict[ProductState, tuple[ProductEdge, ...]]] = []

    for _time in range(length):
        current_layer = layers[-1]
        next_layer: set[ProductState] = set()
        current_edges: dict[ProductState, tuple[ProductEdge, ...]] = {}

        for context_state, acceptor_state in current_layer:
            product_edges = []
            for edge in graph.outgoing(context_state):
                next_acceptor_state = acceptor.next_state(acceptor_state, edge.symbol)
                if next_acceptor_state is None:
                    continue
                dfa_weight = regular_transition_weight(acceptor, acceptor_state, edge.symbol)
                if dfa_weight <= 0.0:
                    continue
                next_product_state = (edge.next_state, next_acceptor_state)
                product_edges.append(
                    ProductEdge(
                        symbol=edge.symbol,
                        probability=edge.probability,
                        transition_weight=dfa_weight,
                        next_state=next_product_state,
                        order_weights=edge.order_weights or ((len(context_state), 1.0),),
                    )
                )
                next_layer.add(next_product_state)
            current_edges[(context_state, acceptor_state)] = tuple(product_edges)

        edges_by_time.append(current_edges)
        layers.append(next_layer)

    betas: list[dict[ProductState, float]] = [dict() for _ in range(length + 1)]
    betas[length] = {
        state: 1.0 if acceptor.is_accepting(state[1]) else 0.0
        for state in layers[length]
    }

    for time in range(length - 1, -1, -1):
        beta_next = betas[time + 1]
        beta_now: dict[ProductState, float] = {}
        for state in layers[time]:
            beta_now[state] = sum(
                edge.probability
                * edge.transition_weight
                * beta_next.get(edge.next_state, 0.0)
                for edge in edges_by_time[time].get(state, ())
            )
        betas[time] = beta_now

    return ProductBPResult(
        graph=graph,
        acceptor=acceptor,
        length=length,
        start_state=start,
        layers=layers,
        edges=edges_by_time,
        betas=betas,
    )


def sample_exact(
    graph: ContextGraph,
    acceptor: DFA,
    *,
    length: int,
    rng: random.Random | int | None = None,
    start_context: Iterable[Symbol] | Context | None = None,
    start_acceptor_state: Hashable | None = None,
) -> tuple[Symbol, ...]:
    """Convenience wrapper: run BP and draw one exact sample."""

    result = run_bp(
        graph,
        acceptor,
        length=length,
        start_context=start_context,
        start_acceptor_state=start_acceptor_state,
    )
    return result.sample(rng=rng)


def _coerce_rng(rng: random.Random | int | None) -> random.Random:
    if rng is None:
        return random.Random()
    if isinstance(rng, int):
        return random.Random(rng)
    return rng


def _sample_order(order_weights: tuple[tuple[int, float], ...], rng: random.Random) -> int:
    if not order_weights:
        return 0
    threshold = rng.random() * sum(weight for _, weight in order_weights)
    cumulative = 0.0
    chosen = order_weights[-1][0]
    for order, weight in order_weights:
        cumulative += weight
        if threshold <= cumulative:
            return order
    return chosen
