"""Backward dynamic programming on reachable context-acceptor products."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Hashable, Iterable, Sequence

from .acceptors import DFA
from .context import Context, ContextGraph, Edge, Symbol, _as_context

ProductState = tuple[Context, Hashable]


@dataclass(frozen=True)
class ProductEdge:
    symbol: Symbol
    probability: float
    next_state: ProductState


@dataclass
class ProductBPResult:
    graph: ContextGraph
    acceptor: DFA
    length: int
    start_state: ProductState
    layers: list[set[ProductState]]
    edges: list[dict[ProductState, tuple[ProductEdge, ...]]]
    betas: list[dict[ProductState, float]]

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
        beta_next = self.betas[time + 1]
        weighted = []
        for edge in self.edges[time].get(state, ()):
            weight = edge.probability * beta_next.get(edge.next_state, 0.0)
            if weight > 0.0:
                weighted.append((edge, weight))
        return tuple(weighted)

    def sample(self, *, rng: random.Random | int | None = None) -> tuple[Symbol, ...]:
        """Draw one exact sample from the constrained distribution."""

        generator = _coerce_rng(rng)
        if self.partition_function <= 0.0:
            raise ValueError("cannot sample because the constrained partition function is zero")

        state = self.start_state
        output: list[Symbol] = []
        for time in range(self.length):
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
                    chosen = edge
                    break
            output.append(chosen.symbol)
            state = chosen.next_state
        return tuple(output)

    def sample_many(self, count: int, *, rng: random.Random | int | None = None) -> list[tuple[Symbol, ...]]:
        generator = _coerce_rng(rng)
        return [self.sample(rng=generator) for _ in range(count)]


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
                next_product_state = (edge.next_state, next_acceptor_state)
                product_edges.append(
                    ProductEdge(
                        symbol=edge.symbol,
                        probability=edge.probability,
                        next_state=next_product_state,
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
                edge.probability * beta_next.get(edge.next_state, 0.0)
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
