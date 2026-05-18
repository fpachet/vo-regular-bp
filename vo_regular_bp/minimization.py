"""Exact quotient/minimization helpers for stochastic context graphs."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .context import Context, ContextGraph, Edge


@dataclass(frozen=True)
class ExactQuotientStats:
    """State/edge counts for an exact graph quotient."""

    states: int
    classes: int
    edges: int
    quotient_edges: int
    max_class_size: int
    refinement_rounds: int

    @property
    def state_reduction(self) -> float:
        return _ratio(self.states, self.classes)

    @property
    def edge_reduction(self) -> float:
        return _ratio(self.edges, self.quotient_edges)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "states": self.states,
            "classes": self.classes,
            "edges": self.edges,
            "quotient_edges": self.quotient_edges,
            "max_class_size": self.max_class_size,
            "refinement_rounds": self.refinement_rounds,
            "state_reduction": self.state_reduction,
            "edge_reduction": self.edge_reduction,
        }


def exact_context_graph_quotient_stats(graph: ContextGraph) -> ExactQuotientStats:
    """Return exact quotient counts for a :class:`ContextGraph`."""

    partition = _context_graph_partition(graph)
    return partition.stats


def minimize_context_graph(graph: ContextGraph) -> ContextGraph:
    """Return a read-only exact quotient of ``graph``.

    The quotient keeps one representative context per equivalence class and
    records aliases so callers may still ask for outgoing edges using original
    context labels.
    """

    partition = _context_graph_partition(graph)
    if partition.stats.classes == partition.stats.states:
        return graph

    representative_by_class = partition.representatives
    representative_for_state = {
        state: representative_by_class[partition.class_by_state[state]]
        for state in partition.states
    }
    edges_by_state: dict[Context, list[Edge]] = {}
    for class_id, representative in enumerate(representative_by_class):
        edges_by_state[representative] = [
            Edge(
                edge.symbol,
                edge.probability,
                representative_for_state[edge.next_state],
                edge.order_weights,
            )
            for edge in graph.outgoing(representative)
        ]

    result = ContextGraph(
        edges_by_state,
        start_state=representative_for_state[graph.start_state],
        max_order=graph.max_order,
        alphabet=graph.alphabet,
        validate=True,
    )
    result._alias_to_state = representative_for_state  # type: ignore[attr-defined]
    result._quotient_stats = partition.stats  # type: ignore[attr-defined]
    result._quotient_classes = tuple(  # type: ignore[attr-defined]
        tuple(
            state
            for state in partition.states
            if partition.class_by_state[state] == class_id
        )
        for class_id in range(len(representative_by_class))
    )
    return result


def exact_fixed_order_graph_quotient_stats(graph: object) -> ExactQuotientStats:
    """Return exact quotient counts for a fixed-order context graph."""

    partition = _fixed_order_partition(graph)
    return partition.stats


def minimize_fixed_order_graph(graph: object) -> object:
    """Return a minimized fixed-order graph, preserving original context lookup."""

    partition = _fixed_order_partition(graph)
    if partition.stats.classes == partition.stats.states:
        return graph

    from .order_stack_bp import FixedOrderContextGraph, StackEdge

    minimized = FixedOrderContextGraph(graph.order)
    representative_by_class = partition.representatives
    minimized.contexts = [graph.contexts[index] for index in representative_by_class]
    minimized.outgoing = [[] for _ in representative_by_class]

    for original_context, original_state in graph.context_to_id.items():
        minimized.context_to_id[original_context] = partition.class_by_state[original_state]

    aliases: list[list[Context]] = [[] for _ in representative_by_class]
    for state, context in enumerate(graph.contexts):
        aliases[partition.class_by_state[state]].append(context)
    minimized.state_aliases = tuple(tuple(group) for group in aliases)
    minimized.quotient_stats = partition.stats

    for class_id, representative_state in enumerate(representative_by_class):
        minimized.outgoing[class_id] = [
            StackEdge(
                src=class_id,
                dst=partition.class_by_state[edge.dst],
                symbol=edge.symbol,
                probability=edge.probability,
                order=edge.order,
            )
            for edge in graph.outgoing[representative_state]
        ]

    return minimized


def minimize_fixed_order_graphs(graphs: dict[int, object]) -> dict[int, object]:
    """Minimize every materialized fixed-order graph in ``graphs``."""

    return {
        order: minimize_fixed_order_graph(graph)
        for order, graph in graphs.items()
    }


@dataclass(frozen=True)
class _ContextGraphPartition:
    states: tuple[Context, ...]
    class_by_state: dict[Context, int]
    representatives: tuple[Context, ...]
    stats: ExactQuotientStats


@dataclass(frozen=True)
class _FixedOrderPartition:
    class_by_state: tuple[int, ...]
    representatives: tuple[int, ...]
    stats: ExactQuotientStats


def _context_graph_partition(graph: ContextGraph) -> _ContextGraphPartition:
    states = tuple(sorted(graph.states, key=_context_sort_key))
    state_index = {state: index for index, state in enumerate(states)}
    classes = [0 for _state in states]
    rounds = 0

    while True:
        signatures = [
            tuple(
                sorted(
                    (
                        _hashable_key(edge.symbol),
                        _float_key(edge.probability),
                        _order_weights_key(edge.order_weights),
                        classes[state_index[edge.next_state]],
                    )
                    for edge in graph.outgoing(state)
                )
            )
            for state in states
        ]
        new_classes, counts = _classes_from_signatures(signatures)
        rounds += 1
        if new_classes == classes:
            representatives = _representative_states(new_classes)
            quotient_edges = sum(
                len(graph.outgoing(states[state]))
                for state in representatives
            )
            return _ContextGraphPartition(
                states=states,
                class_by_state={
                    state: class_id
                    for state, class_id in zip(states, new_classes)
                },
                representatives=tuple(states[state] for state in representatives),
                stats=ExactQuotientStats(
                    states=len(states),
                    classes=len(counts),
                    edges=graph.edge_count(),
                    quotient_edges=quotient_edges,
                    max_class_size=max(counts or [0]),
                    refinement_rounds=rounds,
                ),
            )
        classes = new_classes


def _fixed_order_partition(graph: object) -> _FixedOrderPartition:
    _require_materialized_fixed_order_graph(graph)
    state_count = len(graph.contexts)
    classes = [0 for _state in graph.contexts]
    rounds = 0

    while True:
        signatures = [
            tuple(
                sorted(
                    (
                        _hashable_key(edge.symbol),
                        _float_key(edge.probability),
                        int(edge.order),
                        classes[edge.dst],
                    )
                    for edge in edges
                )
            )
            for edges in graph.outgoing
        ]
        new_classes, counts = _classes_from_signatures(signatures)
        rounds += 1
        if new_classes == classes:
            representatives = _representative_states(new_classes)
            quotient_edges = sum(
                len(graph.outgoing[state])
                for state in representatives
            )
            return _FixedOrderPartition(
                class_by_state=tuple(new_classes),
                representatives=tuple(representatives),
                stats=ExactQuotientStats(
                    states=state_count,
                    classes=len(counts),
                    edges=graph.edge_count,
                    quotient_edges=quotient_edges,
                    max_class_size=max(counts or [0]),
                    refinement_rounds=rounds,
                ),
            )
        classes = new_classes


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


def _require_materialized_fixed_order_graph(graph: object) -> None:
    from .order_stack_bp import FixedOrderContextGraph

    if not isinstance(graph, FixedOrderContextGraph):
        raise TypeError(
            "exact source-graph minimization currently supports materialized "
            "FixedOrderContextGraph instances"
        )
    required = ("order", "contexts", "context_to_id", "outgoing", "edge_count")
    if not all(hasattr(graph, name) for name in required):
        raise TypeError("expected a materialized fixed-order context graph")
    if len(graph.outgoing) != len(graph.contexts):
        raise ValueError(
            "exact minimization requires materialized graph outgoing rows; "
            "lazy graph views are not supported"
        )


def _context_sort_key(context: Context) -> tuple[int, str]:
    return (len(context), repr(context))


def _hashable_key(value: Any) -> Hashable:
    try:
        hash(value)
    except TypeError:
        return (type(value).__name__, repr(value))
    return (type(value).__name__, value)


def _float_key(value: float) -> tuple[int, int]:
    return float(value).as_integer_ratio()


def _order_weights_key(weights: Iterable[tuple[int, float]]) -> tuple[tuple[int, tuple[int, int]], ...]:
    return tuple((int(order), _float_key(weight)) for order, weight in weights)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("inf") if numerator else 1.0
    return float(numerator) / float(denominator)
