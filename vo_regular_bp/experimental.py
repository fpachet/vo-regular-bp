"""Experimental source-model transformations.

The functions in this module are explicit modeling operations. They may change
the source distribution, unlike exact minimization in :mod:`vo_regular_bp.minimization`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
import math
from typing import Any

from .context import Context, ContextGraph, Edge, Symbol


@dataclass(frozen=True)
class AlergiaMergeMetadata:
    """Diagnostics for an experimental ALERGIA-style source merge."""

    method: str
    alpha: float
    min_support: float
    recursive: bool
    original_state_count: int
    merged_state_count: int
    original_edge_count: int
    merged_edge_count: int
    state_to_class: dict[Context, int]
    classes: tuple[tuple[Context, ...], ...]
    conflicting_symbol_destinations: int = 0
    symbol_projection: str | None = None

    @property
    def state_compression_ratio(self) -> float:
        return _ratio(self.original_state_count, self.merged_state_count)

    @property
    def edge_compression_ratio(self) -> float:
        return _ratio(self.original_edge_count, self.merged_edge_count)

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "min_support": self.min_support,
            "recursive": self.recursive,
            "original_state_count": self.original_state_count,
            "merged_state_count": self.merged_state_count,
            "original_edge_count": self.original_edge_count,
            "merged_edge_count": self.merged_edge_count,
            "state_compression_ratio": self.state_compression_ratio,
            "edge_compression_ratio": self.edge_compression_ratio,
            "conflicting_symbol_destinations": self.conflicting_symbol_destinations,
            "symbol_projection": self.symbol_projection,
        }


def alergia_merge(
    graph: ContextGraph,
    *,
    alpha: float = 0.01,
    min_support: int | float = 10,
    recursive: bool = True,
    symbol_projection: Callable[[Symbol], Hashable] | None = None,
) -> ContextGraph:
    """Return an ALERGIA-like approximate source-state merge.

    This function acts only on the unconstrained source graph. ``symbol_projection``
    lets a caller provide the abstraction semantics used by the compatibility
    test, for example pitch class, interval class, or another project-defined
    feature. The returned graph still emits concrete symbols and remains a
    regular :class:`ContextGraph`, so existing BP remains exact with respect to
    the new merged source model.
    """

    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if min_support < 0:
        raise ValueError("min_support must be non-negative")

    counts = _state_counts(graph)
    project = symbol_projection or _identity_projection
    supports = {
        state: float(sum(symbol_counts.values()))
        for state, symbol_counts in counts.items()
    }
    states = tuple(sorted(graph.states, key=_context_sort_key))
    union = _UnionFind(states)

    for state in states:
        if supports.get(state, 0.0) < min_support:
            continue
        for root in _class_roots(union, states):
            if union.find(state) == root:
                break
            members = [
                member
                for member in states
                if union.find(member) == root
            ]
            compatible, pairs = _compatible_with_class(
                graph,
                counts,
                supports,
                state,
                members,
                alpha=float(alpha),
                min_support=float(min_support),
                recursive=recursive,
                symbol_projection=project,
            )
            if compatible:
                union.union(state, members[0])
                for left, right in pairs:
                    union.union(left, right)
                break

    return _build_merged_graph(
        graph,
        counts,
        union,
        alpha=float(alpha),
        min_support=float(min_support),
        recursive=recursive,
        symbol_projection_name=_projection_name(symbol_projection),
    )


def alergia_metadata(graph: ContextGraph) -> AlergiaMergeMetadata | None:
    """Return ALERGIA metadata if ``graph`` was produced by :func:`alergia_merge`."""

    return getattr(graph, "alergia_metadata", None)


def _compatible_with_class(
    graph: ContextGraph,
    counts: dict[Context, dict[Symbol, float]],
    supports: dict[Context, float],
    state: Context,
    members: list[Context],
    *,
    alpha: float,
    min_support: float,
    recursive: bool,
    symbol_projection: Callable[[Symbol], Hashable],
) -> tuple[bool, list[tuple[Context, Context]]]:
    pairs: list[tuple[Context, Context]] = []
    for member in members:
        compatible, member_pairs = _compatible_pair(
            graph,
            counts,
            supports,
            state,
            member,
            alpha=alpha,
            min_support=min_support,
            recursive=recursive,
            symbol_projection=symbol_projection,
            seen=set(),
        )
        if not compatible:
            return False, []
        pairs.extend(member_pairs)
    return True, pairs


def _compatible_pair(
    graph: ContextGraph,
    counts: dict[Context, dict[Symbol, float]],
    supports: dict[Context, float],
    left: Context,
    right: Context,
    *,
    alpha: float,
    min_support: float,
    recursive: bool,
    symbol_projection: Callable[[Symbol], Hashable],
    seen: set[tuple[Context, Context]],
) -> tuple[bool, list[tuple[Context, Context]]]:
    if left == right:
        return True, []
    pair = (left, right) if _context_sort_key(left) <= _context_sort_key(right) else (right, left)
    if pair in seen:
        return True, []
    seen.add(pair)

    left_counts = counts.get(left, {})
    right_counts = counts.get(right, {})
    left_support = supports.get(left, 0.0)
    right_support = supports.get(right, 0.0)
    if left_support == 0.0 and right_support == 0.0 and not left_counts and not right_counts:
        return True, [pair]
    if left_support <= 0.0 or right_support <= 0.0:
        return False, []
    if left_support < min_support or right_support < min_support:
        return False, []

    left_projected = _projected_counts(left_counts, symbol_projection)
    right_projected = _projected_counts(right_counts, symbol_projection)
    bound = _hoeffding_bound(left_support, right_support, alpha)
    for projected_symbol in set(left_projected) | set(right_projected):
        left_probability = left_projected.get(projected_symbol, 0.0) / left_support
        right_probability = right_projected.get(projected_symbol, 0.0) / right_support
        if abs(left_probability - right_probability) > bound:
            return False, []

    pairs = [pair]
    if recursive:
        left_next = _dominant_next_by_projected_symbol(
            graph,
            counts,
            left,
            symbol_projection,
        )
        right_next = _dominant_next_by_projected_symbol(
            graph,
            counts,
            right,
            symbol_projection,
        )
        for projected_symbol in set(left_next) & set(right_next):
            compatible, child_pairs = _compatible_pair(
                graph,
                counts,
                supports,
                left_next[projected_symbol],
                right_next[projected_symbol],
                alpha=alpha,
                min_support=min_support,
                recursive=True,
                symbol_projection=symbol_projection,
                seen=seen,
            )
            if not compatible:
                return False, []
            pairs.extend(child_pairs)
    return True, pairs


def _build_merged_graph(
    graph: ContextGraph,
    counts: dict[Context, dict[Symbol, float]],
    union: "_UnionFind",
    *,
    alpha: float,
    min_support: float,
    recursive: bool,
    symbol_projection_name: str | None,
) -> ContextGraph:
    states = tuple(sorted(graph.states, key=_context_sort_key))
    root_to_members: dict[Context, list[Context]] = defaultdict(list)
    for state in states:
        root_to_members[union.find(state)].append(state)

    roots = tuple(
        sorted(
            root_to_members,
            key=lambda root: _context_sort_key(root_to_members[root][0]),
        )
    )
    root_to_class = {root: class_id for class_id, root in enumerate(roots)}
    representative_by_root = {
        root: tuple(sorted(members, key=_context_sort_key))[0]
        for root, members in root_to_members.items()
    }

    edges_by_state: dict[Context, list[Edge]] = {}
    merged_counts: dict[Context, dict[Symbol, float]] = {}
    merged_supports: dict[Context, float] = {}
    conflicting_symbol_destinations = 0

    for root in roots:
        representative = representative_by_root[root]
        symbol_counts: Counter[Symbol] = Counter()
        destination_counts: dict[Symbol, Counter[Context]] = defaultdict(Counter)
        order_scores: dict[Symbol, Counter[int]] = defaultdict(Counter)

        for member in root_to_members[root]:
            member_counts = counts.get(member, {})
            for edge in graph.outgoing(member):
                count = member_counts.get(edge.symbol, 0.0)
                if count <= 0.0:
                    continue
                symbol_counts[edge.symbol] += count
                destination_counts[edge.symbol][union.find(edge.next_state)] += count
                for order, weight in edge.order_weights:
                    order_scores[edge.symbol][int(order)] += count * float(weight)

        total = float(sum(symbol_counts.values()))
        merged_supports[representative] = total
        merged_counts[representative] = {
            symbol: float(count)
            for symbol, count in symbol_counts.items()
            if count > 0.0
        }
        if total <= 0.0:
            edges_by_state[representative] = []
            continue

        state_edges: list[Edge] = []
        for symbol, count in sorted(symbol_counts.items(), key=lambda item: _hashable_key(item[0])):
            destinations = destination_counts[symbol]
            if len(destinations) > 1:
                conflicting_symbol_destinations += 1
            destination_root = max(
                destinations,
                key=lambda candidate: (destinations[candidate], repr(candidate)),
            )
            order_weights = ()
            if order_scores[symbol]:
                order_total = float(sum(order_scores[symbol].values()))
                if order_total > 0.0:
                    order_weights = tuple(
                        sorted(
                            (
                                (order, score / order_total)
                                for order, score in order_scores[symbol].items()
                            ),
                            reverse=True,
                        )
                    )
            state_edges.append(
                Edge(
                    symbol=symbol,
                    probability=float(count) / total,
                    next_state=representative_by_root[destination_root],
                    order_weights=order_weights,
                )
            )
        edges_by_state[representative] = state_edges

    start_state = representative_by_root[union.find(graph._resolve_state(graph.start_state))]
    merged = ContextGraph(
        edges_by_state,
        start_state=start_state,
        max_order=graph.max_order,
        alphabet=graph.alphabet,
        validate=True,
    )

    alias_to_state = {
        state: representative_by_root[union.find(state)]
        for state in states
    }
    for alias, resolved in getattr(graph, "_alias_to_state", {}).items():
        alias_to_state[alias] = representative_by_root[union.find(resolved)]
    merged._alias_to_state = alias_to_state
    merged._continuation_counts = merged_counts
    merged._state_supports = merged_supports

    state_to_class = {
        state: root_to_class[union.find(state)]
        for state in states
    }
    classes = tuple(
        tuple(sorted(root_to_members[root], key=_context_sort_key))
        for root in roots
    )
    merged.alergia_metadata = AlergiaMergeMetadata(
        method="alergia",
        alpha=alpha,
        min_support=min_support,
        recursive=recursive,
        original_state_count=len(graph.states),
        merged_state_count=len(merged.states),
        original_edge_count=graph.edge_count(),
        merged_edge_count=merged.edge_count(),
        state_to_class=state_to_class,
        classes=classes,
        conflicting_symbol_destinations=conflicting_symbol_destinations,
        symbol_projection=symbol_projection_name,
    )
    return merged


def _state_counts(graph: ContextGraph) -> dict[Context, dict[Symbol, float]]:
    metadata = getattr(graph, "_continuation_counts", {})
    result: dict[Context, dict[Symbol, float]] = {}
    supports = getattr(graph, "_state_supports", {})
    for state in graph.states:
        if state in metadata:
            result[state] = dict(metadata[state])
            continue
        support = float(supports.get(state, 0.0))
        edges = graph.outgoing(state)
        if support <= 0.0 and edges:
            support = 1.0
        result[state] = {
            edge.symbol: edge.probability * support
            for edge in edges
            if edge.probability > 0.0
        }
    return result


def _projected_counts(
    counts: dict[Symbol, float],
    symbol_projection: Callable[[Symbol], Hashable],
) -> dict[Hashable, float]:
    projected: Counter[Hashable] = Counter()
    for symbol, count in counts.items():
        projected[symbol_projection(symbol)] += count
    return dict(projected)


def _dominant_next_by_projected_symbol(
    graph: ContextGraph,
    counts: dict[Context, dict[Symbol, float]],
    state: Context,
    symbol_projection: Callable[[Symbol], Hashable],
) -> dict[Hashable, Context]:
    candidates: dict[Hashable, Counter[Context]] = defaultdict(Counter)
    state_counts = counts.get(state, {})
    for edge in graph.outgoing(state):
        count = state_counts.get(edge.symbol, 0.0)
        if count <= 0.0:
            continue
        candidates[symbol_projection(edge.symbol)][edge.next_state] += count
    return {
        projected_symbol: max(
            destination_counts,
            key=lambda candidate: (destination_counts[candidate], repr(candidate)),
        )
        for projected_symbol, destination_counts in candidates.items()
    }


def _hoeffding_bound(left_support: float, right_support: float, alpha: float) -> float:
    scale = math.log(2.0 / alpha) / 2.0
    return math.sqrt(scale / left_support) + math.sqrt(scale / right_support)


def _class_roots(union: "_UnionFind", states: tuple[Context, ...]) -> tuple[Context, ...]:
    roots = []
    seen = set()
    for state in states:
        root = union.find(state)
        if root in seen:
            continue
        seen.add(root)
        roots.append(root)
    return tuple(roots)


def _context_sort_key(context: Context) -> tuple[int, str]:
    return (len(context), repr(context))


def _hashable_key(value: Any) -> Hashable:
    try:
        hash(value)
    except TypeError:
        return (type(value).__name__, repr(value))
    return (type(value).__name__, value)


def _identity_projection(symbol: Symbol) -> Hashable:
    return symbol


def _projection_name(symbol_projection: Callable[[Symbol], Hashable] | None) -> str | None:
    if symbol_projection is None:
        return None
    return getattr(symbol_projection, "__name__", repr(symbol_projection))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("inf") if numerator else 1.0
    return float(numerator) / float(denominator)


class _UnionFind:
    def __init__(self, states: tuple[Context, ...]) -> None:
        self.parent = {state: state for state in states}

    def find(self, state: Context) -> Context:
        parent = self.parent[state]
        if parent != state:
            parent = self.find(parent)
            self.parent[state] = parent
        return parent

    def union(self, left: Context, right: Context) -> Context:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if _context_sort_key(right_root) < _context_sort_key(left_root):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        return left_root
