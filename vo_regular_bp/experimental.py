"""Experimental source-model transformations.

The functions in this module are explicit modeling operations. They may change
the source distribution, unlike exact minimization in :mod:`vo_regular_bp.minimization`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
import math
import time
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
    transition_projection: str | None = None
    projection_kind: str = "identity"

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
            "transition_projection": self.transition_projection,
            "projection_kind": self.projection_kind,
        }


@dataclass(frozen=True)
class AlergiaFixedOrderMergeMetadata:
    """Diagnostics for an experimental ALERGIA merge of one fixed-order graph."""

    method: str
    order: int
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
    conflicting_edge_orders: int = 0
    symbol_projection: str | None = None
    transition_projection: str | None = None
    projection_kind: str = "identity"
    merge_time_seconds: float = 0.0

    @property
    def state_compression_ratio(self) -> float:
        return _ratio(self.original_state_count, self.merged_state_count)

    @property
    def edge_compression_ratio(self) -> float:
        return _ratio(self.original_edge_count, self.merged_edge_count)

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "order": self.order,
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
            "conflicting_edge_orders": self.conflicting_edge_orders,
            "symbol_projection": self.symbol_projection,
            "transition_projection": self.transition_projection,
            "projection_kind": self.projection_kind,
            "merge_time_seconds": self.merge_time_seconds,
        }


@dataclass(frozen=True)
class AlergiaOrderStackMergeMetadata:
    """Diagnostics for an experimental ALERGIA-merged order-stack model."""

    method: str
    alpha: float
    min_support: float
    recursive: bool
    orders: dict[int, AlergiaFixedOrderMergeMetadata]

    @property
    def original_state_count(self) -> int:
        return sum(metadata.original_state_count for metadata in self.orders.values())

    @property
    def merged_state_count(self) -> int:
        return sum(metadata.merged_state_count for metadata in self.orders.values())

    @property
    def original_edge_count(self) -> int:
        return sum(metadata.original_edge_count for metadata in self.orders.values())

    @property
    def merged_edge_count(self) -> int:
        return sum(metadata.merged_edge_count for metadata in self.orders.values())

    @property
    def merge_time_seconds(self) -> float:
        return sum(metadata.merge_time_seconds for metadata in self.orders.values())

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
            "merge_time_seconds": self.merge_time_seconds,
            "orders": {
                order: metadata.as_dict()
                for order, metadata in sorted(self.orders.items())
            },
        }


@dataclass(frozen=True)
class _ProjectionCache:
    counts: dict[Context, dict[Hashable, float]]
    dominant_next: dict[Context, dict[Hashable, Context]]


_PairMemo = dict[tuple[Context, Context], tuple[bool, tuple[tuple[Context, Context], ...]]]


def alergia_merge(
    graph: ContextGraph,
    *,
    alpha: float = 0.01,
    min_support: int | float = 10,
    recursive: bool = True,
    symbol_projection: Callable[[Symbol], Hashable] | None = None,
    transition_projection: Callable[[Context, Symbol, Edge], Hashable] | None = None,
) -> ContextGraph:
    """Return an ALERGIA-like approximate source-state merge.

    This function acts only on the unconstrained source graph. ``symbol_projection``
    lets a caller compare emitted symbols through a feature map. For
    context-aware semantics, ``transition_projection`` can instead map
    ``(state, symbol, edge)`` to a comparison feature, for example an interval
    from the current context to the emitted symbol. If both projections are
    supplied, ``transition_projection`` takes precedence. The returned graph
    still emits concrete symbols and remains a regular :class:`ContextGraph`,
    so existing BP remains exact with respect to the new merged source model.
    """

    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if min_support < 0:
        raise ValueError("min_support must be non-negative")

    counts = _state_counts(graph)
    project = _transition_projector(
        symbol_projection=symbol_projection,
        transition_projection=transition_projection,
    )
    projection_cache = _precompute_projection_cache(graph, counts, project)
    supports = {
        state: float(sum(symbol_counts.values()))
        for state, symbol_counts in counts.items()
    }
    states = tuple(sorted(graph.states, key=_context_sort_key))
    classes = _ClassRegistry(states)
    pair_memo: _PairMemo = {}

    for state in states:
        if supports.get(state, 0.0) < min_support:
            continue
        for root in classes.roots():
            if classes.find(state) == root:
                break
            members = classes.members(root)
            compatible, pairs = _compatible_with_class(
                graph,
                counts,
                supports,
                state,
                members,
                alpha=float(alpha),
                min_support=float(min_support),
                recursive=recursive,
                projection_cache=projection_cache,
                pair_memo=pair_memo,
            )
            if compatible:
                classes.union(state, members[0])
                for left, right in pairs:
                    classes.union(left, right)
                break

    return _build_merged_graph(
        graph,
        counts,
        classes.union_find,
        alpha=float(alpha),
        min_support=float(min_support),
        recursive=recursive,
        symbol_projection_name=_projection_name(symbol_projection),
        transition_projection_name=_projection_name(transition_projection),
        projection_kind=_projection_kind(symbol_projection, transition_projection),
    )


def alergia_metadata(graph: object) -> (
    AlergiaMergeMetadata
    | AlergiaFixedOrderMergeMetadata
    | AlergiaOrderStackMergeMetadata
    | None
):
    """Return ALERGIA metadata if ``graph`` was produced by an experimental merge."""

    return getattr(graph, "alergia_metadata", None)


def alergia_merge_fixed_order_graph(
    graph: object,
    *,
    alpha: float = 0.01,
    min_support: int | float = 10,
    recursive: bool = True,
    symbol_projection: Callable[[Symbol], Hashable] | None = None,
    transition_projection: Callable[[Context, Symbol, Edge], Hashable] | None = None,
    continuation_counts: Mapping[Context, Mapping[Symbol, int | float]] | None = None,
) -> object:
    """Return an experimental ALERGIA-merged fixed-order order-stack graph.

    The merge is applied to one unconstrained fixed-order source graph. The
    result is still a ``FixedOrderContextGraph`` and can be used by the existing
    order-stack BP backends without a special sampling path.
    """

    started = time.perf_counter()
    context_graph = _fixed_order_to_context_graph(
        graph,
        continuation_counts=continuation_counts,
    )
    merged_context_graph = alergia_merge(
        context_graph,
        alpha=alpha,
        min_support=min_support,
        recursive=recursive,
        symbol_projection=symbol_projection,
        transition_projection=transition_projection,
    )
    merged_graph, conflicting_edge_orders = _context_graph_to_fixed_order_graph(
        merged_context_graph,
        graph,
    )
    source_metadata = alergia_metadata(merged_context_graph)
    if not isinstance(source_metadata, AlergiaMergeMetadata):
        raise RuntimeError("missing ContextGraph ALERGIA metadata")
    merged_graph.alergia_metadata = AlergiaFixedOrderMergeMetadata(
        method="alergia_fixed_order",
        order=int(graph.order),
        alpha=float(alpha),
        min_support=float(min_support),
        recursive=recursive,
        original_state_count=source_metadata.original_state_count,
        merged_state_count=source_metadata.merged_state_count,
        original_edge_count=source_metadata.original_edge_count,
        merged_edge_count=source_metadata.merged_edge_count,
        state_to_class=source_metadata.state_to_class,
        classes=source_metadata.classes,
        conflicting_symbol_destinations=source_metadata.conflicting_symbol_destinations,
        conflicting_edge_orders=conflicting_edge_orders,
        symbol_projection=source_metadata.symbol_projection,
        transition_projection=source_metadata.transition_projection,
        projection_kind=source_metadata.projection_kind,
        merge_time_seconds=time.perf_counter() - started,
    )
    return merged_graph


def alergia_merge_order_stack_model(
    model: object,
    *,
    alpha: float = 0.01,
    min_support: int | float = 10,
    recursive: bool = True,
    symbol_projection: Callable[[Symbol], Hashable] | None = None,
    transition_projection: Callable[[Context, Symbol, Edge], Hashable] | None = None,
) -> "AlergiaOrderStackModel":
    """Return an experimental order-stack model with ALERGIA-merged sources.

    Each fixed-order graph is merged independently. The returned model exposes
    the same ``compile_graph(order)`` interface used by the existing order-stack
    BP preparation functions, so sampling APIs do not need a special path.
    """

    graphs = {}
    metadata_by_order = {}
    for order in range(1, int(model.max_order) + 1):
        graph = model.compile_graph(order)
        merged_graph = alergia_merge_fixed_order_graph(
            graph,
            alpha=alpha,
            min_support=min_support,
            recursive=recursive,
            symbol_projection=symbol_projection,
            transition_projection=transition_projection,
            continuation_counts=_fixed_order_counts_from_model(model, graph),
        )
        graphs[order] = merged_graph
        metadata = alergia_metadata(merged_graph)
        if not isinstance(metadata, AlergiaFixedOrderMergeMetadata):
            raise RuntimeError("missing fixed-order ALERGIA metadata")
        metadata_by_order[order] = metadata

    return AlergiaOrderStackModel(
        base_model=model,
        graphs=graphs,
        metadata=AlergiaOrderStackMergeMetadata(
            method="alergia_order_stack",
            alpha=float(alpha),
            min_support=float(min_support),
            recursive=recursive,
            orders=metadata_by_order,
        ),
    )


class AlergiaOrderStackModel:
    """Experimental order-stack model backed by ALERGIA-merged fixed-order graphs."""

    def __init__(
        self,
        *,
        base_model: object,
        graphs: dict[int, object],
        metadata: AlergiaOrderStackMergeMetadata,
    ) -> None:
        self.base_model = base_model
        self.max_order = int(base_model.max_order)
        self.forbidden_symbols = frozenset(getattr(base_model, "forbidden_symbols", ()))
        self.alphabet = frozenset(getattr(base_model, "alphabet", ()))
        self._graphs = dict(graphs)
        self.alergia_metadata = metadata

    def compile_graph(self, order: int) -> object:
        if order < 1 or order > self.max_order:
            raise ValueError(f"order must be between 1 and {self.max_order}")
        return self._graphs[int(order)]

    def compile_graphs_for_plan(self, *, length: int) -> dict[int, object]:
        return dict(self._graphs)


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
    projection_cache: _ProjectionCache,
    pair_memo: _PairMemo,
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
            projection_cache=projection_cache,
            pair_memo=pair_memo,
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
    projection_cache: _ProjectionCache,
    pair_memo: _PairMemo,
    seen: set[tuple[Context, Context]],
) -> tuple[bool, list[tuple[Context, Context]]]:
    if left == right:
        return True, []
    pair = (left, right) if _context_sort_key(left) <= _context_sort_key(right) else (right, left)
    if pair in seen:
        return True, []
    cache_positive = not seen
    cached = pair_memo.get(pair)
    if cached is not None:
        compatible, pairs = cached
        if not compatible or cache_positive:
            return compatible, list(pairs)
    seen.add(pair)

    left_counts = counts.get(left, {})
    right_counts = counts.get(right, {})
    left_support = supports.get(left, 0.0)
    right_support = supports.get(right, 0.0)
    if left_support == 0.0 and right_support == 0.0 and not left_counts and not right_counts:
        return _cache_pair_result(pair_memo, pair, True, [pair], cache_positive)
    if left_support <= 0.0 or right_support <= 0.0:
        return _cache_pair_result(pair_memo, pair, False, [], cache_positive)
    if left_support < min_support or right_support < min_support:
        return _cache_pair_result(pair_memo, pair, False, [], cache_positive)

    left_projected = projection_cache.counts.get(left, {})
    right_projected = projection_cache.counts.get(right, {})
    bound = _hoeffding_bound(left_support, right_support, alpha)
    for projected_symbol in set(left_projected) | set(right_projected):
        left_probability = left_projected.get(projected_symbol, 0.0) / left_support
        right_probability = right_projected.get(projected_symbol, 0.0) / right_support
        if abs(left_probability - right_probability) > bound:
            return _cache_pair_result(pair_memo, pair, False, [], cache_positive)

    pairs = [pair]
    if recursive:
        left_next = projection_cache.dominant_next.get(left, {})
        right_next = projection_cache.dominant_next.get(right, {})
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
                projection_cache=projection_cache,
                pair_memo=pair_memo,
                seen=seen,
            )
            if not compatible:
                return _cache_pair_result(pair_memo, pair, False, [], cache_positive)
            pairs.extend(child_pairs)
    return _cache_pair_result(pair_memo, pair, True, pairs, cache_positive)


def _build_merged_graph(
    graph: ContextGraph,
    counts: dict[Context, dict[Symbol, float]],
    union: "_UnionFind",
    *,
    alpha: float,
    min_support: float,
    recursive: bool,
    symbol_projection_name: str | None,
    transition_projection_name: str | None,
    projection_kind: str,
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
        transition_projection=transition_projection_name,
        projection_kind=projection_kind,
    )
    return merged


def _fixed_order_to_context_graph(
    graph: object,
    *,
    continuation_counts: Mapping[Context, Mapping[Symbol, int | float]] | None,
) -> ContextGraph:
    contexts = tuple(graph.contexts)
    edges_by_state: dict[Context, list[Edge]] = {}
    count_metadata: dict[Context, dict[Symbol, float]] = {}
    state_supports: dict[Context, float] = {}

    for state_id, context in enumerate(contexts):
        raw_counts = continuation_counts.get(context) if continuation_counts is not None else None
        support = (
            float(sum(count for count in raw_counts.values() if count > 0))
            if raw_counts is not None
            else 1.0
        )
        if support <= 0.0 and graph.outgoing[state_id]:
            support = 1.0
        state_counts: dict[Symbol, float] = {}
        state_edges: list[Edge] = []
        for edge in graph.outgoing[state_id]:
            count = (
                float(raw_counts.get(edge.symbol, edge.probability * support))
                if raw_counts is not None
                else edge.probability * support
            )
            if count > 0.0:
                state_counts[edge.symbol] = count
            state_edges.append(
                Edge(
                    symbol=edge.symbol,
                    probability=edge.probability,
                    next_state=contexts[edge.dst],
                    order_weights=((int(edge.order), 1.0),),
                )
            )
        edges_by_state[context] = state_edges
        count_metadata[context] = state_counts
        state_supports[context] = float(sum(state_counts.values()))

    start_state = contexts[0] if contexts else ()
    context_graph = ContextGraph(
        edges_by_state,
        start_state=start_state,
        max_order=int(graph.order),
        validate=True,
    )
    alias_to_state = {
        alias: contexts[state_id]
        for alias, state_id in graph.context_to_id.items()
        if state_id < len(contexts)
    }
    for state_id, aliases in enumerate(getattr(graph, "state_aliases", ())):
        if state_id >= len(contexts):
            continue
        for alias in aliases:
            alias_to_state[alias] = contexts[state_id]
    context_graph._alias_to_state = alias_to_state
    context_graph._continuation_counts = count_metadata
    context_graph._state_supports = state_supports
    return context_graph


def _context_graph_to_fixed_order_graph(
    merged: ContextGraph,
    source_graph: object,
) -> tuple[object, int]:
    from .order_stack_bp import FixedOrderContextGraph, StackEdge

    metadata = alergia_metadata(merged)
    if not isinstance(metadata, AlergiaMergeMetadata):
        raise RuntimeError("missing ContextGraph ALERGIA metadata")

    fixed = FixedOrderContextGraph(int(source_graph.order))
    representatives = [class_members[0] for class_members in metadata.classes]
    fixed.contexts = representatives
    fixed.context_to_id = {
        context: state_id
        for state_id, context in enumerate(representatives)
    }
    fixed.outgoing = [[] for _context in representatives]

    aliases: list[set[Context]] = [set(class_members) for class_members in metadata.classes]
    for alias, resolved in getattr(merged, "_alias_to_state", {}).items():
        state_id = fixed.context_to_id.get(resolved)
        if state_id is None:
            continue
        fixed.context_to_id[alias] = state_id
        aliases[state_id].add(alias)
    fixed.state_aliases = tuple(
        tuple(sorted(group, key=_context_sort_key))
        for group in aliases
    )

    conflicting_edge_orders = 0
    for src, context in enumerate(fixed.contexts):
        for edge in merged.outgoing(context):
            order, has_conflict = _dominant_order(edge.order_weights, int(source_graph.order))
            if has_conflict:
                conflicting_edge_orders += 1
            fixed.outgoing[src].append(
                StackEdge(
                    src=src,
                    dst=fixed.context_to_id[edge.next_state],
                    symbol=edge.symbol,
                    probability=edge.probability,
                    order=order,
                )
            )
    return fixed, conflicting_edge_orders


def _fixed_order_counts_from_model(
    model: object,
    graph: object,
) -> dict[Context, dict[Symbol, float]] | None:
    model_counts = getattr(model, "counts", None)
    longest_available_suffix = getattr(model, "longest_available_suffix", None)
    if not isinstance(model_counts, Mapping) or longest_available_suffix is None:
        return None

    result: dict[Context, dict[Symbol, float]] = {}
    for state_id, context in enumerate(graph.contexts):
        suffix = longest_available_suffix(context, max_order=int(graph.order))
        suffix_counts = model_counts.get(suffix, {}) if suffix is not None else {}
        state_counts = {
            edge.symbol: float(suffix_counts.get(edge.symbol, 0.0))
            for edge in graph.outgoing[state_id]
            if suffix_counts.get(edge.symbol, 0.0) > 0.0
        }
        if not state_counts and graph.outgoing[state_id]:
            state_counts = {
                edge.symbol: float(edge.probability)
                for edge in graph.outgoing[state_id]
                if edge.probability > 0.0
            }
        result[context] = state_counts
    return result


def _dominant_order(
    order_weights: tuple[tuple[int, float], ...],
    default_order: int,
) -> tuple[int, bool]:
    if not order_weights:
        return default_order, False
    order, _weight = max(order_weights, key=lambda item: (item[1], item[0]))
    return int(order), len(order_weights) > 1


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


def _precompute_projection_cache(
    graph: ContextGraph,
    counts: dict[Context, dict[Symbol, float]],
    transition_projection: Callable[[Context, Symbol, Edge], Hashable],
) -> _ProjectionCache:
    projected_counts: dict[Context, dict[Hashable, float]] = {}
    dominant_next: dict[Context, dict[Hashable, Context]] = {}
    for state in graph.states:
        state_projected: Counter[Hashable] = Counter()
        destination_counts: dict[Hashable, Counter[Context]] = defaultdict(Counter)
        state_counts = counts.get(state, {})
        for edge in graph.outgoing(state):
            count = state_counts.get(edge.symbol, 0.0)
            if count <= 0.0:
                continue
            projected_symbol = transition_projection(state, edge.symbol, edge)
            state_projected[projected_symbol] += count
            destination_counts[projected_symbol][edge.next_state] += count
        projected_counts[state] = dict(state_projected)
        dominant_next[state] = {
            projected_symbol: max(
                destinations,
                key=lambda candidate: (destinations[candidate], repr(candidate)),
            )
            for projected_symbol, destinations in destination_counts.items()
        }
    return _ProjectionCache(counts=projected_counts, dominant_next=dominant_next)


def _cache_pair_result(
    pair_memo: _PairMemo,
    pair: tuple[Context, Context],
    compatible: bool,
    pairs: list[tuple[Context, Context]],
    cache_positive: bool,
) -> tuple[bool, list[tuple[Context, Context]]]:
    frozen_pairs = tuple(pairs)
    if not compatible or cache_positive:
        pair_memo[pair] = (compatible, frozen_pairs)
    return compatible, list(frozen_pairs)


def _hoeffding_bound(left_support: float, right_support: float, alpha: float) -> float:
    scale = math.log(2.0 / alpha) / 2.0
    return math.sqrt(scale / left_support) + math.sqrt(scale / right_support)


class _ClassRegistry:
    def __init__(self, states: tuple[Context, ...]) -> None:
        self.union_find = _UnionFind(states)
        self._state_index = {state: index for index, state in enumerate(states)}
        self._active_roots = list(states)
        self._members_by_root = {state: [state] for state in states}

    def find(self, state: Context) -> Context:
        return self.union_find.find(state)

    def roots(self) -> tuple[Context, ...]:
        return tuple(self._active_roots)

    def members(self, root: Context) -> list[Context]:
        return self._members_by_root[self.find(root)]

    def union(self, left: Context, right: Context) -> Context:
        left_root = self.union_find.find(left)
        right_root = self.union_find.find(right)
        if left_root == right_root:
            return left_root

        root = self.union_find.union(left_root, right_root)
        removed_root = right_root if root == left_root else left_root
        self._members_by_root[root] = self._merge_members(
            self._members_by_root[root],
            self._members_by_root.pop(removed_root),
        )
        self._active_roots.remove(removed_root)
        return root

    def _merge_members(self, left: list[Context], right: list[Context]) -> list[Context]:
        merged: list[Context] = []
        left_index = 0
        right_index = 0
        while left_index < len(left) and right_index < len(right):
            if self._state_index[left[left_index]] <= self._state_index[right[right_index]]:
                merged.append(left[left_index])
                left_index += 1
            else:
                merged.append(right[right_index])
                right_index += 1
        merged.extend(left[left_index:])
        merged.extend(right[right_index:])
        return merged


def _context_sort_key(context: Context) -> tuple[int, str]:
    return (len(context), repr(context))


def _hashable_key(value: Any) -> Hashable:
    try:
        hash(value)
    except TypeError:
        return (type(value).__name__, repr(value))
    return (type(value).__name__, value)


def _identity_projection(_state: Context, symbol: Symbol, _edge: Edge) -> Hashable:
    return symbol


def _transition_projector(
    *,
    symbol_projection: Callable[[Symbol], Hashable] | None,
    transition_projection: Callable[[Context, Symbol, Edge], Hashable] | None,
) -> Callable[[Context, Symbol, Edge], Hashable]:
    if transition_projection is not None:
        return transition_projection
    if symbol_projection is not None:
        return lambda _state, symbol, _edge: symbol_projection(symbol)
    return _identity_projection


def _projection_name(projection: Callable[..., Hashable] | None) -> str | None:
    if projection is None:
        return None
    return getattr(projection, "__name__", repr(projection))


def _projection_kind(
    symbol_projection: Callable[[Symbol], Hashable] | None,
    transition_projection: Callable[[Context, Symbol, Edge], Hashable] | None,
) -> str:
    if transition_projection is not None:
        return "transition"
    if symbol_projection is not None:
        return "symbol"
    return "identity"


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
