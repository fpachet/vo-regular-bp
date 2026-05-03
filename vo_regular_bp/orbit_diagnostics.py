"""Diagnostics for transformation-orbit structure in regular BP products."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .context import Symbol
from .order_stack_bp import RegularOrderStackBPResult
from .positional_bp import PositionConstraint


@dataclass(frozen=True)
class ForbiddenPatternOrbitStats:
    """Orbit counts for forbidden patterns and their DFA prefix states."""

    pattern_count: int
    pattern_orbit_count: int
    prefix_state_count: int
    prefix_state_orbit_count: int
    alphabet_size: int | None = None
    alphabet_orbit_count: int | None = None

    @property
    def pattern_reduction_factor(self) -> float:
        return _ratio(self.pattern_count, self.pattern_orbit_count)

    @property
    def prefix_state_reduction_factor(self) -> float:
        return _ratio(self.prefix_state_count, self.prefix_state_orbit_count)

    @property
    def alphabet_reduction_factor(self) -> float | None:
        if self.alphabet_size is None or self.alphabet_orbit_count is None:
            return None
        return _ratio(self.alphabet_size, self.alphabet_orbit_count)

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "forbidden_patterns": self.pattern_count,
            "forbidden_pattern_orbits": self.pattern_orbit_count,
            "forbidden_pattern_reduction": self.pattern_reduction_factor,
            "dfa_prefix_states": self.prefix_state_count,
            "dfa_prefix_state_orbits": self.prefix_state_orbit_count,
            "dfa_prefix_state_reduction": self.prefix_state_reduction_factor,
            "alphabet_size": self.alphabet_size,
            "alphabet_orbits": self.alphabet_orbit_count,
            "alphabet_reduction": self.alphabet_reduction_factor,
        }


@dataclass(frozen=True)
class RegularProductOrbitStats:
    """Orbit counts for the currently expanded regular BP product."""

    time_indexed_product_states: int
    time_indexed_product_state_orbits: int
    product_states: int
    product_state_orbits: int
    product_edges: int
    time_indexed_product_edge_orbits: int
    product_edge_orbits: int
    expanded_rows: int
    transition_row_shape_orbits: int
    max_time_state_orbit_size: int
    max_product_state_orbit_size: int
    max_time_edge_orbit_size: int
    max_edge_orbit_size: int

    @property
    def time_indexed_product_state_reduction_factor(self) -> float:
        return _ratio(
            self.time_indexed_product_states,
            self.time_indexed_product_state_orbits,
        )

    @property
    def product_state_reduction_factor(self) -> float:
        return _ratio(self.product_states, self.product_state_orbits)

    @property
    def time_indexed_product_edge_reduction_factor(self) -> float:
        return _ratio(self.product_edges, self.time_indexed_product_edge_orbits)

    @property
    def product_edge_reduction_factor(self) -> float:
        return _ratio(self.product_edges, self.product_edge_orbits)

    @property
    def transition_row_shape_reduction_factor(self) -> float:
        return _ratio(self.expanded_rows, self.transition_row_shape_orbits)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "time_indexed_product_states": self.time_indexed_product_states,
            "time_indexed_product_state_orbits": self.time_indexed_product_state_orbits,
            "time_indexed_product_state_reduction": (
                self.time_indexed_product_state_reduction_factor
            ),
            "product_states": self.product_states,
            "product_state_orbits": self.product_state_orbits,
            "product_state_reduction": self.product_state_reduction_factor,
            "product_edges": self.product_edges,
            "time_indexed_product_edge_orbits": self.time_indexed_product_edge_orbits,
            "time_indexed_product_edge_reduction": (
                self.time_indexed_product_edge_reduction_factor
            ),
            "product_edge_orbits": self.product_edge_orbits,
            "product_edge_reduction": self.product_edge_reduction_factor,
            "expanded_rows": self.expanded_rows,
            "transition_row_shape_orbits": self.transition_row_shape_orbits,
            "transition_row_shape_reduction": self.transition_row_shape_reduction_factor,
            "max_time_state_orbit_size": self.max_time_state_orbit_size,
            "max_product_state_orbit_size": self.max_product_state_orbit_size,
            "max_time_edge_orbit_size": self.max_time_edge_orbit_size,
            "max_edge_orbit_size": self.max_edge_orbit_size,
        }


def canonical_integer_shift_key(
    value: Any,
    *,
    fixed_symbols: Iterable[Symbol] = (),
    reference: int | None = None,
) -> Hashable:
    """Canonicalize nested symbols modulo a common integer shift.

    Integer-like symbols are represented relative to the first integer symbol
    unless ``reference`` is supplied. Non-integer symbols, such as sentinels,
    are kept fixed. The function is intentionally diagnostic: it exposes how
    much repeated structure is present under transposition without changing any
    BP semantics.
    """

    fixed = frozenset(fixed_symbols)
    origin = reference
    if origin is None:
        first = _first_shiftable_symbol(value, fixed)
        origin = 0 if first is None else first
    return _canonicalize(value, fixed, int(origin))


def forbidden_pattern_orbit_stats(
    patterns: Iterable[Iterable[Symbol]],
    *,
    alphabet: Iterable[Symbol] | None = None,
    fixed_symbols: Iterable[Symbol] = (),
) -> ForbiddenPatternOrbitStats:
    """Measure integer-shift orbit structure in forbidden n-grams."""

    fixed = frozenset(fixed_symbols)
    pattern_tuple = tuple(dict.fromkeys(tuple(pattern) for pattern in patterns))
    prefixes: set[tuple[Symbol, ...]] = {()}
    for pattern in pattern_tuple:
        for prefix_len in range(1, len(pattern)):
            prefixes.add(pattern[:prefix_len])

    pattern_orbits = {
        canonical_integer_shift_key(pattern, fixed_symbols=fixed)
        for pattern in pattern_tuple
    }
    prefix_orbits = {
        canonical_integer_shift_key(prefix, fixed_symbols=fixed)
        for prefix in prefixes
    }

    alphabet_size = None
    alphabet_orbit_count = None
    if alphabet is not None:
        alphabet_tuple = tuple(dict.fromkeys(alphabet))
        alphabet_size = len(alphabet_tuple)
        alphabet_orbit_count = len(
            {
                canonical_integer_shift_key(symbol, fixed_symbols=fixed)
                for symbol in alphabet_tuple
            }
        )

    return ForbiddenPatternOrbitStats(
        pattern_count=len(pattern_tuple),
        pattern_orbit_count=len(pattern_orbits),
        prefix_state_count=len(prefixes),
        prefix_state_orbit_count=len(prefix_orbits),
        alphabet_size=alphabet_size,
        alphabet_orbit_count=alphabet_orbit_count,
    )


def regular_product_orbit_stats(
    result: RegularOrderStackBPResult,
    *,
    fixed_symbols: Iterable[Symbol] = (),
    include_transition_row_shapes: bool = False,
) -> RegularProductOrbitStats:
    """Measure transposition-orbit repetition in an already-run regular BP."""

    fixed = frozenset(fixed_symbols)
    result.start_order_masses()

    time_state_counts: Counter[Hashable] = Counter()
    state_counts: Counter[Hashable] = Counter()
    time_edge_counts: Counter[Hashable] = Counter()
    edge_counts: Counter[Hashable] = Counter()
    row_shapes: set[Hashable] = set()
    state_seen: set[Hashable] = set()
    state_key_cache: dict[tuple[int, int, Hashable], Hashable] = {}
    acceptor_payload_cache: dict[Hashable, Hashable] = {}
    edge_key_cache: dict[tuple[int, int, Hashable, int, Symbol, Hashable], Hashable] = {}
    scanned_edge_count = 0
    expanded_rows = 0

    for order, cache in result.backwards.items():
        graph = cache.graph
        for time, graph_state, acceptor_state in cache.memo:
            context = graph.contexts[graph_state]
            acceptor_payload = acceptor_payload_cache.get(acceptor_state)
            if acceptor_payload is None:
                acceptor_payload = _acceptor_state_payload(result.acceptor, acceptor_state)
                acceptor_payload_cache[acceptor_state] = acceptor_payload

            raw_state_key = (order, graph_state, acceptor_state)
            state_key = state_key_cache.get(raw_state_key)
            if state_key is None:
                state_payload = (context, acceptor_payload)
                state_key = (
                    order,
                    canonical_integer_shift_key(state_payload, fixed_symbols=fixed),
                )
                state_key_cache[raw_state_key] = state_key
            time_state_key = (time, *state_key)
            time_state_counts[time_state_key] += 1
            if raw_state_key not in state_seen:
                state_seen.add(raw_state_key)
                state_counts[state_key] += 1

            if time == result.length:
                continue

            expanded_rows += 1
            row_edges: list[Hashable] = []
            reference = None
            if include_transition_row_shapes:
                reference = _first_shiftable_symbol((context, acceptor_payload), fixed)
            for transition in cache.accepted_transitions(graph_state, acceptor_state):
                edge = transition.edge
                if not _constraint_allows(result.constraints.get(time), edge.symbol):
                    continue
                next_acceptor_state = transition.next_acceptor_state
                edge_cache_key = (
                    order,
                    graph_state,
                    acceptor_state,
                    edge.dst,
                    edge.symbol,
                    next_acceptor_state,
                )
                edge_key = edge_key_cache.get(edge_cache_key)
                if edge_key is None:
                    next_payload = acceptor_payload_cache.get(next_acceptor_state)
                    if next_payload is None:
                        next_payload = _acceptor_state_payload(
                            result.acceptor,
                            next_acceptor_state,
                        )
                        acceptor_payload_cache[next_acceptor_state] = next_payload
                    dst_context = graph.contexts[edge.dst]
                    edge_payload = (
                        context,
                        acceptor_payload,
                        edge.symbol,
                        dst_context,
                        next_payload,
                        edge.probability,
                    )
                    edge_key = (
                        order,
                        canonical_integer_shift_key(edge_payload, fixed_symbols=fixed),
                    )
                    edge_key_cache[edge_cache_key] = edge_key
                time_edge_key = (time, *edge_key)
                edge_counts[edge_key] += 1
                time_edge_counts[time_edge_key] += 1
                scanned_edge_count += 1

                if include_transition_row_shapes:
                    next_payload = acceptor_payload_cache[next_acceptor_state]
                    dst_context = graph.contexts[edge.dst]
                    row_reference = reference
                    if row_reference is None:
                        row_reference = _first_shiftable_symbol(
                            (edge.symbol, dst_context, next_payload),
                            fixed,
                        )
                    row_edges.append(
                        canonical_integer_shift_key(
                            (edge.symbol, dst_context, next_payload, edge.probability),
                            fixed_symbols=fixed,
                            reference=row_reference,
                        )
                    )

            if include_transition_row_shapes:
                row_key = (
                    order,
                    canonical_integer_shift_key(
                        (context, acceptor_payload),
                        fixed_symbols=fixed,
                        reference=reference,
                    ),
                    tuple(sorted(row_edges, key=repr)),
                )
                row_shapes.add(row_key)

    if scanned_edge_count != result.product_edge_count:
        raise RuntimeError(
            "orbit diagnostic edge scan did not match BP product-edge count: "
            f"{scanned_edge_count} != {result.product_edge_count}"
        )

    return RegularProductOrbitStats(
        time_indexed_product_states=result.time_indexed_product_state_count,
        time_indexed_product_state_orbits=len(time_state_counts),
        product_states=result.product_state_count,
        product_state_orbits=len(state_counts),
        product_edges=result.product_edge_count,
        time_indexed_product_edge_orbits=len(time_edge_counts),
        product_edge_orbits=len(edge_counts),
        expanded_rows=expanded_rows,
        transition_row_shape_orbits=len(row_shapes),
        max_time_state_orbit_size=_max_count(time_state_counts),
        max_product_state_orbit_size=_max_count(state_counts),
        max_time_edge_orbit_size=_max_count(time_edge_counts),
        max_edge_orbit_size=_max_count(edge_counts),
    )


def _acceptor_state_payload(acceptor: Any, state: Hashable) -> Hashable:
    prefixes = getattr(acceptor, "prefixes", None)
    if prefixes is not None and isinstance(state, int) and 0 <= state < len(prefixes):
        return tuple(prefixes[state])
    return ("dfa_state", state)


def _constraint_allows(constraint: PositionConstraint | None, symbol: Symbol) -> bool:
    if constraint is None:
        return True
    if callable(constraint):
        return bool(constraint(symbol))
    return symbol in constraint


def _canonicalize(value: Any, fixed_symbols: frozenset[Symbol], reference: int) -> Hashable:
    if _is_shiftable_integer(value, fixed_symbols):
        return ("rel_int", int(value) - reference)
    if isinstance(value, tuple):
        return tuple(_canonicalize(item, fixed_symbols, reference) for item in value)
    if isinstance(value, list):
        return tuple(_canonicalize(item, fixed_symbols, reference) for item in value)
    return ("fixed", value)


def _first_shiftable_symbol(value: Any, fixed_symbols: frozenset[Symbol]) -> int | None:
    if _is_shiftable_integer(value, fixed_symbols):
        return int(value)
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_shiftable_symbol(item, fixed_symbols)
            if found is not None:
                return found
    return None


def _is_shiftable_integer(value: Any, fixed_symbols: frozenset[Symbol]) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value not in fixed_symbols


def _max_count(counter: Mapping[Hashable, int]) -> int:
    return max(counter.values(), default=0)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
