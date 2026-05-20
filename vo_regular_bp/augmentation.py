"""Virtual data augmentation for order-stack models.

The classes here keep the generated symbol space absolute.  Transformations are
used only to evaluate continuation counts as if transformed copies of the
training sequences had been materialized.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
import time

from .context import Context, Symbol, _as_context
from .order_stack_bp import FixedOrderContextGraph, OrderStackModel
from .order_stack_bp import StackEdge


@dataclass(frozen=True)
class SymbolTransform:
    """A deterministic finite augmentation transform over symbols."""

    name: str
    apply_symbol: Callable[[Symbol], Symbol]
    inverse_symbol: Callable[[Symbol], Symbol]
    weight: float = 1.0

    def apply_context(self, context: Sequence[Symbol]) -> Context:
        return tuple(self.apply_symbol(symbol) for symbol in context)

    def inverse_context(self, context: Sequence[Symbol]) -> Context:
        return tuple(self.inverse_symbol(symbol) for symbol in context)


def integer_shift_transform(
    offset: int,
    *,
    fixed_symbols: Iterable[Symbol] = (),
    name: str | None = None,
) -> SymbolTransform:
    """Return a symbol transform that adds an integer offset.

    ``fixed_symbols`` are left unchanged.  This is useful for start/end sentinel
    symbols in pitch models.
    """

    fixed = frozenset(fixed_symbols)
    shift = int(offset)

    def apply_symbol(symbol: Symbol) -> Symbol:
        if symbol in fixed:
            return symbol
        return int(symbol) + shift

    def inverse_symbol(symbol: Symbol) -> Symbol:
        if symbol in fixed:
            return symbol
        return int(symbol) - shift

    return SymbolTransform(
        name=name or f"shift_{shift:+d}",
        apply_symbol=apply_symbol,
        inverse_symbol=inverse_symbol,
    )


def integer_shift_transforms(
    offsets: Iterable[int],
    *,
    fixed_symbols: Iterable[Symbol] = (),
) -> tuple[SymbolTransform, ...]:
    """Return integer-shift transforms for all requested offsets."""

    return tuple(
        integer_shift_transform(offset, fixed_symbols=fixed_symbols)
        for offset in offsets
    )


def materialize_transformed_sequences(
    sequences: Iterable[Sequence[Symbol]],
    transforms: Iterable[SymbolTransform],
) -> tuple[tuple[Symbol, ...], ...]:
    """Materialize transformed copies for validation and comparison."""

    material = tuple(tuple(sequence) for sequence in sequences)
    return tuple(
        transform.apply_context(sequence)
        for transform in transforms
        for sequence in material
    )


@dataclass
class VirtualAugmentedOrderStackModel:
    """Order-stack model with exact virtual transformed-corpus counts.

    The model answers continuation queries with the same counts as an explicit
    augmented corpus:

    ``C_aug(c -> y) = sum_g C_base(g^-1(c) -> g^-1(y))``.
    """

    base_model: OrderStackModel
    transforms: tuple[SymbolTransform, ...]
    _counts_cache: dict[Context, Counter[Symbol]] = field(default_factory=dict, init=False)
    _distribution_cache: dict[tuple[Context, int | None], tuple[tuple[tuple[Symbol, float], ...], int | None]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _graph_cache: dict[int, FixedOrderContextGraph] = field(default_factory=dict, init=False)
    _contexts_cache: dict[int, tuple[Context, ...]] = field(default_factory=dict, init=False)
    _lazy_graph_cache: dict[tuple[Context, int], dict[int, "LazyVirtualFixedOrderContextGraph"]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _lazy_plan_graph_cache: dict[int, dict[int, "LazyVirtualFixedOrderContextGraph"]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _apply_symbol_cache: dict[tuple[int, Symbol], Symbol] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _inverse_symbol_cache: dict[tuple[int, Symbol], Symbol] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _inverse_context_cache: dict[tuple[int, Context], Context] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _context_materialization_seconds: float = field(default=0.0, init=False, repr=False)
    _context_materialization_calls: int = field(default=0, init=False, repr=False)
    _context_materialization_cache_hits: int = field(default=0, init=False, repr=False)
    _context_materialization_cache_misses: int = field(default=0, init=False, repr=False)
    _augmented_count_calls: int = field(default=0, init=False, repr=False)
    _augmented_count_cache_hits: int = field(default=0, init=False, repr=False)
    _augmented_count_cache_misses: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.transforms:
            raise ValueError("at least one transform is required")
        if any(transform.weight <= 0.0 for transform in self.transforms):
            raise ValueError("transform weights must be positive")
        self.max_order = self.base_model.max_order
        self.forbidden_symbols = frozenset(
            self._apply_symbol(transform_index, symbol)
            for transform_index, _transform in enumerate(self.transforms)
            for symbol in self.base_model.forbidden_symbols
        )
        self.alphabet = frozenset(
            self._apply_symbol(transform_index, symbol)
            for transform_index, _transform in enumerate(self.transforms)
            for symbol in self.base_model.alphabet
        )

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[Sequence[Symbol]],
        *,
        max_order: int,
        transforms: Iterable[SymbolTransform],
        start_symbol: Symbol | None = None,
        end_symbol: Symbol | None = None,
    ) -> "VirtualAugmentedOrderStackModel":
        base_model = OrderStackModel.from_sequences(
            sequences,
            max_order=max_order,
            start_symbol=start_symbol,
            end_symbol=end_symbol,
        )
        return cls(base_model=base_model, transforms=tuple(transforms))

    @property
    def base_context_count(self) -> int:
        return len(self.base_model.counts)

    @property
    def virtual_event_multiplier(self) -> int:
        return len(self.transforms)

    def virtual_context_count(self, order: int | None = None) -> int:
        if order is None:
            order = self.max_order
        return sum(1 for _context in self.iter_contexts(order))

    def iter_contexts(self, order: int) -> Iterable[Context]:
        self._context_materialization_calls += 1
        cached = self._contexts_cache.get(order)
        if cached is not None:
            self._context_materialization_cache_hits += 1
            return cached
        self._context_materialization_cache_misses += 1
        started = time.perf_counter()
        contexts: set[Context] = set()
        for context in self.base_model.counts:
            if len(context) > order:
                continue
            for transform_index, _transform in enumerate(self.transforms):
                contexts.add(
                    tuple(
                        self._apply_symbol(transform_index, symbol)
                        for symbol in context
                    )
                )
        result = tuple(contexts)
        self._contexts_cache[order] = result
        self._context_materialization_seconds += time.perf_counter() - started
        return result

    def augmented_counts(self, context: Iterable[Symbol] | Context) -> Counter[Symbol]:
        self._augmented_count_calls += 1
        state = _as_context(context)
        cached = self._counts_cache.get(state)
        if cached is not None:
            self._augmented_count_cache_hits += 1
            return cached

        self._augmented_count_cache_misses += 1
        counts: Counter[Symbol] = Counter()
        for transform_index, _transform in enumerate(self.transforms):
            inverse_context = self._inverse_context(transform_index, state)
            base_counts = self.base_model.counts.get(inverse_context)
            if not base_counts:
                continue
            for base_symbol, count in base_counts.items():
                counts[self._apply_symbol(transform_index, base_symbol)] += (
                    count * self.transforms[transform_index].weight
                )
        self._counts_cache[state] = counts
        return counts

    def virtual_diagnostics(self) -> dict[str, int | float]:
        return {
            "virtual_context_materialization_seconds": self._context_materialization_seconds,
            "virtual_context_materialization_calls": self._context_materialization_calls,
            "virtual_context_materialization_cache_hits": (
                self._context_materialization_cache_hits
            ),
            "virtual_context_materialization_cache_misses": (
                self._context_materialization_cache_misses
            ),
            "augmented_count_calls": self._augmented_count_calls,
            "augmented_count_cache_hits": self._augmented_count_cache_hits,
            "augmented_count_cache_misses": self._augmented_count_cache_misses,
        }

    def _apply_symbol(self, transform_index: int, symbol: Symbol) -> Symbol:
        key = (transform_index, symbol)
        cached = self._apply_symbol_cache.get(key)
        if cached is not None:
            return cached
        transformed = self.transforms[transform_index].apply_symbol(symbol)
        self._apply_symbol_cache[key] = transformed
        return transformed

    def _inverse_symbol(self, transform_index: int, symbol: Symbol) -> Symbol:
        key = (transform_index, symbol)
        cached = self._inverse_symbol_cache.get(key)
        if cached is not None:
            return cached
        transformed = self.transforms[transform_index].inverse_symbol(symbol)
        self._inverse_symbol_cache[key] = transformed
        return transformed

    def _inverse_context(self, transform_index: int, context: Sequence[Symbol]) -> Context:
        state = _as_context(context)
        key = (transform_index, state)
        cached = self._inverse_context_cache.get(key)
        if cached is not None:
            return cached
        transformed = tuple(self._inverse_symbol(transform_index, symbol) for symbol in state)
        self._inverse_context_cache[key] = transformed
        return transformed

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
            if self.augmented_counts(suffix):
                return suffix
        return None

    def continuation_distribution_with_order(
        self,
        context: Iterable[Symbol] | Context,
        *,
        max_order: int | None = None,
    ) -> tuple[tuple[tuple[Symbol, float], ...], int | None]:
        state = _as_context(context)
        cache_key = (state, max_order)
        cached = self._distribution_cache.get(cache_key)
        if cached is not None:
            return cached

        suffix = self.longest_available_suffix(state, max_order=max_order)
        if suffix is None:
            result = ((), None)
            self._distribution_cache[cache_key] = result
            return result

        counts = self.augmented_counts(suffix)
        total = float(sum(counts.values()))
        if total <= 0.0:
            result = ((), None)
        else:
            result = (
                tuple((symbol, float(count) / total) for symbol, count in counts.items()),
                len(suffix),
            )
        self._distribution_cache[cache_key] = result
        return result

    def compile_graph(self, order: int) -> FixedOrderContextGraph:
        cached = self._graph_cache.get(order)
        if cached is not None:
            return cached
        graph = FixedOrderContextGraph.from_model(self, order=order)
        self._graph_cache[order] = graph
        return graph

    def compile_graphs_for_plan(
        self,
        *,
        length: int,
    ) -> dict[int, "LazyVirtualFixedOrderContextGraph"]:
        if length < 0:
            raise ValueError("length must be non-negative")
        cache_key = int(length)
        cached = self._lazy_plan_graph_cache.get(cache_key)
        if cached is not None:
            return cached

        graphs = self._new_lazy_graphs()
        self._lazy_plan_graph_cache[cache_key] = graphs
        return graphs

    def compile_graphs_for_prefix(
        self,
        *,
        prefix: Sequence[Symbol],
        length: int,
    ) -> dict[int, "LazyVirtualFixedOrderContextGraph"]:
        if length < 0:
            raise ValueError("length must be non-negative")
        prefix_context = tuple(prefix)
        cache_key = (prefix_context, int(length))
        cached = self._lazy_graph_cache.get(cache_key)
        if cached is not None:
            return cached

        graphs = self._new_lazy_graphs()
        self._lazy_graph_cache[cache_key] = graphs
        return graphs

    def _new_lazy_graphs(self) -> dict[int, "LazyVirtualFixedOrderContextGraph"]:
        return {
            order: LazyVirtualFixedOrderContextGraph(
                self,
                order=order,
            )
            for order in range(1, self.max_order + 1)
        }


class _LazyOutgoing:
    def __init__(self, graph: "LazyVirtualFixedOrderContextGraph") -> None:
        self.graph = graph

    def __getitem__(self, state: int) -> list[StackEdge]:
        return self.graph.outgoing_for_state(state)

    def __iter__(self) -> Iterator[list[StackEdge]]:
        for state in range(len(self.graph.contexts)):
            yield self.graph.outgoing_for_state(state)

    def __len__(self) -> int:
        return len(self.graph.contexts)


class LazyVirtualFixedOrderContextGraph:
    """Lazily materialized fixed-order graph for virtual augmentation."""

    def __init__(
        self,
        model: VirtualAugmentedOrderStackModel,
        *,
        order: int,
        allowed_contexts: Iterable[Context] | None = None,
    ) -> None:
        if order < 1 or order > model.max_order:
            raise ValueError(f"order must be between 1 and {model.max_order}")
        self.model = model
        self.order = int(order)
        self._allowed_contexts_source = (
            None if allowed_contexts is None else tuple(allowed_contexts)
        )
        self._allowed_contexts: set[Context] | None = None
        self.contexts: list[Context] = []
        self.context_to_id: dict[Context, int] = {}
        self._outgoing_cache: dict[int, list[StackEdge]] = {}
        self.outgoing = _LazyOutgoing(self)
        self.outgoing_row_calls = 0
        self.outgoing_row_cache_hits = 0
        self.outgoing_row_cache_misses = 0
        self.outgoing_row_materialization_seconds = 0.0

    @property
    def allowed_contexts(self) -> set[Context]:
        return self._ensure_allowed_contexts()

    def truncate_context(self, context: Iterable[Symbol] | Context) -> Context:
        state = _as_context(context)
        if len(state) <= self.order:
            return state
        return state[-self.order:]

    def next_context(self, context: Iterable[Symbol] | Context, symbol: Symbol) -> Context:
        return self.truncate_context(_as_context(context) + (symbol,))

    def state_id(self, context: Iterable[Symbol] | Context) -> int | None:
        state = self.truncate_context(context)
        if state not in self._ensure_allowed_contexts():
            return None
        return self._add_context(state)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.outgoing)

    def outgoing_for_state(self, state: int) -> list[StackEdge]:
        self.outgoing_row_calls += 1
        cached = self._outgoing_cache.get(state)
        if cached is not None:
            self.outgoing_row_cache_hits += 1
            return cached
        self.outgoing_row_cache_misses += 1
        started = time.perf_counter()
        context = self.contexts[state]
        distribution, effective_order = self.model.continuation_distribution_with_order(
            context,
            max_order=self.order,
        )
        edges: list[StackEdge] = []
        order = self.order
        allowed_contexts = self._ensure_allowed_contexts()
        for symbol, probability in distribution:
            candidate_context = context + (symbol,)
            dst_context = (
                candidate_context
                if len(candidate_context) <= order
                else candidate_context[-order:]
            )
            if dst_context not in allowed_contexts:
                continue
            dst = self._add_context(dst_context)
            edges.append(
                StackEdge(
                    src=state,
                    dst=dst,
                    symbol=symbol,
                    probability=float(probability),
                    order=effective_order or 0,
                )
            )
        self._outgoing_cache[state] = edges
        self.outgoing_row_materialization_seconds += time.perf_counter() - started
        return edges

    def _ensure_allowed_contexts(self) -> set[Context]:
        allowed_contexts = self._allowed_contexts
        if allowed_contexts is not None:
            return allowed_contexts
        source = (
            self.model.iter_contexts(self.order)
            if self._allowed_contexts_source is None
            else self._allowed_contexts_source
        )
        allowed_contexts = {self.truncate_context(context) for context in source}
        self._allowed_contexts = allowed_contexts
        return allowed_contexts

    def _add_context(self, context: Context) -> int:
        found = self.context_to_id.get(context)
        if found is not None:
            return found
        context_id = len(self.contexts)
        self.context_to_id[context] = context_id
        self.contexts.append(context)
        return context_id
