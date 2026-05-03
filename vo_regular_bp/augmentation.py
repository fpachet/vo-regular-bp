"""Virtual data augmentation for order-stack models.

The classes here keep the generated symbol space absolute.  Transformations are
used only to evaluate continuation counts as if transformed copies of the
training sequences had been materialized.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

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

    def __post_init__(self) -> None:
        if not self.transforms:
            raise ValueError("at least one transform is required")
        if any(transform.weight <= 0.0 for transform in self.transforms):
            raise ValueError("transform weights must be positive")
        self.max_order = self.base_model.max_order
        self.forbidden_symbols = frozenset(
            transform.apply_symbol(symbol)
            for transform in self.transforms
            for symbol in self.base_model.forbidden_symbols
        )
        self.alphabet = frozenset(
            transform.apply_symbol(symbol)
            for transform in self.transforms
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
        cached = self._contexts_cache.get(order)
        if cached is not None:
            return cached
        contexts: set[Context] = set()
        for context in self.base_model.counts:
            if len(context) > order:
                continue
            for transform in self.transforms:
                contexts.add(transform.apply_context(context))
        result = tuple(contexts)
        self._contexts_cache[order] = result
        return result

    def augmented_counts(self, context: Iterable[Symbol] | Context) -> Counter[Symbol]:
        state = _as_context(context)
        cached = self._counts_cache.get(state)
        if cached is not None:
            return cached

        counts: Counter[Symbol] = Counter()
        for transform in self.transforms:
            inverse_context = transform.inverse_context(state)
            base_counts = self.base_model.counts.get(inverse_context)
            if not base_counts:
                continue
            for base_symbol, count in base_counts.items():
                counts[transform.apply_symbol(base_symbol)] += count * transform.weight
        self._counts_cache[state] = counts
        return counts

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

        graphs = {
            order: LazyVirtualFixedOrderContextGraph(
                self,
                order=order,
                allowed_contexts=self.iter_contexts(order),
            )
            for order in range(1, self.max_order + 1)
        }
        self._lazy_graph_cache[cache_key] = graphs
        return graphs


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
        allowed_contexts: Iterable[Context],
    ) -> None:
        if order < 1 or order > model.max_order:
            raise ValueError(f"order must be between 1 and {model.max_order}")
        self.model = model
        self.order = int(order)
        self.allowed_contexts = {self.truncate_context(context) for context in allowed_contexts}
        self.contexts: list[Context] = []
        self.context_to_id: dict[Context, int] = {}
        self._outgoing_cache: dict[int, list[StackEdge]] = {}
        self.outgoing = _LazyOutgoing(self)

    def truncate_context(self, context: Iterable[Symbol] | Context) -> Context:
        state = _as_context(context)
        if len(state) <= self.order:
            return state
        return state[-self.order:]

    def next_context(self, context: Iterable[Symbol] | Context, symbol: Symbol) -> Context:
        return self.truncate_context(_as_context(context) + (symbol,))

    def state_id(self, context: Iterable[Symbol] | Context) -> int | None:
        state = self.truncate_context(context)
        if state not in self.allowed_contexts:
            return None
        return self._add_context(state)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.outgoing)

    def outgoing_for_state(self, state: int) -> list[StackEdge]:
        cached = self._outgoing_cache.get(state)
        if cached is not None:
            return cached
        context = self.contexts[state]
        distribution, effective_order = self.model.continuation_distribution_with_order(
            context,
            max_order=self.order,
        )
        edges: list[StackEdge] = []
        for symbol, probability in distribution:
            dst_context = self.next_context(context, symbol)
            if dst_context not in self.allowed_contexts:
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
        return edges

    def _add_context(self, context: Context) -> int:
        found = self.context_to_id.get(context)
        if found is not None:
            return found
        context_id = len(self.contexts)
        self.context_to_id[context] = context_id
        self.contexts.append(context)
        return context_id
