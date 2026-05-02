"""Sparse variable-order context graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Hashable, Iterable, Mapping, Sequence

Symbol = Hashable
Context = tuple[Symbol, ...]


@dataclass(frozen=True)
class Edge:
    """A labeled probabilistic transition in a context graph."""

    symbol: Symbol
    probability: float
    next_state: Context


def _as_context(value: Iterable[Symbol] | Context | None) -> Context:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    return tuple(value)


class ContextGraph:
    """Sparse context graph induced by a variable-order/backoff model.

    Nodes are canonical contexts represented as tuples of symbols. Each outgoing
    edge emits one symbol and moves to the canonical suffix context after that
    emission.
    """

    def __init__(
        self,
        edges_by_state: Mapping[Iterable[Symbol] | Context, Sequence[Edge]],
        *,
        start_state: Iterable[Symbol] | Context = (),
        max_order: int | None = None,
        alphabet: Iterable[Symbol] | None = None,
        validate: bool = True,
    ) -> None:
        normalized: dict[Context, tuple[Edge, ...]] = {}
        states: set[Context] = {_as_context(start_state)}
        emitted: set[Symbol] = set(alphabet or ())

        for raw_state, raw_edges in edges_by_state.items():
            state = _as_context(raw_state)
            edges = tuple(
                Edge(edge.symbol, float(edge.probability), _as_context(edge.next_state))
                for edge in raw_edges
            )
            normalized[state] = edges
            states.add(state)
            for edge in edges:
                states.add(edge.next_state)
                emitted.add(edge.symbol)

        self._edges = normalized
        self.start_state = _as_context(start_state)
        self.states = frozenset(states)
        self.alphabet = frozenset(emitted)
        self.max_order = (
            max_order
            if max_order is not None
            else max((len(state) for state in self.states), default=0)
        )

        if self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if validate:
            self._validate()

    @classmethod
    def from_counts(
        cls,
        continuation_counts: Mapping[Iterable[Symbol] | Context, Mapping[Symbol, int | float]],
        *,
        max_order: int | None = None,
        start_state: Iterable[Symbol] | Context = (),
    ) -> "ContextGraph":
        """Build a graph by normalizing explicit continuation counts."""

        contexts = {_as_context(context) for context in continuation_counts}
        contexts.add(_as_context(start_state))
        if max_order is None:
            max_order = max((len(context) for context in contexts), default=0)

        edges_by_state: dict[Context, list[Edge]] = {}
        for raw_context, counts in continuation_counts.items():
            context = _as_context(raw_context)
            total = float(sum(counts.values()))
            if total <= 0.0:
                raise ValueError(f"context {context!r} has no positive continuation mass")
            edges_by_state[context] = [
                Edge(symbol, float(count) / total, cls._canon(context, symbol, contexts, max_order))
                for symbol, count in counts.items()
                if count > 0
            ]
        return cls(
            edges_by_state,
            start_state=start_state,
            max_order=max_order,
            validate=True,
        )

    @classmethod
    def from_probabilities(
        cls,
        probabilities: Mapping[Iterable[Symbol] | Context, Mapping[Symbol, float]],
        *,
        max_order: int | None = None,
        start_state: Iterable[Symbol] | Context = (),
    ) -> "ContextGraph":
        """Build a graph from explicit conditional probabilities."""

        contexts = {_as_context(context) for context in probabilities}
        contexts.add(_as_context(start_state))
        if max_order is None:
            max_order = max((len(context) for context in contexts), default=0)

        edges_by_state: dict[Context, list[Edge]] = {}
        for raw_context, probs in probabilities.items():
            context = _as_context(raw_context)
            edges_by_state[context] = [
                Edge(symbol, float(prob), cls._canon(context, symbol, contexts, max_order))
                for symbol, prob in probs.items()
                if prob > 0.0
            ]
        return cls(
            edges_by_state,
            start_state=start_state,
            max_order=max_order,
            validate=True,
        )

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[Sequence[Symbol]],
        *,
        max_order: int,
        start_state: Iterable[Symbol] | Context = (),
    ) -> "ContextGraph":
        """Estimate normalized continuation counts for all suffix contexts.

        For every token position, the implementation records the continuation
        from each suffix context of orders 0..max_order available before that
        token. This gives a sparse variable-order graph whose transitions are
        normalized continuation counts at each observed context.
        """

        if max_order < 0:
            raise ValueError("max_order must be non-negative")

        counts: dict[Context, Counter[Symbol]] = defaultdict(Counter)
        observed_contexts: set[Context] = {(), _as_context(start_state)}

        for sequence in sequences:
            tokens = tuple(sequence)
            for index, symbol in enumerate(tokens):
                order_limit = min(max_order, index)
                for order in range(order_limit + 1):
                    context = tokens[index - order : index] if order else ()
                    counts[context][symbol] += 1
                    observed_contexts.add(context)

        return cls.from_counts(counts, max_order=max_order, start_state=start_state)

    @staticmethod
    def _canon(
        context: Context,
        symbol: Symbol,
        known_contexts: set[Context] | frozenset[Context],
        max_order: int,
    ) -> Context:
        candidate = context + (symbol,)
        limit = min(max_order, len(candidate))
        for order in range(limit, -1, -1):
            suffix = candidate[-order:] if order else ()
            if suffix in known_contexts:
                return suffix
        return ()

    def canonical_context(self, context: Iterable[Symbol] | Context, symbol: Symbol) -> Context:
        """Return the longest known suffix after emitting ``symbol``."""

        return self._canon(_as_context(context), symbol, self.states, self.max_order)

    def outgoing(self, state: Iterable[Symbol] | Context) -> tuple[Edge, ...]:
        """Outgoing edges for a context, or an empty tuple for dead contexts."""

        return self._edges.get(_as_context(state), ())

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._edges.values())

    def probability(self, sequence: Sequence[Symbol], *, start_state: Iterable[Symbol] | Context | None = None) -> float:
        """Unconstrained probability of a sequence under the context graph."""

        state = self.start_state if start_state is None else _as_context(start_state)
        prob = 1.0
        for symbol in sequence:
            edge = next((edge for edge in self.outgoing(state) if edge.symbol == symbol), None)
            if edge is None:
                return 0.0
            prob *= edge.probability
            state = edge.next_state
        return prob

    def _validate(self) -> None:
        if self.start_state not in self.states:
            raise ValueError("start_state must be present in graph states")
        if self.max_order < max((len(state) for state in self.states), default=0):
            raise ValueError("max_order is smaller than at least one context state")

        for state, edges in self._edges.items():
            seen_symbols: set[Symbol] = set()
            total = 0.0
            for edge in edges:
                if edge.symbol in seen_symbols:
                    raise ValueError(f"state {state!r} has duplicate edge for symbol {edge.symbol!r}")
                seen_symbols.add(edge.symbol)
                if edge.next_state not in self.states:
                    raise ValueError(f"edge from {state!r} points to unknown state {edge.next_state!r}")
                if not math.isfinite(edge.probability) or edge.probability < 0.0:
                    raise ValueError(f"invalid probability {edge.probability!r} on edge {edge!r}")
                total += edge.probability
            if edges and not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"outgoing probabilities from {state!r} sum to {total}, not 1")
