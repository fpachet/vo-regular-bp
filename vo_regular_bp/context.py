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
    order_weights: tuple[tuple[int, float], ...] = ()


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
                Edge(
                    edge.symbol,
                    float(edge.probability),
                    _as_context(edge.next_state),
                    tuple((int(order), float(weight)) for order, weight in edge.order_weights),
                )
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
        self._alias_to_state: dict[Context, Context] = {}
        self._continuation_counts: dict[Context, dict[Symbol, float]] = {}
        self._state_supports: dict[Context, float] = {}
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
        count_metadata: dict[Context, dict[Symbol, float]] = {}
        state_supports: dict[Context, float] = {}
        for raw_context, counts in continuation_counts.items():
            context = _as_context(raw_context)
            total = float(sum(counts.values()))
            if total <= 0.0:
                raise ValueError(f"context {context!r} has no positive continuation mass")
            state_supports[context] = total
            count_metadata[context] = {
                symbol: float(count)
                for symbol, count in counts.items()
                if count > 0
            }
            edges_by_state[context] = [
                Edge(symbol, float(count) / total, cls._canon(context, symbol, contexts, max_order))
                for symbol, count in counts.items()
                if count > 0
            ]
        graph = cls(
            edges_by_state,
            start_state=start_state,
            max_order=max_order,
            validate=True,
        )
        graph._continuation_counts = count_metadata
        graph._state_supports = state_supports
        return graph

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
        continuation_counts: dict[Context, dict[Symbol, float]] = {}
        state_supports: dict[Context, float] = {}
        for raw_context, probs in probabilities.items():
            context = _as_context(raw_context)
            continuation_counts[context] = {
                symbol: float(prob)
                for symbol, prob in probs.items()
                if prob > 0.0
            }
            state_supports[context] = float(sum(continuation_counts[context].values()))
            edges_by_state[context] = [
                Edge(symbol, float(prob), cls._canon(context, symbol, contexts, max_order))
                for symbol, prob in probs.items()
                if prob > 0.0
            ]
        graph = cls(
            edges_by_state,
            start_state=start_state,
            max_order=max_order,
            validate=True,
        )
        graph._continuation_counts = continuation_counts
        graph._state_supports = state_supports
        return graph

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

        return cls.from_weighted_sequences(
            ((1.0, sequence) for sequence in sequences),
            max_order=max_order,
            start_state=start_state,
        )

    @classmethod
    def from_weighted_sequences(
        cls,
        weighted_sequences: Iterable[tuple[int | float, Sequence[Symbol]]],
        *,
        max_order: int,
        start_state: Iterable[Symbol] | Context = (),
    ) -> "ContextGraph":
        """Estimate continuation counts from a weighted sequence multiset."""

        if max_order < 0:
            raise ValueError("max_order must be non-negative")

        counts: dict[Context, Counter[Symbol]] = defaultdict(Counter)

        for weight, sequence in weighted_sequences:
            if weight <= 0:
                continue
            tokens = tuple(sequence)
            for index, symbol in enumerate(tokens):
                order_limit = min(max_order, index)
                for order in range(order_limit + 1):
                    context = tokens[index - order : index] if order else ()
                    counts[context][symbol] += weight

        return cls.from_counts(counts, max_order=max_order, start_state=start_state)

    @classmethod
    def from_backoff_sequences(
        cls,
        sequences: Iterable[Sequence[Symbol]],
        *,
        max_order: int,
        backoff_weight: float = 0.25,
        start_state: Iterable[Symbol] | Context = (),
    ) -> "ContextGraph":
        """Build a sparse graph with explicit lower-order backoff support.

        Each context receives a weighted mixture of normalized continuation
        distributions from itself and its suffixes. This is useful for exact
        experiments where a strict longest-context MLE model would make every
        ``K+1``-gram transition copied from training by construction.
        """

        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        if not 0.0 <= backoff_weight <= 1.0:
            raise ValueError("backoff_weight must be in [0, 1]")

        counts: dict[Context, Counter[Symbol]] = defaultdict(Counter)
        for sequence in sequences:
            tokens = tuple(sequence)
            for index, symbol in enumerate(tokens):
                order_limit = min(max_order, index)
                for order in range(order_limit + 1):
                    context = tokens[index - order : index] if order else ()
                    counts[context][symbol] += 1

        contexts = set(counts)
        contexts.add(_as_context(start_state))
        edges_by_state: dict[Context, list[Edge]] = {}
        continuation_counts: dict[Context, dict[Symbol, float]] = {}
        state_supports: dict[Context, float] = {}

        for context in contexts:
            scores: Counter[Symbol] = Counter()
            order_scores: dict[Symbol, Counter[int]] = defaultdict(Counter)
            context_order = len(context)
            for order in range(context_order, -1, -1):
                suffix = context[-order:] if order else ()
                if suffix not in counts:
                    continue
                total = float(sum(counts[suffix].values()))
                if total <= 0.0:
                    continue
                weight = backoff_weight ** (context_order - order)
                for symbol, count in counts[suffix].items():
                    contribution = weight * (float(count) / total)
                    scores[symbol] += contribution
                    order_scores[symbol][order] += contribution

            total_score = float(sum(scores.values()))
            if total_score <= 0.0:
                continue
            state_supports[context] = total_score
            continuation_counts[context] = {
                symbol: float(score)
                for symbol, score in scores.items()
                if score > 0.0
            }
            edges_by_state[context] = [
                Edge(
                    symbol,
                    score / total_score,
                    cls._canon(context, symbol, contexts, max_order),
                    tuple(
                        sorted(
                            (
                                (order, contribution / score)
                                for order, contribution in order_scores[symbol].items()
                            ),
                            reverse=True,
                        )
                    ),
                )
                for symbol, score in scores.items()
                if score > 0.0
            ]

        graph = cls(
            edges_by_state,
            start_state=start_state,
            max_order=max_order,
            validate=True,
        )
        graph._continuation_counts = continuation_counts
        graph._state_supports = state_supports
        return graph

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

        candidate = _as_context(context) + (symbol,)
        limit = min(self.max_order, len(candidate))
        for order in range(limit, -1, -1):
            suffix = candidate[-order:] if order else ()
            state = self._resolve_state(suffix)
            if state in self.states:
                return state
        return ()

    def outgoing(self, state: Iterable[Symbol] | Context) -> tuple[Edge, ...]:
        """Outgoing edges for a context, or an empty tuple for dead contexts."""

        return self._edges.get(self._resolve_state(state), ())

    def _resolve_state(self, state: Iterable[Symbol] | Context) -> Context:
        context = _as_context(state)
        return self._alias_to_state.get(context, context)

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
                order_total = 0.0
                for order, weight in edge.order_weights:
                    if order < 0:
                        raise ValueError(f"invalid negative order {order!r} on edge {edge!r}")
                    if not math.isfinite(weight) or weight < 0.0:
                        raise ValueError(f"invalid order weight {weight!r} on edge {edge!r}")
                    order_total += weight
                if edge.order_weights and not math.isclose(order_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"order weights on edge {edge!r} sum to {order_total}, not 1")
                total += edge.probability
            if edges and not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"outgoing probabilities from {state!r} sum to {total}, not 1")
