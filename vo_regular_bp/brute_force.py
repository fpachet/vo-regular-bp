"""Tiny exhaustive checks for exactness tests."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Hashable, Iterable, Mapping

from .acceptors import DFA
from .context import Context, ContextGraph, Symbol, _as_context


def brute_force_distribution(
    graph: ContextGraph,
    acceptor: DFA,
    *,
    length: int,
    alphabet: Iterable[Symbol] | None = None,
    start_context: Iterable[Symbol] | Context | None = None,
    start_acceptor_state: Hashable | None = None,
) -> dict[tuple[Symbol, ...], float]:
    """Enumerate all accepted length-``n`` strings and unnormalized masses."""

    if length < 0:
        raise ValueError("length must be non-negative")

    context0 = graph.start_state if start_context is None else _as_context(start_context)
    acceptor0 = acceptor.start_state if start_acceptor_state is None else start_acceptor_state
    masses: dict[tuple[Symbol, ...], float] = defaultdict(float)

    if alphabet is not None:
        for sequence in product(tuple(alphabet), repeat=length):
            state = acceptor0
            accepted = True
            for symbol in sequence:
                state = acceptor.next_state(state, symbol)
                if state is None:
                    accepted = False
                    break
            if not accepted or not acceptor.is_accepting(state):
                continue
            probability = graph.probability(sequence, start_state=context0)
            if probability > 0.0:
                masses[sequence] += probability
        return dict(masses)

    def visit(
        time: int,
        context_state: Context,
        acceptor_state: Hashable,
        prefix: tuple[Symbol, ...],
        probability: float,
    ) -> None:
        if time == length:
            if acceptor.is_accepting(acceptor_state):
                masses[prefix] += probability
            return

        for edge in graph.outgoing(context_state):
            next_acceptor_state = acceptor.next_state(acceptor_state, edge.symbol)
            if next_acceptor_state is None:
                continue
            visit(
                time + 1,
                edge.next_state,
                next_acceptor_state,
                prefix + (edge.symbol,),
                probability * edge.probability,
            )

    visit(0, context0, acceptor0, (), 1.0)
    return dict(masses)


def brute_force_partition_function(
    graph: ContextGraph,
    acceptor: DFA,
    *,
    length: int,
    alphabet: Iterable[Symbol] | None = None,
    start_context: Iterable[Symbol] | Context | None = None,
    start_acceptor_state: Hashable | None = None,
) -> float:
    return sum(
        brute_force_distribution(
            graph,
            acceptor,
            length=length,
            alphabet=alphabet,
            start_context=start_context,
            start_acceptor_state=start_acceptor_state,
        ).values()
    )


def conditional_distribution(masses: Mapping[tuple[Symbol, ...], float]) -> dict[tuple[Symbol, ...], float]:
    total = float(sum(masses.values()))
    if total <= 0.0:
        return {}
    return {sequence: mass / total for sequence, mass in masses.items()}
