"""Deterministic regular acceptors used by product BP."""

from __future__ import annotations

from itertools import product
from typing import Callable, Hashable, Iterable, Mapping, Sequence

Symbol = Hashable
State = Hashable


class DFA:
    """Small deterministic acceptor interface.

    A transition returning ``None`` means that the emitted symbol is rejected
    from that state. The product BP code only relies on ``start_state``,
    ``next_state`` and ``is_accepting``.
    """

    def __init__(
        self,
        *,
        start_state: State,
        accept_states: Iterable[State] | None = None,
        transitions: Mapping[State, Mapping[Symbol, State]] | None = None,
        states: Iterable[State] | None = None,
        alphabet: Iterable[Symbol] | None = None,
        transition_func: Callable[[State, Symbol], State | None] | None = None,
        accept_func: Callable[[State], bool] | None = None,
        name: str = "dfa",
    ) -> None:
        if accept_states is None and accept_func is None:
            raise ValueError("either accept_states or accept_func is required")
        if transitions is None and transition_func is None:
            raise ValueError("either transitions or transition_func is required")

        self.start_state = start_state
        self.accept_states = frozenset(accept_states or ())
        self.transitions = {
            state: dict(symbols)
            for state, symbols in (transitions or {}).items()
        }
        self.states = None if states is None else frozenset(states)
        self.alphabet = None if alphabet is None else frozenset(alphabet)
        self._transition_func = transition_func
        self._accept_func = accept_func
        self.name = name

    def next_state(self, state: State, symbol: Symbol) -> State | None:
        if self._transition_func is not None:
            return self._transition_func(state, symbol)
        return self.transitions.get(state, {}).get(symbol)

    def is_accepting(self, state: State) -> bool:
        if self._accept_func is not None:
            return bool(self._accept_func(state))
        return state in self.accept_states

    def accepts(self, sequence: Sequence[Symbol]) -> bool:
        state = self.start_state
        for symbol in sequence:
            state = self.next_state(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)

    def state_count(self) -> int | None:
        if self.states is None:
            return None
        return len(self.states)


def true_acceptor(*, name: str = "true") -> DFA:
    """Accept every finite sequence over any alphabet."""

    return DFA(
        start_state=0,
        accept_states={0},
        states={0},
        transition_func=lambda state, symbol: 0,
        name=name,
    )


def all_of(*acceptors: DFA, name: str = "all_of") -> DFA:
    """Intersection of deterministic acceptors."""

    if not acceptors:
        return true_acceptor(name=name)

    start = tuple(acceptor.start_state for acceptor in acceptors)

    def transition(state: State, symbol: Symbol) -> State | None:
        next_parts = []
        for acceptor, part in zip(acceptors, state):
            next_part = acceptor.next_state(part, symbol)
            if next_part is None:
                return None
            next_parts.append(next_part)
        return tuple(next_parts)

    def accepting(state: State) -> bool:
        return all(acceptor.is_accepting(part) for acceptor, part in zip(acceptors, state))

    states = None
    if all(acceptor.states is not None for acceptor in acceptors):
        sizes = [len(acceptor.states or ()) for acceptor in acceptors]
        product_size = 1
        for size in sizes:
            product_size *= size
        if product_size <= 100_000:
            state_sets = [acceptor.states or () for acceptor in acceptors]
            states = set(product(*state_sets))

    alphabets = [acceptor.alphabet for acceptor in acceptors if acceptor.alphabet is not None]
    alphabet = set.intersection(*map(set, alphabets)) if alphabets else None

    return DFA(
        start_state=start,
        states=states,
        alphabet=alphabet,
        transition_func=transition,
        accept_func=accepting,
        name=name,
    )


def positional_acceptor(
    length: int,
    constraints: Mapping[int, Iterable[Symbol] | Symbol] | None = None,
    *,
    alphabet: Iterable[Symbol] | None = None,
    name: str = "positional",
) -> DFA:
    """Accept length-``n`` strings satisfying per-position allowed sets.

    Positions are zero-based. Constraint values may be a single symbol or an
    iterable of allowed symbols.
    """

    if length < 0:
        raise ValueError("length must be non-negative")

    allowed_by_pos: dict[int, set[Symbol]] = {}
    for position, allowed in (constraints or {}).items():
        if position < 0 or position >= length:
            raise ValueError(f"position {position} is outside length {length}")
        if isinstance(allowed, (str, bytes)):
            allowed_by_pos[position] = {allowed}
        else:
            try:
                allowed_by_pos[position] = set(allowed)  # type: ignore[arg-type]
            except TypeError:
                allowed_by_pos[position] = {allowed}  # type: ignore[list-item]

    def transition(position: State, symbol: Symbol) -> State | None:
        if not isinstance(position, int) or position >= length:
            return None
        allowed = allowed_by_pos.get(position)
        if allowed is not None and symbol not in allowed:
            return None
        return position + 1

    return DFA(
        start_state=0,
        accept_states={length},
        states=set(range(length + 1)),
        alphabet=alphabet,
        transition_func=transition,
        name=name,
    )


def meter_acceptor(
    pattern: Sequence[Hashable | Iterable[Hashable] | None],
    symbol_to_meter: Mapping[Symbol, Hashable] | Callable[[Symbol], Hashable],
    *,
    alphabet: Iterable[Symbol] | None = None,
    name: str = "meter",
) -> DFA:
    """Accept strings whose per-symbol meter classes match ``pattern``.

    Each pattern entry may be a single class, an iterable of allowed classes, or
    ``None`` as a wildcard. This intentionally keeps meter generic: callers can
    map symbols to stress, duration, pitch class, or any other finite class.
    """

    allowed_by_pos: list[set[Hashable] | None] = []
    for expected in pattern:
        if expected is None:
            allowed_by_pos.append(None)
        elif isinstance(expected, (str, bytes)):
            allowed_by_pos.append({expected})
        else:
            try:
                allowed_by_pos.append(set(expected))  # type: ignore[arg-type]
            except TypeError:
                allowed_by_pos.append({expected})  # type: ignore[list-item]

    if callable(symbol_to_meter):
        meter_of = symbol_to_meter
    else:
        meter_map = dict(symbol_to_meter)
        meter_of = lambda symbol: meter_map[symbol]

    length = len(pattern)

    def transition(position: State, symbol: Symbol) -> State | None:
        if not isinstance(position, int) or position >= length:
            return None
        allowed = allowed_by_pos[position]
        if allowed is not None and meter_of(symbol) not in allowed:
            return None
        return position + 1

    return DFA(
        start_state=0,
        accept_states={length},
        states=set(range(length + 1)),
        alphabet=alphabet,
        transition_func=transition,
        name=name,
    )


def forbidden_substring_acceptor(
    forbidden_patterns: Iterable[Sequence[Symbol]],
    *,
    alphabet: Iterable[Symbol] | None = None,
    name: str = "forbidden_substring",
) -> DFA:
    """Accept strings that contain none of the forbidden substrings.

    The states are Aho-Corasick-style prefixes: the current state stores the
    longest suffix of the generated sequence that is a proper prefix of a
    forbidden pattern. Completing a forbidden pattern rejects the transition.
    """

    patterns = tuple(tuple(pattern) for pattern in forbidden_patterns)
    if any(len(pattern) == 0 for pattern in patterns):
        raise ValueError("empty forbidden patterns would reject every sequence")
    if not patterns:
        return true_acceptor(name=name)

    prefixes: set[tuple[Symbol, ...]] = {()}
    for pattern in patterns:
        for prefix_len in range(1, len(pattern)):
            prefixes.add(pattern[:prefix_len])

    max_prefix_len = max((len(prefix) for prefix in prefixes), default=0)

    def completes_forbidden(candidate: tuple[Symbol, ...]) -> bool:
        return any(
            len(candidate) >= len(pattern) and candidate[-len(pattern) :] == pattern
            for pattern in patterns
        )

    def next_prefix(state: State, symbol: Symbol) -> State | None:
        if not isinstance(state, tuple):
            raise TypeError("forbidden-substring states must be tuples")
        candidate = state + (symbol,)
        if completes_forbidden(candidate):
            return None
        limit = min(len(candidate), max_prefix_len)
        for size in range(limit, -1, -1):
            suffix = candidate[-size:] if size else ()
            if suffix in prefixes:
                return suffix
        return ()

    transitions = None
    if alphabet is not None:
        alphabet_set = frozenset(alphabet)
        transitions = {
            state: {
                symbol: next_state
                for symbol in alphabet_set
                if (next_state := next_prefix(state, symbol)) is not None
            }
            for state in prefixes
        }
    else:
        alphabet_set = None

    return DFA(
        start_state=(),
        accept_states=prefixes,
        transitions=transitions,
        states=prefixes,
        alphabet=alphabet_set,
        transition_func=None if transitions is not None else next_prefix,
        name=name,
    )


def max_order_acceptor(
    reference_sequences: Iterable[Sequence[Symbol]],
    max_order: int,
    *,
    alphabet: Iterable[Symbol] | None = None,
    name: str = "max_order",
) -> DFA:
    """Forbid every reference substring of length ``max_order + 1``.

    This is the standard regular way to enforce "do not copy beyond max order":
    any generated window longer than ``max_order`` that appears in the reference
    material is rejected.
    """

    if max_order < 0:
        raise ValueError("max_order must be non-negative")

    window = max_order + 1
    forbidden: set[tuple[Symbol, ...]] = set()
    for sequence in reference_sequences:
        tokens = tuple(sequence)
        for index in range(0, len(tokens) - window + 1):
            forbidden.add(tokens[index : index + window])
    return forbidden_substring_acceptor(forbidden, alphabet=alphabet, name=name)
