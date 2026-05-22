"""Deterministic regular acceptors used by product BP."""

from __future__ import annotations

import math
from itertools import product
from typing import Callable, Hashable, Iterable, Mapping, Protocol, Sequence

Symbol = Hashable
State = Hashable


class SupportsWeightedTransitions(Protocol):
    """Structural acceptor interface with optional soft transition weights."""

    start_state: State

    def next_state(self, state: State, symbol: Symbol) -> State | None:
        ...

    def is_accepting(self, state: State) -> bool:
        ...

    def transition_weight(self, state: State, symbol: Symbol) -> float:
        ...


class DFA:
    """Small deterministic acceptor interface.

    A transition returning ``None`` means that the emitted symbol is rejected
    from that state. Legal transitions can optionally carry nonnegative
    multiplicative weights through ``transition_weight``; the default is
    ``1.0``, so ordinary hard DFAs are unchanged.
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
        transition_weights: Mapping[State, Mapping[Symbol, float]] | None = None,
        transition_weight_func: Callable[[State, Symbol], float] | None = None,
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
        self._transition_weights = {
            state: dict(symbols)
            for state, symbols in (transition_weights or {}).items()
        }
        self.states = None if states is None else frozenset(states)
        self.alphabet = None if alphabet is None else frozenset(alphabet)
        self._transition_func = transition_func
        self._transition_weight_func = transition_weight_func
        self._accept_func = accept_func
        self.name = name

    def next_state(self, state: State, symbol: Symbol) -> State | None:
        if self._transition_func is not None:
            return self._transition_func(state, symbol)
        return self.transitions.get(state, {}).get(symbol)

    def transition_weight(self, state: State, symbol: Symbol) -> float:
        transition_weight_func = getattr(self, "_transition_weight_func", None)
        if transition_weight_func is not None:
            return float(transition_weight_func(state, symbol))
        transition_weights = getattr(self, "_transition_weights", {})
        return float(transition_weights.get(state, {}).get(symbol, 1.0))

    def is_accepting(self, state: State) -> bool:
        if self._accept_func is not None:
            return bool(self._accept_func(state))
        return state in self.accept_states

    def accepts(self, sequence: Sequence[Symbol]) -> bool:
        state = self.start_state
        for symbol in sequence:
            current_state = state
            state = self.next_state(state, symbol)
            if state is None:
                return False
            if transition_weight(self, current_state, symbol) <= 0.0:
                return False
        return self.is_accepting(state)

    def state_count(self) -> int | None:
        if self.states is None:
            return None
        return len(self.states)


class WeightedDFA(DFA):
    """DFA subclass documenting weighted regular-constraint semantics.

    Subclasses may override ``transition_weight``. The inherited implementation
    supports either a transition-weight mapping or a callable passed to
    ``DFA.__init__``.
    """


class DenseForbiddenSubstringDFA(DFA):
    """Finite-alphabet forbidden-substring DFA with dense integer states.

    States are integer ids for the same proper-prefix states used by
    :func:`forbidden_substring_acceptor`. A transition value of ``-1`` means
    rejection. The public methods mirror :class:`DFA`, while
    ``dense_transition_by_symbol`` gives hot loops a direct table lookup path.
    """

    rejected_state = -1

    def __init__(
        self,
        *,
        prefixes: Sequence[tuple[Symbol, ...]],
        alphabet: Iterable[Symbol],
        transition_table: Sequence[Sequence[int]],
        name: str = "dense_forbidden_substring",
    ) -> None:
        alphabet_tuple = tuple(alphabet)
        self.start_state = 0
        self.prefixes = tuple(prefixes)
        self.states = frozenset(range(len(self.prefixes)))
        self.accept_states = self.states
        self.alphabet = frozenset(alphabet_tuple)
        self.transition_table = tuple(tuple(int(next_state) for next_state in row) for row in transition_table)
        self.symbol_to_index = {symbol: index for index, symbol in enumerate(alphabet_tuple)}
        self.dense_transition_by_symbol = {
            symbol: tuple(row[index] for row in self.transition_table)
            for symbol, index in self.symbol_to_index.items()
        }
        self.name = name

    def next_state(self, state: State, symbol: Symbol) -> State | None:
        if not isinstance(state, int):
            raise TypeError("dense forbidden-substring states must be integers")
        transitions = self.dense_transition_by_symbol.get(symbol)
        if transitions is None:
            return None
        next_state = transitions[state]
        if next_state == self.rejected_state:
            return None
        return next_state

    def is_accepting(self, state: State) -> bool:
        return isinstance(state, int) and 0 <= state < len(self.prefixes)

    def accepts(self, sequence: Sequence[Symbol]) -> bool:
        state = self.start_state
        for symbol in sequence:
            next_state = self.next_state(state, symbol)
            if next_state is None:
                return False
            if transition_weight(self, state, symbol) <= 0.0:
                return False
            state = next_state
        return self.is_accepting(state)

    def state_count(self) -> int:
        return len(self.prefixes)

    def edge_count(self) -> int:
        return sum(
            1
            for row in self.transition_table
            for next_state in row
            if next_state != self.rejected_state
        )


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

    def weight(state: State, symbol: Symbol) -> float:
        total = 1.0
        for acceptor, part in zip(acceptors, state):
            total *= transition_weight(acceptor, part, symbol)
        return total

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

    dfa = DFA(
        start_state=start,
        states=states,
        alphabet=alphabet,
        transition_func=transition,
        transition_weight_func=weight,
        accept_func=accepting,
        name=name,
    )
    dfa.component_acceptors = tuple(acceptors)  # type: ignore[attr-defined]
    return dfa


def transition_weight(acceptor: object, state: State, symbol: Symbol) -> float:
    """Return a validated multiplicative weight for a legal transition.

    Acceptors without ``transition_weight`` are treated as ordinary hard DFAs
    with weight ``1.0``. A weight of ``0.0`` is allowed and acts as hard
    rejection in BP; negative or non-finite weights are invalid.
    """

    method = getattr(acceptor, "transition_weight", None)
    raw_weight = 1.0 if method is None else method(state, symbol)
    weight = float(raw_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"transition weight for state {state!r} and symbol {symbol!r} "
            f"must be a finite nonnegative number, got {raw_weight!r}"
        )
    return weight


def has_custom_transition_weights(acceptor: object) -> bool:
    method = getattr(type(acceptor), "transition_weight", None)
    default_method = getattr(DFA, "transition_weight")
    if method is not None and method is not default_method:
        return True
    return bool(
        getattr(acceptor, "_transition_weights", None)
        or getattr(acceptor, "_transition_weight_func", None) is not None
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


def cumulative_meter_acceptor(
    length: int,
    cost: Mapping[Symbol, int] | Callable[[Symbol], int],
    predicate: Callable[[int, Symbol, int], bool] | None = None,
    *,
    alphabet: Iterable[Symbol] | None = None,
    max_cost: int | None = None,
    accept_costs: Iterable[int] | Callable[[int], bool] | None = None,
    end_symbol: Symbol | None = None,
    name: str = "cumulative_meter",
) -> DFA:
    """Accept strings satisfying a cumulative meter predicate.

    This is the regular-constraint form of the paper-style meter predicate
    ``pi(c, x, k)``: before emitting symbol ``x`` at 1-based position ``k``,
    the DFA state stores the cumulative cost ``c`` of previous symbols. The
    transition is accepted iff ``predicate(c, x, k)`` holds, then the symbol
    cost is added to the state.

    ``accept_costs`` can be used for final total-cost requirements. If
    ``end_symbol`` is provided, once it appears, all later symbols must be the
    same padding symbol. Passing ``max_cost`` bounds transitions and lets the
    DFA expose an explicit finite state set.
    """

    if length < 0:
        raise ValueError("length must be non-negative")
    if max_cost is not None and max_cost < 0:
        raise ValueError("max_cost must be non-negative")

    if callable(cost):
        cost_of = cost
    else:
        cost_map = dict(cost)

        def cost_of(symbol: Symbol) -> int:
            return cost_map[symbol]

    if accept_costs is None:
        accepts_cost = lambda total: True
    elif callable(accept_costs):
        accepts_cost = accept_costs
    else:
        accepted_totals = frozenset(int(total) for total in accept_costs)
        accepts_cost = lambda total: total in accepted_totals

    def transition(state: State, symbol: Symbol) -> State | None:
        if not (
            isinstance(state, tuple)
            and len(state) == 3
            and isinstance(state[0], int)
            and isinstance(state[1], int)
            and isinstance(state[2], bool)
        ):
            raise TypeError("cumulative-meter states must be (position, total_cost, ended)")

        position, total_cost, ended = state
        if position >= length:
            return None
        if ended and symbol != end_symbol:
            return None

        symbol_cost = _nonnegative_int_cost(cost_of(symbol), symbol)
        next_position = position + 1
        next_total = total_cost + symbol_cost
        if max_cost is not None and next_total > max_cost:
            return None
        if predicate is not None and not predicate(total_cost, symbol, next_position):
            return None

        next_ended = ended or (end_symbol is not None and symbol == end_symbol)
        return (next_position, next_total, next_ended)

    def accepting(state: State) -> bool:
        return (
            isinstance(state, tuple)
            and len(state) == 3
            and state[0] == length
            and isinstance(state[1], int)
            and bool(accepts_cost(state[1]))
        )

    states = None
    if max_cost is not None:
        ended_values = (False, True) if end_symbol is not None else (False,)
        states = {
            (position, total, ended)
            for position in range(length + 1)
            for total in range(max_cost + 1)
            for ended in ended_values
        }

    return DFA(
        start_state=(0, 0, False),
        states=states,
        alphabet=alphabet,
        transition_func=transition,
        accept_func=accepting,
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
    patterns_by_len: dict[int, set[tuple[Symbol, ...]]] = {}
    for pattern in patterns:
        patterns_by_len.setdefault(len(pattern), set()).add(pattern)

    if len(patterns_by_len) == 1:
        forbidden_len, forbidden_set = next(iter(patterns_by_len.items()))

        def completes_forbidden(candidate: tuple[Symbol, ...]) -> bool:
            return len(candidate) >= forbidden_len and candidate[-forbidden_len:] in forbidden_set

    else:

        def completes_forbidden(candidate: tuple[Symbol, ...]) -> bool:
            return any(
                len(candidate) >= pattern_len and candidate[-pattern_len:] in pattern_set
                for pattern_len, pattern_set in patterns_by_len.items()
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


def dense_forbidden_substring_acceptor(
    forbidden_patterns: Iterable[Sequence[Symbol]],
    *,
    alphabet: Iterable[Symbol],
    name: str = "dense_forbidden_substring",
) -> DenseForbiddenSubstringDFA:
    """Dense finite-alphabet acceptor for strings with no forbidden substring."""

    patterns = tuple(tuple(pattern) for pattern in forbidden_patterns)
    if any(len(pattern) == 0 for pattern in patterns):
        raise ValueError("empty forbidden patterns would reject every sequence")

    alphabet_tuple = tuple(alphabet)
    prefix_to_id: dict[tuple[Symbol, ...], int] = {(): 0}
    prefixes: list[tuple[Symbol, ...]] = [()]
    for pattern in patterns:
        for prefix_len in range(1, len(pattern)):
            prefix = pattern[:prefix_len]
            if prefix not in prefix_to_id:
                prefix_to_id[prefix] = len(prefixes)
                prefixes.append(prefix)

    max_prefix_len = max((len(prefix) for prefix in prefixes), default=0)
    patterns_by_len: dict[int, set[tuple[Symbol, ...]]] = {}
    for pattern in patterns:
        patterns_by_len.setdefault(len(pattern), set()).add(pattern)

    if len(patterns_by_len) == 1:
        forbidden_len, forbidden_set = next(iter(patterns_by_len.items()))

        def completes_forbidden(candidate: tuple[Symbol, ...]) -> bool:
            return len(candidate) >= forbidden_len and candidate[-forbidden_len:] in forbidden_set

    else:

        def completes_forbidden(candidate: tuple[Symbol, ...]) -> bool:
            return any(
                len(candidate) >= pattern_len and candidate[-pattern_len:] in pattern_set
                for pattern_len, pattern_set in patterns_by_len.items()
            )

    def next_prefix(state: tuple[Symbol, ...], symbol: Symbol) -> tuple[Symbol, ...] | None:
        candidate = state + (symbol,)
        if completes_forbidden(candidate):
            return None
        limit = min(len(candidate), max_prefix_len)
        for size in range(limit, -1, -1):
            suffix = candidate[-size:] if size else ()
            if suffix in prefix_to_id:
                return suffix
        return ()

    transition_table: list[list[int]] = []
    for prefix in prefixes:
        row: list[int] = []
        for symbol in alphabet_tuple:
            next_state = next_prefix(prefix, symbol)
            row.append(DenseForbiddenSubstringDFA.rejected_state if next_state is None else prefix_to_id[next_state])
        transition_table.append(row)

    return DenseForbiddenSubstringDFA(
        prefixes=prefixes,
        alphabet=alphabet_tuple,
        transition_table=transition_table,
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


def _nonnegative_int_cost(value: int, symbol: Symbol) -> int:
    if not isinstance(value, int):
        raise TypeError(f"cost for symbol {symbol!r} must be an integer")
    if value < 0:
        raise ValueError(f"cost for symbol {symbol!r} must be non-negative")
    return value
