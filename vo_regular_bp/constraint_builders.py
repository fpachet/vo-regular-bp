"""Convenience builders for common fixed-horizon constraints."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from .constraints import ConstraintSet, CumulativeMeterConstraint, MeterConstraint
from .context import Symbol
from .positional_bp import PositionConstraint, _allows


def combine_constraints(*constraints: ConstraintSet | None) -> ConstraintSet:
    """Merge constraint sets into one specification.

    Positional constraints at the same position are intersected.  The current
    public ``ConstraintSet`` supports one meter and one cumulative-meter
    constraint; passing multiple distinct constraints of either kind raises.
    """

    positional: dict[int, PositionConstraint] = {}
    forbidden_substrings: list[Sequence[Symbol]] = []
    regular_acceptors = []
    meter = None
    cumulative_meter = None

    for spec in constraints:
        if spec is None:
            continue
        for position, constraint in spec.positional.items():
            if position in positional:
                positional[position] = _intersect_position_constraints(
                    positional[position],
                    constraint,
                )
            else:
                positional[position] = constraint
        forbidden_substrings.extend(spec.forbidden_substrings)
        regular_acceptors.extend(spec.regular_acceptors)
        if spec.meter is not None:
            if meter is not None and spec.meter != meter:
                raise ValueError("combine_constraints supports at most one meter constraint")
            meter = spec.meter
        if spec.cumulative_meter is not None:
            if cumulative_meter is not None and spec.cumulative_meter != cumulative_meter:
                raise ValueError(
                    "combine_constraints supports at most one cumulative meter constraint"
                )
            cumulative_meter = spec.cumulative_meter

    return ConstraintSet(
        positional=positional,
        forbidden_substrings=tuple(forbidden_substrings),
        regular_acceptors=tuple(regular_acceptors),
        meter=meter,
        cumulative_meter=cumulative_meter,
    )


def at_position(position: int, allowed: PositionConstraint | Symbol) -> ConstraintSet:
    """Constrain one zero-based generated position."""

    return ConstraintSet(positional={int(position): _position_constraint(allowed)})


def final_symbol(symbol: Symbol, *, length: int) -> ConstraintSet:
    """Constrain the final generated symbol to one value."""

    return final_symbols({symbol}, length=length)


def final_symbols(allowed: PositionConstraint | Symbol, *, length: int) -> ConstraintSet:
    """Constrain the final generated position."""

    if length <= 0:
        raise ValueError("length must be positive for a final-position constraint")
    return at_position(length - 1, allowed)


def final_pitch_class(
    pitch_class: int,
    *,
    length: int,
    modulo: int = 12,
    symbol_to_pitch: Mapping[Symbol, int] | Callable[[Symbol], int] | None = None,
) -> ConstraintSet:
    """Constrain the final generated symbol by pitch class.

    By default the symbol itself is interpreted as an integer pitch.  Richer
    symbol types can pass a mapping or callable through ``symbol_to_pitch``.
    """

    if modulo <= 0:
        raise ValueError("modulo must be positive")
    target = int(pitch_class) % int(modulo)

    def allows(symbol: Symbol) -> bool:
        pitch = _lookup(symbol_to_pitch, symbol)
        return int(pitch) % int(modulo) == target

    return final_symbols(allows, length=length)


def avoid_copied_ngrams(reference: Sequence[Symbol], ngram_length: int) -> ConstraintSet:
    """Reject generated substrings copied from a reference sequence."""

    if ngram_length <= 0:
        raise ValueError("ngram_length must be positive")
    forbidden = tuple(
        dict.fromkeys(
            tuple(reference[index : index + ngram_length])
            for index in range(0, len(reference) - ngram_length + 1)
        )
    )
    return ConstraintSet(forbidden_substrings=forbidden)


def meter_pattern(
    pattern: Sequence[object | Iterable[object] | None],
    symbol_to_meter: Mapping[Symbol, object] | Callable[[Symbol], object],
    *,
    name: str = "meter",
) -> ConstraintSet:
    """Constrain per-position meter/classes."""

    return ConstraintSet(
        meter=MeterConstraint(
            pattern=pattern,
            symbol_to_meter=symbol_to_meter,
            name=name,
        )
    )


def cumulative_meter(
    cost: Mapping[Symbol, int] | Callable[[Symbol], int],
    predicate: Callable[[int, Symbol, int], bool] | None = None,
    *,
    length: int | None = None,
    max_cost: int | None = None,
    accept_costs: Iterable[int] | Callable[[int], bool] | None = None,
    end_symbol: Symbol | None = None,
    name: str = "cumulative_meter",
) -> ConstraintSet:
    """Constrain cumulative symbolic cost such as duration or beat position."""

    return ConstraintSet(
        cumulative_meter=CumulativeMeterConstraint(
            cost=cost,
            predicate=predicate,
            length=length,
            max_cost=max_cost,
            accept_costs=accept_costs,
            end_symbol=end_symbol,
            name=name,
        )
    )


def padded_duration_total(
    total: int,
    *,
    length: int,
    pad_symbol: Symbol,
    symbol_to_duration: Mapping[Symbol, int] | Callable[[Symbol], int],
    allow_zero_duration_events: bool = False,
    name: str = "padded_duration_total",
) -> ConstraintSet:
    """Constrain generated duration with trailing PAD symbols.

    The generated sequence has fixed length, but its musical prefix may be
    shorter. The PAD symbol has duration zero, may appear only once ``total`` is
    reached, and is absorbing once emitted.
    """

    if total < 0:
        raise ValueError("total must be non-negative")
    if length < 0:
        raise ValueError("length must be non-negative")
    target = int(total)

    def duration_of(symbol: Symbol) -> int:
        if symbol == pad_symbol:
            return 0
        return _lookup(symbol_to_duration, symbol)

    def predicate(current_total: int, symbol: Symbol, one_based_position: int) -> bool:
        del one_based_position
        if symbol == pad_symbol:
            return current_total == target
        duration = duration_of(symbol)
        if duration == 0 and not allow_zero_duration_events:
            return False
        if current_total == target:
            return False
        return True

    return cumulative_meter(
        duration_of,
        predicate,
        length=length,
        max_cost=target,
        accept_costs={target},
        end_symbol=pad_symbol,
        name=name,
    )


def _intersect_position_constraints(
    left: PositionConstraint,
    right: PositionConstraint,
) -> PositionConstraint:
    def allows(symbol: Symbol) -> bool:
        return _allows(left, symbol) and _allows(right, symbol)

    return allows


def _position_constraint(allowed: PositionConstraint | Symbol) -> PositionConstraint:
    if callable(allowed):
        return allowed
    if isinstance(allowed, (str, bytes)):
        return {allowed}
    try:
        iter(allowed)  # type: ignore[arg-type]
    except TypeError:
        return {allowed}  # type: ignore[return-value]
    return allowed  # type: ignore[return-value]


def _lookup(
    mapping_or_callable: Mapping[Symbol, int] | Callable[[Symbol], int] | None,
    symbol: Symbol,
) -> int:
    if mapping_or_callable is None:
        return int(symbol)
    if isinstance(mapping_or_callable, Mapping):
        return int(mapping_or_callable[symbol])
    return int(mapping_or_callable(symbol))
