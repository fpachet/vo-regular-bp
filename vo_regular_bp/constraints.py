"""Public constraint specifications and compiler helpers.

The classes in this module are intentionally project-agnostic: symbols can be
pitches, durations, tokens, events, or any other hashable objects.  The compiler
keeps purely positional constraints as time-indexed masks and compiles regular
constraints into deterministic acceptors.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Hashable

from .acceptors import (
    DFA,
    all_of,
    cumulative_meter_acceptor,
    dense_forbidden_substring_acceptor,
    forbidden_substring_acceptor,
    meter_acceptor,
)
from .context import Symbol
from .positional_bp import PositionConstraint, PositionConstraints


MeterClass = Hashable


@dataclass(frozen=True)
class MeterConstraint:
    """Per-position meter/classes for emitted symbols.

    ``pattern`` entries may be a single class, an iterable of allowed classes,
    or ``None`` as a wildcard. ``symbol_to_meter`` maps each emitted symbol to
    its meter class.
    """

    pattern: Sequence[MeterClass | Iterable[MeterClass] | None]
    symbol_to_meter: Mapping[Symbol, MeterClass] | Callable[[Symbol], MeterClass]
    name: str = "meter"


@dataclass(frozen=True)
class CumulativeMeterConstraint:
    """Cumulative-cost meter predicate.

    The DFA state stores the cumulative cost before each emission. ``predicate``
    is called as ``predicate(total_cost_before, symbol, one_based_position)``.
    If ``length`` is omitted, the compiler uses the requested generation length.
    """

    cost: Mapping[Symbol, int] | Callable[[Symbol], int]
    predicate: Callable[[int, Symbol, int], bool] | None = None
    length: int | None = None
    max_cost: int | None = None
    accept_costs: Iterable[int] | Callable[[int], bool] | None = None
    end_symbol: Symbol | None = None
    name: str = "cumulative_meter"


@dataclass(frozen=True)
class ConstraintSet:
    """Constraints for fixed-horizon symbolic generation.

    Positional constraints are represented as masks and do not inflate regular
    DFA state. Regular constraints are compiled into acceptors and intersected.
    """

    positional: PositionConstraints = field(default_factory=dict)
    forbidden_substrings: Iterable[Sequence[Symbol]] = field(default_factory=tuple)
    regular_acceptors: Sequence[DFA] = field(default_factory=tuple)
    meter: MeterConstraint | None = None
    cumulative_meter: CumulativeMeterConstraint | None = None


@dataclass(frozen=True)
class CompiledConstraints:
    """Compiled constraints consumed by BP backends."""

    positional: dict[int, PositionConstraint]
    regular_acceptor: DFA | None = None

    @property
    def has_regular(self) -> bool:
        return self.regular_acceptor is not None


def compile_constraints(
    constraints: ConstraintSet | None,
    *,
    length: int,
    alphabet: Iterable[Symbol],
    prefer_dense_forbidden: bool = True,
) -> CompiledConstraints:
    """Compile public constraint specs into masks plus an optional DFA."""

    if length < 0:
        raise ValueError("length must be non-negative")
    spec = constraints or ConstraintSet()
    alphabet_tuple = tuple(alphabet)

    positional = dict(spec.positional)
    _validate_positions(positional, length)

    regulars: list[DFA] = list(spec.regular_acceptors)
    forbidden = tuple(tuple(pattern) for pattern in spec.forbidden_substrings)
    if forbidden:
        if prefer_dense_forbidden:
            regulars.append(
                dense_forbidden_substring_acceptor(
                    forbidden,
                    alphabet=alphabet_tuple,
                    name="dense_forbidden_substring",
                )
            )
        else:
            regulars.append(
                forbidden_substring_acceptor(
                    forbidden,
                    alphabet=alphabet_tuple,
                    name="forbidden_substring",
                )
            )

    if spec.meter is not None:
        if len(spec.meter.pattern) != length:
            raise ValueError("meter pattern length must match generation length")
        regulars.append(
            meter_acceptor(
                spec.meter.pattern,
                spec.meter.symbol_to_meter,
                alphabet=alphabet_tuple,
                name=spec.meter.name,
            )
        )

    if spec.cumulative_meter is not None:
        cumulative = spec.cumulative_meter
        if cumulative.length is not None and cumulative.length != length:
            raise ValueError("cumulative meter length must match generation length")
        regulars.append(
            cumulative_meter_acceptor(
                cumulative.length if cumulative.length is not None else length,
                cumulative.cost,
                cumulative.predicate,
                alphabet=alphabet_tuple,
                max_cost=cumulative.max_cost,
                accept_costs=cumulative.accept_costs,
                end_symbol=cumulative.end_symbol,
                name=cumulative.name,
            )
        )

    regular_acceptor = _combine_regulars(regulars)
    return CompiledConstraints(
        positional=positional,
        regular_acceptor=regular_acceptor,
    )


def _combine_regulars(regulars: Sequence[DFA]) -> DFA | None:
    if not regulars:
        return None
    if len(regulars) == 1:
        return regulars[0]
    return all_of(*regulars, name="constraint_set")


def _validate_positions(constraints: Mapping[int, PositionConstraint], length: int) -> None:
    for position in constraints:
        if position < 0 or position >= length:
            raise IndexError(f"constraint position {position} is outside length {length}")
