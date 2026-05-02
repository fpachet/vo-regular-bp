"""Continuator-style facade over the reusable constrained BP backend.

This module does not import Continuator.  It provides a small compatibility
surface with Continuator vocabulary so an external Continuator project can swap
its generation backend to ``vo_regular_bp`` with minimal glue code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar

from .adapters import (
    EventCodec,
    EventOrderStackBackend,
    infer_symbol_to_event as infer_symbol_decoder,
    prepare_constrained_order_stack_from_events,
)
from .constraint_builders import (
    cumulative_meter,
    final_pitch_class,
    meter_pattern,
)
from .constraints import ConstraintSet
from .context import Symbol
from .order_stack_bp import OrderPolicy, SingletonAvoidingBackoffPolicy


EventT = TypeVar("EventT")


def prepare_continuation_backend(
    training_events: Iterable[Sequence[EventT]],
    *,
    prefix: Sequence[EventT],
    horizon: int,
    max_order: int,
    constraints: ConstraintSet | None = None,
    event_to_symbol: Callable[[EventT], Symbol] | None = None,
    symbol_to_event: Mapping[Symbol, EventT] | Callable[[Symbol], EventT] | None = None,
    infer_decoder: bool = True,
    strict_decoder: bool = True,
    policy: OrderPolicy | None = None,
    start_event: EventT | None = None,
    end_event: EventT | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
) -> EventOrderStackBackend[EventT]:
    """Prepare a constrained order-stack backend in Continuator vocabulary.

    ``training_events`` and ``prefix`` are project-level event objects.
    ``event_to_symbol`` maps those events to hashable symbols consumed by the
    backend.  If no explicit ``symbol_to_event`` decoder is supplied, the
    function can infer one from the training material when symbols identify
    unique events.
    """

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    material = tuple(tuple(sequence) for sequence in training_events)
    prefix_tuple = tuple(prefix)

    codec = _build_codec(
        material,
        prefix_tuple,
        event_to_symbol=event_to_symbol,
        symbol_to_event=symbol_to_event,
        infer_decoder=infer_decoder,
        strict_decoder=strict_decoder,
    )
    active_policy = policy if policy is not None else SingletonAvoidingBackoffPolicy()

    return prepare_constrained_order_stack_from_events(
        material,
        constraints,
        codec=codec,
        max_order=max_order,
        length=horizon,
        prefix=prefix_tuple,
        policy=active_policy,
        start_event=start_event,
        end_event=end_event,
        alphabet=alphabet,
        prefer_dense_forbidden=prefer_dense_forbidden,
    )


def final_pitch_class_constraint(
    pitch_class: int,
    *,
    horizon: int,
    modulo: int = 12,
    symbol_to_pitch: Mapping[Symbol, int] | Callable[[Symbol], int] | None = None,
) -> ConstraintSet:
    """Constrain the final generated symbol to a pitch class."""

    return final_pitch_class(
        pitch_class,
        length=horizon,
        modulo=modulo,
        symbol_to_pitch=symbol_to_pitch,
    )


def duration_total_constraint(
    total: int,
    *,
    horizon: int | None = None,
    symbol_to_duration: Mapping[Symbol, int] | Callable[[Symbol], int],
    max_total: int | None = None,
    name: str = "duration_total",
) -> ConstraintSet:
    """Constrain the total generated duration/cost."""

    if total < 0:
        raise ValueError("total must be non-negative")
    return cumulative_meter(
        symbol_to_duration,
        length=horizon,
        max_cost=total if max_total is None else max_total,
        accept_costs={total},
        name=name,
    )


def meter_cycle_constraint(
    cycle: Sequence[object | Iterable[object] | None],
    *,
    horizon: int,
    symbol_to_meter: Mapping[Symbol, object] | Callable[[Symbol], object],
    offset: int = 0,
    name: str = "meter_cycle",
) -> ConstraintSet:
    """Constrain generated positions by a repeating meter/class cycle."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if not cycle:
        raise ValueError("cycle must not be empty")
    pattern = tuple(cycle[(offset + position) % len(cycle)] for position in range(horizon))
    return meter_pattern(pattern, symbol_to_meter, name=name)


def _build_codec(
    training_events: Sequence[Sequence[EventT]],
    prefix: Sequence[EventT],
    *,
    event_to_symbol: Callable[[EventT], Symbol] | None,
    symbol_to_event: Mapping[Symbol, EventT] | Callable[[Symbol], EventT] | None,
    infer_decoder: bool,
    strict_decoder: bool,
) -> EventCodec[EventT]:
    if event_to_symbol is None:
        codec = EventCodec.identity()
        if symbol_to_event is not None:
            return codec.with_decoder(symbol_to_event)  # type: ignore[arg-type, return-value]
        return codec  # type: ignore[return-value]

    decoder = symbol_to_event
    if decoder is None and infer_decoder:
        decoder = infer_symbol_decoder(
            tuple(training_events) + (tuple(prefix),),
            event_to_symbol,
            strict=strict_decoder,
        )
    return EventCodec(event_to_symbol=event_to_symbol, symbol_to_event=decoder)
