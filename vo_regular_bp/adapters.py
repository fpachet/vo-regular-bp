"""Adapters for projects that use rich event objects instead of symbols."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Generic, TypeVar

from .backend import (
    BackendDiagnostics,
    ConstrainedOrderStackBackend,
    ConstrainedOrderStackPlan,
    prepare_constrained_order_stack_plan,
)
from .constraints import ConstraintSet
from .context import Symbol
from .order_stack_bp import OrderPolicy, OrderStackModel


EventT = TypeVar("EventT")


@dataclass(frozen=True)
class EventCodec(Generic[EventT]):
    """Encode project-specific events as hashable symbols.

    ``symbol_to_event`` is optional.  It is only required when callers want to
    decode generated symbols back into event objects.
    """

    event_to_symbol: Callable[[EventT], Symbol]
    symbol_to_event: Mapping[Symbol, EventT] | Callable[[Symbol], EventT] | None = None

    @classmethod
    def identity(cls) -> "EventCodec[Symbol]":
        return cls(lambda event: event, lambda symbol: symbol)

    def encode_event(self, event: EventT) -> Symbol:
        return self.event_to_symbol(event)

    def encode_sequence(self, sequence: Sequence[EventT]) -> tuple[Symbol, ...]:
        return tuple(self.encode_event(event) for event in sequence)

    def encode_sequences(
        self,
        sequences: Iterable[Sequence[EventT]],
    ) -> tuple[tuple[Symbol, ...], ...]:
        return tuple(self.encode_sequence(sequence) for sequence in sequences)

    def decode_symbol(self, symbol: Symbol) -> EventT:
        decoder = self.symbol_to_event
        if decoder is None:
            raise ValueError("EventCodec has no symbol_to_event decoder")
        if isinstance(decoder, Mapping):
            return decoder[symbol]
        return decoder(symbol)

    def decode_sequence(self, sequence: Sequence[Symbol]) -> tuple[EventT, ...]:
        return tuple(self.decode_symbol(symbol) for symbol in sequence)

    def symbol_property(
        self,
        event_property: Callable[[EventT], object],
    ) -> Callable[[Symbol], object]:
        """Lift an event property function to generated symbols."""

        def property_from_symbol(symbol: Symbol) -> object:
            return event_property(self.decode_symbol(symbol))

        return property_from_symbol

    def with_decoder(
        self,
        symbol_to_event: Mapping[Symbol, EventT] | Callable[[Symbol], EventT],
    ) -> "EventCodec[EventT]":
        return EventCodec(self.event_to_symbol, symbol_to_event)


@dataclass(frozen=True)
class GeneratedEvents(Generic[EventT]):
    """Generated events with their encoded symbols and optional order trace."""

    events: tuple[EventT, ...]
    symbols: tuple[Symbol, ...]
    orders: tuple[int, ...] = ()


@dataclass(frozen=True)
class EventOrderStackBackend(Generic[EventT]):
    """Prepared constrained order-stack sampler for event objects."""

    backend: ConstrainedOrderStackBackend
    codec: EventCodec[EventT]

    @property
    def diagnostics(self) -> BackendDiagnostics:
        return self.backend.diagnostics

    def sample_symbols(self, *, rng: random.Random | int | None = None) -> tuple[Symbol, ...]:
        return self.backend.sample(rng=rng)

    def sample_events(self, *, rng: random.Random | int | None = None) -> tuple[EventT, ...]:
        return self.codec.decode_sequence(self.sample_symbols(rng=rng))

    def sample(self, *, rng: random.Random | int | None = None) -> tuple[EventT, ...]:
        return self.sample_events(rng=rng)

    def sample_events_with_orders(
        self,
        *,
        rng: random.Random | int | None = None,
    ) -> GeneratedEvents[EventT]:
        generated = self.backend.sample_with_orders(rng=rng)
        return GeneratedEvents(
            events=self.codec.decode_sequence(generated.sequence),
            symbols=generated.sequence,
            orders=generated.orders,
        )

    def sample_many_events(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[tuple[EventT, ...]]:
        return [
            self.codec.decode_sequence(sequence)
            for sequence in self.backend.sample_many(count, rng=rng)
        ]

    def sample_many_events_with_orders(
        self,
        count: int,
        *,
        rng: random.Random | int | None = None,
    ) -> list[GeneratedEvents[EventT]]:
        return [
            GeneratedEvents(
                events=self.codec.decode_sequence(generated.sequence),
                symbols=generated.sequence,
                orders=generated.orders,
            )
            for generated in self.backend.sample_many_with_orders(count, rng=rng)
        ]


@dataclass(frozen=True)
class EventOrderStackPlan(Generic[EventT]):
    """Prefix-independent event sampler plan with encode/decode glue."""

    plan: ConstrainedOrderStackPlan
    codec: EventCodec[EventT]

    @property
    def length(self) -> int:
        return self.plan.length

    def for_prefix(self, prefix: Sequence[EventT]) -> EventOrderStackBackend[EventT]:
        backend = self.plan.for_prefix(self.codec.encode_sequence(prefix))
        return EventOrderStackBackend(backend=backend, codec=self.codec)

    def sample_events(
        self,
        *,
        prefix: Sequence[EventT],
        rng: random.Random | int | None = None,
    ) -> tuple[EventT, ...]:
        return self.for_prefix(prefix).sample_events(rng=rng)

    def sample_events_with_orders(
        self,
        *,
        prefix: Sequence[EventT],
        rng: random.Random | int | None = None,
    ) -> GeneratedEvents[EventT]:
        return self.for_prefix(prefix).sample_events_with_orders(rng=rng)


def prepare_constrained_order_stack_from_events(
    sequences: Iterable[Sequence[EventT]],
    constraints: ConstraintSet | None = None,
    *,
    codec: EventCodec[EventT],
    max_order: int,
    length: int,
    prefix: Sequence[EventT],
    policy: OrderPolicy | None = None,
    start_event: EventT | None = None,
    end_event: EventT | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
    minimize_source_graphs: bool = False,
) -> EventOrderStackBackend[EventT]:
    """Build and prepare a constrained order-stack backend from event sequences."""

    plan = prepare_constrained_order_stack_plan_from_events(
        sequences,
        constraints,
        codec=codec,
        max_order=max_order,
        length=length,
        policy=policy,
        start_event=start_event,
        end_event=end_event,
        alphabet=alphabet,
        prefer_dense_forbidden=prefer_dense_forbidden,
        minimize_source_graphs=minimize_source_graphs,
    )
    return plan.for_prefix(prefix)


def prepare_constrained_order_stack_plan_from_events(
    sequences: Iterable[Sequence[EventT]],
    constraints: ConstraintSet | None = None,
    *,
    codec: EventCodec[EventT],
    max_order: int,
    length: int,
    policy: OrderPolicy | None = None,
    start_event: EventT | None = None,
    end_event: EventT | None = None,
    alphabet: Iterable[Symbol] | None = None,
    prefer_dense_forbidden: bool = True,
    minimize_source_graphs: bool = False,
) -> EventOrderStackPlan[EventT]:
    """Build an order-stack model from events and prepare a prefixless plan."""

    encoded_sequences = codec.encode_sequences(sequences)
    model = OrderStackModel.from_sequences(
        encoded_sequences,
        max_order=max_order,
        start_symbol=None if start_event is None else codec.encode_event(start_event),
        end_symbol=None if end_event is None else codec.encode_event(end_event),
    )
    plan = prepare_constrained_order_stack_plan(
        model,
        constraints,
        length=length,
        policy=policy,
        alphabet=alphabet,
        prefer_dense_forbidden=prefer_dense_forbidden,
        minimize_source_graphs=minimize_source_graphs,
    )
    return EventOrderStackPlan(plan=plan, codec=codec)


def infer_symbol_to_event(
    sequences: Iterable[Sequence[EventT]],
    event_to_symbol: Callable[[EventT], Symbol],
    *,
    strict: bool = True,
) -> dict[Symbol, EventT]:
    """Infer a decoder lookup from training events.

    In strict mode, two unequal events mapping to the same symbol raise because
    decoding would be ambiguous.  With ``strict=False``, the first event seen for
    each symbol is kept.
    """

    lookup: dict[Symbol, EventT] = {}
    for sequence in sequences:
        for event in sequence:
            symbol = event_to_symbol(event)
            if symbol in lookup:
                if strict and lookup[symbol] != event:
                    raise ValueError(f"symbol {symbol!r} maps to multiple events")
                continue
            lookup[symbol] = event
    return lookup
