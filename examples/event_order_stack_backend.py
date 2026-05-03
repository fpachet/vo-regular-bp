"""Event-object order-stack generation with symbolic constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (
    EventCodec,
    LongestFeasiblePolicy,
    combine_constraints,
    final_pitch_class,
    meter_pattern,
    prepare_constrained_order_stack_from_events,
)


@dataclass(frozen=True)
class Note:
    pitch: int
    duration: int


training = (
    Note(60, 1),
    Note(62, 1),
    Note(64, 2),
    Note(60, 1),
    Note(65, 1),
    Note(72, 2),
    Note(60, 1),
)

codec = EventCodec(
    event_to_symbol=lambda note: (note.pitch, note.duration),
    symbol_to_event=lambda symbol: Note(symbol[0], symbol[1]),
)

constraints = combine_constraints(
    final_pitch_class(0, length=2, symbol_to_pitch=lambda symbol: symbol[0]),
    meter_pattern(
        ("short", "long"),
        lambda symbol: "long" if symbol[1] == 2 else "short",
    ),
)

backend = prepare_constrained_order_stack_from_events(
    [training],
    constraints,
    codec=codec,
    max_order=1,
    length=2,
    prefix=(Note(60, 1),),
    policy=LongestFeasiblePolicy(),
)

generated = backend.sample_events_with_orders(rng=0)
print(generated.events)
print(generated.orders)
print(backend.diagnostics.as_dict())
