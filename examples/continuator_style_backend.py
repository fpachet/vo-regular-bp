"""Dependency-free Continuator-style constrained continuation example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (  # noqa: E402
    combine_constraints,
    duration_total_constraint,
    final_pitch_class_constraint,
    meter_cycle_constraint,
    prepare_continuation_backend,
)


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    duration: int
    velocity: int = 96


training = (
    NoteEvent(60, 1),
    NoteEvent(65, 1),
    NoteEvent(72, 2),
    NoteEvent(60, 1),
    NoteEvent(67, 1),
    NoteEvent(72, 2),
    NoteEvent(60, 1),
)

constraints = combine_constraints(
    final_pitch_class_constraint(
        0,
        horizon=2,
        symbol_to_pitch=lambda symbol: symbol[0],
    ),
    duration_total_constraint(
        3,
        horizon=2,
        symbol_to_duration=lambda symbol: symbol[1],
    ),
    meter_cycle_constraint(
        ("short", "long"),
        horizon=2,
        symbol_to_meter=lambda symbol: "long" if symbol[1] == 2 else "short",
    ),
)

backend = prepare_continuation_backend(
    [training],
    prefix=(NoteEvent(60, 1),),
    horizon=2,
    max_order=1,
    constraints=constraints,
    event_to_symbol=lambda note: (note.pitch, note.duration),
)

generated = backend.sample_events_with_orders(rng=0)
print(generated.events)
print(generated.orders)
print(backend.diagnostics.as_dict())
