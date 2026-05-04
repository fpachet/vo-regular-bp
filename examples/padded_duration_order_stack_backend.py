"""Generate variable musical length with fixed-horizon PAD symbols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (  # noqa: E402
    append_padding,
    padded_duration_total_constraint,
    prepare_continuation_backend,
)


@dataclass(frozen=True)
class Note:
    pitch: int
    duration: int


pad = Note(-1, 0)
training = (
    Note(60, 4),
    Note(62, 4),
    Note(64, 4),
    Note(65, 4),
    Note(67, 4),
)


def encode(note: Note):
    if note == pad:
        return "<PAD>"
    return (note.pitch, note.duration)


backend = prepare_continuation_backend(
    append_padding([training], pad_symbol=pad, pad_count=2),
    prefix=(Note(60, 4),),
    horizon=6,
    max_order=1,
    constraints=padded_duration_total_constraint(
        16,
        horizon=6,
        pad_symbol="<PAD>",
        symbol_to_duration=lambda symbol: symbol[1],
    ),
    event_to_symbol=encode,
)

generated = backend.sample_events_with_orders(rng=0)
print(generated.events)
print(f"real_duration={sum(note.duration for note in generated.events if note != pad)}")
print(f"orders={generated.orders}")
print(backend.diagnostics.as_dict())
