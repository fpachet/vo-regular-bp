"""Generate note events with an exact total generated duration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import duration_total_constraint, prepare_continuation_backend  # noqa: E402


@dataclass(frozen=True)
class Note:
    pitch: int
    duration: int


training = (
    Note(60, 4),
    Note(62, 4),
    Note(64, 4),
    Note(65, 4),
    Note(67, 4),
    Note(69, 4),
    Note(71, 4),
    Note(72, 4),
    Note(60, 4),
    Note(64, 8),
    Note(65, 4),
    Note(67, 4),
    Note(60, 4),
    Note(62, 4),
    Note(65, 8),
    Note(67, 4),
    Note(69, 4),
    Note(60, 4),
)

constraints = duration_total_constraint(
    32,
    horizon=8,
    symbol_to_duration=lambda symbol: symbol[1],
)

backend = prepare_continuation_backend(
    [training],
    prefix=(Note(60, 4),),
    horizon=8,
    max_order=1,
    constraints=constraints,
    event_to_symbol=lambda note: (note.pitch, note.duration),
)

generated = backend.sample_events_with_orders(rng=0)
print(generated.events)
print(f"total_duration={sum(note.duration for note in generated.events)}")
print(f"orders={generated.orders}")
print(backend.diagnostics.as_dict())
