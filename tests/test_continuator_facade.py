from dataclasses import dataclass
import random

import pytest

from vo_regular_bp import (
    append_padding,
    combine_constraints,
    duration_total_constraint,
    final_pitch_class_constraint,
    final_symbol,
    meter_cycle_constraint,
    padded_duration_total_constraint,
    prepare_continuation_backend,
)


@dataclass(frozen=True)
class Note:
    pitch: int
    duration: int


def test_prepare_continuation_backend_generates_decoded_events():
    training = (
        Note(60, 1),
        Note(65, 1),
        Note(72, 2),
        Note(60, 1),
        Note(67, 1),
        Note(72, 2),
        Note(60, 1),
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
        prefix=(Note(60, 1),),
        horizon=2,
        max_order=1,
        constraints=constraints,
        event_to_symbol=lambda note: (note.pitch, note.duration),
    )

    generated = backend.sample_events_with_orders(rng=random.Random(0))
    assert tuple(note.duration for note in generated.events) == (1, 2)
    assert generated.events[-1].pitch % 12 == 0
    assert generated.symbols == tuple((note.pitch, note.duration) for note in generated.events)
    assert len(generated.orders) == 2
    assert backend.diagnostics.backend == "order_stack_regular"


def test_prepare_continuation_backend_enforces_total_duration_32():
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

    backend = prepare_continuation_backend(
        [training],
        prefix=(Note(60, 4),),
        horizon=8,
        max_order=1,
        constraints=duration_total_constraint(
            32,
            horizon=8,
            symbol_to_duration=lambda symbol: symbol[1],
        ),
        event_to_symbol=lambda note: (note.pitch, note.duration),
    )

    for seed in range(10):
        generated = backend.sample_events_with_orders(rng=random.Random(seed))
        assert sum(note.duration for note in generated.events) == 32
        assert len(generated.events) == 8

    assert backend.diagnostics.backend == "order_stack_regular"


def test_prepare_continuation_backend_enforces_padded_total_duration():
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

    for seed in range(10):
        generated = backend.sample_events_with_orders(rng=random.Random(seed))
        events = generated.events
        assert sum(note.duration for note in events if note != pad) == 16
        first_pad = events.index(pad)
        assert all(note == pad for note in events[first_pad:])
        assert all(note != pad for note in events[:first_pad])

    assert backend.diagnostics.backend == "order_stack_regular"


def test_prepare_continuation_backend_accepts_identity_symbols():
    backend = prepare_continuation_backend(
        [("a", "b", "a")],
        prefix=("a",),
        horizon=1,
        max_order=1,
        constraints=final_symbol("b", length=1),
    )

    assert backend.sample(rng=random.Random(0)) == ("b",)


def test_prepare_continuation_backend_detects_ambiguous_inferred_decoder():
    training = (
        Note(60, 1),
        Note(60, 2),
    )

    with pytest.raises(ValueError, match="maps to multiple events"):
        prepare_continuation_backend(
            [training],
            prefix=(Note(60, 1),),
            horizon=1,
            max_order=1,
            event_to_symbol=lambda note: note.pitch,
        )
