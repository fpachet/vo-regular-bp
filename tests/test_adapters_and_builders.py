from dataclasses import dataclass
import random

import pytest

from vo_regular_bp import (
    EventCodec,
    GeneratedEvents,
    LongestFeasiblePolicy,
    append_padding,
    avoid_copied_ngrams,
    combine_constraints,
    cumulative_meter,
    final_pitch_class,
    final_symbol,
    infer_symbol_to_event,
    meter_pattern,
    padded_duration_total,
    prepare_constrained_order_stack,
    prepare_constrained_order_stack_from_events,
)
from vo_regular_bp import OrderStackModel


@dataclass(frozen=True)
class Note:
    pitch: int
    duration: int


def test_constraint_builders_compose_with_order_stack_backend():
    training = (60, 62, 64, 65, 67, 69, 72, 60)
    model = OrderStackModel.from_sequences([training], max_order=1)
    constraints = combine_constraints(
        final_pitch_class(0, length=2),
        avoid_copied_ngrams(training, 3),
    )

    backend = prepare_constrained_order_stack(
        model,
        constraints,
        length=2,
        prefix=(67,),
        policy=LongestFeasiblePolicy(),
    )

    sample = backend.sample(rng=random.Random(0))
    assert sample[-1] % 12 == 0
    assert all(
        tuple(sample[index : index + 3]) not in constraints.forbidden_substrings
        for index in range(len(sample) - 2)
    )


def test_event_codec_backend_decodes_generated_events_with_meter_constraints():
    sequence = (
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
        [sequence],
        constraints,
        codec=codec,
        max_order=1,
        length=2,
        prefix=(Note(60, 1),),
        policy=LongestFeasiblePolicy(),
    )

    generated = backend.sample_events_with_orders(rng=random.Random(1))
    assert isinstance(generated, GeneratedEvents)
    assert tuple(note.duration for note in generated.events) == (1, 2)
    assert generated.events[-1].pitch % 12 == 0
    assert generated.symbols == tuple((note.pitch, note.duration) for note in generated.events)
    assert len(generated.orders) == 2


def test_event_codec_symbol_property_lifts_event_functions():
    lookup = {
        (60, 1): Note(60, 1),
        (72, 2): Note(72, 2),
    }
    codec = EventCodec(lambda note: (note.pitch, note.duration), lookup)
    duration_of_symbol = codec.symbol_property(lambda note: note.duration)

    assert duration_of_symbol((72, 2)) == 2


def test_infer_symbol_to_event_detects_ambiguous_decoding():
    sequences = [
        (Note(60, 1), Note(60, 2)),
    ]

    with pytest.raises(ValueError, match="maps to multiple events"):
        infer_symbol_to_event(sequences, lambda note: note.pitch)

    lookup = infer_symbol_to_event(sequences, lambda note: note.pitch, strict=False)
    assert lookup[60] == Note(60, 1)


def test_cumulative_meter_builder_accepts_exact_total_cost():
    sequence = (1, 2, 3, 1, 2, 1, 3, 2, 1)
    model = OrderStackModel.from_sequences([sequence], max_order=1)

    backend = prepare_constrained_order_stack(
        model,
        combine_constraints(
            final_symbol(1, length=2),
            cumulative_meter({1: 1, 2: 2, 3: 3}, max_cost=4, accept_costs={4}),
        ),
        length=2,
        prefix=(1,),
        policy=LongestFeasiblePolicy(),
    )

    samples = backend.sample_many(10, rng=random.Random(3))
    assert all(sample[-1] == 1 for sample in samples)
    assert all(sum(sample) == 4 for sample in samples)


def test_padded_duration_builder_forces_absorbing_pad_after_total():
    pad = "<PAD>"
    sequence = ((60, 4), (62, 4), (64, 4), (65, 4))
    model = OrderStackModel.from_sequences(
        append_padding([sequence], pad_symbol=pad, pad_count=2),
        max_order=1,
    )

    backend = prepare_constrained_order_stack(
        model,
        padded_duration_total(
            12,
            length=5,
            pad_symbol=pad,
            symbol_to_duration=lambda symbol: symbol[1],
        ),
        length=5,
        prefix=((60, 4),),
        policy=LongestFeasiblePolicy(),
    )

    sample = backend.sample(rng=random.Random(4))
    first_pad = sample.index(pad)
    assert sum(symbol[1] for symbol in sample if symbol != pad) == 12
    assert all(symbol != pad for symbol in sample[:first_pad])
    assert all(symbol == pad for symbol in sample[first_pad:])


def test_append_padding_rejects_empty_padding():
    with pytest.raises(ValueError, match="pad_count"):
        append_padding([(1, 2, 3)], pad_symbol=0, pad_count=0)
