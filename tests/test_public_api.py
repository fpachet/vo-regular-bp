from dataclasses import dataclass

import vo_regular_bp as vbp


def test_public_exports_resolve():
    assert vbp.__version__ == "0.1.0"
    missing = [name for name in vbp.__all__ if not hasattr(vbp, name)]
    assert not missing
    private = [
        name
        for name in vbp.__all__
        if name.startswith("_") and name != "__version__"
    ]
    assert not private


def test_public_symbolic_order_stack_backend_smoke():
    training = (60, 62, 64, 60, 65, 67, 60, 69, 72, 60, 62, 67, 72, 60)
    horizon = 4
    forbidden_length = 4
    model = vbp.OrderStackModel.from_sequences([training], max_order=3)
    constraints = vbp.combine_constraints(
        vbp.final_pitch_class(0, length=horizon),
        vbp.avoid_copied_ngrams(training, forbidden_length),
    )

    backend = vbp.prepare_constrained_order_stack(
        model,
        constraints,
        length=horizon,
        prefix=training[:3],
        policy=vbp.LongestFeasiblePolicy(),
    )
    generated = backend.sample_with_orders(rng=7)
    forbidden = {
        tuple(training[index : index + forbidden_length])
        for index in range(len(training) - forbidden_length + 1)
    }

    assert generated.sequence[-1] % 12 == 0
    assert all(
        tuple(generated.sequence[index : index + forbidden_length]) not in forbidden
        for index in range(len(generated.sequence) - forbidden_length + 1)
    )
    assert backend.diagnostics.backend == "order_stack_regular"
    assert len(generated.orders) == horizon


def test_public_event_and_meter_backend_smoke():
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
    codec = vbp.EventCodec(
        event_to_symbol=lambda note: (note.pitch, note.duration),
        symbol_to_event=lambda symbol: Note(symbol[0], symbol[1]),
    )
    constraints = vbp.combine_constraints(
        vbp.final_pitch_class(0, length=2, symbol_to_pitch=lambda symbol: symbol[0]),
        vbp.meter_pattern(
            ("short", "long"),
            lambda symbol: "long" if symbol[1] == 2 else "short",
        ),
    )

    backend = vbp.prepare_constrained_order_stack_from_events(
        [training],
        constraints,
        codec=codec,
        max_order=1,
        length=2,
        prefix=(Note(60, 1),),
        policy=vbp.LongestFeasiblePolicy(),
    )
    generated = backend.sample_events_with_orders(rng=0)

    assert generated.events[-1].pitch % 12 == 0
    assert tuple("long" if note.duration == 2 else "short" for note in generated.events) == (
        "short",
        "long",
    )
    assert generated.symbols == tuple(codec.encode_event(note) for note in generated.events)
