from itertools import product
import math
import random

from vo_regular_bp import (
    ConstraintSet,
    CumulativeMeterConstraint,
    LongestFeasiblePolicy,
    MeterConstraint,
    OrderStackModel,
    run_constrained_order_stack,
)


def test_backend_positional_only_uses_masks():
    model = OrderStackModel.from_sequences([(0, 1, 0, 2, 0, 1, 0)], max_order=1)

    result = run_constrained_order_stack(
        model,
        ConstraintSet(positional={1: {0}}),
        length=2,
        prefix=(0,),
        policy=LongestFeasiblePolicy(),
    )

    assert result.constraints == {1: {0}}
    assert not hasattr(result, "acceptor")
    samples = [result.sample(rng=random.Random(seed)) for seed in range(10)]
    assert all(sample[-1] == 0 for sample in samples)


def test_backend_forbidden_substrings_keep_positional_mask_separate():
    training = (0, 1, 2, 1, 0, 2, 1, 2, 0)
    forbidden = {tuple(training[index : index + 3]) for index in range(len(training) - 2)}
    model = OrderStackModel.from_sequences([training], max_order=1)

    result = run_constrained_order_stack(
        model,
        ConstraintSet(
            positional={2: {0}},
            forbidden_substrings=forbidden,
        ),
        length=3,
        prefix=training[:2],
        policy=LongestFeasiblePolicy(),
    )

    assert result.acceptor.state_count() < 20
    assert result.constraints == {2: {0}}
    assert result.success_mass == 1.0
    assert result.start_order_masses() == ((1, 1.0 / 9.0),)

    samples = [result.sample(rng=random.Random(seed)) for seed in range(20)]
    assert all(sample[-1] == 0 for sample in samples)
    assert all(
        all(tuple(sample[index : index + 3]) not in forbidden for index in range(len(sample) - 2))
        for sample in samples
    )


def test_backend_meter_pattern_constraint():
    model = OrderStackModel.from_sequences([(1, 2, 1, 2, 3, 2, 1, 2)], max_order=1)
    meter = MeterConstraint(
        pattern=("even", "odd", "even"),
        symbol_to_meter=lambda symbol: "even" if int(symbol) % 2 == 0 else "odd",
    )

    result = run_constrained_order_stack(
        model,
        ConstraintSet(meter=meter),
        length=3,
        prefix=(1,),
        policy=LongestFeasiblePolicy(),
    )

    samples = [result.sample(rng=random.Random(seed)) for seed in range(20)]
    assert all(
        tuple("even" if int(symbol) % 2 == 0 else "odd" for symbol in sample)
        == ("even", "odd", "even")
        for sample in samples
    )


def test_backend_cumulative_meter_constraint_matches_enumerated_support():
    model = OrderStackModel.from_sequences([(1, 2, 3, 1, 2, 1, 3, 2, 1)], max_order=1)
    cumulative = CumulativeMeterConstraint(
        cost={1: 1, 2: 2, 3: 3},
        max_cost=4,
        accept_costs={4},
    )

    result = run_constrained_order_stack(
        model,
        ConstraintSet(cumulative_meter=cumulative),
        length=2,
        prefix=(1,),
        policy=LongestFeasiblePolicy(),
    )

    support = {
        sequence
        for sequence in product((1, 2, 3), repeat=2)
        if result.acceptor.accepts(sequence)
    }
    assert support == {(1, 3), (2, 2), (3, 1)}
    assert math.isclose(result.start_order_masses()[0][1], 1.0 / 6.0)

    samples = [result.sample(rng=random.Random(seed)) for seed in range(20)]
    assert all(sum(sample) == 4 for sample in samples)
