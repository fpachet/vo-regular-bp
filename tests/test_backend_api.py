from itertools import product
import math
import random

from vo_regular_bp import (
    BackendDiagnostics,
    ConstraintSet,
    CumulativeMeterConstraint,
    GeneratedSequence,
    LongestFeasiblePolicy,
    MeterConstraint,
    OrderStackModel,
    prepare_constrained_order_stack,
    prepare_constrained_order_stack_plan,
    prepare_constrained_order_stack_plan_from_sequences,
    prepare_constrained_order_stack_from_sequences,
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


def test_prepared_backend_samples_and_reports_diagnostics():
    model = OrderStackModel.from_sequences([(0, 1, 0, 2, 0, 1, 0)], max_order=1)

    backend = prepare_constrained_order_stack(
        model,
        ConstraintSet(positional={1: {0}}),
        length=2,
        prefix=(0,),
        policy=LongestFeasiblePolicy(),
    )

    diagnostics = backend.diagnostics
    assert isinstance(diagnostics, BackendDiagnostics)
    assert diagnostics.backend == "order_stack_positional"
    assert diagnostics.length == 2
    assert diagnostics.max_order == 1
    assert diagnostics.context_states > 0
    assert diagnostics.context_edges > 0
    assert diagnostics.success_mass == 1.0
    assert diagnostics.start_order_masses == ((1, 1.0),)
    assert diagnostics.as_dict()["backend"] == "order_stack_positional"

    sample = backend.sample_with_orders(rng=random.Random(0))
    assert isinstance(sample, GeneratedSequence)
    assert sample.sequence[-1] == 0
    assert len(sample.orders) == 2
    traced_sequence, trace = backend.sample_with_trace(rng=random.Random(3))
    ordered_sample = backend.sample_with_orders(rng=random.Random(3))
    assert ordered_sample.sequence == traced_sequence
    assert ordered_sample.orders == tuple(step.order for step in trace)

    samples = backend.sample_many_with_orders(5, rng=random.Random(1))
    assert all(isinstance(item, GeneratedSequence) for item in samples)
    assert all(item.sequence[-1] == 0 for item in samples)


def test_prefixless_positional_plan_matches_prefix_bound_backend():
    model = OrderStackModel.from_sequences([(0, 1, 0, 2, 0, 1, 0)], max_order=1)
    constraints = ConstraintSet(positional={1: {0}})

    plan = prepare_constrained_order_stack_plan(
        model,
        constraints,
        length=2,
        policy=LongestFeasiblePolicy(),
    )
    planned = plan.for_prefix((0,))
    direct = prepare_constrained_order_stack(
        model,
        constraints,
        length=2,
        prefix=(0,),
        policy=LongestFeasiblePolicy(),
    )

    assert not plan.is_regular
    assert plan.length == 2
    assert planned.sample_with_orders(rng=random.Random(3)) == direct.sample_with_orders(
        rng=random.Random(3)
    )
    assert planned.diagnostics.start_order_masses == direct.diagnostics.start_order_masses


def test_prepared_backend_from_sequences_supports_regular_constraints():
    training = (0, 1, 2, 1, 0, 2, 1, 2, 0)
    forbidden = {tuple(training[index : index + 3]) for index in range(len(training) - 2)}

    backend = prepare_constrained_order_stack_from_sequences(
        [training],
        ConstraintSet(
            positional={2: {0}},
            forbidden_substrings=forbidden,
        ),
        max_order=1,
        length=3,
        prefix=training[:2],
        policy=LongestFeasiblePolicy(),
    )

    diagnostics = backend.diagnostics
    assert diagnostics.backend == "order_stack_regular"
    assert diagnostics.regular_product_states is not None
    assert diagnostics.regular_product_edges is not None
    assert diagnostics.success_mass == 1.0
    traced_sequence, trace = backend.sample_with_trace(rng=random.Random(3))
    ordered_sample = backend.sample_with_orders(rng=random.Random(3))
    assert ordered_sample.sequence == traced_sequence
    assert ordered_sample.orders == tuple(step.order for step in trace)

    samples = backend.sample_many(10, rng=random.Random(2))
    assert all(sample[-1] == 0 for sample in samples)
    assert all(
        all(tuple(sample[index : index + 3]) not in forbidden for index in range(len(sample) - 2))
        for sample in samples
    )


def test_prefixless_regular_plan_reuses_backward_cache_across_prefixes():
    training = (0, 1, 2, 1, 0, 2, 1, 2, 0)
    forbidden = {tuple(training[index : index + 3]) for index in range(len(training) - 2)}
    constraints = ConstraintSet(
        positional={2: {0}},
        forbidden_substrings=forbidden,
    )

    plan = prepare_constrained_order_stack_plan_from_sequences(
        [training],
        constraints,
        max_order=1,
        length=3,
        policy=LongestFeasiblePolicy(),
    )
    assert plan.is_regular
    assert sum(len(cache.memo) for cache in plan.plan.backwards.values()) == 0

    first = plan.for_prefix(training[:2])
    direct = prepare_constrained_order_stack_from_sequences(
        [training],
        constraints,
        max_order=1,
        length=3,
        prefix=training[:2],
        policy=LongestFeasiblePolicy(),
    )
    warmed = sum(len(cache.memo) for cache in plan.plan.backwards.values())
    assert warmed > 0
    assert first.sample_with_orders(rng=random.Random(3)) == direct.sample_with_orders(
        rng=random.Random(3)
    )

    second = plan.for_prefix((1, 2))
    assert second.result.backwards[1] is first.result.backwards[1]
    assert sum(len(cache.memo) for cache in plan.plan.backwards.values()) >= warmed

    sample = second.sample(rng=random.Random(4))
    assert sample[-1] == 0
    assert all(
        tuple(sample[index : index + 3]) not in forbidden
        for index in range(len(sample) - 2)
    )
