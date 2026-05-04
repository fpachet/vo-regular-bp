import random

import pytest

from vo_regular_bp import (
    ConstraintSet,
    LongestFeasiblePolicy,
    OrderStackModel,
    prepare_constrained_order_stack,
    prepare_until_end_order_stack,
    prepare_until_order_stack,
)


def test_until_exact_symbol_stop_returns_first_hit_suffix():
    model = OrderStackModel.from_sequences([("A", "B", "C")], max_order=1)

    backend = prepare_until_order_stack(
        model,
        prefix=("A",),
        stop="C",
        min_length=1,
        max_length=3,
        policy=LongestFeasiblePolicy(),
    )

    assert backend.feasible_lengths == (2,)
    sample = backend.sample(rng=random.Random(0))
    assert sample == ("B", "C")
    assert sample[-1] == "C"
    assert "C" not in sample[:-1]


def test_until_end_stop_allows_learned_end_symbol_at_final_position():
    end = "<END>"
    model = OrderStackModel.from_sequences(
        [("A", "B", "C")],
        max_order=1,
        end_symbol=end,
    )

    unconstrained = prepare_constrained_order_stack(
        model,
        length=3,
        prefix=("A",),
        policy=LongestFeasiblePolicy(),
    )
    with pytest.raises(ValueError):
        unconstrained.sample(rng=random.Random(0))

    backend = prepare_until_end_order_stack(
        model,
        prefix=("A",),
        end_symbol=end,
        min_length=3,
        max_length=3,
        policy=LongestFeasiblePolicy(),
    )

    assert backend.sample(rng=random.Random(0)) == ("B", "C", end)


def test_until_set_stop_stops_on_either_symbol_and_forbids_both_earlier():
    model = OrderStackModel.from_sequences(
        [
            ("A", "B", "C"),
            ("A", "B", "D"),
        ],
        max_order=1,
    )

    backend = prepare_until_order_stack(
        model,
        prefix=("A",),
        stop={"C", "D"},
        min_length=2,
        max_length=2,
        policy=LongestFeasiblePolicy(),
    )

    samples = backend.sample_many(20, rng=random.Random(2))
    assert {sample[-1] for sample in samples} <= {"C", "D"}
    assert all(not ({"C", "D"} & set(sample[:-1])) for sample in samples)


def test_until_predicate_stop_supports_symbol_predicates():
    model = OrderStackModel.from_sequences([(1, 2, 12)], max_order=1)

    backend = prepare_until_order_stack(
        model,
        prefix=(1,),
        stop=lambda symbol: symbol % 12 == 0,
        min_length=1,
        max_length=2,
        policy=LongestFeasiblePolicy(),
    )

    sample = backend.sample(rng=random.Random(0))
    assert sample == (2, 12)
    assert sample[-1] % 12 == 0
    assert all(symbol % 12 != 0 for symbol in sample[:-1])


def test_until_stop_supports_tuple_valued_symbols():
    start = (60, 1)
    middle = (62, 1)
    stop = (64, 2)
    model = OrderStackModel.from_sequences([(start, middle, stop)], max_order=1)

    backend = prepare_until_order_stack(
        model,
        prefix=(start,),
        stop=stop,
        min_length=1,
        max_length=2,
        policy=LongestFeasiblePolicy(),
    )

    assert backend.sample(rng=random.Random(0)) == (middle, stop)


def test_until_no_feasible_first_hit_path_raises_cleanly():
    model = OrderStackModel.from_sequences([("A", "B", "C")], max_order=1)

    with pytest.raises(ValueError, match="No feasible first-hit"):
        prepare_until_order_stack(
            model,
            prefix=("A",),
            stop="Z",
            min_length=1,
            max_length=3,
            policy=LongestFeasiblePolicy(),
        )


def test_until_validates_lengths_and_prefix():
    model = OrderStackModel.from_sequences([("A", "B", "C")], max_order=1)

    with pytest.raises(ValueError, match="min_length"):
        prepare_until_order_stack(
            model,
            prefix=("A",),
            stop="C",
            min_length=0,
        )
    with pytest.raises(ValueError, match="max_length"):
        prepare_until_order_stack(
            model,
            prefix=("A",),
            stop="C",
            min_length=3,
            max_length=2,
        )
    with pytest.raises(ValueError, match="non-empty prefix"):
        prepare_until_order_stack(
            model,
            prefix=(),
            stop="C",
        )


def test_until_composes_with_caller_positional_constraint():
    model = OrderStackModel.from_sequences(
        [
            ("A", "B", "C"),
            ("A", "X", "C"),
        ],
        max_order=1,
    )

    backend = prepare_until_order_stack(
        model,
        prefix=("A",),
        stop="C",
        min_length=2,
        max_length=2,
        constraints=ConstraintSet(positional={0: {"B"}}),
        policy=LongestFeasiblePolicy(),
    )

    assert backend.sample(rng=random.Random(0)) == ("B", "C")


def test_until_treats_length_specific_caller_constraints_as_candidate_infeasible():
    model = OrderStackModel.from_sequences([("A", "B", "C")], max_order=1)

    backend = prepare_until_order_stack(
        model,
        prefix=("A",),
        stop="C",
        min_length=1,
        max_length=2,
        constraints=ConstraintSet(positional={1: {"C"}}),
        policy=LongestFeasiblePolicy(),
    )

    assert backend.feasible_lengths == (2,)
    assert backend.sample(rng=random.Random(0)) == ("B", "C")


def test_until_composes_with_forbidden_substrings():
    model = OrderStackModel.from_sequences(
        [
            ("A", "B", "C"),
            ("A", "D", "C"),
        ],
        max_order=1,
    )

    backend = prepare_until_order_stack(
        model,
        prefix=("A",),
        stop="C",
        min_length=2,
        max_length=2,
        constraints=ConstraintSet(forbidden_substrings={("B", "C")}),
        policy=LongestFeasiblePolicy(),
    )

    assert backend.diagnostics.feasible_lengths == (2,)
    assert backend.sample(rng=random.Random(0)) == ("D", "C")
