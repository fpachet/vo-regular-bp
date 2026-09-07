import pytest

from vo_regular_bp import (
    ConstraintSet,
    LongestFeasiblePolicy,
    SingletonAvoidingBackoffPolicy,
    OrderStackModel,
    prepare_constrained_order_stack_plan,
    prepare_until_order_stack,
    true_acceptor,
)
from vo_regular_bp.order_stack_bp import _TransitionRows


def test_transition_budget_bounds_empty_and_oversized_rows():
    cache = _TransitionRows(max_entries=3)
    cache[0] = (1, 2)
    cache[1] = ()
    cache[2] = (3,)
    assert list(cache) == [0, 1]
    cache[3] = (1, 2, 3, 4)
    assert 3 not in cache
    assert cache.entries <= 3 and cache.skipped_rows == 2


@pytest.mark.parametrize("regular", [False, True])
@pytest.mark.parametrize(
    "policy", [LongestFeasiblePolicy(), SingletonAvoidingBackoffPolicy()]
)
def test_plain_orders_and_trace_paths_keep_same_seed_sequences(regular, policy):
    model = OrderStackModel.from_sequences([(0, 1, 0, 2, 0, 1, 2, 1, 0)], max_order=3)
    constraints = (
        ConstraintSet(regular_acceptors=(true_acceptor(),))
        if regular
        else ConstraintSet()
    )
    plan = prepare_constrained_order_stack_plan(
        model, constraints, length=5, policy=policy
    )
    for seed in range(30):
        plain = plan.for_prefix((0, 1)).sample(rng=seed)
        orders = plan.for_prefix((0, 1)).sample_with_orders(rng=seed)
        traced, trace = plan.for_prefix((0, 1)).sample_with_trace(rng=seed)
        assert plain == orders.sequence == traced
        assert orders.orders == tuple(step.order for step in trace)


def test_regular_plans_share_symbol_transitions_across_orders_but_not_plans():
    model = OrderStackModel.from_sequences([(0, 1, 0, 2, 1)], max_order=3)
    constraints = ConstraintSet(regular_acceptors=(true_acceptor(),))
    plan = prepare_constrained_order_stack_plan(model, constraints, length=4).plan
    assert (
        len({id(c.acceptor_symbol_transitions) for c in plan.backwards.values()}) == 1
    )
    other = prepare_constrained_order_stack_plan(model, constraints, length=4).plan
    assert (
        plan.backwards[1].acceptor_symbol_transitions
        is not other.backwards[1].acceptor_symbol_transitions
    )


def test_shared_first_hit_tables_match_generic_length_backends():
    model = OrderStackModel.from_sequences(
        [("a", "a", "b", "a", "b", "b")], max_order=2
    )
    shared = prepare_until_order_stack(model, prefix=("a", "a"), stop="b", max_length=8)
    # An explicit empty ConstraintSet selects the generic fallback.
    generic = prepare_until_order_stack(
        model, prefix=("a", "a"), stop="b", max_length=8, constraints=ConstraintSet()
    )
    assert shared.feasible_lengths == generic.feasible_lengths
    assert shared.length_weights == generic.length_weights
    for left, right in zip(shared.backends, generic.backends):
        assert left.result.start_order_masses() == right.result.start_order_masses()
        for seed in range(10):
            assert left.sample_with_orders(rng=seed) == right.sample_with_orders(
                rng=seed
            )
    assert len({id(b.result.backwards[1].rows) for b in shared.backends}) == 1
