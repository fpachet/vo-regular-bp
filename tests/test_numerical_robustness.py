import math
from collections import Counter

import pytest

from vo_regular_bp import (
    ContextGraph,
    ConstraintSet,
    OrderStackModel,
    WeightedDFA,
    prepare_constrained_order_stack,
    prepare_until_order_stack,
    run_bp,
    run_positional_bp,
    run_order_stack_dfa_bp,
    true_acceptor,
)


def iid_stack():
    return OrderStackModel(
        {(s,): Counter({"a": 1, "b": 1}) for s in ("s", "a", "b")}, max_order=1
    )


@pytest.mark.parametrize("weight", [1e-20, 1e20, 1e-300, 1e300])
def test_common_soft_weight_cancels_even_outside_float_mass_range(weight):
    graph = ContextGraph.from_counts({(): {"a": 1, "b": 1}}, max_order=0)
    acceptor = WeightedDFA(
        start_state=0,
        accept_states={0},
        transition_func=lambda q, s: 0,
        transition_weight_func=lambda q, s: weight,
    )
    result = run_bp(graph, acceptor, length=20)
    assert math.isclose(result.log_partition_function, 20 * math.log(weight))
    assert math.isclose(result.conditional_probability(("a",) * 20), 2**-20)
    assert result.sample_many(20, rng=3) == run_bp(
        graph, true_acceptor(), length=20
    ).sample_many(20, rng=3)
    stack = run_order_stack_dfa_bp(iid_stack(), acceptor, length=20, prefix=("s",))
    reference = run_order_stack_dfa_bp(
        iid_stack(), true_acceptor(), length=20, prefix=("s",)
    )
    assert math.isclose(stack.start_log_order_masses()[0][1], 20 * math.log(weight))
    assert stack.sample_many_with_orders(
        20, rng=4
    ) == reference.sample_many_with_orders(20, rng=4)


def test_impossible_weighted_language_remains_impossible():
    graph = ContextGraph.from_counts({(): {"a": 1}}, max_order=0)
    acceptor = WeightedDFA(
        start_state=0,
        accept_states={0},
        transition_func=lambda q, s: 0,
        transition_weight_func=lambda q, s: 0,
    )
    result = run_bp(graph, acceptor, length=1200)
    assert result.log_partition_function == -math.inf
    with pytest.raises(ValueError):
        result.sample()


def test_underflow_in_single_edge_product_keeps_support():
    graph = ContextGraph.from_probabilities({(): {"a": 5e-324, "b": 1.0}}, max_order=0)
    acceptor = WeightedDFA(
        start_state=0,
        accept_states={0},
        transition_func=lambda q, s: 0,
        transition_weight_func=lambda q, s: 0.5 if s == "a" else 0,
    )
    result = run_bp(graph, acceptor, length=1)
    # exp(log_mass) may round to either adjacent subnormal at this boundary.
    assert result.partition_function <= 5e-324
    assert math.isfinite(result.log_partition_function)
    assert result.sample(rng=0) == ("a",)


def test_long_regular_horizon_does_not_require_python_recursion():
    result = run_order_stack_dfa_bp(
        iid_stack(), true_acceptor(), length=1200, prefix=("s",)
    )
    assert result.start_order_masses() == ((1, 1.0),)
    assert len(result.sample(rng=1)) == 1200


def test_long_positional_horizon_and_rare_constraint_keep_support():
    graph = ContextGraph.from_counts({(): {"a": 1, "b": 1}}, max_order=0)
    constraints = {t: {"a"} for t in range(1200)}
    result = run_positional_bp(graph, length=1200, constraints=constraints)
    assert result.partition_function == 0
    assert math.isclose(result.log_partition_function, -1200 * math.log(2))
    assert result.sample(rng=1) == ("a",) * 1200
    assert math.isclose(result.conditional_probability(("a",) * 1200), 1)
    backend = prepare_constrained_order_stack(
        iid_stack(),
        ConstraintSet(positional=constraints),
        length=1200,
        prefix=("s",),
    )
    assert math.isclose(
        backend.result.start_log_order_masses()[0][1], -1200 * math.log(2)
    )
    assert backend.sample(rng=1) == ("a",) * 1200


def test_first_hit_lengths_with_unrepresentable_masses_are_weighted_relatively():
    # Exact length masses are 2^-n. Restrict to two lengths in the underflow
    # range; the shorter length must have twice the probability of the longer.
    result = prepare_until_order_stack(
        iid_stack(), prefix=("s",), stop="b", min_length=1200, max_length=1201
    )
    assert result.feasible_lengths == (1200, 1201)
    assert math.isclose(result.length_weights[0] / result.length_weights[1], 2)
    sample = result.sample(rng=1)
    assert sample[-1] == "b" and set(sample[:-1]) == {"a"}


def test_product_transition_rows_are_shared_without_changing_layer_diagnostics():
    graph = ContextGraph.from_counts({(): {"a": 1, "b": 1}}, max_order=0)
    result = run_bp(graph, true_acceptor(), length=100)
    assert result.product_edge_count == 200
    assert len({id(row) for layer in result.edges for row in layer.values()}) == 1


def test_empty_horizon_backend_diagnostics_do_not_index_a_future_layer():
    model = iid_stack()
    for constraints in (None, ConstraintSet(regular_acceptors=(true_acceptor(),))):
        backend = prepare_constrained_order_stack(model, constraints, length=0, prefix=('s',))
        assert backend.sample(rng=0) == ()
        assert backend.diagnostics.success_mass == 1
