import math

from vo_regular_bp import (
    ContextGraph,
    LongestFeasiblePolicy,
    OrderStackModel,
    WeightedDFA,
    brute_force_distribution,
    conditional_distribution,
    run_bp,
    run_order_stack_dfa_bp,
    true_acceptor,
)


class SymbolWeightedDFA(WeightedDFA):
    def __init__(self, weights=None):
        super().__init__(
            start_state=0,
            accept_states={0},
            states={0},
            transition_func=lambda _state, _symbol: 0,
            name="symbol_weighted",
        )
        self.weights = dict(weights or {})

    def transition_weight(self, _state, symbol) -> float:
        return float(self.weights.get(symbol, 1.0))


def test_product_bp_unit_weights_reproduce_unweighted_behavior():
    graph = _one_step_graph()
    baseline = run_bp(graph, true_acceptor(), length=1)
    weighted = run_bp(graph, SymbolWeightedDFA(), length=1)

    assert weighted.partition_function == baseline.partition_function
    assert weighted.transition_weights(0, weighted.start_state) == baseline.transition_weights(
        0,
        baseline.start_state,
    )
    assert weighted.conditional_probability(("A",)) == baseline.conditional_probability(("A",))
    assert weighted.conditional_probability(("B",)) == baseline.conditional_probability(("B",))
    assert weighted.sample_many(20, rng=123) == baseline.sample_many(20, rng=123)


def test_product_bp_zero_weight_is_a_hard_ban():
    graph = _one_step_graph()
    acceptor = SymbolWeightedDFA({"B": 0.0})
    result = run_bp(graph, acceptor, length=1)

    assert math.isclose(result.partition_function, 0.5)
    assert result.conditional_probability(("A",)) == 1.0
    assert result.conditional_probability(("B",)) == 0.0
    assert not acceptor.accepts(("B",))
    assert set(result.sample_many(50, rng=123)) == {("A",)}


def test_product_bp_soft_weight_lowers_but_keeps_probability():
    graph = _one_step_graph()
    acceptor = SymbolWeightedDFA({"B": 0.25})
    result = run_bp(graph, acceptor, length=1)
    masses = brute_force_distribution(graph, acceptor, length=1)
    exact = conditional_distribution(masses)

    assert math.isclose(result.partition_function, 0.625)
    assert masses == {("A",): 0.5, ("B",): 0.125}
    assert math.isclose(exact[("A",)], 0.8)
    assert math.isclose(exact[("B",)], 0.2)
    assert math.isclose(result.conditional_probability(("A",)), exact[("A",)])
    assert math.isclose(result.conditional_probability(("B",)), exact[("B",)])


def test_product_bp_sampling_frequencies_reflect_soft_weights():
    graph = _one_step_graph()
    result = run_bp(graph, SymbolWeightedDFA({"B": 0.25}), length=1)
    samples = result.sample_many(5_000, rng=999)
    b_rate = samples.count(("B",)) / len(samples)

    assert 0.16 <= b_rate <= 0.24


def test_order_stack_bp_uses_regular_transition_weights_for_sampling():
    model = _one_step_order_stack_model()
    result = run_order_stack_dfa_bp(
        model,
        SymbolWeightedDFA({"B": 0.25}),
        length=1,
        prefix=("S",),
        policy=LongestFeasiblePolicy(),
    )

    assert result.start_order_masses() == ((1, 0.625),)
    candidate_set = result._candidate_sets(0, ("S",), result.start_acceptor_state)[0]
    weights_by_symbol = {
        edge.symbol: weight
        for edge, weight in zip(candidate_set.edges, candidate_set.weights)
    }
    assert weights_by_symbol == {"A": 0.5, "B": 0.125}

    samples = result.sample_many_with_orders(5_000, rng=321)
    b_rate = sum(sequence == ("B",) for sequence, _orders in samples) / len(samples)
    assert 0.16 <= b_rate <= 0.24


def test_order_stack_bp_zero_weight_is_a_hard_ban():
    result = run_order_stack_dfa_bp(
        _one_step_order_stack_model(),
        SymbolWeightedDFA({"B": 0.0}),
        length=1,
        prefix=("S",),
        policy=LongestFeasiblePolicy(),
    )

    assert result.start_order_masses() == ((1, 0.5),)
    candidate_set = result._candidate_sets(0, ("S",), result.start_acceptor_state)[0]
    assert tuple(edge.symbol for edge in candidate_set.edges) == ("A",)
    assert {sequence for sequence, _orders in result.sample_many_with_orders(50, rng=123)} == {
        ("A",)
    }


def _one_step_graph():
    return ContextGraph.from_probabilities(
        {
            (): {"A": 0.5, "B": 0.5},
        },
        max_order=0,
    )


def _one_step_order_stack_model():
    return OrderStackModel.from_sequences(
        [
            ("S", "A"),
            ("S", "B"),
        ],
        max_order=1,
    )
