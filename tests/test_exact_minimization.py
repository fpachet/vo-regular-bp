import math
from collections import Counter
from itertools import product

from vo_regular_bp import (
    ContextGraph,
    LongestFeasiblePolicy,
    OrderStackModel,
    dense_forbidden_substring_acceptor,
    exact_context_graph_quotient_stats,
    exact_fixed_order_graph_quotient_stats,
    minimize_context_graph,
    minimize_fixed_order_graph,
    run_order_stack_bp,
    run_order_stack_masked_dfa_bp,
)


def test_minimize_context_graph_preserves_original_context_lookup():
    graph = ContextGraph.from_counts(
        {
            ("a",): {"x": 1, "y": 1},
            ("b",): {"x": 1, "y": 1},
            ("x",): {"x": 1},
            ("y",): {"y": 1},
        },
        max_order=1,
        start_state=("a",),
    )

    stats = exact_context_graph_quotient_stats(graph)
    minimized = minimize_context_graph(graph)

    assert stats.states == 4
    assert stats.classes == 3
    assert minimized.outgoing(("a",)) == minimized.outgoing(("b",))
    assert math.isclose(
        minimized.probability(("x", "x"), start_state=("b",)),
        graph.probability(("x", "x"), start_state=("b",)),
    )


def test_minimize_fixed_order_graph_preserves_aliases_and_orders():
    model = OrderStackModel(
        {
            ("a",): Counter({"x": 1, "y": 1}),
            ("b",): Counter({"x": 1, "y": 1}),
            ("x",): Counter({"x": 1}),
            ("y",): Counter({"y": 1}),
        },
        max_order=1,
    )
    graph = model.compile_graph(1)

    stats = exact_fixed_order_graph_quotient_stats(graph)
    minimized = minimize_fixed_order_graph(graph)

    assert stats.states == 4
    assert stats.classes == 3
    assert minimized.state_id(("a",)) == minimized.state_id(("b",))
    assert {edge.order for edge in minimized.outgoing[minimized.state_id(("a",))]} == {1}


def test_minimized_positional_order_stack_matches_distribution():
    training = ("a", "x", "x", "b", "x", "x", "a", "y", "y", "b", "y", "y")
    alphabet = tuple(sorted(set(training)))
    model = OrderStackModel.from_sequences([training], max_order=1)

    baseline = run_order_stack_bp(
        model,
        length=2,
        prefix=("a",),
        constraints={1: {"x", "y"}},
        policy=LongestFeasiblePolicy(),
    )
    minimized = run_order_stack_bp(
        model,
        length=2,
        prefix=("a",),
        constraints={1: {"x", "y"}},
        policy=LongestFeasiblePolicy(),
        minimize_source_graphs=True,
    )

    assert _policy_distribution(baseline, alphabet) == _policy_distribution(minimized, alphabet)
    traced_sequence, trace = minimized.sample_with_trace(rng=0)
    assert traced_sequence
    assert trace[0].context == ("a",)


def test_minimized_regular_order_stack_matches_distribution():
    training = (0, 1, 2, 1, 0, 2, 1, 2, 0)
    alphabet = tuple(sorted(set(training)))
    forbidden = {
        tuple(training[index : index + 3])
        for index in range(len(training) - 2)
    }
    acceptor = dense_forbidden_substring_acceptor(forbidden, alphabet=alphabet)
    model = OrderStackModel.from_sequences([training], max_order=1)

    baseline = run_order_stack_masked_dfa_bp(
        model,
        acceptor,
        length=3,
        prefix=training[:2],
        constraints={2: {0}},
        policy=LongestFeasiblePolicy(),
    )
    minimized = run_order_stack_masked_dfa_bp(
        model,
        acceptor,
        length=3,
        prefix=training[:2],
        constraints={2: {0}},
        policy=LongestFeasiblePolicy(),
        minimize_source_graphs=True,
    )

    assert baseline.start_order_masses() == minimized.start_order_masses()
    assert _policy_distribution(baseline, alphabet) == _policy_distribution(minimized, alphabet)


def _policy_distribution(result, alphabet):
    return {
        sequence: _policy_probability(result, sequence)
        for sequence in product(alphabet, repeat=result.length)
    }


def _policy_probability(result, sequence):
    acceptor_state = getattr(result, "start_acceptor_state", None)
    history = list(result.prefix)
    probability = 1.0
    for position, symbol in enumerate(sequence):
        if acceptor_state is None:
            candidate_sets = result._candidate_sets(position, history)
        else:
            candidate_sets = result._candidate_sets(position, history, acceptor_state)
        if not candidate_sets:
            return 0.0
        chosen_set = candidate_sets[0]
        total = sum(chosen_set.weights)
        for edge, weight in zip(chosen_set.edges, chosen_set.weights):
            if edge.symbol != symbol:
                continue
            probability *= weight / total
            history.append(symbol)
            if acceptor_state is not None:
                acceptor_state = result.backwards[chosen_set.order].next_acceptor_state(
                    acceptor_state,
                    symbol,
                )
                if acceptor_state is None:
                    return 0.0
            break
        else:
            return 0.0
    if acceptor_state is not None and not result.acceptor.is_accepting(acceptor_state):
        return 0.0
    return probability
