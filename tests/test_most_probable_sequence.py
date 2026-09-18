from fractions import Fraction
from itertools import product
import math
import random

import pytest

from vo_regular_bp import (
    ContextGraph,
    DFA,
    Edge,
    LazyBackoffContextModel,
    MostProbableSequenceResult,
    WeightedDFA,
    all_of,
    forbidden_substring_acceptor,
    most_probable_sequence,
    positional_acceptor,
    run_bp,
    true_acceptor,
)


def enumerate_weights(graph, acceptor, length, *, start_context=None, start_acceptor_state=None):
    """Independent oracle: enumerate words and multiply exact input-float ratios."""
    weights = {}
    for sequence in product(graph.alphabet, repeat=length):
        context = graph.start_state if start_context is None else tuple(start_context)
        state = acceptor.start_state if start_acceptor_state is None else start_acceptor_state
        weight = Fraction(1)
        for symbol in sequence:
            edge = next((edge for edge in graph.outgoing(context) if edge.symbol == symbol), None)
            next_state = acceptor.next_state(state, symbol)
            if edge is None or next_state is None:
                weight = Fraction(0)
                break
            weight *= Fraction(edge.probability) * Fraction(acceptor.transition_weight(state, symbol))
            context, state = edge.next_state, next_state
        if weight > 0 and acceptor.is_accepting(state):
            weights[sequence] = weight
    return weights


def assert_optimum(graph, acceptor, length, **kwargs):
    weights = enumerate_weights(graph, acceptor, length, **kwargs)
    result = most_probable_sequence(graph, acceptor, length=length, **kwargs)
    assert isinstance(result, MostProbableSequenceResult)
    assert result.feasible == bool(weights)
    if not weights:
        assert result.sequence is None
        assert result.log_weight == -math.inf
    else:
        best = max(weights.values())
        assert result.sequence in weights
        assert weights[result.sequence] == best
        expected_log = math.log(best.numerator) - math.log(best.denominator)
        assert result.log_weight == pytest.approx(expected_log, abs=1e-11)
    return result, weights


@pytest.mark.parametrize("variable_order", [False, True])
@pytest.mark.parametrize("length", range(6))
@pytest.mark.parametrize("constraint_kind", ["hard", "positional_regular", "weighted"])
def test_exhaustive_context_graphs_and_existing_bp(variable_order, length, constraint_kind):
    probabilities = {
        (): {"a": 0.5, "b": 0.5},
        ("a",): {"a": 0.25, "b": 0.75},
        ("b",): {"a": 0.5, "b": 0.5},
    }
    if variable_order:
        probabilities[("a", "b")] = {"a": 0.875, "b": 0.125}
    graph = ContextGraph.from_probabilities(probabilities)
    acceptor = true_acceptor()
    if constraint_kind != "hard":
        acceptor = all_of(
            positional_acceptor(length, {length - 1: {"a"}} if length else {}),
            forbidden_substring_acceptor([("a", "a")]),
        )
    if constraint_kind == "weighted":
        acceptor = all_of(
            acceptor,
            WeightedDFA(
                start_state=0,
                accept_states={0},
                transitions={0: {"a": 1, "b": 0}, 1: {"a": 0, "b": 1}},
                transition_weights={0: {"a": 2.0, "b": 0.25}, 1: {"a": 0.5, "b": 0.0}},
            ),
        )
    result, weights = assert_optimum(graph, acceptor, length)
    bp = run_bp(graph, acceptor, length=length)
    total = sum(weights.values(), Fraction(0))
    assert bp.partition_function == pytest.approx(float(total))
    if result.feasible:
        assert all(sequence in weights for sequence in bp.sample_many(20, rng=3))
        for sequence, weight in weights.items():
            assert bp.conditional_probability(sequence) == pytest.approx(float(weight / total))
    else:
        with pytest.raises(ValueError, match="partition function is zero"):
            bp.sample()


def test_random_small_weighted_products_against_exact_enumeration():
    rng = random.Random(41)
    for _ in range(30):
        probabilities = {}
        for context in ((), (0,), (1,), (0, 1), (1, 1)):
            p = rng.randrange(9) / 8
            probabilities[context] = {0: p, 1: 1 - p}
        graph = ContextGraph.from_probabilities(probabilities)
        acceptor = WeightedDFA(
            start_state=0,
            accept_states={rng.randrange(3)},
            transitions={q: {s: rng.choice([None, 0, 1, 2]) for s in (0, 1)} for q in range(3)},
            transition_weights={
                q: {s: rng.choice([0.0, 0.25, 1.0, 2.0]) for s in (0, 1)} for q in range(3)
            },
        )
        assert_optimum(graph, acceptor, 5)


@pytest.mark.parametrize("accepting", [False, True])
def test_empty_sequence_and_start_overrides(accepting):
    graph = ContextGraph({}, start_state=("unused",))
    acceptor = DFA(start_state="reject", accept_states={"accept"}, transitions={})
    result, _ = assert_optimum(
        graph, acceptor, 0, start_context=["other"],
        start_acceptor_state="accept" if accepting else "reject",
    )
    assert result.sequence == (() if accepting else None)
    assert result.log_weight == (0.0 if accepting else -math.inf)


def test_nonempty_start_overrides_and_terminal_acceptance():
    graph = ContextGraph.from_probabilities({(): {"a": 1.0}, ("b",): {"b": 1.0}})
    acceptor = DFA(
        start_state=0, accept_states={2}, transitions={0: {"b": 1}, 1: {"b": 2}}
    )
    assert not assert_optimum(graph, acceptor, 1, start_context=["b"])[0].feasible
    result, _ = assert_optimum(graph, acceptor, 1, start_context=["b"], start_acceptor_state=1)
    assert result.sequence == ("b",)


@pytest.mark.parametrize("kind", ["source_zero", "acceptor_zero", "rejected", "dead", "nonterminal"])
def test_infeasible_models(kind):
    graph = ContextGraph({(): (Edge("zero", 0.0, ()), Edge("live", 1.0, ()))})
    acceptor = true_acceptor()
    if kind == "source_zero":
        acceptor = positional_acceptor(1, {0: {"zero"}})
    elif kind == "acceptor_zero":
        acceptor = WeightedDFA(
            start_state=0, accept_states={0}, transition_func=lambda q, s: 0,
            transition_weight_func=lambda q, s: 0.0,
        )
    elif kind == "rejected":
        acceptor = DFA(start_state=0, accept_states={0}, transitions={})
    elif kind == "dead":
        graph = ContextGraph({})
    else:
        acceptor = DFA(start_state=0, accept_states={1}, transition_func=lambda q, s: 0)
    assert_optimum(graph, acceptor, 1)


def test_zero_source_edge_cannot_win_a_tie():
    graph = ContextGraph({(): (Edge("zero", 0.0, ()), Edge("live", 1.0, ()))})
    assert assert_optimum(graph, true_acceptor(), 3)[0].sequence == ("live",) * 3


def test_ties_follow_outgoing_order_with_noncomparable_symbols():
    graph = ContextGraph({
        (): (Edge("z", 0.5, ("left",)), Edge(7, 0.5, ("right",))),
        ("left",): (Edge(None, 0.5, ()), Edge((1, 2), 0.5, ())),
        ("right",): (Edge((1, 2), 0.5, ()), Edge(None, 0.5, ())),
    })
    for _ in range(3):
        result, _ = assert_optimum(graph, true_acceptor(), 4)
        assert result.sequence == ("z", None, "z", None)


def test_no_tolerance_collapses_distinct_scores():
    p = 0.5 + 1e-14
    graph = ContextGraph.from_probabilities({(): {"first": 1 - p, "best": p}})
    assert assert_optimum(graph, true_acceptor(), 1)[0].sequence == ("best",)


@pytest.mark.parametrize("weight", [0.5, 1e-300, 1e300])
def test_tiny_probabilities_and_extreme_acceptor_weights(weight):
    graph = ContextGraph.from_probabilities({(): {"rare": 5e-324, "common": 1.0}})
    acceptor = WeightedDFA(
        start_state=0, accept_states={0}, transition_func=lambda q, s: 0,
        transition_weight_func=lambda q, s: weight if s == "rare" else 0.0,
    )
    result, _ = assert_optimum(graph, acceptor, 3)
    assert result.sequence == ("rare",) * 3
    assert math.isfinite(result.log_weight)


def test_large_positive_log_weight():
    graph = ContextGraph.from_probabilities({(): {"a": 1.0}})
    acceptor = WeightedDFA(
        start_state=0, accept_states={0}, transition_func=lambda q, s: 0,
        transition_weight_func=lambda q, s: 1e300,
    )
    assert assert_optimum(graph, acceptor, 3)[0].log_weight > 2000


def test_greedy_conditional_sampling_is_not_sequence_optimization():
    graph = ContextGraph.from_probabilities({
        (): {"a": 0.6, "b": 0.4},
        ("a",): {"x": 0.5, "y": 0.5},
        ("b",): {"x": 1.0},
    })
    acceptor = positional_acceptor(2, {1: {"x", "y"}})
    bp = run_bp(graph, acceptor, length=2)
    state = bp.start_state
    greedy = []
    for time in range(2):
        edge, _ = max(bp.transition_weights(time, state), key=lambda item: item[1])
        greedy.append(edge.symbol)
        state = edge.next_state
    result, weights = assert_optimum(graph, acceptor, 2)
    assert tuple(greedy) == ("a", "x")
    assert result.sequence == ("b", "x")
    assert weights[tuple(greedy)] < weights[result.sequence]


@pytest.mark.parametrize("track_orders", [False, True])
def test_lazy_and_eager_explicit_backoff_mixtures(track_orders):
    sequences = [("a", "b", "a", "a"), ("b", "b", "a")]
    graph = ContextGraph.from_backoff_sequences(sequences, max_order=2, backoff_weight=0.25)
    lazy = LazyBackoffContextModel.from_sequences(
        sequences, max_order=2, backoff_weight=0.25, track_order_weights=track_orders,
    )
    acceptor = all_of(positional_acceptor(4, {3: {"a"}}), forbidden_substring_acceptor([("a", "a")]))
    expected, _ = assert_optimum(graph, acceptor, 4)
    actual = most_probable_sequence(lazy, acceptor, length=4)
    assert actual.sequence == expected.sequence
    assert actual.log_weight == pytest.approx(expected.log_weight)


def test_sparse_lazy_rows_are_cached_and_partition_is_never_computed(monkeypatch):
    import vo_regular_bp.product_bp as implementation

    def unexpected(*args, **kwargs):
        pytest.fail("optimizer must not compute partition/sampling tables or enumerate all states")

    monkeypatch.setattr(implementation, "run_bp", unexpected)
    monkeypatch.setattr(implementation, "log_sum", unexpected)

    class LazyGraph:
        start_state = ()
        calls = []

        @property
        def states(self):
            return unexpected()

        def outgoing(self, context):
            self.calls.append(context)
            return (Edge("a", 1.0, ()), Edge("blocked", 0.0, ("unreachable",)))

    graph = LazyGraph()
    acceptor = DFA(start_state=0, accept_states={0}, transitions={0: {"a": 0}})
    result = most_probable_sequence(graph, acceptor, length=1200)
    assert result.sequence == ("a",) * 1200
    assert result.log_weight == 0.0
    assert graph.calls == [()]


def test_lazy_horizon_does_not_expand_terminal_layer():
    lazy = LazyBackoffContextModel.from_sequences(
        [("a", "b", "c")], max_order=2, backoff_weight=0.25,
    )
    most_probable_sequence(lazy, true_acceptor(), length=0)
    assert not lazy._outgoing_cache
    most_probable_sequence(lazy, true_acceptor(), length=1)
    assert set(lazy._outgoing_cache) == {()}


def test_negative_horizon():
    with pytest.raises(ValueError, match="length must be non-negative"):
        most_probable_sequence(ContextGraph({}), true_acceptor(), length=-1)


@pytest.mark.parametrize("weight", [-1.0, math.inf, math.nan])
def test_invalid_acceptor_weights_use_existing_validation(weight):
    graph = ContextGraph.from_probabilities({(): {"a": 1.0}})
    acceptor = WeightedDFA(
        start_state=0, accept_states={0}, transition_func=lambda q, s: 0,
        transition_weight_func=lambda q, s: weight,
    )
    with pytest.raises(ValueError, match="finite nonnegative"):
        most_probable_sequence(graph, acceptor, length=1)
