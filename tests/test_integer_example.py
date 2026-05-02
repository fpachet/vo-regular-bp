from math import isclose

from vo_regular_bp import (
    ContextGraph,
    brute_force_distribution,
    conditional_distribution,
    positional_acceptor,
    run_bp,
)


def test_integer_example_recovers_ten_elevenths_and_one_eleventh():
    graph = ContextGraph.from_counts({(): {0: 10, 1: 1}}, max_order=0)
    acceptor = positional_acceptor(1, alphabet=graph.alphabet)

    bp = run_bp(graph, acceptor, length=1)
    masses = brute_force_distribution(graph, acceptor, length=1)
    exact = conditional_distribution(masses)

    assert isclose(bp.partition_function, 1.0)
    assert isclose(exact[(0,)], 10 / 11)
    assert isclose(exact[(1,)], 1 / 11)

    weighted = dict((edge.symbol, weight / bp.partition_function) for edge, weight in bp.transition_weights(0, bp.start_state))
    assert isclose(weighted[0], 10 / 11)
    assert isclose(weighted[1], 1 / 11)
