from math import isclose

from vo_regular_bp import (
    ContextGraph,
    LazyBackoffContextModel,
    positional_acceptor,
    run_bp,
    run_positional_bp,
)


def paper_integer_graph() -> ContextGraph:
    return ContextGraph.from_weighted_sequences(
        [
            (10, [0, 1, 2, 4]),
            (10, [0, 1, 3, 5]),
            (1, [0, 1, 3, 4]),
            (1000, [6, 2, 5]),
            (1000, [6, 3, 4]),
        ],
        max_order=2,
    )


def test_positional_bp_matches_positional_acceptor_product():
    graph = paper_integer_graph()
    acceptor = positional_acceptor(2, {1: {4}}, alphabet=range(7))

    product = run_bp(graph, acceptor, length=2, start_context=(0, 1))
    positional = run_positional_bp(
        graph,
        length=2,
        start_context=(0, 1),
        constraints={1: {4}},
    )

    assert isclose(positional.partition_function, 11 / 21, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(positional.partition_function, product.partition_function, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(positional.conditional_probability((2, 4)), 10 / 11, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(positional.conditional_probability((3, 4)), 1 / 11, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(
        positional.conditional_probability((2, 4)),
        product.conditional_probability((2, 4)),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_lazy_backoff_model_matches_eager_backoff_graph():
    sequences = [[0, 1, 2, 3, 2, 3], [0, 1, 3, 2, 2, 0]]
    graph = ContextGraph.from_backoff_sequences(sequences, max_order=2, backoff_weight=0.25)
    lazy = LazyBackoffContextModel.from_sequences(sequences, max_order=2, backoff_weight=0.25)
    constraints = {0: {2, 3}, 3: lambda symbol: symbol != 1}

    eager_result = run_positional_bp(
        graph,
        length=4,
        start_context=(0, 1),
        constraints=constraints,
    )
    lazy_result = run_positional_bp(
        lazy,
        length=4,
        start_context=lazy.prefix_context((0, 1)),
        constraints=constraints,
    )

    assert isclose(lazy_result.partition_function, eager_result.partition_function, rel_tol=1e-12, abs_tol=1e-12)
    assert lazy_result.time_indexed_state_count == eager_result.time_indexed_state_count
    assert lazy_result.edge_count == eager_result.edge_count
