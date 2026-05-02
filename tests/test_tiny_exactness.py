from math import isclose
import random

from vo_regular_bp import (
    ContextGraph,
    all_of,
    brute_force_distribution,
    brute_force_partition_function,
    conditional_distribution,
    empirical_distribution,
    meter_acceptor,
    positional_acceptor,
    run_bp,
    total_variation,
)


def tiny_graph():
    return ContextGraph.from_probabilities(
        {
            (): {"A": 0.50, "B": 0.50},
            ("A",): {"A": 0.20, "B": 0.30, "C": 0.50},
            ("B",): {"A": 0.40, "B": 0.10, "C": 0.50},
            ("C",): {"A": 0.70, "B": 0.30},
        },
        max_order=1,
    )


def test_positional_meter_bp_partition_matches_brute_force():
    graph = tiny_graph()
    length = 4
    positional = positional_acceptor(length, {0: {"A", "B"}, 3: {"A"}}, alphabet=graph.alphabet)
    meter = meter_acceptor([None, 0, None, 1], {"A": 1, "B": 0, "C": 0}, alphabet=graph.alphabet)
    acceptor = all_of(positional, meter)

    bp = run_bp(graph, acceptor, length=length)
    z_brute = brute_force_partition_function(graph, acceptor, length=length)

    assert z_brute > 0.0
    assert isclose(bp.partition_function, z_brute, rel_tol=1e-12, abs_tol=1e-12)
    assert bp.unique_product_state_count <= bp.time_indexed_product_state_count
    assert bp.product_edge_count > 0


def test_samples_empirically_match_exact_tiny_distribution():
    graph = tiny_graph()
    length = 3
    acceptor = positional_acceptor(length, {2: {"A", "B"}}, alphabet=graph.alphabet)
    bp = run_bp(graph, acceptor, length=length)
    exact = conditional_distribution(brute_force_distribution(graph, acceptor, length=length))

    rng = random.Random(123)
    samples = bp.sample_many(20_000, rng=rng)
    empirical = empirical_distribution(samples)

    assert total_variation(empirical, exact) < 0.025
