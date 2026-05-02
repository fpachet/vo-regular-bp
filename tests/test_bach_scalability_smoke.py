import math
import random
import time

from scripts.eval_bach_positional_direct import (
    LazyBackoffContextModel,
    run_direct_positional_bp,
    run_memo_lazy_direct_positional_bp,
)
from scripts.eval_bach_scalability import (
    BachConfig,
    accepts_from,
    load_bach_pitches,
    prefix_context,
    run_configuration,
)
from vo_regular_bp import ContextGraph


def test_bach_scalability_smoke():
    pitches = load_bach_pitches()
    config = BachConfig(max_order=3, horizon=16, forbidden_ngram=4)

    t0 = time.perf_counter()
    result = run_configuration(
        pitches,
        config,
        prefix=pitches[:6],
        samples=10,
        seed=123,
    )
    elapsed = time.perf_counter() - t0

    assert len(pitches) > 100
    assert math.isfinite(result.bp_s)
    assert math.isfinite(elapsed)
    assert result.bp_unique_states > 0
    assert result.bp_edges > 0
    assert len(result.samples) == 10
    assert len(result.sample_orders) == 10
    assert all(len(orders) == config.horizon for orders in result.sample_orders)
    assert all(0 <= order <= config.max_order for orders in result.sample_orders for order in orders)
    assert result.selected_order_avg > 0.0
    assert result.constraint_violations == 0
    assert all(
        accepts_from(result.acceptor, sample, result.start_acceptor_state)
        for sample in result.samples
    )


def test_optimized_positional_bp_matches_full_graph_baseline():
    pitches = load_bach_pitches()
    max_order = 3
    horizon = 16
    pitch_class = 0
    backoff_weight = 0.25
    prefix = pitches[:6]
    predicate = lambda pitch: int(pitch) % 12 == pitch_class
    constraints = {0: predicate, horizon - 1: predicate}

    graph = ContextGraph.from_backoff_sequences(
        [pitches],
        max_order=max_order,
        backoff_weight=backoff_weight,
    )
    baseline = run_direct_positional_bp(
        graph,
        length=horizon,
        start_context=prefix_context(graph, prefix, max_order),
        constraints=constraints,
    )

    model = LazyBackoffContextModel.from_sequences(
        [pitches],
        max_order=max_order,
        backoff_weight=backoff_weight,
    )
    optimized = run_memo_lazy_direct_positional_bp(
        model,
        length=horizon,
        start_context=model.prefix_context(prefix),
        constraints=constraints,
    )

    assert math.isclose(
        optimized.partition_function,
        baseline.partition_function,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert optimized.time_indexed_state_count == baseline.time_indexed_state_count
    assert optimized.edge_count == baseline.edge_count

    rng = random.Random(123)
    samples = [optimized.sample(rng) for _ in range(20)]
    assert all(sample[0] % 12 == pitch_class for sample in samples)
    assert all(sample[-1] % 12 == pitch_class for sample in samples)
