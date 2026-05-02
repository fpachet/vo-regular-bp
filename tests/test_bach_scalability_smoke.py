import math
import time

from scripts.eval_bach_scalability import (
    BachConfig,
    accepts_from,
    load_bach_pitches,
    run_configuration,
)


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
