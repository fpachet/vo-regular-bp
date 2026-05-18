import math

from vo_regular_bp import ContextGraph, positional_acceptor, run_bp
from vo_regular_bp.experimental import alergia_merge, alergia_metadata


def test_alergia_does_not_merge_incompatible_distributions():
    graph = ContextGraph.from_counts(
        {
            ("a",): {"x": 9000, "y": 1000},
            ("b",): {"x": 1000, "y": 9000},
            ("x",): {"x": 10000},
            ("y",): {"y": 10000},
        },
        max_order=1,
        start_state=("a",),
    )

    merged = alergia_merge(graph, alpha=0.01, min_support=10)
    metadata = alergia_metadata(merged)

    assert metadata is not None
    assert metadata.original_state_count == metadata.merged_state_count
    assert merged.outgoing(("a",)) != merged.outgoing(("b",))


def test_alergia_merges_statistically_compatible_distributions():
    graph = ContextGraph.from_counts(
        {
            ("a",): {"x": 50, "y": 50},
            ("b",): {"x": 52, "y": 48},
            ("x",): {"x": 100},
            ("y",): {"y": 100},
        },
        max_order=1,
        start_state=("a",),
    )

    merged = alergia_merge(graph, alpha=0.01, min_support=10)
    metadata = alergia_metadata(merged)

    assert metadata is not None
    assert metadata.merged_state_count < metadata.original_state_count
    assert metadata.state_to_class[("a",)] == metadata.state_to_class[("b",)]
    assert merged.outgoing(("a",)) == merged.outgoing(("b",))


def test_alergia_respects_min_support():
    graph = ContextGraph.from_counts(
        {
            ("a",): {"x": 1, "y": 1},
            ("b",): {"x": 1, "y": 1},
            ("x",): {"x": 100},
            ("y",): {"y": 100},
        },
        max_order=1,
        start_state=("a",),
    )

    merged = alergia_merge(graph, alpha=0.01, min_support=10)
    metadata = alergia_metadata(merged)

    assert metadata is not None
    assert metadata.original_state_count == metadata.merged_state_count
    assert metadata.state_to_class[("a",)] != metadata.state_to_class[("b",)]


def test_alergia_recursive_compatibility_blocks_bad_successor_merge():
    graph = ContextGraph.from_counts(
        {
            ("a",): {"x": 100},
            ("b",): {"x": 100},
            ("a", "x"): {"m": 100},
            ("b", "x"): {"n": 100},
            ("m",): {"m": 100},
            ("n",): {"n": 100},
        },
        max_order=2,
        start_state=("a",),
    )

    recursive = alergia_merge(graph, alpha=0.01, min_support=10, recursive=True)
    one_step = alergia_merge(graph, alpha=0.01, min_support=10, recursive=False)

    assert alergia_metadata(recursive).state_to_class[("a",)] != alergia_metadata(
        recursive
    ).state_to_class[("b",)]
    assert alergia_metadata(one_step).state_to_class[("a",)] == alergia_metadata(
        one_step
    ).state_to_class[("b",)]


def test_alergia_result_is_normalized_and_usable_by_bp():
    graph = ContextGraph.from_counts(
        {
            ("a",): {"x": 50, "y": 50},
            ("b",): {"x": 52, "y": 48},
            ("x",): {"x": 100},
            ("y",): {"y": 100},
        },
        max_order=1,
        start_state=("a",),
    )

    merged = alergia_merge(graph, alpha=0.01, min_support=10)
    for state in merged.states:
        outgoing = merged.outgoing(state)
        if outgoing:
            assert math.isclose(sum(edge.probability for edge in outgoing), 1.0)

    acceptor = positional_acceptor(2, alphabet=merged.alphabet)
    result = run_bp(merged, acceptor, length=2, start_context=("b",))

    assert result.partition_function > 0.0
    assert len(result.sample(rng=0)) == 2


def test_alergia_changes_model_only_when_explicitly_called():
    graph = ContextGraph.from_counts(
        {
            ("a",): {"x": 50, "y": 50},
            ("b",): {"x": 50, "y": 50},
            ("x",): {"x": 100},
            ("y",): {"y": 100},
        },
        max_order=1,
        start_state=("a",),
    )

    merged = alergia_merge(graph, alpha=0.01, min_support=10)

    assert len(graph.states) == 4
    assert len(merged.states) == 3
    assert not hasattr(graph, "alergia_metadata")


def test_alergia_symbol_projection_supplies_client_semantics():
    graph = ContextGraph.from_counts(
        {
            ("a",): {60: 50, 62: 50},
            ("b",): {72: 52, 74: 48},
            (60,): {60: 100},
            (62,): {62: 100},
            (72,): {72: 100},
            (74,): {74: 100},
        },
        max_order=1,
        start_state=("a",),
    )

    raw = alergia_merge(graph, alpha=0.01, min_support=10)
    projected = alergia_merge(
        graph,
        alpha=0.01,
        min_support=10,
        symbol_projection=lambda symbol: symbol % 12 if isinstance(symbol, int) else symbol,
    )

    raw_metadata = alergia_metadata(raw)
    projected_metadata = alergia_metadata(projected)
    assert raw_metadata is not None
    assert projected_metadata is not None
    assert raw_metadata.state_to_class[("a",)] != raw_metadata.state_to_class[("b",)]
    assert projected_metadata.state_to_class[("a",)] == projected_metadata.state_to_class[("b",)]
    assert projected_metadata.symbol_projection == "<lambda>"

    outgoing_symbols = {edge.symbol for edge in projected.outgoing(("a",))}
    assert outgoing_symbols == {60, 62, 72, 74}


def test_alergia_transition_projection_supplies_context_aware_semantics():
    graph = ContextGraph.from_counts(
        {
            ("a", 60): {62: 100},
            ("b", 72): {74: 100},
            (60, 62): {64: 100},
            (72, 74): {76: 100},
            (62, 64): {64: 100},
            (74, 76): {76: 100},
        },
        max_order=2,
        start_state=("a", 60),
    )

    raw = alergia_merge(graph, alpha=0.01, min_support=10)
    projected = alergia_merge(
        graph,
        alpha=0.01,
        min_support=10,
        transition_projection=_interval_projection,
    )

    raw_metadata = alergia_metadata(raw)
    projected_metadata = alergia_metadata(projected)
    assert raw_metadata is not None
    assert projected_metadata is not None
    assert raw_metadata.state_to_class[("a", 60)] != raw_metadata.state_to_class[("b", 72)]
    assert projected_metadata.state_to_class[("a", 60)] == projected_metadata.state_to_class[("b", 72)]
    assert projected_metadata.state_to_class[(60, 62)] == projected_metadata.state_to_class[(72, 74)]
    assert projected_metadata.transition_projection == "_interval_projection"
    assert projected_metadata.projection_kind == "transition"

    for state in projected.states:
        outgoing = projected.outgoing(state)
        if outgoing:
            assert math.isclose(sum(edge.probability for edge in outgoing), 1.0)

    acceptor = positional_acceptor(2, alphabet=projected.alphabet)
    result = run_bp(projected, acceptor, length=2, start_context=("b", 72))
    assert result.partition_function > 0.0


def test_alergia_transition_projection_takes_precedence_over_symbol_projection():
    graph = ContextGraph.from_counts(
        {
            ("a", 60): {62: 100},
            ("b", 72): {74: 100},
            (60, 62): {62: 100},
            (72, 74): {74: 100},
        },
        max_order=2,
        start_state=("a", 60),
    )

    merged = alergia_merge(
        graph,
        alpha=0.01,
        min_support=10,
        symbol_projection=lambda symbol: ("raw", symbol),
        transition_projection=_interval_projection,
    )
    metadata = alergia_metadata(merged)

    assert metadata is not None
    assert metadata.projection_kind == "transition"
    assert metadata.symbol_projection == "<lambda>"
    assert metadata.transition_projection == "_interval_projection"
    assert metadata.state_to_class[("a", 60)] == metadata.state_to_class[("b", 72)]


def _interval_projection(state, symbol, _edge):
    if state and isinstance(state[-1], int) and isinstance(symbol, int):
        return symbol - state[-1]
    return symbol
