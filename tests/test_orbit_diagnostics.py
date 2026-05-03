import vo_regular_bp as vbp


def test_canonical_integer_shift_key_collapses_transposed_tuples():
    assert vbp.canonical_integer_shift_key((60, 64, 67)) == (
        vbp.canonical_integer_shift_key((65, 69, 72))
    )
    assert vbp.canonical_integer_shift_key(("<START>", 60, 64)) == (
        vbp.canonical_integer_shift_key(("<START>", 72, 76))
    )


def test_forbidden_pattern_orbit_stats_counts_shifted_patterns_once():
    stats = vbp.forbidden_pattern_orbit_stats(
        [(60, 64, 67), (61, 65, 68), (60, 65, 67)],
        alphabet=[60, 61, 64, 65, 67, 68],
    )

    assert stats.pattern_count == 3
    assert stats.pattern_orbit_count == 2
    assert stats.pattern_reduction_factor == 1.5
    assert stats.prefix_state_orbit_count < stats.prefix_state_count


def test_regular_product_orbit_stats_match_bp_edge_count():
    base = (0, 2, 4, 2, 0, 5, 7, 0)
    transforms = vbp.integer_shift_transforms([0, 12])
    augmented = vbp.materialize_transformed_sequences([base], transforms)
    model = vbp.VirtualAugmentedOrderStackModel.from_sequences(
        [base],
        max_order=2,
        transforms=transforms,
    )
    alphabet = tuple(sorted(model.alphabet))
    forbidden = {
        tuple(sequence[index : index + 3])
        for sequence in augmented
        for index in range(len(sequence) - 3 + 1)
    }
    acceptor = vbp.dense_forbidden_substring_acceptor(forbidden, alphabet=alphabet)
    bp = vbp.run_order_stack_masked_dfa_bp(
        model,
        acceptor,
        length=3,
        prefix=(0, 2),
        constraints={2: {0, 12}},
        policy=vbp.LongestFeasiblePolicy(),
    )

    stats = vbp.regular_product_orbit_stats(bp)

    assert stats.product_edges == bp.product_edge_count
    assert stats.product_edge_orbits <= stats.product_edges
    assert stats.time_indexed_product_state_orbits <= stats.time_indexed_product_states
    assert bp.regular_transition_row_count <= bp.product_state_count
    assert bp.regular_transition_row_count > 0
    assert bp.regular_accepted_transition_count > 0
    assert bp.regular_transition_row_cache_misses > 0
