import math
from itertools import product

import vo_regular_bp as vbp


def test_virtual_augmented_counts_match_explicit_transformed_sequences():
    base = (0, 2, 4, 2, 0)
    transforms = vbp.integer_shift_transforms([0, 2])
    explicit_sequences = vbp.materialize_transformed_sequences([base], transforms)
    explicit = vbp.OrderStackModel.from_sequences(explicit_sequences, max_order=2)
    virtual = vbp.VirtualAugmentedOrderStackModel.from_sequences(
        [base],
        max_order=2,
        transforms=transforms,
    )

    assert virtual.base_context_count < len(explicit.counts)
    assert virtual.virtual_context_count(2) == len(explicit.counts)
    assert virtual.augmented_counts((2,)) == explicit.counts[(2,)]
    assert explicit.counts[(2,)] == {4: 2, 0: 1}
    for context in explicit.counts:
        assert virtual.augmented_counts(context) == explicit.counts[context]
        assert virtual.continuation_distribution_with_order(
            context,
            max_order=2,
        ) == explicit.continuation_distribution_with_order(context, max_order=2)


def test_virtual_augmented_full_graph_matches_explicit_graph():
    base = (0, 1, 0, 2)
    transforms = vbp.integer_shift_transforms([0, 10])
    explicit = vbp.OrderStackModel.from_sequences(
        vbp.materialize_transformed_sequences([base], transforms),
        max_order=2,
    )
    virtual = vbp.VirtualAugmentedOrderStackModel.from_sequences(
        [base],
        max_order=2,
        transforms=transforms,
    )

    for order in (1, 2):
        explicit_graph = explicit.compile_graph(order)
        virtual_graph = virtual.compile_graph(order)
        assert set(virtual_graph.contexts) == set(explicit_graph.contexts)
        assert _edge_signature(virtual_graph) == _edge_signature(explicit_graph)


def test_virtual_augmented_reachable_graph_matches_explicit_distribution():
    base = (0, 2, 4, 2, 0, 5, 7, 0)
    transforms = vbp.integer_shift_transforms([0, 2])
    explicit_sequences = vbp.materialize_transformed_sequences([base], transforms)
    explicit = vbp.OrderStackModel.from_sequences(explicit_sequences, max_order=2)
    virtual = vbp.VirtualAugmentedOrderStackModel.from_sequences(
        [base],
        max_order=2,
        transforms=transforms,
    )
    alphabet = tuple(sorted(explicit.alphabet))
    horizon = 3
    prefix = (0, 2)
    constraints = {horizon - 1: {0, 2}}

    explicit_bp = vbp.run_order_stack_bp(
        explicit,
        length=horizon,
        prefix=prefix,
        constraints=constraints,
        policy=vbp.LongestFeasiblePolicy(),
    )
    virtual_bp = vbp.run_order_stack_bp(
        virtual,
        length=horizon,
        prefix=prefix,
        constraints=constraints,
        policy=vbp.LongestFeasiblePolicy(),
    )

    assert virtual_bp.success_mass == explicit_bp.success_mass
    assert virtual_bp.start_order_masses() == explicit_bp.start_order_masses()
    for sequence in product(alphabet, repeat=horizon):
        assert math.isclose(
            _policy_stack_probability(virtual_bp, sequence),
            _policy_stack_probability(explicit_bp, sequence),
        )


def test_virtual_augmented_regular_bp_matches_explicit_distribution():
    base = (0, 2, 4, 2, 0, 5, 7, 0)
    transforms = vbp.integer_shift_transforms([0, 2])
    explicit_sequences = vbp.materialize_transformed_sequences([base], transforms)
    explicit = vbp.OrderStackModel.from_sequences(explicit_sequences, max_order=2)
    virtual = vbp.VirtualAugmentedOrderStackModel.from_sequences(
        [base],
        max_order=2,
        transforms=transforms,
    )
    alphabet = tuple(sorted(explicit.alphabet))
    horizon = 3
    prefix = (0, 2)
    forbidden_length = 4
    forbidden = {
        tuple(sequence[index : index + forbidden_length])
        for sequence in explicit_sequences
        for index in range(len(sequence) - forbidden_length + 1)
    }
    acceptor = vbp.dense_forbidden_substring_acceptor(forbidden, alphabet=alphabet)
    constraints = {horizon - 1: {0, 2}}

    explicit_bp = vbp.run_order_stack_masked_dfa_bp(
        explicit,
        acceptor,
        length=horizon,
        prefix=prefix,
        constraints=constraints,
        policy=vbp.LongestFeasiblePolicy(),
    )
    virtual_bp = vbp.run_order_stack_masked_dfa_bp(
        virtual,
        acceptor,
        length=horizon,
        prefix=prefix,
        constraints=constraints,
        policy=vbp.LongestFeasiblePolicy(),
    )

    assert virtual_bp.context_state_count < explicit_bp.context_state_count
    assert virtual_bp.success_mass == explicit_bp.success_mass
    assert virtual_bp.start_order_masses() == explicit_bp.start_order_masses()
    for sequence in product(alphabet, repeat=horizon):
        assert math.isclose(
            _regular_policy_stack_probability(virtual_bp, sequence),
            _regular_policy_stack_probability(explicit_bp, sequence),
        )


def _edge_signature(graph):
    return {
        (
            graph.contexts[edge.src],
            edge.symbol,
            graph.contexts[edge.dst],
            edge.probability,
            edge.order,
        )
        for edges in graph.outgoing
        for edge in edges
    }


def _policy_stack_probability(result, sequence):
    history = list(result.prefix)
    probability = 1.0
    for position, symbol in enumerate(sequence):
        candidate_sets = result._candidate_sets(position, history)
        if not candidate_sets:
            return 0.0
        chosen_set = candidate_sets[0]
        total = sum(chosen_set.weights)
        for edge, weight in zip(chosen_set.edges, chosen_set.weights):
            if edge.symbol != symbol:
                continue
            probability *= weight / total
            history.append(symbol)
            break
        else:
            return 0.0
    return probability


def _regular_policy_stack_probability(result, sequence):
    acceptor_state = result.start_acceptor_state
    history = list(result.prefix)
    probability = 1.0
    for position, symbol in enumerate(sequence):
        candidate_sets = result._candidate_sets(position, history, acceptor_state)
        if not candidate_sets:
            return 0.0
        chosen_set = candidate_sets[0]
        total = sum(chosen_set.weights)
        for edge, weight in zip(chosen_set.edges, chosen_set.weights):
            if edge.symbol != symbol:
                continue
            next_state = result.backwards[chosen_set.order].next_acceptor_state(
                acceptor_state,
                symbol,
            )
            if next_state is None:
                return 0.0
            probability *= weight / total
            acceptor_state = next_state
            history.append(symbol)
            break
        else:
            return 0.0
    if not result.acceptor.is_accepting(acceptor_state):
        return 0.0
    return probability
