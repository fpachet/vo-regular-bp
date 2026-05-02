import math
import random
from itertools import product

from scripts.eval_bach_scalability import forbidden_windows
from vo_regular_bp import (
    LongestFeasiblePolicy,
    OrderStackModel,
    all_of,
    dense_forbidden_substring_acceptor,
    forbidden_substring_acceptor,
    positional_acceptor,
    run_order_stack_dfa_bp,
    run_order_stack_masked_dfa_bp,
)


def test_masked_dense_policy_stack_matches_generic_composite_distribution():
    training = (0, 1, 2, 1, 0, 2, 1, 2, 0)
    alphabet = tuple(sorted(set(training)))
    prefix = training[:2]
    horizon = 3
    forbidden_ngram = 3
    final_symbols = frozenset({0})

    model = OrderStackModel.from_sequences([training], max_order=1)
    final_acceptor = positional_acceptor(
        horizon,
        {horizon - 1: final_symbols},
        alphabet=alphabet,
    )
    generic_forbidden = forbidden_substring_acceptor(
        forbidden_windows(training, forbidden_ngram),
        alphabet=alphabet,
    )
    dense_forbidden = dense_forbidden_substring_acceptor(
        forbidden_windows(training, forbidden_ngram),
        alphabet=alphabet,
    )

    baseline = run_order_stack_dfa_bp(
        model,
        all_of(final_acceptor, generic_forbidden),
        length=horizon,
        prefix=prefix,
        policy=LongestFeasiblePolicy(),
    )
    optimized = run_order_stack_masked_dfa_bp(
        model,
        dense_forbidden,
        length=horizon,
        prefix=prefix,
        constraints={horizon - 1: final_symbols},
        policy=LongestFeasiblePolicy(),
    )

    assert optimized.success_mass == baseline.success_mass
    assert optimized.start_order_masses() == baseline.start_order_masses()

    baseline_distribution = {
        sequence: _policy_stack_probability(baseline, sequence)
        for sequence in product(alphabet, repeat=horizon)
    }
    optimized_distribution = {
        sequence: _policy_stack_probability(optimized, sequence)
        for sequence in product(alphabet, repeat=horizon)
    }
    assert baseline_distribution == optimized_distribution
    assert math.isclose(sum(optimized_distribution.values()), 1.0)

    rng = random.Random(123)
    samples = [optimized.sample(rng=rng) for _ in range(25)]
    forbidden = forbidden_windows(training, forbidden_ngram)
    assert all(sample[-1] in final_symbols for sample in samples)
    assert all(
        all(
            tuple(sample[index : index + forbidden_ngram]) not in forbidden
            for index in range(len(sample) - forbidden_ngram + 1)
        )
        for sample in samples
    )


def _policy_stack_probability(result, sequence):
    state = result.start_acceptor_state
    history = list(result.prefix)
    probability = 1.0
    for position, symbol in enumerate(sequence):
        candidate_sets = result._candidate_sets(position, history, state)
        if not candidate_sets:
            return 0.0
        chosen_set = candidate_sets[0]
        total = sum(chosen_set.weights)
        for edge, weight in zip(chosen_set.edges, chosen_set.weights):
            if edge.symbol != symbol:
                continue
            next_state = result.backwards[chosen_set.order].next_acceptor_state(state, symbol)
            if next_state is None:
                return 0.0
            probability *= weight / total
            state = next_state
            history.append(symbol)
            break
        else:
            return 0.0
    if not result.acceptor.is_accepting(state):
        return 0.0
    return probability
