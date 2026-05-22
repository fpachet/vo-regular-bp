import math
import random
from itertools import product

from scripts.eval_bach_scalability import forbidden_windows
from vo_regular_bp import (
    DFA,
    LongestFeasiblePolicy,
    OrderStackModel,
    all_of,
    append_padding,
    dense_forbidden_substring_acceptor,
    exact_fixed_order_graph_quotient_stats,
    forbidden_substring_acceptor,
    padded_melody_duration_view_quotient_diagnostics,
    positional_acceptor,
    run_order_stack_dfa_bp,
    run_order_stack_masked_dfa_bp,
)
from vo_regular_bp.order_stack_bp import FixedOrderContextGraph, StackEdge


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


def test_padded_melody_fast_path_matches_generic_product_distribution():
    start = "<START>"
    pad = "<PAD>"
    durations = {
        "N1": 1,
        "N2": 2,
        "R1": 1,
        "R2": 2,
    }
    training = append_padding(
        [
            ("N1", "R1", "N2", "N1"),
            ("N2", "N1", "R2", "N1"),
            ("R1", "N1", "N1", "N2"),
        ],
        pad_symbol=pad,
        pad_count=2,
    )
    model = OrderStackModel.from_sequences(training, max_order=1, start_symbol=start)
    length = 4
    target = 3
    min_notes = 2

    duration_acceptor = _lsdb_style_duration_acceptor(
        durations,
        pad_symbol=pad,
        target=target,
    )
    final_acceptor = _lsdb_style_final_note_acceptor(pad_symbol=pad)
    min_note_acceptor = _lsdb_style_min_note_acceptor(
        min_notes,
        pad_symbol=pad,
    )
    optimized = run_order_stack_dfa_bp(
        model,
        all_of(duration_acceptor, final_acceptor, min_note_acceptor),
        length=length,
        prefix=(start,),
        policy=LongestFeasiblePolicy(),
    )
    generic = run_order_stack_dfa_bp(
        model,
        _manual_product_acceptor(
            duration_acceptor,
            final_acceptor,
            min_note_acceptor,
        ),
        length=length,
        prefix=(start,),
        policy=LongestFeasiblePolicy(),
    )

    assert optimized.start_order_masses() == generic.start_order_masses()
    alphabet = tuple(model.alphabet)
    optimized_distribution = {
        sequence: _policy_stack_probability(optimized, sequence)
        for sequence in product(alphabet, repeat=length)
    }
    generic_distribution = {
        sequence: _policy_stack_probability(generic, sequence)
        for sequence in product(alphabet, repeat=length)
    }
    assert optimized_distribution == generic_distribution
    assert math.isclose(sum(optimized_distribution.values()), 1.0)

    sample = optimized.sample(rng=random.Random(7))
    first_pad = sample.index(pad) if pad in sample else len(sample)
    visible = sample[:first_pad]
    assert sum(durations[symbol] for symbol in visible) == target
    assert sum(1 for symbol in visible if symbol.startswith("N")) >= min_notes
    assert visible[-1].startswith("N")
    assert all(symbol == pad for symbol in sample[first_pad:])


def test_duration_view_quotient_reports_exact_compression():
    pad = "<PAD>"
    graph = FixedOrderContextGraph(order=1)
    graph.contexts = [("A",), ("B",)]
    graph.context_to_id = {("A",): 0, ("B",): 1}
    graph.outgoing = [
        [
            StackEdge(0, 0, "N1", 0.50, 1),
            StackEdge(0, 1, "N2", 0.25, 1),
            StackEdge(0, 0, pad, 0.25, 1),
        ],
        [
            StackEdge(1, 1, "N3", 0.50, 1),
            StackEdge(1, 0, "N4", 0.25, 1),
            StackEdge(1, 1, pad, 0.25, 1),
        ],
    ]
    duration_acceptor = _lsdb_style_duration_acceptor(
        {"N1": 1, "N2": 1, "N3": 1, "N4": 1},
        pad_symbol=pad,
        target=2,
    )
    acceptor = all_of(
        duration_acceptor,
        _lsdb_style_final_note_acceptor(pad_symbol=pad),
    )

    ordinary = exact_fixed_order_graph_quotient_stats(graph)
    duration_view = padded_melody_duration_view_quotient_diagnostics(
        {1: graph},
        acceptor,
    )

    assert ordinary.classes == 2
    assert duration_view is not None
    assert duration_view.states == 2
    assert duration_view.classes == 1
    assert duration_view.state_reduction == 2.0
    assert duration_view.projected_edges == 6
    assert duration_view.quotient_edges == 2


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


def _lsdb_style_duration_acceptor(durations, *, pad_symbol, target):
    def transition(state, symbol):
        current_total, ended = state
        if ended and symbol != pad_symbol:
            return None
        if symbol == pad_symbol:
            if current_total != target:
                return None
            return (current_total, True)
        if current_total == target:
            return None
        duration = durations.get(symbol)
        if duration is None or duration <= 0:
            return None
        next_total = current_total + duration
        if next_total > target:
            return None
        return (next_total, False)

    return DFA(
        start_state=(0, False),
        states={(total, ended) for total in range(target + 1) for ended in (False, True)},
        transition_func=transition,
        accept_func=lambda state: state[0] == target,
        name="padded_melody_duration_total",
    )


def _lsdb_style_final_note_acceptor(*, pad_symbol):
    def transition(state, symbol):
        last_real_was_note, ended = state
        if ended and symbol != pad_symbol:
            return None
        if symbol == pad_symbol:
            return (last_real_was_note, True)
        return (str(symbol).startswith("N"), False)

    return DFA(
        start_state=(False, False),
        states={(is_note, ended) for is_note in (False, True) for ended in (False, True)},
        transition_func=transition,
        accept_func=lambda state: bool(state[0]),
        name="final_real_note",
    )


def _lsdb_style_min_note_acceptor(min_notes, *, pad_symbol):
    def transition(state, symbol):
        note_count, ended = state
        if ended and symbol != pad_symbol:
            return None
        if symbol == pad_symbol:
            return (note_count, True)
        next_count = note_count + (1 if str(symbol).startswith("N") else 0)
        return (min(min_notes, next_count), False)

    return DFA(
        start_state=(0, False),
        states={(count, ended) for count in range(min_notes + 1) for ended in (False, True)},
        transition_func=transition,
        accept_func=lambda state: state[0] >= min_notes,
        name="min_real_note_count",
    )


def _manual_product_acceptor(*acceptors):
    def transition(state, symbol):
        parts = []
        for acceptor, part in zip(acceptors, state):
            next_part = acceptor.next_state(part, symbol)
            if next_part is None:
                return None
            parts.append(next_part)
        return tuple(parts)

    def accepting(state):
        return all(acceptor.is_accepting(part) for acceptor, part in zip(acceptors, state))

    return DFA(
        start_state=tuple(acceptor.start_state for acceptor in acceptors),
        transition_func=transition,
        accept_func=accepting,
        name="manual_product",
    )
