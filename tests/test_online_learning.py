"""Online snapshots must agree with independent full rebuilds and path sums."""
import gc
from fractions import Fraction
from itertools import product
import math
import random
import weakref

import pytest

from vo_regular_bp import (
    ConstraintSet, LongestFeasiblePolicy, OrderStackModel,
    SingletonAvoidingBackoffPolicy, VirtualAugmentedOrderStackModel,
    integer_shift_transforms, materialize_transformed_sequences,
    prepare_constrained_order_stack, prepare_until_end_order_stack,
)


def graph_rows(graph):
    return {
        context: tuple((edge.symbol, edge.probability, edge.order, graph.contexts[edge.dst])
                       for edge in graph.outgoing[state])
        for state, context in enumerate(graph.contexts)
    }


@pytest.mark.parametrize("markers", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_repeated_and_deferred_updates_match_rebuild(markers, reuse):
    rng = random.Random(712)
    options = dict(max_order=4)
    if markers:
        options.update(start_symbol="START", end_symbol="END")
    memory = [(0, 1, 0, 2, 1)]
    model = OrderStackModel.from_sequences(memory, **options)
    for cycle in range(16):
        # Leave consecutive updates pending to exercise coalescing.
        if cycle % 4 == 0:
            for order in range(1, 5):
                model.compile_graph(order)
        phrase = (memory[0] if cycle % 3 == 0 else
                  tuple(rng.randrange(5) for _ in range(rng.randrange(10))))
        memory.append(phrase)
        old_counts = {c: row.copy() for c, row in model.counts.items()}
        previous = model
        model = model.with_sequences(iter([phrase]), reuse_graphs=reuse)
        fresh = OrderStackModel.from_sequences(memory, **options)
        assert previous.counts == old_counts
        assert model.counts == fresh.counts
        assert model.alphabet == fresh.alphabet
        assert model.forbidden_symbols == fresh.forbidden_symbols
        if cycle % 4 in (2, 3):
            for order in range(1, 5):
                assert graph_rows(model.compile_graph(order)) == graph_rows(fresh.compile_graph(order))


def test_new_suffix_replaces_terminal_and_backoff_closure_rows():
    model = OrderStackModel.from_sequences([(0, 1, 2), (1, 3, 4)], max_order=3)
    graphs = [model.compile_graph(k) for k in range(1, 4)]
    before = [graph_rows(g) for g in graphs]
    # (1, 2) is initially a terminal with no outgoing edges. It gets counts;
    # its longer graph states can now back off through this new suffix.
    updated = model.with_sequences([(1, 2, 5), (4, 6)])
    fresh = OrderStackModel.from_sequences([(0, 1, 2), (1, 3, 4), (1, 2, 5), (4, 6)], max_order=3)
    for order, old in enumerate(graphs, 1):
        graph = updated.compile_graph(order)
        assert graph_rows(graph) == graph_rows(fresh.compile_graph(order))
        assert graph_rows(old) == before[order - 1]


def test_support_shrink_falls_back_and_removes_orphan_prefix():
    memory = [(0, 1, 2), (2, 3, 4), (2, 5, 6)]
    model = OrderStackModel.from_sequences(memory, max_order=3)
    previous = model.compile_graph(3)
    assert previous.state_id((1, 2, 5)) is not None
    updated = model.with_sequences([(1, 2, 3)])
    fresh = OrderStackModel.from_sequences(memory + [(1, 2, 3)], max_order=3)
    graph = updated.compile_graph(3)
    assert graph_rows(graph) == graph_rows(fresh.compile_graph(3))
    assert graph.state_id((1, 2, 5)) is None


def test_snapshots_share_only_unchanged_rows_and_release_ancestors():
    model = OrderStackModel.from_sequences([(0, 0, 1, 0), (7, 7, 8, 7)], max_order=2)
    old_graph = model.compile_graph(2)
    old_row = old_graph.outgoing[old_graph.state_id((7, 7))]
    updated = model.with_sequences([(0, 0, 2)])
    graph = updated.compile_graph(2)
    assert graph.outgoing[graph.state_id((7, 7))] is old_row
    assert updated.counts[(7, 7)] is model.counts[(7, 7)]
    assert updated.counts[(0, 0)] is not model.counts[(0, 0)]
    assert graph.outgoing[graph.state_id((0, 0))] is not old_graph.outgoing[old_graph.state_id((0, 0))]
    model_ref, graph_ref = weakref.ref(model), weakref.ref(old_graph)
    del model, old_graph
    gc.collect()
    assert model_ref() is None and graph_ref() is None
    assert not updated._graph_seeds


def test_many_unprepared_updates_do_not_retain_model_chain():
    model = OrderStackModel.from_sequences([(0, 1, 0)], max_order=2)
    model.compile_graph(2)
    refs = []
    for _ in range(100):
        refs.append(weakref.ref(model))
        model = model.with_sequences([(0, 1, 0)])
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert len(model._graph_seeds) == 1
    assert len(model._graph_seeds[2][1]) <= len(model.counts)
    assert model.counts[(0,)][1] == 101
    model.compile_graph(2)
    assert not model._graph_seeds


def test_repeated_preparation_does_not_retain_old_graphs_or_inference():
    model = OrderStackModel.from_sequences([(0, 1, 0, 2, 0)], max_order=2)
    refs = []
    for cycle in range(40):
        backend = prepare_constrained_order_stack(model, length=3, prefix=(0,))
        backend.sample(rng=cycle)
        refs.extend([weakref.ref(model), weakref.ref(backend.result),
                     *(weakref.ref(graph) for graph in backend.result.graphs.values())])
        model = model.with_sequences([(0, 1, 0, 2, 0)])
        del backend
    for order in (1, 2):
        model.compile_graph(order)
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert len(model.counts) == 6
    assert not model._graph_seeds


@pytest.mark.parametrize("regular", [False, True])
@pytest.mark.parametrize("policy", [LongestFeasiblePolicy(), SingletonAvoidingBackoffPolicy()])
def test_learning_keeps_old_inference_valid_and_new_inference_fresh(regular, policy):
    memory = [(0, 1, 0, 2, 1, 0)]
    model = OrderStackModel.from_sequences(memory, max_order=3, start_symbol="START", end_symbol="END")
    constraints = ConstraintSet(positional={4: {0}},
                                forbidden_substrings=((2, 2),) if regular else ())
    def prepare(m):
        return prepare_constrained_order_stack(m, constraints, length=5, prefix=(0, 1), policy=policy)
    old = prepare(model)
    old_samples = [old.sample_with_trace(rng=i) for i in range(20)]
    for phrase in [memory[0], (1, 3, 0, 1, 3, 0)]:
        memory.append(phrase)
        model = model.with_sequences([phrase])
        fresh = OrderStackModel.from_sequences(memory, max_order=3, start_symbol="START", end_symbol="END")
        left, right = prepare(model), prepare(fresh)
        assert left.result.start_order_masses() == right.result.start_order_masses()
        assert left.result.start_log_order_masses() == right.result.start_log_order_masses()
        assert left.result.backwards is not old.result.backwards
        for seed in range(20):
            a, b = left.sample_with_trace(rng=seed), right.sample_with_trace(rng=seed)
            assert a == b
            assert a[0][-1] == 0 and "START" not in a[0] and "END" not in a[0]
            if regular:
                assert all(pair != (2, 2) for pair in zip(a[0], a[0][1:]))
    assert [old.sample_with_trace(rng=i) for i in range(20)] == old_samples


def reference_probability(model, sequence, prefix, masks, banned):
    """Exact rational policy probability, independent of graphs/BP tables."""
    length = len(sequence)

    def row(context, order):
        for k in range(min(order, len(context)), 0, -1):
            counts = model.counts.get(context[-k:])
            if counts:
                total = sum(counts.values())
                return [(s, Fraction(c, total)) for s, c in counts.items()]
        return []

    def children(history, order, position):
        return [(s, p) for s, p in row(history, order)
                if s not in model.forbidden_symbols and s in masks.get(position, model.alphabet)
                and (not history or (history[-1], s) not in banned)]

    def beta(history, order, position):
        if position == length:
            return Fraction(1)
        return sum((p * beta((history + (s,))[-order:], order, position + 1)
                    for s, p in children(history, order, position)), Fraction(0))

    probability = Fraction(1)
    history = prefix
    for position, symbol in enumerate(sequence):
        for order in range(min(model.max_order, len(history)), 0, -1):
            weighted = [(s, p * beta((history + (s,))[-order:], order, position + 1))
                        for s, p in children(history, order, position)]
            total = sum((w for _, w in weighted), Fraction(0))
            if total:
                probability *= sum((w for s, w in weighted if s == symbol), Fraction(0)) / total
                break
        else:
            return Fraction(0)
        history += (symbol,)
    return probability


def result_probability(result, sequence, regular):
    history = list(result.prefix)
    state = result.start_acceptor_state if regular else None
    probability = 1.0
    for position, symbol in enumerate(sequence):
        candidates = (result._candidate_sets(position, history, state) if regular
                      else result._candidate_sets(position, history))
        if not candidates:
            return 0.0
        chosen = candidates[0]
        weight = sum(w for e, w in zip(chosen.edges, chosen.weights) if e.symbol == symbol)
        probability *= weight / sum(chosen.weights)
        if weight == 0:
            return 0.0
        if regular:
            state = result.backwards[chosen.order].next_acceptor_state(state, symbol)
        history.append(symbol)
    return probability


@pytest.mark.parametrize("regular", [False, True])
def test_exhaustive_probabilities_after_duplicate_and_new_symbol(regular):
    memory = [(0, 0, 1, 0, 1, 1, 0)]
    model = OrderStackModel.from_sequences(memory, max_order=2)
    for k in (1, 2):
        model.compile_graph(k)
    for phrase in [memory[0], (0, 2, 0, 2, 1, 2, 0)]:
        model = model.with_sequences([phrase])
        memory.append(phrase)
        fresh = OrderStackModel.from_sequences(memory, max_order=2)
        masks, banned = {2: {0, 1}}, {(1, 1)} if regular else set()
        constraints = ConstraintSet(positional=masks, forbidden_substrings=banned)
        results = [prepare_constrained_order_stack(m, constraints, length=3, prefix=(0,),
                                                  policy=LongestFeasiblePolicy()).result
                   for m in (model, fresh)]
        total = Fraction(0)
        for sequence in product(sorted(model.alphabet), repeat=3):
            expected = reference_probability(model, sequence, (0,), masks, banned)
            total += expected
            for result in results:
                actual = result_probability(result, sequence, regular)
                assert actual == pytest.approx(float(expected), abs=1e-14)
                assert (actual > 0) == (expected > 0)
        assert total == 1


def test_first_hit_end_after_learning_matches_fresh_and_preserves_markers():
    memory = [(0, 1, 0), (0, 2, 0)]
    model = OrderStackModel.from_sequences(memory, max_order=3, start_symbol="S", end_symbol="E")
    prepare_until_end_order_stack(model, prefix=(0,), end_symbol="E", max_length=5)
    for phrase in [memory[0], (0, 3, 1)]:
        model = model.with_sequences([phrase])
        memory.append(phrase)
        fresh = OrderStackModel.from_sequences(memory, max_order=3, start_symbol="S", end_symbol="E")
        left, right = [prepare_until_end_order_stack(m, prefix=(0,), end_symbol="E", max_length=5)
                       for m in (model, fresh)]
        assert left.feasible_lengths == right.feasible_lengths
        assert left.length_weights == right.length_weights
        for seed in range(30):
            sample = left.sample_with_orders(rng=seed)
            assert sample == right.sample_with_orders(rng=seed)
            assert sample.sequence[-1] == "E" and "E" not in sample.sequence[:-1]
            assert "S" not in sample.sequence


@pytest.mark.parametrize("regular", [False, True])
def test_virtual_updates_match_fresh_and_explicit_augmentation(regular):
    memory = [(60, 61, 60, 62, 60)]
    transforms = integer_shift_transforms((0, 1, 2), fixed_symbols=("S", "E"))
    options = dict(max_order=2, start_symbol="S", end_symbol="E")
    model = VirtualAugmentedOrderStackModel.from_sequences(memory, transforms=transforms, **options)
    constraints = ConstraintSet(positional={2: {60, 61, 62}},
                                forbidden_substrings=((61, 61),) if regular else ())
    def prepare(m):
        return prepare_constrained_order_stack(m, constraints, length=3, prefix=(60,), policy=LongestFeasiblePolicy())
    old = prepare(model)
    old_samples = old.sample_many(10, rng=42)
    for phrase in [memory[0], (60, 65, 60, 62)]:
        memory.append(phrase)
        model = model.with_sequences([phrase])
        fresh = VirtualAugmentedOrderStackModel.from_sequences(memory, transforms=transforms, **options)
        explicit = OrderStackModel.from_sequences(materialize_transformed_sequences(memory, transforms), **options)
        backends = [prepare(m) for m in (model, fresh, explicit)]
        for backend in backends[1:]:
            assert backend.result.start_order_masses() == backends[0].result.start_order_masses()
            for seed in range(20):
                assert backend.sample_with_orders(rng=seed) == backends[0].sample_with_orders(rng=seed)
    assert old.sample_many(10, rng=42) == old_samples


def test_incremental_update_keeps_log_underflow_fallback():
    # Repeating a very unlikely accepted emission makes the total mass underflow.
    from collections import Counter
    model = OrderStackModel({(0,): Counter({0: 1, 1: 10**30}),
                             (1,): Counter({0: 1, 1: 1})}, max_order=1)
    model.compile_graph(1)
    updated = model.with_sequences([(0, 0)])
    result = prepare_constrained_order_stack(updated, ConstraintSet(positional={i: {0} for i in range(20)}),
                                             length=20, prefix=(0,)).result
    assert result.start_order_masses() == ((1, 0.0),)
    assert math.isfinite(result.start_log_order_masses()[0][1])
    assert result.sample(rng=0) == (0,) * 20


def test_minimized_views_and_weighted_regular_messages_are_rebuilt():
    from vo_regular_bp import DFA
    memory = [(0, 1, 0, 2, 0)]
    model = OrderStackModel.from_sequences(memory, max_order=2)
    for order in (1, 2):
        model.compile_minimized_graph(order)
    updated = model.with_sequences([(0, 3, 0, 1, 0)])
    fresh = OrderStackModel.from_sequences(memory + [(0, 3, 0, 1, 0)], max_order=2)
    assert not updated._minimized_graph_cache
    acceptor = DFA(start_state=0, transition_func=lambda q, s: q,
                   accept_func=lambda q: True,
                   transition_weight_func=lambda q, s: 0.25 if s == 3 else 2.0)
    constraints = ConstraintSet(regular_acceptors=(acceptor,))
    backends = [prepare_constrained_order_stack(m, constraints, length=3, prefix=(0,),
                                                minimize_source_graphs=True)
                for m in (updated, fresh)]
    assert backends[0].result.start_order_masses() == backends[1].result.start_order_masses()
    for seed in range(20):
        assert backends[0].sample_with_orders(rng=seed) == backends[1].sample_with_orders(rng=seed)
