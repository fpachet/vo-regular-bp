from vo_regular_bp import (
    ContextGraph,
    all_of,
    cumulative_meter_acceptor,
    dense_forbidden_substring_acceptor,
    forbidden_substring_acceptor,
    max_order_acceptor,
    meter_acceptor,
    positional_acceptor,
    run_bp,
)


def test_positional_acceptor():
    acceptor = positional_acceptor(3, {0: {"A"}, 2: {"C"}})

    assert acceptor.accepts(("A", "B", "C"))
    assert not acceptor.accepts(("B", "B", "C"))
    assert not acceptor.accepts(("A", "B"))
    assert not acceptor.accepts(("A", "B", "B"))


def test_meter_acceptor():
    acceptor = meter_acceptor([1, 0, 1], {"A": 1, "B": 0, "C": 1})

    assert acceptor.accepts(("A", "B", "C"))
    assert not acceptor.accepts(("A", "C", "B"))


def test_cumulative_meter_acceptor_matches_running_example():
    acceptor = _six_beat_meter_acceptor()

    assert acceptor.accepts((1, 2, 3, 4, 2, 3, 2, 1) + (0,) * 10)
    assert not acceptor.accepts((1, 2, 4, 3, 2, 3, 2, 1) + (0,) * 10)
    assert not acceptor.accepts((1, 2, 3, 4, 3, 4, 2) + (0,) * 11)


def test_cumulative_meter_acceptor_conditions_bp_samples():
    graph = ContextGraph.from_probabilities(
        {
            (): {1: 1.0},
            (0,): {0: 1.0},
            (1,): {2: 0.8, 0: 0.2},
            (2,): {1: 0.4, 3: 0.2, 4: 0.2, 0: 0.2},
            (3,): {2: 0.4, 4: 0.4, 0: 0.2},
            (4,): {2: 0.4, 3: 0.4, 0: 0.2},
        },
        max_order=1,
    )
    acceptor = _six_beat_meter_acceptor()

    bp = run_bp(graph, acceptor, length=18)
    samples = bp.sample_many(100, rng=123)

    assert bp.partition_function > 0.0
    assert all(acceptor.accepts(sample) for sample in samples)


def test_forbidden_substring_acceptor():
    acceptor = forbidden_substring_acceptor([("A", "B"), ("C", "A", "C")])

    assert acceptor.accepts(("A", "C", "B", "C"))
    assert not acceptor.accepts(("A", "B", "C"))
    assert not acceptor.accepts(("C", "A", "C"))


def test_dense_forbidden_substring_acceptor_matches_generic():
    alphabet = (0, 1, 2)
    patterns = ((0, 1, 2), (1, 1), (2, 0))
    generic = forbidden_substring_acceptor(patterns, alphabet=alphabet)
    dense = dense_forbidden_substring_acceptor(patterns, alphabet=alphabet)

    sequences = [()]
    for length in range(1, 6):
        sequences.extend(_tuples(alphabet, length))

    assert dense.state_count() == generic.state_count()
    assert all(dense.accepts(sequence) == generic.accepts(sequence) for sequence in sequences)


def test_max_order_acceptor_forbids_reference_windows():
    acceptor = max_order_acceptor([("A", "B", "C", "D")], max_order=2)

    assert acceptor.accepts(("A", "B", "D"))
    assert not acceptor.accepts(("A", "B", "C"))
    assert not acceptor.accepts(("B", "C", "D"))


def test_all_of_intersects_acceptors():
    positional = positional_acceptor(2, {0: {"A"}})
    meter = meter_acceptor([1, 0], {"A": 1, "B": 0})
    acceptor = all_of(positional, meter)

    assert acceptor.accepts(("A", "B"))
    assert not acceptor.accepts(("B", "A"))


def _tuples(alphabet, length):
    if length == 0:
        return [()]
    return [
        prefix + (symbol,)
        for prefix in _tuples(alphabet, length - 1)
        for symbol in alphabet
    ]


def _six_beat_meter_acceptor():
    length = 18
    costs = {symbol: symbol for symbol in range(5)}

    def predicate(total, symbol, _position):
        symbol_cost = costs[symbol]
        new_total = total + symbol_cost
        if new_total > length:
            return False
        if symbol_cost == 0:
            return True
        if symbol_cost > 6:
            return False
        return total // 6 == (new_total - 1) // 6 or total % 6 == 0

    return cumulative_meter_acceptor(
        length,
        costs,
        predicate,
        alphabet=range(5),
        max_cost=length,
        accept_costs={length},
        end_symbol=0,
    )
