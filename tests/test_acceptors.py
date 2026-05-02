from vo_regular_bp import (
    all_of,
    dense_forbidden_substring_acceptor,
    forbidden_substring_acceptor,
    max_order_acceptor,
    meter_acceptor,
    positional_acceptor,
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
