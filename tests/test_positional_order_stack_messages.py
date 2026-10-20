"""Check numeric positional messages against independently enumerated paths."""

import math
from collections import Counter

import pytest

from vo_regular_bp import OrderStackModel
from vo_regular_bp.order_stack_bp import FixedOrderContextGraph, StackEdge, _backward_messages


@pytest.mark.parametrize('kind', ['none', 'sets', 'predicate', 'impossible'])
@pytest.mark.parametrize('allow_stop', [False, True])
def test_messages_match_full_path_enumeration(kind, allow_stop):
    model = OrderStackModel({(s,): Counter(a=2, b=1, stop=1) for s in ('s', 'a', 'b', 'stop')},
                            max_order=1, forbidden_symbols=frozenset({'stop'}))
    graph = model.compile_graph(1)
    constraints = {
        'none': {},
        'sets': {0: {'a', 'b'}, 2: {'b'}, 3: {'stop', 'a'}},
        'predicate': {1: lambda s: s != 'b', 3: lambda s: s in {'stop', 'a'}},
        'impossible': {2: set()},
    }[kind]
    allowed = {3: frozenset({'stop'})} if allow_stop else {}
    result = _backward_messages(graph, length=4, constraints=constraints,
                                forbidden_symbols=model.forbidden_symbols,
                                allowed_forbidden_symbols=allowed)

    def enumerate_mass(state, position):
        if position == 4:
            return 1.0
        terms = []
        for edge in graph.outgoing[state]:
            if edge.symbol in model.forbidden_symbols and edge.symbol not in allowed.get(position, ()):
                continue
            mask = constraints.get(position)
            if mask is not None and not (mask(edge.symbol) if callable(mask) else edge.symbol in mask):
                continue
            terms.append(edge.probability * enumerate_mass(edge.dst, position + 1))
        return math.fsum(terms)

    for position in range(5):
        for state in range(len(graph.contexts)):
            assert result[position][state] == pytest.approx(enumerate_mass(state, position))
    assert result.log_values is None


def test_single_product_underflow_keeps_finite_log_mass():
    graph = FixedOrderContextGraph(1)
    graph.contexts = [('s',), ('a',), ('b',)]
    graph.outgoing = [[StackEdge(0, 1, 'a', 5e-324, 1)],
                      [StackEdge(1, 2, 'b', .5, 1)], []]
    result = _backward_messages(graph, length=2, constraints={}, forbidden_symbols=frozenset(),
                                allowed_forbidden_symbols={})
    assert result.log_values is not None
    assert math.isclose(result.log_values[0][0], math.log(5e-324) + math.log(.5))
    assert result.log_values[0][2] == -math.inf
