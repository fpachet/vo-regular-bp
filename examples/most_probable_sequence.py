"""Optimize and sample the same constrained context model.

Run from the repository root with:
    python examples/most_probable_sequence.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (  # noqa: E402
    ContextGraph,
    most_probable_sequence,
    positional_acceptor,
    run_bp,
)


def main():
    graph = ContextGraph.from_probabilities({
        (): {"a": 0.6, "b": 0.4},
        ("a",): {"x": 0.5, "y": 0.5},
        ("b",): {"x": 1.0},
    })
    acceptor = positional_acceptor(2, {1: {"x", "y"}})

    optimum = most_probable_sequence(graph, acceptor, length=2)
    print("Optimum:", optimum.sequence, "log weight:", optimum.log_weight)
    # ('b', 'x') has weight 0.4; ('a', 'x') and ('a', 'y') each have weight 0.3.
    assert optimum.feasible and optimum.sequence == ("b", "x")

    bp = run_bp(graph, acceptor, length=2)
    print("Samples:", bp.sample_many(10, rng=7))
    # Sampling explores all three words with probabilities 0.4, 0.3, and 0.3.
    # Greedily taking the largest sampling weight first chooses 'a' (mass 0.6),
    # whose best individual completion has weight only 0.3.


if __name__ == "__main__":
    main()
