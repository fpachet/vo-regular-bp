#!/usr/bin/env python
"""Tiny exactness check: BP partition, brute force partition, sample TV."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (
    ContextGraph,
    all_of,
    brute_force_distribution,
    conditional_distribution,
    empirical_distribution,
    meter_acceptor,
    positional_acceptor,
    run_bp,
    total_variation,
)


def build_example():
    graph = ContextGraph.from_probabilities(
        {
            (): {"A": 0.55, "B": 0.45},
            ("A",): {"A": 0.20, "B": 0.50, "C": 0.30},
            ("B",): {"A": 0.60, "B": 0.10, "C": 0.30},
            ("C",): {"A": 0.40, "B": 0.60},
        },
        max_order=1,
    )
    length = 4
    alphabet = graph.alphabet
    positional = positional_acceptor(length, {3: {"A"}}, alphabet=alphabet)
    meter = meter_acceptor(
        [None, 0, None, 1],
        {"A": 1, "B": 0, "C": 0},
        alphabet=alphabet,
    )
    return graph, all_of(positional, meter), length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    graph, acceptor, length = build_example()
    bp = run_bp(graph, acceptor, length=length)
    brute = brute_force_distribution(graph, acceptor, length=length)
    exact = conditional_distribution(brute)
    rng = random.Random(args.seed)
    samples = bp.sample_many(args.samples, rng=rng)
    empirical = empirical_distribution(samples)

    print(f"Z_brute: {sum(brute.values()):.12g}")
    print(f"Z_BP:    {bp.partition_function:.12g}")
    print(f"abs_err: {abs(sum(brute.values()) - bp.partition_function):.3g}")
    print(f"support: {len(exact)}")
    print(f"samples: {args.samples}")
    print(f"TV(empirical, exact): {total_variation(empirical, exact):.6g}")


if __name__ == "__main__":
    main()
