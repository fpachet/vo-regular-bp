#!/usr/bin/env python
"""Tiny exactness experiments for VO context graphs under regular constraints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (  # noqa: E402
    ContextGraph,
    DFA,
    brute_force_distribution,
    conditional_distribution,
    empirical_distribution,
    positional_acceptor,
    run_bp,
    total_variation,
)
from vo_regular_bp.product_bp import ProductBPResult  # noqa: E402


@dataclass(frozen=True)
class TinyExperiment:
    name: str
    graph: ContextGraph
    acceptor: DFA
    alphabet: tuple[int, ...]
    prefix: tuple[int, ...]
    horizon: int


@dataclass(frozen=True)
class ExperimentResult:
    experiment: TinyExperiment
    bp: ProductBPResult
    brute: dict[tuple[int, ...], float]
    exact: dict[tuple[int, ...], float]
    bp_distribution: dict[tuple[int, ...], float]
    empirical: dict[tuple[int, ...], float]
    samples: list[tuple[int, ...]]

    @property
    def z_brute(self) -> float:
        return sum(self.brute.values())

    @property
    def z_bp(self) -> float:
        return self.bp.partition_function

    @property
    def tv_exact_bp(self) -> float:
        return total_variation(self.exact, self.bp_distribution)

    @property
    def tv_exact_empirical(self) -> float:
        return total_variation(self.exact, self.empirical)

    @property
    def sample_violations(self) -> int:
        return sum(
            1
            for sequence in self.samples
            if not self.experiment.acceptor.accepts(sequence)
        )


def build_paper_integer_experiment() -> TinyExperiment:
    """Paper integer example: choose x0 while accounting for x1 = 4."""

    alphabet = tuple(range(7))
    weighted_sequences = [
        (10, (0, 1, 2, 4)),
        (10, (0, 1, 3, 5)),
        (1, (0, 1, 3, 4)),
        (1000, (6, 2, 5)),
        (1000, (6, 3, 4)),
    ]
    graph = ContextGraph.from_weighted_sequences(weighted_sequences, max_order=2)
    acceptor = positional_acceptor(
        2,
        {1: {4}},
        alphabet=alphabet,
        name="second_symbol_is_4",
    )
    return TinyExperiment(
        name="Experiment 1: positional future constraint x1 = 4",
        graph=graph,
        acceptor=acceptor,
        alphabet=alphabet,
        prefix=(0, 1),
        horizon=2,
    )


def build_forbidden_substring_experiment() -> TinyExperiment:
    """Tiny non-positional regular constraint: generated sequence avoids 22."""

    alphabet = (0, 1, 2, 3)
    weighted_sequences = [
        (10, (0, 1, 2, 2, 0, 1, 3, 0)),
        (5, (0, 1, 2, 3, 0, 1, 0, 3)),
        (4, (0, 1, 0, 2, 1, 3, 2, 0)),
        (3, (0, 1, 3, 2, 2, 1, 0, 2)),
        (2, (2, 0, 1, 2, 0, 3, 1, 2)),
    ]
    graph = ContextGraph.from_weighted_sequences(weighted_sequences, max_order=2)

    transitions = {
        "q0": {symbol: ("q1" if symbol == 2 else "q0") for symbol in alphabet},
        "q1": {symbol: ("dead" if symbol == 2 else "q0") for symbol in alphabet},
        "dead": {symbol: "dead" for symbol in alphabet},
    }
    acceptor = DFA(
        start_state="q0",
        accept_states={"q0", "q1"},
        transitions=transitions,
        states={"q0", "q1", "dead"},
        alphabet=alphabet,
        name="avoid_22",
    )
    return TinyExperiment(
        name="Experiment 2: non-positional forbidden substring 22",
        graph=graph,
        acceptor=acceptor,
        alphabet=alphabet,
        prefix=(0, 1),
        horizon=4,
    )


def analyze_experiment(
    experiment: TinyExperiment,
    *,
    sample_count: int,
    seed: int,
) -> ExperimentResult:
    bp = run_bp(
        experiment.graph,
        experiment.acceptor,
        length=experiment.horizon,
        start_context=experiment.prefix,
    )
    brute = brute_force_distribution(
        experiment.graph,
        experiment.acceptor,
        length=experiment.horizon,
        alphabet=experiment.alphabet,
        start_context=experiment.prefix,
    )
    exact = conditional_distribution(brute)
    bp_distribution = {
        sequence: probability
        for sequence in product(experiment.alphabet, repeat=experiment.horizon)
        if (probability := bp.conditional_probability(sequence)) > 0.0
    }

    rng = random.Random(seed)
    samples = bp.sample_many(sample_count, rng=rng)
    empirical = empirical_distribution(samples)

    return ExperimentResult(
        experiment=experiment,
        bp=bp,
        brute=brute,
        exact=exact,
        bp_distribution=bp_distribution,
        empirical=empirical,
        samples=samples,
    )


def format_sequence(sequence: tuple[int, ...]) -> str:
    return "(" + ",".join(str(symbol) for symbol in sequence) + ")"


def print_result(result: ExperimentResult, *, max_rows: int | None = None) -> None:
    experiment = result.experiment
    print()
    print(experiment.name)
    print("-" * len(experiment.name))
    print(f"prefix: {experiment.prefix}  horizon: {experiment.horizon}")
    print()

    support = sorted(
        set(result.exact) | set(result.bp_distribution) | set(result.empirical),
        key=lambda sequence: (-result.exact.get(sequence, 0.0), sequence),
    )
    if max_rows is not None:
        support = support[:max_rows]

    print("sequence  brute_prob      exact_cond      bp_cond         sample_freq")
    for sequence in support:
        print(
            f"{format_sequence(sequence):<9} "
            f"{result.brute.get(sequence, 0.0):<15.10g} "
            f"{result.exact.get(sequence, 0.0):<15.10g} "
            f"{result.bp_distribution.get(sequence, 0.0):<15.10g} "
            f"{result.empirical.get(sequence, 0.0):<15.10g}"
        )

    print()
    print(f"Z_brute: {result.z_brute:.12g}")
    print(f"Z_BP: {result.z_bp:.12g}")
    print(f"abs_error: {abs(result.z_brute - result.z_bp):.3g}")
    print(f"TV(exact, BP): {result.tv_exact_bp:.6g}")
    print(f"TV(exact, empirical): {result.tv_exact_empirical:.6g}")
    print(f"sample constraint violations: {result.sample_violations}")
    print(f"context states: {len(experiment.graph.states)}")
    print(f"context edges: {experiment.graph.edge_count()}")
    print(f"acceptor states: {experiment.acceptor.state_count() or 'unknown'}")
    print(f"reachable product states: {result.bp.unique_product_state_count}")
    print(f"reachable product states, time-indexed: {result.bp.time_indexed_product_state_count}")
    print(f"reachable product edges: {result.bp.product_edge_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    for offset, experiment in enumerate(
        [
            build_paper_integer_experiment(),
            build_forbidden_substring_experiment(),
        ]
    ):
        result = analyze_experiment(
            experiment,
            sample_count=args.samples,
            seed=args.seed + offset,
        )
        print_result(result, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
