#!/usr/bin/env python
"""Report reachable product sizes and BP/sampling timings."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (
    ContextGraph,
    all_of,
    max_order_acceptor,
    positional_acceptor,
    run_bp,
)


def load_sequences(path: Path | None) -> list[tuple[str, ...]]:
    if path is None:
        motifs = [
            "C D E G A G E D".split(),
            "C E G B A G E C".split(),
            "D F A C B A F D".split(),
            "G A B D E D B A".split(),
        ]
        rng = random.Random(0)
        sequences = []
        for _ in range(300):
            motif = rng.choice(motifs)
            rotation = rng.randrange(len(motif))
            sequences.append(tuple(motif[rotation:] + motif[:rotation]))
        return sequences

    sequences = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = tuple(line.split())
        if tokens:
            sequences.append(tokens)
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forbid-max-order", action="store_true")
    args = parser.parse_args()

    sequences = load_sequences(args.corpus)
    graph = ContextGraph.from_sequences(sequences, max_order=args.max_order)
    alphabet = graph.alphabet

    forced_end = next(iter(sorted(alphabet, key=str)))
    acceptor = positional_acceptor(args.length, {args.length - 1: {forced_end}}, alphabet=alphabet)
    if args.forbid_max_order:
        acceptor = all_of(
            acceptor,
            max_order_acceptor(sequences, args.max_order, alphabet=alphabet),
            name="forced_end_and_max_order",
        )

    rng = random.Random(args.seed)
    tracemalloc.start()
    t0 = time.perf_counter()
    bp = run_bp(graph, acceptor, length=args.length)
    bp_time = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t1 = time.perf_counter()
    for _ in range(args.samples):
        if bp.partition_function > 0.0:
            bp.sample(rng=rng)
    sampling_time = time.perf_counter() - t1

    print(f"training sequences: {len(sequences)}")
    print(f"|T| context states: {len(graph.states)}")
    print(f"|E_T| context edges: {graph.edge_count()}")
    print(f"|Q| acceptor states: {acceptor.state_count() or 'unknown'}")
    print(f"reachable acceptor states: {bp.reachable_acceptor_state_count}")
    print(f"reachable product states, unique: {bp.unique_product_state_count}")
    print(f"reachable product states, time-indexed: {bp.time_indexed_product_state_count}")
    print(f"reachable product edges: {bp.product_edge_count}")
    print(f"Z_BP: {bp.partition_function:.12g}")
    print(f"BP time: {bp_time:.6f}s")
    print(f"sampling time ({args.samples}): {sampling_time:.6f}s")
    print(f"peak traced memory: {peak / (1024 * 1024):.3f} MiB")


if __name__ == "__main__":
    main()
