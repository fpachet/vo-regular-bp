"""Measure complete backend lifecycles in isolated, resource-bounded workers.

Example: python scripts/bench_backend_lifecycle.py --case lsdb --repeats 3
Use --source-root to compare a saved checkout with the same benchmark driver.
Memory tracing is deliberately separate from ordinary timing runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import tracemalloc


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--case", choices=("lsdb", "bach", "until"), default="lsdb")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--samples", type=int, default=20, help="0 measures preparation only"
    )
    p.add_argument("--prefixes", type=int, default=0)
    p.add_argument("--tracks", type=int, default=16)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--max-order", type=int, default=3)
    p.add_argument("--memory", action="store_true")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--max-rss-mib", type=float, default=2048)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def rss_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2**20 if sys.platform == "darwin" else 1024)


def emit(**values):
    print(json.dumps(values, sort_keys=True), flush=True)


def worker(args):
    sys.path.insert(0, str(args.source_root.resolve()))
    from vo_regular_bp import (
        ConstraintSet,
        LongestFeasiblePolicy,
        OrderStackModel,
        VirtualAugmentedOrderStackModel,
        prepare_constrained_order_stack_plan,
        prepare_until_order_stack,
    )

    def monitor():
        while True:
            if rss_mib() > args.max_rss_mib:
                emit(stage="limit", reason="rss", peak_rss_mib=rss_mib())
                os._exit(2)
            time.sleep(0.2)

    threading.Thread(target=monitor, daemon=True).start()
    if args.memory:
        tracemalloc.start()
    started = time.perf_counter()
    if args.case == "lsdb":
        from scripts.bench_lsdb_virtual_order_stack import (
            START,
            make_sequences,
            transposition_transforms,
            build_integrated_melody_dfa,
        )

        sequences = make_sequences(tracks=args.tracks, sequence_length=96, seed=1729)
        model = VirtualAugmentedOrderStackModel.from_sequences(
            sequences,
            max_order=args.max_order,
            transforms=transposition_transforms(tuple(range(12))),
            start_symbol=START,
        )
        dfa = build_integrated_melody_dfa(
            length=args.length,
            total_beats=args.length / 2,
            phase_quantum=0.25,
            max_leap=12,
            min_pitch=36,
            max_pitch=96,
            weighted=True,
        )
        constraints = ConstraintSet(regular_acceptors=(dfa,))
        prefix = (START,)
    elif args.case == "bach":
        contents = (
            args.source_root / "data/bach_prelude_c_major_pitches.txt"
        ).read_text()
        sequence = tuple(
            int(token)
            for line in contents.splitlines()
            for token in line.split("#", 1)[0].split()
        )
        sequences = [sequence]
        model = OrderStackModel.from_sequences(sequences, max_order=args.max_order)
        constraints = ConstraintSet(
            positional={args.length - 1: {s for s in sequence if s % 12 == 0}},
            forbidden_substrings={
                sequence[i : i + 5] for i in range(len(sequence) - 4)
            },
        )
        prefix = sequence[: args.max_order]
    else:
        sequences = [("a", "a", "b")]
        model = OrderStackModel.from_sequences(sequences, max_order=1)
        prefix = ("a",)
    emit(stage="build", seconds=time.perf_counter() - started)
    started = time.perf_counter()
    if args.case == "until":
        backend = prepare_until_order_stack(
            model,
            prefix=prefix,
            stop="b",
            max_length=args.length,
            policy=LongestFeasiblePolicy(),
        )
        masses = backend.length_weights
    else:
        plan = prepare_constrained_order_stack_plan(
            model,
            constraints,
            length=args.length,
            policy=LongestFeasiblePolicy(),
        )
        backend = plan.for_prefix(prefix)
        masses = backend.result.start_order_masses()
    emit(stage="prepare", seconds=time.perf_counter() - started, masses=masses)
    if args.samples:
        started = time.perf_counter()
        first = backend.sample(rng=0)
        emit(
            stage="first_sample", seconds=time.perf_counter() - started, sequence=first
        )
        started = time.perf_counter()
        samples = backend.sample_many(args.samples, rng=1)
        emit(
            stage="sample_many",
            seconds=time.perf_counter() - started,
            count=len(samples),
        )
        started = time.perf_counter()
        backend.sample_many(args.samples, rng=1)
        emit(
            stage="warm_sample_many",
            seconds=time.perf_counter() - started,
            count=args.samples,
        )
        if args.case != "until":
            import random

            rng = random.Random(2)
            started = time.perf_counter()
            for _ in range(args.samples):
                backend.sample_with_trace(rng=rng)
            emit(
                stage="trace_many",
                seconds=time.perf_counter() - started,
                count=args.samples,
            )
            rng = random.Random(2)
            started = time.perf_counter()
            for _ in range(args.samples):
                backend.sample_with_trace(rng=rng)
            emit(
                stage="warm_trace_many",
                seconds=time.perf_counter() - started,
                count=args.samples,
            )
    if args.prefixes and args.case != "until":
        started = time.perf_counter()
        for index in range(args.prefixes):
            sequence = sequences[index % len(sequences)]
            plan.for_prefix(sequence[: args.max_order]).sample(rng=index)
        emit(
            stage="prefixes", seconds=time.perf_counter() - started, count=args.prefixes
        )
    gc.collect()
    memory = dict(peak_rss_mib=rss_mib())
    if args.memory:
        current, peak = tracemalloc.get_traced_memory()
        memory.update(retained_mib=current / 2**20, peak_traced_mib=peak / 2**20)
    emit(stage="memory", **memory)


def main():
    args = arguments()
    if args.worker:
        worker(args)
        return
    for repeat in range(args.repeats):
        emit(
            stage="run", repeat=repeat + 1, case=args.case, source=str(args.source_root)
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker",
        ]
        try:
            result = subprocess.run(command, timeout=args.timeout)
            if result.returncode:
                raise SystemExit(result.returncode)
        except subprocess.TimeoutExpired:
            emit(stage="limit", reason="timeout", seconds=args.timeout)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
