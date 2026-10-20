"""Reproducible learn -> graph -> backward -> sample lifecycle benchmark.

Run from the repository root. Workers isolate cases, methods, repeats and memory
tracing. JSONL includes raw cycles; --memory is a separate allocation-traced run.
No MIDI parser, external Continuator, or runtime dependency is required.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import json
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
# corpus, notes, order, horizon, constraint kind
CASES = {
    "tiny": ("synthetic", 64, 1, 16, "fixed"),
    "bach549": ("bach", 549, 4, 64, "fixed"),
    "bach-o2": ("bach", 549, 2, 16, "fixed"),
    "bach-o6": ("bach", 549, 6, 128, "fixed"),
    "growing4k": ("synthetic", 4096, 4, 64, "fixed"),
    "growing16k": ("synthetic", 16384, 4, 64, "fixed"),
    "novel-phrases": ("synthetic", 4096, 4, 64, "novel"),
    "regular": ("bach", 549, 4, 64, "regular"),
    "until-end": ("bach", 549, 4, 64, "until"),
    "virtual": ("bach", 549, 4, 32, "virtual"),
    "virtual-regular": ("bach", 549, 4, 16, "virtual-regular"),
}


def worker(args):
    sys.path.insert(0, str(args.source_root.resolve()))
    from vo_regular_bp import (
        ConstraintSet, LongestFeasiblePolicy, OrderStackModel,
        VirtualAugmentedOrderStackModel, integer_shift_transforms,
        prepare_constrained_order_stack, prepare_until_end_order_stack,
    )
    source, notes, order, length, kind = CASES[args.case]
    if source == "bach":
        sequence = tuple(int(t) for line in (ROOT / "data/bach_prelude_c_major_pitches.txt")
                         .read_text().splitlines() for t in line.split("#")[0].split())[:notes]
    else:
        rng = random.Random(1729)
        sequence = tuple(60 + rng.randrange(12) for _ in range(notes))
    # Eight transformations exercise overlapping absolute contexts.
    virtual = kind.startswith("virtual")
    transforms = integer_shift_transforms(range(8), fixed_symbols=("END",))
    model_type = VirtualAugmentedOrderStackModel if virtual else OrderStackModel
    options = dict(max_order=order, end_symbol="END") if kind == "until" else dict(max_order=order)
    if virtual:
        options["transforms"] = transforms
    memory = [sequence]
    prefix = sequence[:order]
    policy = LongestFeasiblePolicy()
    final = {s for s in sequence if s % 12 == 0}
    constraints = ConstraintSet(
        positional={length - 1: final},
        forbidden_substrings=((60, 60, 60),) if "regular" in kind else (),
    )

    def graphs(model):
        # Regular virtual preparation uses lazy source graphs. Their expansion
        # occurs in inference/first sample, and is not mislabeled graph setup.
        if virtual and "regular" in kind:
            model.compile_graphs_for_plan(length=length)
        else:
            for k in range(1, order + 1):
                model.compile_graph(k)

    def prepare(model):
        if kind == "until":
            return prepare_until_end_order_stack(
                model, prefix=prefix, end_symbol="END", max_length=length, policy=policy,
            )
        return prepare_constrained_order_stack(
            model, constraints, length=length, prefix=prefix, policy=policy,
        )

    if args.memory:
        tracemalloc.start()
    model = backend = None
    rows = []
    for cycle in range(args.updates + 1):
        # Drop obsolete inference before learning. The model update itself must
        # retain only reusable source snapshots, never the previous BP tables.
        if args.memory:
            gc.collect()
        started = time.perf_counter()
        backend = None
        invalidated = time.perf_counter()
        if cycle:
            # Alternate duplicates, familiar phrases and genuinely new symbols.
            offset = ((cycle - 1) * 17) % (len(sequence) - 16 + 1)
            phrase = sequence[offset:offset + 16]
            if kind == "novel":
                phrase_rng = random.Random(9000 + cycle)
                phrase = tuple(60 + phrase_rng.randrange(12) for _ in range(16))
            if cycle % 4 == 0:
                phrase = phrase[:-1] + (84 + cycle // 4,)
            memory.append(phrase)
        stored = time.perf_counter()
        if not cycle or args.method == "rebuild":
            model = model_type.from_sequences(memory, **options)
        elif virtual:
            model = model.with_sequences([phrase])
        else:
            model = model.with_sequences([phrase], reuse_graphs=args.method == "incremental")
        learned = time.perf_counter()
        graphs(model)
        graphed = time.perf_counter()
        backend = prepare(model)
        prepared = time.perf_counter()
        backend.sample(rng=0)
        sampled = time.perf_counter()
        rng = random.Random(1)
        for _ in range(args.samples):
            backend.sample(rng=rng)
        warmed = time.perf_counter()
        row = dict(
            cycle=cycle, phase="post_learning" if cycle else "cold",
            invalidate_ms=(invalidated - started) * 1000,
            store_ms=(stored - invalidated) * 1000,
            counts_ms=(learned - stored) * 1000,
            graph_ms=(graphed - learned) * 1000,
            backward_prepare_ms=(prepared - graphed) * 1000,
            first_sample_ms=(sampled - prepared) * 1000,
            cached_sample_ms=(warmed - sampled) * 1000 / args.samples,
            learn_to_first_ms=(sampled - started) * 1000,
        )
        if args.memory:
            gc.collect()
            retained, peak = tracemalloc.get_traced_memory()
            row.update(retained_mib=retained / 2**20, peak_mib=peak / 2**20)
        rows.append(row)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        case=args.case, method=args.method, memory=args.memory,
        repeat=args.repeat, python=sys.version, platform=platform.platform(),
        source_root=str(args.source_root), notes=notes, order=order, length=length,
        updates=args.updates, samples=args.samples, cycles=rows,
        peak_rss_mib=rss / (2**20 if sys.platform == "darwin" else 1024),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--method", choices=["all", "rebuild", "counts", "incremental"], default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--profile", type=Path, help="write cProfile data (one worker)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or args.updates < 0 or args.repeats < 1:
        parser.error("samples/repeats must be positive; updates nonnegative")
    if args.worker:
        profiler = cProfile.Profile() if args.profile else None
        if profiler:
            profiler.enable()
        result = worker(args)
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(args.profile))
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    cases = CASES if args.case == "all" else [args.case]
    methods = ["rebuild", "counts", "incremental"] if args.method == "all" else [args.method]
    for case in cases:
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic thermal/load effects.
            for method in methods[::(-1 if repeat % 2 else 1)]:
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--case", case, "--method", method, "--repeat", str(repeat),
                           "--updates", str(args.updates), "--samples", str(args.samples),
                           "--source-root", str(args.source_root)]
                if args.memory:
                    command.append("--memory")
                if args.profile:
                    command.extend(["--profile", str(args.profile)])
                subprocess.run(command, check=True, timeout=180)


if __name__ == "__main__":
    main()
