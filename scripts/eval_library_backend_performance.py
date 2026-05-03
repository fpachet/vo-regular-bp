#!/usr/bin/env python
"""Benchmark library-facing order-stack APIs and current hot spots.

Set ``VO_REGULAR_BP_IMPORT_ROOT`` to benchmark an archived checkout while using
this script from the current tree.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import pstats
import random
import statistics
import sys
import time
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOT = Path(os.environ.get("VO_REGULAR_BP_IMPORT_ROOT", str(REPO_ROOT))).resolve()
sys.path.insert(0, str(IMPORT_ROOT))

from scripts.eval_bach_scalability import (  # noqa: E402
    build_policy_stack_optimized_constraint,
    forbidden_windows,
    load_bach_pitches,
)
import vo_regular_bp as vbp  # noqa: E402


@dataclass(frozen=True)
class TimingSummary:
    median_s: float
    min_s: float
    max_s: float
    runs_s: tuple[float, ...]

    @classmethod
    def from_runs(cls, runs: Iterable[float]) -> "TimingSummary":
        values = tuple(runs)
        if not values:
            return cls(0.0, 0.0, 0.0, ())
        return cls(
            median_s=statistics.median(values),
            min_s=min(values),
            max_s=max(values),
            runs_s=values,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="current")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--api-order", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--maxorder-grams", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile-runs", type=int, default=20)
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "label": args.label,
        "import_root": str(IMPORT_ROOT),
        "python": sys.version.replace("\n", " "),
        "vo_regular_bp_file": getattr(vbp, "__file__", None),
        "horizon": args.horizon,
        "maxorder_grams": args.maxorder_grams,
        "samples": args.samples,
        "seed": args.seed,
        "warmups": args.warmups,
        "repeats": args.repeats,
    }

    pitches = load_bach_pitches()
    alphabet = tuple(sorted(set(pitches)))
    prefix = tuple(pitches[:6])
    metadata.update(
        {
            "training_length": len(pitches),
            "alphabet_size": len(alphabet),
            "prefix": prefix,
        }
    )

    raw_rows = run_raw_order_stack_rows(args, pitches, alphabet, prefix)
    write_csv(args.output_dir / f"{args.label}_raw_order_stack.csv", raw_rows)

    api_rows: list[dict[str, Any]] = []
    if not args.skip_api:
        api_rows = run_api_overhead_rows(args, pitches, alphabet, prefix)
        write_csv(args.output_dir / f"{args.label}_api_overhead.csv", api_rows)

    profiles: dict[str, str] = {}
    if not args.skip_profile:
        profiles = write_profiles(args, pitches, alphabet, prefix)

    summary = {
        "metadata": metadata,
        "raw_order_stack": raw_rows,
        "api_overhead": api_rows,
        "profiles": profiles,
    }
    (args.output_dir / f"{args.label}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )


def run_raw_order_stack_rows(
    args: argparse.Namespace,
    pitches: tuple[int, ...],
    alphabet: tuple[int, ...],
    prefix: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bundle, constraint_s = build_policy_stack_optimized_constraint(
        pitches,
        alphabet,
        args.horizon,
        args.maxorder_grams,
        0,
    )

    for order in args.orders:
        model = vbp.OrderStackModel.from_sequences([pitches], max_order=order)
        build_graphs = {
            graph_order: model.compile_graph(graph_order)
            for graph_order in range(1, model.max_order + 1)
        }
        graph_build_edges = sum(graph.edge_count for graph in build_graphs.values())
        graph_build_states = sum(len(graph.contexts) for graph in build_graphs.values())

        def build_bp():
            return vbp.run_order_stack_masked_dfa_bp(
                model,
                bundle.regular_acceptor,
                length=args.horizon,
                prefix=prefix,
                constraints=bundle.positional_constraints,
                policy=vbp.LongestFeasiblePolicy(),
            )

        bp, bp_timing = timed_build(build_bp, warmups=args.warmups, repeats=args.repeats)
        bp.sample_many_with_orders(args.samples, rng=random.Random(args.seed))
        sample_timing = timed_sample(
            lambda count, rng: bp.sample_many_with_orders(count, rng=rng),
            args.samples,
            seed=args.seed,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        rows.append(
            {
                "label": args.label,
                "method": "raw_run_order_stack_masked_dfa_bp",
                "K": order,
                "context_states": graph_build_states,
                "context_edges": graph_build_edges,
                "regular_states": bundle.regular_acceptor.state_count(),
                "reachable_product_states": bp.product_state_count,
                "reachable_product_states_time_indexed": bp.time_indexed_product_state_count,
                "reachable_product_edges": bp.product_edge_count,
                "constraint_build_s": f"{constraint_s:.9f}",
                "bp_median_s": f"{bp_timing.median_s:.9f}",
                "bp_min_s": f"{bp_timing.min_s:.9f}",
                "bp_max_s": f"{bp_timing.max_s:.9f}",
                "sampling_median_s": f"{sample_timing.median_s:.9f}",
                "sampling_per_sequence_median_s": f"{sample_timing.median_s / args.samples:.9f}",
                "sampling_min_s": f"{sample_timing.min_s:.9f}",
                "sampling_max_s": f"{sample_timing.max_s:.9f}",
                "bp_runs_s": json.dumps(bp_timing.runs_s),
                "sampling_runs_s": json.dumps(sample_timing.runs_s),
            }
        )
    return rows


def run_api_overhead_rows(
    args: argparse.Namespace,
    pitches: tuple[int, ...],
    alphabet: tuple[int, ...],
    prefix: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not hasattr(vbp, "ConstraintSet"):
        return []

    order = args.api_order
    final_pitches = frozenset(pitch for pitch in alphabet if pitch % 12 == 0)
    constraints = vbp.ConstraintSet(
        positional={args.horizon - 1: final_pitches},
        forbidden_substrings=forbidden_windows(pitches, args.maxorder_grams),
    )
    methods: list[tuple[str, Callable[[], Any], Callable[[Any, int, random.Random], Any]]] = []

    bundle, _constraint_s = build_policy_stack_optimized_constraint(
        pitches,
        alphabet,
        args.horizon,
        args.maxorder_grams,
        0,
    )
    model = vbp.OrderStackModel.from_sequences([pitches], max_order=order)
    for graph_order in range(1, model.max_order + 1):
        model.compile_graph(graph_order)

    def raw_build():
        return vbp.run_order_stack_masked_dfa_bp(
            model,
            bundle.regular_acceptor,
            length=args.horizon,
            prefix=prefix,
            constraints=bundle.positional_constraints,
            policy=vbp.LongestFeasiblePolicy(),
        )

    methods.append(("raw_prebuilt_model_and_constraint", raw_build, sample_result))

    if hasattr(vbp, "prepare_constrained_order_stack"):
        model_for_prepare = vbp.OrderStackModel.from_sequences([pitches], max_order=order)

        def prepare_model_build():
            return vbp.prepare_constrained_order_stack(
                model_for_prepare,
                constraints,
                length=args.horizon,
                prefix=prefix,
                policy=vbp.LongestFeasiblePolicy(),
            )

        methods.append(("prepare_constrained_order_stack", prepare_model_build, sample_backend))

    if hasattr(vbp, "prepare_constrained_order_stack_from_events") and hasattr(vbp, "EventCodec"):
        codec = vbp.EventCodec.identity()

        def prepare_events_build():
            return vbp.prepare_constrained_order_stack_from_events(
                [pitches],
                constraints,
                codec=codec,
                max_order=order,
                length=args.horizon,
                prefix=prefix,
                policy=vbp.LongestFeasiblePolicy(),
            )

        methods.append(("prepare_constrained_order_stack_from_events", prepare_events_build, sample_event_backend))

    if hasattr(vbp, "prepare_continuation_backend"):
        def continuator_build():
            return vbp.prepare_continuation_backend(
                [pitches],
                prefix=prefix,
                horizon=args.horizon,
                max_order=order,
                constraints=constraints,
                policy=vbp.LongestFeasiblePolicy(),
            )

        methods.append(("prepare_continuation_backend", continuator_build, sample_event_backend))

    rows = []
    for method, builder, sampler in methods:
        result, prepare_timing = timed_build(builder, warmups=args.warmups, repeats=args.repeats)
        sampler(result, args.samples, random.Random(args.seed))
        sample_timing = timed_sample(
            lambda count, rng: sampler(result, count, rng),
            args.samples,
            seed=args.seed,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        diagnostics = diagnostics_for(result)
        rows.append(
            {
                "label": args.label,
                "method": method,
                "K": order,
                "prepare_median_s": f"{prepare_timing.median_s:.9f}",
                "prepare_min_s": f"{prepare_timing.min_s:.9f}",
                "prepare_max_s": f"{prepare_timing.max_s:.9f}",
                "sampling_median_s": f"{sample_timing.median_s:.9f}",
                "sampling_per_sequence_median_s": f"{sample_timing.median_s / args.samples:.9f}",
                "sampling_min_s": f"{sample_timing.min_s:.9f}",
                "sampling_max_s": f"{sample_timing.max_s:.9f}",
                "diagnostics": json.dumps(diagnostics, default=str),
                "prepare_runs_s": json.dumps(prepare_timing.runs_s),
                "sampling_runs_s": json.dumps(sample_timing.runs_s),
            }
        )
    return rows


def write_profiles(
    args: argparse.Namespace,
    pitches: tuple[int, ...],
    alphabet: tuple[int, ...],
    prefix: tuple[int, ...],
) -> dict[str, str]:
    order = args.api_order
    model = vbp.OrderStackModel.from_sequences([pitches], max_order=order)
    for graph_order in range(1, model.max_order + 1):
        model.compile_graph(graph_order)
    bundle, _ = build_policy_stack_optimized_constraint(
        pitches,
        alphabet,
        args.horizon,
        args.maxorder_grams,
        0,
    )

    def build_bp():
        return vbp.run_order_stack_masked_dfa_bp(
            model,
            bundle.regular_acceptor,
            length=args.horizon,
            prefix=prefix,
            constraints=bundle.positional_constraints,
            policy=vbp.LongestFeasiblePolicy(),
        )

    def profile_bp():
        for _ in range(args.profile_runs):
            build_bp()

    bp = build_bp()
    bp.sample_many_with_orders(args.samples, rng=random.Random(args.seed))

    def profile_sampling():
        for repeat in range(args.profile_runs):
            bp.sample_many_with_orders(args.samples, rng=random.Random(args.seed + repeat))

    profile_paths = {
        "bp": args.output_dir / f"{args.label}_profile_bp.txt",
        "sampling": args.output_dir / f"{args.label}_profile_sampling.txt",
    }
    profile_paths["bp"].write_text(profile_text(profile_bp), encoding="utf-8")
    profile_paths["sampling"].write_text(profile_text(profile_sampling), encoding="utf-8")
    return {name: str(path) for name, path in profile_paths.items()}


def timed_build(
    builder: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> tuple[Any, TimingSummary]:
    result = None
    for _ in range(warmups):
        result = builder()
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = builder()
        runs.append(time.perf_counter() - start)
    return result, TimingSummary.from_runs(runs)


def timed_sample(
    sampler: Callable[[int, random.Random], Any],
    count: int,
    *,
    seed: int,
    warmups: int,
    repeats: int,
) -> TimingSummary:
    for warmup in range(warmups):
        sampler(count, random.Random(seed + warmup))
    runs = []
    for repeat in range(repeats):
        start = time.perf_counter()
        sampler(count, random.Random(seed + warmups + repeat))
        runs.append(time.perf_counter() - start)
    return TimingSummary.from_runs(runs)


def sample_result(result: Any, count: int, rng: random.Random) -> Any:
    return result.sample_many_with_orders(count, rng=rng)


def sample_backend(result: Any, count: int, rng: random.Random) -> Any:
    return result.sample_many_with_orders(count, rng=rng)


def sample_event_backend(result: Any, count: int, rng: random.Random) -> Any:
    return result.sample_many_events_with_orders(count, rng=rng)


def diagnostics_for(result: Any) -> dict[str, Any]:
    if hasattr(result, "diagnostics"):
        diagnostics = result.diagnostics
        if hasattr(diagnostics, "as_dict"):
            return dict(diagnostics.as_dict())
    return {
        "backend": "raw",
        "context_states": getattr(result, "context_state_count", None),
        "context_edges": getattr(result, "context_edge_count", None),
        "product_states": getattr(result, "product_state_count", None),
        "product_edges": getattr(result, "product_edge_count", None),
        "success_mass": getattr(result, "success_mass", None),
    }


def profile_text(fn: Callable[[], None]) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
    return stream.getvalue()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
