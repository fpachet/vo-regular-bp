#!/usr/bin/env python
"""Compare positional Bach generation across three variable-order engines.

The task is the same positional-only benchmark as ``eval_bach_positional_direct``:
generate a fixed horizon from Bach Prelude in C pitches, with generated positions
0 and n-1 constrained to pitch class C.  This script compares:

* this package's sparse context-graph BP with direct positional masks;
* Continuator's classic variable-order model;
* Continuator's ContextBP model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_bach_positional_direct import (  # noqa: E402
    LazyBackoffContextModel,
    run_direct_positional_bp,
    run_memo_lazy_direct_positional_bp,
)
from scripts.eval_bach_scalability import DATA_PATH, load_bach_pitches, prefix_context  # noqa: E402
from vo_regular_bp import ContextGraph  # noqa: E402


DEFAULT_CONTINUATOR_ROOT = Path("/Users/francoispachet/IdeaProjects/continuator")


@dataclass(frozen=True)
class RunResult:
    method: str
    parse_s: float
    build_s: float
    bp_s: float
    sample_s: float
    total_s: float
    violations: int
    failures: int
    sequence: tuple[int, ...] | None
    order_sequence: tuple[int, ...] = ()
    partition_function: float | None = None
    context_states: int | None = None
    context_edges: int | None = None
    reachable_states: int | None = None
    reachable_edges: int | None = None


def c_pitch_values(pitches: Sequence[int], pitch_class: int) -> set[int]:
    values = {int(pitch) for pitch in pitches if int(pitch) % 12 == pitch_class}
    if not values:
        raise ValueError(f"training sequence has no pitch with pitch class {pitch_class}")
    return values


def violates_pitch_class(sequence: Sequence[int] | None, pitch_class: int) -> bool:
    if not sequence:
        return True
    return int(sequence[0]) % 12 != pitch_class or int(sequence[-1]) % 12 != pitch_class


def run_direct(args: argparse.Namespace, seed: int) -> RunResult:
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    pitches = tuple(int(pitch) for pitch in load_bach_pitches(args.data))
    parse_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    if args.direct_engine == "baseline":
        graph = ContextGraph.from_backoff_sequences(
            [pitches],
            max_order=args.max_order,
            backoff_weight=args.backoff_weight,
        )
        model = None
    else:
        model = LazyBackoffContextModel.from_sequences(
            [pitches],
            max_order=args.max_order,
            backoff_weight=args.backoff_weight,
        )
        graph = None
    build_s = time.perf_counter() - t1

    prefix = tuple(pitches[: args.prefix_length])
    start_context = (
        prefix_context(graph, prefix, args.max_order)
        if graph is not None
        else model.prefix_context(prefix)
    )
    predicate = lambda pitch: int(pitch) % 12 == args.pitch_class

    t2 = time.perf_counter()
    if graph is not None:
        bp = run_direct_positional_bp(
            graph,
            length=args.horizon,
            start_context=start_context,
            constraints={0: predicate, args.horizon - 1: predicate},
        )
        context_states = len(graph.states)
        context_edges = graph.edge_count()
    else:
        bp = run_memo_lazy_direct_positional_bp(
            model,
            length=args.horizon,
            start_context=start_context,
            constraints={0: predicate, args.horizon - 1: predicate},
        )
        context_states = len(model.contexts)
        context_edges = model.materialized_edge_count
    bp_s = time.perf_counter() - t2

    rng = random.Random(seed)
    sequence: tuple[int, ...] | None = None
    violations = 0
    failures = 0
    t3 = time.perf_counter()
    try:
        for _ in range(args.samples):
            sequence = tuple(int(pitch) for pitch in bp.sample(rng))
            violations += int(violates_pitch_class(sequence, args.pitch_class))
    except Exception:
        failures += 1
    sample_s = time.perf_counter() - t3

    return RunResult(
        method="vo_regular_bp_direct",
        parse_s=parse_s,
        build_s=build_s,
        bp_s=bp_s,
        sample_s=sample_s,
        total_s=time.perf_counter() - total_start,
        violations=violations,
        failures=failures,
        sequence=sequence,
        partition_function=bp.partition_function,
        context_states=context_states,
        context_edges=context_edges,
        reachable_states=bp.time_indexed_state_count,
        reachable_edges=bp.edge_count,
    )


def import_continuator(continuator_root: Path) -> tuple[Any, Any, Any, Any]:
    root = continuator_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Continuator root does not exist: {root}")
    sys.path.insert(0, str(root))
    from ctor.classic.variable_order_markov import Variable_order_Markov
    from ctor.constraints import ConstraintProblem
    from ctor.context_bp.model import ContextBPModel
    from ctor.context_bp.order_policy import SingletonAvoidingBackoffPolicy

    return Variable_order_Markov, ContextBPModel, ConstraintProblem, SingletonAvoidingBackoffPolicy


def make_constraint_problem(ConstraintProblem: Any, *, length: int, allowed_values: set[int]) -> Any:
    problem = ConstraintProblem(length=length)
    problem.require_one_of(0, allowed_values)
    problem.require_one_of(length - 1, allowed_values)
    return problem


def run_classic(args: argparse.Namespace, seed: int, imports: tuple[Any, Any, Any]) -> RunResult:
    Variable_order_Markov, _, ConstraintProblem, _ = imports
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    pitches = tuple(int(pitch) for pitch in load_bach_pitches(args.data))
    parse_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = Variable_order_Markov(
        sequence_of_stuff=None,
        vp_lambda=lambda pitch: int(pitch),
        kmax=args.max_order,
        decay_fast_half_life=10,
        decay_slow_half_life=80,
        seed=seed,
        store_realizations=False,
    )
    model.learn_sequence(list(pitches))
    build_s = time.perf_counter() - t1

    prefix = list(pitches[: args.prefix_length])
    constraints = make_constraint_problem(
        ConstraintProblem,
        length=args.horizon,
        allowed_values=c_pitch_values(pitches, args.pitch_class),
    )

    # The classic path solves first-order chain BP inside sampling.
    np.random.seed(seed)
    sequence: tuple[int, ...] | None = None
    violations = 0
    failures = 0
    t2 = time.perf_counter()
    for _ in range(args.samples):
        try:
            generated = model.continue_sequence(
                prefix,
                length=args.horizon,
                constraints=constraints,
                relax_prefix_on_fail=False,
                relax_pos0_on_fail=False,
                raise_on_fail=True,
            )
        except Exception:
            failures += 1
            continue
        sequence = tuple(int(pitch) for pitch in generated)
        violations += int(violates_pitch_class(sequence, args.pitch_class))
    sample_s = time.perf_counter() - t2

    return RunResult(
        method="continuator_classic",
        parse_s=parse_s,
        build_s=build_s,
        bp_s=0.0,
        sample_s=sample_s,
        total_s=time.perf_counter() - total_start,
        violations=violations,
        failures=failures,
        sequence=sequence,
        context_states=len(model.ctx_to_continuations),
        context_edges=sum(len(counter.full) for counter in model.ctx_to_continuations.values()),
    )


def run_context_bp(args: argparse.Namespace, seed: int, imports: tuple[Any, Any, Any]) -> RunResult:
    _, ContextBPModel, ConstraintProblem, SingletonAvoidingBackoffPolicy = imports
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    pitches = tuple(int(pitch) for pitch in load_bach_pitches(args.data))
    parse_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = ContextBPModel(
        kmax=args.max_order,
        viewpoint_fn=lambda pitch: int(pitch),
        seed=seed,
        order_policy=SingletonAvoidingBackoffPolicy(),
    )
    model.learn_sequence(list(pitches))
    build_s = time.perf_counter() - t1

    prefix = list(pitches[: args.prefix_length])
    constraints = make_constraint_problem(
        ConstraintProblem,
        length=args.horizon,
        allowed_values=c_pitch_values(pitches, args.pitch_class),
    )

    sequence: tuple[int, ...] | None = None
    order_sequence: tuple[int, ...] = ()
    violations = 0
    failures = 0
    t2 = time.perf_counter()
    for _ in range(args.samples):
        try:
            result = model.sample_sequence_with_trace(
                length=args.horizon,
                prefix=prefix,
                constraints=constraints,
                raise_on_fail=True,
            )
        except Exception:
            failures += 1
            continue
        if result is None:
            failures += 1
            continue
        generated, trace = result
        sequence = tuple(int(pitch) for pitch in generated)
        order_sequence = tuple(int(step.order) for step in trace)
        violations += int(violates_pitch_class(sequence, args.pitch_class))
    sample_s = time.perf_counter() - t2

    compiled = model.compile_graph(prefix=prefix, order=args.max_order)
    return RunResult(
        method="continuator_context_bp",
        parse_s=parse_s,
        build_s=build_s,
        bp_s=0.0,
        sample_s=sample_s,
        total_s=time.perf_counter() - total_start,
        violations=violations,
        failures=failures,
        sequence=sequence,
        order_sequence=order_sequence,
        context_states=len(compiled.contexts),
        context_edges=sum(len(edges) for edges in compiled.outgoing),
    )


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(method: str, results: list[RunResult]) -> dict[str, Any]:
    first = results[0]
    return {
        "method": method,
        "parse_mean_s": mean([result.parse_s for result in results]),
        "build_mean_s": mean([result.build_s for result in results]),
        "bp_mean_s": mean([result.bp_s for result in results]),
        "sample_mean_s": mean([result.sample_s for result in results]),
        "total_mean_s": mean([result.total_s for result in results]),
        "total_stdev_s": stdev([result.total_s for result in results]),
        "violations": sum(result.violations for result in results),
        "failures": sum(result.failures for result in results),
        "partition_function": first.partition_function,
        "context_states": first.context_states,
        "context_edges": first.context_edges,
        "reachable_states": first.reachable_states,
        "reachable_edges": first.reachable_edges,
        "example_sequence": first.sequence,
        "example_orders": first.order_sequence,
    }


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, tuple):
        return " ".join(str(item) for item in value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--continuator-root", type=Path, default=DEFAULT_CONTINUATOR_ROOT)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument("--pitch-class", type=int, default=0)
    parser.add_argument("--backoff-weight", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direct-engine", choices=("optimized", "baseline"), default="optimized")
    args = parser.parse_args()

    imports = import_continuator(args.continuator_root)
    runners = (run_direct, run_classic, run_context_bp)
    grouped: dict[str, list[RunResult]] = {}
    for runner in runners:
        method_results = []
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            if runner is run_direct:
                method_results.append(runner(args, seed))
            else:
                method_results.append(runner(args, seed, imports))
        grouped[method_results[0].method] = method_results

    pitches = tuple(int(pitch) for pitch in load_bach_pitches(args.data))
    print("Bach positional-only comparison")
    print(
        f"K={args.max_order} n={args.horizon} samples_per_repeat={args.samples} "
        f"repeats={args.repeats} pitch_class={args.pitch_class}"
    )
    print(f"prefix: {tuple(pitches[: args.prefix_length])}")
    print()
    header = (
        "method,parse_mean_s,build_mean_s,bp_mean_s,sample_mean_s,total_mean_s,"
        "total_stdev_s,violations,failures,partition_function,context_states,"
        "context_edges,reachable_states,reachable_edges"
    )
    print(header)
    summaries = [summarize(method, results) for method, results in grouped.items()]
    for row in summaries:
        print(
            ",".join(
                format_optional(row[key])
                for key in (
                    "method",
                    "parse_mean_s",
                    "build_mean_s",
                    "bp_mean_s",
                    "sample_mean_s",
                    "total_mean_s",
                    "total_stdev_s",
                    "violations",
                    "failures",
                    "partition_function",
                    "context_states",
                    "context_edges",
                    "reachable_states",
                    "reachable_edges",
                )
            )
        )

    print()
    print("examples")
    for row in summaries:
        print(f"{row['method']}: {format_optional(row['example_sequence'])}")
        if row["example_orders"]:
            print(f"{row['method']}_orders: {format_optional(row['example_orders'])}")


if __name__ == "__main__":
    main()
