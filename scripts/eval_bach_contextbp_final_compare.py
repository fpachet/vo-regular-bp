#!/usr/bin/env python
"""Benchmark Continuator ContextBP against vo-regular-bp order-stack BP.

The comparison is intentionally positional-only: final pitch class C at the
last generated event. Continuator's current public ConstraintProblem does not
encode MAXORDER/forbidden-substring regular constraints, so this script omits
MAXORDER for both engines.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_bach_scalability import (  # noqa: E402
    DATA_PATH,
    forbidden_windows,
    load_bach_pitches,
    longest_copied_span,
)
from vo_regular_bp import LongestFeasiblePolicy as VoLongestFeasiblePolicy  # noqa: E402
from vo_regular_bp import OrderStackModel, run_order_stack_bp  # noqa: E402


DEFAULT_CONTINUATOR_ROOT = Path("/Users/francoispachet/IdeaProjects/continuator")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "bach_contextbp_final_compare.csv"


@dataclass(frozen=True)
class BenchmarkRow:
    engine: str
    K: int
    parse_s: float
    model_build_s: float
    graph_build_s: float
    constraint_build_s: float
    bp_s: float
    sampling_s: float
    samples_generated: int
    failures: int
    violations: int
    context_states: int
    context_edges: int
    reachable_states_time_indexed: int | None
    reachable_edges: int | None
    longest_copy_avg: float
    selected_order_avg: float
    selected_order_hist: tuple[tuple[int, int], ...]
    repeat: int

    def as_dict(self) -> dict[str, object]:
        sample_per_sequence = self.sampling_s / self.samples_generated if self.samples_generated else 0.0
        return {
            "engine": self.engine,
            "K": self.K,
            "parse_s": f"{self.parse_s:.9f}",
            "model_build_s": f"{self.model_build_s:.9f}",
            "graph_build_s": f"{self.graph_build_s:.9f}",
            "constraint_build_s": f"{self.constraint_build_s:.9f}",
            "bp_s": f"{self.bp_s:.9f}",
            "sampling_s": f"{self.sampling_s:.9f}",
            "sampling_per_sequence_s": f"{sample_per_sequence:.9f}",
            "samples_generated": self.samples_generated,
            "failures": self.failures,
            "violations": self.violations,
            "context_states": self.context_states,
            "context_edges": self.context_edges,
            "reachable_states_time_indexed": self.reachable_states_time_indexed or "",
            "reachable_edges": self.reachable_edges or "",
            "longest_copy_avg": f"{self.longest_copy_avg:.6f}",
            "selected_order_avg": f"{self.selected_order_avg:.6f}",
            "selected_order_hist": " ".join(
                f"{order}:{count}" for order, count in self.selected_order_hist
            ),
            "repeat": self.repeat,
        }


def c_pitch_values(pitches: Sequence[int], pitch_class: int) -> set[int]:
    values = {int(pitch) for pitch in pitches if int(pitch) % 12 == pitch_class}
    if not values:
        raise ValueError(f"training sequence has no pitch with pitch class {pitch_class}")
    return values


def violates_final_pitch_class(sequence: Sequence[int], pitch_class: int) -> bool:
    return not sequence or int(sequence[-1]) % 12 != pitch_class


def order_stats(order_sequences: Sequence[Sequence[int]]) -> tuple[float, tuple[tuple[int, int], ...]]:
    counts: dict[int, int] = {}
    for orders in order_sequences:
        for order in orders:
            counts[int(order)] = counts.get(int(order), 0) + 1
    total = sum(counts.values())
    avg = sum(order * count for order, count in counts.items()) / total if total else 0.0
    return avg, tuple(sorted(counts.items()))


def copied_span_stats(samples: Sequence[Sequence[int]], pitches: Sequence[int]) -> float:
    if not samples:
        return 0.0
    max_len = max(len(sample) for sample in samples)
    windows = {length: forbidden_windows(pitches, length) for length in range(1, max_len + 1)}
    copied = [longest_copied_span(sample, windows) for sample in samples]
    return sum(copied) / len(copied)


def vo_reachable_counts(result: Any) -> tuple[int, int]:
    states = 0
    edges = 0
    for order, graph in result.graphs.items():
        backward = result.backwards[order]
        for position in range(result.length + 1):
            states += sum(1 for value in backward[position] if value > 0.0)
        for position in range(result.length):
            constraint = result.constraints.get(position)
            beta_next = backward[position + 1]
            for state, beta in enumerate(backward[position]):
                if beta <= 0.0:
                    continue
                for edge in graph.outgoing[state]:
                    if edge.symbol in result.model.forbidden_symbols:
                        continue
                    if not _allows_positional(constraint, edge.symbol):
                        continue
                    if edge.probability * beta_next[edge.dst] > 0.0:
                        edges += 1
    return states, edges


def continuator_reachable_counts(
    graphs: dict[int, Any],
    backwards: dict[int, Any],
    allowed_by_position: list[set[int]],
) -> tuple[int, int]:
    states = 0
    edges = 0
    length = len(allowed_by_position)
    for order, graph in graphs.items():
        backward = backwards[order]
        for position in range(length + 1):
            states += int((backward[position] > 0.0).sum())
        for position in range(length):
            allowed = allowed_by_position[position]
            beta_next = backward[position + 1]
            for state, beta in enumerate(backward[position]):
                if beta <= 0.0:
                    continue
                for edge in graph.outgoing[state]:
                    if edge.symbol in allowed and edge.weight * beta_next[edge.dst] > 0.0:
                        edges += 1
    return states, edges


def _allows_positional(constraint: Any, symbol: Any) -> bool:
    if constraint is None:
        return True
    if callable(constraint):
        return bool(constraint(symbol))
    return symbol in constraint


def run_vo_regular_bp(
    *,
    data_path: Path,
    K: int,
    horizon: int,
    prefix_length: int,
    pitch_class: int,
    samples: int,
    seed: int,
    repeat: int,
) -> BenchmarkRow:
    total_parse = time.perf_counter()
    pitches = tuple(int(pitch) for pitch in load_bach_pitches(data_path))
    parse_s = time.perf_counter() - total_parse
    prefix = tuple(pitches[:prefix_length])
    allowed_c = c_pitch_values(pitches, pitch_class)

    t0 = time.perf_counter()
    model = OrderStackModel.from_sequences([pitches], max_order=K)
    model_build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    for order in range(1, K + 1):
        model.compile_graph(order)
    graph_build_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    constraints = {horizon - 1: allowed_c}
    constraint_build_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    bp = run_order_stack_bp(
        model,
        length=horizon,
        prefix=prefix,
        constraints=constraints,
        policy=VoLongestFeasiblePolicy(),
    )
    bp_s = time.perf_counter() - t3

    rng = random.Random(seed)
    generated: list[tuple[int, ...]] = []
    generated_orders: list[tuple[int, ...]] = []
    failures = 0
    t4 = time.perf_counter()
    for _ in range(samples):
        try:
            sequence, orders = bp.sample_with_orders(rng=rng)
        except Exception:
            failures += 1
            continue
        generated.append(tuple(int(symbol) for symbol in sequence))
        generated_orders.append(tuple(int(order) for order in orders))
    sampling_s = time.perf_counter() - t4

    reachable_states, reachable_edges = vo_reachable_counts(bp)
    order_avg, order_hist = order_stats(generated_orders)
    return BenchmarkRow(
        engine="vo_regular_bp_policy_stack",
        K=K,
        parse_s=parse_s,
        model_build_s=model_build_s,
        graph_build_s=graph_build_s,
        constraint_build_s=constraint_build_s,
        bp_s=bp_s,
        sampling_s=sampling_s,
        samples_generated=len(generated),
        failures=failures,
        violations=sum(violates_final_pitch_class(sample, pitch_class) for sample in generated),
        context_states=bp.context_state_count,
        context_edges=bp.context_edge_count,
        reachable_states_time_indexed=reachable_states,
        reachable_edges=reachable_edges,
        longest_copy_avg=copied_span_stats(generated, pitches),
        selected_order_avg=order_avg,
        selected_order_hist=order_hist,
        repeat=repeat,
    )


def run_continuator_context_bp(
    *,
    continuator_root: Path,
    data_path: Path,
    K: int,
    horizon: int,
    prefix_length: int,
    pitch_class: int,
    samples: int,
    seed: int,
    repeat: int,
) -> BenchmarkRow:
    root = str(continuator_root.expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ctor.constraints import ConstraintProblem
    from ctor.context_bp.inference import backward_messages
    from ctor.context_bp.model import ContextBPModel
    from ctor.context_bp.order_policy import LongestFeasiblePolicy

    total_parse = time.perf_counter()
    pitches = tuple(int(pitch) for pitch in load_bach_pitches(data_path))
    parse_s = time.perf_counter() - total_parse
    prefix = tuple(pitches[:prefix_length])
    allowed_c = c_pitch_values(pitches, pitch_class)

    t0 = time.perf_counter()
    model = ContextBPModel(
        kmax=K,
        viewpoint_fn=lambda pitch: int(pitch),
        seed=seed,
        order_policy=LongestFeasiblePolicy(),
    )
    model.learn_sequence(list(pitches))
    model_build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    graphs = {order: model.compile_graph(prefix=prefix, order=order) for order in range(1, K + 1)}
    graph_build_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    constraints = ConstraintProblem(length=horizon)
    constraints.require_one_of(horizon - 1, allowed_c)
    allowed = model._allowed_symbols_by_position(horizon, constraints)
    constraint_build_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    backwards = {
        order: backward_messages(
            graph,
            length=horizon,
            allowed_symbols_by_position=allowed,
        )
        for order, graph in graphs.items()
    }
    bp_s = time.perf_counter() - t3

    rng = random.Random(seed)
    generated: list[tuple[int, ...]] = []
    generated_orders: list[tuple[int, ...]] = []
    failures = 0
    order_data = {order: (graphs[order], backwards[order]) for order in graphs}
    t4 = time.perf_counter()
    for _ in range(samples):
        try:
            sequence, orders = sample_continuator_with_reused_messages(
                model,
                order_data,
                allowed,
                prefix,
                horizon,
                rng,
            )
        except Exception:
            failures += 1
            continue
        generated.append(sequence)
        generated_orders.append(orders)
    sampling_s = time.perf_counter() - t4

    reachable_states, reachable_edges = continuator_reachable_counts(graphs, backwards, allowed)
    order_avg, order_hist = order_stats(generated_orders)
    return BenchmarkRow(
        engine="continuator_context_bp",
        K=K,
        parse_s=parse_s,
        model_build_s=model_build_s,
        graph_build_s=graph_build_s,
        constraint_build_s=constraint_build_s,
        bp_s=bp_s,
        sampling_s=sampling_s,
        samples_generated=len(generated),
        failures=failures,
        violations=sum(violates_final_pitch_class(sample, pitch_class) for sample in generated),
        context_states=sum(len(graph.contexts) for graph in graphs.values()),
        context_edges=sum(sum(len(edges) for edges in graph.outgoing) for graph in graphs.values()),
        reachable_states_time_indexed=reachable_states,
        reachable_edges=reachable_edges,
        longest_copy_avg=copied_span_stats(generated, pitches),
        selected_order_avg=order_avg,
        selected_order_hist=order_hist,
        repeat=repeat,
    )


def sample_continuator_with_reused_messages(
    model: Any,
    order_data: dict[int, tuple[Any, Any]],
    allowed_symbols_by_position: list[set[int]],
    prefix: Sequence[int],
    length: int,
    rng: random.Random,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    history = [model.vocabulary.start_id]
    history.extend(model._encode_value(item) for item in prefix)
    sequence: list[int] = []
    orders: list[int] = []
    for position in range(length):
        candidate_sets = model._candidate_sets_for_position(
            order_data,
            history,
            allowed_symbols_by_position,
            position,
        )
        chosen = model.order_policy.choose(candidate_sets, rng)
        if chosen is None:
            raise RuntimeError("no feasible continuation")
        edge = chosen.edge
        sequence.append(edge.symbol)
        orders.append(int(edge.order))
        history.append(edge.symbol)
    decoded = tuple(int(model.vocabulary.decode(symbol)) for symbol in sequence)
    return decoded, tuple(orders)


def median_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_fields = [
        "parse_s",
        "model_build_s",
        "graph_build_s",
        "constraint_build_s",
        "bp_s",
        "sampling_s",
        "sampling_per_sequence_s",
        "longest_copy_avg",
        "selected_order_avg",
    ]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["engine"]), int(row["K"])), []).append(row)

    medians: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        row = dict(group[0])
        for field in numeric_fields:
            row[field] = f"{statistics.median(float(item[field]) for item in group):.9f}"
        row["repeat"] = "median"
        medians.append(row)
    return medians


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, object]]) -> None:
    columns = [
        "engine",
        "K",
        "model_build_s",
        "graph_build_s",
        "constraint_build_s",
        "bp_s",
        "sampling_per_sequence_s",
        "context_states",
        "context_edges",
        "reachable_states_time_indexed",
        "reachable_edges",
        "violations",
        "longest_copy_avg",
        "selected_order_avg",
        "selected_order_hist",
    ]
    print(",".join(columns))
    for row in rows:
        print(",".join(str(row[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--continuator-root", type=Path, default=DEFAULT_CONTINUATOR_ROOT)
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument("--pitch-class", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=None)
    args = parser.parse_args()

    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")

    raw_rows: list[dict[str, object]] = []
    for repeat in range(-args.warmups, args.repeats):
        measured = repeat >= 0
        for K in args.orders:
            for runner in (run_continuator_context_bp, run_vo_regular_bp):
                row = runner(
                    continuator_root=args.continuator_root,
                    data_path=args.data,
                    K=K,
                    horizon=args.horizon,
                    prefix_length=args.prefix_length,
                    pitch_class=args.pitch_class,
                    samples=args.samples,
                    seed=args.seed + max(repeat, 0) * 100 + K,
                    repeat=repeat,
                ) if runner is run_continuator_context_bp else runner(
                    data_path=args.data,
                    K=K,
                    horizon=args.horizon,
                    prefix_length=args.prefix_length,
                    pitch_class=args.pitch_class,
                    samples=args.samples,
                    seed=args.seed + max(repeat, 0) * 100 + K,
                    repeat=repeat,
                )
                if measured:
                    raw_rows.append(row.as_dict())

    rows = median_rows(raw_rows)
    print_table(rows)
    write_csv(args.output, rows)
    raw_output = args.raw_output or args.output.with_name(f"{args.output.stem}_raw{args.output.suffix}")
    write_csv(raw_output, raw_rows)
    print(f"\nCSV written to {args.output}")
    print(f"Raw CSV written to {raw_output}")


if __name__ == "__main__":
    main()
