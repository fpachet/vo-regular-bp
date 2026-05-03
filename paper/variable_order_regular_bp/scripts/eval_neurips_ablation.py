#!/usr/bin/env python
"""NeurIPS ablation for sparse VO regular-constrained BP on Bach pitches.

This is an internal ablation of the proposed constrained order-stack method.
The fixed pure longest-observed-suffix source is recorded only as a diagnostic
constrained mass; it is not sampled and is not treated as a competing method.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import statistics
import sys
import time
import tracemalloc
from typing import Hashable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from paper.variable_order_regular_bp.scripts.eval_bach_scalability import (  # noqa: E402
    BachConfig,
    ConstraintCache,
    DATA_PATH,
    GraphCache,
    PolicyStackConstraintBundle,
    StackModelCache,
    accepts_from,
    build_policy_stack_optimized_constraint,
    forbidden_windows,
    load_bach_pitches,
    longest_copied_span,
    run_configuration,
)
from vo_regular_bp import (  # noqa: E402
    LongestFeasiblePolicy,
    OrderStackModel,
    RegularOrderStackBPResult,
    run_order_stack_masked_dfa_bp,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "neurips_ablation"
DEFAULT_HORIZONS = (16, 32, 64, 96, 128)
DEFAULT_MAXORDER_VALUES = (3, 4, 5, 6, 7)


@dataclass(frozen=True)
class AblationRun:
    sweep: str
    repeat: int
    label: str
    K: int
    n: int
    M: int
    final_pitch_class: int
    alphabet_size: int
    training_length: int
    prefix: tuple[int, ...]
    context_build_s: float
    acceptor_build_s: float
    bp_s: float
    sampling_s: float
    peak_memory_mib: float
    context_states: int
    context_edges: int
    acceptor_states: int
    acceptor_edges: int
    reachable_product_states: int
    reachable_product_states_time_indexed: int
    reachable_product_edges: int
    policy_reachable_states: int
    policy_reachable_terminal_states: int
    success_mass: float
    diagnostic_pure_longest_suffix_mass: float
    diagnostic_pure_bp_s: float
    samples_requested: int
    samples_generated: int
    violation_count: int
    direct_violation_count: int
    max_copied_span: int
    longest_copy_avg: float
    selected_order_avg: float
    selected_order_hist: tuple[tuple[int, int], ...]
    order_start_masses: tuple[tuple[int, float], ...]

    @property
    def sampling_per_sequence_s(self) -> float:
        if not self.samples_generated:
            return 0.0
        return self.sampling_s / self.samples_generated

    def row(self) -> dict[str, object]:
        return {
            "sweep": self.sweep,
            "repeat": self.repeat,
            "label": self.label,
            "K": self.K,
            "n": self.n,
            "M": self.M,
            "final_pitch_class": self.final_pitch_class,
            "alphabet_size": self.alphabet_size,
            "training_length": self.training_length,
            "context_states": self.context_states,
            "context_edges": self.context_edges,
            "acceptor_states": self.acceptor_states,
            "acceptor_edges": self.acceptor_edges,
            "reachable_product_states": self.reachable_product_states,
            "reachable_product_states_time_indexed": self.reachable_product_states_time_indexed,
            "reachable_product_edges": self.reachable_product_edges,
            "policy_reachable_states": self.policy_reachable_states,
            "policy_reachable_terminal_states": self.policy_reachable_terminal_states,
            "context_build_s": f"{self.context_build_s:.6f}",
            "acceptor_build_s": f"{self.acceptor_build_s:.6f}",
            "bp_s": f"{self.bp_s:.6f}",
            "sampling_s": f"{self.sampling_s:.6f}",
            "sampling_per_sequence_s": f"{self.sampling_per_sequence_s:.9f}",
            "peak_memory_mib": f"{self.peak_memory_mib:.3f}",
            "success_mass": f"{self.success_mass:.12g}",
            "mass_kind": "policy_success_mass",
            "diagnostic_pure_longest_suffix_mass": f"{self.diagnostic_pure_longest_suffix_mass:.12g}",
            "diagnostic_pure_mass_kind": "diagnostic_fixed_pure_longest_suffix",
            "diagnostic_pure_bp_s": f"{self.diagnostic_pure_bp_s:.6f}",
            "samples_requested": self.samples_requested,
            "samples_generated": self.samples_generated,
            "violation_count": self.violation_count,
            "direct_violation_count": self.direct_violation_count,
            "max_copied_span": self.max_copied_span,
            "longest_copy_avg": f"{self.longest_copy_avg:.3f}",
            "selected_order_avg": f"{self.selected_order_avg:.3f}",
            "selected_order_hist": format_hist(self.selected_order_hist, self.K),
            "order_start_masses": " ".join(
                f"{order}:{mass:.12g}" for order, mass in self.order_start_masses
            ),
            "prefix": " ".join(map(str, self.prefix)),
        }


def format_hist(hist: Sequence[tuple[int, int]], max_order: int) -> str:
    counts = dict(hist)
    return " ".join(f"{order}:{counts.get(order, 0)}" for order in range(1, max_order + 1))


def parse_hist(text: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    for part in text.split():
        order, count = part.split(":", maxsplit=1)
        counts[int(order)] += int(count)
    return counts


def count_direct_violations(
    samples: Sequence[Sequence[int]],
    *,
    training_pitches: Sequence[int],
    forbidden_ngram: int,
    final_pitch_class: int,
) -> int:
    forbidden = forbidden_windows(training_pitches, forbidden_ngram)
    violations = 0
    for sample in samples:
        final_ok = bool(sample) and int(sample[-1]) % 12 == final_pitch_class
        maxorder_ok = all(
            tuple(sample[index : index + forbidden_ngram]) not in forbidden
            for index in range(0, len(sample) - forbidden_ngram + 1)
        )
        if not final_ok or not maxorder_ok:
            violations += 1
    return violations


def expand_longest_feasible_policy_reachable(bp: RegularOrderStackBPResult) -> tuple[int, int]:
    """Expand all policy-reachable suffix/DFA states for deterministic counts."""

    max_order = bp.model.max_order
    frontier: set[tuple[tuple[Hashable, ...], Hashable]] = {
        (tuple(bp.prefix[-max_order:]), bp.start_acceptor_state)
    }
    state_total = len(frontier)

    for position in range(bp.length):
        next_frontier: set[tuple[tuple[Hashable, ...], Hashable]] = set()
        for suffix, acceptor_state in frontier:
            candidate_sets = bp._candidate_sets(position, suffix, acceptor_state)
            if not candidate_sets:
                continue
            selected = candidate_sets[0]
            cache = bp.backwards[selected.order]
            for edge in selected.edges:
                next_acceptor_state = cache.next_acceptor_state(acceptor_state, edge.symbol)
                if next_acceptor_state is None:
                    continue
                next_suffix = (suffix + (edge.symbol,))[-max_order:]
                next_frontier.add((next_suffix, next_acceptor_state))
        frontier = next_frontier
        state_total += len(frontier)

    terminal_accepting = sum(1 for _suffix, state in frontier if bp.acceptor.is_accepting(state))
    return state_total, terminal_accepting


def run_policy_stack_ablation(
    pitches: Sequence[int],
    *,
    sweep: str,
    repeat: int,
    label: str,
    K: int,
    n: int,
    M: int,
    prefix: Sequence[int],
    samples: int,
    seed: int,
    final_pitch_class: int,
    stack_model_cache: StackModelCache,
    constraint_cache: ConstraintCache,
    diagnostic_pure_longest_suffix_mass: float,
    diagnostic_pure_bp_s: float,
    copy_windows_cache: dict[int, dict[int, set[tuple[int, ...]]]],
) -> AblationRun:
    if K in stack_model_cache:
        model = stack_model_cache[K]
        context_build_s = 0.0
    else:
        t0 = time.perf_counter()
        model = OrderStackModel.from_sequences([pitches], max_order=K)
        context_build_s = time.perf_counter() - t0
        stack_model_cache[K] = model

    alphabet = tuple(sorted(model.alphabet))
    constraint_key = ("neurips_policy_stack", alphabet, n, M, final_pitch_class)
    if constraint_key in constraint_cache:
        constraint_bundle = constraint_cache[constraint_key]
        if not isinstance(constraint_bundle, PolicyStackConstraintBundle):
            raise TypeError("cached NeurIPS policy-stack constraint has unexpected type")
        acceptor_build_s = 0.0
    else:
        constraint_bundle, acceptor_build_s = build_policy_stack_optimized_constraint(
            pitches,
            alphabet,
            n,
            M,
            final_pitch_class,
        )
        constraint_cache[constraint_key] = constraint_bundle

    tracemalloc.start()
    t1 = time.perf_counter()
    bp = run_order_stack_masked_dfa_bp(
        model,
        constraint_bundle.regular_acceptor,
        length=n,
        prefix=prefix,
        constraints=constraint_bundle.positional_constraints,
        start_acceptor_state=constraint_bundle.regular_start_state,
        policy=LongestFeasiblePolicy(),
    )
    policy_reachable_states, policy_reachable_terminal_states = expand_longest_feasible_policy_reachable(bp)
    success_mass = bp.success_mass
    order_start_masses = bp.start_order_masses()
    bp_s = time.perf_counter() - t1
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rng = random.Random(seed)
    t2 = time.perf_counter()
    generated_with_orders = bp.sample_many_with_orders(samples, rng=rng) if success_mass > 0.0 else []
    sampling_s = time.perf_counter() - t2
    generated = [tuple(int(symbol) for symbol in sample) for sample, _orders in generated_with_orders]
    generated_orders = [orders for _sample, orders in generated_with_orders]

    if success_mass > 0.0 and len(generated) != samples:
        raise RuntimeError(f"expected {samples} samples for n={n}, M={M}, got {len(generated)}")

    if n not in copy_windows_cache:
        copy_windows_cache[n] = {length: forbidden_windows(pitches, length) for length in range(1, n + 1)}
    copied = [longest_copied_span(sample, copy_windows_cache[n]) for sample in generated]

    validation_violations = sum(
        1
        for sample in generated
        if not accepts_from(
            constraint_bundle.validation_acceptor,
            sample,
            constraint_bundle.validation_start_state,
        )
    )
    direct_violations = count_direct_violations(
        generated,
        training_pitches=pitches,
        forbidden_ngram=M,
        final_pitch_class=final_pitch_class,
    )
    if validation_violations or direct_violations:
        raise RuntimeError(
            "sample verification failed for "
            f"n={n}, M={M}: validation={validation_violations}, direct={direct_violations}"
        )

    order_counts: Counter[int] = Counter()
    for orders in generated_orders:
        order_counts.update(int(order) for order in orders)
    order_total = sum(order_counts.values())
    selected_order_avg = (
        sum(order * count for order, count in order_counts.items()) / order_total
        if order_total
        else 0.0
    )

    return AblationRun(
        sweep=sweep,
        repeat=repeat,
        label=label,
        K=K,
        n=n,
        M=M,
        final_pitch_class=final_pitch_class,
        alphabet_size=len(set(pitches)),
        training_length=len(pitches),
        prefix=tuple(prefix),
        context_build_s=context_build_s,
        acceptor_build_s=acceptor_build_s,
        bp_s=bp_s,
        sampling_s=sampling_s,
        peak_memory_mib=peak / (1024 * 1024),
        context_states=bp.context_state_count,
        context_edges=bp.context_edge_count,
        acceptor_states=constraint_bundle.acceptor_states,
        acceptor_edges=constraint_bundle.acceptor_edges,
        reachable_product_states=bp.product_state_count,
        reachable_product_states_time_indexed=bp.time_indexed_product_state_count,
        reachable_product_edges=bp.product_edge_count,
        policy_reachable_states=policy_reachable_states,
        policy_reachable_terminal_states=policy_reachable_terminal_states,
        success_mass=success_mass,
        diagnostic_pure_longest_suffix_mass=diagnostic_pure_longest_suffix_mass,
        diagnostic_pure_bp_s=diagnostic_pure_bp_s,
        samples_requested=samples,
        samples_generated=len(generated),
        violation_count=validation_violations,
        direct_violation_count=direct_violations,
        max_copied_span=max(copied, default=0),
        longest_copy_avg=sum(copied) / len(copied) if copied else 0.0,
        selected_order_avg=selected_order_avg,
        selected_order_hist=tuple(sorted(order_counts.items())),
        order_start_masses=order_start_masses,
    )


def run_diagnostic_pure_mass(
    pitches: Sequence[int],
    *,
    K: int,
    n: int,
    M: int,
    prefix: Sequence[int],
    final_pitch_class: int,
    graph_cache: GraphCache,
    constraint_cache: ConstraintCache,
) -> tuple[float, float]:
    result = run_configuration(
        pitches,
        BachConfig(
            max_order=K,
            horizon=n,
            forbidden_ngram=M,
            final_pitch_class=final_pitch_class,
            source_policy="pure",
        ),
        prefix=prefix,
        samples=0,
        seed=0,
        graph_cache=graph_cache,
        constraint_cache=constraint_cache,
    )
    return result.partition_function, result.bp_s


def median_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    median_fields = {
        "context_build_s",
        "acceptor_build_s",
        "bp_s",
        "sampling_s",
        "sampling_per_sequence_s",
        "peak_memory_mib",
        "success_mass",
        "diagnostic_pure_longest_suffix_mass",
        "diagnostic_pure_bp_s",
        "longest_copy_avg",
        "selected_order_avg",
    }
    median_int_fields = {
        "context_states",
        "context_edges",
        "acceptor_states",
        "acceptor_edges",
        "reachable_product_states",
        "reachable_product_states_time_indexed",
        "reachable_product_edges",
        "policy_reachable_states",
        "policy_reachable_terminal_states",
    }
    sum_int_fields = {
        "samples_generated",
        "violation_count",
        "direct_violation_count",
    }
    max_int_fields = {"max_copied_span"}
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in raw_rows:
        key = (row["sweep"], row["K"], row["n"], row["M"], row["final_pitch_class"])
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        row = dict(group[0])
        row["repeat"] = "median"
        row["label"] = "median"
        for field in median_fields:
            row[field] = f"{statistics.median(float(item[field]) for item in group):.12g}"
        for field in median_int_fields:
            row[field] = int(statistics.median(int(item[field]) for item in group))
        for field in sum_int_fields:
            row[field] = sum(int(item[field]) for item in group)
        for field in max_int_fields:
            row[field] = max(int(item[field]) for item in group)
        row["samples_requested"] = sum(int(item["samples_requested"]) for item in group)
        hist: Counter[int] = Counter()
        for item in group:
            hist.update(parse_hist(str(item["selected_order_hist"])))
        max_order = int(row["K"])
        row["selected_order_hist"] = format_hist(tuple(sorted(hist.items())), max_order)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_number(value: object) -> str:
    return f"{int(value):,}"


def latex_seconds(value: object) -> str:
    return f"{float(value):.3f}"


def latex_millis(value: object) -> str:
    return f"{1000.0 * float(value):.3f}"


def latex_mass(value: object) -> str:
    numeric = float(value)
    if numeric == 0.0:
        return "0"
    if abs(numeric) < 1e-3:
        return f"{numeric:.1e}"
    return f"{numeric:.3g}"


def write_horizon_table(path: Path, rows: list[dict[str, object]]) -> None:
    K = rows[0]["K"] if rows else 6
    M = rows[0]["M"] if rows else 5
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Horizon ablation for constrained order-stack BP on Bach Prelude pitches ($K={K}$, MAXORDER $M={M}$). The fixed pure longest-suffix mass is a diagnostic only.}}",
        r"\label{tab:vo-bp-horizon-ablation}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$n$ & $|V_\otimes|$ & $|E_\otimes|$ & BP (s) & samp. (ms) & succ. mass & diag. $Z_{\rm pure}$ & max copy \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=lambda item: int(item["n"])):
        lines.append(
            " & ".join(
                [
                    str(row["n"]),
                    latex_number(row["reachable_product_states_time_indexed"]),
                    latex_number(row["reachable_product_edges"]),
                    latex_seconds(row["bp_s"]),
                    latex_millis(row["sampling_per_sequence_s"]),
                    latex_mass(row["success_mass"]),
                    latex_mass(row["diagnostic_pure_longest_suffix_mass"]),
                    str(row["max_copied_span"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_maxorder_table(path: Path, rows: list[dict[str, object]]) -> None:
    K = rows[0]["K"] if rows else 6
    n = rows[0]["n"] if rows else 32
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{MAXORDER-strength ablation for constrained order-stack BP on Bach Prelude pitches ($K={K}$, horizon $n={n}$). The fixed pure longest-suffix mass is a diagnostic only.}}",
        r"\label{tab:vo-bp-maxorder-ablation}",
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r"$M$ & $|Q|$ & $|V_\otimes|$ & $|E_\otimes|$ & BP (s) & samp. (ms) & succ. mass & diag. $Z_{\rm pure}$ & max copy \\",
        r"\midrule",
    ]
    for row in sorted(rows, key=lambda item: int(item["M"])):
        lines.append(
            " & ".join(
                [
                    str(row["M"]),
                    latex_number(row["acceptor_states"]),
                    latex_number(row["reachable_product_states_time_indexed"]),
                    latex_number(row["reachable_product_edges"]),
                    latex_seconds(row["bp_s"]),
                    latex_millis(row["sampling_per_sequence_s"]),
                    latex_mass(row["success_mass"]),
                    latex_mass(row["diagnostic_pure_longest_suffix_mass"]),
                    str(row["max_copied_span"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reproduce_commands(path: Path, args: argparse.Namespace) -> None:
    command = (
        "python scripts/eval_neurips_ablation.py "
        f"--data {args.data} "
        f"--output-dir {args.output_dir} "
        f"--max-order {args.max_order} "
        "--horizons "
        + " ".join(map(str, args.horizons))
        + " "
        f"--horizon-maxorder {args.horizon_maxorder} "
        f"--maxorder-horizon {args.maxorder_horizon} "
        "--maxorder-values "
        + " ".join(map(str, args.maxorder_values))
        + " "
        f"--samples {args.samples} "
        f"--warmups {args.warmups} "
        f"--repeats {args.repeats} "
        f"--seed {args.seed} "
        f"--final-pitch-class {args.final_pitch_class} "
        f"--prefix-length {args.prefix_length} "
        "--sweeps "
        + " ".join(args.sweeps)
    )
    text = (
        "Reproduce the NeurIPS ablation sweeps with:\n\n"
        f"{command}\n\n"
        "The run uses the Bach Prelude pitch-only corpus, alphabet size 25, "
        "final pitch-class C, constrained longest-feasible order-stack policy, "
        f"{args.warmups} warmup run(s), fixed seeds, and "
        f"{args.samples} sample(s) per measured repeat. "
        "The pure longest-observed-suffix constrained mass is diagnostic only, "
        "not a baseline.\n"
    )
    path.write_text(text, encoding="utf-8")


def run_sweep(
    *,
    sweep: str,
    configs: Sequence[tuple[int, int]],
    pitches: Sequence[int],
    prefix: Sequence[int],
    args: argparse.Namespace,
    graph_cache: GraphCache,
    stack_model_cache: StackModelCache,
    constraint_cache: ConstraintCache,
    copy_windows_cache: dict[int, dict[int, set[tuple[int, ...]]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_runs: list[AblationRun] = []
    total_configs = len(configs)
    for repeat in range(-args.warmups, args.repeats):
        measured = repeat >= 0
        for index, (n, M) in enumerate(configs):
            pure_mass, pure_bp_s = run_diagnostic_pure_mass(
                pitches,
                K=args.max_order,
                n=n,
                M=M,
                prefix=prefix,
                final_pitch_class=args.final_pitch_class,
                graph_cache=graph_cache,
                constraint_cache=constraint_cache,
            )
            seed_offset = (repeat if measured else 100_000 - repeat) * total_configs + index
            run = run_policy_stack_ablation(
                pitches,
                sweep=sweep,
                repeat=repeat,
                label=f"repeat_{repeat}" if measured else f"warmup_{abs(repeat)}",
                K=args.max_order,
                n=n,
                M=M,
                prefix=prefix,
                samples=args.samples,
                seed=args.seed + seed_offset,
                final_pitch_class=args.final_pitch_class,
                stack_model_cache=stack_model_cache,
                constraint_cache=constraint_cache,
                diagnostic_pure_longest_suffix_mass=pure_mass,
                diagnostic_pure_bp_s=pure_bp_s,
                copy_windows_cache=copy_windows_cache,
            )
            if measured:
                raw_runs.append(run)

    raw_rows = [run.row() for run in raw_runs]
    return median_rows(raw_rows), raw_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--horizon-maxorder", type=int, default=5)
    parser.add_argument("--maxorder-horizon", type=int, default=32)
    parser.add_argument("--maxorder-values", nargs="+", type=int, default=list(DEFAULT_MAXORDER_VALUES))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--final-pitch-class", type=int, default=0)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument("--sweeps", nargs="+", choices=("horizon", "maxorder"), default=["horizon", "maxorder"])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_order < 1:
        raise ValueError("--max-order must be positive")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.warmups < 1:
        raise ValueError("--warmups must be at least 1 for this ablation")
    if args.repeats < 5:
        raise ValueError("--repeats must be at least 5 for this ablation")

    pitches = load_bach_pitches(args.data)
    if len(set(pitches)) != 25:
        raise ValueError(f"expected Bach pitch alphabet size 25, got {len(set(pitches))}")
    if len(pitches) < args.prefix_length:
        raise ValueError("prefix length exceeds pitch sequence length")
    prefix = tuple(pitches[: args.prefix_length])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_cache: GraphCache = {}
    stack_model_cache: StackModelCache = {}
    constraint_cache: ConstraintCache = {}
    copy_windows_cache: dict[int, dict[int, set[tuple[int, ...]]]] = {}

    if "horizon" in args.sweeps:
        configs = [(horizon, args.horizon_maxorder) for horizon in args.horizons]
        rows, raw_rows = run_sweep(
            sweep="horizon",
            configs=configs,
            pitches=pitches,
            prefix=prefix,
            args=args,
            graph_cache=graph_cache,
            stack_model_cache=stack_model_cache,
            constraint_cache=constraint_cache,
            copy_windows_cache=copy_windows_cache,
        )
        write_csv(args.output_dir / "horizon_sweep.csv", rows)
        write_csv(args.output_dir / "horizon_sweep_raw.csv", raw_rows)
        write_horizon_table(args.output_dir / "horizon_sweep_table.tex", rows)

    if "maxorder" in args.sweeps:
        configs = [(args.maxorder_horizon, maxorder) for maxorder in args.maxorder_values]
        rows, raw_rows = run_sweep(
            sweep="maxorder",
            configs=configs,
            pitches=pitches,
            prefix=prefix,
            args=args,
            graph_cache=graph_cache,
            stack_model_cache=stack_model_cache,
            constraint_cache=constraint_cache,
            copy_windows_cache=copy_windows_cache,
        )
        write_csv(args.output_dir / "maxorder_sweep.csv", rows)
        write_csv(args.output_dir / "maxorder_sweep_raw.csv", raw_rows)
        write_maxorder_table(args.output_dir / "maxorder_sweep_table.tex", rows)

    write_reproduce_commands(args.output_dir / "reproduce_commands.txt", args)
    print(f"NeurIPS ablation outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
