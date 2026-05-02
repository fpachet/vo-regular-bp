#!/usr/bin/env python
"""Scalability evaluation for exact VO product BP on Bach Prelude pitches."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import statistics
import sys
import time
import tracemalloc
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vo_regular_bp import (  # noqa: E402
    ContextGraph,
    DFA,
    LongestFeasiblePolicy,
    OrderStackModel,
    all_of,
    dense_forbidden_substring_acceptor,
    forbidden_substring_acceptor,
    positional_acceptor,
    run_bp,
    run_order_stack_masked_dfa_bp,
)

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "bach_prelude_c_major_pitches.txt"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "outputs" / "bach_scalability.csv"

GraphCache = dict[tuple[str, int, float], ContextGraph]
StackModelCache = dict[int, OrderStackModel]
ConstraintCache = dict[tuple[object, ...], object]


@dataclass(frozen=True)
class GenericConstraintBundle:
    acceptor: DFA
    start_state: object
    acceptor_states: int
    acceptor_edges: int


@dataclass(frozen=True)
class PolicyStackConstraintBundle:
    regular_acceptor: DFA
    regular_start_state: object
    positional_constraints: dict[int, frozenset[int]]
    validation_acceptor: DFA
    validation_start_state: object
    acceptor_states: int
    acceptor_edges: int


@dataclass(frozen=True)
class BachConfig:
    max_order: int
    horizon: int
    forbidden_ngram: int
    final_pitch_class: int = 0
    backoff_weight: float = 0.25
    source_policy: str = "pure"


@dataclass(frozen=True)
class BachResult:
    config: BachConfig
    graph: ContextGraph | None
    acceptor: DFA
    start_context: tuple[int, ...]
    start_acceptor_state: object
    prefix: tuple[int, ...]
    samples: list[tuple[int, ...]]
    sample_orders: list[tuple[int, ...]]
    context_build_s: float
    acceptor_build_s: float
    bp_s: float
    sampling_s: float
    peak_memory_mib: float
    context_states: int
    context_edges: int
    acceptor_states: int
    acceptor_edges: int
    full_product_edge_upper_bound: int
    dense_lifted_state_count: int
    constraint_violations: int
    longest_copy_max: int
    longest_copy_avg: float
    selected_order_avg: float
    selected_order_hist: tuple[tuple[int, int], ...]
    partition_function: float
    mass_kind: str
    order_start_masses: tuple[tuple[int, float], ...] = ()

    def row(self, train_length: int, alphabet_size: int, sample_count: int) -> dict[str, object]:
        sampling_per_sequence = self.sampling_s / len(self.samples) if self.samples else 0.0
        sampling_per_event = (
            self.sampling_s / (len(self.samples) * self.config.horizon)
            if self.samples and self.config.horizon
            else 0.0
        )
        return {
            "source_policy": self.config.source_policy,
            "K": self.config.max_order,
            "n": self.config.horizon,
            "M": self.config.forbidden_ngram,
            "final_pitch_class": self.config.final_pitch_class,
            "backoff_weight": self.config.backoff_weight,
            "alphabet_size": alphabet_size,
            "training_length": train_length,
            "context_states": self.context_states,
            "context_edges": self.context_edges,
            "acceptor_states": self.acceptor_states,
            "acceptor_edges": self.acceptor_edges,
            "reachable_product_states": self.bp_unique_states,
            "reachable_product_states_time_indexed": self.bp_time_indexed_states,
            "reachable_product_edges": self.bp_edges,
            "full_product_edge_upper_bound": self.full_product_edge_upper_bound,
            "dense_lifted_state_count": self.dense_lifted_state_count,
            "context_build_s": f"{self.context_build_s:.6f}",
            "acceptor_build_s": f"{self.acceptor_build_s:.6f}",
            "bp_s": f"{self.bp_s:.6f}",
            "sampling_s": f"{self.sampling_s:.6f}",
            "sampling_per_sequence_s": f"{sampling_per_sequence:.8f}",
            "sampling_per_event_s": f"{sampling_per_event:.10f}",
            "peak_memory_mib": f"{self.peak_memory_mib:.3f}",
            "partition_function": f"{self.partition_function:.12g}",
            "mass_kind": self.mass_kind,
            "order_start_masses": " ".join(
                f"{order}:{mass:.12g}" for order, mass in self.order_start_masses
            ),
            "requested_samples": sample_count,
            "samples_generated": len(self.samples),
            "constraint_violations": self.constraint_violations,
            "longest_copy_max": self.longest_copy_max,
            "longest_copy_avg": f"{self.longest_copy_avg:.3f}",
            "selected_order_avg": f"{self.selected_order_avg:.3f}",
            "selected_order_hist": " ".join(
                f"{order}:{count}" for order, count in self.selected_order_hist
            ),
            "prefix": " ".join(map(str, self.prefix)),
        }

    @property
    def bp_unique_states(self) -> int:
        return getattr(self, "_bp_unique_states")

    @property
    def bp_time_indexed_states(self) -> int:
        return getattr(self, "_bp_time_indexed_states")

    @property
    def bp_edges(self) -> int:
        return getattr(self, "_bp_edges")


def load_bach_pitches(path: Path = DATA_PATH) -> tuple[int, ...]:
    tokens: list[int] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        tokens.extend(int(token) for token in line.split())
    if not tokens:
        raise ValueError(f"no pitch tokens found in {path}")
    return tuple(tokens)


def forbidden_windows(sequence: Sequence[int], length: int) -> set[tuple[int, ...]]:
    if length <= 0:
        raise ValueError("forbidden n-gram length must be positive")
    return {
        tuple(sequence[index : index + length])
        for index in range(0, len(sequence) - length + 1)
    }


def prefix_context(graph: ContextGraph, prefix: Sequence[int], max_order: int) -> tuple[int, ...]:
    for order in range(min(max_order, len(prefix)), -1, -1):
        suffix = tuple(prefix[-order:]) if order else ()
        if suffix in graph.states:
            return suffix
    return graph.start_state


def accepts_from(acceptor: DFA, sequence: Sequence[int], start_state: object) -> bool:
    state = start_state
    for symbol in sequence:
        state = acceptor.next_state(state, symbol)
        if state is None:
            return False
    return acceptor.is_accepting(state)


def build_constraint_acceptor(
    training_pitches: Sequence[int],
    alphabet: Iterable[int],
    horizon: int,
    forbidden_ngram: int,
    final_pitch_class: int,
    prefix: Sequence[int],
) -> tuple[DFA, object, int, int, float]:
    alphabet_tuple = tuple(sorted(alphabet))
    t0 = time.perf_counter()

    final_pitches = {pitch for pitch in alphabet_tuple if pitch % 12 == final_pitch_class}
    if not final_pitches:
        raise ValueError(f"alphabet has no pitch class {final_pitch_class}")

    final_acceptor = positional_acceptor(
        horizon,
        {horizon - 1: final_pitches},
        alphabet=alphabet_tuple,
        name="final_pitch_class",
    )
    forbidden_acceptor = forbidden_substring_acceptor(
        forbidden_windows(training_pitches, forbidden_ngram),
        alphabet=alphabet_tuple,
        name=f"avoid_training_{forbidden_ngram}grams",
    )
    acceptor = all_of(final_acceptor, forbidden_acceptor, name="final_pc_and_maxorder")
    start_state = acceptor.start_state

    final_states = final_acceptor.state_count() or 0
    forbidden_states = forbidden_acceptor.state_count() or 0
    acceptor_states = final_states * forbidden_states
    acceptor_edges = count_composite_edges(final_acceptor, forbidden_acceptor, alphabet_tuple)
    return acceptor, start_state, acceptor_states, acceptor_edges, time.perf_counter() - t0


def build_policy_stack_optimized_constraint(
    training_pitches: Sequence[int],
    alphabet: Iterable[int],
    horizon: int,
    forbidden_ngram: int,
    final_pitch_class: int,
) -> tuple[PolicyStackConstraintBundle, float]:
    alphabet_tuple = tuple(sorted(alphabet))
    t0 = time.perf_counter()

    final_pitches = frozenset(pitch for pitch in alphabet_tuple if pitch % 12 == final_pitch_class)
    if not final_pitches:
        raise ValueError(f"alphabet has no pitch class {final_pitch_class}")

    regular_acceptor = dense_forbidden_substring_acceptor(
        forbidden_windows(training_pitches, forbidden_ngram),
        alphabet=alphabet_tuple,
        name=f"dense_avoid_training_{forbidden_ngram}grams",
    )
    positional_constraints = {horizon - 1: final_pitches}

    final_acceptor = positional_acceptor(
        horizon,
        positional_constraints,
        alphabet=alphabet_tuple,
        name="final_pitch_class",
    )
    validation_acceptor = all_of(
        final_acceptor,
        regular_acceptor,
        name="validation_final_pc_and_maxorder",
    )

    bundle = PolicyStackConstraintBundle(
        regular_acceptor=regular_acceptor,
        regular_start_state=regular_acceptor.start_state,
        positional_constraints=positional_constraints,
        validation_acceptor=validation_acceptor,
        validation_start_state=validation_acceptor.start_state,
        acceptor_states=regular_acceptor.state_count() or 0,
        acceptor_edges=count_dfa_edges(regular_acceptor, alphabet_tuple),
    )
    return bundle, time.perf_counter() - t0


def count_dfa_edges(acceptor: DFA, alphabet: Sequence[int]) -> int:
    if hasattr(acceptor, "edge_count"):
        return int(acceptor.edge_count())
    if acceptor.states is None:
        return 0
    edges = 0
    for state in acceptor.states:
        for symbol in alphabet:
            if acceptor.next_state(state, symbol) is not None:
                edges += 1
    return edges


def count_composite_edges(final_acceptor: DFA, forbidden_acceptor: DFA, alphabet: Sequence[int]) -> int:
    if final_acceptor.states is None or forbidden_acceptor.states is None:
        return 0
    edges = 0
    for final_state in final_acceptor.states:
        for forbidden_state in forbidden_acceptor.states:
            for symbol in alphabet:
                if final_acceptor.next_state(final_state, symbol) is None:
                    continue
                if forbidden_acceptor.next_state(forbidden_state, symbol) is None:
                    continue
                edges += 1
    return edges


def longest_copied_span(sample: Sequence[int], training_windows: dict[int, set[tuple[int, ...]]]) -> int:
    for length in range(len(sample), 0, -1):
        windows = training_windows.get(length, set())
        if any(tuple(sample[index : index + length]) in windows for index in range(len(sample) - length + 1)):
            return length
    return 0


def run_configuration(
    pitches: Sequence[int],
    config: BachConfig,
    *,
    prefix: Sequence[int],
    samples: int,
    seed: int,
    graph_cache: GraphCache | None = None,
    stack_model_cache: StackModelCache | None = None,
    constraint_cache: ConstraintCache | None = None,
) -> BachResult:
    if config.source_policy == "policy_stack":
        return run_policy_stack_configuration(
            pitches,
            config,
            prefix=prefix,
            samples=samples,
            seed=seed,
            stack_model_cache=stack_model_cache,
            constraint_cache=constraint_cache,
        )

    graph_key = (config.source_policy, config.max_order, config.backoff_weight)
    if graph_cache is not None and graph_key in graph_cache:
        graph = graph_cache[graph_key]
        context_build_s = 0.0
    else:
        t0 = time.perf_counter()
        graph = build_source_graph(pitches, config)
        context_build_s = time.perf_counter() - t0
        if graph_cache is not None:
            graph_cache[graph_key] = graph

    alphabet = tuple(sorted(graph.alphabet))
    start_context = prefix_context(graph, prefix, config.max_order)
    constraint_key = (
        "generic",
        alphabet,
        config.horizon,
        config.forbidden_ngram,
        config.final_pitch_class,
    )
    if constraint_cache is not None and constraint_key in constraint_cache:
        bundle = constraint_cache[constraint_key]
        if not isinstance(bundle, GenericConstraintBundle):
            raise TypeError("cached generic constraint has unexpected type")
        acceptor = bundle.acceptor
        start_acceptor_state = bundle.start_state
        acceptor_states = bundle.acceptor_states
        acceptor_edges = bundle.acceptor_edges
        acceptor_build_s = 0.0
    else:
        acceptor, start_acceptor_state, acceptor_states, acceptor_edges, acceptor_build_s = build_constraint_acceptor(
            pitches,
            alphabet,
            config.horizon,
            config.forbidden_ngram,
            config.final_pitch_class,
            prefix,
        )
        if constraint_cache is not None:
            constraint_cache[constraint_key] = GenericConstraintBundle(
                acceptor=acceptor,
                start_state=start_acceptor_state,
                acceptor_states=acceptor_states,
                acceptor_edges=acceptor_edges,
            )

    tracemalloc.start()
    t1 = time.perf_counter()
    bp = run_bp(
        graph,
        acceptor,
        length=config.horizon,
        start_context=start_context,
        start_acceptor_state=start_acceptor_state,
    )
    bp_s = time.perf_counter() - t1
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rng = random.Random(seed)
    generated: list[tuple[int, ...]] = []
    generated_orders: list[tuple[int, ...]] = []
    t2 = time.perf_counter()
    if bp.partition_function > 0.0:
        generated_with_orders = bp.sample_many_with_orders(samples, rng=rng)
        generated = [sample for sample, _orders in generated_with_orders]
        generated_orders = [orders for _sample, orders in generated_with_orders]
    sampling_s = time.perf_counter() - t2

    training_windows = {
        length: forbidden_windows(pitches, length)
        for length in range(1, config.horizon + 1)
    }
    copied = [longest_copied_span(sample, training_windows) for sample in generated]
    violations = sum(
        1
        for sample in generated
        if not accepts_from(acceptor, sample, start_acceptor_state)
    )
    order_counts: dict[int, int] = {}
    for orders in generated_orders:
        for order in orders:
            order_counts[order] = order_counts.get(order, 0) + 1
    order_total = sum(order_counts.values())
    selected_order_avg = (
        sum(order * count for order, count in order_counts.items()) / order_total
        if order_total
        else 0.0
    )

    result = BachResult(
        config=config,
        graph=graph,
        acceptor=acceptor,
        start_context=start_context,
        start_acceptor_state=start_acceptor_state,
        prefix=tuple(prefix),
        samples=generated,
        sample_orders=generated_orders,
        context_build_s=context_build_s,
        acceptor_build_s=acceptor_build_s,
        bp_s=bp_s,
        sampling_s=sampling_s,
        peak_memory_mib=peak / (1024 * 1024),
        context_states=len(graph.states),
        context_edges=graph.edge_count(),
        acceptor_states=acceptor_states,
        acceptor_edges=acceptor_edges,
        full_product_edge_upper_bound=acceptor_states * graph.edge_count(),
        dense_lifted_state_count=len(alphabet) ** config.max_order,
        constraint_violations=violations,
        longest_copy_max=max(copied, default=0),
        longest_copy_avg=sum(copied) / len(copied) if copied else 0.0,
        selected_order_avg=selected_order_avg,
        selected_order_hist=tuple(sorted(order_counts.items())),
        partition_function=bp.partition_function,
        mass_kind="partition_function",
    )
    object.__setattr__(result, "_bp_unique_states", bp.unique_product_state_count)
    object.__setattr__(result, "_bp_time_indexed_states", bp.time_indexed_product_state_count)
    object.__setattr__(result, "_bp_edges", bp.product_edge_count)
    return result


def run_policy_stack_configuration(
    pitches: Sequence[int],
    config: BachConfig,
    *,
    prefix: Sequence[int],
    samples: int,
    seed: int,
    stack_model_cache: StackModelCache | None = None,
    constraint_cache: ConstraintCache | None = None,
) -> BachResult:
    if config.max_order < 1:
        raise ValueError("policy_stack requires max_order >= 1")

    if stack_model_cache is not None and config.max_order in stack_model_cache:
        model = stack_model_cache[config.max_order]
        context_build_s = 0.0
    else:
        t0 = time.perf_counter()
        model = OrderStackModel.from_sequences([pitches], max_order=config.max_order)
        context_build_s = time.perf_counter() - t0
        if stack_model_cache is not None:
            stack_model_cache[config.max_order] = model

    alphabet = tuple(sorted(model.alphabet))
    constraint_key = (
        "policy_stack_optimized",
        alphabet,
        config.horizon,
        config.forbidden_ngram,
        config.final_pitch_class,
    )
    if constraint_cache is not None and constraint_key in constraint_cache:
        constraint_bundle = constraint_cache[constraint_key]
        if not isinstance(constraint_bundle, PolicyStackConstraintBundle):
            raise TypeError("cached policy-stack constraint has unexpected type")
        acceptor_build_s = 0.0
    else:
        constraint_bundle, acceptor_build_s = build_policy_stack_optimized_constraint(
            pitches,
            alphabet,
            config.horizon,
            config.forbidden_ngram,
            config.final_pitch_class,
        )
        if constraint_cache is not None:
            constraint_cache[constraint_key] = constraint_bundle

    tracemalloc.start()
    t1 = time.perf_counter()
    bp = run_order_stack_masked_dfa_bp(
        model,
        constraint_bundle.regular_acceptor,
        length=config.horizon,
        prefix=prefix,
        constraints=constraint_bundle.positional_constraints,
        start_acceptor_state=constraint_bundle.regular_start_state,
        policy=LongestFeasiblePolicy(),
    )
    order_start_masses = bp.start_order_masses()
    success_mass = bp.success_mass
    bp_s = time.perf_counter() - t1
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rng = random.Random(seed)
    generated: list[tuple[int, ...]] = []
    generated_orders: list[tuple[int, ...]] = []
    t2 = time.perf_counter()
    if success_mass > 0.0:
        generated_with_orders = bp.sample_many_with_orders(samples, rng=rng)
        generated = [tuple(int(symbol) for symbol in sample) for sample, _orders in generated_with_orders]
        generated_orders = [orders for _sample, orders in generated_with_orders]
    sampling_s = time.perf_counter() - t2

    training_windows = {
        length: forbidden_windows(pitches, length)
        for length in range(1, config.horizon + 1)
    }
    copied = [longest_copied_span(sample, training_windows) for sample in generated]
    violations = sum(
        1
        for sample in generated
        if not accepts_from(
            constraint_bundle.validation_acceptor,
            sample,
            constraint_bundle.validation_start_state,
        )
    )
    order_counts: dict[int, int] = {}
    for orders in generated_orders:
        for order in orders:
            order_counts[order] = order_counts.get(order, 0) + 1
    order_total = sum(order_counts.values())
    selected_order_avg = (
        sum(order * count for order, count in order_counts.items()) / order_total
        if order_total
        else 0.0
    )

    context_states = bp.context_state_count
    context_edges = bp.context_edge_count
    start_context = tuple(prefix[-min(config.max_order, len(prefix)) :])
    result = BachResult(
        config=config,
        graph=None,
        acceptor=constraint_bundle.validation_acceptor,
        start_context=start_context,
        start_acceptor_state=constraint_bundle.validation_start_state,
        prefix=tuple(prefix),
        samples=generated,
        sample_orders=generated_orders,
        context_build_s=context_build_s,
        acceptor_build_s=acceptor_build_s,
        bp_s=bp_s,
        sampling_s=sampling_s,
        peak_memory_mib=peak / (1024 * 1024),
        context_states=context_states,
        context_edges=context_edges,
        acceptor_states=constraint_bundle.acceptor_states,
        acceptor_edges=constraint_bundle.acceptor_edges,
        full_product_edge_upper_bound=constraint_bundle.acceptor_states * context_edges,
        dense_lifted_state_count=len(alphabet) ** config.max_order,
        constraint_violations=violations,
        longest_copy_max=max(copied, default=0),
        longest_copy_avg=sum(copied) / len(copied) if copied else 0.0,
        selected_order_avg=selected_order_avg,
        selected_order_hist=tuple(sorted(order_counts.items())),
        partition_function=success_mass,
        mass_kind="policy_success_mass",
        order_start_masses=order_start_masses,
    )
    object.__setattr__(result, "_bp_unique_states", bp.product_state_count)
    object.__setattr__(result, "_bp_time_indexed_states", bp.time_indexed_product_state_count)
    object.__setattr__(result, "_bp_edges", bp.product_edge_count)
    return result


def build_source_graph(pitches: Sequence[int], config: BachConfig) -> ContextGraph:
    """Build the stochastic source graph for the Bach experiment."""

    if config.source_policy == "pure":
        return ContextGraph.from_sequences([pitches], max_order=config.max_order)
    if config.source_policy == "mixture":
        return ContextGraph.from_backoff_sequences(
            [pitches],
            max_order=config.max_order,
            backoff_weight=config.backoff_weight,
        )
    raise ValueError(
        f"unknown source policy {config.source_policy!r}; expected 'pure', 'mixture', or 'policy_stack'"
    )


def parse_ints(values: list[str]) -> list[int]:
    return [int(value) for value in values]


def print_table(rows: list[dict[str, object]]) -> None:
    columns = [
        "source_policy",
        "K",
        "n",
        "M",
        "alphabet_size",
        "training_length",
        "context_states",
        "context_edges",
        "acceptor_states",
        "acceptor_edges",
        "reachable_product_states",
        "reachable_product_edges",
        "full_product_edge_upper_bound",
        "dense_lifted_state_count",
        "bp_s",
        "sampling_per_sequence_s",
        "partition_function",
        "mass_kind",
        "constraint_violations",
        "longest_copy_avg",
        "selected_order_avg",
        "order_start_masses",
    ]
    print(",".join(columns))
    for row in rows:
        print(",".join(str(row[column]) for column in columns))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _raw_output_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_raw{output.suffix}")


def _config_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["source_policy"],
        row["K"],
        row["n"],
        row["M"],
        row["final_pitch_class"],
        row["backoff_weight"],
    )


def median_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    median_fields = {
        "context_build_s",
        "acceptor_build_s",
        "bp_s",
        "sampling_s",
        "sampling_per_sequence_s",
        "sampling_per_event_s",
        "peak_memory_mib",
        "partition_function",
        "longest_copy_avg",
        "selected_order_avg",
    }
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault(_config_key(row), []).append(row)

    rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        row = dict(group[0])
        for field in median_fields:
            row[field] = f"{statistics.median(float(item[field]) for item in group):.12g}"
        row["label"] = "median"
        row["repeat"] = "median"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--orders", nargs="+", default=["1", "2", "3", "4", "5", "6"])
    parser.add_argument("--horizons", nargs="+", default=["32"])
    parser.add_argument("--maxorder-grams", nargs="+", default=["5"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-pitch-class", type=int, default=0)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument("--backoff-weight", type=float, default=0.25)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument(
        "--source-policy",
        choices=("policy_stack", "pure", "mixture", "both", "all"),
        default="policy_stack",
        help=(
            "'policy_stack' uses constrained longest-feasible order-stack backoff; "
            "'pure' uses one fixed longest observed suffix MLE source; "
            "'mixture' uses the smoothed suffix-mixture source; "
            "'both' runs pure+mixture; 'all' runs policy_stack+pure+mixture."
        ),
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")

    pitches = load_bach_pitches(args.data)
    if len(pitches) < args.prefix_length:
        raise ValueError("prefix length exceeds pitch sequence length")
    prefix = tuple(pitches[: args.prefix_length])

    raw_rows: list[dict[str, object]] = []
    results: list[BachResult] = []
    graph_cache: GraphCache = {}
    stack_model_cache: StackModelCache = {}
    constraint_cache: ConstraintCache = {}
    if args.source_policy == "both":
        source_policies = ["pure", "mixture"]
    elif args.source_policy == "all":
        source_policies = ["policy_stack", "pure", "mixture"]
    else:
        source_policies = [args.source_policy]
    configs = [
        BachConfig(
            order,
            horizon,
            maxorder_gram,
            args.final_pitch_class,
            args.backoff_weight,
            source_policy,
        )
        for source_policy in source_policies
        for order in parse_ints(args.orders)
        for horizon in parse_ints(args.horizons)
        for maxorder_gram in parse_ints(args.maxorder_grams)
    ]

    for repeat in range(-args.warmups, args.repeats):
        measured = repeat >= 0
        for index, config in enumerate(configs):
            seed_offset = (repeat if measured else 100_000 - repeat) * len(configs) + index
            result = run_configuration(
                pitches,
                config,
                prefix=prefix,
                samples=args.samples,
                seed=args.seed + seed_offset,
                graph_cache=graph_cache,
                stack_model_cache=stack_model_cache,
                constraint_cache=constraint_cache,
            )
            if measured:
                results.append(result)
                row = result.row(len(pitches), len(set(pitches)), args.samples)
                row["repeat"] = repeat
                row["label"] = f"repeat_{repeat}"
                raw_rows.append(row)

    rows = raw_rows if args.repeats == 1 else median_rows(raw_rows)
    print_table(rows)
    write_csv(args.output, rows)
    if args.repeats > 1 or args.warmups:
        raw_output = args.raw_output or _raw_output_path(args.output)
        write_csv(raw_output, raw_rows)
        print(f"\nRaw CSV written to {raw_output}")
    print(f"\nCSV written to {args.output}")

    representative = next(
        (
            result
            for result in results
            if result.config.max_order == 4
            and result.config.horizon == 32
            and result.config.forbidden_ngram == 5
            and result.samples
        ),
        next((result for result in results if result.samples), None),
    )
    if representative is not None:
        config = representative.config
        print(f"\nExample generated sequences for K={config.max_order}, n={config.horizon}, M={config.forbidden_ngram}:")
        for sample, orders in zip(representative.samples[:5], representative.sample_orders[:5]):
            print(" ".join(map(str, sample)))
            print("orders:", " ".join(map(str, orders)))


if __name__ == "__main__":
    main()
