#!/usr/bin/env python
"""Direct positional-constraint benchmark on Bach Prelude pitches.

This benchmark intentionally avoids a regular-acceptor product: positional
constraints are applied as time-indexed masks on context-graph edges.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Callable, Hashable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_bach_scalability import DATA_PATH, load_bach_pitches, prefix_context  # noqa: E402
from vo_regular_bp import ContextGraph  # noqa: E402
from vo_regular_bp.context import Context, Edge, Symbol  # noqa: E402

PositionPredicate = Callable[[Symbol], bool]


class LazyBackoffContextModel:
    """Backoff continuation model with outgoing edges compiled on demand."""

    def __init__(
        self,
        counts: dict[Context, Counter[Symbol]],
        *,
        max_order: int,
        backoff_weight: float,
        start_state: Sequence[Symbol] = (),
    ) -> None:
        if max_order < 0:
            raise ValueError("max_order must be non-negative")
        if not 0.0 <= backoff_weight <= 1.0:
            raise ValueError("backoff_weight must be in [0, 1]")
        self.counts = counts
        self.max_order = int(max_order)
        self.backoff_weight = float(backoff_weight)
        self.start_state = tuple(start_state)
        contexts = set(counts)
        contexts.add(self.start_state)
        self.contexts = frozenset(contexts)
        self.alphabet = frozenset(symbol for counter in counts.values() for symbol in counter)
        self.distributions = {
            context: tuple((symbol, float(count) / total) for symbol, count in counter.items())
            for context, counter in counts.items()
            for total in (float(sum(counter.values())),)
            if total > 0.0
        }
        self._outgoing_cache: dict[Context, tuple[Edge, ...]] = {}

    @classmethod
    def from_sequences(
        cls,
        sequences: Sequence[Sequence[Symbol]],
        *,
        max_order: int,
        backoff_weight: float,
        start_state: Sequence[Symbol] = (),
    ) -> "LazyBackoffContextModel":
        counts: dict[Context, Counter[Symbol]] = defaultdict(Counter)
        for sequence in sequences:
            tokens = tuple(sequence)
            for index, symbol in enumerate(tokens):
                order_limit = min(max_order, index)
                for order in range(order_limit + 1):
                    context = tokens[index - order : index] if order else ()
                    counts[context][symbol] += 1
        return cls(
            dict(counts),
            max_order=max_order,
            backoff_weight=backoff_weight,
            start_state=start_state,
        )

    def outgoing(self, context: Sequence[Symbol]) -> tuple[Edge, ...]:
        state = tuple(context)
        cached = self._outgoing_cache.get(state)
        if cached is not None:
            return cached

        scores: dict[Symbol, float] = {}
        context_order = len(state)
        distributions = self.distributions
        backoff_weight = self.backoff_weight
        for order in range(context_order, -1, -1):
            suffix = state[-order:] if order else ()
            distribution = distributions.get(suffix)
            if not distribution:
                continue
            weight = backoff_weight ** (context_order - order)
            for symbol, probability in distribution:
                scores[symbol] = scores.get(symbol, 0.0) + weight * probability

        total_score = float(sum(scores.values()))
        if total_score <= 0.0:
            edges: tuple[Edge, ...] = ()
        else:
            edges = tuple(
                Edge(
                    symbol,
                    score / total_score,
                    ContextGraph._canon(state, symbol, self.contexts, self.max_order),
                )
                for symbol, score in scores.items()
                if score > 0.0
            )
        self._outgoing_cache[state] = edges
        return edges

    def prefix_context(self, prefix: Sequence[Symbol]) -> Context:
        for order in range(min(self.max_order, len(prefix)), -1, -1):
            suffix = tuple(prefix[-order:]) if order else ()
            if suffix in self.contexts:
                return suffix
        return self.start_state

    @property
    def materialized_edge_count(self) -> int:
        return sum(len(edges) for edges in self._outgoing_cache.values())


@dataclass
class DirectPositionalBPResult:
    graph: ContextGraph
    length: int
    start_context: Context
    layers: list[set[Context]]
    edges: list[dict[Context, tuple[Edge, ...]]]
    betas: list[dict[Context, float]]

    @property
    def partition_function(self) -> float:
        return self.betas[0].get(self.start_context, 0.0)

    @property
    def unique_state_count(self) -> int:
        return len(set().union(*self.layers)) if self.layers else 0

    @property
    def time_indexed_state_count(self) -> int:
        return sum(len(layer) for layer in self.layers)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for layer in self.edges for edges in layer.values())

    def sample(self, rng: random.Random) -> tuple[Symbol, ...]:
        if self.partition_function <= 0.0:
            raise ValueError("cannot sample because the constrained partition function is zero")

        context = self.start_context
        output: list[Symbol] = []
        for time in range(self.length):
            weighted = []
            for edge in self.edges[time].get(context, ()):
                weight = edge.probability * self.betas[time + 1].get(edge.next_state, 0.0)
                if weight > 0.0:
                    weighted.append((edge, weight))
            total = sum(weight for _, weight in weighted)
            if total <= 0.0:
                raise RuntimeError("no positive continuation in BP table")
            threshold = rng.random() * total
            cumulative = 0.0
            chosen = weighted[-1][0]
            for edge, weight in weighted:
                cumulative += weight
                if threshold <= cumulative:
                    chosen = edge
                    break
            output.append(chosen.symbol)
            context = chosen.next_state
        return tuple(output)


@dataclass
class MemoDirectPositionalBPResult:
    model: LazyBackoffContextModel
    length: int
    start_context: Context
    constraints: dict[int, PositionPredicate]
    betas: list[dict[Context, float]]
    allowed_edge_counts: list[dict[Context, int]]

    @property
    def partition_function(self) -> float:
        return self.betas[0].get(self.start_context, 0.0)

    @property
    def unique_state_count(self) -> int:
        return len(set().union(*self.betas)) if self.betas else 0

    @property
    def time_indexed_state_count(self) -> int:
        return sum(len(layer) for layer in self.betas)

    @property
    def edge_count(self) -> int:
        return sum(sum(layer.values()) for layer in self.allowed_edge_counts)

    def sample(self, rng: random.Random) -> tuple[Symbol, ...]:
        if self.partition_function <= 0.0:
            raise ValueError("cannot sample because the constrained partition function is zero")

        context = self.start_context
        output: list[Symbol] = []
        for time in range(self.length):
            predicate = self.constraints.get(time)
            weighted = []
            beta_next = self.betas[time + 1]
            for edge in self.model.outgoing(context):
                if predicate is not None and not predicate(edge.symbol):
                    continue
                weight = edge.probability * beta_next.get(edge.next_state, 0.0)
                if weight > 0.0:
                    weighted.append((edge.symbol, edge.next_state, weight))
            total = sum(weight for _, _, weight in weighted)
            if total <= 0.0:
                raise RuntimeError("no positive continuation in BP table")
            threshold = rng.random() * total
            cumulative = 0.0
            chosen_symbol, chosen_context, _ = weighted[-1]
            for symbol, next_context, weight in weighted:
                cumulative += weight
                if threshold <= cumulative:
                    chosen_symbol = symbol
                    chosen_context = next_context
                    break
            output.append(chosen_symbol)
            context = chosen_context
        return tuple(output)


def run_direct_positional_bp(
    graph: ContextGraph,
    *,
    length: int,
    start_context: Sequence[Symbol],
    constraints: dict[int, PositionPredicate],
) -> DirectPositionalBPResult:
    context0 = tuple(start_context)
    layers: list[set[Context]] = [{context0}]
    edges_by_time: list[dict[Context, tuple[Edge, ...]]] = []

    for time in range(length):
        predicate = constraints.get(time)
        current_edges: dict[Context, tuple[Edge, ...]] = {}
        next_layer: set[Context] = set()
        for context in layers[-1]:
            outgoing = []
            for edge in graph.outgoing(context):
                if predicate is not None and not predicate(edge.symbol):
                    continue
                outgoing.append(edge)
                next_layer.add(edge.next_state)
            current_edges[context] = tuple(outgoing)
        edges_by_time.append(current_edges)
        layers.append(next_layer)

    betas: list[dict[Context, float]] = [dict() for _ in range(length + 1)]
    betas[length] = {context: 1.0 for context in layers[length]}
    for time in range(length - 1, -1, -1):
        beta_next = betas[time + 1]
        betas[time] = {
            context: sum(
                edge.probability * beta_next.get(edge.next_state, 0.0)
                for edge in edges_by_time[time].get(context, ())
            )
            for context in layers[time]
        }

    return DirectPositionalBPResult(
        graph=graph,
        length=length,
        start_context=context0,
        layers=layers,
        edges=edges_by_time,
        betas=betas,
    )


def run_memo_lazy_direct_positional_bp(
    model: LazyBackoffContextModel,
    *,
    length: int,
    start_context: Sequence[Symbol],
    constraints: dict[int, PositionPredicate],
) -> MemoDirectPositionalBPResult:
    context0 = tuple(start_context)
    betas: list[dict[Context, float]] = [dict() for _ in range(length + 1)]
    allowed_edge_counts: list[dict[Context, int]] = [dict() for _ in range(length)]

    def beta(time_index: int, context: Context) -> float:
        cached = betas[time_index].get(context)
        if cached is not None:
            return cached
        if time_index == length:
            betas[time_index][context] = 1.0
            return 1.0

        predicate = constraints.get(time_index)
        total = 0.0
        allowed_count = 0
        for edge in model.outgoing(context):
            if predicate is not None and not predicate(edge.symbol):
                continue
            allowed_count += 1
            total += edge.probability * beta(time_index + 1, edge.next_state)

        allowed_edge_counts[time_index][context] = allowed_count
        betas[time_index][context] = total
        return total

    beta(0, context0)
    return MemoDirectPositionalBPResult(
        model=model,
        length=length,
        start_context=context0,
        constraints=constraints,
        betas=betas,
        allowed_edge_counts=allowed_edge_counts,
    )


def timed_once_baseline(args: argparse.Namespace, seed: int) -> dict[str, float | int | tuple[Hashable, ...] | str]:
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    pitches = load_bach_pitches(args.data)
    parse_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    graph = ContextGraph.from_backoff_sequences(
        [pitches],
        max_order=args.max_order,
        backoff_weight=args.backoff_weight,
    )
    graph_s = time.perf_counter() - t1

    prefix = tuple(pitches[: args.prefix_length])
    start_context = prefix_context(graph, prefix, args.max_order)
    is_pc_c = lambda pitch: int(pitch) % 12 == args.pitch_class
    constraints = {0: is_pc_c, args.horizon - 1: is_pc_c}

    t2 = time.perf_counter()
    bp = run_direct_positional_bp(
        graph,
        length=args.horizon,
        start_context=start_context,
        constraints=constraints,
    )
    bp_s = time.perf_counter() - t2

    rng = random.Random(seed)
    violations = 0
    t3 = time.perf_counter()
    for _ in range(args.samples):
        sample = bp.sample(rng)
        if not sample or sample[0] % 12 != args.pitch_class or sample[-1] % 12 != args.pitch_class:
            violations += 1
    sample_s = time.perf_counter() - t3

    total_s = time.perf_counter() - total_start
    return {
        "engine": "baseline",
        "parse_s": parse_s,
        "graph_s": graph_s,
        "bp_s": bp_s,
        "sample_s": sample_s,
        "total_s": total_s,
        "partition_function": bp.partition_function,
        "context_states": len(graph.states),
        "context_edges": graph.edge_count(),
        "reachable_states": bp.unique_state_count,
        "reachable_states_time_indexed": bp.time_indexed_state_count,
        "reachable_edges": bp.edge_count,
        "violations": violations,
        "prefix": prefix,
        "start_context": start_context,
    }


def timed_once_optimized(args: argparse.Namespace, seed: int) -> dict[str, float | int | tuple[Hashable, ...] | str]:
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    pitches = load_bach_pitches(args.data)
    parse_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = LazyBackoffContextModel.from_sequences(
        [pitches],
        max_order=args.max_order,
        backoff_weight=args.backoff_weight,
    )
    graph_s = time.perf_counter() - t1

    prefix = tuple(pitches[: args.prefix_length])
    start_context = model.prefix_context(prefix)
    is_pc_c = lambda pitch: int(pitch) % 12 == args.pitch_class
    constraints = {0: is_pc_c, args.horizon - 1: is_pc_c}

    t2 = time.perf_counter()
    bp = run_memo_lazy_direct_positional_bp(
        model,
        length=args.horizon,
        start_context=start_context,
        constraints=constraints,
    )
    bp_s = time.perf_counter() - t2

    rng = random.Random(seed)
    violations = 0
    t3 = time.perf_counter()
    for _ in range(args.samples):
        sample = bp.sample(rng)
        if not sample or sample[0] % 12 != args.pitch_class or sample[-1] % 12 != args.pitch_class:
            violations += 1
    sample_s = time.perf_counter() - t3

    total_s = time.perf_counter() - total_start
    return {
        "engine": "optimized",
        "parse_s": parse_s,
        "graph_s": graph_s,
        "bp_s": bp_s,
        "sample_s": sample_s,
        "total_s": total_s,
        "partition_function": bp.partition_function,
        "context_states": len(model.contexts),
        "context_edges": model.materialized_edge_count,
        "reachable_states": bp.unique_state_count,
        "reachable_states_time_indexed": bp.time_indexed_state_count,
        "reachable_edges": bp.edge_count,
        "violations": violations,
        "prefix": prefix,
        "start_context": start_context,
    }


def timed_once(args: argparse.Namespace, seed: int) -> dict[str, float | int | tuple[Hashable, ...]]:
    if args.engine == "baseline":
        return timed_once_baseline(args, seed)
    return timed_once_optimized(args, seed)


def print_results(args: argparse.Namespace, results: list[dict[str, float | int | tuple[Hashable, ...] | str]]) -> None:
    first = results[0]
    phases = ["parse_s", "graph_s", "bp_s", "sample_s", "total_s"]

    print("Bach positional-only direct BP benchmark")
    print(f"engine={first['engine']}")
    print(f"K={args.max_order} n={args.horizon} samples_per_repeat={args.samples} repeats={len(results)}")
    print(f"constraint: generated first and last pitch class = {args.pitch_class}")
    print(f"prefix: {first['prefix']}")
    print(f"start_context: {first['start_context']}")
    print(f"partition_function: {float(first['partition_function']):.12g}")
    print(f"context_states: {first['context_states']}")
    print(f"context_edges: {first['context_edges']}")
    print(f"reachable_states: {first['reachable_states']}")
    print(f"reachable_states_time_indexed: {first['reachable_states_time_indexed']}")
    print(f"reachable_edges: {first['reachable_edges']}")
    print(f"violations_total: {sum(int(result['violations']) for result in results)}")
    print()
    print("phase,mean_s,stdev_s")
    for phase in phases:
        values = [float(result[phase]) for result in results]
        print(f"{phase},{mean(values):.6f},{stdev(values):.6f}")


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument("--pitch-class", type=int, default=0)
    parser.add_argument("--backoff-weight", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engine", choices=("optimized", "baseline"), default="optimized")
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args()

    if args.compare_baseline:
        for engine in ("baseline", "optimized"):
            args.engine = engine
            results = [timed_once(args, args.seed + repeat) for repeat in range(args.repeats)]
            print_results(args, results)
            print()
        return

    results = [timed_once(args, args.seed + repeat) for repeat in range(args.repeats)]
    print_results(args, results)


if __name__ == "__main__":
    main()
