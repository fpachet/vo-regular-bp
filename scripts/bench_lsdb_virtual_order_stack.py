"""LSDB-shaped benchmark for virtual order-stack regular BP preparation.

The benchmark is intentionally independent from LSDB, but mirrors the workload
that matters for FlowComposer-style melody generation:

* pitch-duration string tokens such as ``"60@0.5"``;
* virtual chromatic transposition;
* max_order 4;
* a length-32, total-duration regular constraint;
* previous-pitch state and optional transition weights in the same DFA.
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import pstats
import random
import sys
import time
from typing import Hashable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vo_regular_bp import (  # noqa: E402
    ConstraintSet,
    DFA,
    LongestFeasiblePolicy,
    OrderStackModel,
    SymbolTransform,
    VirtualAugmentedOrderStackModel,
    prepare_constrained_order_stack,
)


START = "<START>"
REST = "REST"


@dataclass(frozen=True)
class BenchResult:
    build_seconds: float
    prepare_seconds: float
    diagnostics: dict[str, object]


@lru_cache(maxsize=500_000)
def parse_token(symbol: object) -> tuple[int | None, float]:
    text = str(symbol)
    if "@" not in text:
        raise ValueError(f"expected pitch-duration token, got {symbol!r}")
    pitch_text, duration_text = text.split("@", 1)
    pitch = None if pitch_text == REST else int(pitch_text)
    return pitch, float(duration_text)


def make_token(pitch: int | None, duration: float) -> str:
    label = REST if pitch is None else str(int(pitch))
    return f"{label}@{duration:g}"


@lru_cache(maxsize=500_000)
def transpose_token(symbol: Hashable, offset: int) -> Hashable:
    if symbol == START:
        return symbol
    if not isinstance(symbol, str) or "@" not in symbol:
        return symbol
    pitch, duration = parse_token(symbol)
    if pitch is None:
        return symbol
    return make_token(pitch + int(offset), duration)


def transposition_transforms(offsets: tuple[int, ...]) -> tuple[SymbolTransform, ...]:
    return tuple(
        SymbolTransform(
            name=f"transpose_{offset:+d}",
            apply_symbol=lambda symbol, offset=offset: transpose_token(symbol, offset),
            inverse_symbol=lambda symbol, offset=offset: transpose_token(symbol, -offset),
        )
        for offset in offsets
    )


def make_sequences(
    *,
    tracks: int,
    sequence_length: int,
    seed: int,
) -> list[tuple[str, ...]]:
    rng = random.Random(seed)
    durations = (0.25, 0.5, 0.5, 0.5, 0.75, 1.0)
    scale = (0, 2, 4, 5, 7, 9, 11, 12)
    sequences: list[tuple[str, ...]] = []
    for track in range(tracks):
        tonic = 54 + (track % 12)
        contour = rng.choice((-1, 1))
        pitch_index = (track * 3) % len(scale)
        tokens: list[str] = []
        for step in range(sequence_length):
            duration = durations[(step + track * 2 + (step // 7)) % len(durations)]
            if (step + track) % 23 == 0:
                tokens.append(make_token(None, duration))
                continue
            if step % 9 == 0:
                contour *= -1
            pitch_index = (pitch_index + contour + rng.choice((-1, 0, 1))) % len(scale)
            octave = 12 * (((step // 16) + track) % 2)
            pitch = tonic + scale[pitch_index] + octave
            tokens.append(make_token(pitch, duration))
        sequences.append(tuple(tokens))
    return sequences


def build_integrated_melody_dfa(
    *,
    length: int,
    total_beats: float,
    phase_quantum: float,
    max_leap: int,
    min_pitch: int,
    max_pitch: int,
    weighted: bool,
) -> DFA:
    target = int(round(total_beats / phase_quantum))
    min_units = 1
    max_units = int(round(1.0 / phase_quantum))
    no_previous = -1

    def duration_units(symbol: str) -> int:
        _pitch, duration = parse_token(symbol)
        return max(1, int(round(duration / phase_quantum)))

    def transition(state: object, symbol: str) -> object | None:
        position, current_total, previous_pitch = state  # type: ignore[misc]
        if position >= length:
            return None
        units = duration_units(symbol)
        next_total = current_total + units
        if next_total > target:
            return None
        remaining = length - position - 1
        if next_total + remaining * min_units > target:
            return None
        if next_total + remaining * max_units < target:
            return None

        pitch, _duration = parse_token(symbol)
        if pitch is None:
            return (position + 1, next_total, previous_pitch)
        if pitch < min_pitch or pitch > max_pitch:
            return None
        if previous_pitch != no_previous and abs(pitch - previous_pitch) > max_leap:
            return None
        return (position + 1, next_total, int(pitch))

    def accepting(state: object) -> bool:
        position, current_total, _previous_pitch = state  # type: ignore[misc]
        return position == length and current_total == target

    def weight(state: object, symbol: str) -> float:
        _position, _current_total, previous_pitch = state  # type: ignore[misc]
        pitch, _duration = parse_token(symbol)
        if pitch is None:
            return 0.35
        score = math.exp(-abs(pitch - 66) * 0.015)
        if previous_pitch != no_previous:
            leap = abs(int(pitch) - int(previous_pitch))
            score *= math.exp(-max(0, leap - 2) * 0.08)
        return score

    return DFA(
        start_state=(0, 0, no_previous),
        transition_func=transition,
        transition_weight_func=weight if weighted else None,
        accept_func=accepting,
        name="bench_integrated_melody",
    )


def run_once(args: argparse.Namespace, sequences: list[tuple[str, ...]]) -> BenchResult:
    started = time.perf_counter()
    if args.transforms:
        offsets = tuple(range(args.transforms))
        model: OrderStackModel | VirtualAugmentedOrderStackModel = (
            VirtualAugmentedOrderStackModel.from_sequences(
                sequences,
                max_order=args.max_order,
                transforms=transposition_transforms(offsets),
                start_symbol=START,
            )
        )
    else:
        model = OrderStackModel.from_sequences(
            sequences,
            max_order=args.max_order,
            start_symbol=START,
        )
    build_seconds = time.perf_counter() - started

    acceptor = build_integrated_melody_dfa(
        length=args.length,
        total_beats=args.total_beats,
        phase_quantum=args.phase_quantum,
        max_leap=args.max_leap,
        min_pitch=args.min_pitch,
        max_pitch=args.max_pitch,
        weighted=args.weighted,
    )
    constraints = ConstraintSet(regular_acceptors=(acceptor,))

    started = time.perf_counter()
    backend = prepare_constrained_order_stack(
        model,
        constraints,
        length=args.length,
        prefix=(START,),
        policy=LongestFeasiblePolicy(),
    )
    prepare_seconds = time.perf_counter() - started
    diagnostics = dict(backend.diagnostics.as_dict())
    diagnostics.update(_extra_diagnostics(model, backend.result))
    return BenchResult(
        build_seconds=build_seconds,
        prepare_seconds=prepare_seconds,
        diagnostics=diagnostics,
    )


def _extra_diagnostics(model: object, result: object) -> dict[str, object]:
    extra: dict[str, object] = {}
    model_diagnostics = getattr(model, "virtual_diagnostics", None)
    if callable(model_diagnostics):
        extra.update(model_diagnostics())
    graphs = getattr(result, "graphs", {})
    extra["virtual_outgoing_row_calls"] = sum(
        getattr(graph, "outgoing_row_calls", 0) for graph in graphs.values()
    )
    extra["virtual_outgoing_row_cache_hits"] = sum(
        getattr(graph, "outgoing_row_cache_hits", 0) for graph in graphs.values()
    )
    extra["virtual_outgoing_row_cache_misses"] = sum(
        getattr(graph, "outgoing_row_cache_misses", 0) for graph in graphs.values()
    )
    for name in (
        "regular_transition_row_count",
        "regular_transition_row_cache_hits",
        "regular_transition_row_cache_misses",
        "regular_accepted_transition_count",
    ):
        value = getattr(result, name, None)
        if value is not None:
            extra[name] = value
    return extra


def print_result(label: str, result: BenchResult) -> None:
    diagnostics = result.diagnostics
    print(
        f"{label}: build={result.build_seconds:.4f}s "
        f"prepare={result.prepare_seconds:.4f}s "
        f"context_states={diagnostics.get('context_states')} "
        f"context_edges={diagnostics.get('context_edges')} "
        f"product_states={diagnostics.get('regular_product_states')} "
        f"time_indexed={diagnostics.get('regular_product_states_time_indexed')} "
        f"product_edges={diagnostics.get('regular_product_edges')} "
        "transition_rows="
        f"{diagnostics.get('regular_transition_rows', diagnostics.get('regular_transition_row_count'))}"
    )
    interesting = (
        "virtual_context_materialization_seconds",
        "virtual_context_materialization_calls",
        "virtual_context_materialization_cache_hits",
        "virtual_context_materialization_cache_misses",
        "augmented_count_calls",
        "augmented_count_cache_hits",
        "augmented_count_cache_misses",
        "virtual_outgoing_row_calls",
        "virtual_outgoing_row_cache_hits",
        "virtual_outgoing_row_cache_misses",
        "regular_transition_row_cache_hits",
        "regular_transition_row_cache_misses",
        "regular_acceptor_symbol_transition_cache_hits",
        "regular_acceptor_symbol_transition_cache_misses",
        "regular_accepted_transition_count",
        "regular_beta_state_expansions",
        "regular_beta_cache_hits",
        "regular_beta_cache_misses",
    )
    for key in interesting:
        if key in diagnostics:
            print(f"  {key}={diagnostics[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=int, default=120)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--total-beats", type=float, default=16.0)
    parser.add_argument("--phase-quantum", type=float, default=0.25)
    parser.add_argument("--transforms", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-leap", type=int, default=12)
    parser.add_argument("--min-pitch", type=int, default=36)
    parser.add_argument("--max-pitch", type=int, default=96)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-limit", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequences = make_sequences(
        tracks=args.tracks,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    print(
        f"tracks={args.tracks} sequence_length={args.sequence_length} "
        f"length={args.length} max_order={args.max_order} "
        f"transforms={args.transforms} weighted={args.weighted}"
    )
    if args.profile:
        profiler = cProfile.Profile()
        result = profiler.runcall(run_once, args, sequences)
        print_result("profiled", result)
        stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
        stats.print_stats(args.profile_limit)
        return 0

    results = [run_once(args, sequences) for _ in range(args.repeats)]
    for index, result in enumerate(results, start=1):
        print_result(f"run {index}", result)
    if len(results) > 1:
        mean_prepare = sum(result.prepare_seconds for result in results) / len(results)
        print(f"mean_prepare={mean_prepare:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
