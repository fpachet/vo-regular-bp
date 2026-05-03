#!/usr/bin/env python
"""Diagnose transposition-orbit repetition in virtual augmentation products."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Callable, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import vo_regular_bp as vbp  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "bach_prelude_c_major_pitches.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "virtual_product_orbits"
START_SYMBOL = "<START>"
DEFAULT_ASCENDING_ANCHOR_OFFSETS = (0, 2, 4, 5, 7, 9, 11, 12)
T = TypeVar("T")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--maxorder-grams", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--offsets", type=int, nargs="+", default=list(range(-6, 6)))
    parser.add_argument(
        "--constraint-mode",
        choices=("ascending_anchors", "final_pitch_class"),
        default="ascending_anchors",
    )
    parser.add_argument("--anchor-period", type=int, default=8)
    parser.add_argument("--final-pitch-class", type=int, default=0)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument(
        "--include-row-shapes",
        action="store_true",
        help="also canonicalize full outgoing row shapes; slower but more detailed",
    )
    args = parser.parse_args()

    pitches = load_pitches(args.data)
    transforms = vbp.integer_shift_transforms(args.offsets, fixed_symbols={START_SYMBOL})
    augmented_sequences = vbp.materialize_transformed_sequences([pitches], transforms)

    if args.constraint_mode == "ascending_anchors":
        prefix = (START_SYMBOL,)
        model = vbp.VirtualAugmentedOrderStackModel.from_sequences(
            [pitches],
            max_order=args.max_order,
            transforms=transforms,
            start_symbol=START_SYMBOL,
        )
        positional_constraints = ascending_anchor_constraints(
            horizon=args.horizon,
            period=args.anchor_period,
            start_pitch=pitches[0],
        )
    else:
        prefix = tuple(pitches[: args.prefix_length])
        model = vbp.VirtualAugmentedOrderStackModel.from_sequences(
            [pitches],
            max_order=args.max_order,
            transforms=transforms,
        )
        positional_constraints = {
            args.horizon - 1: {
                symbol
                for symbol in model.alphabet
                if int(symbol) % 12 == args.final_pitch_class
            }
        }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for forbidden_len in args.maxorder_grams:
        row = run_case(
            pitches=pitches,
            augmented_sequences=augmented_sequences,
            model=model,
            prefix=prefix,
            positional_constraints=positional_constraints,
            max_order=args.max_order,
            horizon=args.horizon,
            forbidden_len=forbidden_len,
            offsets=tuple(args.offsets),
            constraint_mode=args.constraint_mode,
            include_row_shapes=args.include_row_shapes,
        )
        rows.append(row)
        print_summary(row)

    output_path = output_dir / "virtual_product_orbit_diagnostics.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"output={output_path}")


def run_case(
    *,
    pitches: tuple[int, ...],
    augmented_sequences: tuple[tuple[int, ...], ...],
    model: vbp.VirtualAugmentedOrderStackModel,
    prefix: tuple[object, ...],
    positional_constraints: dict[int, object],
    max_order: int,
    horizon: int,
    forbidden_len: int,
    offsets: tuple[int, ...],
    constraint_mode: str,
    include_row_shapes: bool,
) -> dict[str, object]:
    forbidden = forbidden_windows(augmented_sequences, forbidden_len)
    alphabet = tuple(sorted(model.alphabet))
    pattern_stats = vbp.forbidden_pattern_orbit_stats(forbidden, alphabet=alphabet)
    acceptor = vbp.dense_forbidden_substring_acceptor(forbidden, alphabet=alphabet)

    bp, bp_s = timed(
        lambda: vbp.run_order_stack_masked_dfa_bp(
            model,
            acceptor,
            length=horizon,
            prefix=prefix,
            constraints=positional_constraints,
            policy=vbp.LongestFeasiblePolicy(),
        )
    )
    bp_transition_row_count = bp.regular_transition_row_count
    bp_accepted_transition_count = bp.regular_accepted_transition_count
    bp_transition_row_cache_hits = bp.regular_transition_row_cache_hits
    bp_transition_row_cache_misses = bp.regular_transition_row_cache_misses
    orbit_stats, orbit_s = timed(
        lambda: vbp.regular_product_orbit_stats(
            bp,
            fixed_symbols={START_SYMBOL},
            include_transition_row_shapes=include_row_shapes,
        )
    )
    row_signature_stats, row_signature_s = timed(
        lambda: vbp.regular_row_signature_stats(
            bp,
            fixed_symbols={START_SYMBOL},
        )
    )

    row: dict[str, object] = {
        "constraint_mode": constraint_mode,
        "training_tokens": len(pitches),
        "virtual_tokens": len(pitches) * len(offsets),
        "offsets": " ".join(str(offset) for offset in offsets),
        "transform_count": len(offsets),
        "max_order": max_order,
        "horizon": horizon,
        "copy_ngram": forbidden_len,
        "alphabet_size": len(alphabet),
        "context_states": bp.context_state_count,
        "context_edges": bp.context_edge_count,
        "success_mass": f"{bp.success_mass:.12g}",
        "bp_s": f"{bp_s:.6f}",
        "orbit_diagnostic_s": f"{orbit_s:.6f}",
        "row_signature_diagnostic_s": f"{row_signature_s:.6f}",
        "transition_rows": bp_transition_row_count,
        "accepted_transitions_cached": bp_accepted_transition_count,
        "transition_row_cache_hits": bp_transition_row_cache_hits,
        "transition_row_cache_misses": bp_transition_row_cache_misses,
    }
    row.update(format_stats(pattern_stats.as_dict()))
    row.update(format_stats(orbit_stats.as_dict()))
    row.update(format_stats(row_signature_stats.as_dict()))
    return row


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def load_pitches(path: Path) -> tuple[int, ...]:
    pitches: list[int] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line:
            pitches.extend(int(token) for token in line.split())
    if not pitches:
        raise ValueError(f"no pitches found in {path}")
    return tuple(pitches)


def forbidden_windows(
    sequences: tuple[tuple[int, ...], ...],
    length: int,
) -> tuple[tuple[int, ...], ...]:
    if length <= 0:
        raise ValueError("forbidden window length must be positive")
    return tuple(
        dict.fromkeys(
            tuple(sequence[index : index + length])
            for sequence in sequences
            for index in range(0, len(sequence) - length + 1)
        )
    )


def ascending_anchor_constraints(
    *,
    horizon: int,
    period: int,
    start_pitch: int,
) -> dict[int, set[int]]:
    anchors = tuple(range(0, horizon, period))
    anchor_pitches = tuple(
        int(start_pitch) + offset
        for offset in DEFAULT_ASCENDING_ANCHOR_OFFSETS[: len(anchors)]
    )
    if len(anchor_pitches) != len(anchors):
        raise ValueError("not enough default anchor pitches for the requested horizon")
    return {
        position: {pitch}
        for position, pitch in zip(anchors, anchor_pitches)
    }


def format_stats(stats: dict[str, object]) -> dict[str, object]:
    formatted: dict[str, object] = {}
    for key, value in stats.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.6f}"
        else:
            formatted[key] = value
    return formatted


def print_summary(row: dict[str, object]) -> None:
    print(
        f"copy_ngram={row['copy_ngram']}",
        f"bp_s={row['bp_s']}",
        f"product_edges={row['product_edges']}",
        f"time_edge_orbits={row['time_indexed_product_edge_orbits']}",
        f"time_edge_reduction={row['time_indexed_product_edge_reduction']}",
        f"edge_orbits={row['product_edge_orbits']}",
        f"edge_reduction={row['product_edge_reduction']}",
        f"transition_rows={row['transition_rows']}",
        f"exact_row_reduction={row['exact_product_row_reduction']}",
        f"exact_time_masked_row_reduction={row['exact_time_masked_row_reduction']}",
        f"prefix_state_reduction={row['dfa_prefix_state_reduction']}",
    )


if __name__ == "__main__":
    main()
