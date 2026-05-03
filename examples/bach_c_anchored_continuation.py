"""Bach pitch-only Continuator example with bar anchors and anti-copy constraints.

This example uses the static Bach Prelude pitch corpus shipped with the repo.
It generates a fixed-length continuation that:

- emits exact requested anchor pitches at each 8-note bar start;
- defaults to an ascending C-major line C D E F G A B C;
- renders the final anchor C as a full-measure MIDI note;
- rejects generated n-grams copied from the training data.

The bar-start rule is a positional phrase/bar-anchor constraint. It is meter-like
if each pitch token represents a fixed metrical unit, but it is not a
duration-aware meter constraint.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import random
import struct
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vo_regular_bp import (  # noqa: E402
    LongestFeasiblePolicy,
    SingletonAvoidingBackoffPolicy,
    at_position,
    avoid_copied_ngrams,
    combine_constraints,
    prepare_constrained_order_stack_from_sequences,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bach_prelude_c_major_pitches.txt"
MIDI_DIR = Path(__file__).resolve().parents[1] / "outputs" / "bach_c_anchored_midi"
TRACE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "bach_c_anchored_traces"
START_SYMBOL = "<START>"
DEFAULT_ASCENDING_ANCHOR_OFFSETS = (0, 2, 4, 5, 7, 9, 11, 12)


def load_pitches(path: Path) -> tuple[int, ...]:
    pitches: list[int] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        pitches.extend(int(token) for token in line.split())
    if not pitches:
        raise ValueError(f"no pitch tokens found in {path}")
    return tuple(pitches)


def default_ascending_anchor_pitches(start_pitch: int) -> tuple[int, ...]:
    return tuple(int(start_pitch) + offset for offset in DEFAULT_ASCENDING_ANCHOR_OFFSETS)


def anchor_positions(*, horizon: int, period: int) -> tuple[int, ...]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if period <= 0:
        raise ValueError("period must be positive")
    return tuple(range(0, horizon, period))


def bar_anchor_constraints(
    *,
    horizon: int,
    period: int,
    anchor_pitches: Sequence[int],
) -> list:
    positions = anchor_positions(horizon=horizon, period=period)
    if len(anchor_pitches) != len(positions):
        raise ValueError(
            f"{len(anchor_pitches)} anchor pitches supplied for {len(positions)} bar starts"
        )

    return [
        at_position(position, int(anchor_pitch))
        for position, anchor_pitch in zip(positions, anchor_pitches)
    ]


def copied_windows(reference: Sequence[int], length: int) -> set[tuple[int, ...]]:
    return {
        tuple(reference[index : index + length])
        for index in range(0, len(reference) - length + 1)
    }


def longest_training_copy(
    generated: Sequence[int],
    reference: Sequence[int],
) -> tuple[int, tuple[int, ...]]:
    max_length = min(len(generated), len(reference))
    for length in range(max_length, 0, -1):
        forbidden = copied_windows(reference, length)
        for index in range(0, len(generated) - length + 1):
            window = tuple(generated[index : index + length])
            if window in forbidden:
                return length, window
    return 0, ()


def format_order_histogram(orders: Sequence[int]) -> str:
    counts = Counter(orders)
    return ", ".join(f"{order}:{counts[order]}" for order in sorted(counts))


def make_policy(name: str):
    if name == "longest":
        return LongestFeasiblePolicy()
    if name == "singleton":
        return SingletonAvoidingBackoffPolicy()
    raise ValueError(f"unknown policy {name!r}")


def write_midi(
    pitches: Sequence[int],
    path: Path,
    *,
    ticks_per_quarter: int = 480,
    note_ticks: int = 120,
    tempo_bpm: int = 120,
    velocity: int = 84,
    accent_velocity: int = 104,
    accent_period: int | None = 16,
    final_hold_start: int | None = None,
    channel: int = 0,
    program: int = 0,
) -> None:
    """Write a monophonic fixed-duration MIDI file for a pitch sequence."""

    if ticks_per_quarter <= 0:
        raise ValueError("ticks_per_quarter must be positive")
    if note_ticks <= 0:
        raise ValueError("note_ticks must be positive")
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be positive")
    if not 0 <= channel <= 15:
        raise ValueError("channel must be in 0..15")
    if not 0 <= program <= 127:
        raise ValueError("program must be in 0..127")
    if final_hold_start is not None and not 0 <= final_hold_start < len(pitches):
        raise ValueError("final_hold_start must be a valid pitch index")

    track = bytearray()
    tempo_us_per_quarter = round(60_000_000 / tempo_bpm)
    track.extend(_midi_varlen(0))
    track.extend(b"\xff\x51\x03")
    track.extend(tempo_us_per_quarter.to_bytes(3, byteorder="big"))
    track.extend(_midi_varlen(0))
    track.extend(bytes((0xC0 | channel, program)))

    for index, pitch in enumerate(pitches):
        midi_pitch = int(pitch)
        if not 0 <= midi_pitch <= 127:
            raise ValueError(f"pitch {midi_pitch} is outside MIDI range 0..127")
        duration_ticks = (
            note_ticks * (len(pitches) - index)
            if final_hold_start is not None and index == final_hold_start
            else note_ticks
        )
        note_velocity = (
            accent_velocity
            if accent_period is not None and accent_period > 0 and index % accent_period == 0
            else velocity
        )
        track.extend(_midi_varlen(0))
        track.extend(bytes((0x90 | channel, midi_pitch, note_velocity)))
        track.extend(_midi_varlen(duration_ticks))
        track.extend(bytes((0x80 | channel, midi_pitch, 0)))
        if final_hold_start is not None and index == final_hold_start:
            break

    track.extend(_midi_varlen(0))
    track.extend(b"\xff\x2f\x00")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"MThd" + struct.pack(">LHHH", 6, 0, 1, ticks_per_quarter)
    chunk = b"MTrk" + struct.pack(">L", len(track)) + bytes(track)
    path.write_bytes(header + chunk)


def _midi_varlen(value: int) -> bytes:
    if value < 0:
        raise ValueError("variable-length MIDI values must be non-negative")
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
        value >>= 7

    result = bytearray()
    while True:
        result.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(result)


def write_order_trace(
    pitches: Sequence[int],
    orders: Sequence[int],
    path: Path,
    *,
    anchors: Sequence[int],
    anchor_pitches: Sequence[int],
    final_hold_start: int | None,
) -> None:
    """Write selected generation order per emitted position as CSV."""

    if len(pitches) != len(orders):
        raise ValueError("pitches and orders must have the same length")
    anchor_by_position = dict(zip(anchors, anchor_pitches))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "position",
                "pitch",
                "order",
                "is_anchor",
                "anchor_pitch",
                "anchor_match",
                "audible_in_midi",
                "held_full_measure",
            ],
        )
        writer.writeheader()
        for position, (pitch, order) in enumerate(zip(pitches, orders)):
            anchor_pitch = anchor_by_position.get(position)
            audible = final_hold_start is None or position <= final_hold_start
            writer.writerow(
                {
                    "position": position,
                    "pitch": int(pitch),
                    "order": int(order),
                    "is_anchor": anchor_pitch is not None,
                    "anchor_pitch": "" if anchor_pitch is None else int(anchor_pitch),
                    "anchor_match": "" if anchor_pitch is None else int(pitch) == int(anchor_pitch),
                    "audible_in_midi": audible,
                    "held_full_measure": final_hold_start is not None
                    and position == final_hold_start,
                }
            )


def run_case(
    *,
    pitches: tuple[int, ...],
    horizon: int,
    model_max_order: int,
    copy_ngram: int,
    anchor_period: int,
    anchor_pitches: tuple[int, ...],
    policy_name: str,
    rng: random.Random,
    midi_dir: Path | None,
    trace_dir: Path | None,
) -> None:
    constraints = combine_constraints(
        *bar_anchor_constraints(
            horizon=horizon,
            period=anchor_period,
            anchor_pitches=anchor_pitches,
        ),
        avoid_copied_ngrams(pitches, copy_ngram),
    )

    prefix = (START_SYMBOL,)
    backend = prepare_constrained_order_stack_from_sequences(
        [pitches],
        constraints,
        max_order=model_max_order,
        length=horizon,
        prefix=prefix,
        policy=make_policy(policy_name),
        start_symbol=START_SYMBOL,
    )

    generated = backend.sample_with_orders(rng=rng)
    sequence = generated.sequence
    longest_len, longest_copy = longest_training_copy(sequence, pitches)

    positions = anchor_positions(horizon=horizon, period=anchor_period)
    last_measure_start = positions[-1]
    actual_anchors = tuple(sequence[position] for position in positions)
    audible_orders = generated.orders[: last_measure_start + 1]
    print(f"\ncopy_ngram={copy_ngram}")
    print(f"context_prefix={prefix}")
    print(f"policy={policy_name}")
    print(f"anchor_pitches={anchor_pitches}")
    print(f"anchors={positions}")
    print(f"last_measure_start={last_measure_start}")
    print(f"sequence={sequence}")
    print(f"orders={generated.orders}")
    print(f"order_histogram={format_order_histogram(generated.orders)}")
    print(f"generated_max_order={max(generated.orders)}")
    print(f"audible_max_order={max(audible_orders)}")
    print(f"diagnostics={backend.diagnostics.as_dict()}")
    print(f"actual_anchor_pitches={actual_anchors}")
    print(f"anchors_match={actual_anchors == anchor_pitches}")
    print(f"final_pitch={sequence[-1]}")
    print(f"longest_training_copy={longest_len}: {longest_copy}")
    print(f"violates_copy_ngram={longest_len >= copy_ngram}")
    if midi_dir is not None:
        midi_path = midi_dir / (
            f"bach_c_anchor_h{horizon}_K{model_max_order}_{policy_name}_copy{copy_ngram}.mid"
        )
        write_midi(
            sequence,
            midi_path,
            accent_period=anchor_period,
            final_hold_start=last_measure_start,
        )
        print(f"midi_c_full_note_ticks={anchor_period * 120}")
        print(f"midi={midi_path}")
    if trace_dir is not None:
        trace_path = trace_dir / (
            f"bach_c_anchor_h{horizon}_K{model_max_order}_{policy_name}_copy{copy_ngram}_orders.csv"
        )
        write_order_trace(
            sequence,
            generated.orders,
            trace_path,
            anchors=positions,
            anchor_pitches=anchor_pitches,
            final_hold_start=last_measure_start,
        )
        print(f"order_trace={trace_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--model-max-order", type=int, default=4)
    parser.add_argument(
        "--policy",
        choices=("singleton", "longest"),
        default="singleton",
        help="order-selection policy",
    )
    parser.add_argument("--copy-ngram", type=int, nargs="+", default=[8, 7, 6])
    parser.add_argument("--anchor-period", type=int, default=8)
    parser.add_argument(
        "--anchor-pitch",
        type=int,
        default=None,
        help="repeat one exact MIDI pitch at every bar anchor",
    )
    parser.add_argument(
        "--anchor-pitches",
        type=int,
        nargs="+",
        default=None,
        help="exact MIDI pitches required at successive bar anchors",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--midi-dir",
        type=Path,
        default=MIDI_DIR,
        help="directory for generated MIDI files",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=TRACE_DIR,
        help="directory for generated order-trace CSV files",
    )
    parser.add_argument("--no-midi", action="store_true", help="skip MIDI export")
    parser.add_argument("--no-trace", action="store_true", help="skip order-trace CSV export")
    args = parser.parse_args()

    pitches = load_pitches(args.data)
    rng = random.Random(args.seed)
    midi_dir = None if args.no_midi else args.midi_dir
    trace_dir = None if args.no_trace else args.trace_dir
    if args.anchor_pitches is not None:
        anchor_pitches = tuple(int(pitch) for pitch in args.anchor_pitches)
    elif args.anchor_pitch is not None:
        anchor_pitches = (int(args.anchor_pitch),)
    else:
        anchor_pitches = default_ascending_anchor_pitches(pitches[0])
    horizon = int(args.horizon if args.horizon is not None else len(anchor_pitches) * args.anchor_period)
    positions = anchor_positions(horizon=horizon, period=args.anchor_period)
    if len(anchor_pitches) == 1 and len(positions) > 1:
        anchor_pitches = anchor_pitches * len(positions)
    if len(anchor_pitches) != len(positions):
        raise ValueError(
            f"{len(anchor_pitches)} anchor pitches supplied for {len(positions)} bar starts"
        )

    print(f"training_tokens={len(pitches)}")
    print(f"alphabet={tuple(sorted(set(pitches)))}")
    print(f"horizon={horizon}")
    print(f"model_max_order={args.model_max_order}")
    print(f"policy={args.policy}")
    print(f"anchor_pitches={anchor_pitches}")

    for copy_ngram in args.copy_ngram:
        try:
            run_case(
                pitches=pitches,
                horizon=horizon,
                model_max_order=args.model_max_order,
                copy_ngram=copy_ngram,
                anchor_period=args.anchor_period,
                anchor_pitches=anchor_pitches,
                policy_name=args.policy,
                rng=rng,
                midi_dir=midi_dir,
                trace_dir=trace_dir,
            )
        except ValueError as exc:
            print(f"\ncopy_ngram={copy_ngram}")
            print(f"infeasible={exc}")


if __name__ == "__main__":
    main()
