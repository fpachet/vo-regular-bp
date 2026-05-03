#!/usr/bin/env python
"""Compare explicit and virtual transposition augmentation on Bach pitches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from paper.variable_order_regular_bp.scripts.eval_bach_scalability import (  # noqa: E402
    DATA_PATH,
    forbidden_windows,
    load_bach_pitches,
)
import vo_regular_bp as vbp  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "virtual_augmentation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--maxorder-grams", type=int, default=5)
    parser.add_argument("--final-pitch-class", type=int, default=0)
    parser.add_argument("--prefix-length", type=int, default=6)
    parser.add_argument("--offsets", type=int, nargs="+", default=list(range(12)))
    args = parser.parse_args()

    pitches = load_bach_pitches(args.data)
    prefix = tuple(pitches[: args.prefix_length])
    transforms = vbp.integer_shift_transforms(args.offsets)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_sequences, explicit_materialize_s = timed(
        lambda: vbp.materialize_transformed_sequences([pitches], transforms)
    )
    original_model, original_model_s = timed(
        lambda: vbp.OrderStackModel.from_sequences([pitches], max_order=args.max_order)
    )
    explicit_model, explicit_model_s = timed(
        lambda: vbp.OrderStackModel.from_sequences(
            explicit_sequences,
            max_order=args.max_order,
        )
    )
    virtual_model, virtual_model_s = timed(
        lambda: vbp.VirtualAugmentedOrderStackModel.from_sequences(
            [pitches],
            max_order=args.max_order,
            transforms=transforms,
        )
    )

    original_graph_stats = graph_stats(original_model, args.max_order)
    explicit_graph_stats = graph_stats(explicit_model, args.max_order)
    virtual_full_graph_stats = graph_stats(virtual_model, args.max_order)

    original_alphabet = tuple(sorted(original_model.alphabet))
    explicit_alphabet = tuple(sorted(explicit_model.alphabet))
    original_forbidden = forbidden_windows(pitches, args.maxorder_grams)
    explicit_forbidden = {
        tuple(sequence[index : index + args.maxorder_grams])
        for sequence in explicit_sequences
        for index in range(len(sequence) - args.maxorder_grams + 1)
    }
    original_acceptor = vbp.dense_forbidden_substring_acceptor(
        original_forbidden,
        alphabet=original_alphabet,
    )
    explicit_acceptor = vbp.dense_forbidden_substring_acceptor(
        explicit_forbidden,
        alphabet=explicit_alphabet,
    )
    original_final = {
        symbol for symbol in original_alphabet if int(symbol) % 12 == args.final_pitch_class
    }
    explicit_final = {
        symbol for symbol in explicit_alphabet if int(symbol) % 12 == args.final_pitch_class
    }
    constraints = {args.horizon - 1: original_final}
    augmented_constraints = {args.horizon - 1: explicit_final}

    original_bp, original_bp_s = timed(
        lambda: vbp.run_order_stack_masked_dfa_bp(
            original_model,
            original_acceptor,
            length=args.horizon,
            prefix=prefix,
            constraints=constraints,
            policy=vbp.LongestFeasiblePolicy(),
        )
    )
    explicit_bp, explicit_bp_s = timed(
        lambda: vbp.run_order_stack_masked_dfa_bp(
            explicit_model,
            explicit_acceptor,
            length=args.horizon,
            prefix=prefix,
            constraints=augmented_constraints,
            policy=vbp.LongestFeasiblePolicy(),
        )
    )
    virtual_bp, virtual_bp_s = timed(
        lambda: vbp.run_order_stack_masked_dfa_bp(
            virtual_model,
            explicit_acceptor,
            length=args.horizon,
            prefix=prefix,
            constraints=augmented_constraints,
            policy=vbp.LongestFeasiblePolicy(),
        )
    )

    rows = [
        model_row(
            "original",
            model=original_model,
            model_s=original_model_s,
            materialize_s=0.0,
            graph=original_graph_stats,
            bp=original_bp,
            bp_s=original_bp_s,
            stored_sequences=1,
            stored_events=len(pitches),
            virtual_events=len(pitches),
            transform_count=1,
            forbidden_count=len(original_forbidden),
        ),
        model_row(
            "explicit_augmentation",
            model=explicit_model,
            model_s=explicit_model_s,
            materialize_s=explicit_materialize_s,
            graph=explicit_graph_stats,
            bp=explicit_bp,
            bp_s=explicit_bp_s,
            stored_sequences=len(explicit_sequences),
            stored_events=sum(len(sequence) for sequence in explicit_sequences),
            virtual_events=sum(len(sequence) for sequence in explicit_sequences),
            transform_count=len(transforms),
            forbidden_count=len(explicit_forbidden),
        ),
        model_row(
            "virtual_augmentation_lazy_bp",
            model=virtual_model,
            model_s=virtual_model_s,
            materialize_s=0.0,
            graph=virtual_full_graph_stats,
            bp=virtual_bp,
            bp_s=virtual_bp_s,
            stored_sequences=1,
            stored_events=len(pitches),
            virtual_events=sum(len(sequence) for sequence in explicit_sequences),
            transform_count=len(transforms),
            forbidden_count=len(explicit_forbidden),
        ),
    ]

    rows[-1]["explicit_success_mass_diff"] = (
        float(virtual_bp.success_mass) - float(explicit_bp.success_mass)
    )
    rows[-1]["explicit_start_masses_match"] = (
        virtual_bp.start_order_masses() == explicit_bp.start_order_masses()
    )

    output_path = output_dir / "virtual_augmentation_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"output={output_path}")
    for row in rows:
        print(
            row["method"],
            f"stored_events={row['stored_events']}",
            f"full_graph_states={row['full_graph_states']}",
            f"bp_graph_states={row['bp_graph_states']}",
            f"bp_s={float(row['bp_s']):.6f}",
            f"product_edges={row['product_edges']}",
        )
    print(f"virtual_matches_explicit={rows[-1]['explicit_start_masses_match']}")


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def graph_stats(model, max_order: int) -> dict[str, float | int]:
    def build():
        graphs = {order: model.compile_graph(order) for order in range(1, max_order + 1)}
        return {
            "full_graph_states": sum(len(graph.contexts) for graph in graphs.values()),
            "full_graph_edges": sum(graph.edge_count for graph in graphs.values()),
        }

    stats, elapsed = timed(build)
    stats["full_graph_s"] = elapsed
    return stats


def model_row(
    method: str,
    *,
    model,
    model_s: float,
    materialize_s: float,
    graph: dict[str, float | int],
    bp,
    bp_s: float,
    stored_sequences: int,
    stored_events: int,
    virtual_events: int,
    transform_count: int,
    forbidden_count: int,
) -> dict[str, object]:
    return {
        "method": method,
        "stored_sequences": stored_sequences,
        "stored_events": stored_events,
        "virtual_events": virtual_events,
        "transform_count": transform_count,
        "model_count_contexts": model_count_contexts(model),
        "base_count_contexts": getattr(model, "base_context_count", ""),
        "virtual_contexts": (
            model.virtual_context_count(model.max_order)
            if hasattr(model, "virtual_context_count")
            else ""
        ),
        "forbidden_ngrams": forbidden_count,
        "materialize_s": f"{materialize_s:.6f}",
        "model_s": f"{model_s:.6f}",
        "full_graph_s": f"{float(graph['full_graph_s']):.6f}",
        "full_graph_states": graph["full_graph_states"],
        "full_graph_edges": graph["full_graph_edges"],
        "bp_s": f"{bp_s:.6f}",
        "bp_graph_states": bp.context_state_count,
        "bp_graph_edges": bp.context_edge_count,
        "product_states": bp.product_state_count,
        "time_product_states": bp.time_indexed_product_state_count,
        "product_edges": bp.product_edge_count,
        "success_mass": f"{bp.success_mass:.12g}",
        "start_order_masses": json.dumps(bp.start_order_masses()),
        "explicit_success_mass_diff": "",
        "explicit_start_masses_match": "",
    }


def model_count_contexts(model) -> int:
    counts = getattr(model, "counts", None)
    if counts is not None:
        return len(counts)
    return len(model.base_model.counts)


if __name__ == "__main__":
    main()
