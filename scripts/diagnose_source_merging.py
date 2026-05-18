"""Report exact and experimental source-graph compression diagnostics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from vo_regular_bp import (
    ContextGraph,
    exact_context_graph_quotient_stats,
    minimize_context_graph,
)
from vo_regular_bp.experimental import alergia_merge, alergia_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="whitespace-separated token corpus")
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--min-support", type=float, default=10.0)
    parser.add_argument(
        "--projection",
        choices=("identity", "pitch-class", "interval"),
        default="identity",
        help="simple built-in projection for diagnostics",
    )
    parser.add_argument("--heldout", type=Path)
    args = parser.parse_args()

    sequence = _read_tokens(args.input)
    graph = ContextGraph.from_sequences([sequence], max_order=args.max_order)
    exact_stats = exact_context_graph_quotient_stats(graph)
    exact = minimize_context_graph(graph)
    merge_kwargs = _projection_kwargs(args.projection)
    merged = alergia_merge(
        graph,
        alpha=args.alpha,
        min_support=args.min_support,
        recursive=True,
        **merge_kwargs,
    )
    merged_metadata = alergia_metadata(merged)

    rows = [
        ("raw states", len(graph.states)),
        ("raw edges", graph.edge_count()),
        ("exact-minimized states", len(exact.states)),
        ("exact-minimized edges", exact.edge_count()),
        ("exact state reduction", f"{exact_stats.state_reduction:.3f}x"),
        ("ALERGIA states", len(merged.states)),
        ("ALERGIA edges", merged.edge_count()),
    ]
    if merged_metadata is not None:
        rows.extend(
            [
                (
                    "ALERGIA state reduction",
                    f"{merged_metadata.state_compression_ratio:.3f}x",
                ),
                (
                    "ALERGIA edge reduction",
                    f"{merged_metadata.edge_compression_ratio:.3f}x",
                ),
                (
                    "ALERGIA conflicting destinations",
                    merged_metadata.conflicting_symbol_destinations,
                ),
                ("ALERGIA projection kind", merged_metadata.projection_kind),
                ("ALERGIA symbol projection", merged_metadata.symbol_projection),
                ("ALERGIA transition projection", merged_metadata.transition_projection),
            ]
        )

    if args.heldout is not None:
        heldout = _read_tokens(args.heldout)
        rows.extend(
            [
                ("raw heldout avg logprob", _average_log_probability(graph, heldout)),
                ("ALERGIA heldout avg logprob", _average_log_probability(merged, heldout)),
            ]
        )

    width = max(len(label) for label, _value in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


def _read_tokens(path: Path) -> tuple[object, ...]:
    values: list[object] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        for token in line.split():
            values.append(_coerce_token(token))
    return tuple(values)


def _coerce_token(token: str) -> object:
    try:
        return int(token)
    except ValueError:
        return token


def _projection_kwargs(name: str) -> dict[str, object]:
    if name == "identity":
        return {}
    if name == "pitch-class":
        return {"symbol_projection": _pitch_class_projection}
    if name == "interval":
        return {"transition_projection": _interval_projection}
    raise ValueError(f"unknown projection {name!r}")


def _pitch_class_projection(symbol: object) -> object:
    return symbol % 12 if isinstance(symbol, int) else symbol


def _interval_projection(state: tuple[object, ...], symbol: object, _edge: object) -> object:
    if state and isinstance(state[-1], int) and isinstance(symbol, int):
        return symbol - state[-1]
    return symbol


def _average_log_probability(graph: ContextGraph, sequence: tuple[object, ...]) -> float:
    if not sequence:
        return float("nan")
    state = graph.start_state
    total = 0.0
    count = 0
    for symbol in sequence:
        edge = next((edge for edge in graph.outgoing(state) if edge.symbol == symbol), None)
        if edge is None or edge.probability <= 0.0:
            return float("-inf")
        total += math.log(edge.probability)
        count += 1
        state = edge.next_state
    return total / count


if __name__ == "__main__":
    main()
