"""Utilities for fixed-horizon padded generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar


SymbolT = TypeVar("SymbolT")


def append_padding(
    sequences: Iterable[Sequence[SymbolT]],
    *,
    pad_symbol: SymbolT,
    pad_count: int,
) -> tuple[tuple[SymbolT, ...], ...]:
    """Append repeated PAD symbols to each training sequence.

    This is the standard fixed-horizon encoding of variable-length generation:
    train on phrase/bar sequences followed by enough PAD symbols for the chosen
    model order, then constrain PAD to be zero-cost and absorbing.

    Probabilities are those of the padded model: the first PAD transition is a
    modeled stop/ending event, while trailing PAD self-loops should become
    deterministic when enough padding is appended.

    Use at least ``max_order + 1`` PAD symbols when building an order-stack
    model so every fixed-order PAD context has a PAD self-loop.
    """

    if pad_count < 1:
        raise ValueError("pad_count must be at least 1")
    padding = (pad_symbol,) * int(pad_count)
    return tuple(tuple(sequence) + padding for sequence in sequences)
