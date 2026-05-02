"""Small metrics for exactness experiments."""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable, Mapping

Symbol = Hashable


def total_variation(
    p: Mapping[tuple[Symbol, ...], float],
    q: Mapping[tuple[Symbol, ...], float],
) -> float:
    support = set(p) | set(q)
    return 0.5 * sum(abs(float(p.get(item, 0.0)) - float(q.get(item, 0.0))) for item in support)


def empirical_distribution(samples: Iterable[tuple[Symbol, ...]]) -> dict[tuple[Symbol, ...], float]:
    counts = Counter(samples)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {sample: count / total for sample, count in counts.items()}
