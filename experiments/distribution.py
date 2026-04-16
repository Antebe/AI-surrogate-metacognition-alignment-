"""
Distributional metrics over the 4-category response space:
Correct (0), Lure (1), Unsure (2), Other (3).

1-Wasserstein on the ordered categorical.
Jensen-Shannon divergence (symmetric KL) with additive smoothing.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

CATEGORIES: tuple[str, ...] = ("Correct", "Lure", "Unsure", "Other")


def category_distribution(cats: Iterable[str]) -> dict[str, float]:
    """Normalized frequency over the 4-category space. Missing = 0."""
    counts = Counter(cats)
    total = sum(counts.values())
    if total == 0:
        return {c: 0.0 for c in CATEGORIES}
    return {c: counts.get(c, 0) / total for c in CATEGORIES}


def _as_vec(dist: dict[str, float]) -> np.ndarray:
    return np.array([dist.get(c, 0.0) for c in CATEGORIES], dtype=float)


def wasserstein_4cat(a: dict[str, float], b: dict[str, float]) -> float:
    """1-Wasserstein (Earth Mover's) over the ordered 4-category support."""
    pa, pb = _as_vec(a), _as_vec(b)
    return float(np.sum(np.abs(np.cumsum(pa) - np.cumsum(pb))))


def js_divergence_4cat(
    a: dict[str, float], b: dict[str, float], eps: float = 1e-12
) -> float:
    """Symmetric Jensen-Shannon divergence (base-2), smoothed."""
    pa, pb = _as_vec(a) + eps, _as_vec(b) + eps
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        return float(np.sum(p * (np.log2(p) - np.log2(q))))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
