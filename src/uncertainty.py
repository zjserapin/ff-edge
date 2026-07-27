"""Interval estimates, kept in one place so every result carries error bars.

Three functions because this project has three kinds of uncertainty, and using
the wrong one understates the error by an order of magnitude:

  wilson_interval        A rate from n independent trials. Correct for a base
                         rate over player-seasons.
  bootstrap_ci           Any statistic over independent observations, when there
                         is no closed form (AUC, a calibration bin's lift).
  cluster_bootstrap_ci   The one that matters most, and the one most likely to
                         be skipped. When observations come in correlated
                         groups — 5,000 simulations that all replay the same
                         four NFL seasons — resampling observations pretends you
                         have 5,000 independent draws. You have four. Resample
                         the groups instead.

The difference is not academic. On the strategy simulation, a naive Wilson
interval reports about +/-1 point on a title rate; resampling seasons reports
something wide enough to conclude that most strategies are indistinguishable,
which is the honest answer and the opposite conclusion.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate.

    Wilson rather than the normal approximation because the rates here are near
    0.2 on samples of a few hundred, where the textbook p +/- z*sqrt(p(1-p)/n)
    misbehaves and can hand back a lower bound below zero.
    """
    if n <= 0:
        return (0.0, 0.0)

    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `stat` over independent observations."""
    values = np.asarray(values)
    if values.size == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.shape[0], size=(n_boot, values.shape[0]))
    draws = np.array([stat(values[i]) for i in idx])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def cluster_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap that resamples whole groups, for correlated observations.

    Draw `n_groups` groups with replacement, pool their observations, compute
    the statistic. The width then reflects how much the answer moves when you
    swap which seasons you happened to observe — which is the real question when
    the answer is being generalized to a season you have not seen.

    With only a handful of groups the interval will be wide. That is not a
    defect of the method; it is the sample size showing up where it belongs.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)
    if values.size == 0:
        return (float("nan"), float("nan"))

    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)

    draws = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([members[i] for i in picked])
        draws.append(stat(values[idx]))

    arr = np.array(draws)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2)))


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the rank-sum identity.

    Implemented here rather than imported so `bootstrap_ci` can call it a few
    thousand times without sklearn's validation overhead on every draw, and so
    a resample containing one class returns nan instead of raising.
    """
    y = np.asarray(y_true).astype(float)
    s = np.asarray(scores).astype(float)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)

    # Average ranks within ties, or tied scores bias the statistic.
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]

    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
