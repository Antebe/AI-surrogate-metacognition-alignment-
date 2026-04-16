"""
Brier-based composite loss for calibration alignment.

Loss (advisor-finalized, 2026-04-15):
  L = |Brier_model − Brier_human|
    + alpha · |rho(conf, acc)_model − rho(conf, acc)_human|
    + beta  · max(0, acc_human − acc_steered)

Per advisor: human 1-4 responses → probability via EMPIRICAL per-level
accuracy rate on the calibration split.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

LEVELS: tuple[int, ...] = (1, 2, 3, 4)


def build_human_prob_map(
    calib_df: pd.DataFrame,
    mode: Literal["empirical", "linear"] = "empirical",
) -> dict[int, float]:
    """Map 1-4 confidence response → probability of correctness.

    Empirical: per-level hit rate; missing levels fill with global mean.
    Linear:    (r - 1) / 3.
    """
    if mode == "linear":
        return {r: (r - 1) / 3 for r in LEVELS}

    if mode != "empirical":
        raise ValueError(f"Unknown mode: {mode!r}")

    global_rate = float(calib_df["Correct"].mean())
    by_level = (
        calib_df.groupby("Confidence")["Correct"].mean().to_dict()
    )
    return {r: float(by_level.get(r, global_rate)) for r in LEVELS}


def human_prob_for_response(
    responses: np.ndarray, prob_map: dict[int, float]
) -> np.ndarray:
    """Apply prob_map to an integer array of 1-4 responses."""
    out = np.asarray(responses, dtype=float).copy()
    for r, p in prob_map.items():
        out[np.asarray(responses) == r] = p
    return out


def brier_score(probs: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared deviation between predicted prob and ground truth."""
    p = np.asarray(probs, dtype=float)
    c = np.asarray(correct, dtype=float)
    return float(np.mean((p - c) ** 2))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def composite_loss(
    model_conf: np.ndarray,
    model_acc: np.ndarray,
    human_resp: np.ndarray,
    human_correct: np.ndarray,
    human_prob_map: dict[int, float],
    alpha: float,
    beta: float,
) -> tuple[float, dict[str, float]]:
    """Compute L and return a breakdown dict."""
    p_human = human_prob_for_response(human_resp, human_prob_map)
    brier_m = brier_score(model_conf, model_acc)
    brier_h = brier_score(p_human, human_correct)
    rho_m   = _pearson(model_conf, model_acc)
    rho_h   = _pearson(p_human, human_correct)
    acc_m   = float(np.asarray(model_acc, dtype=float).mean())
    acc_h   = float(np.asarray(human_correct, dtype=float).mean())

    brier_diff  = abs(brier_m - brier_h)
    rho_diff    = abs(rho_m - rho_h)
    acc_penalty = max(0.0, acc_h - acc_m)

    L = brier_diff + alpha * rho_diff + beta * acc_penalty
    return L, {
        "brier_model":  brier_m,
        "brier_human":  brier_h,
        "brier_diff":   brier_diff,
        "rho_model":    rho_m,
        "rho_human":    rho_h,
        "rho_diff":     rho_diff,
        "acc_model":    acc_m,
        "acc_human":    acc_h,
        "acc_penalty":  acc_penalty,
    }


def limiting_behavior_report(alpha: float, beta: float) -> dict[str, dict[str, float]]:
    """Per-term best/worst limits — advisor's sanity check."""
    return {
        "brier_diff": {"best": 0.0, "worst": 1.0},
        "rho_diff":   {"best": 0.0, "worst": 2.0 * alpha},
        "acc_penalty":{"best": 0.0, "worst": 1.0 * beta},
    }
