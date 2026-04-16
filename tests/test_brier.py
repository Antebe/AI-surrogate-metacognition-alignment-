import numpy as np
import pandas as pd
import pytest

from brier import (
    build_human_prob_map,
    human_prob_for_response,
    brier_score,
    composite_loss,
    limiting_behavior_report,
)


def test_build_human_prob_map_empirical_matches_hand_computation():
    df = pd.DataFrame({
        "Confidence": [1, 1, 2, 2, 3, 3, 4, 4],
        "Correct":    [0, 0, 0, 1, 1, 1, 1, 1],   # acc by level: 0, .5, 1, 1
    })
    m = build_human_prob_map(df, mode="empirical")
    assert m == {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.0}


def test_build_human_prob_map_linear_is_fixed():
    df = pd.DataFrame({"Confidence": [1, 2, 3, 4], "Correct": [0, 0, 0, 0]})
    m = build_human_prob_map(df, mode="linear")
    assert m == {1: 0.0, 2: 1/3, 3: 2/3, 4: 1.0}


def test_build_human_prob_map_fills_missing_levels_with_global_rate():
    df = pd.DataFrame({
        "Confidence": [2, 2, 4, 4],   # no 1s, no 3s
        "Correct":    [0, 1, 1, 1],
    })
    m = build_human_prob_map(df, mode="empirical")
    # level 2: 0.5, level 4: 1.0, missing levels fall back to global mean 0.75
    assert m[2] == 0.5
    assert m[4] == 1.0
    assert m[1] == 0.75
    assert m[3] == 0.75


def test_human_prob_for_response_vectorizes():
    m = {1: 0.1, 2: 0.4, 3: 0.7, 4: 0.95}
    arr = human_prob_for_response(np.array([1, 2, 3, 4, 2]), m)
    np.testing.assert_allclose(arr, [0.1, 0.4, 0.7, 0.95, 0.4])


def test_brier_score_perfect_calibration_is_zero():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    correct = np.array([1, 0, 1, 0])
    assert brier_score(probs, correct) == pytest.approx(0.0)


def test_brier_score_worst_case_is_one():
    probs = np.array([1.0, 0.0])
    correct = np.array([0, 1])
    assert brier_score(probs, correct) == pytest.approx(1.0)


def test_composite_loss_identical_dists_is_zero():
    # Model confidences exactly equal human probabilities,
    # model accuracy exactly equals human correctness.
    mc = np.array([0.0, 0.5, 0.8, 1.0])
    ma = np.array([0.0, 0.5, 1.0, 1.0])
    hr = np.array([1, 2, 3, 4])
    h_acc = ma.copy()
    hmap = {1: 0.0, 2: 0.5, 3: 0.8, 4: 1.0}   # p_human = mc
    L, parts = composite_loss(
        model_conf=mc, model_acc=ma,
        human_resp=hr, human_correct=h_acc,
        human_prob_map=hmap,
        alpha=1.0, beta=10.0,
    )
    assert parts["brier_diff"] == pytest.approx(0.0, abs=1e-12)
    assert parts["rho_diff"] == pytest.approx(0.0, abs=1e-9)
    assert parts["acc_penalty"] == pytest.approx(0.0)
    assert L == pytest.approx(0.0, abs=1e-9)


def test_composite_loss_accuracy_penalty_fires_only_below_human():
    # parts["acc_penalty"] is the UNWEIGHTED term max(0, acc_h - acc_m).
    # The total L applies beta to it.
    mc = np.array([0.5, 0.5, 0.5, 0.5])
    ma = np.array([0.0, 0.0, 0.0, 0.0])     # model worthless
    hr = np.array([1, 1, 1, 1])
    hc = np.array([1, 1, 1, 1])             # humans all correct → acc_h = 1.0
    hmap = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    L, parts = composite_loss(
        model_conf=mc, model_acc=ma,
        human_resp=hr, human_correct=hc,
        human_prob_map=hmap,
        alpha=0.0, beta=10.0,
    )
    assert parts["acc_penalty"] == pytest.approx(1.0, abs=1e-6)
    # Total loss applies beta: brier_diff + 0 + 10 * 1.0 = brier_diff + 10
    assert L == pytest.approx(parts["brier_diff"] + 10.0, abs=1e-6)

    # Opposite: model at 1, humans at 0 → penalty = 0
    ma2 = np.ones_like(ma)
    _, parts2 = composite_loss(
        model_conf=mc, model_acc=ma2,
        human_resp=hr, human_correct=np.zeros_like(hc),
        human_prob_map=hmap,
        alpha=0.0, beta=10.0,
    )
    assert parts2["acc_penalty"] == 0.0


def test_limiting_behavior_report_has_required_rows():
    out = limiting_behavior_report(alpha=1.0, beta=10.0)
    for key in ("brier_diff", "rho_diff", "acc_penalty"):
        assert key in out
        assert "best" in out[key] and "worst" in out[key]
