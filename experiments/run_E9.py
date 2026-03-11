#!/usr/bin/env python3
"""
E9 -- Comprehensive E1 (Baseline) vs E8 (Steered) Analysis
============================================================
Detailed statistical comparison: confidence separation between response
categories, human alignment, accuracy preservation, and metacognitive
sensitivity. Uses effect sizes, bootstrap CIs, and paired tests.

Depends on: E1, E8

Saves:
  results/E9/E9_analysis.json
  results/E9/plots/*.png
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import *
from shared import (
    setup_logging, setup_viz, save_plot,
    load_human_data, get_participant_split,
    compute_human_norms, align_model_human, collect_response_categories,
    EG_COLORS, BASELINE_COLOR, OPTIMAL_COLOR, HUMAN_COLOR,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for two independent samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    if pooled_std < 1e-12:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def bootstrap_ci(data: np.ndarray, stat_fn=np.mean, n_boot: int = 5000,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high)."""
    rng = np.random.default_rng(RANDOM_SEED)
    point = float(stat_fn(data))
    boots = np.array([stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return point, float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))


def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, stat_fn=np.mean,
                      n_boot: int = 5000, ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap CI for the difference stat_fn(a) - stat_fn(b)."""
    rng = np.random.default_rng(RANDOM_SEED)
    point = float(stat_fn(a) - stat_fn(b))
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(stat_fn(sa) - stat_fn(sb))
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    return point, float(np.percentile(diffs, 100 * alpha)), float(np.percentile(diffs, 100 * (1 - alpha)))


def load_results(experiment: str, eg: int) -> dict:
    """Load per-item results JSON for an experiment and exposure group."""
    if experiment == "E1":
        path = RESULTS_DIR / "E1" / f"E1_baseline_EG{eg}.json"
    elif experiment == "E8":
        path = RESULTS_DIR / "E8" / f"E8_steered_EG{eg}.json"
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path) as f:
        return json.load(f)


def build_item_df(results: dict, label: str) -> pd.DataFrame:
    """Build per-item DataFrame with mean confidence, accuracy, and modal category."""
    rows = []
    for iid, v in results.items():
        confs = v["confidence_scores"]
        cats = v["response_categories"]
        accs = v.get("accuracies", [])
        # modal category
        cat_counts = pd.Series(cats).value_counts()
        modal_cat = cat_counts.index[0] if len(cat_counts) > 0 else "Other"
        rows.append({
            "item_id": iid,
            "mean_conf": float(np.mean(confs)),
            "mean_acc": float(np.mean(accs)) if accs else float(v.get("mean_accuracy", np.nan)),
            "modal_category": modal_cat,
            "n_samples": len(confs),
            "condition": label,
        })
    return pd.DataFrame(rows)


def build_sample_df(results: dict, label: str) -> pd.DataFrame:
    """Build per-sample (long-form) DataFrame."""
    rows = []
    for iid, v in results.items():
        for i, (conf, cat) in enumerate(zip(v["confidence_scores"], v["response_categories"])):
            rows.append({
                "item_id": iid,
                "sample_idx": i,
                "confidence": conf,
                "category": cat,
                "condition": label,
            })
    return pd.DataFrame(rows)


def main():
    log = setup_logging("E9")
    setup_viz()

    out_dir = RESULTS_DIR / "E9"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dependencies ─────────────────────────────────────────────────
    for dep in ["E1/E1_summary.json", "E8/E8_summary.json"]:
        if not (RESULTS_DIR / dep).exists():
            raise FileNotFoundError(f"Missing {dep}. Run E1 and E8 first.")

    with open(RESULTS_DIR / "E1" / "E1_summary.json") as f:
        e1_summary = json.load(f)
    with open(RESULTS_DIR / "E8" / "E8_summary.json") as f:
        e8_summary = json.load(f)

    human_df = load_human_data()
    calib_ids, test_ids = get_participant_split(human_df)

    analysis = {}
    cat_order = ["Correct", "Lure", "Unsure", "Other"]
    cat_pairs = list(itertools.combinations(cat_order, 2))

    for eg in EXPOSURE_GROUPS:
        log.info(f"\n{'='*60}")
        log.info(f"Analyzing EG{eg}")
        log.info(f"{'='*60}")

        e1_raw = load_results("E1", eg)
        e8_raw = load_results("E8", eg)

        e1_items = build_item_df(e1_raw, "Baseline")
        e8_items = build_item_df(e8_raw, "Steered")
        e1_samples = build_sample_df(e1_raw, "Baseline")
        e8_samples = build_sample_df(e8_raw, "Steered")

        human_norms = compute_human_norms(human_df, calib_ids, exposure_group=eg)
        if len(human_norms) == 0:
            human_norms = compute_human_norms(human_df, calib_ids)

        eg_key = f"EG{eg}"
        eg_analysis = {}

        # ── 1. Summary metrics comparison ─────────────────────────────
        e1_s = e1_summary.get(eg_key, {})
        e8_s = e8_summary.get(eg_key, {})

        eg_analysis["summary_comparison"] = {
            "metric": ["ECE", "Conf-Acc Corr", "Loss", "r(model,human)", "Mean Conf", "Mean Acc"],
            "baseline": [e1_s.get("ece"), e1_s.get("conf_acc_corr"), e1_s.get("loss"),
                         e1_s.get("r_human"), e1_s.get("mean_model_conf"), e1_s.get("mean_model_acc")],
            "steered": [e8_s.get("ece"), e8_s.get("conf_acc_corr"), e8_s.get("loss"),
                        e8_s.get("r_human"), e8_s.get("mean_model_conf"), e8_s.get("mean_model_acc")],
        }

        log.info("Summary comparison:")
        for m, b, s in zip(eg_analysis["summary_comparison"]["metric"],
                           eg_analysis["summary_comparison"]["baseline"],
                           eg_analysis["summary_comparison"]["steered"]):
            delta = (s - b) if (b is not None and s is not None) else None
            log.info(f"  {m:20s}  baseline={b:.4f}  steered={s:.4f}  delta={delta:+.4f}" if delta is not None else f"  {m:20s}  baseline={b}  steered={s}")

        # ── 2. Pairwise confidence separation between categories ──────
        log.info("\nPairwise confidence separation (Cohen's d):")
        separation_results = {}
        for cat_a, cat_b in cat_pairs:
            pair_key = f"{cat_a}_vs_{cat_b}"
            e1_a = e1_samples[e1_samples["category"] == cat_a]["confidence"].values
            e1_b = e1_samples[e1_samples["category"] == cat_b]["confidence"].values
            e8_a = e8_samples[e8_samples["category"] == cat_a]["confidence"].values
            e8_b = e8_samples[e8_samples["category"] == cat_b]["confidence"].values

            d_e1 = cohens_d(e1_a, e1_b)
            d_e8 = cohens_d(e8_a, e8_b)

            # Mann-Whitney U for each
            if len(e1_a) > 0 and len(e1_b) > 0:
                u1, p1 = stats.mannwhitneyu(e1_a, e1_b, alternative="two-sided")
            else:
                u1, p1 = float("nan"), float("nan")
            if len(e8_a) > 0 and len(e8_b) > 0:
                u8, p8 = stats.mannwhitneyu(e8_a, e8_b, alternative="two-sided")
            else:
                u8, p8 = float("nan"), float("nan")

            separation_results[pair_key] = {
                "baseline_d": d_e1, "steered_d": d_e8,
                "baseline_p": float(p1), "steered_p": float(p8),
                "baseline_n": (len(e1_a), len(e1_b)),
                "steered_n": (len(e8_a), len(e8_b)),
                "delta_d": d_e8 - d_e1 if not (np.isnan(d_e1) or np.isnan(d_e8)) else None,
            }
            log.info(f"  {pair_key:25s}  d_base={d_e1:+.3f}  d_steer={d_e8:+.3f}  delta={d_e8-d_e1:+.3f}" if not np.isnan(d_e1) and not np.isnan(d_e8) else f"  {pair_key:25s}  insufficient data")

        eg_analysis["pairwise_separation"] = separation_results

        # ── 3. Per-item paired analysis ───────────────────────────────
        # Match items between E1 and E8
        common_items = sorted(set(e1_items["item_id"]) & set(e8_items["item_id"]))
        e1_matched = e1_items.set_index("item_id").loc[common_items].sort_index()
        e8_matched = e8_items.set_index("item_id").loc[common_items].sort_index()

        conf_diff = e8_matched["mean_conf"].values - e1_matched["mean_conf"].values
        acc_diff = e8_matched["mean_acc"].values - e1_matched["mean_acc"].values

        # Paired t-tests
        t_conf, p_conf = stats.ttest_rel(e8_matched["mean_conf"].values, e1_matched["mean_conf"].values)
        t_acc, p_acc = stats.ttest_rel(e8_matched["mean_acc"].values, e1_matched["mean_acc"].values)

        # Wilcoxon signed-rank (non-parametric)
        try:
            w_conf, wp_conf = stats.wilcoxon(conf_diff[conf_diff != 0])
        except ValueError:
            w_conf, wp_conf = float("nan"), float("nan")
        try:
            w_acc, wp_acc = stats.wilcoxon(acc_diff[acc_diff != 0])
        except ValueError:
            w_acc, wp_acc = float("nan"), float("nan")

        # Bootstrap CIs
        conf_ci = bootstrap_ci(conf_diff)
        acc_ci = bootstrap_ci(acc_diff)

        eg_analysis["paired_item_analysis"] = {
            "n_items": len(common_items),
            "confidence_delta": {"mean": conf_ci[0], "ci_low": conf_ci[1], "ci_high": conf_ci[2],
                                 "paired_t": float(t_conf), "p_ttest": float(p_conf),
                                 "wilcoxon_p": float(wp_conf)},
            "accuracy_delta": {"mean": acc_ci[0], "ci_low": acc_ci[1], "ci_high": acc_ci[2],
                               "paired_t": float(t_acc), "p_ttest": float(p_acc),
                               "wilcoxon_p": float(wp_acc)},
        }

        log.info(f"\nPaired item analysis (n={len(common_items)}):")
        log.info(f"  Confidence delta: {conf_ci[0]:+.4f} [{conf_ci[1]:+.4f}, {conf_ci[2]:+.4f}]  p={p_conf:.4f}")
        log.info(f"  Accuracy delta:   {acc_ci[0]:+.4f} [{acc_ci[1]:+.4f}, {acc_ci[2]:+.4f}]  p={p_acc:.4f}")

        # ── 4. Human alignment comparison ─────────────────────────────
        e1_aligned = align_model_human(e1_raw, human_norms)
        e8_aligned = align_model_human(e8_raw, human_norms)

        # Per-item residuals: |model_conf - human_conf|
        e1_resid = np.abs(e1_aligned["model_conf"].values - e1_aligned["human_conf_norm"].values)
        e8_resid = np.abs(e8_aligned["model_conf"].values - e8_aligned["human_conf_norm"].values)

        resid_ci = bootstrap_ci_diff(e8_resid, e1_resid)
        t_resid, p_resid = stats.ttest_rel(e8_resid, e1_resid)

        eg_analysis["human_alignment"] = {
            "baseline_mean_residual": float(e1_resid.mean()),
            "steered_mean_residual": float(e8_resid.mean()),
            "delta_residual": resid_ci[0], "ci_low": resid_ci[1], "ci_high": resid_ci[2],
            "paired_t": float(t_resid), "p_value": float(p_resid),
        }
        log.info(f"\nHuman alignment (mean |model - human| residual):")
        log.info(f"  Baseline: {e1_resid.mean():.4f}  Steered: {e8_resid.mean():.4f}")
        log.info(f"  Delta: {resid_ci[0]:+.4f} [{resid_ci[1]:+.4f}, {resid_ci[2]:+.4f}]  p={p_resid:.4f}")

        # ── 5. Metacognitive sensitivity: conf-acc correlation ────────
        e1_conf_vals = e1_aligned["model_conf"].values
        e1_acc_vals = e1_aligned["model_acc"].values
        e8_conf_vals = e8_aligned["model_conf"].values
        e8_acc_vals = e8_aligned["model_acc"].values

        r_e1, p_r_e1 = stats.pearsonr(e1_conf_vals, e1_acc_vals)
        r_e8, p_r_e8 = stats.pearsonr(e8_conf_vals, e8_acc_vals)

        # Fisher z-test for comparing correlations
        z_e1 = np.arctanh(r_e1)
        z_e8 = np.arctanh(r_e8)
        n_items = len(e1_conf_vals)
        se_diff = np.sqrt(2.0 / (n_items - 3))
        z_diff = (z_e8 - z_e1) / se_diff
        p_fisher = 2 * (1 - stats.norm.cdf(abs(z_diff)))

        eg_analysis["metacognitive_sensitivity"] = {
            "baseline_r": float(r_e1), "baseline_p": float(p_r_e1),
            "steered_r": float(r_e8), "steered_p": float(p_r_e8),
            "fisher_z": float(z_diff), "fisher_p": float(p_fisher),
        }
        log.info(f"\nMetacognitive sensitivity (conf-acc r):")
        log.info(f"  Baseline: r={r_e1:.4f} (p={p_r_e1:.4f})")
        log.info(f"  Steered:  r={r_e8:.4f} (p={p_r_e8:.4f})")
        log.info(f"  Fisher z-test: z={z_diff:.3f}, p={p_fisher:.4f}")

        # ── 6. Category distribution shift ────────────────────────────
        e1_cats = collect_response_categories(e1_raw)
        e8_cats = collect_response_categories(e8_raw)

        e1_dist = e1_cats["category"].value_counts(normalize=True).reindex(cat_order, fill_value=0)
        e8_dist = e8_cats["category"].value_counts(normalize=True).reindex(cat_order, fill_value=0)

        eg_analysis["category_distribution"] = {
            "baseline": {c: float(e1_dist[c]) for c in cat_order},
            "steered": {c: float(e8_dist[c]) for c in cat_order},
        }
        log.info(f"\nCategory distributions:")
        for c in cat_order:
            log.info(f"  {c:10s}  baseline={e1_dist[c]:.3f}  steered={e8_dist[c]:.3f}  delta={e8_dist[c]-e1_dist[c]:+.3f}")

        analysis[eg_key] = eg_analysis

        # ══════════════════════════════════════════════════════════════
        # VISUALIZATIONS for this EG
        # ══════════════════════════════════════════════════════════════

        # ── Plot 1: Pairwise confidence separation (Cohen's d) ────────
        fig, ax = plt.subplots(figsize=(12, 6))
        pair_labels = [f"{a}\nvs\n{b}" for a, b in cat_pairs]
        d_base = [separation_results[f"{a}_vs_{b}"]["baseline_d"] for a, b in cat_pairs]
        d_steer = [separation_results[f"{a}_vs_{b}"]["steered_d"] for a, b in cat_pairs]
        x = np.arange(len(cat_pairs))
        w = 0.35
        bars1 = ax.bar(x - w/2, d_base, w, label="E1 Baseline", color=BASELINE_COLOR)
        bars2 = ax.bar(x + w/2, d_steer, w, label="E8 Steered", color=OPTIMAL_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=9)
        ax.set_ylabel("Cohen's d")
        ax.set_title(f"E9: Pairwise Confidence Separation — {EG_LABELS[eg]}")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend()
        for bar, val in zip(bars1, d_base):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02 * np.sign(val),
                        f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
        for bar, val in zip(bars2, d_steer):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02 * np.sign(val),
                        f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E9_pairwise_separation_EG{eg}")

        # ── Plot 2: Confidence by category (E1 vs E8 side by side) ────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (samples, label, color) in zip(axes, [
            (e1_samples, "E1 Baseline", BASELINE_COLOR),
            (e8_samples, "E8 Steered", OPTIMAL_COLOR),
        ]):
            present_cats = [c for c in cat_order if c in samples["category"].values]
            if present_cats:
                sns.boxplot(data=samples[samples["category"].isin(present_cats)],
                            x="category", y="confidence", order=present_cats,
                            palette="Set2", ax=ax)
                sns.stripplot(data=samples[samples["category"].isin(present_cats)],
                              x="category", y="confidence", order=present_cats,
                              color="black", alpha=0.2, size=2, ax=ax)
            ax.set_xlabel("Response Category")
            ax.set_ylabel("Confidence")
            ax.set_title(label)
        fig.suptitle(f"E9: Confidence by Response Category — {EG_LABELS[eg]}", fontsize=14)
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E9_conf_by_category_EG{eg}")

        # ── Plot 3: Per-item confidence change (scatter) ──────────────
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(e1_matched["mean_conf"], e8_matched["mean_conf"],
                   alpha=0.6, s=50, edgecolor="white", c=OPTIMAL_COLOR)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="No change")
        ax.set_xlabel("E1 Baseline Confidence")
        ax.set_ylabel("E8 Steered Confidence")
        ax.set_title(f"E9: Per-Item Confidence Shift — {EG_LABELS[eg]}")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E9_item_conf_shift_EG{eg}")

        # ── Plot 4: Per-item confidence delta distribution ────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(conf_diff, bins=20, alpha=0.7, color=OPTIMAL_COLOR, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(conf_diff.mean(), color="red", linewidth=2, label=f"Mean={conf_diff.mean():+.4f}")
        ax.set_xlabel("Confidence Delta (Steered - Baseline)")
        ax.set_ylabel("Count")
        ax.set_title(f"E9: Per-Item Confidence Change Distribution — {EG_LABELS[eg]}")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E9_conf_delta_hist_EG{eg}")

        # ── Plot 5: Model vs Human (E1 and E8 overlay) ───────────────
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(e1_aligned["human_conf_norm"], e1_aligned["model_conf"],
                   alpha=0.5, s=40, color=BASELINE_COLOR, label="E1 Baseline", edgecolor="white")
        ax.scatter(e8_aligned["human_conf_norm"], e8_aligned["model_conf"],
                   alpha=0.5, s=40, color=OPTIMAL_COLOR, label="E8 Steered", edgecolor="white")
        ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Perfect calibration")
        ax.set_xlabel("Human Confidence (normalized)")
        ax.set_ylabel("Model Confidence")
        ax.set_title(f"E9: Model vs Human — {EG_LABELS[eg]}\n"
                     f"Baseline r={e1_s.get('r_human', 0):.3f}, Steered r={e8_s.get('r_human', 0):.3f}")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E9_model_vs_human_overlay_EG{eg}")

        # ── Plot 6: Residual comparison (|model - human|) ────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(e1_resid, bins=15, alpha=0.5, color=BASELINE_COLOR, label="E1 Baseline", edgecolor="white")
        ax.hist(e8_resid, bins=15, alpha=0.5, color=OPTIMAL_COLOR, label="E8 Steered", edgecolor="white")
        ax.axvline(e1_resid.mean(), color=BASELINE_COLOR, linewidth=2, linestyle="--")
        ax.axvline(e8_resid.mean(), color=OPTIMAL_COLOR, linewidth=2, linestyle="--")
        ax.set_xlabel("|Model Confidence - Human Confidence|")
        ax.set_ylabel("Count")
        ax.set_title(f"E9: Alignment Residuals — {EG_LABELS[eg]}")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E9_residuals_EG{eg}")

        # ── Plot 7: Conf-Acc scatter overlay ──────────────────────────
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(e1_aligned["model_conf"], e1_aligned["model_acc"],
                   alpha=0.5, s=40, color=BASELINE_COLOR, label=f"E1 (r={r_e1:.3f})", edgecolor="white")
        ax.scatter(e8_aligned["model_conf"], e8_aligned["model_acc"],
                   alpha=0.5, s=40, color=OPTIMAL_COLOR, label=f"E8 (r={r_e8:.3f})", edgecolor="white")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
        ax.set_xlabel("Model Confidence")
        ax.set_ylabel("Model Accuracy")
        ax.set_title(f"E9: Confidence vs Accuracy — {EG_LABELS[eg]}")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E9_conf_acc_overlay_EG{eg}")

        # ── Plot 8: Category distribution shift ──────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(cat_order))
        w = 0.25
        ax.bar(x - w, [e1_dist.get(c, 0) for c in cat_order], w,
               label="E1 Baseline", color=BASELINE_COLOR)
        ax.bar(x, [e8_dist.get(c, 0) for c in cat_order], w,
               label="E8 Steered", color=OPTIMAL_COLOR)
        # Add human distribution
        h_sub = human_df[
            (human_df["ID_1"].isin(calib_ids)) &
            (human_df["Exposure"] == eg)
        ]
        if len(h_sub) == 0:
            h_sub = human_df[human_df["ID_1"].isin(calib_ids)]
        human_dist = {
            "Correct": h_sub["Correct"].mean(),
            "Lure": h_sub["Error"].mean(),
            "Unsure": h_sub["Unsure"].mean(),
            "Other": h_sub["Other"].mean(),
        }
        ax.bar(x + w, [human_dist[c] for c in cat_order], w,
               label="Human", color=HUMAN_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_order)
        ax.set_ylabel("Proportion")
        ax.set_title(f"E9: Response Category Distribution — {EG_LABELS[eg]}")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E9_category_dist_EG{eg}")

        # ── Plot 9: Per-item accuracy change ─────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(acc_diff, bins=20, alpha=0.7, color=OPTIMAL_COLOR, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(acc_diff.mean(), color="red", linewidth=2, label=f"Mean={acc_diff.mean():+.4f}")
        ax.set_xlabel("Accuracy Delta (Steered - Baseline)")
        ax.set_ylabel("Count")
        ax.set_title(f"E9: Per-Item Accuracy Change Distribution — {EG_LABELS[eg]}")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E9_acc_delta_hist_EG{eg}")

    # ── Cross-EG summary table (plot 10) ──────────────────────────────
    summary_metrics = ["ECE", "Conf-Acc Corr", "Loss", "r(model,human)", "Mean Conf", "Mean Acc"]
    summary_data = []
    for eg in EXPOSURE_GROUPS:
        eg_key = f"EG{eg}"
        comp = analysis[eg_key]["summary_comparison"]
        for i, m in enumerate(comp["metric"]):
            b = comp["baseline"][i]
            s = comp["steered"][i]
            if b is not None and s is not None:
                summary_data.append({
                    "EG": EG_LABELS[eg], "Metric": m,
                    "Baseline": b, "Steered": s, "Delta": s - b,
                })

    if summary_data:
        sdf = pd.DataFrame(summary_data)
        fig, ax = plt.subplots(figsize=(14, max(3, len(summary_data) * 0.4 + 1)))

        pivot = sdf.pivot(index="Metric", columns="EG", values="Delta")
        pivot = pivot.reindex(summary_metrics)
        sns.heatmap(pivot, annot=True, fmt="+.4f", cmap="RdYlGn_r", center=0,
                    ax=ax, linewidths=1, cbar_kws={"label": "Delta (Steered - Baseline)"})
        ax.set_title("E9: Metric Deltas (Steered - Baseline)")
        plt.tight_layout()
        save_plot(fig, plot_dir, "E9_summary_delta_heatmap")

    # ── Plot 11: Grand comparison bar chart ───────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    metrics_to_plot = [
        ("ece", "ECE (lower = better)"),
        ("conf_acc_corr", "Conf-Acc Correlation (higher = better)"),
        ("loss", "Loss (lower = better)"),
        ("r_human", "r(model, human) (higher = better)"),
        ("mean_model_conf", "Mean Confidence"),
        ("mean_model_acc", "Mean Accuracy"),
    ]
    for ax, (metric, label) in zip(axes, metrics_to_plot):
        for i, eg in enumerate(EXPOSURE_GROUPS):
            eg_key = f"EG{eg}"
            e1_val = e1_summary.get(eg_key, {}).get(metric, 0)
            e8_val = e8_summary.get(eg_key, {}).get(metric, 0)
            if e1_val is None: e1_val = 0
            if e8_val is None: e8_val = 0
            x_pos = np.array([0, 1]) + i * 3
            bars = ax.bar(x_pos, [e1_val, e8_val],
                          color=[BASELINE_COLOR, OPTIMAL_COLOR], width=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(["Baseline", "Steered"])
            for bar, val in zip(bars, [e1_val, e8_val]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(label)
        ax.set_ylabel("Value")
    fig.suptitle("E9: Baseline vs Steered — All Metrics", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E9_grand_comparison")

    # ── Save analysis ─────────────────────────────────────────────────
    with open(out_dir / "E9_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    log.info(f"\nAnalysis saved to {out_dir / 'E9_analysis.json'}")
    log.info(f"All plots saved to {plot_dir}")
    log.info("E9 complete.")


if __name__ == "__main__":
    main()
