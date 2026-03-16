#!/usr/bin/env python3
"""
E5 -- Item-Type Asymmetry
==========================
Stratifies results by Validity (Accurate vs Inaccurate) and Exposure Group
(EG0, EG1). Tests whether steering disproportionately improves calibration
on false-exposure items (the misinformation effect).

Depends on: E1, E2 results.

Saves:
  results/E5/E5_asymmetry.json
  results/E5/plots/*.png
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import *
from shared import (
    setup_logging, setup_viz, save_plot,
    load_statements, load_human_data, get_participant_split,
    EG_COLORS, BASELINE_COLOR, OPTIMAL_COLOR, HUMAN_COLOR,
)


def main():
    log = setup_logging("E5")
    setup_viz()

    out_dir  = RESULTS_DIR / "E5"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prerequisite results ────────────────────────────────────────
    for dep in ["E1/E1_summary.json", "E2/E2_optimal_alpha.json"]:
        if not (RESULTS_DIR / dep).exists():
            raise FileNotFoundError(f"Dependency missing: {dep}")

    # E1 baseline results per EG
    e1_results = {}
    for eg in EXPOSURE_GROUPS:
        path = RESULTS_DIR / "E1" / f"E1_baseline_EG{eg}.json"
        if path.exists():
            with open(path) as f:
                e1_results[eg] = json.load(f)

    # E2 optimal alpha and sweep
    with open(RESULTS_DIR / "E2" / "E2_optimal_alpha.json") as f:
        opt = json.load(f)
    OPT_KEY = opt["optimal_alpha"]

    e2_sweep = {}
    for eg in EXPOSURE_GROUPS:
        path = RESULTS_DIR / "E2" / f"E2_sweep_EG{eg}.json"
        if path.exists():
            with open(path) as f:
                e2_sweep[eg] = json.load(f)

    log.info(f"Optimal alpha*: {OPT_KEY}")

    # Baseline accuracy per EG (from E2 sweep at 0,0)
    baseline_acc_by_eg: dict[int, float] = {}
    for eg in EXPOSURE_GROUPS:
        if eg in e2_sweep and "0,0" in e2_sweep[eg]:
            baseline_acc_by_eg[eg] = float(np.mean(e2_sweep[eg]["0,0"]["model_accs"]))
    log.info(f"Baseline accuracy by EG: {baseline_acc_by_eg}")

    # ── Load human data with Validity labels ─────────────────────────────
    human_df = load_human_data()
    calib_ids, _ = get_participant_split(human_df)
    calib_df = human_df[human_df["ID_1"].isin(calib_ids)]

    # Per-item human stats WITH Validity label
    item_stats = (
        calib_df.groupby("Item")
        .agg(
            human_conf_norm=("Confidence", lambda x: (x.mean() - 1) / 3),
            human_accuracy=("Correct", "mean"),
            validity=("Validity", "first"),
            exposure=("Exposure", "first"),
        )
        .reset_index()
        .rename(columns={"Item": "item_id"})
    )
    log.info(f"Validity counts:\n{item_stats['validity'].value_counts()}")

    # ── Load item order for sweep alignment ──────────────────────────────
    statements = load_statements()
    item_order = statements["item_id"].tolist()

    # ── Human per-item ECE (self-calibration) ────────────────────────────
    item_stats["human_ece"] = np.abs(
        item_stats["human_conf_norm"] - item_stats["human_accuracy"]
    )

    # ── Build analysis DataFrame per EG ──────────────────────────────────
    results_by_validity = {}

    for eg in EXPOSURE_GROUPS:
        if eg not in e1_results or eg not in e2_sweep:
            log.info(f"Skipping EG{eg}: missing E1 or E2 data")
            continue

        # Baseline confidence per item
        baseline_confs = {k: v["mean_confidence"] for k, v in e1_results[eg].items()}

        # Optimal alpha confidence per item
        opt_data = e2_sweep[eg].get(OPT_KEY, {})
        if not opt_data:
            log.info(f"Skipping EG{eg}: optimal alpha not in sweep")
            continue

        opt_confs = {iid: c for iid, c in zip(
            opt_data.get("item_ids", item_order),
            opt_data["model_confs"],
        )}
        opt_accs = {iid: a for iid, a in zip(
            opt_data.get("item_ids", item_order),
            opt_data["model_accs"],
        )}

        # Merge with human stats
        df = item_stats.copy()
        df["baseline_conf"] = df["item_id"].map(baseline_confs)
        df["opt_conf"] = df["item_id"].map(opt_confs)
        df["opt_acc"] = df["item_id"].map(opt_accs)
        df = df.dropna(subset=["baseline_conf", "opt_conf"])
        df["eg"] = eg

        log.info(f"EG{eg}: {len(df)} items with full data")

        # Baseline accuracy for this EG
        baseline_confs_dict = {k: v for k, v in e1_results[eg].items()}
        baseline_accs = {k: v["mean_accuracy"] for k, v in baseline_confs_dict.items()}

        # Compute ΔECE by Validity
        for validity in df["validity"].unique():
            sub = df[df["validity"] == validity]
            ece_base = float(np.mean(np.abs(sub["baseline_conf"] - sub["human_conf_norm"])))
            ece_opt  = float(np.mean(np.abs(sub["opt_conf"] - sub["human_conf_norm"])))
            delta = ece_base - ece_opt  # positive = improvement

            # Human self-calibration ECE: |conf - accuracy| per item
            human_ece = float(sub["human_ece"].mean())

            # Accuracy metrics
            base_acc_vals = [baseline_accs.get(iid, np.nan) for iid in sub["item_id"]]
            mean_base_acc = float(np.nanmean(base_acc_vals))
            mean_opt_acc = float(sub["opt_acc"].mean())
            mean_human_acc = float(sub["human_accuracy"].mean())

            key = f"EG{eg}_{validity}"
            results_by_validity[key] = {
                "exposure_group": eg,
                "validity": validity,
                "ece_baseline": ece_base,
                "ece_optimal": ece_opt,
                "ece_human": human_ece,
                "delta_ece": delta,
                "n_items": len(sub),
                "mean_human_conf": float(sub["human_conf_norm"].mean()),
                "mean_baseline_conf": float(sub["baseline_conf"].mean()),
                "mean_opt_conf": float(sub["opt_conf"].mean()),
                "mean_baseline_acc": mean_base_acc,
                "mean_opt_acc": mean_opt_acc,
                "mean_human_acc": mean_human_acc,
            }
            log.info(
                f"  {key}: ECE_base={ece_base:.4f} ECE_opt={ece_opt:.4f} ECE_human={human_ece:.4f} "
                f"ΔECE={delta:+.4f} acc_base={mean_base_acc:.4f} acc_opt={mean_opt_acc:.4f} "
                f"acc_human={mean_human_acc:.4f}"
            )

    # Save results
    with open(out_dir / "E5_asymmetry.json", "w") as f:
        json.dump(results_by_validity, f, indent=2)
    log.info("Saved E5_asymmetry.json")

    # ══════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════
    log.info("Generating plots...")

    rdf = pd.DataFrame(results_by_validity.values())
    if len(rdf) == 0:
        log.info("No data for visualization. Exiting.")
        return

    # ── 1. ΔECE by Validity x Exposure Group ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_delta = rdf.pivot(index="validity", columns="exposure_group", values="delta_ece")
    pivot_delta.columns = [EG_LABELS[c] for c in pivot_delta.columns]
    pivot_delta.plot(kind="bar", ax=ax, color=[EG_COLORS[0], EG_COLORS[1]])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ΔECE (positive = improvement)")
    ax.set_title("E5: ΔECE by Item Validity x Exposure Group")
    ax.set_xlabel("Validity")
    plt.xticks(rotation=0)
    ax.legend(title="Exposure")
    save_plot(fig, plot_dir, "E5_delta_ece_by_validity_eg")

    # ── 2. ECE Before/After by Validity (incl. Human) ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        sub = rdf[rdf["exposure_group"] == eg]
        if len(sub) == 0:
            continue
        x = np.arange(len(sub))
        w = 0.25
        ax.bar(x - w, sub["ece_baseline"], w, label="Baseline", color=BASELINE_COLOR)
        ax.bar(x, sub["ece_optimal"], w, label="Optimal α*", color=OPTIMAL_COLOR)
        ax.bar(x + w, sub["ece_human"], w, label="Human", color=HUMAN_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["validity"])
        ax.set_ylabel("ECE")
        ax.set_title(f"{EG_LABELS[eg]}")
        ax.legend()
    fig.suptitle("E5: ECE by Validity — Model vs Human", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E5_ece_before_after")

    # ── 3. Interaction Heatmap (Exposure x Validity) ─────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for ax, metric, title, cmap in zip(
        axes,
        ["delta_ece", "ece_baseline", "ece_optimal", "ece_human"],
        ["ΔECE (improvement)", "ECE Baseline", "ECE Optimal", "ECE Human"],
        ["RdYlGn", "YlOrRd", "YlOrRd", "YlOrRd"],
    ):
        pivot = rdf.pivot(
            index="exposure_group", columns="validity", values=metric
        )
        pivot.index = [EG_LABELS[i] for i in pivot.index]
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax,
                    linewidths=1, cbar_kws={"label": metric})
        ax.set_title(title)

    fig.suptitle("E5: Exposure x Validity Interaction", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E5_interaction_heatmap")

    # ── 4. Confidence Distributions by Validity ──────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row_idx, eg in enumerate(EXPOSURE_GROUPS):
        for col_idx, validity in enumerate(rdf["validity"].unique()):
            ax = axes[row_idx, col_idx]
            key = f"EG{eg}_{validity}"
            if key not in results_by_validity:
                continue
            r = results_by_validity[key]

            # Get per-item data
            sub_items = item_stats[item_stats["validity"] == validity]
            if eg in e1_results:
                base_confs = [e1_results[eg].get(iid, {}).get("mean_confidence", np.nan)
                              for iid in sub_items["item_id"]]
            else:
                base_confs = []
            human_confs = sub_items["human_conf_norm"].values

            if len(base_confs) > 0:
                ax.hist(base_confs, bins=10, alpha=0.5, density=True,
                        color=BASELINE_COLOR, label="Model (baseline)")
            ax.hist(human_confs, bins=10, alpha=0.5, density=True,
                    color=HUMAN_COLOR, label="Human")
            ax.set_xlabel("Confidence")
            ax.set_title(f"{EG_LABELS[eg]} - {validity}")
            ax.legend(fontsize=8)

    fig.suptitle("E5: Confidence Distributions by Validity x Exposure", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E5_conf_distributions")

    # ── 5. Mean Confidence Comparison ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_data = []
    for _, r in rdf.iterrows():
        key = f"{EG_LABELS[r['exposure_group']]}\n{r['validity']}"
        bar_data.append({"Group": key, "Source": "Human", "Conf": r["mean_human_conf"]})
        bar_data.append({"Group": key, "Source": "Baseline", "Conf": r["mean_baseline_conf"]})
        bar_data.append({"Group": key, "Source": "Optimal", "Conf": r["mean_opt_conf"]})
    bdf = pd.DataFrame(bar_data)
    sns.barplot(data=bdf, x="Group", y="Conf", hue="Source",
                palette=[HUMAN_COLOR, BASELINE_COLOR, OPTIMAL_COLOR], ax=ax)
    ax.set_ylabel("Mean Confidence (normalized)")
    ax.set_title("E5: Mean Confidence by Condition")
    ax.legend(title="Source")
    save_plot(fig, plot_dir, "E5_mean_conf_comparison")

    # ── 6. Accuracy by Validity (incl. Human) ────────────────────────
    if "mean_baseline_acc" in rdf.columns:
        fig, axes = plt.subplots(1, max(len(EXPOSURE_GROUPS), 2), figsize=(14, 5))
        if len(EXPOSURE_GROUPS) == 1:
            axes = [axes] if not hasattr(axes, '__len__') else axes
        for ax, eg in zip(axes, EXPOSURE_GROUPS):
            sub = rdf[rdf["exposure_group"] == eg]
            if len(sub) == 0:
                continue
            x = np.arange(len(sub))
            w = 0.25
            ax.bar(x - w, sub["mean_baseline_acc"], w, label="Baseline", color=BASELINE_COLOR)
            ax.bar(x, sub["mean_opt_acc"], w, label="Optimal α*", color=OPTIMAL_COLOR)
            ax.bar(x + w, sub["mean_human_acc"], w, label="Human", color=HUMAN_COLOR)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["validity"])
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{EG_LABELS[eg]}")
            ax.legend()
        fig.suptitle("E5: Accuracy by Validity — Model vs Human", fontsize=14)
        plt.tight_layout()
        save_plot(fig, plot_dir, "E5_accuracy_before_after")

    # ── Hypothesis verdict ───────────────────────────────────────────────
    inaccurate_deltas = rdf[rdf["validity"] == "Inaccurate"]["delta_ece"].values
    accurate_deltas   = rdf[rdf["validity"] == "Accurate"]["delta_ece"].values

    if len(inaccurate_deltas) > 0 and len(accurate_deltas) > 0:
        mean_inacc = np.mean(inaccurate_deltas)
        mean_acc   = np.mean(accurate_deltas)
        log.info(f"\nΔECE(Inaccurate) = {mean_inacc:+.4f}")
        log.info(f"ΔECE(Accurate)   = {mean_acc:+.4f}")
        if mean_inacc > mean_acc:
            log.info("HYPOTHESIS SUPPORTED: Steering improves Inaccurate items more.")
        else:
            log.info("NULL: Symmetric or reversed improvement pattern.")

    log.info(f"All plots saved to {plot_dir}")
    log.info("E5 complete.")


if __name__ == "__main__":
    main()
