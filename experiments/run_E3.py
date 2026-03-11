#!/usr/bin/env python3
"""
E3 -- Ablation: Specificity
=============================
Tests whether the steering improvement is specific to hesitation/uncertainty
latents or a generic artifact. Runs target latents and 10 control latents
at optimal alpha*, for both EG0 and EG1.

Saves:
  results/E3/E3_specificity.json
  results/E3/plots/*.png
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
from tqdm import tqdm

from config import *
from shared import (
    setup_logging, setup_viz, save_plot,
    load_items, load_human_data, get_participant_split,
    compute_human_norms, run_condition, compute_loss, align_model_human,
    make_target_interventions, make_control_interventions, get_client,
    load_model, load_saes,
    EG_COLORS, BASELINE_COLOR, OPTIMAL_COLOR, CONTROL_COLOR,
)


def main():
    log = setup_logging("E3")
    setup_viz()

    out_dir  = RESULTS_DIR / "E3"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load optimal alpha from E2 ──────────────────────────────────────
    opt_path = RESULTS_DIR / "E2" / "E2_optimal_alpha.json"
    if not opt_path.exists():
        raise FileNotFoundError(f"E2 must be run first: {opt_path}")
    with open(opt_path) as f:
        opt = json.load(f)
    OPT_UNSURE  = opt["unsure_coeff"]
    OPT_REFUSAL = opt["refusal_coeff"]
    log.info(f"Optimal alpha*: unsure={OPT_UNSURE}, refusal={OPT_REFUSAL}")

    # ── Load E1 baseline ─────────────────────────────────────────────────
    e1_summary_path = RESULTS_DIR / "E1" / "E1_summary.json"
    if e1_summary_path.exists():
        with open(e1_summary_path) as f:
            e1_sum = json.load(f)
    else:
        e1_sum = None
        log.info("Warning: E1 summary not found")

    # Baseline accuracy per EG (from E1 or E2 sweep at 0,0)
    baseline_acc_by_eg: dict[int, float] = {}
    for eg in EXPOSURE_GROUPS:
        e2_sweep_path = RESULTS_DIR / "E2" / f"E2_sweep_EG{eg}.json"
        if e2_sweep_path.exists():
            with open(e2_sweep_path) as f:
                e2_sweep = json.load(f)
            if "0,0" in e2_sweep:
                baseline_acc_by_eg[eg] = float(np.mean(e2_sweep["0,0"]["model_accs"]))
    log.info(f"Baseline accuracy by EG: {baseline_acc_by_eg}")

    # ── Load model + SAEs ────────────────────────────────────────────────
    log.info("Loading model + SAEs...")
    model  = load_model(hf_token=HF_TOKEN)
    saes   = load_saes(layers=SAE_LAYERS)
    client = get_client()
    log.info("Model loaded.")

    # ── Load data ────────────────────────────────────────────────────────
    human_df = load_human_data()
    calib_ids, _ = get_participant_split(human_df)

    # ── Results storage (resumable) ──────────────────────────────────────
    save_path = out_dir / "E3_specificity.json"
    if save_path.exists():
        with open(save_path) as f:
            e3_results = json.load(f)
        log.info(f"Loaded {len(e3_results)} cached conditions")
    else:
        e3_results = {}

    # ── Define conditions ────────────────────────────────────────────────
    conditions = {"target": make_target_interventions(saes, OPT_UNSURE, OPT_REFUSAL)}
    for name in CONTROL_LATENTS:
        conditions[name] = make_control_interventions(saes, name, OPT_UNSURE)

    # ── Run each condition x each EG ─────────────────────────────────────
    for cond_name, ivs in tqdm(conditions.items(), desc="Conditions"):
        for eg in EXPOSURE_GROUPS:
            result_key = f"{cond_name}_EG{eg}"
            if result_key in e3_results:
                continue

            log.info(f"Running {cond_name} x EG{eg}...")
            items = load_items(eg)
            human_norms = compute_human_norms(human_df, calib_ids, exposure_group=eg)
            if len(human_norms) == 0:
                human_norms = compute_human_norms(human_df, calib_ids)
            h_map = dict(zip(human_norms["item_id"], human_norms["human_conf_norm"]))

            results = run_condition(
                model, items, interventions=ivs, client=client,
                n_samples=E3_N_SAMPLES, logger=log,
            )
            df = align_model_human(results, human_norms)
            loss, ece, corr = compute_loss(
                df["model_conf"].tolist(),
                df["human_conf_norm"].tolist(),
                df["model_acc"].tolist(),
            )

            mean_acc = float(df["model_acc"].mean())
            base_acc = baseline_acc_by_eg.get(eg, 1.0)
            acc_ratio = base_acc / mean_acc if mean_acc > 0 else 1.0
            adjusted_loss = loss * acc_ratio

            e3_results[result_key] = {
                "condition": cond_name,
                "exposure_group": eg,
                "loss": adjusted_loss,
                "raw_loss": loss,
                "ece": ece,
                "corr": corr,
                "mean_accuracy": mean_acc,
                "acc_ratio": acc_ratio,
                "label": cond_name,
            }
            with open(save_path, "w") as f:
                json.dump(e3_results, f, indent=2)
            log.info(
                f"  {result_key}: L_adj={adjusted_loss:.4f} L_raw={loss:.4f} "
                f"ECE={ece:.4f} corr={corr:.4f} acc={mean_acc:.4f}"
            )

    log.info(f"All conditions complete: {len(e3_results)} entries")

    # ── Aggregate results ────────────────────────────────────────────────
    rdf = pd.DataFrame(e3_results.values())

    # Combined loss per condition (mean across EGs)
    combined = rdf.groupby("condition").agg(
        mean_loss=("loss", "mean"),
        mean_ece=("ece", "mean"),
        mean_corr=("corr", "mean"),
    ).reset_index()
    combined["is_target"] = combined["condition"] == "target"

    target_loss = combined.loc[combined["is_target"], "mean_loss"].values[0]
    ctrl_losses = combined.loc[~combined["is_target"], "mean_loss"].values
    pct_ctrl_worse = np.mean(ctrl_losses > target_loss) * 100

    log.info(f"\nTarget loss:   {target_loss:.4f}")
    log.info(f"Control mean:  {np.mean(ctrl_losses):.4f} +/- {np.std(ctrl_losses):.4f}")
    log.info(f"Control > target: {pct_ctrl_worse:.0f}%")

    if pct_ctrl_worse >= 80:
        log.info("SPECIFICITY CONFIRMED: Target latents outperform most controls.")
    else:
        log.info("SPECIFICITY WEAK: Controls achieve similar improvement.")

    # ══════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════
    log.info("Generating plots...")

    # ── 1. Target vs Control Losses (combined) ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.scatter(
        [0] * len(ctrl_losses), ctrl_losses, alpha=0.7, s=100,
        color=CONTROL_COLOR, label="Control latents", zorder=3, edgecolor="white",
    )
    ax.scatter([1], [target_loss], marker="*", s=400, color=OPTIMAL_COLOR,
               label="Target latents (alpha*)", zorder=4)
    if e1_sum and "combined" in e1_sum:
        ax.axhline(e1_sum["combined"]["loss"], color="gray", linestyle="--",
                    label="Baseline (alpha=0)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control\nLatents", "Target\nLatents"])
    ax.set_ylabel("Combined Adjusted Loss")
    ax.set_title("E3: Target vs Control (Combined)")
    ax.legend()

    # Bar chart of all conditions
    ax = axes[1]
    sorted_combined = combined.sort_values("mean_loss")
    colors = [OPTIMAL_COLOR if t else CONTROL_COLOR for t in sorted_combined["is_target"]]
    ax.barh(sorted_combined["condition"], sorted_combined["mean_loss"], color=colors)
    if e1_sum and "combined" in e1_sum:
        ax.axvline(e1_sum["combined"]["loss"], color="gray", linestyle="--",
                    alpha=0.5, label="Baseline")
    ax.set_xlabel("Combined Adjusted Loss")
    ax.set_title("Adjusted Loss by Condition (lower = better)")
    ax.legend()

    fig.suptitle("E3: Specificity Ablation", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E3_target_vs_control")

    # ── 2. Per-EG Breakdown ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        eg_data = rdf[rdf["exposure_group"] == eg].copy()
        eg_data["is_target"] = eg_data["condition"] == "target"
        eg_data = eg_data.sort_values("loss")
        colors = [OPTIMAL_COLOR if t else CONTROL_COLOR for t in eg_data["is_target"]]
        ax.barh(eg_data["condition"], eg_data["loss"], color=colors)
        ax.set_xlabel("Loss")
        ax.set_title(f"{EG_LABELS[eg]}")
    fig.suptitle("E3: Per-EG Specificity", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E3_per_eg_specificity")

    # ── 3. Boxplot: Control loss distribution ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ctrl_data = rdf[rdf["condition"] != "target"].copy()
    ctrl_data["EG"] = ctrl_data["exposure_group"].map(EG_LABELS)
    sns.boxplot(data=ctrl_data, x="EG", y="loss", color=CONTROL_COLOR, ax=ax)
    # Overlay target
    for eg in EXPOSURE_GROUPS:
        t_val = rdf[(rdf["condition"] == "target") & (rdf["exposure_group"] == eg)]["loss"].values
        if len(t_val) > 0:
            ax.scatter([EXPOSURE_GROUPS.index(eg)], t_val, marker="*", s=300,
                       color=OPTIMAL_COLOR, zorder=5, label="Target" if eg == 0 else "")
    ax.set_ylabel("Loss")
    ax.set_title("E3: Control Distribution vs Target")
    ax.legend()
    save_plot(fig, plot_dir, "E3_boxplot_controls")

    # ── 4. Heatmap: Condition x EG ───────────────────────────────────
    pivot_loss = rdf.pivot(index="condition", columns="exposure_group", values="loss")
    pivot_loss.columns = [EG_LABELS[c] for c in pivot_loss.columns]
    pivot_loss = pivot_loss.sort_values(pivot_loss.columns[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(pivot_loss, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Loss"})
    ax.set_title("E3: Loss by Condition x Exposure Group")
    save_plot(fig, plot_dir, "E3_heatmap_condition_eg")

    # ── 5. Accuracy by Condition ──────────────────────────────────────
    if "mean_accuracy" in rdf.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        acc_combined = rdf.groupby("condition").agg(
            mean_acc=("mean_accuracy", "mean"),
        ).reset_index()
        acc_combined["is_target"] = acc_combined["condition"] == "target"
        acc_combined = acc_combined.sort_values("mean_acc", ascending=False)
        colors = [OPTIMAL_COLOR if t else CONTROL_COLOR for t in acc_combined["is_target"]]
        ax.barh(acc_combined["condition"], acc_combined["mean_acc"], color=colors)
        ax.set_xlabel("Mean Accuracy")
        ax.set_title("E3: Accuracy by Condition")
        save_plot(fig, plot_dir, "E3_accuracy_by_condition")

    log.info(f"All plots saved to {plot_dir}")
    log.info("E3 complete.")


if __name__ == "__main__":
    main()
