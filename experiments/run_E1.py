#!/usr/bin/env python3
"""
E1 -- Baseline: Unsteered Model Performance
============================================
Runs unsteered Gemma on all 80 items x 2 exposure groups (EG0, EG1).
Two-turn confidence elicitation: Gemma generates completion -> expresses
confidence -> Claude scores 1-4. Claude also codes responses as
Correct/Lure/Unsure/Other.

Saves:
  results/E1/E1_baseline_EG0.json
  results/E1/E1_baseline_EG1.json
  results/E1/E1_summary.json
  results/E1/plots/*.png
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
    load_items, load_human_data, get_participant_split,
    compute_human_norms, align_model_human, collect_response_categories,
    run_condition, compute_loss, get_client, load_model, load_saes,
    EG_COLORS, BASELINE_COLOR, HUMAN_COLOR,
)


def main():
    log = setup_logging("E1")
    setup_viz()

    out_dir  = RESULTS_DIR / "E1"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────
    log.info("Loading model + SAEs...")
    model  = load_model(hf_token=HF_TOKEN)
    client = get_client()
    log.info("Model loaded.")

    # ── Load human data & split ──────────────────────────────────────────
    human_df = load_human_data()
    calib_ids, test_ids = get_participant_split(human_df)
    log.info(f"Participants: {len(calib_ids)} calib, {len(test_ids)} test")

    # ── Run baseline for each exposure group ─────────────────────────────
    eg_results = {}
    eg_summaries = {}

    for eg in EXPOSURE_GROUPS:
        log.info(f"\n{'='*60}")
        log.info(f"Running EG{eg} baseline (unsteered)...")
        log.info(f"{'='*60}")

        items = load_items(eg)
        save_path = out_dir / f"E1_baseline_EG{eg}.json"

        # Resume from checkpoint
        existing = {}
        if save_path.exists():
            with open(save_path) as f:
                existing = json.load(f)
            log.info(f"Resuming: {len(existing)}/{len(items)} items cached")

        results = run_condition(
            model, items, interventions=None, client=client,
            n_samples=N_SAMPLES, logger=log,
            checkpoint_path=save_path, existing_results=existing,
        )

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved {save_path}")

        eg_results[eg] = results

        # ── Compute metrics against human norms for this EG ──────────
        human_norms = compute_human_norms(human_df, calib_ids, exposure_group=eg)

        if len(human_norms) == 0:
            log.info(f"  No human data for Exposure={eg}, using all calib data")
            human_norms = compute_human_norms(human_df, calib_ids)

        df = align_model_human(results, human_norms)
        loss, ece, conf_acc_r = compute_loss(
            df["model_conf"].tolist(),
            df["human_conf_norm"].tolist(),
            df["model_acc"].tolist(),
        )
        r_human, p_human = stats.pearsonr(df["model_conf"], df["human_conf_norm"])

        summary = {
            "exposure_group": eg,
            "ece": ece,
            "conf_acc_corr": conf_acc_r,
            "loss": loss,
            "r_human": float(r_human),
            "p_human": float(p_human),
            "n_items": len(df),
            "mean_model_conf": float(df["model_conf"].mean()),
            "mean_human_conf": float(df["human_conf_norm"].mean()),
            "mean_model_acc": float(df["model_acc"].mean()),
        }
        eg_summaries[f"EG{eg}"] = summary

        log.info(f"EG{eg} Results:")
        log.info(f"  ECE:            {ece:.4f}")
        log.info(f"  Conf-Acc corr:  {conf_acc_r:.4f}")
        log.info(f"  Loss:           {loss:.4f}")
        log.info(f"  r(model,human): {r_human:.4f}  p={p_human:.6f}")
        log.info(f"  Model conf:     {df['model_conf'].mean():.3f}")
        log.info(f"  Human conf:     {df['human_conf_norm'].mean():.3f}")

    # ── Combined summary ─────────────────────────────────────────────────
    combined_loss = np.mean([s["loss"] for s in eg_summaries.values()])
    combined_ece  = np.mean([s["ece"]  for s in eg_summaries.values()])
    eg_summaries["combined"] = {"loss": combined_loss, "ece": combined_ece}

    with open(out_dir / "E1_summary.json", "w") as f:
        json.dump(eg_summaries, f, indent=2)
    log.info(f"\nCombined loss: {combined_loss:.4f}, Combined ECE: {combined_ece:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════
    log.info("Generating plots...")

    # Prepare merged DataFrames for each EG
    dfs = {}
    for eg in EXPOSURE_GROUPS:
        human_norms = compute_human_norms(human_df, calib_ids, exposure_group=eg)
        if len(human_norms) == 0:
            human_norms = compute_human_norms(human_df, calib_ids)
        df = align_model_human(eg_results[eg], human_norms)
        df["eg"] = eg
        df["eg_label"] = EG_LABELS[eg]
        dfs[eg] = df

    all_df = pd.concat(dfs.values(), ignore_index=True)

    # ── 1. Model vs Human Confidence Scatter (per EG) ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        d = dfs[eg]
        ax.scatter(d["human_conf_norm"], d["model_conf"],
                   alpha=0.6, color=EG_COLORS[eg], s=50, edgecolor="white")
        ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Perfect calibration")
        r_val = eg_summaries[f"EG{eg}"]["r_human"]
        ece_val = eg_summaries[f"EG{eg}"]["ece"]
        ax.set_xlabel("Human Confidence (normalized)")
        ax.set_ylabel("Model Confidence")
        ax.set_title(f"{EG_LABELS[eg]}\nr={r_val:.3f}, ECE={ece_val:.3f}")
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("E1 Baseline: Model vs Human Confidence", fontsize=14, y=1.02)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E1_scatter_model_vs_human")

    # ── 2. Joint distribution (jointplot per EG) ─────────────────────
    for eg in EXPOSURE_GROUPS:
        d = dfs[eg]
        g = sns.jointplot(
            data=d, x="human_conf_norm", y="model_conf",
            kind="reg", color=EG_COLORS[eg], height=7,
            marginal_kws=dict(bins=15, kde=True),
            joint_kws=dict(scatter_kws=dict(alpha=0.5, s=40)),
        )
        g.set_axis_labels("Human Confidence", "Model Confidence")
        g.figure.suptitle(f"E1: Joint Distribution - {EG_LABELS[eg]}", y=1.02)
        save_plot(g.figure, plot_dir, f"E1_jointplot_EG{eg}")

    # ── 3. Confidence Distribution Comparison ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        d = dfs[eg]
        ax.hist(d["model_conf"], bins=12, alpha=0.6, density=True,
                color=EG_COLORS[eg], label="Model", edgecolor="white")
        ax.hist(d["human_conf_norm"], bins=12, alpha=0.5, density=True,
                color=HUMAN_COLOR, label="Human", edgecolor="white")
        ax.set_xlabel("Confidence (normalized)")
        ax.set_ylabel("Density")
        ax.set_title(EG_LABELS[eg])
        ax.legend()
    fig.suptitle("E1: Confidence Distributions (Model vs Human)", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E1_confidence_distributions")

    # ── 4. Violin plot: Model confidence by EG ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    violin_data = []
    for eg in EXPOSURE_GROUPS:
        for iid, v in eg_results[eg].items():
            for cs in v["confidence_scores"]:
                violin_data.append({"EG": EG_LABELS[eg], "Confidence": cs})
    vdf = pd.DataFrame(violin_data)
    sns.violinplot(data=vdf, x="EG", y="Confidence", palette=EG_COLORS.values(),
                   inner="box", ax=ax)
    ax.set_title("E1: Model Confidence Distribution by Exposure Group")
    ax.set_ylabel("Confidence (normalized)")
    save_plot(fig, plot_dir, "E1_violin_confidence_by_EG")

    # ── 5. Response Category Distribution (model vs human) ───────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cat_order = ["Correct", "Lure", "Unsure", "Other"]

    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        # Model categories
        model_cats = collect_response_categories(eg_results[eg])
        model_counts = model_cats["category"].value_counts(normalize=True)
        model_counts = model_counts.reindex(cat_order, fill_value=0)

        # Human categories from human_df
        h_sub = human_df[
            (human_df["ID_1"].isin(calib_ids)) &
            (human_df["Exposure"] == eg)
        ]
        if len(h_sub) == 0:
            h_sub = human_df[human_df["ID_1"].isin(calib_ids)]

        human_cat_map = {
            "Correct": h_sub["Correct"].mean(),
            "Lure": h_sub["Error"].mean(),
            "Unsure": h_sub["Unsure"].mean(),
            "Other": h_sub["Other"].mean(),
        }

        x = np.arange(len(cat_order))
        w = 0.35
        ax.bar(x - w / 2, [model_counts.get(c, 0) for c in cat_order],
               w, label="Model", color=EG_COLORS[eg], alpha=0.8)
        ax.bar(x + w / 2, [human_cat_map[c] for c in cat_order],
               w, label="Human", color=HUMAN_COLOR, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_order)
        ax.set_ylabel("Proportion")
        ax.set_title(EG_LABELS[eg])
        ax.legend()

    fig.suptitle("E1: Response Category Proportions (Model vs Human)", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E1_response_categories")

    # ── 6. Boxplot: Confidence by Response Category ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        box_data = []
        for iid, v in eg_results[eg].items():
            for cs, cat in zip(v["confidence_scores"], v["response_categories"]):
                box_data.append({"Category": cat, "Confidence": cs})
        bdf = pd.DataFrame(box_data)
        if len(bdf) > 0:
            sns.boxplot(data=bdf, x="Category", y="Confidence",
                        order=cat_order, palette="Set2", ax=ax)
            sns.stripplot(data=bdf, x="Category", y="Confidence",
                          order=cat_order, color="black", alpha=0.3, size=3, ax=ax)
        ax.set_title(EG_LABELS[eg])
        ax.set_ylabel("Confidence (normalized)")
    fig.suptitle("E1: Confidence by Response Category", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E1_boxplot_conf_by_category")

    # ── 7. Conf-Acc Correlation Scatter ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        d = dfs[eg]
        ax.scatter(d["model_conf"], d["model_acc"], alpha=0.6,
                   color=EG_COLORS[eg], s=50, edgecolor="white")
        if len(d) > 2 and d["model_conf"].std() > 1e-8:
            m_fit, b_fit = np.polyfit(d["model_conf"], d["model_acc"], 1)
            x_line = np.linspace(d["model_conf"].min(), d["model_conf"].max(), 100)
            y_line = np.clip(m_fit * x_line + b_fit, 0, 1)
            ax.plot(x_line, y_line, "r-", alpha=0.7)
        r_val = eg_summaries[f"EG{eg}"]["conf_acc_corr"]
        ax.set_xlabel("Model Confidence")
        ax.set_ylabel("Model Accuracy")
        ax.set_title(f"{EG_LABELS[eg]}\nr(conf,acc)={r_val:.3f}")
    fig.suptitle("E1: Confidence vs Accuracy", fontsize=14, y=1.02)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E1_conf_vs_accuracy")

    # ── 8. Summary Heatmap: EG x Metric ─────────────────────────────
    metrics = ["ece", "conf_acc_corr", "loss", "r_human", "mean_model_conf"]
    metric_labels = ["ECE", "Conf-Acc r", "Loss", "r(model,human)", "Mean Conf"]
    heat_data = []
    for eg in EXPOSURE_GROUPS:
        row = [eg_summaries[f"EG{eg}"][m] for m in metrics]
        heat_data.append(row)
    heat_df = pd.DataFrame(heat_data, index=[EG_LABELS[eg] for eg in EXPOSURE_GROUPS],
                           columns=metric_labels)

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=ax,
                linewidths=1, cbar_kws={"label": "Value"})
    ax.set_title("E1: Baseline Metrics by Exposure Group")
    save_plot(fig, plot_dir, "E1_metrics_heatmap")

    # ── 9. EG Comparison Bar Chart ───────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, label in zip(
        axes,
        ["ece", "conf_acc_corr", "loss"],
        ["ECE (lower = better)", "Conf-Acc Corr (higher = better)", "Joint Loss (lower = better)"],
    ):
        vals = [eg_summaries[f"EG{eg}"][metric] for eg in EXPOSURE_GROUPS]
        bars = ax.bar(
            [EG_LABELS[eg] for eg in EXPOSURE_GROUPS], vals,
            color=[EG_COLORS[eg] for eg in EXPOSURE_GROUPS],
        )
        ax.set_title(label)
        ax.set_ylabel("Value")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    fig.suptitle("E1: Baseline Comparison Across Exposure Groups", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E1_eg_comparison_bars")

    log.info(f"All plots saved to {plot_dir}")
    log.info("E1 complete.")


if __name__ == "__main__":
    main()
