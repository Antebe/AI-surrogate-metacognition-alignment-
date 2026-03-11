#!/usr/bin/env python3
"""
E8 -- Steered Model Performance (mirrors E1 with optimal steering)
===================================================================
Runs Gemma with optimal steering (from E2) on all 80 items x exposure groups.
Same two-turn confidence elicitation as E1, but with SAE interventions active.

Depends on: E2 (optimal alpha)

Saves:
  results/E8/E8_steered_EG0.json
  results/E8/E8_steered_EG1.json
  results/E8/E8_summary.json
  results/E8/plots/*.png
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
    make_target_interventions,
    EG_COLORS, BASELINE_COLOR, HUMAN_COLOR, OPTIMAL_COLOR,
)


def main():
    log = setup_logging("E8")
    setup_viz()

    out_dir  = RESULTS_DIR / "E8"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Check E2 dependency ───────────────────────────────────────────────
    opt_path = RESULTS_DIR / "E2" / "E2_optimal_alpha.json"
    if not opt_path.exists():
        raise FileNotFoundError(f"E2 optimal alpha not found: {opt_path}. Run E2 first.")
    with open(opt_path) as f:
        opt = json.load(f)
    OPT_UNSURE  = opt["unsure_coeff"]
    OPT_REFUSAL = opt["refusal_coeff"]
    log.info(f"Optimal alpha*: unsure={OPT_UNSURE}, refusal={OPT_REFUSAL}")

    # ── Load model + SAEs ─────────────────────────────────────────────────
    log.info("Loading model + SAEs...")
    model = load_model(hf_token=HF_TOKEN)
    saes  = load_saes(layers=SAE_LAYERS)
    client = get_client()
    log.info("Model loaded.")

    interventions = make_target_interventions(saes, OPT_UNSURE, OPT_REFUSAL)
    log.info(f"Steering with {len(interventions)} interventions")

    # ── Load human data & split ───────────────────────────────────────────
    human_df = load_human_data()
    calib_ids, test_ids = get_participant_split(human_df)
    log.info(f"Participants: {len(calib_ids)} calib, {len(test_ids)} test")

    # ── Run steered model for each exposure group ─────────────────────────
    eg_results = {}
    eg_summaries = {}

    for eg in EXPOSURE_GROUPS:
        log.info(f"\n{'='*60}")
        log.info(f"Running EG{eg} steered (unsure={OPT_UNSURE}, refusal={OPT_REFUSAL})...")
        log.info(f"{'='*60}")

        items = load_items(eg)
        save_path = out_dir / f"E8_steered_EG{eg}.json"

        # Resume from checkpoint
        existing = {}
        if save_path.exists():
            with open(save_path) as f:
                existing = json.load(f)
            log.info(f"Resuming: {len(existing)}/{len(items)} items cached")

        results = run_condition(
            model, items, interventions=interventions, client=client,
            n_samples=N_SAMPLES, logger=log,
            checkpoint_path=save_path, existing_results=existing,
        )

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved {save_path}")

        eg_results[eg] = results

        # ── Compute metrics against human norms for this EG ───────────
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
            "unsure_coeff": OPT_UNSURE,
            "refusal_coeff": OPT_REFUSAL,
        }
        eg_summaries[f"EG{eg}"] = summary

        log.info(f"EG{eg} Results:")
        log.info(f"  ECE:            {ece:.4f}")
        log.info(f"  Conf-Acc corr:  {conf_acc_r:.4f}")
        log.info(f"  Loss:           {loss:.4f}")
        log.info(f"  r(model,human): {r_human:.4f}  p={p_human:.6f}")
        log.info(f"  Model conf:     {df['model_conf'].mean():.3f}")
        log.info(f"  Human conf:     {df['human_conf_norm'].mean():.3f}")
        log.info(f"  Model acc:      {df['model_acc'].mean():.3f}")

    # ── Combined summary ──────────────────────────────────────────────────
    combined_loss = np.mean([s["loss"] for s in eg_summaries.values()])
    combined_ece  = np.mean([s["ece"]  for s in eg_summaries.values()])
    eg_summaries["combined"] = {"loss": combined_loss, "ece": combined_ece}

    with open(out_dir / "E8_summary.json", "w") as f:
        json.dump(eg_summaries, f, indent=2)
    log.info(f"\nCombined loss: {combined_loss:.4f}, Combined ECE: {combined_ece:.4f}")

    # ── Load E1 baseline for comparison ───────────────────────────────────
    e1_summary_path = RESULTS_DIR / "E1" / "E1_summary.json"
    e1_summaries = {}
    if e1_summary_path.exists():
        with open(e1_summary_path) as f:
            e1_summaries = json.load(f)
        log.info("Loaded E1 baseline for comparison plots")

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

    # ── 1. Model vs Human Confidence Scatter (per EG) ─────────────────
    fig, axes = plt.subplots(1, len(EXPOSURE_GROUPS), figsize=(7 * len(EXPOSURE_GROUPS), 6),
                             squeeze=False)
    axes = axes[0]
    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        d = dfs[eg]
        ax.scatter(d["human_conf_norm"], d["model_conf"],
                   alpha=0.6, color=OPTIMAL_COLOR, s=50, edgecolor="white")
        ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="Perfect calibration")
        r_val = eg_summaries[f"EG{eg}"]["r_human"]
        ece_val = eg_summaries[f"EG{eg}"]["ece"]
        ax.set_xlabel("Human Confidence (normalized)")
        ax.set_ylabel("Model Confidence")
        ax.set_title(f"{EG_LABELS[eg]} (steered)\nr={r_val:.3f}, ECE={ece_val:.3f}")
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("E8 Steered: Model vs Human Confidence", fontsize=14, y=1.02)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E8_scatter_model_vs_human")

    # ── 2. Joint distribution (jointplot per EG) ──────────────────────
    for eg in EXPOSURE_GROUPS:
        d = dfs[eg]
        g = sns.jointplot(
            data=d, x="human_conf_norm", y="model_conf",
            kind="reg", color=OPTIMAL_COLOR, height=7,
            marginal_kws=dict(bins=15, kde=True),
            joint_kws=dict(scatter_kws=dict(alpha=0.5, s=40)),
        )
        g.set_axis_labels("Human Confidence", "Model Confidence")
        g.figure.suptitle(f"E8: Joint Distribution - {EG_LABELS[eg]} (steered)", y=1.02)
        save_plot(g.figure, plot_dir, f"E8_jointplot_EG{eg}")

    # ── 3. Confidence Distribution Comparison ─────────────────────────
    fig, axes = plt.subplots(1, len(EXPOSURE_GROUPS), figsize=(7 * len(EXPOSURE_GROUPS), 5),
                             squeeze=False)
    axes = axes[0]
    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        d = dfs[eg]
        ax.hist(d["model_conf"], bins=12, alpha=0.6, density=True,
                color=OPTIMAL_COLOR, label="Model (steered)", edgecolor="white")
        ax.hist(d["human_conf_norm"], bins=12, alpha=0.5, density=True,
                color=HUMAN_COLOR, label="Human", edgecolor="white")
        ax.set_xlabel("Confidence (normalized)")
        ax.set_ylabel("Density")
        ax.set_title(EG_LABELS[eg])
        ax.legend()
    fig.suptitle("E8: Confidence Distributions (Steered Model vs Human)", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E8_confidence_distributions")

    # ── 4. Violin plot: Model confidence by EG ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    violin_data = []
    for eg in EXPOSURE_GROUPS:
        for iid, v in eg_results[eg].items():
            for cs in v["confidence_scores"]:
                violin_data.append({"EG": EG_LABELS[eg], "Confidence": cs})
    vdf = pd.DataFrame(violin_data)
    sns.violinplot(data=vdf, x="EG", y="Confidence", palette=[OPTIMAL_COLOR] * len(EXPOSURE_GROUPS),
                   inner="box", ax=ax)
    ax.set_title("E8: Steered Model Confidence Distribution by Exposure Group")
    ax.set_ylabel("Confidence (normalized)")
    save_plot(fig, plot_dir, "E8_violin_confidence_by_EG")

    # ── 5. Response Category Distribution (model vs human) ────────────
    fig, axes = plt.subplots(1, len(EXPOSURE_GROUPS), figsize=(7 * len(EXPOSURE_GROUPS), 5),
                             squeeze=False)
    axes = axes[0]
    cat_order = ["Correct", "Lure", "Unsure", "Other"]

    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        model_cats = collect_response_categories(eg_results[eg])
        model_counts = model_cats["category"].value_counts(normalize=True)
        model_counts = model_counts.reindex(cat_order, fill_value=0)

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
               w, label="Model (steered)", color=OPTIMAL_COLOR, alpha=0.8)
        ax.bar(x + w / 2, [human_cat_map[c] for c in cat_order],
               w, label="Human", color=HUMAN_COLOR, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_order)
        ax.set_ylabel("Proportion")
        ax.set_title(EG_LABELS[eg])
        ax.legend()

    fig.suptitle("E8: Response Category Proportions (Steered Model vs Human)", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E8_response_categories")

    # ── 6. Boxplot: Confidence by Response Category ───────────────────
    fig, axes = plt.subplots(1, len(EXPOSURE_GROUPS), figsize=(7 * len(EXPOSURE_GROUPS), 5),
                             squeeze=False)
    axes = axes[0]
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
    fig.suptitle("E8: Confidence by Response Category (Steered)", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E8_boxplot_conf_by_category")

    # ── 7. Conf-Acc Correlation Scatter ───────────────────────────────
    fig, axes = plt.subplots(1, len(EXPOSURE_GROUPS), figsize=(7 * len(EXPOSURE_GROUPS), 6),
                             squeeze=False)
    axes = axes[0]
    for ax, eg in zip(axes, EXPOSURE_GROUPS):
        d = dfs[eg]
        ax.scatter(d["model_conf"], d["model_acc"], alpha=0.6,
                   color=OPTIMAL_COLOR, s=50, edgecolor="white")
        if len(d) > 2 and d["model_conf"].std() > 1e-8:
            m_fit, b_fit = np.polyfit(d["model_conf"], d["model_acc"], 1)
            x_line = np.linspace(d["model_conf"].min(), d["model_conf"].max(), 100)
            y_line = np.clip(m_fit * x_line + b_fit, 0, 1)
            ax.plot(x_line, y_line, "r-", alpha=0.7)
        r_val = eg_summaries[f"EG{eg}"]["conf_acc_corr"]
        ax.set_xlabel("Model Confidence")
        ax.set_ylabel("Model Accuracy")
        ax.set_title(f"{EG_LABELS[eg]} (steered)\nr(conf,acc)={r_val:.3f}")
    fig.suptitle("E8: Confidence vs Accuracy (Steered)", fontsize=14, y=1.02)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E8_conf_vs_accuracy")

    # ── 8. Summary Heatmap: EG x Metric ──────────────────────────────
    metrics = ["ece", "conf_acc_corr", "loss", "r_human", "mean_model_conf", "mean_model_acc"]
    metric_labels = ["ECE", "Conf-Acc r", "Loss", "r(model,human)", "Mean Conf", "Mean Acc"]
    heat_data = []
    for eg in EXPOSURE_GROUPS:
        row = [eg_summaries[f"EG{eg}"][m] for m in metrics]
        heat_data.append(row)
    heat_df = pd.DataFrame(heat_data, index=[EG_LABELS[eg] for eg in EXPOSURE_GROUPS],
                           columns=metric_labels)

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=ax,
                linewidths=1, cbar_kws={"label": "Value"})
    ax.set_title("E8: Steered Metrics by Exposure Group")
    save_plot(fig, plot_dir, "E8_metrics_heatmap")

    # ── 9. EG Comparison Bar Chart ────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, metric, label in zip(
        axes,
        ["ece", "conf_acc_corr", "loss", "mean_model_acc"],
        ["ECE (lower = better)", "Conf-Acc Corr (higher = better)",
         "Joint Loss (lower = better)", "Accuracy"],
    ):
        vals = [eg_summaries[f"EG{eg}"][metric] for eg in EXPOSURE_GROUPS]
        bars = ax.bar(
            [EG_LABELS[eg] for eg in EXPOSURE_GROUPS], vals,
            color=[OPTIMAL_COLOR] * len(EXPOSURE_GROUPS),
        )
        ax.set_title(label)
        ax.set_ylabel("Value")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    fig.suptitle("E8: Steered Comparison Across Exposure Groups", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E8_eg_comparison_bars")

    # ── 10. E1 vs E8 Comparison (if E1 results available) ────────────
    if e1_summaries:
        compare_metrics = ["ece", "conf_acc_corr", "loss"]
        compare_labels = ["ECE", "Conf-Acc Corr", "Loss"]

        for eg in EXPOSURE_GROUPS:
            eg_key = f"EG{eg}"
            if eg_key not in e1_summaries:
                continue

            fig, axes = plt.subplots(1, len(compare_metrics), figsize=(5 * len(compare_metrics), 5))
            if len(compare_metrics) == 1:
                axes = [axes]
            for ax, metric, label in zip(axes, compare_metrics, compare_labels):
                e1_val = e1_summaries[eg_key].get(metric, 0)
                e8_val = eg_summaries[eg_key][metric]
                bars = ax.bar(
                    ["E1 Baseline", "E8 Steered"],
                    [e1_val, e8_val],
                    color=[BASELINE_COLOR, OPTIMAL_COLOR],
                )
                ax.set_title(label)
                ax.set_ylabel("Value")
                for bar, val in zip(bars, [e1_val, e8_val]):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=11)
            fig.suptitle(f"E1 Baseline vs E8 Steered — {EG_LABELS[eg]}", fontsize=14)
            plt.tight_layout()
            save_plot(fig, plot_dir, f"E8_vs_E1_EG{eg}")

        # Accuracy comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, eg in enumerate(EXPOSURE_GROUPS):
            eg_key = f"EG{eg}"
            if eg_key not in e1_summaries:
                continue
            e1_acc = e1_summaries[eg_key].get("mean_model_acc", 0)
            e8_acc = eg_summaries[eg_key]["mean_model_acc"]
            x_pos = np.array([0, 1]) + i * 3
            bars = ax.bar(x_pos, [e1_acc, e8_acc],
                          color=[BASELINE_COLOR, OPTIMAL_COLOR], width=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(["E1 Baseline", "E8 Steered"])
            for bar, val in zip(bars, [e1_acc, e8_acc]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=11)
        ax.set_ylabel("Mean Accuracy")
        ax.set_title("E1 Baseline vs E8 Steered: Accuracy")
        plt.tight_layout()
        save_plot(fig, plot_dir, "E8_vs_E1_accuracy")

        log.info("E1 vs E8 comparison plots saved")

    log.info(f"All plots saved to {plot_dir}")
    log.info("E8 complete.")


if __name__ == "__main__":
    main()
