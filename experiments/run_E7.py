#!/usr/bin/env python3
"""
E7 -- Individual Differences: Participant Coverage
====================================================
Tests whether varying alpha across the sweep traces a meaningful path
through human metacognitive variance. Maps model profiles alongside human
profiles in PCA space, computes Wasserstein distances, and measures coverage.

Depends on: E2 results.

Saves:
  results/E7/E7_individual_diffs.json
  results/E7/plots/*.png
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
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from config import *
from shared import (
    setup_logging, setup_viz, save_plot,
    load_statements, load_human_data, get_participant_split,
    EG_COLORS, BASELINE_COLOR, OPTIMAL_COLOR, HUMAN_COLOR,
)


def main():
    log = setup_logging("E7")
    setup_viz()

    out_dir  = RESULTS_DIR / "E7"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load human data ──────────────────────────────────────────────────
    human_df = load_human_data()
    calib_ids, test_ids = get_participant_split(human_df)
    all_ids = calib_ids + test_ids

    # Per-participant confidence profile (items as dimensions)
    pivot = (
        human_df[human_df["ID_1"].isin(all_ids)]
        .groupby(["ID_1", "Item"])["Confidence"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    pivot_norm = (pivot - 1) / 3  # normalize to [0,1]
    item_cols = pivot_norm.columns.tolist()
    n_items = len(item_cols)

    # Drop participants with >10% missing
    complete_mask = pivot_norm.isna().mean(axis=1) < 0.1
    profiles_human = pivot_norm[complete_mask].fillna(pivot_norm.mean(axis=0)).values
    participant_ids = pivot_norm[complete_mask].index.tolist()
    log.info(f"Human profiles: {profiles_human.shape} (participants x items)")

    # Per-participant metacognitive ability (conf-acc correlation)
    per_participant = (
        human_df[human_df["ID_1"].isin(all_ids)]
        .groupby("ID_1")
        .apply(lambda g: pd.Series({
            "conf_acc_corr": np.corrcoef(
                (g["Confidence"] - 1) / 3, g["Correct"]
            )[0, 1] if len(g) > 5 and g["Correct"].std() > 0 else np.nan,
            "mean_conf": (g["Confidence"].mean() - 1) / 3,
        }))
        .dropna()
    )
    log.info(f"Metacognitive profiles: {len(per_participant)} participants")

    # ── Load item order ──────────────────────────────────────────────────
    statements = load_statements()
    item_order = statements["item_id"].tolist()
    item_to_idx = {iid: i for i, iid in enumerate(item_cols)}

    # ── Process each EG separately ───────────────────────────────────────
    e7_results = {}

    for eg in EXPOSURE_GROUPS:
        log.info(f"\n{'='*50}")
        log.info(f"EG{eg}")
        log.info(f"{'='*50}")

        sweep_path = RESULTS_DIR / "E2" / f"E2_sweep_EG{eg}.json"
        if not sweep_path.exists():
            log.info(f"  Skipping: E2 sweep for EG{eg} not found")
            continue

        with open(sweep_path) as f:
            sweep = json.load(f)

        # Build model profiles: reorder to match human pivot columns
        model_profiles = []
        alpha_labels = []

        # Compute acc_ratio relative to baseline (0,0)
        base_acc = float(np.mean(sweep["0,0"]["model_accs"])) if "0,0" in sweep else 1.0

        for key, data in sweep.items():
            mc = data["model_confs"]
            sweep_item_ids = data.get("item_ids", item_order)
            profile = np.full(n_items, np.nan)
            for i, iid in enumerate(sweep_item_ids):
                if iid in item_to_idx and i < len(mc):
                    profile[item_to_idx[iid]] = mc[i]
            # Fill NaN with column mean from human data
            nan_mask = np.isnan(profile)
            if nan_mask.any():
                profile[nan_mask] = np.nanmean(profiles_human[:, nan_mask], axis=0)
            model_profiles.append(profile)
            mean_acc = float(np.mean(data["model_accs"]))
            acc_ratio = base_acc / mean_acc if mean_acc > 0 else 1.0
            adjusted_loss = data["loss"] * acc_ratio
            alpha_labels.append({
                "key": key,
                "unsure": data["unsure_coeff"],
                "refusal": data["refusal_coeff"],
                "loss": adjusted_loss,
                "raw_loss": data["loss"],
                "mean_accuracy": mean_acc,
            })

        model_profiles = np.array(model_profiles)
        alpha_df = pd.DataFrame(alpha_labels)
        log.info(f"  Model profiles: {model_profiles.shape}")

        # ── Per-alpha: nearest human and distance metrics ────────────
        results = []
        for i, (m_profile, row) in enumerate(zip(model_profiles, alpha_df.itertuples())):
            dists = np.linalg.norm(profiles_human - m_profile, axis=1)
            nearest_i = int(np.argmin(dists))
            mean_dist = float(np.mean(dists))
            min_dist = float(np.min(dists))

            w_dist_mean = float(np.mean([
                wasserstein_distance(m_profile, h_profile)
                for h_profile in profiles_human
            ]))

            results.append({
                "key": row.key,
                "unsure_coeff": row.unsure,
                "refusal_coeff": row.refusal,
                "loss": row.loss,
                "nearest_participant": participant_ids[nearest_i],
                "l2_to_nearest": min_dist,
                "l2_mean": mean_dist,
                "wasserstein_mean": w_dist_mean,
            })

        results_df = pd.DataFrame(results)

        # Coverage
        threshold = float(profiles_human.std(axis=0).mean())
        covered = set()
        for r in results:
            if r["l2_to_nearest"] < threshold:
                covered.add(r["nearest_participant"])
        coverage_pct = len(covered) / len(participant_ids) * 100

        log.info(f"  Coverage: {coverage_pct:.1f}% (threshold={threshold:.3f})")
        log.info(f"  Wasserstein (best alpha): {results_df['wasserstein_mean'].min():.4f}")

        e7_results[f"EG{eg}"] = {
            "per_alpha": results,
            "coverage_pct": coverage_pct,
            "coverage_threshold": threshold,
        }

        # ══════════════════════════════════════════════════════════════
        # PER-EG VISUALIZATIONS
        # ══════════════════════════════════════════════════════════════

        # ── 1. PCA: Humans + Model alpha-sweep ──────────────────────
        all_profiles = np.vstack([profiles_human, model_profiles])
        n_human = len(profiles_human)
        n_model = len(model_profiles)

        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        pca_coords = pca.fit_transform(all_profiles)
        human_pca = pca_coords[:n_human]
        model_pca = pca_coords[n_human:]
        explained = pca.explained_variance_ratio_

        # Find optimal alpha index
        opt_alpha = json.load(open(RESULTS_DIR / "E2" / "E2_optimal_alpha.json"))
        opt_key = opt_alpha["optimal_alpha"]
        opt_mask = alpha_df["key"] == opt_key
        opt_idx = alpha_df[opt_mask].index[0] if opt_mask.any() else 0

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(human_pca[:, 0], human_pca[:, 1], alpha=0.3, s=25,
                   c=HUMAN_COLOR, label="Human participants")
        sc = ax.scatter(model_pca[:, 0], model_pca[:, 1],
                        c=alpha_df["loss"], cmap="viridis_r", s=70, zorder=3,
                        edgecolor="white", linewidth=0.5, label="Model (alpha grid)")
        plt.colorbar(sc, ax=ax, label="Adjusted Loss")
        ax.scatter(model_pca[opt_idx, 0], model_pca[opt_idx, 1],
                   marker="*", s=400, c="red", zorder=5, label="alpha*")
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        ax.set_title(f"E7: PCA — Human Participants + Model alpha-Sweep ({EG_LABELS[eg]})")
        ax.legend()
        save_plot(fig, plot_dir, f"E7_pca_EG{eg}")

        # ── 2. Wasserstein Heatmap ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        pivot_wass = results_df.pivot(
            index="unsure_coeff", columns="refusal_coeff", values="wasserstein_mean"
        ).sort_index(ascending=False)
        sns.heatmap(pivot_wass, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Mean Wasserstein"})
        ax.set_title(f"E7: Wasserstein Distance to Humans ({EG_LABELS[eg]})")
        save_plot(fig, plot_dir, f"E7_wasserstein_heatmap_EG{eg}")

        # ── 3. L2 Distance Heatmap ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        pivot_l2 = results_df.pivot(
            index="unsure_coeff", columns="refusal_coeff", values="l2_to_nearest"
        ).sort_index(ascending=False)
        sns.heatmap(pivot_l2, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "L2 to Nearest Human"})
        ax.set_title(f"E7: L2 Distance to Nearest Human ({EG_LABELS[eg]})")
        save_plot(fig, plot_dir, f"E7_l2_heatmap_EG{eg}")

        # ── 4. Coverage Curve ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        thresholds = np.linspace(0, profiles_human.std(axis=0).max(), 50)
        coverage_curve = []
        for t in thresholds:
            covered_t = set()
            for r in results:
                if r["l2_to_nearest"] < t:
                    covered_t.add(r["nearest_participant"])
            coverage_curve.append(len(covered_t) / len(participant_ids) * 100)

        ax.plot(thresholds, coverage_curve, linewidth=2, color=EG_COLORS[eg])
        ax.axvline(threshold, color="r", linestyle="--",
                   label=f"1sigma threshold={threshold:.2f}")
        ax.axhline(coverage_pct, color="g", linestyle="--",
                   label=f"Coverage={coverage_pct:.1f}%")
        ax.set_xlabel("L2 Distance Threshold")
        ax.set_ylabel("% Participants Covered")
        ax.set_title(f"E7: Coverage by alpha-Sweep ({EG_LABELS[eg]})")
        ax.legend()
        ax.set_ylim(0, 105)
        save_plot(fig, plot_dir, f"E7_coverage_curve_EG{eg}")

    # ── 5. Metacognitive Ability: Human vs Model ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(per_participant["conf_acc_corr"].dropna(), bins=20, alpha=0.6,
            density=True, label="Human participants", color=HUMAN_COLOR, edgecolor="white")

    # Model conf-acc corrs from sweep (all EGs combined)
    model_corrs = []
    for eg in EXPOSURE_GROUPS:
        sweep_path = RESULTS_DIR / "E2" / f"E2_sweep_EG{eg}.json"
        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep = json.load(f)
            model_corrs.extend([v["corr"] for v in sweep.values()])

    if model_corrs:
        ax.hist(model_corrs, bins=20, alpha=0.6, density=True,
                label="Model (all alpha)", color=OPTIMAL_COLOR, edgecolor="white")

    ax.set_xlabel("Confidence-Accuracy Correlation")
    ax.set_ylabel("Density")
    ax.set_title("E7: Model alpha-Sweep vs Human Metacognitive Ability Distribution")
    ax.legend()
    save_plot(fig, plot_dir, "E7_metacog_distribution")

    # ── 6. Mean Confidence: Human vs Model ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(per_participant["mean_conf"].dropna(), bins=20, alpha=0.6,
            density=True, label="Human participants", color=HUMAN_COLOR, edgecolor="white")

    model_mean_confs = []
    for eg in EXPOSURE_GROUPS:
        sweep_path = RESULTS_DIR / "E2" / f"E2_sweep_EG{eg}.json"
        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep = json.load(f)
            model_mean_confs.extend([np.mean(v["model_confs"]) for v in sweep.values()])

    if model_mean_confs:
        ax.hist(model_mean_confs, bins=20, alpha=0.6, density=True,
                label="Model (all alpha)", color=OPTIMAL_COLOR, edgecolor="white")

    ax.set_xlabel("Mean Confidence (normalized)")
    ax.set_ylabel("Density")
    ax.set_title("E7: Mean Confidence Distribution — Human vs Model")
    ax.legend()
    save_plot(fig, plot_dir, "E7_mean_conf_distribution")

    # ── 7. Model Accuracy Distribution ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    model_accs_all = []
    for eg in EXPOSURE_GROUPS:
        sweep_path = RESULTS_DIR / "E2" / f"E2_sweep_EG{eg}.json"
        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep = json.load(f)
            model_accs_all.extend([float(np.mean(v["model_accs"])) for v in sweep.values()])
    if model_accs_all:
        ax.hist(model_accs_all, bins=20, alpha=0.6, density=True,
                label="Model (all alpha)", color=OPTIMAL_COLOR, edgecolor="white")
    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel("Density")
    ax.set_title("E7: Model Accuracy Distribution Across alpha-Sweep")
    ax.legend()
    save_plot(fig, plot_dir, "E7_accuracy_distribution")

    # ── 8. Jointplot: L2 distance vs Loss ────────────────────────────
    for eg in EXPOSURE_GROUPS:
        eg_key = f"EG{eg}"
        if eg_key not in e7_results:
            continue
        rdf = pd.DataFrame(e7_results[eg_key]["per_alpha"])
        g = sns.jointplot(
            data=rdf, x="loss", y="l2_to_nearest",
            kind="reg", color=EG_COLORS[eg], height=7,
        )
        g.set_axis_labels("Loss L(alpha)", "L2 Distance to Nearest Human")
        g.figure.suptitle(f"E7: Loss vs L2 Distance ({EG_LABELS[eg]})", y=1.02)
        save_plot(g.figure, plot_dir, f"E7_jointplot_loss_l2_EG{eg}")

    # ── Save all results ─────────────────────────────────────────────────
    with open(out_dir / "E7_individual_diffs.json", "w") as f:
        json.dump(e7_results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    for eg_key, data in e7_results.items():
        cov = data["coverage_pct"]
        log.info(f"\n{eg_key}:")
        log.info(f"  Coverage: {cov:.1f}%")
        if cov > 50:
            log.info(f"  HYPOTHESIS SUPPORTED: alpha sweep covers >50% of individuals.")
        else:
            log.info(f"  PARTIAL: Model variance spans only part of human individual-difference space.")

    log.info(f"\nAll plots saved to {plot_dir}")
    log.info("E7 complete.")


if __name__ == "__main__":
    main()
