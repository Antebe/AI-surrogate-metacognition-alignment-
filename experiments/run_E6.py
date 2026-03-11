#!/usr/bin/env python3
"""
E6 -- Generalization to Held-Out Participants
===============================================
Tests whether alpha* (optimized on 100 calib participants) transfers to
the 66 held-out test participants. Computed for both EG0 and EG1.

Depends on: E1, E2 results.

Saves:
  results/E6/E6_generalization.json
  results/E6/plots/*.png
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
    compute_human_norms,
    EG_COLORS, BASELINE_COLOR, OPTIMAL_COLOR, HUMAN_COLOR,
)


def compute_loss_from_dicts(
    model_conf_dict: dict, human_norm_df: pd.DataFrame,
    model_acc_dict: dict | None = None,
    baseline_acc: float = 1.0,
) -> tuple[float, float, float, float, float, list]:
    """Evaluate adjusted loss given model dicts and human norms DataFrame.

    Returns (adjusted_loss, ece, corr, mean_acc, acc_ratio, items).
    """
    h_map = dict(zip(human_norm_df["item_id"], human_norm_df["human_conf_norm"]))
    items = sorted(set(model_conf_dict.keys()) & set(h_map.keys()))
    mc = np.array([model_conf_dict[i] for i in items])
    hc = np.array([h_map[i] for i in items])
    ece = float(np.mean(np.abs(mc - hc)))
    if model_acc_dict and np.std(mc) > 1e-8:
        ma = np.array([model_acc_dict.get(i, 0.5) for i in items])
        corr = float(np.corrcoef(mc, ma)[0, 1]) if np.std(ma) > 1e-8 else 0.0
        mean_acc = float(np.mean(ma))
    else:
        corr = 0.0
        mean_acc = 0.0
    raw_loss = LAMBDA1 * ece + LAMBDA2 * (1 - corr)
    acc_ratio = baseline_acc / mean_acc if mean_acc > 0 else 1.0
    adjusted_loss = raw_loss * acc_ratio
    return adjusted_loss, ece, corr, mean_acc, acc_ratio, items


def main():
    log = setup_logging("E6")
    setup_viz()

    out_dir  = RESULTS_DIR / "E6"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prerequisites ───────────────────────────────────────────────
    with open(RESULTS_DIR / "E2" / "E2_optimal_alpha.json") as f:
        opt = json.load(f)
    OPT_KEY = opt["optimal_alpha"]
    log.info(f"Optimal alpha*: {OPT_KEY}")

    # ── Load participant split ───────────────────────────────────────────
    human_df = load_human_data()
    calib_ids, test_ids = get_participant_split(human_df)
    log.info(f"Calib: {len(calib_ids)}, Test: {len(test_ids)}")

    # ── Load item order ──────────────────────────────────────────────────
    statements = load_statements()
    item_order = statements["item_id"].tolist()

    # ── Evaluate on calib vs test for each EG ────────────────────────────
    e6_results = {}

    for eg in EXPOSURE_GROUPS:
        log.info(f"\n{'='*50}")
        log.info(f"EG{eg}")
        log.info(f"{'='*50}")

        # Human norms for calib and test splits
        calib_norms = compute_human_norms(human_df, calib_ids, exposure_group=eg)
        test_norms  = compute_human_norms(human_df, test_ids, exposure_group=eg)

        if len(calib_norms) == 0:
            calib_norms = compute_human_norms(human_df, calib_ids)
        if len(test_norms) == 0:
            test_norms = compute_human_norms(human_df, test_ids)

        # Load E1 baseline
        e1_path = RESULTS_DIR / "E1" / f"E1_baseline_EG{eg}.json"
        if e1_path.exists():
            with open(e1_path) as f:
                e1_data = json.load(f)
            zero_confs = {k: v["mean_confidence"] for k, v in e1_data.items()}
        else:
            zero_confs = {}
            log.info(f"  Warning: E1 baseline for EG{eg} not found")

        # Load E2 sweep for optimal alpha
        sweep_path = RESULTS_DIR / "E2" / f"E2_sweep_EG{eg}.json"
        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            opt_data = sweep_data.get(OPT_KEY, {})
            opt_confs = {iid: c for iid, c in zip(
                opt_data.get("item_ids", item_order),
                opt_data.get("model_confs", []),
            )}
            opt_accs = {iid: a for iid, a in zip(
                opt_data.get("item_ids", item_order),
                opt_data.get("model_accs", []),
            )}
        else:
            opt_confs = {}
            opt_accs = {}
            log.info(f"  Warning: E2 sweep for EG{eg} not found")

        if not zero_confs or not opt_confs:
            continue

        # Baseline accuracy for acc_ratio computation
        zero_accs = {k: v["mean_accuracy"] for k, v in e1_data.items()}
        _, _, _, base_acc, _, _ = compute_loss_from_dicts(zero_confs, calib_norms, zero_accs)

        # Evaluate on calib split
        loss_cb, ece_cb, corr_cb, acc_cb, _, _ = compute_loss_from_dicts(
            zero_confs, calib_norms, zero_accs, baseline_acc=base_acc)
        loss_co, ece_co, corr_co, acc_co, ratio_co, _ = compute_loss_from_dicts(
            opt_confs, calib_norms, opt_accs, baseline_acc=base_acc)

        # Evaluate on TEST split
        loss_tb, ece_tb, corr_tb, acc_tb, _, _ = compute_loss_from_dicts(
            zero_confs, test_norms, zero_accs, baseline_acc=base_acc)
        loss_to, ece_to, corr_to, acc_to, ratio_to, _ = compute_loss_from_dicts(
            opt_confs, test_norms, opt_accs, baseline_acc=base_acc)

        delta_calib = loss_cb - loss_co
        delta_test  = loss_tb - loss_to
        gen_ratio = delta_test / max(delta_calib, 1e-8)

        e6_results[f"EG{eg}"] = {
            "calib": {
                "loss_baseline": loss_cb, "loss_optimal": loss_co,
                "ece_baseline": ece_cb, "ece_optimal": ece_co,
                "acc_baseline": acc_cb, "acc_optimal": acc_co,
                "delta_loss": delta_calib,
            },
            "test": {
                "loss_baseline": loss_tb, "loss_optimal": loss_to,
                "ece_baseline": ece_tb, "ece_optimal": ece_to,
                "acc_baseline": acc_tb, "acc_optimal": acc_to,
                "delta_loss": delta_test,
            },
            "generalization_ratio": gen_ratio,
        }

        log.info(f"                    Baseline    Optimal    ΔLOSS")
        log.info(f"  CALIB split:      {loss_cb:.4f}      {loss_co:.4f}     {delta_calib:+.4f}  acc={acc_co:.4f}")
        log.info(f"  TEST  split:      {loss_tb:.4f}      {loss_to:.4f}     {delta_test:+.4f}  acc={acc_to:.4f}")
        log.info(f"  Generalization ratio: {gen_ratio:.3f}")

        if delta_test > 0:
            log.info(f"  GENERALIZATION CONFIRMED (ratio={gen_ratio:.2f})")
        else:
            log.info(f"  NULL: No transfer to test split")

    # Save
    with open(out_dir / "E6_generalization.json", "w") as f:
        json.dump(e6_results, f, indent=2)
    log.info("\nSaved E6_generalization.json")

    # ══════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════
    log.info("Generating plots...")

    if not e6_results:
        log.info("No results to visualize.")
        return

    # ── 1. Loss Comparison: Calib vs Test (per EG) ───────────────────
    fig, axes = plt.subplots(1, len(e6_results), figsize=(7 * len(e6_results), 5))
    if len(e6_results) == 1:
        axes = [axes]

    for ax, (eg_key, data) in zip(axes, e6_results.items()):
        x = np.arange(2)
        w = 0.35
        calib_vals = [data["calib"]["loss_baseline"], data["calib"]["loss_optimal"]]
        test_vals  = [data["test"]["loss_baseline"],  data["test"]["loss_optimal"]]
        ax.bar(x - w / 2, calib_vals, w, label="Calib split", color=BASELINE_COLOR, alpha=0.8)
        ax.bar(x + w / 2, test_vals, w, label="Test split", color=OPTIMAL_COLOR, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline (alpha=0)", "Optimal alpha*"])
        ax.set_ylabel("Adjusted Loss")
        ax.set_title(f"{eg_key}: Calib vs Test")
        ax.legend()
    fig.suptitle("E6: Generalization — Calib vs Test Split", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E6_calib_vs_test_loss")

    # ── 2a. Accuracy Comparison ───────────────────────────────────────
    fig, axes = plt.subplots(1, len(e6_results), figsize=(7 * len(e6_results), 5))
    if len(e6_results) == 1:
        axes = [axes]
    for ax, (eg_key, data) in zip(axes, e6_results.items()):
        labels = ["Calib\nBaseline", "Calib\nOptimal", "Test\nBaseline", "Test\nOptimal"]
        vals = [
            data["calib"].get("acc_baseline", 0), data["calib"].get("acc_optimal", 0),
            data["test"].get("acc_baseline", 0), data["test"].get("acc_optimal", 0),
        ]
        colors_a = [BASELINE_COLOR, BASELINE_COLOR, OPTIMAL_COLOR, OPTIMAL_COLOR]
        alphas_a = [0.5, 1.0, 0.5, 1.0]
        bars = ax.bar(labels, vals, color=colors_a)
        for bar, a in zip(bars, alphas_a):
            bar.set_alpha(a)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{eg_key}: Accuracy")
    fig.suptitle("E6: Accuracy — Calib vs Test Splits", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E6_accuracy_comparison")

    # ── 2. ECE Comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(e6_results), figsize=(7 * len(e6_results), 5))
    if len(e6_results) == 1:
        axes = [axes]

    for ax, (eg_key, data) in zip(axes, e6_results.items()):
        labels = ["Calib\nBaseline", "Calib\nOptimal", "Test\nBaseline", "Test\nOptimal"]
        vals = [
            data["calib"]["ece_baseline"], data["calib"]["ece_optimal"],
            data["test"]["ece_baseline"], data["test"]["ece_optimal"],
        ]
        colors = [BASELINE_COLOR, BASELINE_COLOR, OPTIMAL_COLOR, OPTIMAL_COLOR]
        alphas = [0.5, 1.0, 0.5, 1.0]
        bars = ax.bar(labels, vals, color=colors)
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)
        ax.set_ylabel("ECE")
        ax.set_title(f"{eg_key}: ECE")
    fig.suptitle("E6: ECE — Calib vs Test Splits", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E6_ece_comparison")

    # ── 3. Generalization Ratio ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    eg_keys = list(e6_results.keys())
    ratios = [e6_results[k]["generalization_ratio"] for k in eg_keys]
    colors = ["green" if r > 0 else "red" for r in ratios]
    ax.barh(eg_keys, ratios, color=colors, height=0.5)
    ax.axvline(1.0, color="black", linestyle="--", label="Full transfer")
    ax.axvline(0.0, color="gray", linestyle="-", alpha=0.5, label="No transfer")
    ax.set_xlabel("ΔLOSS(test) / ΔLOSS(calib)")
    ax.set_title("E6: Generalization Ratio by Exposure Group")
    ax.legend()
    ax.set_xlim(-0.5, max(1.5, max(ratios) + 0.3))
    save_plot(fig, plot_dir, "E6_generalization_ratio")

    # ── 4. Delta Loss Comparison ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_data = []
    for eg_key, data in e6_results.items():
        bar_data.append({"EG": eg_key, "Split": "Calib", "ΔLOSS": data["calib"]["delta_loss"]})
        bar_data.append({"EG": eg_key, "Split": "Test",  "ΔLOSS": data["test"]["delta_loss"]})
    bdf = pd.DataFrame(bar_data)
    sns.barplot(data=bdf, x="EG", y="ΔLOSS", hue="Split",
                palette=[BASELINE_COLOR, OPTIMAL_COLOR], ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ΔLOSS adjusted (positive = steering helps)")
    ax.set_title("E6: Adjusted ΔLOSS by Split and Exposure Group")
    ax.legend(title="Split")
    save_plot(fig, plot_dir, "E6_delta_loss_comparison")

    # ── 5. Summary Heatmap ───────────────────────────────────────────
    heat_rows = []
    for eg_key, data in e6_results.items():
        heat_rows.append({
            "EG": eg_key,
            "Calib ΔLOSS": data["calib"]["delta_loss"],
            "Test ΔLOSS": data["test"]["delta_loss"],
            "Gen Ratio": data["generalization_ratio"],
            "Test ECE (base)": data["test"]["ece_baseline"],
            "Test ECE (opt)": data["test"]["ece_optimal"],
            "Acc (base)": data["test"].get("acc_baseline", 0),
            "Acc (opt)": data["test"].get("acc_optimal", 0),
        })
    heat_df = pd.DataFrame(heat_rows).set_index("EG")

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                linewidths=1, center=0)
    ax.set_title("E6: Generalization Summary")
    save_plot(fig, plot_dir, "E6_summary_heatmap")

    log.info(f"All plots saved to {plot_dir}")
    log.info("E6 complete.")


if __name__ == "__main__":
    main()
