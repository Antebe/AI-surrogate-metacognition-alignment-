#!/usr/bin/env python3
"""
E2 -- Core Steering: Optimal Coefficient Search
=================================================
Grid-searches over alpha = (unsure_coeff, refusal_coeff) for both EG0 and EG1.
Finds alpha* that minimizes joint calibration loss across both exposure groups.

Loss: L(alpha) = lambda1 * ECE(model, humans) + lambda2 * (1 - corr(conf, acc))

Saves:
  results/E2/E2_sweep_EG{0,1}.json
  results/E2/E2_optimal_alpha.json
  results/E2/E2_summary.json
  results/E2/plots/*.png
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
    compute_human_norms, compute_loss, get_client,
    build_prompt, get_confidence, code_response,
    make_target_interventions,
    load_model, load_saes, generate,
    EG_COLORS, BASELINE_COLOR, OPTIMAL_COLOR,
)


def main():
    log = setup_logging("E2")
    setup_viz()

    out_dir  = RESULTS_DIR / "E2"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + SAEs ────────────────────────────────────────────────
    log.info("Loading model + SAEs...")
    model  = load_model(hf_token=HF_TOKEN)
    saes   = load_saes(layers=SAE_LAYERS)
    client = get_client()
    log.info("Model loaded.")

    # ── Load data ────────────────────────────────────────────────────────
    human_df = load_human_data()
    calib_ids, _ = get_participant_split(human_df)

    items_by_eg = {eg: load_items(eg) for eg in EXPOSURE_GROUPS}

    human_norms_by_eg = {}
    for eg in EXPOSURE_GROUPS:
        norms = compute_human_norms(human_df, calib_ids, exposure_group=eg)
        if len(norms) == 0:
            norms = compute_human_norms(human_df, calib_ids)
        human_norms_by_eg[eg] = dict(zip(norms["item_id"], norms["human_conf_norm"]))
    log.info(f"Human norms loaded: {[len(v) for v in human_norms_by_eg.values()]} items per EG")

    # ── Grid sweep ───────────────────────────────────────────────────────
    grid_size = len(SWEEP_UNSURE_VALS) * len(SWEEP_REFUSAL_VALS)
    log.info(f"Grid: {len(SWEEP_UNSURE_VALS)} x {len(SWEEP_REFUSAL_VALS)} = {grid_size} configs")

    for eg in EXPOSURE_GROUPS:
        sweep_path = out_dir / f"E2_sweep_EG{eg}.json"

        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep_results = json.load(f)
            log.info(f"EG{eg}: Loaded {len(sweep_results)}/{grid_size} cached configs")
        else:
            sweep_results = {}

        items = items_by_eg[eg]
        h_map = human_norms_by_eg[eg]

        for u in tqdm(SWEEP_UNSURE_VALS, desc=f"EG{eg} unsure"):
            for r in tqdm(SWEEP_REFUSAL_VALS, desc="refusal", leave=False):
                key = f"{u},{r}"
                if key in sweep_results:
                    continue

                ivs = make_target_interventions(saes, u, r)
                model_confs, model_accs, item_ids = [], [], []

                for _, row in items.iterrows():
                    prompt = build_prompt(row["exposure_statement"], row["question"])
                    raw_comps = generate(
                        model, prompt, interventions=ivs,
                        max_new_tokens=MAX_NEW_TOKENS, n=E2_N_SAMPLES, temperature=TEMPERATURE,
                    )
                    comps = [
                        o[len(prompt):].strip() if o.startswith(prompt) else o.strip()
                        for o in raw_comps
                    ]
                    confs = [get_confidence(model, prompt, c, client)[0] for c in comps]
                    coded = [
                        code_response(
                            row["exposure_statement"], row["question"],
                            row["correct_answer"], c, client,
                        )
                        for c in comps
                    ]
                    accs = [acc for _, acc in coded]
                    model_confs.append(float(np.mean(confs)))
                    model_accs.append(float(np.mean(accs)))
                    item_ids.append(row["item_id"])

                # Align with human norms
                hconfs = [h_map.get(i, np.nan) for i in item_ids]
                mask = [not np.isnan(h) for h in hconfs]
                mc_f = [model_confs[i] for i in range(len(mask)) if mask[i]]
                hc_f = [hconfs[i]      for i in range(len(mask)) if mask[i]]
                ma_f = [model_accs[i]  for i in range(len(mask)) if mask[i]]

                loss, ece, corr = compute_loss(mc_f, hc_f, ma_f)
                sweep_results[key] = {
                    "unsure_coeff": u, "refusal_coeff": r,
                    "loss": loss, "ece": ece, "corr": corr,
                    "mean_accuracy": float(np.mean(model_accs)),
                    "model_confs": model_confs, "model_accs": model_accs,
                    "item_ids": item_ids,
                }

                with open(sweep_path, "w") as f:
                    json.dump(sweep_results, f, indent=2)

                log.info(f"  EG{eg} alpha=({u},{r}): L={loss:.4f} ECE={ece:.4f} r={corr:.4f}")

        log.info(f"EG{eg} sweep complete: {len(sweep_results)} configs")

    # ── Find optimal alpha (combined across available EGs) ───────────────
    log.info("Finding optimal alpha...")

    sweep_by_eg: dict[int, dict] = {}
    for eg in [0, 1]:
        path = out_dir / f"E2_sweep_EG{eg}.json"
        if path.exists():
            with open(path) as f:
                sweep_by_eg[eg] = json.load(f)
            log.info(f"Loaded sweep for EG{eg}: {len(sweep_by_eg[eg])} configs")
        else:
            log.info(f"EG{eg} sweep file not found — skipping")

    if not sweep_by_eg:
        raise RuntimeError("No sweep results found for any EG. Run the sweep first.")

    # Compute per-config accuracy and acc_ratio relative to baseline (0,0)
    baseline_key = "0,0"
    for eg, data in sweep_by_eg.items():
        base_acc = float(np.mean(data[baseline_key]["model_accs"])) if baseline_key in data else 1.0
        for key, vals in data.items():
            acc = float(np.mean(vals["model_accs"]))
            vals["mean_accuracy"] = acc
            vals["acc_ratio"] = base_acc / acc if acc > 0 else 1.0
            vals["adjusted_loss"] = vals["loss"] * vals["acc_ratio"]
        log.info(f"EG{eg}: baseline acc={base_acc:.4f}")

    # Combine adjusted losses across whichever EGs are available
    all_keys = set()
    for data in sweep_by_eg.values():
        all_keys.update(data.keys())

    combined_losses = {}
    for key in all_keys:
        losses = [sweep_by_eg[eg][key]["adjusted_loss"] for eg in sweep_by_eg if key in sweep_by_eg[eg]]
        combined_losses[key] = float(np.mean(losses))

    best_key = min(combined_losses, key=combined_losses.get)
    best_loss = combined_losses[best_key]
    best_u, best_r = map(int, best_key.split(","))

    optimal = {
        "optimal_alpha": best_key,
        "unsure_coeff": best_u,
        "refusal_coeff": best_r,
        "combined_loss": best_loss,
    }
    for eg in [0, 1]:
        eg_key = f"EG{eg}"
        if eg in sweep_by_eg and best_key in sweep_by_eg[eg]:
            d = sweep_by_eg[eg][best_key]
            optimal[eg_key] = {
                "loss": d["loss"], "ece": d["ece"], "corr": d["corr"],
                "adjusted_loss": d["adjusted_loss"],
                "mean_accuracy": d["mean_accuracy"], "acc_ratio": d["acc_ratio"],
            }
        else:
            optimal[eg_key] = {
                "loss": None, "ece": None, "corr": None,
                "adjusted_loss": None, "mean_accuracy": None, "acc_ratio": None,
            }

    with open(out_dir / "E2_optimal_alpha.json", "w") as f:
        json.dump(optimal, f, indent=2)

    log.info(f"Optimal alpha*: unsure={optimal['unsure_coeff']}, refusal={optimal['refusal_coeff']}")
    log.info(f"  Combined adjusted loss: {best_loss:.4f}")
    for eg in [0, 1]:
        eg_key = f"EG{eg}"
        d = optimal[eg_key]
        if d["loss"] is not None:
            log.info(
                f"  {eg_key}: L_raw={d['loss']:.4f} L_adj={d['adjusted_loss']:.4f} "
                f"ECE={d['ece']:.4f} acc={d['mean_accuracy']:.4f} acc_ratio={d['acc_ratio']:.4f}"
            )

    # ── Load E1 baseline for comparison ──────────────────────────────────
    e1_summary_path = RESULTS_DIR / "E1" / "E1_summary.json"
    if e1_summary_path.exists():
        with open(e1_summary_path) as f:
            e1_sum = json.load(f)
    else:
        e1_sum = None
        log.info("Warning: E1 summary not found; skipping baseline comparison in some plots")

    # ══════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════
    log.info("Generating plots...")

    for eg, sweep_data in sweep_by_eg.items():
        rows = pd.DataFrame([
            {"u": v["unsure_coeff"], "r": v["refusal_coeff"],
             "loss": v["adjusted_loss"], "raw_loss": v["loss"],
             "ece": v["ece"], "corr": v["corr"],
             "acc": v["mean_accuracy"], "acc_ratio": v["acc_ratio"]}
            for v in sweep_data.values()
        ])

        # ── 1. Loss Landscape Heatmaps (per EG) ─────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(26, 5))
        for ax, col, title, cmap in zip(
            axes,
            ["loss", "ece", "corr", "acc"],
            ["Adjusted Loss", "ECE (lower=better)", "Conf-Acc Corr (higher=better)", "Accuracy"],
            ["viridis_r", "viridis_r", "viridis", "viridis"],
        ):
            pivot = rows.pivot(index="u", columns="r", values=col)
            im = ax.imshow(
                pivot.values, cmap=cmap, aspect="auto",
                extent=[
                    min(SWEEP_REFUSAL_VALS) - 10, max(SWEEP_REFUSAL_VALS) + 10,
                    max(SWEEP_UNSURE_VALS) + 10, min(SWEEP_UNSURE_VALS) - 10,
                ],
            )
            plt.colorbar(im, ax=ax)
            ax.scatter(
                [optimal["refusal_coeff"]], [optimal["unsure_coeff"]],
                marker="*", s=250, c="red", zorder=5, label="alpha*",
            )
            ax.set_xlabel("refusal_coeff")
            ax.set_ylabel("unsure_coeff")
            ax.set_title(title)
            ax.legend()
        fig.suptitle(f"E2: Loss Landscape - {EG_LABELS[eg]}", fontsize=14, y=1.02)
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E2_loss_landscape_EG{eg}")

        # ── 2. Seaborn heatmap (cleaner) ─────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(26, 5))
        for ax, col, title, cmap in zip(
            axes,
            ["loss", "ece", "corr", "acc"],
            ["Adjusted Loss", "ECE", "Conf-Acc r", "Accuracy"],
            ["YlOrRd", "YlOrRd", "YlGn", "YlGn"],
        ):
            pivot = rows.pivot(index="u", columns="r", values=col).sort_index(ascending=False)
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax,
                        linewidths=0.5, cbar_kws={"label": col})
            ax.set_xlabel("refusal_coeff")
            ax.set_ylabel("unsure_coeff")
            ax.set_title(title)
        fig.suptitle(f"E2: Coefficient Grid - {EG_LABELS[eg]}", fontsize=14, y=1.02)
        plt.tight_layout()
        save_plot(fig, plot_dir, f"E2_heatmap_EG{eg}")

    # ── 3. Combined Loss Landscape ───────────────────────────────────
    combined_rows = []
    for key, comb_loss in combined_losses.items():
        u, r = map(int, key.split(","))
        row = {"u": u, "r": r, "loss": comb_loss}
        for eg in sweep_by_eg:
            row[f"ece_eg{eg}"] = sweep_by_eg[eg].get(key, {}).get("ece", np.nan)
        combined_rows.append(row)
    cdf = pd.DataFrame(combined_rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = cdf.pivot(index="u", columns="r", values="loss").sort_index(ascending=False)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Combined Adjusted Loss"})
    ax.set_xlabel("refusal_coeff")
    ax.set_ylabel("unsure_coeff")
    ax.set_title(f"E2: Combined Adjusted Loss\nalpha*=({optimal['unsure_coeff']},{optimal['refusal_coeff']})")
    save_plot(fig, plot_dir, "E2_combined_loss_heatmap")

    # ── 4. Baseline vs Optimal Comparison ────────────────────────────
    if e1_sum:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        w = 0.35

        # ECE, Adjusted Loss (lower = better)
        for ax, metric, label in zip(
            axes[:2],
            ["ece", "adjusted_loss"],
            ["ECE", "Adjusted Loss"],
        ):
            baseline_vals = []
            optimal_vals = []
            eg_names = []
            for eg_key in ["EG0", "EG1"]:
                if eg_key in e1_sum and eg_key in optimal and optimal[eg_key].get(metric) is not None:
                    # Baseline adjusted_loss == raw loss (acc_ratio=1 at 0,0)
                    base_metric = metric if metric != "adjusted_loss" else "loss"
                    baseline_vals.append(e1_sum[eg_key].get(base_metric, e1_sum[eg_key].get("loss", 0)))
                    optimal_vals.append(optimal[eg_key][metric])
                    eg_names.append(eg_key)

            if eg_names:
                x = np.arange(len(eg_names))
                ax.bar(x - w / 2, baseline_vals, w, label="Baseline (alpha=0)",
                       color=BASELINE_COLOR, alpha=0.8)
                ax.bar(x + w / 2, optimal_vals, w, label="Optimal alpha*",
                       color=OPTIMAL_COLOR, alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(eg_names)
                ax.set_title(f"{label} (lower = better)")
                ax.set_ylabel(label)
                ax.legend()

        # Conf-Acc correlation (higher = better)
        ax = axes[2]
        corr_base, corr_opt, eg_names_c = [], [], []
        for eg_key in ["EG0", "EG1"]:
            if eg_key in e1_sum and eg_key in optimal and optimal[eg_key].get("corr") is not None:
                corr_base.append(e1_sum[eg_key].get("conf_acc_corr", 0))
                corr_opt.append(optimal[eg_key]["corr"])
                eg_names_c.append(eg_key)
        if eg_names_c:
            x = np.arange(len(eg_names_c))
            ax.bar(x - w / 2, corr_base, w, label="Baseline", color=BASELINE_COLOR, alpha=0.8)
            ax.bar(x + w / 2, corr_opt, w, label="Optimal", color=OPTIMAL_COLOR, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(eg_names_c)
            ax.set_title("Conf-Acc Corr (higher = better)")
            ax.legend()

        # Accuracy (higher = better)
        ax = axes[3]
        acc_base, acc_opt, eg_names_a = [], [], []
        for eg_key in ["EG0", "EG1"]:
            if eg_key in e1_sum and eg_key in optimal and optimal[eg_key].get("mean_accuracy") is not None:
                acc_base.append(e1_sum[eg_key].get("mean_accuracy", e1_sum[eg_key].get("acc", 0)))
                acc_opt.append(optimal[eg_key]["mean_accuracy"])
                eg_names_a.append(eg_key)
        if eg_names_a:
            x = np.arange(len(eg_names_a))
            ax.bar(x - w / 2, acc_base, w, label="Baseline", color=BASELINE_COLOR, alpha=0.8)
            ax.bar(x + w / 2, acc_opt, w, label="Optimal", color=OPTIMAL_COLOR, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(eg_names_a)
            ax.set_title("Accuracy (higher = better)")
            ax.legend()

        fig.suptitle("E2: Baseline vs Optimal Steering", fontsize=14)
        plt.tight_layout()
        save_plot(fig, plot_dir, "E2_baseline_vs_optimal")

    # ── 5. Loss trajectory across unsure_coeff (marginalizing refusal) ─
    n_egs = len(sweep_by_eg)
    fig, axes = plt.subplots(1, max(n_egs, 2), figsize=(7 * max(n_egs, 2), 5))
    if n_egs == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    for ax, (eg, sweep_data) in zip(axes, sweep_by_eg.items()):
        sdf = pd.DataFrame([
            {"u": v["unsure_coeff"], "r": v["refusal_coeff"], "loss": v["adjusted_loss"]}
            for v in sweep_data.values()
        ])
        # Marginal over refusal
        marginal_u = sdf.groupby("u")["loss"].agg(["mean", "std"]).reset_index()
        ax.errorbar(marginal_u["u"], marginal_u["mean"], yerr=marginal_u["std"],
                    marker="o", capsize=4, color=EG_COLORS[eg])
        ax.set_xlabel("unsure_coeff")
        ax.set_ylabel("Adj. Loss (mean +/- std over refusal_coeff)")
        ax.set_title(f"{EG_LABELS[eg]}")
    fig.suptitle("E2: Loss vs unsure_coeff (marginalized over refusal)", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E2_loss_vs_unsure_marginal")

    # ── 6. Loss trajectory across refusal_coeff ──────────────────────
    fig, axes = plt.subplots(1, max(n_egs, 2), figsize=(7 * max(n_egs, 2), 5))
    if n_egs == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    for ax, (eg, sweep_data) in zip(axes, sweep_by_eg.items()):
        sdf = pd.DataFrame([
            {"u": v["unsure_coeff"], "r": v["refusal_coeff"], "loss": v["adjusted_loss"]}
            for v in sweep_data.values()
        ])
        marginal_r = sdf.groupby("r")["loss"].agg(["mean", "std"]).reset_index()
        ax.errorbar(marginal_r["r"], marginal_r["mean"], yerr=marginal_r["std"],
                    marker="s", capsize=4, color=EG_COLORS[eg])
        ax.set_xlabel("refusal_coeff")
        ax.set_ylabel("Adj. Loss (mean +/- std over unsure_coeff)")
        ax.set_title(f"{EG_LABELS[eg]}")
    fig.suptitle("E2: Loss vs refusal_coeff (marginalized over unsure)", fontsize=14)
    plt.tight_layout()
    save_plot(fig, plot_dir, "E2_loss_vs_refusal_marginal")

    # ── Save summary ─────────────────────────────────────────────────────
    summary = {
        "optimal": optimal,
        "grid_size": grid_size,
        "n_configs_by_eg": {f"EG{eg}": len(data) for eg, data in sweep_by_eg.items()},
    }
    with open(out_dir / "E2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"All plots saved to {plot_dir}")
    log.info("E2 complete.")


if __name__ == "__main__":
    main()
