#!/usr/bin/env python3
"""
E2 -- Core Steering: optimal (unsure, refusal) coefficient search.

Uses the Brier-based composite loss. Sweep is persona-averaged over a
rotating subset of N_PERSONAS_SWEEP; 0-shot only.

Saves:
  results/E2/E2_sweep_EG{eg}.json
  results/E2/E2_optimal_alpha.json
  results/E2/E2_summary.json
  results/E2/limiting_behavior.md
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import numpy as np
from tqdm import tqdm

from config import (
    RESULTS_DIR, HF_TOKEN, SAE_LAYERS, SWEEP_UNSURE_VALS, SWEEP_REFUSAL_VALS,
    N_PERSONAS_SWEEP, PERSONA_SEED, N_RESAMPLES_SWEEP, EXPOSURE_GROUPS_PRIMARY,
    LOSS_ALPHA, LOSS_BETA, HUMAN_CONF_MAP,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    get_participant_split, load_model, load_saes, get_client, run_condition,
    make_target_interventions, log_eta, count_scored_items,
)
from persona import load_personas
from brier import build_human_prob_map, composite_loss, limiting_behavior_report


def _flatten(results: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flatten persona-keyed results to (conf[], acc[]) arrays."""
    confs, accs = [], []
    for persona_results in results.values():
        for v in persona_results.values():
            confs.append(v["mean_confidence"])
            accs.append(v["mean_accuracy"])
    return np.array(confs), np.array(accs)


def main():
    log = setup_logging("E2")
    setup_viz()
    out_dir = RESULTS_DIR / "E2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    log.info("Loading model + SAEs...")
    model = load_model(hf_token=HF_TOKEN)
    saes  = load_saes(layers=SAE_LAYERS)
    client = get_client()

    human_df = load_human_data()
    calib_ids, _ = get_participant_split(human_df)
    calib_df = human_df[human_df["ID_1"].isin(calib_ids)]
    human_prob_map = build_human_prob_map(calib_df, mode=HUMAN_CONF_MAP)
    log.info(f"Human prob map: {human_prob_map}")

    personas = load_personas(human_df, n=N_PERSONAS_SWEEP, seed=PERSONA_SEED)

    grid_size = len(SWEEP_UNSURE_VALS) * len(SWEEP_REFUSAL_VALS)
    n_scored = count_scored_items(load_items(EXPOSURE_GROUPS_PRIMARY[0]))
    total_completions = (
        grid_size * len(EXPOSURE_GROUPS_PRIMARY) * len(personas)
        * n_scored * N_RESAMPLES_SWEEP
    )
    log_eta(log, total_completions, label="E2 total")

    limits = limiting_behavior_report(LOSS_ALPHA, LOSS_BETA)
    with open(out_dir / "limiting_behavior.md", "w") as f:
        f.write("# Loss term limits\n\n")
        for k, v in limits.items():
            f.write(f"- **{k}**: best={v['best']:.3f}, worst={v['worst']:.3f}\n")

    best = {"loss": float("inf"), "u": None, "r": None}
    all_summaries: dict[str, dict] = {}

    for eg in EXPOSURE_GROUPS_PRIMARY:
        items = load_items(eg)
        sub = calib_df[calib_df["Exposure"] == eg]
        h_resp = sub["Confidence"].to_numpy()
        h_corr = sub["Correct"].to_numpy()

        sweep_path = out_dir / f"E2_sweep_EG{eg}.json"
        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep = json.load(f)
        else:
            sweep = {}

        grid = [(u, r) for u in SWEEP_UNSURE_VALS for r in SWEEP_REFUSAL_VALS]

        for u, r in tqdm(grid, desc=f"EG{eg} grid"):
            key = f"{u},{r}"
            if key in sweep:
                if sweep[key]["loss"] < best["loss"]:
                    best = {"loss": sweep[key]["loss"], "u": u, "r": r}
                continue

            ivs = make_target_interventions(saes, u, r)
            results = run_condition(
                model=model, items=items, personas=personas,
                interventions=ivs, client=client,
                n_shot=0, n_resamples=N_RESAMPLES_SWEEP, logger=log,
            )
            mc, ma = _flatten(results)
            L, parts = composite_loss(
                model_conf=mc, model_acc=ma,
                human_resp=h_resp, human_correct=h_corr,
                human_prob_map=human_prob_map,
                alpha=LOSS_ALPHA, beta=LOSS_BETA,
            )
            sweep[key] = {
                "unsure_coeff": u, "refusal_coeff": r,
                "loss": L, **parts,
                "n_completions": int(len(mc)),
            }
            with open(sweep_path, "w") as f:
                json.dump(sweep, f, indent=2)

            log.info(
                f"  alpha=({u},{r}) L={L:.4f} brier_diff={parts['brier_diff']:.3f} "
                f"rho_diff={parts['rho_diff']:.3f} acc_pen={parts['acc_penalty']:.3f}"
            )

            if L < best["loss"]:
                best = {"loss": L, "u": u, "r": r}

        all_summaries[f"EG{eg}"] = sweep

    with open(out_dir / "E2_optimal_alpha.json", "w") as f:
        json.dump({
            "optimal_alpha": f"{best['u']},{best['r']}",
            "unsure_coeff": best["u"], "refusal_coeff": best["r"],
            "combined_loss": best["loss"],
        }, f, indent=2)
    with open(out_dir / "E2_summary.json", "w") as f:
        json.dump({"sweeps": all_summaries, "best": best}, f, indent=2)

    log.info(f"alpha* = ({best['u']}, {best['r']})  L* = {best['loss']:.4f}")


if __name__ == "__main__":
    main()
