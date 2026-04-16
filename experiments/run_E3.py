#!/usr/bin/env python3
"""
E3 -- Specificity ablation: do control latents produce the same gain?

Runs the N_PERSONAS_SWEEP subset x N_RESAMPLES_SWEEP x scored items at
the optimal coefficients but with CONTROL_LATENTS instead of the target set.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import numpy as np

from config import (
    RESULTS_DIR, HF_TOKEN, SAE_LAYERS, CONTROL_LATENTS,
    N_PERSONAS_SWEEP, PERSONA_SEED, N_RESAMPLES_SWEEP,
    EXPOSURE_GROUPS_PRIMARY, HUMAN_CONF_MAP, LOSS_ALPHA, LOSS_BETA,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    get_participant_split, load_model, load_saes, get_client,
    run_condition, make_control_interventions,
)
from persona import load_personas
from brier import build_human_prob_map, composite_loss


def _flatten(results):
    confs, accs = [], []
    for p in results.values():
        for v in p.values():
            confs.append(v["mean_confidence"])
            accs.append(v["mean_accuracy"])
    return np.array(confs), np.array(accs)


def main():
    log = setup_logging("E3")
    setup_viz()
    out_dir = RESULTS_DIR / "E3"
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = json.loads((RESULTS_DIR / "E2" / "E2_optimal_alpha.json").read_text())
    u_opt, r_opt = opt["unsure_coeff"], opt["refusal_coeff"]

    model = load_model(hf_token=HF_TOKEN)
    saes = load_saes(layers=SAE_LAYERS)
    client = get_client()

    human_df = load_human_data()
    calib_ids, _ = get_participant_split(human_df)
    calib_df = human_df[human_df["ID_1"].isin(calib_ids)]
    human_prob_map = build_human_prob_map(calib_df, mode=HUMAN_CONF_MAP)
    personas = load_personas(human_df, n=N_PERSONAS_SWEEP, seed=PERSONA_SEED)

    findings = {}
    for ctrl_name in CONTROL_LATENTS:
        for eg in EXPOSURE_GROUPS_PRIMARY:
            items = load_items(eg)
            ivs = make_control_interventions(saes, ctrl_name, max(u_opt, r_opt))
            ckpt = out_dir / f"E3_{ctrl_name}_EG{eg}.json"
            existing = json.loads(ckpt.read_text()) if ckpt.exists() else {}
            results = run_condition(
                model=model, items=items, personas=personas,
                interventions=ivs, client=client,
                n_shot=0, n_resamples=N_RESAMPLES_SWEEP, logger=log,
                checkpoint_path=ckpt, existing_results=existing,
            )
            with open(ckpt, "w") as f:
                json.dump(results, f, indent=2)

            mc, ma = _flatten(results)
            sub = calib_df[calib_df["Exposure"] == eg]
            L, parts = composite_loss(
                model_conf=mc, model_acc=ma,
                human_resp=sub["Confidence"].to_numpy(),
                human_correct=sub["Correct"].to_numpy(),
                human_prob_map=human_prob_map,
                alpha=LOSS_ALPHA, beta=LOSS_BETA,
            )
            findings[f"{ctrl_name}_EG{eg}"] = {"loss": L, **parts}
            log.info(f"  {ctrl_name} EG{eg}: L={L:.4f}")

    with open(out_dir / "E3_summary.json", "w") as f:
        json.dump(findings, f, indent=2)


if __name__ == "__main__":
    main()
