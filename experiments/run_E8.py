#!/usr/bin/env python3
"""
E8 -- Steered Model Performance at alpha* across n-shot conditions.

For each (n_shot in N_SHOT_E8): run N_PERSONAS_FULL personas x
N_RESAMPLES_E8 x scored items under optimal steering alpha* (from E2).

Saves:
  results/E8/E8_steered_EG{eg}_shot{k}_rs{rs}.json
  results/E8/E8_summary.json
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import (
    RESULTS_DIR, HF_TOKEN, SAE_LAYERS, N_PERSONAS_FULL, PERSONA_SEED,
    N_RESAMPLES_E8, N_SHOT_E8, EXPOSURE_GROUPS_PRIMARY,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    load_model, load_saes, get_client, run_condition,
    make_target_interventions, log_eta, count_scored_items,
)
from persona import load_personas

from tqdm import tqdm


def main():
    log = setup_logging("E8")
    setup_viz()
    out_dir = RESULTS_DIR / "E8"
    out_dir.mkdir(parents=True, exist_ok=True)

    opt_path = RESULTS_DIR / "E2" / "E2_optimal_alpha.json"
    with open(opt_path) as f:
        opt = json.load(f)
    u, r = opt["unsure_coeff"], opt["refusal_coeff"]
    log.info(f"alpha* = ({u}, {r})")

    log.info("Loading model + SAEs...")
    model = load_model(hf_token=HF_TOKEN)
    saes  = load_saes(layers=SAE_LAYERS)
    client = get_client()

    ivs = make_target_interventions(saes, u, r)

    human_df = load_human_data()
    personas = load_personas(human_df, n=N_PERSONAS_FULL, seed=PERSONA_SEED)
    log.info(f"Personas: {len(personas)}")

    summary = {"optimal_alpha": opt, "conditions": []}

    conditions = [
        (eg, k, rs)
        for eg in EXPOSURE_GROUPS_PRIMARY
        for k in N_SHOT_E8
        for rs in range(N_RESAMPLES_E8)
    ]
    items_by_eg = {eg: load_items(eg) for eg in EXPOSURE_GROUPS_PRIMARY}
    total_completions = sum(
        count_scored_items(items_by_eg[eg]) * len(personas)
        for (eg, _, _) in conditions
    )
    log_eta(log, total_completions, label="E8 total")

    for (eg, k, rs) in tqdm(conditions, desc="E8 conditions"):
        items = items_by_eg[eg]
        cond = f"EG{eg}_shot{k}_rs{rs}"
        ckpt = out_dir / f"E8_steered_{cond}.json"
        existing = {}
        if ckpt.exists():
            with open(ckpt) as f:
                existing = json.load(f)

        results = run_condition(
            model=model, items=items, personas=personas,
            interventions=ivs, client=client,
            n_shot=k, n_resamples=1, logger=log,
            checkpoint_path=ckpt, existing_results=existing,
        )
        with open(ckpt, "w") as f:
            json.dump(results, f, indent=2)
        summary["conditions"].append({
            "exposure_group": eg, "n_shot": k, "resample": rs,
            "n_personas": len(results),
            "file": str(ckpt.name),
        })

    with open(out_dir / "E8_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
