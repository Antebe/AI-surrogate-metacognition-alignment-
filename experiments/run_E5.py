#!/usr/bin/env python3
"""
E5 -- Item-Type Asymmetry: does steering improve calibration more
on lure-exposure items (the misinformation effect)?

Runs BOTH EG0 and EG1 at baseline and alpha* (N_PERSONAS_FULL personas,
1 resample), 0-shot. Downstream analysis stratifies by Validity.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import (
    RESULTS_DIR, HF_TOKEN, SAE_LAYERS, N_PERSONAS_FULL, PERSONA_SEED,
    EXPOSURE_GROUPS_E5,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    load_model, load_saes, get_client, run_condition,
    make_target_interventions,
)
from persona import load_personas


def main():
    log = setup_logging("E5")
    setup_viz()
    out_dir = RESULTS_DIR / "E5"
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = json.loads((RESULTS_DIR / "E2" / "E2_optimal_alpha.json").read_text())
    u, r = opt["unsure_coeff"], opt["refusal_coeff"]

    model = load_model(hf_token=HF_TOKEN)
    saes = load_saes(layers=SAE_LAYERS)
    client = get_client()

    human_df = load_human_data()
    personas = load_personas(human_df, n=N_PERSONAS_FULL, seed=PERSONA_SEED)

    ivs_base = None
    ivs_opt = make_target_interventions(saes, u, r)

    for eg in EXPOSURE_GROUPS_E5:
        items = load_items(eg)
        for cond_name, ivs in [("baseline", ivs_base), ("steered", ivs_opt)]:
            ckpt = out_dir / f"E5_{cond_name}_EG{eg}.json"
            existing = json.loads(ckpt.read_text()) if ckpt.exists() else {}
            results = run_condition(
                model=model, items=items, personas=personas,
                interventions=ivs, client=client,
                n_shot=0, n_resamples=1, logger=log,
                checkpoint_path=ckpt, existing_results=existing,
            )
            with open(ckpt, "w") as f:
                json.dump(results, f, indent=2)
            log.info(f"  E5 {cond_name} EG{eg}: {len(results)} personas")


if __name__ == "__main__":
    main()
