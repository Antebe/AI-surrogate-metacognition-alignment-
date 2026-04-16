#!/usr/bin/env python3
"""
E6 -- Generalization: does alpha* transfer to held-out (test) participants?

Runs the 66 held-out personas x N_RESAMPLES_E1 resamples x scored items
at alpha*, 0-shot, EG0.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import (
    RESULTS_DIR, HF_TOKEN, SAE_LAYERS, PERSONA_SEED,
    N_RESAMPLES_E1, EXPOSURE_GROUPS_PRIMARY,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    get_participant_split, load_model, load_saes, get_client,
    run_condition, make_target_interventions,
)
from persona import load_personas


def main():
    log = setup_logging("E6")
    setup_viz()
    out_dir = RESULTS_DIR / "E6"
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = json.loads((RESULTS_DIR / "E2" / "E2_optimal_alpha.json").read_text())
    u, r = opt["unsure_coeff"], opt["refusal_coeff"]

    model = load_model(hf_token=HF_TOKEN)
    saes = load_saes(layers=SAE_LAYERS)
    client = get_client()

    human_df = load_human_data()
    _, test_ids = get_participant_split(human_df)
    test_df = human_df[human_df["ID_1"].isin(test_ids)]
    personas = load_personas(test_df, n=None, seed=PERSONA_SEED)
    log.info(f"Held-out personas: {len(personas)}")

    ivs = make_target_interventions(saes, u, r)

    for eg in EXPOSURE_GROUPS_PRIMARY:
        items = load_items(eg)
        for rs in range(N_RESAMPLES_E1):
            ckpt = out_dir / f"E6_heldout_EG{eg}_rs{rs}.json"
            existing = json.loads(ckpt.read_text()) if ckpt.exists() else {}
            results = run_condition(
                model=model, items=items, personas=personas,
                interventions=ivs, client=client,
                n_shot=0, n_resamples=1, logger=log,
                checkpoint_path=ckpt, existing_results=existing,
            )
            with open(ckpt, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
