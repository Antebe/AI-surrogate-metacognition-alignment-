#!/usr/bin/env python3
"""
E1 -- Baselines: unsteered Gemma with persona prompting, across n-shot.

For each (n_shot in N_SHOT_E1) x EXPOSURE_GROUPS_PRIMARY:
  run N_PERSONAS_FULL personas x N_RESAMPLES_E1 resamples x scored items,
  unsteered.

Saves:
  results/E1/E1_baseline_EG{eg}_shot{k}_rs{rs}.json
  results/E1/E1_summary.json
  results/E1/brier_human_prob_map.json
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import (
    RESULTS_DIR, HF_TOKEN, N_PERSONAS_FULL, PERSONA_SEED,
    N_RESAMPLES_E1, N_SHOT_E1, EXPOSURE_GROUPS_PRIMARY,
    HUMAN_CONF_MAP,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    get_participant_split, load_model, get_client, run_condition,
    log_eta, count_scored_items,
)
from persona import load_personas
from brier import build_human_prob_map

from tqdm import tqdm


def main():
    log = setup_logging("E1")
    setup_viz()
    out_dir = RESULTS_DIR / "E1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    log.info("Loading model...")
    model = load_model(hf_token=HF_TOKEN)
    client = get_client()

    human_df = load_human_data()
    calib_ids, _ = get_participant_split(human_df)
    calib_df = human_df[human_df["ID_1"].isin(calib_ids)]

    human_prob_map = build_human_prob_map(calib_df, mode=HUMAN_CONF_MAP)
    with open(out_dir / "brier_human_prob_map.json", "w") as f:
        json.dump({str(k): v for k, v in human_prob_map.items()}, f, indent=2)
    log.info(f"Human prob map: {human_prob_map}")

    personas = load_personas(human_df, n=N_PERSONAS_FULL, seed=PERSONA_SEED)
    log.info(f"Loaded {len(personas)} personas")

    summary = {"conditions": []}

    # Build the full condition list up-front for a single outer tqdm bar.
    conditions = [
        (eg, k, rs)
        for eg in EXPOSURE_GROUPS_PRIMARY
        for k in N_SHOT_E1
        for rs in range(N_RESAMPLES_E1)
    ]
    items_by_eg = {eg: load_items(eg) for eg in EXPOSURE_GROUPS_PRIMARY}
    total_completions = sum(
        count_scored_items(items_by_eg[eg]) * len(personas)
        for (eg, _, _) in conditions
    )
    log_eta(log, total_completions, label="E1 total")

    for (eg, k, rs) in tqdm(conditions, desc="E1 conditions"):
        items = items_by_eg[eg]
        cond_tag = f"EG{eg}_shot{k}_rs{rs}"
        ckpt = out_dir / f"E1_baseline_{cond_tag}.json"
        existing = {}
        if ckpt.exists():
            with open(ckpt) as f:
                existing = json.load(f)
        log.info(f"Condition {cond_tag}: cached personas={len(existing)}")

        results = run_condition(
            model=model, items=items, personas=personas,
            interventions=None, client=client,
            n_shot=k, n_resamples=1, logger=log,
            checkpoint_path=ckpt, existing_results=existing,
        )
        with open(ckpt, "w") as f:
            json.dump(results, f, indent=2)

        summary["conditions"].append({
            "exposure_group": eg,
            "n_shot": k,
            "resample": rs,
            "n_personas": len(results),
            "file": str(ckpt.name),
        })

    with open(out_dir / "E1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("E1 complete.")


if __name__ == "__main__":
    main()
