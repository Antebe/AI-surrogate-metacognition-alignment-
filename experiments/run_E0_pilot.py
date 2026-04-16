#!/usr/bin/env python3
"""
E0 -- Pilot: measure seconds per completion on this box.

Runs 1 persona x 3 items x 1 resample, unsteered, 0-shot.
Writes results/E0/pilot_timing.json with observed_s_per_completion
so downstream cost estimates use ground truth.
"""
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import (
    RESULTS_DIR, HF_TOKEN, PERSONA_SEED,
)
from shared import (
    setup_logging, load_items, load_human_data,
    load_model, get_client, run_condition,
)
from persona import load_personas


def main():
    log = setup_logging("E0")
    out_dir = RESULTS_DIR / "E0"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model...")
    model = load_model(hf_token=HF_TOKEN)
    client = get_client()

    human_df = load_human_data()
    personas = load_personas(human_df, n=1, seed=PERSONA_SEED)

    items = load_items(0).head(3)

    t0 = time.perf_counter()
    results = run_condition(
        model=model, items=items, personas=personas,
        interventions=None, client=client,
        n_shot=0, n_resamples=1, logger=log,
    )
    dt = time.perf_counter() - t0

    n_completions = sum(
        len(v["completions"]) for p in results.values() for v in p.values()
    )
    rate = dt / max(n_completions, 1)

    for p in results.values():
        for v in p.values():
            for probs in v["confidence_probs"]:
                s = sum(probs.values())
                assert abs(s - 1.0) < 1e-4, f"probs don't sum to 1: {probs}"

    summary = {
        "n_completions": n_completions,
        "elapsed_s": dt,
        "observed_s_per_completion": rate,
        "persona_id": str(personas.iloc[0]["ID_1"]),
    }
    with open(out_dir / "pilot_timing.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Pilot: {n_completions} completions in {dt:.1f}s = {rate:.2f}s/completion")


if __name__ == "__main__":
    main()
