#!/usr/bin/env python3
"""
E9 -- E1 (baseline) vs E8 (steered) comprehensive analysis.

Compares baseline to steered at matched shot settings:
  - Brier, correlation, accuracy, composite loss
  - 4-category distributional metrics (Wasserstein, JS)
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import numpy as np
import pandas as pd

from config import (
    RESULTS_DIR, N_SHOT_E8, EXPOSURE_GROUPS_PRIMARY,
    HUMAN_CONF_MAP, LOSS_ALPHA, LOSS_BETA,
)
from shared import setup_logging, setup_viz, load_human_data, get_participant_split
from brier import build_human_prob_map, composite_loss
from distribution import (
    category_distribution, wasserstein_4cat, js_divergence_4cat,
)


def _load_condition(prefix_dir: str, file_prefix: str, eg: int, k: int) -> dict:
    """Merge all resample files for a given file prefix/EG/shot into one persona dict."""
    merged: dict[str, dict] = defaultdict(dict)
    pat = f"{file_prefix}_EG{eg}_shot{k}_rs*.json"
    for p in sorted((RESULTS_DIR / prefix_dir).glob(pat)):
        data = json.loads(p.read_text())
        for pid, items in data.items():
            merged[pid].update(items)
    return dict(merged)


def _flatten(results: dict):
    confs, accs, cats = [], [], []
    for p in results.values():
        for v in p.values():
            confs.append(v["mean_confidence"])
            accs.append(v["mean_accuracy"])
            cats.extend(v["response_categories"])
    return np.array(confs), np.array(accs), cats


def main():
    log = setup_logging("E9")
    setup_viz()
    out_dir = RESULTS_DIR / "E9"
    out_dir.mkdir(parents=True, exist_ok=True)

    human_df = load_human_data()
    calib_ids, _ = get_participant_split(human_df)
    calib_df = human_df[human_df["ID_1"].isin(calib_ids)]
    human_prob_map = build_human_prob_map(calib_df, mode=HUMAN_CONF_MAP)

    report: dict[str, dict] = {}

    for eg in EXPOSURE_GROUPS_PRIMARY:
        sub = calib_df[calib_df["Exposure"] == eg]
        h_resp, h_corr = sub["Confidence"].to_numpy(), sub["Correct"].to_numpy()

        human_cats = []
        for _, r in sub.iterrows():
            if r["Correct"]:    human_cats.append("Correct")
            elif r["Error"]:    human_cats.append("Lure")
            elif r["Unsure"]:   human_cats.append("Unsure")
            else:               human_cats.append("Other")
        hd = category_distribution(human_cats)

        for k in N_SHOT_E8:
            key = f"EG{eg}_shot{k}"
            e1 = _load_condition("E1", "E1_baseline", eg, k)
            e8 = _load_condition("E8", "E8_steered", eg, k)

            def _panel(results):
                mc, ma, cats = _flatten(results)
                if len(mc) == 0:
                    return {"error": "no data"}
                L, parts = composite_loss(
                    model_conf=mc, model_acc=ma,
                    human_resp=h_resp, human_correct=h_corr,
                    human_prob_map=human_prob_map,
                    alpha=LOSS_ALPHA, beta=LOSS_BETA,
                )
                md = category_distribution(cats)
                return {
                    **parts, "loss": L,
                    "wasserstein_vs_human": wasserstein_4cat(md, hd),
                    "js_vs_human": js_divergence_4cat(md, hd),
                    "model_dist": md,
                }

            rep = {"baseline": _panel(e1), "steered": _panel(e8), "human_dist": hd}
            report[key] = rep
            if "loss" in rep["baseline"] and "loss" in rep["steered"]:
                log.info(
                    f"  {key} base_L={rep['baseline']['loss']:.4f} "
                    f"steered_L={rep['steered']['loss']:.4f}"
                )

    (out_dir / "E9_summary.json").write_text(json.dumps(report, indent=2))
    log.info("E9 complete.")


if __name__ == "__main__":
    main()
