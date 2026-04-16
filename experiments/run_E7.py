#!/usr/bin/env python3
"""
E7 -- Individual Differences: does the alpha-sweep trace through human variance?

Reads E2 per-config summary stats (persona-averaged) and builds a 3-D
profile per sweep cell (brier_model, rho_model, acc_model). Compares to
per-participant human profiles via PCA + 1-D Wasserstein per axis.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from config import RESULTS_DIR, EXPOSURE_GROUPS_PRIMARY
from shared import setup_logging, setup_viz, load_human_data


def _load_sweep(eg: int) -> dict:
    return json.loads((RESULTS_DIR / "E2" / f"E2_sweep_EG{eg}.json").read_text())


def _config_profile(cell: dict) -> np.ndarray:
    return np.array([
        cell.get("brier_model", np.nan),
        cell.get("rho_model", np.nan),
        cell.get("acc_model", np.nan),
    ])


def main():
    log = setup_logging("E7")
    setup_viz()
    out_dir = RESULTS_DIR / "E7"
    out_dir.mkdir(parents=True, exist_ok=True)

    human_df = load_human_data()

    for eg in EXPOSURE_GROUPS_PRIMARY:
        sweep = _load_sweep(eg)
        profiles = np.stack([_config_profile(v) for v in sweep.values()])

        pca = PCA(n_components=min(2, profiles.shape[1]))
        emb = pca.fit_transform(profiles)

        per_p = (
            human_df[human_df["Exposure"] == eg]
            .groupby("ID_1")
            .apply(lambda g: pd.Series({
                "brier": ((g["Confidence"] - 1) / 3 - g["Correct"]).pow(2).mean(),
                "rho":   np.corrcoef((g["Confidence"] - 1) / 3, g["Correct"])[0, 1]
                         if g["Correct"].std() > 0 else 0.0,
                "acc":   g["Correct"].mean(),
            }))
            .dropna()
        )
        human_profiles = per_p.to_numpy()
        human_emb = pca.transform(human_profiles)

        w = [wasserstein_distance(emb[:, j], human_emb[:, j]) for j in range(emb.shape[1])]

        (out_dir / f"E7_EG{eg}.json").write_text(json.dumps({
            "sweep_pca": emb.tolist(),
            "human_pca": human_emb.tolist(),
            "wasserstein_per_axis": w,
        }, indent=2))
        log.info(f"  EG{eg}: W per axis = {w}")


if __name__ == "__main__":
    main()
