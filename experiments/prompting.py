"""
n-shot prompt assembly.

Exemplars are drawn from a fixed pool of held-out item numbers
(NSHOT_POOL_ITEM_NUMS in config). Scored items exclude the pool so
there's no train/test leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scored_items_mask(items: pd.DataFrame, pool_item_nums: list[int]) -> pd.Series:
    """Boolean mask over `items` for rows that should be scored (= not in pool)."""
    return ~items["item_num"].isin(pool_item_nums)


def select_exemplars(
    items: pd.DataFrame,
    pool_item_nums: list[int],
    k: int,
    seed: int,
) -> pd.DataFrame:
    """Return k exemplars drawn without replacement from the pool."""
    if k == 0:
        return items.iloc[:0]
    pool = items[items["item_num"].isin(pool_item_nums)]
    if len(pool) < k:
        raise ValueError(
            f"Need {k} exemplars but pool has only {len(pool)} items"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=k, replace=False)
    return pool.iloc[np.sort(idx)].reset_index(drop=True)


def _stem(question: str) -> str:
    return question.replace("____", "").strip().rstrip(".")


def build_nshot_prompt(
    persona_prefix: str,
    row: pd.Series,
    exemplars: pd.DataFrame,
) -> str:
    """Assemble a single persona-conditioned n-shot completion prompt.

    Exemplars must NOT include the target item.
    """
    if len(exemplars) > 0 and row["item_num"] in set(exemplars["item_num"]):
        raise ValueError(
            f"Exemplar pool overlaps target item: {row['item_num']}"
        )

    parts: list[str] = [persona_prefix.strip(), ""]

    # n-shot exemplars
    for i, (_, ex) in enumerate(exemplars.iterrows(), start=1):
        parts += [
            f"Example {i}:",
            f"The following fact was recently shared with you:",
            ex["exposure_statement"],
            "",
            f"Complete the sentence with ONE word or short phrase:",
            f"{_stem(ex['question'])} is/are {ex['correct_answer']}",
            "",
        ]

    # Target item
    parts += [
        "Now answer the following in the same format.",
        "",
        "The following fact was recently shared with you:",
        row["exposure_statement"],
        "",
        "Now complete this sentence with ONE word or short phrase:",
        f"{_stem(row['question'])} is/are",
    ]
    return "\n".join(parts)
