"""
Persona prompting: load unique participants and format a persona prefix
from their demographics.

The Uncertaintydata_long_share.csv columns:
  ID_1, Age_4 (raw age, misleading name), Gender, Education, Ethnicity,
  UncertaintyCondition, Exposure, Validity, Item, ...

Gender codebook (inferred; confirm with Andre):
  1=male, 2=female, 3=non-binary, 5=other/prefer-not-to-say.
Education/Ethnicity: raw category codes pass through; prose mode wraps
them in human-readable phrases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GENDER_MAP: dict[int, str] = {
    1: "male",
    2: "female",
    3: "non-binary",
    5: "other",
}

PERSONA_COLS = [
    "ID_1", "Age_4", "Gender", "Education",
    "Ethnicity", "UncertaintyCondition", "Exposure",
]


def load_personas(
    human_df: pd.DataFrame,
    n: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Return one row per unique participant (ID_1), optionally subsampled.

    Demographics are taken from the first row per ID_1 (stable within a
    participant by construction).
    """
    per_p = (
        human_df[PERSONA_COLS]
        .drop_duplicates(subset=["ID_1"])
        .reset_index(drop=True)
    )
    if n is not None and n < len(per_p):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(per_p), size=n, replace=False)
        per_p = per_p.iloc[np.sort(idx)].reset_index(drop=True)
    return per_p


def build_persona_prefix(row: pd.Series, style: str) -> str:
    """Render a single persona as a string prefix."""
    if style not in ("prose", "structured"):
        raise ValueError(f"Unknown persona style: {style!r}")

    age    = int(row["Age_4"])
    gender = int(row["Gender"])
    edu    = int(row["Education"])
    eth    = str(row["Ethnicity"])

    if style == "structured":
        return (
            f"Participant demographics: age={age}, gender={gender}, "
            f"education={edu}, ethnicity={eth}."
        )

    # prose
    gender_word = GENDER_MAP.get(gender, "participant")
    return (
        f"You are a {age}-year-old {gender_word} participant in a memory "
        f"experiment. Your demographic profile: education level {edu}, "
        f"ethnicity categories {eth}. Answer as this person would."
    )
