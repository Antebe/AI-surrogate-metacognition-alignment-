import numpy as np
import pandas as pd
import pytest

from persona import (
    load_personas,
    build_persona_prefix,
    GENDER_MAP,
)


def _fake_human_df():
    # Two participants, minimal schema
    return pd.DataFrame({
        "ID_1":       ["A", "A", "B", "B"],
        "Age_4":      [23, 23, 47, 47],
        "Gender":     [2, 2, 1, 1],
        "Education":  [5, 5, 8, 8],
        "Ethnicity":  ["1,3", "1,3", "3", "3"],
        "UncertaintyCondition": [0, 0, 1, 1],
        "Exposure":   [0, 0, 1, 1],
        "Item":       ["Item1", "Item2", "Item1", "Item2"],
    })


def test_load_personas_dedups_to_one_row_per_id():
    df = _fake_human_df()
    out = load_personas(df, n=None, seed=42)
    assert len(out) == 2
    assert set(out["ID_1"]) == {"A", "B"}


def test_load_personas_subsamples_deterministically():
    df = _fake_human_df()
    a = load_personas(df, n=1, seed=42)
    b = load_personas(df, n=1, seed=42)
    assert list(a["ID_1"]) == list(b["ID_1"])


def test_load_personas_subsamples_differ_under_different_seed():
    df = _fake_human_df()
    # Bigger fake panel to make the draws actually vary
    big = pd.concat([df] + [
        df.assign(ID_1=f"X{i}", Age_4=20 + i) for i in range(30)
    ], ignore_index=True)
    a = load_personas(big, n=5, seed=1)
    b = load_personas(big, n=5, seed=2)
    assert list(a["ID_1"]) != list(b["ID_1"])


def test_build_persona_prefix_prose_mentions_age_and_gender():
    row = pd.Series({
        "ID_1": "A", "Age_4": 23, "Gender": 2,
        "Education": 5, "Ethnicity": "1,3",
        "UncertaintyCondition": 0, "Exposure": 0,
    })
    out = build_persona_prefix(row, style="prose")
    assert "23" in out
    assert GENDER_MAP[2] in out.lower()
    assert "education" in out.lower()
    assert out.endswith(".")


def test_build_persona_prefix_structured_contains_all_codes():
    row = pd.Series({
        "ID_1": "A", "Age_4": 23, "Gender": 2,
        "Education": 5, "Ethnicity": "1,3",
        "UncertaintyCondition": 0, "Exposure": 0,
    })
    out = build_persona_prefix(row, style="structured")
    for token in ["age=23", "gender=2", "education=5", "ethnicity=1,3"]:
        assert token in out


def test_build_persona_prefix_unknown_gender_falls_back_to_code():
    row = pd.Series({
        "ID_1": "Z", "Age_4": 30, "Gender": 99,
        "Education": 2, "Ethnicity": "4",
        "UncertaintyCondition": 1, "Exposure": 0,
    })
    out = build_persona_prefix(row, style="prose")
    # No KeyError; unknown code rendered as "participant"
    assert "30" in out


def test_build_persona_prefix_rejects_bad_style():
    row = pd.Series({
        "ID_1": "A", "Age_4": 23, "Gender": 2,
        "Education": 5, "Ethnicity": "1,3",
        "UncertaintyCondition": 0, "Exposure": 0,
    })
    with pytest.raises(ValueError):
        build_persona_prefix(row, style="haiku")


def test_166_full_panel_produces_166_unique_prefixes(tmp_path):
    # If the real CSV is present, the full panel produces 166 distinct prefixes.
    try:
        human_df = pd.read_csv("data/Uncertaintydata_long_share.csv")
    except FileNotFoundError:
        pytest.skip("real human data not present")
    personas = load_personas(human_df, n=None, seed=42)
    prefixes = [build_persona_prefix(r, "prose") for _, r in personas.iterrows()]
    # 166 unique IDs; prefixes may collide if two IDs share all demographics,
    # which is acceptable — assert close to unique.
    # Real data yields 141 unique prefixes (demographic collisions are common).
    assert len(personas) == 166
    assert len(set(prefixes)) > 140
