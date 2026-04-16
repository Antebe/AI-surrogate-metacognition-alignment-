import pandas as pd
import pytest

from prompting import (
    build_nshot_prompt,
    select_exemplars,
    scored_items_mask,
)


def _fake_items():
    return pd.DataFrame({
        "item_num":  list(range(1, 11)),
        "item_id":   [f"Item{i}" for i in range(1, 11)],
        "question":  [f"Stem{i} is ____ ." for i in range(1, 11)],
        "correct_answer": [f"ans{i}" for i in range(1, 11)],
        "accurate_statement": [f"Stem{i} is ans{i}." for i in range(1, 11)],
        "exposure_statement": [f"Stem{i} is lure{i}." for i in range(1, 11)],
        "exposure_group": [0] * 10,
    })


def test_scored_items_mask_excludes_pool():
    items = _fake_items()
    mask = scored_items_mask(items, pool_item_nums=[8, 9, 10])
    assert mask.sum() == 7
    assert not mask.iloc[7] and not mask.iloc[8] and not mask.iloc[9]


def test_select_exemplars_returns_k_from_pool_only():
    items = _fake_items()
    ex = select_exemplars(items, pool_item_nums=[8, 9, 10], k=2, seed=0)
    assert len(ex) == 2
    assert set(ex["item_num"]).issubset({8, 9, 10})


def test_select_exemplars_is_seed_stable():
    items = _fake_items()
    a = select_exemplars(items, pool_item_nums=[1,2,3,4,5], k=3, seed=7)
    b = select_exemplars(items, pool_item_nums=[1,2,3,4,5], k=3, seed=7)
    assert list(a["item_num"]) == list(b["item_num"])


def test_build_nshot_prompt_zero_shot_has_no_exemplars(monkeypatch):
    items = _fake_items()
    row = items.iloc[0]
    out = build_nshot_prompt(
        persona_prefix="You are a 23-year-old female participant.",
        row=row,
        exemplars=items.iloc[:0],  # empty
    )
    assert "Example" not in out
    assert "You are a 23-year-old female participant." in out
    assert row["exposure_statement"] in out


def test_build_nshot_prompt_k_shot_includes_k_examples():
    items = _fake_items()
    row = items.iloc[0]
    exemplars = items.iloc[[7, 8]]   # 2 exemplars
    out = build_nshot_prompt(
        persona_prefix="You are a participant.",
        row=row,
        exemplars=exemplars,
    )
    # Each exemplar contributes a labeled "Example N:" block
    assert out.count("Example ") == 2
    assert "ans8" in out  # correct answer of exemplar shows in completion
    assert "ans9" in out


def test_build_nshot_prompt_exemplars_cannot_overlap_target_item():
    items = _fake_items()
    row = items.iloc[3]   # Item4
    exemplars = items.iloc[[3, 7]]   # deliberate collision
    with pytest.raises(ValueError):
        build_nshot_prompt(
            persona_prefix="p",
            row=row,
            exemplars=exemplars,
        )
