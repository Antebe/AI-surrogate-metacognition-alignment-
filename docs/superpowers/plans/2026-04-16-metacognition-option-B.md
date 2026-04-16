# Option B — Metacognition / SAE-Steering Pipeline Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the existing SAE-steering pipeline to match the advisor's locked-in feedback — 166-persona simulation with demographics, 0/1/5-shot prompt baselines, Brier-based composite loss, and 4-category distributional comparisons — while keeping the `src/` SAE library untouched.

**Architecture:** Pure-Python extension of `experiments/`. Four new modules (`persona.py`, `prompting.py`, `brier.py`, `distribution.py`) provide the new primitives with unit tests. One new pilot script (`run_E0_pilot.py`) measures per-completion latency. The existing `run_E*.py` scripts are refactored to consume the new primitives; old result JSON is archived. All knobs live in `config.py`.

**Tech Stack:** Python 3.11 · uv · pytest · pandas · numpy · scipy · transformer_lens + sae_lens (via existing `src/`) · anthropic SDK · matplotlib + seaborn.

---

## File Structure

**Created:**
- `experiments/persona.py` — demographics → persona prefix
- `experiments/prompting.py` — n-shot exemplar selection + prompt assembly
- `experiments/brier.py` — Brier loss, human→probability mapping, composite loss
- `experiments/distribution.py` — Wasserstein + JS-divergence on 4-cat
- `experiments/run_E0_pilot.py` — timing probe
- `tests/__init__.py`
- `tests/test_persona.py`
- `tests/test_prompting.py`
- `tests/test_brier.py`
- `tests/test_distribution.py`
- `pyproject.toml` — pytest config
- `.env.example` — token template

**Modified:**
- `experiments/config.py` — new knobs + env-based token loading
- `experiments/shared.py` — import new modules, extend `run_condition` with persona/shot loops
- `experiments/run_E1.py` — rewritten for personas × shots × resamples
- `experiments/run_E2.py` — rewritten for Brier loss + persona-aware sweep
- `experiments/run_E8.py` — rewritten for α\* × {0,1,5}-shot
- `experiments/run_E3.py` — persona-aware, 0-shot
- `experiments/run_E5.py` — EG1 replication at base + α\*
- `experiments/run_E6.py` — held-out personas
- `experiments/run_E7.py` — read new E1/E2 schema
- `experiments/run_E9.py` — E1 vs E8 at matching shot settings
- `experiments/run_all.sh` — pilot-gated, per-step logs
- `requirements.txt` — add `pytest`, `python-dotenv`

**Archived (untouched on disk, renamed under `results/_archive_pre_option_B/`):**
- Existing `results/E{1,2,3,5,6,7,8,9}/` (schema incompatible).

---

## Phase 0 — Environment, tests scaffold, config knobs

### Task 1: Add pytest scaffold and dev deps

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create `pyproject.toml`**

Write to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
pythonpath = ["experiments", "."]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Create `tests/__init__.py` (empty file)**

Write an empty file at `tests/__init__.py`.

- [ ] **Step 3: Append to `requirements.txt`**

Append two lines (preserve existing content):

```
pytest
python-dotenv
```

- [ ] **Step 4: Install dev deps**

Run:
```bash
cd /home/cs29824/andre/sae_steering_toolkit
uv pip install pytest python-dotenv
```

Expected: pytest 8.x installs; no errors.

- [ ] **Step 5: Verify pytest discovers an empty test dir**

Run: `cd /home/cs29824/andre/sae_steering_toolkit && uv run pytest`
Expected: "no tests ran in 0.XXs" (exit 5 — acceptable, no tests yet).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/__init__.py requirements.txt
git commit -m "chore: add pytest scaffold and dev dependencies"
```

---

### Task 2: Env-based API tokens and `.env.example`

**Files:**
- Modify: `experiments/config.py` (top of file, after imports)
- Create: `.env.example`
- Modify: `.gitignore` (ensure `.env` is ignored)

- [ ] **Step 1: Read current `config.py`**

Run: `cat /home/cs29824/andre/sae_steering_toolkit/experiments/config.py`
Note the existing structure — add imports + keys block after the header comment but before the `Paths` block.

- [ ] **Step 2: Modify `experiments/config.py`**

Replace the `# ── API Keys ──` block (currently empty lines 13-17) with:

```python
# ── API Keys (read from environment; never commit them) ──────────────
import os
from dotenv import load_dotenv

# Load .env if present at project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_KEY")
    or os.environ.get("HUGGINGFACE_WRITE_KEY")
)
ANTHROPIC_KEY = (
    os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("ANTHROPIC_KEY")
)

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
```

(The existing `CLAUDE_MODEL` line should be *deleted* below the block to avoid a duplicate.)

- [ ] **Step 3: Create `.env.example`**

Write to `.env.example`:

```
# Copy to .env and fill in. .env is gitignored.
HF_TOKEN=hf_xxx_replace_me
ANTHROPIC_API_KEY=sk-ant-xxx_replace_me
```

- [ ] **Step 4: Verify `.gitignore` covers `.env`**

Run: `grep -q '^\.env$' .gitignore || echo '.env' >> .gitignore`
Expected: silent if present, or `.env` is appended.

- [ ] **Step 5: Smoke-test config imports**

Run:
```bash
cd /home/cs29824/andre/sae_steering_toolkit/experiments
uv run python -c "from config import HF_TOKEN, ANTHROPIC_KEY; print('HF=', bool(HF_TOKEN), 'ANTHROPIC=', bool(ANTHROPIC_KEY))"
```

Expected: `HF= True ANTHROPIC= False` (or True if you've set it — either is OK; we test the import, not the value).

- [ ] **Step 6: Commit**

```bash
git add experiments/config.py .env.example .gitignore
git commit -m "feat(config): load API tokens from env with fallbacks"
```

---

### Task 3: Add all Option-B knobs to `config.py`

**Files:**
- Modify: `experiments/config.py` (bottom of file)

- [ ] **Step 1: Append the Option-B knobs block**

Append to `experiments/config.py` (after the existing `EG_FILES` dict):

```python
# ══════════════════════════════════════════════════════════════════════════
# OPTION-B KNOBS
# ══════════════════════════════════════════════════════════════════════════

# ── Persona panel ────────────────────────────────────────────────────────
N_PERSONAS_FULL   = 166
N_PERSONAS_SWEEP  = 20
PERSONA_STYLE     = "prose"     # "prose" | "structured"
PERSONA_SEED      = 42

# ── Resamples (independent panel draws) ──────────────────────────────────
N_RESAMPLES_E1    = 3
N_RESAMPLES_SWEEP = 1
N_RESAMPLES_E8    = 5

# ── n-shot conditions (prompt-engineering baselines) ─────────────────────
N_SHOT_E1    = [0, 1, 5]
N_SHOT_SWEEP = [0]
N_SHOT_E8    = [0, 1, 5]

# Exemplar pool for n-shot: fixed held-out item numbers (never scored).
NSHOT_POOL_ITEM_NUMS = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80]

# ── Exposure groups ──────────────────────────────────────────────────────
EXPOSURE_GROUPS_PRIMARY = [0]
EXPOSURE_GROUPS_E5      = [0, 1]

# ── Loss (Brier-based) ───────────────────────────────────────────────────
LOSS_ALPHA      = 1.0    # |rho_model - rho_human|
LOSS_BETA       = 10.0   # max(0, acc_human - acc_steered)
HUMAN_CONF_MAP  = "empirical"   # "empirical" | "linear"

# ── Items held out for n-shot exemplars are excluded from scoring ────────
# Scored items = 80 − |NSHOT_POOL_ITEM_NUMS| = 70 by default.
```

- [ ] **Step 2: Verify imports**

Run:
```bash
cd /home/cs29824/andre/sae_steering_toolkit/experiments
uv run python -c "from config import N_PERSONAS_FULL, LOSS_BETA, N_SHOT_E1, NSHOT_POOL_ITEM_NUMS; print(N_PERSONAS_FULL, LOSS_BETA, N_SHOT_E1, len(NSHOT_POOL_ITEM_NUMS))"
```

Expected: `166 10.0 [0, 1, 5] 10`

- [ ] **Step 3: Commit**

```bash
git add experiments/config.py
git commit -m "feat(config): add Option-B knobs (persona, n-shot, Brier loss)"
```

---

## Phase 1 — New modules (TDD)

### Task 4: `persona.py` — load and format demographics

**Files:**
- Create: `experiments/persona.py`
- Create: `tests/test_persona.py`

- [ ] **Step 1: Write failing tests**

Write to `tests/test_persona.py`:

```python
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
    assert len(personas) == 166
    assert len(set(prefixes)) > 150
```

- [ ] **Step 2: Run test — expect import failure**

Run: `cd /home/cs29824/andre/sae_steering_toolkit && uv run pytest tests/test_persona.py -v`
Expected: `ModuleNotFoundError: No module named 'persona'`.

- [ ] **Step 3: Write `experiments/persona.py`**

```python
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
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `cd /home/cs29824/andre/sae_steering_toolkit && uv run pytest tests/test_persona.py -v`
Expected: 8 passed (the 166-unique test may skip if the CSV isn't found at cwd — that's fine from project root).

- [ ] **Step 5: Commit**

```bash
git add experiments/persona.py tests/test_persona.py
git commit -m "feat(persona): demographics → persona prefix loader + prompt renderer"
```

---

### Task 5: `prompting.py` — n-shot prompts with held-out exemplars

**Files:**
- Create: `experiments/prompting.py`
- Create: `tests/test_prompting.py`

- [ ] **Step 1: Write failing tests**

Write to `tests/test_prompting.py`:

```python
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
```

- [ ] **Step 2: Run test — expect import failure**

Run: `uv run pytest tests/test_prompting.py -v`
Expected: `ModuleNotFoundError: No module named 'prompting'`.

- [ ] **Step 3: Write `experiments/prompting.py`**

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

Run: `uv run pytest tests/test_prompting.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/prompting.py tests/test_prompting.py
git commit -m "feat(prompting): n-shot exemplar selection + persona-conditioned prompts"
```

---

### Task 6: `brier.py` — Brier loss, human-probability map, composite loss

**Files:**
- Create: `experiments/brier.py`
- Create: `tests/test_brier.py`

- [ ] **Step 1: Write failing tests**

Write to `tests/test_brier.py`:

```python
import numpy as np
import pandas as pd
import pytest

from brier import (
    build_human_prob_map,
    human_prob_for_response,
    brier_score,
    composite_loss,
    limiting_behavior_report,
)


def test_build_human_prob_map_empirical_matches_hand_computation():
    df = pd.DataFrame({
        "Confidence": [1, 1, 2, 2, 3, 3, 4, 4],
        "Correct":    [0, 0, 0, 1, 1, 1, 1, 1],   # acc by level: 0, .5, 1, 1
    })
    m = build_human_prob_map(df, mode="empirical")
    assert m == {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.0}


def test_build_human_prob_map_linear_is_fixed():
    df = pd.DataFrame({"Confidence": [1, 2, 3, 4], "Correct": [0, 0, 0, 0]})
    m = build_human_prob_map(df, mode="linear")
    assert m == {1: 0.0, 2: 1/3, 3: 2/3, 4: 1.0}


def test_build_human_prob_map_fills_missing_levels_with_global_rate():
    df = pd.DataFrame({
        "Confidence": [2, 2, 4, 4],   # no 1s, no 3s
        "Correct":    [0, 1, 1, 1],
    })
    m = build_human_prob_map(df, mode="empirical")
    # level 2: 0.5, level 4: 1.0, missing levels fall back to global mean 0.75
    assert m[2] == 0.5
    assert m[4] == 1.0
    assert m[1] == 0.75
    assert m[3] == 0.75


def test_human_prob_for_response_vectorizes():
    m = {1: 0.1, 2: 0.4, 3: 0.7, 4: 0.95}
    arr = human_prob_for_response(np.array([1, 2, 3, 4, 2]), m)
    np.testing.assert_allclose(arr, [0.1, 0.4, 0.7, 0.95, 0.4])


def test_brier_score_perfect_calibration_is_zero():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    correct = np.array([1, 0, 1, 0])
    assert brier_score(probs, correct) == pytest.approx(0.0)


def test_brier_score_worst_case_is_one():
    probs = np.array([1.0, 0.0])
    correct = np.array([0, 1])
    assert brier_score(probs, correct) == pytest.approx(1.0)


def test_composite_loss_identical_dists_is_zero():
    # Identical calibration, identical correlation, accuracy matched.
    mc = np.array([0.2, 0.5, 0.8, 1.0])
    ma = np.array([0.0, 0.5, 1.0, 1.0])
    hr = np.array([1, 2, 3, 4])          # maps to [0.0, 0.33, 0.67, 1.0] under linear
    h_acc = ma.copy()
    hmap = {1: 0.0, 2: 0.5, 3: 0.8, 4: 1.0}  # match model conf exactly
    L, parts = composite_loss(
        model_conf=mc, model_acc=ma,
        human_resp=hr, human_correct=h_acc,
        human_prob_map=hmap,
        alpha=1.0, beta=10.0,
    )
    assert parts["brier_diff"] == pytest.approx(0.0)
    assert parts["rho_diff"] == pytest.approx(0.0, abs=1e-9)
    assert parts["acc_penalty"] == pytest.approx(0.0)
    assert L == pytest.approx(0.0)


def test_composite_loss_accuracy_penalty_fires_only_below_human():
    mc = np.array([0.5, 0.5, 0.5, 0.5])
    ma = np.array([0.0, 0.0, 0.0, 0.0])     # model worthless
    hr = np.array([1, 1, 1, 1])
    hc = np.array([1, 1, 1, 1])             # humans all correct → acc_h = 1.0
    hmap = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    L, parts = composite_loss(
        model_conf=mc, model_acc=ma,
        human_resp=hr, human_correct=hc,
        human_prob_map=hmap,
        alpha=0.0, beta=10.0,
    )
    assert parts["acc_penalty"] == pytest.approx(10.0, abs=1e-6)
    # opposite: model at 1, humans at 0 → penalty = 0
    ma2 = np.ones_like(ma)
    L2, parts2 = composite_loss(
        model_conf=mc, model_acc=ma2,
        human_resp=hr, human_correct=np.zeros_like(hc),
        human_prob_map=hmap,
        alpha=0.0, beta=10.0,
    )
    assert parts2["acc_penalty"] == 0.0


def test_limiting_behavior_report_has_required_rows():
    out = limiting_behavior_report(alpha=1.0, beta=10.0)
    for key in ("brier_diff", "rho_diff", "acc_penalty"):
        assert key in out
        assert "best" in out[key] and "worst" in out[key]
```

- [ ] **Step 2: Run test — expect import failure**

Run: `uv run pytest tests/test_brier.py -v`
Expected: `ModuleNotFoundError: No module named 'brier'`.

- [ ] **Step 3: Write `experiments/brier.py`**

```python
"""
Brier-based composite loss for calibration alignment.

Loss (advisor-finalized, 2026-04-15):
  L = |Brier_model − Brier_human|
    + alpha · |rho(conf, acc)_model − rho(conf, acc)_human|
    + beta  · max(0, acc_human − acc_steered)

Per advisor: human 1-4 responses → probability via EMPIRICAL per-level
accuracy rate on the calibration split.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

LEVELS: tuple[int, ...] = (1, 2, 3, 4)


def build_human_prob_map(
    calib_df: pd.DataFrame,
    mode: Literal["empirical", "linear"] = "empirical",
) -> dict[int, float]:
    """Map 1-4 confidence response → probability of correctness.

    Empirical: per-level hit rate; missing levels fill with global mean.
    Linear:    (r - 1) / 3.
    """
    if mode == "linear":
        return {r: (r - 1) / 3 for r in LEVELS}

    if mode != "empirical":
        raise ValueError(f"Unknown mode: {mode!r}")

    global_rate = float(calib_df["Correct"].mean())
    by_level = (
        calib_df.groupby("Confidence")["Correct"].mean().to_dict()
    )
    return {r: float(by_level.get(r, global_rate)) for r in LEVELS}


def human_prob_for_response(
    responses: np.ndarray, prob_map: dict[int, float]
) -> np.ndarray:
    """Apply prob_map to an integer array of 1-4 responses."""
    out = np.asarray(responses, dtype=float).copy()
    for r, p in prob_map.items():
        out[np.asarray(responses) == r] = p
    return out


def brier_score(probs: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared deviation between predicted prob and ground truth."""
    p = np.asarray(probs, dtype=float)
    c = np.asarray(correct, dtype=float)
    return float(np.mean((p - c) ** 2))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def composite_loss(
    model_conf: np.ndarray,
    model_acc: np.ndarray,
    human_resp: np.ndarray,
    human_correct: np.ndarray,
    human_prob_map: dict[int, float],
    alpha: float,
    beta: float,
) -> tuple[float, dict[str, float]]:
    """Compute L and return a breakdown dict."""
    p_human = human_prob_for_response(human_resp, human_prob_map)
    brier_m = brier_score(model_conf, model_acc)
    brier_h = brier_score(p_human, human_correct)
    rho_m   = _pearson(model_conf, model_acc)
    rho_h   = _pearson(p_human, human_correct)
    acc_m   = float(np.asarray(model_acc, dtype=float).mean())
    acc_h   = float(np.asarray(human_correct, dtype=float).mean())

    brier_diff  = abs(brier_m - brier_h)
    rho_diff    = abs(rho_m - rho_h)
    acc_penalty = max(0.0, acc_h - acc_m)

    L = brier_diff + alpha * rho_diff + beta * acc_penalty
    return L, {
        "brier_model":  brier_m,
        "brier_human":  brier_h,
        "brier_diff":   brier_diff,
        "rho_model":    rho_m,
        "rho_human":    rho_h,
        "rho_diff":     rho_diff,
        "acc_model":    acc_m,
        "acc_human":    acc_h,
        "acc_penalty":  acc_penalty,
    }


def limiting_behavior_report(alpha: float, beta: float) -> dict[str, dict[str, float]]:
    """Per-term best/worst limits — advisor's sanity check."""
    return {
        "brier_diff": {"best": 0.0, "worst": 1.0},
        "rho_diff":   {"best": 0.0, "worst": 2.0 * alpha},
        "acc_penalty":{"best": 0.0, "worst": 1.0 * beta},
    }
```

- [ ] **Step 4: Run tests — expect pass**

Run: `uv run pytest tests/test_brier.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/brier.py tests/test_brier.py
git commit -m "feat(brier): composite loss with empirical human-prob mapping"
```

---

### Task 7: `distribution.py` — 4-category response distributional metrics

**Files:**
- Create: `experiments/distribution.py`
- Create: `tests/test_distribution.py`

- [ ] **Step 1: Write failing tests**

Write to `tests/test_distribution.py`:

```python
import numpy as np
import pytest

from distribution import (
    category_distribution,
    wasserstein_4cat,
    js_divergence_4cat,
    CATEGORIES,
)


def test_category_distribution_normalizes_to_one():
    cats = ["Correct", "Correct", "Lure", "Unsure", "Other"]
    dist = category_distribution(cats)
    assert set(dist) == set(CATEGORIES)
    assert dist["Correct"] == pytest.approx(0.4)
    assert dist["Lure"] == pytest.approx(0.2)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_category_distribution_handles_missing_categories():
    dist = category_distribution(["Correct"] * 3)
    assert dist["Correct"] == 1.0
    assert dist["Lure"] == 0.0


def test_wasserstein_identical_is_zero():
    a = {"Correct": .5, "Lure": .3, "Unsure": .1, "Other": .1}
    assert wasserstein_4cat(a, a) == pytest.approx(0.0)


def test_wasserstein_adjacent_shift_is_one():
    # 100% mass at Correct vs 100% at Lure
    a = {"Correct": 1.0, "Lure": 0.0, "Unsure": 0.0, "Other": 0.0}
    b = {"Correct": 0.0, "Lure": 1.0, "Unsure": 0.0, "Other": 0.0}
    assert wasserstein_4cat(a, b) == pytest.approx(1.0)


def test_wasserstein_max_spread_is_three():
    a = {"Correct": 1.0, "Lure": 0.0, "Unsure": 0.0, "Other": 0.0}
    b = {"Correct": 0.0, "Lure": 0.0, "Unsure": 0.0, "Other": 1.0}
    # 4-category ordering 0..3, total cost = 3
    assert wasserstein_4cat(a, b) == pytest.approx(3.0)


def test_js_divergence_identical_is_zero():
    a = {"Correct": .5, "Lure": .3, "Unsure": .1, "Other": .1}
    assert js_divergence_4cat(a, a) == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_symmetric():
    a = {"Correct": .5, "Lure": .3, "Unsure": .1, "Other": .1}
    b = {"Correct": .25, "Lure": .25, "Unsure": .25, "Other": .25}
    ab = js_divergence_4cat(a, b)
    ba = js_divergence_4cat(b, a)
    assert ab == pytest.approx(ba, abs=1e-12)
```

- [ ] **Step 2: Run test — expect import failure**

Run: `uv run pytest tests/test_distribution.py -v`
Expected: `ModuleNotFoundError: No module named 'distribution'`.

- [ ] **Step 3: Write `experiments/distribution.py`**

```python
"""
Distributional metrics over the 4-category response space:
Correct (0), Lure (1), Unsure (2), Other (3).

1-Wasserstein on the ordered categorical.
Jensen–Shannon divergence (symmetric KL) with additive smoothing.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

CATEGORIES: tuple[str, ...] = ("Correct", "Lure", "Unsure", "Other")


def category_distribution(cats: Iterable[str]) -> dict[str, float]:
    """Normalized frequency over the 4-category space. Missing = 0."""
    counts = Counter(cats)
    total = sum(counts.values())
    if total == 0:
        return {c: 0.0 for c in CATEGORIES}
    return {c: counts.get(c, 0) / total for c in CATEGORIES}


def _as_vec(dist: dict[str, float]) -> np.ndarray:
    return np.array([dist.get(c, 0.0) for c in CATEGORIES], dtype=float)


def wasserstein_4cat(a: dict[str, float], b: dict[str, float]) -> float:
    """1-Wasserstein (Earth Mover's) over the ordered 4-category support."""
    pa, pb = _as_vec(a), _as_vec(b)
    # Cumulative differences give the 1-Wasserstein for 1D distributions
    return float(np.sum(np.abs(np.cumsum(pa) - np.cumsum(pb))))


def js_divergence_4cat(
    a: dict[str, float], b: dict[str, float], eps: float = 1e-12
) -> float:
    """Symmetric Jensen–Shannon divergence (base-2), smoothed."""
    pa, pb = _as_vec(a) + eps, _as_vec(b) + eps
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        return float(np.sum(p * (np.log2(p) - np.log2(q))))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `uv run pytest tests/test_distribution.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/distribution.py tests/test_distribution.py
git commit -m "feat(distribution): 1-Wasserstein + JS-divergence on 4-category response space"
```

---

### Task 8: Wire new modules into `shared.py`

**Files:**
- Modify: `experiments/shared.py` (imports + `run_condition` extension)

- [ ] **Step 1: Read current `shared.py` `run_condition()`**

Run: `sed -n '340,410p' experiments/shared.py`
Note the single items × samples loop on line 357. We replace it with a persona × resample × items loop.

- [ ] **Step 2: Add imports (top of shared.py, after `from src import *`)**

Insert after line `from src import *`:

```python
from persona import load_personas, build_persona_prefix
from prompting import build_nshot_prompt, select_exemplars
from brier import (
    build_human_prob_map, human_prob_for_response, brier_score,
    composite_loss, limiting_behavior_report,
)
from distribution import (
    category_distribution, wasserstein_4cat, js_divergence_4cat, CATEGORIES,
)
```

- [ ] **Step 3: Replace `run_condition` with a persona-aware version**

Replace the `def run_condition(...)` block (lines ~341–407) with:

```python
def run_condition(
    model,
    items: pd.DataFrame,
    personas: pd.DataFrame,
    interventions: list | None,
    client: anthropic.Anthropic,
    n_shot: int,
    n_resamples: int,
    logger: Optional[logging.Logger] = None,
    checkpoint_path: Optional[Path] = None,
    existing_results: Optional[dict] = None,
) -> dict:
    """Run every (persona, item) at a single (intervention, n_shot) configuration.

    Returns: {persona_id: {item_id: {...}}}. Resumable via checkpoint.
    """
    results: dict[str, dict[str, dict]] = dict(existing_results) if existing_results else {}

    scored = items[scored_items_mask(items, NSHOT_POOL_ITEM_NUMS)].reset_index(drop=True)

    for _, prow in tqdm(personas.iterrows(), total=len(personas), desc="Personas"):
        pid = str(prow["ID_1"])
        prefix = build_persona_prefix(prow, style=PERSONA_STYLE)
        results.setdefault(pid, {})

        # n-shot exemplars: seed by persona so the same persona sees
        # the same exemplars across items (consistent within participant).
        persona_seed = (PERSONA_SEED + hash(pid)) & 0xFFFFFFFF
        exemplars = select_exemplars(
            items, pool_item_nums=NSHOT_POOL_ITEM_NUMS,
            k=n_shot, seed=persona_seed,
        )

        for _, row in scored.iterrows():
            iid = str(row["item_id"])
            if iid in results[pid]:
                continue

            prompt = build_nshot_prompt(prefix, row, exemplars)
            raw = generate(
                model, prompt, interventions=interventions,
                max_new_tokens=MAX_NEW_TOKENS, n=n_resamples,
                temperature=TEMPERATURE,
            )
            completions = [
                o[len(prompt):].strip() if o.startswith(prompt) else o.strip()
                for o in raw
            ]
            conf_data = [get_confidence(model, prompt, c, client) for c in completions]
            confs    = [cd[0] for cd in conf_data]
            probs    = [cd[1]["probs"] for cd in conf_data]
            coded    = [
                code_response(row["exposure_statement"], row["question"],
                              row["correct_answer"], c, client)
                for c in completions
            ]
            cats = [cat for cat, _ in coded]
            accs = [acc for _, acc in coded]

            results[pid][iid] = {
                "prompt": prompt,
                "completions": completions,
                "confidence_probs": probs,
                "confidence_scores": confs,
                "response_categories": cats,
                "accuracy_scores": accs,
                "mean_confidence": float(np.mean(confs)),
                "mean_accuracy": float(np.mean(accs)),
            }

            if checkpoint_path:
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f, indent=2)

        if logger:
            n_items_done = len(results[pid])
            logger.info(f"  persona={pid} items={n_items_done}")

    return results
```

- [ ] **Step 4: Run the pure tests (persona/prompting/brier/distribution) to confirm nothing broke**

Run: `cd /home/cs29824/andre/sae_steering_toolkit && uv run pytest tests -v`
Expected: All unit tests pass (model-independent).

- [ ] **Step 5: Commit**

```bash
git add experiments/shared.py
git commit -m "feat(shared): persona/shot-aware run_condition, wired to new modules"
```

---

## Phase 2 — Pilot + E1

### Task 9: `run_E0_pilot.py` — per-completion timing probe

**Files:**
- Create: `experiments/run_E0_pilot.py`

- [ ] **Step 1: Write the pilot script**

```python
#!/usr/bin/env python3
"""
E0 -- Pilot: measure seconds per completion on this box.

Runs 1 persona × 3 items × 1 resample, unsteered, 0-shot.
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

import pandas as pd

from config import (
    RESULTS_DIR, HF_TOKEN, EG_FILES, PERSONA_SEED, PERSONA_STYLE,
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

    items = load_items(0).head(3)   # 3 items only

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

    # Sanity: every confidence_probs vector sums to 1
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
```

- [ ] **Step 2: Dry-run the script with a tiny smoke check (no model load)**

Only attempt a run if the box has HF + Anthropic tokens. Otherwise verify it parses:

```bash
cd /home/cs29824/andre/sae_steering_toolkit
uv run python -c "import py_compile; py_compile.compile('experiments/run_E0_pilot.py', doraise=True); print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add experiments/run_E0_pilot.py
git commit -m "feat(E0): pilot timing probe"
```

---

### Task 10: Archive old results and rewrite `run_E1.py`

**Files:**
- Modify: `experiments/run_E1.py` (full rewrite)
- Move: existing `results/E1`, `results/E2`, `results/E8` → `results/_archive_pre_option_B/`

- [ ] **Step 1: Archive old results (schema incompatible)**

```bash
cd /home/cs29824/andre/sae_steering_toolkit
mkdir -p results/_archive_pre_option_B
for d in E1 E2 E3 E5 E6 E7 E8 E9; do
    [ -d "results/$d" ] && git mv "results/$d" "results/_archive_pre_option_B/$d" || true
done
git commit -m "chore: archive pre-Option-B results (incompatible schema)"
```

- [ ] **Step 2: Overwrite `experiments/run_E1.py` with the persona/shot-aware version**

```python
#!/usr/bin/env python3
"""
E1 -- Baselines: unsteered Gemma with persona prompting, across n-shot.

For each (n_shot in N_SHOT_E1) × EXPOSURE_GROUPS_PRIMARY:
  run 166 personas × N_RESAMPLES_E1 resamples × 80 items, unsteered.

Saves:
  results/E1/E1_baseline_EG{eg}_shot{k}.json
  results/E1/E1_summary.json
  results/E1/brier_human_prob_map.json
  results/E1/plots/*.png
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import numpy as np
import pandas as pd

from config import (
    RESULTS_DIR, HF_TOKEN, N_PERSONAS_FULL, PERSONA_SEED,
    N_RESAMPLES_E1, N_SHOT_E1, EXPOSURE_GROUPS_PRIMARY,
    HUMAN_CONF_MAP, LOSS_ALPHA, LOSS_BETA,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    get_participant_split, load_model, get_client, run_condition,
)
from persona import load_personas
from brier import build_human_prob_map, composite_loss


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

    # Human-prob map — computed once, saved for downstream scripts
    human_prob_map = build_human_prob_map(calib_df, mode=HUMAN_CONF_MAP)
    with open(out_dir / "brier_human_prob_map.json", "w") as f:
        json.dump(human_prob_map, f, indent=2)
    log.info(f"Human prob map: {human_prob_map}")

    personas = load_personas(human_df, n=N_PERSONAS_FULL, seed=PERSONA_SEED)
    log.info(f"Loaded {len(personas)} personas")

    summary = {"conditions": []}

    for eg in EXPOSURE_GROUPS_PRIMARY:
        items = load_items(eg)

        for k in N_SHOT_E1:
            for rs in range(N_RESAMPLES_E1):
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
```

- [ ] **Step 3: Syntax-check**

```bash
uv run python -c "import py_compile; py_compile.compile('experiments/run_E1.py', doraise=True); print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add experiments/run_E1.py
git commit -m "feat(E1): rewrite for personas × n-shot × resamples"
```

---

## Phase 3 — E2 Brier-based sweep

### Task 11: Rewrite `run_E2.py`

**Files:**
- Modify: `experiments/run_E2.py` (full rewrite)

- [ ] **Step 1: Overwrite `experiments/run_E2.py`**

```python
#!/usr/bin/env python3
"""
E2 -- Core Steering: optimal (unsure, refusal) coefficient search.

Uses the Brier-based composite loss. Sweep is persona-averaged over a
rotating subset of N_PERSONAS_SWEEP; 0-shot only.

Saves:
  results/E2/E2_sweep_EG{eg}.json
  results/E2/E2_optimal_alpha.json
  results/E2/E2_summary.json
  results/E2/limiting_behavior.md
  results/E2/plots/*.png
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    RESULTS_DIR, HF_TOKEN, SAE_LAYERS, SWEEP_UNSURE_VALS, SWEEP_REFUSAL_VALS,
    N_PERSONAS_SWEEP, PERSONA_SEED, N_RESAMPLES_SWEEP, EXPOSURE_GROUPS_PRIMARY,
    LOSS_ALPHA, LOSS_BETA, HUMAN_CONF_MAP,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    get_participant_split, load_model, load_saes, get_client, run_condition,
    make_target_interventions,
)
from persona import load_personas
from brier import build_human_prob_map, composite_loss, limiting_behavior_report


def _flatten(results: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flatten persona-keyed results to (conf[], acc[]) arrays."""
    confs, accs = [], []
    for persona_results in results.values():
        for v in persona_results.values():
            confs.append(v["mean_confidence"])
            accs.append(v["mean_accuracy"])
    return np.array(confs), np.array(accs)


def _human_conf_acc(human_df: pd.DataFrame, item_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (Confidence, Correct) from the calibration split, restricted to scored items."""
    sub = human_df[human_df["Item"].isin(item_ids)]
    return sub["Confidence"].to_numpy(), sub["Correct"].to_numpy()


def main():
    log = setup_logging("E2")
    setup_viz()
    out_dir = RESULTS_DIR / "E2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    log.info("Loading model + SAEs...")
    model = load_model(hf_token=HF_TOKEN)
    saes  = load_saes(layers=SAE_LAYERS)
    client = get_client()

    human_df = load_human_data()
    calib_ids, _ = get_participant_split(human_df)
    calib_df = human_df[human_df["ID_1"].isin(calib_ids)]
    human_prob_map = build_human_prob_map(calib_df, mode=HUMAN_CONF_MAP)
    log.info(f"Human prob map: {human_prob_map}")

    personas = load_personas(human_df, n=N_PERSONAS_SWEEP, seed=PERSONA_SEED)

    # Limiting-behavior sanity check (advisor-requested)
    limits = limiting_behavior_report(LOSS_ALPHA, LOSS_BETA)
    with open(out_dir / "limiting_behavior.md", "w") as f:
        f.write("# Loss term limits\n\n")
        for k, v in limits.items():
            f.write(f"- **{k}**: best={v['best']:.3f}, worst={v['worst']:.3f}\n")

    best = {"loss": float("inf"), "u": None, "r": None}
    all_summaries: dict[str, dict] = {}

    for eg in EXPOSURE_GROUPS_PRIMARY:
        items = load_items(eg)
        scored_ids = set(items["item_id"])
        h_resp, h_corr = _human_conf_acc(
            calib_df[calib_df["Exposure"] == eg], scored_ids
        )

        sweep_path = out_dir / f"E2_sweep_EG{eg}.json"
        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep = json.load(f)
        else:
            sweep = {}

        grid = [(u, r) for u in SWEEP_UNSURE_VALS for r in SWEEP_REFUSAL_VALS]

        for u, r in tqdm(grid, desc=f"EG{eg} grid"):
            key = f"{u},{r}"
            if key in sweep:
                continue

            ivs = make_target_interventions(saes, u, r)
            results = run_condition(
                model=model, items=items, personas=personas,
                interventions=ivs, client=client,
                n_shot=0, n_resamples=N_RESAMPLES_SWEEP, logger=log,
            )
            mc, ma = _flatten(results)
            L, parts = composite_loss(
                model_conf=mc, model_acc=ma,
                human_resp=h_resp, human_correct=h_corr,
                human_prob_map=human_prob_map,
                alpha=LOSS_ALPHA, beta=LOSS_BETA,
            )
            sweep[key] = {
                "unsure_coeff": u, "refusal_coeff": r,
                "loss": L, **parts,
                "n_completions": int(len(mc)),
            }
            with open(sweep_path, "w") as f:
                json.dump(sweep, f, indent=2)

            log.info(
                f"  alpha=({u},{r}) L={L:.4f} brier_diff={parts['brier_diff']:.3f} "
                f"rho_diff={parts['rho_diff']:.3f} acc_pen={parts['acc_penalty']:.3f}"
            )

            if L < best["loss"]:
                best = {"loss": L, "u": u, "r": r}

        all_summaries[f"EG{eg}"] = sweep

    with open(out_dir / "E2_optimal_alpha.json", "w") as f:
        json.dump({
            "optimal_alpha": f"{best['u']},{best['r']}",
            "unsure_coeff": best["u"], "refusal_coeff": best["r"],
            "combined_loss": best["loss"],
        }, f, indent=2)
    with open(out_dir / "E2_summary.json", "w") as f:
        json.dump({"sweeps": all_summaries, "best": best}, f, indent=2)

    log.info(f"α* = ({best['u']}, {best['r']})  L* = {best['loss']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

```bash
uv run python -c "import py_compile; py_compile.compile('experiments/run_E2.py', doraise=True); print('ok')"
```

- [ ] **Step 3: Commit**

```bash
git add experiments/run_E2.py
git commit -m "feat(E2): Brier-based sweep with persona-averaged evaluation"
```

---

## Phase 4 — E8 steered + E3/E5/E6/E7/E9 patches

### Task 12: Rewrite `run_E8.py`

**Files:**
- Modify: `experiments/run_E8.py`

- [ ] **Step 1: Overwrite `experiments/run_E8.py`**

```python
#!/usr/bin/env python3
"""
E8 -- Steered Model Performance at α* across n-shot conditions.

For each (n_shot in N_SHOT_E8): run 166 personas × N_RESAMPLES_E8 × 80 items
under optimal steering α* (from E2).

Saves:
  results/E8/E8_steered_EG{eg}_shot{k}.json
  results/E8/E8_summary.json
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import (
    RESULTS_DIR, HF_TOKEN, SAE_LAYERS, N_PERSONAS_FULL, PERSONA_SEED,
    N_RESAMPLES_E8, N_SHOT_E8, EXPOSURE_GROUPS_PRIMARY,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    load_model, load_saes, get_client, run_condition,
    make_target_interventions,
)
from persona import load_personas


def main():
    log = setup_logging("E8")
    setup_viz()
    out_dir = RESULTS_DIR / "E8"
    out_dir.mkdir(parents=True, exist_ok=True)

    opt_path = RESULTS_DIR / "E2" / "E2_optimal_alpha.json"
    with open(opt_path) as f:
        opt = json.load(f)
    u, r = opt["unsure_coeff"], opt["refusal_coeff"]
    log.info(f"α* = ({u}, {r})")

    log.info("Loading model + SAEs...")
    model = load_model(hf_token=HF_TOKEN)
    saes  = load_saes(layers=SAE_LAYERS)
    client = get_client()

    ivs = make_target_interventions(saes, u, r)

    human_df = load_human_data()
    personas = load_personas(human_df, n=N_PERSONAS_FULL, seed=PERSONA_SEED)
    log.info(f"Personas: {len(personas)}")

    summary = {"optimal_alpha": opt, "conditions": []}

    for eg in EXPOSURE_GROUPS_PRIMARY:
        items = load_items(eg)

        for k in N_SHOT_E8:
            for rs in range(N_RESAMPLES_E8):
                cond = f"EG{eg}_shot{k}_rs{rs}"
                ckpt = out_dir / f"E8_steered_{cond}.json"
                existing = {}
                if ckpt.exists():
                    with open(ckpt) as f:
                        existing = json.load(f)

                results = run_condition(
                    model=model, items=items, personas=personas,
                    interventions=ivs, client=client,
                    n_shot=k, n_resamples=1, logger=log,
                    checkpoint_path=ckpt, existing_results=existing,
                )
                with open(ckpt, "w") as f:
                    json.dump(results, f, indent=2)
                summary["conditions"].append({
                    "exposure_group": eg, "n_shot": k, "resample": rs,
                    "n_personas": len(results),
                    "file": str(ckpt.name),
                })

    with open(out_dir / "E8_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check + commit**

```bash
uv run python -c "import py_compile; py_compile.compile('experiments/run_E8.py', doraise=True); print('ok')"
git add experiments/run_E8.py
git commit -m "feat(E8): steered α* × n-shot conditions over full 166-persona panel"
```

---

### Task 13: Patch E3, E5, E6 to use the new schema

**Files:**
- Modify: `experiments/run_E3.py` (persona-aware, 0-shot)
- Modify: `experiments/run_E5.py` (EG1 replication at baseline + α\*)
- Modify: `experiments/run_E6.py` (held-out personas)

- [ ] **Step 1: Rewrite `experiments/run_E3.py`**

```python
#!/usr/bin/env python3
"""
E3 -- Specificity ablation: do 3 control latents produce the same gain?

Runs the N_PERSONAS_SWEEP subset × N_RESAMPLES_SWEEP × 80 items at the
optimal coefficients but with CONTROL_LATENTS instead of the target set.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import numpy as np

from config import (
    RESULTS_DIR, HF_TOKEN, SAE_LAYERS, CONTROL_LATENTS,
    N_PERSONAS_SWEEP, PERSONA_SEED, N_RESAMPLES_SWEEP,
    EXPOSURE_GROUPS_PRIMARY, HUMAN_CONF_MAP, LOSS_ALPHA, LOSS_BETA,
)
from shared import (
    setup_logging, setup_viz, load_items, load_human_data,
    get_participant_split, load_model, load_saes, get_client,
    run_condition, make_control_interventions,
)
from persona import load_personas
from brier import build_human_prob_map, composite_loss


def _flatten(results):
    import numpy as np
    confs, accs = [], []
    for p in results.values():
        for v in p.values():
            confs.append(v["mean_confidence"]); accs.append(v["mean_accuracy"])
    return np.array(confs), np.array(accs)


def main():
    log = setup_logging("E3")
    setup_viz()
    out_dir = RESULTS_DIR / "E3"
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = json.loads((RESULTS_DIR / "E2" / "E2_optimal_alpha.json").read_text())
    u_opt, r_opt = opt["unsure_coeff"], opt["refusal_coeff"]

    model = load_model(hf_token=HF_TOKEN)
    saes = load_saes(layers=SAE_LAYERS)
    client = get_client()

    human_df = load_human_data()
    calib_ids, _ = get_participant_split(human_df)
    calib_df = human_df[human_df["ID_1"].isin(calib_ids)]
    human_prob_map = build_human_prob_map(calib_df, mode=HUMAN_CONF_MAP)
    personas = load_personas(human_df, n=N_PERSONAS_SWEEP, seed=PERSONA_SEED)

    findings = {}
    for ctrl_name in CONTROL_LATENTS:
        for eg in EXPOSURE_GROUPS_PRIMARY:
            items = load_items(eg)
            ivs = make_control_interventions(saes, ctrl_name, max(u_opt, r_opt))
            ckpt = out_dir / f"E3_{ctrl_name}_EG{eg}.json"
            existing = json.loads(ckpt.read_text()) if ckpt.exists() else {}
            results = run_condition(
                model=model, items=items, personas=personas,
                interventions=ivs, client=client,
                n_shot=0, n_resamples=N_RESAMPLES_SWEEP, logger=log,
                checkpoint_path=ckpt, existing_results=existing,
            )
            with open(ckpt, "w") as f:
                json.dump(results, f, indent=2)

            mc, ma = _flatten(results)
            sub = calib_df[calib_df["Exposure"] == eg]
            L, parts = composite_loss(
                model_conf=mc, model_acc=ma,
                human_resp=sub["Confidence"].to_numpy(),
                human_correct=sub["Correct"].to_numpy(),
                human_prob_map=human_prob_map,
                alpha=LOSS_ALPHA, beta=LOSS_BETA,
            )
            findings[f"{ctrl_name}_EG{eg}"] = {"loss": L, **parts}
            log.info(f"  {ctrl_name} EG{eg}: L={L:.4f}")

    with open(out_dir / "E3_summary.json", "w") as f:
        json.dump(findings, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rewrite `experiments/run_E5.py` (EG1 replication)**

```python
#!/usr/bin/env python3
"""
E5 -- Item-Type Asymmetry: does steering improve calibration more
on lure-exposure items (the misinformation effect)?

Runs BOTH EG0 and EG1 at baseline and α* (N_PERSONAS_FULL × 3 resamples),
0-shot. Then stratifies by Validity.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import (
    RESULTS_DIR, HF_TOKEN, SAE_LAYERS, N_PERSONAS_FULL, PERSONA_SEED,
    N_RESAMPLES_E1, EXPOSURE_GROUPS_E5,
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
```

- [ ] **Step 3: Rewrite `experiments/run_E6.py` (held-out personas)**

```python
#!/usr/bin/env python3
"""
E6 -- Generalization: does α* transfer to held-out (test) participants?

Runs 66 held-out personas × 3 resamples × 80 items at α*, 0-shot, EG0.
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
```

- [ ] **Step 4: Syntax-check all three**

```bash
for f in run_E3.py run_E5.py run_E6.py; do
  uv run python -c "import py_compile; py_compile.compile('experiments/$f', doraise=True); print('$f ok')"
done
```

- [ ] **Step 5: Commit**

```bash
git add experiments/run_E3.py experiments/run_E5.py experiments/run_E6.py
git commit -m "feat(E3/E5/E6): persona-aware rewrites consistent with new schema"
```

---

### Task 14: Patch `run_E7.py` and `run_E9.py` (analysis-only)

**Files:**
- Modify: `experiments/run_E7.py` (read new schema)
- Modify: `experiments/run_E9.py` (consume E1/E8 across shot conditions)

- [ ] **Step 1: Overwrite `experiments/run_E7.py` with schema-adapted analysis**

```python
#!/usr/bin/env python3
"""
E7 -- Individual Differences: does the α-sweep trace through human variance?

Reads E2 per-config results (now persona-keyed) and builds per-sweep-config
model profiles; compares to per-participant human profiles via PCA and
Wasserstein.
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


def _config_profile(sweep_entry: dict) -> np.ndarray:
    """A per-config vector = mean_confidence across items (order-stable)."""
    # sweep_entry is the E2 cell; we need per-item means which we don't
    # store post-rewrite. Fall back to the summary stats stored in E2:
    # (brier_model, rho_model, acc_model) as a 3-d "profile".
    return np.array([
        sweep_entry.get("brier_model", np.nan),
        sweep_entry.get("rho_model", np.nan),
        sweep_entry.get("acc_model", np.nan),
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

        # PCA on sweep profiles
        pca = PCA(n_components=min(2, profiles.shape[1]))
        emb = pca.fit_transform(profiles)

        # Per-participant: (brier_participant, rho_participant, acc_participant)
        per_p = (
            human_df[human_df["Exposure"] == eg]
            .groupby("ID_1")
            .apply(lambda g: pd.Series({
                "brier": ((g["Confidence"] - 1) / 3 - g["Correct"]).pow(2).mean(),
                "rho":   np.corrcoef((g["Confidence"] - 1)/3, g["Correct"])[0,1]
                         if g["Correct"].std() > 0 else 0.0,
                "acc":   g["Correct"].mean(),
            }))
            .dropna()
        )
        human_profiles = per_p.to_numpy()
        human_emb = pca.transform(human_profiles)

        # Wasserstein on each PCA axis
        w = [wasserstein_distance(emb[:, j], human_emb[:, j]) for j in range(emb.shape[1])]

        (out_dir / f"E7_EG{eg}.json").write_text(json.dumps({
            "sweep_pca": emb.tolist(),
            "human_pca": human_emb.tolist(),
            "wasserstein_per_axis": w,
        }, indent=2))
        log.info(f"  EG{eg}: W per axis = {w}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Overwrite `experiments/run_E9.py` with cross-condition comparison**

```python
#!/usr/bin/env python3
"""
E9 -- E1 vs E8 comprehensive analysis.

Compares baseline (E1) to steered (E8) at matched shot settings, using:
  - Brier, correlation, accuracy
  - 4-category distributional metrics (Wasserstein, JS)
  - Per-category confidence means
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


def _load_condition(prefix: str, eg: int, k: int) -> dict:
    """Merge all resample files for a given prefix/EG/shot into one persona dict."""
    merged = defaultdict(dict)
    pat = f"{prefix}_EG{eg}_shot{k}_rs*.json"
    for p in sorted((RESULTS_DIR / prefix.split("_")[0]).glob(pat)):
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
        # Human categorical distribution
        human_cats = []
        for _, r in sub.iterrows():
            if r["Correct"]:    human_cats.append("Correct")
            elif r["Error"]:    human_cats.append("Lure")
            elif r["Unsure"]:   human_cats.append("Unsure")
            else:               human_cats.append("Other")
        hd = category_distribution(human_cats)

        for k in N_SHOT_E8:
            key = f"EG{eg}_shot{k}"
            e1 = _load_condition("E1_baseline", eg, k)
            e8 = _load_condition("E8_steered", eg, k)

            def _panel(results):
                mc, ma, cats = _flatten(results)
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

            r = {"baseline": _panel(e1), "steered": _panel(e8), "human_dist": hd}
            report[key] = r
            log.info(
                f"  {key} base_L={r['baseline']['loss']:.4f} "
                f"steered_L={r['steered']['loss']:.4f}  "
                f"W(base,hu)={r['baseline']['wasserstein_vs_human']:.3f} "
                f"W(steer,hu)={r['steered']['wasserstein_vs_human']:.3f}"
            )

    (out_dir / "E9_summary.json").write_text(json.dumps(report, indent=2))
    log.info("E9 complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax-check + commit**

```bash
for f in run_E7.py run_E9.py; do
  uv run python -c "import py_compile; py_compile.compile('experiments/$f', doraise=True); print('$f ok')"
done
git add experiments/run_E7.py experiments/run_E9.py
git commit -m "feat(E7/E9): analysis scripts adapted to new persona schema"
```

---

### Task 15: Update `run_all.sh` and README

**Files:**
- Modify: `experiments/run_all.sh`
- Modify: `README.md` — update Quick Start, API tokens section

- [ ] **Step 1: Overwrite `experiments/run_all.sh`**

```bash
#!/usr/bin/env bash
# Option B pipeline — tmux-friendly, resumable, pilot-gated.
# Usage:
#   tmux new -s mcog 'cd experiments && bash run_all.sh; exec bash'
set -euo pipefail
cd "$(dirname "$0")"

source /home/cs29824/.venv/bin/activate
: "${HF_TOKEN:?HF_TOKEN not set — see README 'API tokens' section}"
: "${ANTHROPIC_API_KEY:=${ANTHROPIC_KEY:-}}"
export ANTHROPIC_API_KEY
[ -n "$ANTHROPIC_API_KEY" ] || { echo "ANTHROPIC_API_KEY not set"; exit 1; }

LOG_ROOT="../logs/run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
echo "Logs: $LOG_ROOT"

run () {
    local name=$1
    echo "──── $name  $(date +%H:%M:%S) ────"
    uv run python "run_${name}.py" 2>&1 | tee "$LOG_ROOT/${name}.log"
    echo "──── $name done $(date +%H:%M:%S) ────"
}

run E0_pilot
run E1
run E2
run E8
run E5
run E3
run E6
run E7
run E9

echo "ALL DONE $(date)"
```

Then: `chmod +x experiments/run_all.sh`.

- [ ] **Step 2: Prepend an "API tokens" section to README.md**

Insert before the "Quick Start" section in `README.md`:

```markdown
## API tokens

This pipeline needs two tokens:

| Var | Where to get it |
|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens (accept Gemma-2-9B license first) |
| `ANTHROPIC_API_KEY` | https://console.anthropic.com/settings/keys |

Set them persistently:

```bash
echo 'export HF_TOKEN="hf_..."'              >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

Or copy `.env.example` → `.env` and fill in (loaded automatically by `config.py`).
```

- [ ] **Step 3: Commit**

```bash
chmod +x experiments/run_all.sh
git add experiments/run_all.sh README.md
git commit -m "chore: tmux-friendly run_all.sh + API tokens doc"
```

---

## Self-review

**Spec coverage check:**
- Persona simulation (166, demographics): Tasks 4, 8, 10, 12.
- 0/1/5-shot baselines: Tasks 5, 10, 12.
- Logit confidence → Brier: Tasks 6, 8, 11, 14.
- Brier composite loss: Tasks 6, 11, 13, 14.
- Empirical human→prob map: Tasks 6, 10 (cache writer), 11, 13, 14.
- Wasserstein + KL on 4-cat: Tasks 7, 14.
- Merged E1/E8: Task 14 (E9 does the direct comparison).
- Limiting-behavior sanity doc: Task 11.
- EG1 replication (E5): Task 13.
- README / tokens doc: Task 15.

**Placeholder scan:** All code blocks contain executable code; all commands have expected output; no "TBD"s.

**Type/name consistency:** `run_condition` signature fixed (Task 8) — all callers in Tasks 9–13 pass `personas=`, `n_shot=`, `n_resamples=`. Checkpoint files use `EG{eg}_shot{k}_rs{rs}` convention in E1/E8; E9 loads them via the glob `E1_baseline_EG{eg}_shot{k}_rs*.json`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-16-metacognition-option-B.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
