# Option B — Metacognition / SAE-Steering Pipeline Rewrite

**Date:** 2026-04-16
**Author:** Claude (Opus 4.6) + Matthew (proxy for Andre)
**Target venue:** ICML 2026 Mech-Interp workshop (May 9); ICML Phil-ML (May 11 secondary)
**Status:** Approved scope; implementation pending plan

---

## 1. Motivation

The advisor (Hullman) asked for five structural changes to the pipeline before
re-running experiments. This doc specs those changes as Option B of the A/B/C
triage: "pragmatic paper" — full-fidelity on the primary claim, trimmed
ablations. The existing `src/` SAE-steering library is unchanged.

**Locked-in changes (per the email thread):**

1. **166-persona simulation** on the same Gemma-2-9B-IT model, using demographics
   from `Uncertaintydata_long_share.csv`, with ≥ 3 independent resamples for E1
   baselines and ≥ 5 for the final E8 steered run.
2. **Prompt-engineering baselines at 0-shot / 1-shot / 5-shot** on the unsteered
   model. Compare response *distributions* to humans, not just accuracy.
3. **Logit-based confidence** — keep. Already implemented; extract the softmax
   over the "1/2/3/4" tokens for the Brier probability directly (no re-prompt).
4. **Loss rewrite** to
   `L = |Brier_model − Brier_human| + α·|ρ_model − ρ_human| + β·max(0, acc_human − acc_steered)`
   with α = 1, β = 10. Humans' 1–4 confidence → probability by the
   **empirical** per-level accuracy rate on the calibration split.
5. **Distributional comparison** on the four response categories
   (Correct / Lure / Unsure / Other): Wasserstein + KL between model and human
   category distributions.
6. **Merge E1 and E8** into a unified runner (baseline vs. steered, same plots,
   direct overlay charts).

## 2. Non-goals

- Changing `src/` SAE library (hooks, loading, generate).
- New models / new SAEs.
- Political-affiliation persona variable (column not in the share CSV; dropped
  from methods unless Andre surfaces it later).
- Full 5-resample uncertainty on the E2 sweep (kept cheap to save GPU time).
- Full EG1 coverage outside the E5 replication.

## 3. Scope summary

| Phase | Shots | Personas | Resamples | EGs | Items | Sweep |
|---|---|---:|---:|---|---:|---|
| E0 pilot (timing) | 0 | 1 | 1 | 0 | 60 | — |
| E1 baselines | {0,1,5} | 166 | 3 | 0 | 80 | — |
| E2 sweep | 0 | 20 (rotating) | 1 | 0 | 80 | 5×5 = 25 |
| E8 steered @ α\* | {0,1,5} | 166 | 5 | 0 | 80 | — |
| E5 asymmetry (EG1 rep) | 0 | 166 | 3 | 0 & 1 | 80 | base + α\* |
| E3 specificity | 0 | 20 | 1 | 0 | 80 | 3 control latents |
| E6 test-set transfer | 0 | 66 (held-out) | 3 | 0 | 80 | α\* only |
| E7 individual diffs | — | — | — | — | — | reanalyzes E2 sweep |
| E9 E1-vs-E8 analysis | — | — | — | — | — | reanalyzes E1+E8 |

Total Gemma completions: ~440 k (~180 GPU-hrs @ 1.5 s/completion).
Total Claude calls (response-coding fallback): ~132 k ($30 Haiku 4.5).

## 4. Architecture

### 4.1 New modules

```
experiments/
├── config.py              # ← all knobs (existing, to extend)
├── shared.py              # ← existing, extended with imports below
├── persona.py             # NEW  build persona prefix from demographics
├── prompting.py           # NEW  n-shot exemplar selection + prompt assembly
├── brier.py               # NEW  Brier loss, human-prob mapping, composite loss
├── distribution.py        # NEW  Wasserstein/KL on 4-cat + confidence hist
├── run_E0_pilot.py        # NEW  60-item timing probe; writes pilot_timing.json
├── run_E1.py              # REWRITTEN  baselines (0/1/5-shot × personas × resamples)
├── run_E2.py              # REWRITTEN  sweep (persona-aware, 0-shot, Brier loss)
├── run_E3.py              # PATCHED    persona-aware, 0-shot
├── run_E5.py              # PATCHED    EG1 replication at base + α*
├── run_E6.py              # PATCHED    held-out personas
├── run_E7.py              # PATCHED    reads new E1/E2 output schema
├── run_E8.py              # REWRITTEN  α* × 3 shot conditions
├── run_E9.py              # PATCHED    consumes merged E1/E8 summaries
└── run_all.sh             # PATCHED    pilot-gated, logs per-step
```

### 4.2 Config knobs (in `config.py`)

```python
# Persona panel
N_PERSONAS_FULL   = 166
N_PERSONAS_SWEEP  = 20
PERSONA_STYLE     = "prose"   # "prose" | "structured"
PERSONA_SEED      = 42

# Resamples
N_RESAMPLES_E1    = 3
N_RESAMPLES_SWEEP = 1
N_RESAMPLES_E8    = 5

# n-shot
N_SHOT_E1         = [0, 1, 5]
N_SHOT_SWEEP      = [0]
N_SHOT_E8         = [0, 1, 5]

# EGs
EXPOSURE_GROUPS_PRIMARY = [0]
EXPOSURE_GROUPS_E5      = [0, 1]

# Loss
LOSS_ALPHA     = 1.0
LOSS_BETA      = 10.0
HUMAN_CONF_MAP = "empirical"   # "empirical" | "linear"

# Existing knobs kept as-is:
#   SWEEP_UNSURE_VALS, SWEEP_REFUSAL_VALS, TEMPERATURE, MAX_NEW_TOKENS,
#   UNSURE_LATENTS, REFUSAL_LATENTS, CONTROL_LATENTS, RANDOM_SEED.
```

### 4.3 Persona prompting

**`persona.py`**

```python
def load_personas(seed:int, n:int|None) -> pd.DataFrame
    # one row per ID_1 with Age_4, Gender, Education, Ethnicity,
    # UncertaintyCondition, Exposure. Optionally subsample n with seed.

def build_persona_prefix(row: pd.Series, style: str) -> str
    # style="prose":       "You are a 23-year-old female participant..."
    # style="structured":  "Participant demographics: age=23, gender=2, ..."
```

**Key design decisions:**

- `Age_4` column is raw age despite its name (verified: 18–79). Passed literally.
- `Gender`: 1→male, 2→female, 3→non-binary, 5→other. Documented in
  `persona.py` as an assumption, flagged if Andre provides a codebook.
- `Ethnicity`: multi-code strings ("1,3", "2,4") passed through as a
  category-code list in prose mode.
- `Education`: integer level, passed as `"education level N"`.
- Prose mode is the default. Structured mode is a config-flag fallback for
  reviewers who object to coding assumptions.

**Prompt shape (prose mode, 0-shot):**

```
<persona_prefix>

The following fact was recently shared with you:
<exposure_stmt>

Now complete this sentence with ONE word or short phrase:
<stem> is/are
```

1-shot and 5-shot prepend *k* examples drawn from a fixed held-out pool of 10
items (never scored in E1/E8). Example pool is defined in `prompting.py`
and seeded for reproducibility.

### 4.4 Confidence → probability (model side)

Already in `shared.get_confidence()`: returns `logit_details["probs"]` over
`{1,2,3,4}` after softmax, plus a scalar `confidence_score ∈ [0,1]` computed as
`(EV − 1) / 3`. For **Brier**, use that existing scalar directly as the model's
predicted probability of being correct — matches advisor's phrasing ("I
normalized logits that produce a value in [0,1], so I can compute the model
Brier directly").

```
p_correct_model = confidence_score = (EV − 1) / 3     # ∈ [0, 1]
Brier_model_i = (p_correct_model_i − accuracy_i)^2     # per item × resample
Brier_model   = mean over items × resamples × personas
```

The full 4-way probability distribution is retained in JSON for post-hoc
re-analysis (e.g., to swap in a different mapping without re-running Gemma).

### 4.5 Confidence → probability (human side)

**`brier.build_human_prob_map()`** — on the calibration split (100
participants), compute `p_correct_at_level[r]` for `r ∈ {1,2,3,4}` as the
empirical hit-rate of participants reporting confidence `r`. Then
`p_correct_human(response) = p_correct_at_level[response]`.

The map is item-aggregated (per advisor: "from the calibration split, compute
the actual accuracy rate for participants who reported each confidence level").
The map is computed once and cached in `results/brier_human_prob_map.json`.

### 4.6 Loss

**`brier.compute_loss(alpha, beta)` →  `(L, brier_diff, rho_diff, acc_pen)`**

```
L = |Brier_model − Brier_human|
  + alpha · |ρ(conf,acc)_model − ρ(conf,acc)_human|
  + beta  · max(0, acc_human − acc_steered)
```

**Limiting-behavior sanity check** (per advisor) — included in
`brier.limiting_behavior_report()`:

| Term | Best case (value) | Worst case (value) |
|---|---|---|
| `|Brier_m − Brier_h|` | 0 (identical calibration) | ~1 (max disagreement) |
| `α · |ρ_m − ρ_h|` | 0 (matched sensitivity) | 2·α (anti-correlated vs. matched) |
| `β · max(0, acc_h − acc_s)` | 0 (steered ≥ human) | β · acc_h (steered at 0) |

Report is printed on every E2 sweep step and saved to
`results/E2/limiting_behavior.md`.

### 4.7 Distributional metrics

**`distribution.py`** — two functions used by E1, E8, E9, and E5:

```python
def wasserstein_4cat(model_dist: dict, human_dist: dict) -> float
def kl_4cat(model_dist: dict, human_dist: dict,
            eps: float = 1e-6) -> float
```

Input is the normalized 4-category histogram (Correct / Lure / Unsure / Other)
aggregated across items × resamples. Over a categorical ordering (Correct <
Lure < Unsure < Other) Wasserstein reduces to the cumulative-difference
1-Wasserstein; KL is symmetrized (Jensen–Shannon) to avoid the zero-bin blow-up.

### 4.8 Data schema changes

Per-item result now indexed by `(persona_id, shot_count, item_id)`:

```jsonc
// results/E1/E1_baseline_EG0_shot{0,1,5}.json
{
  "<persona_id>": {
    "<item_id>": {
      "prompt": "...",
      "completions": [...],           // length = N_RESAMPLES
      "confidence_probs": [...],      // softmax over 1/2/3/4, per resample
      "confidence_scores": [...],     // expected-value-normalized, [0,1]
      "p_correct_model": [...],       // confidence_scores (same)
      "response_categories": [...],
      "accuracy_scores": [...],
      "mean_confidence": 0.72,
      "mean_accuracy": 0.83
    }
  }
}
```

E2 sweep checkpoint gains a `persona_id` key. Existing E1/E2 results stay
on disk but are **not** reused — schema is incompatible. The old results are
archived to `results/_archive_pre_option_B/`.

## 5. Testing plan

Unit tests live in `tests/` (new dir, `uv run pytest tests/`):

- `tests/test_persona.py` — prefix construction for each style; handles
  missing/multi-code ethnicity; 166 unique prefixes for the full panel.
- `tests/test_brier.py` — loss reproduces toy cases: identical calibration →
  0; opposite → expected value; accuracy-clamp only fires when below human.
- `tests/test_distribution.py` — Wasserstein/JS on canned 4-cat distributions
  match hand-computed values.
- `tests/test_prompting.py` — n-shot exemplar selection excludes any scored
  item; seeded determinism.

**Integration smoke test** (`run_E0_pilot.py`):

- 1 persona, 0-shot, 3 items, unsteered → measures s/completion, writes
  `results/E0/pilot_timing.json`, asserts that `get_confidence()` returns a
  4-way probability distribution summing to 1.

No mocked model in any test.

## 6. Rollout

1. Merge config + `persona.py` + `prompting.py` + `brier.py` + tests
   (compile-clean, tests green, no runs yet).
2. Run `E0_pilot` — update README's "cost per completion" from the measurement.
3. Run `E1` (smallest real run; produces Brier-human, human-prob-map cache).
4. Run `E2` (uses the E1-derived Brier-human baseline).
5. Run `E8` with the emitted `α*`.
6. Run `E5`, `E3`, `E6`, `E7`, `E9`.

Each `run_E*.py` is **resumable from checkpoint** via the existing
`(key → dict)` write-on-every-cell pattern.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Wall-clock blows past May 9 | E0 pilot + estimate before committing sweep; drop 1/5-shot on E8 if tight |
| Reviewer objects to gender/education coding assumptions | Structured mode available as ablation; document in the appendix |
| Brier EV/4 mapping loses info | Document as limitation; keep full 4-way probs in JSON for post-hoc re-analysis |
| HF gated-model auth drops | Retry with existing `HUGGINGFACE_WRITE_KEY` fallback in `config.py` |
| E2 finds α\* at the grid edge | Config-level `refine_sweep()` that adds a 3×3 finer grid around the edge winner; invoked manually if needed |

## 8. Open questions (blocking confirm from Andre)

- **Political affiliation** — Andre listed it as a persona variable in his
  email but it's not in the share CSV. Confirm the column was dropped or
  point to the file that has it.
- **Gender/Education/Ethnicity codebook** — reasonable default assumptions
  documented in §4.3. Confirm acceptable or ship the real codebook.
- **5 resamples vs. ≥ 5** — advisor said "at least 5" for the panel; we spec
  exactly 5 for E8 to stay in budget. Flag if she wants ≥ 10.
