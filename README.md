# SAE Steering for Simulating Human Metacognition

## Overview

This project investigates whether adjusting monosemantic metacognitive neurons — identified via Sparse Autoencoder (SAE) latent decomposition — can improve large language model simulations of human behavioral data. We simulate human memory retrieval on an open-ended general knowledge task after exposure to true and false information, using **Gemma-2-9B-IT** with **GemmaScope SAEs**.

The primary contribution is to improve the theoretical tools by which cognitive scientists might hypothesize and test human cognition. Metacognitive sensitivity — the ability to accurately monitor one's own knowledge — is essential to mechanistic claims about how and why humans reproduce inaccurate information. Improving confidence simulation *before* experimental validation would improve the efficiency of theory-testing.

## Hypothesis

We predict that the **unsteered** Gemma model will poorly simulate human metacognition:

1. **Overconfidence**: The model will exhibit general overconfidence on knowledge judgments (consistently rating "completely confident") relative to human data.
2. **Inflated performance**: The model will produce greater task accuracy than expected of humans, with reduced frequency of Unsure/Other responses.
3. **Poor metacognitive sensitivity**: With confidence at ceiling, the unsteered model will show weaker confidence–accuracy correlation than observed in humans.

We hypothesize that **steering** pseudo-monosemantic neurons related to uncertainty and hesitation — adjusted to reduce model confidence — will bring simulated task performance and confidence closer to human behavioral data, thereby improving metacognitive sensitivity.

## Explanatory Mechanism

We steer Gemma using SAE-identified latent directions for *uncertainty* and *refusal*, amplifying these features to lower the model's expressed confidence. This targets the model's internal representations rather than prompt engineering, allowing fine-grained control over metacognitive behavior. The steering coefficients are optimized against human calibration data to minimize a joint loss combining Expected Calibration Error (ECE) and confidence–accuracy decorrelation.

## Human Behavioral Data

Human data comes from **Agnoli & Rapp (in prep)**: *n* = 166 participants completed 80 fill-in-the-blank general knowledge items. Participants were assigned to one of two **counterbalanced exposure groups** (EG0 or EG1). Each group saw a mix of accurate and inaccurate (lure) versions of statements — items that are accurate in EG0 are lure in EG1, and vice versa — so that every item appears in both conditions across groups. Participants rated their confidence on a 1–4 scale for each response. This design enables studying the **misinformation effect** — the tendency to reproduce false information with inappropriate confidence.

## Prior Work

- **Salovich, N. A., & Rapp, D. N. (2021)**. Misinformed and unaware? Metacognition and the influence of inaccurate information. *Journal of Experimental Psychology: Learning, Memory, and Cognition*.
- **Geers, A. L., et al. (2024)**. Metacognitive confidence and susceptibility to misinformation.
- **Salovich, N. A., & Rapp, D. N. (2023)**. Behavioral simulation approaches for testing cognitive theories.
- **Templeton, A., et al. (2024)**. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Anthropic Research*.
- **Bricken, T., et al. (2023)**. Towards monosemanticity: Decomposing language models with dictionary learning. *Anthropic Research*.

## Experiment Sequence

### E1 — Baseline: Unsteered Model Performance

Runs unsteered Gemma on all 80 items for both exposure groups. Each EG file contains a counterbalanced mix of accurate and lure statements (see Data section). Uses logit-based confidence elicitation: Gemma generates a completion, then rates confidence via next-token logits over a 1–4 scale. Claude codes each response as Correct/Lure/Unsure/Other, analogous to researcher coding of human data.

### E2 — Core Steering: Optimal Coefficient Search

Steers 4 target latents — 2 "unsure" and 2 "refusal" features identified via pilot experiments. Grid-searches over `alpha = (unsure_coeff, refusal_coeff)` with 36 configurations, evaluated on 100 calibration participants across both exposure groups. Finds the `alpha*` that minimizes an accuracy-adjusted joint loss `L = [0.5 * ECE + 0.5 * (1 - corr)] * (acc_baseline / acc)`, where `acc_baseline` is the unsteered (0,0) model accuracy. This penalizes accuracy drops: if steering reduces accuracy, the ratio > 1 inflates the loss.

### E3 — Ablation: Specificity

Tests whether the improvement is specific to the metacognitive target latents or a generic steering artifact. Runs 10 semantically unrelated control latents (topic, syntactic, random features) at the same optimal coefficients. If target latents outperform most controls, the effect is specific to uncertainty/refusal features.

### E4 -- Ablation: confidence elicitation

We rate confidence via next-token logits over a 1–4 scale, find expected value, and then normalize. But we can also use other techniques: another-LLM-as-a-judge or explicit self-eval (CoT). Logits was the most grounded method.  

### E5 — Item-Type Asymmetry

Stratifies results by Validity (whether a given item was presented accurately or as a lure within its EG file) and Exposure Group (EG0 vs. EG1). Tests whether steering disproportionately improves calibration on lure-exposure items, where humans show inflated confidence — the core misinformation effect. Examines the Exposure x Validity interaction.

### E6 — Generalization to Held-Out Participants

Tests whether `alpha*` (optimized on 100 calibration participants) transfers to the 66 held-out test participants without retuning. Computes a generalization ratio: `DLOSS(test) / DLOSS(calib)` — values near 1.0 indicate full transfer.

### E7 — Individual Differences: Participant Coverage

Treats each participant's per-item confidence as an 80-dimensional profile. Maps model profiles at each alpha alongside human profiles in PCA space. Measures what fraction of individual participants are "covered" by some alpha in the sweep, and computes Wasserstein distances between model and human profile distributions. Tests whether the alpha sweep traces a meaningful path through human metacognitive variance.

## Technical Details

| Component | Detail |
|-----------|--------|
| **Model** | Gemma-2-9B-IT via TransformerLens |
| **SAEs** | GemmaScope (layers 9, 20, 31; width 16k) |
| **Target latents** | sae_20:11868, sae_9:11212 (unsure); sae_9:16203, sae_31:12190 (refusal) |
| **Confidence elicitation** | Two-turn self-report (Gemma expresses → Claude scores 1–4) |
| **Response coding** | Claude Sonnet classifies as Correct/Lure/Unsure/Other |
| **Temperature** | 0.5 |
| **Loss function** | `L(alpha) = [0.5 * ECE + 0.5 * (1 - correlation)] * (acc_baseline / acc_alpha)` |
| **Human data** | *n* = 166 (100 calib / 66 test), 80 items, 2 exposure groups |

## Project Structure

```
sae_steering_toolkit/
├── README.md
├── requirements.txt
├── src/                        # Core SAE steering library
│   ├── config.py               #   Device/model defaults
│   ├── loading.py              #   Model & SAE loading
│   ├── hooks.py                #   Steering/ablation interventions
│   ├── generate.py             #   Text generation & sweeps
│   └── viz.py                  #   Plotly visualization helpers
├── experiments/                # Experiment scripts (tmux-friendly)
│   ├── config.py               #   Central experiment configuration
│   ├── shared.py               #   Shared utilities (data, logging, viz)
│   ├── run_E1.py               #   Baseline
│   ├── run_E2.py               #   Core steering
│   ├── run_E3.py               #   Ablation specificity
│   ├── run_E5.py               #   Item-type asymmetry
│   ├── run_E6.py               #   Generalization
│   ├── run_E7.py               #   Individual differences
│   └── run_all.sh              #   Sequential runner
├── data/
│   ├── 80 statements.csv       #   Item templates (80 trivia items)
│   ├── 80 statements_EG0.csv   #   Exposure group 0 (counterbalanced: mix of accurate & lure)
│   ├── 80 statements_EG1.csv   #   Exposure group 1 (counterbalanced: complement of EG0)
│   └── Uncertaintydata_long_share.csv  # Human behavioral data (n=166)
├── results/                    #   Per-experiment results & plots
│   ├── E1/
│   ├── E2/
│   └── ...
└── logs/                       #   Per-experiment log files
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments sequentially
cd experiments
bash run_all.sh

# Or run individual experiments
python run_E1.py
python run_E2.py
# ...

# tmux session
tmux new -s experiments 'cd experiments && bash run_all.sh'
```

## Configuration

All experiment parameters are centralized in `experiments/config.py`:
- API keys (HuggingFace, Anthropic)
- Model and SAE layer settings
- Target and control latent indices
- Sweep grid values
- Sample sizes, temperature, loss weights
- Exposure group definitions

Change settings there — individual experiment scripts read from config.

## SAE Steering Library (src/)

The core steering toolkit is adapted from the [Northwestern AI Safety Club steering demo](https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/Hooked_SAE_Transformer_Demo.ipynb).

```python
from src import *

model = load_model(hf_token="hf_...")
sae_20 = load_sae(layer=20)

# Steer, ablate, clamp, sweep
result = compare(model, "Hello", [steer(sae_20, 12082, coeff=240)])
sweep = sweep_coefficients(model, "Hello", [steer(sae_20, 12082, coeff=0)],
                           coefficients=[0, 50, 100, 200, 400])
```
