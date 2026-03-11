"""
Central configuration for all SAE steering experiments.
Change settings here, not in individual scripts.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
LOGS_DIR     = PROJECT_ROOT / "logs"

# ── API Keys ─────────────────────────────────────────────────────────────
HF_TOKEN      = ""
ANTHROPIC_KEY  = ""

CLAUDE_MODEL   = "claude-haiku-4-5-20251001"

# ── SAE Layers ───────────────────────────────────────────────────────────
SAE_LAYERS = [9, 20, 31]

# ── Target Latents (from pilot experiments in main.ipynb) ────────────────
# Format: (layer, latent_idx)
UNSURE_LATENTS  = [(20, 11868), (9, 11212)]
REFUSAL_LATENTS = [(9, 16203), (31, 12190)]

# ── Control Latents (for E3 ablation) ────────────────────────────────────
CONTROL_LATENTS = {
    "ctrl_topic_A":  (31, 3000),
    "ctrl_random_B": (20, 7777),
    "ctrl_random_C": (9, 9999),
}

# ── Experiment Parameters ────────────────────────────────────────────────
N_SAMPLES    = 6       # completions per item (E1, E5, E6)
E2_N_SAMPLES = 3       # reduced for sweep tractability
E3_N_SAMPLES = 3
TEMPERATURE  = 0.5
MAX_NEW_TOKENS = 15     # max tokens per Gemma completion (answers should be 1-3 words)
LAMBDA1      = 0.5     # ECE weight in joint loss
LAMBDA2      = 0.5     # (1 - corr) weight in joint loss
N_CALIB      = 100     # calibration participants
RANDOM_SEED  = 42

# ── E2 Sweep Grid ───────────────────────────────────────────────────────
SWEEP_UNSURE_VALS  = [0, 20, 40, 60, 120]
SWEEP_REFUSAL_VALS = [0, 20, 40, 60, 120]

# ── Exposure Groups ─────────────────────────────────────────────────────
EXPOSURE_GROUPS = [0] #0, 1 is too much  # Counterbalanced: each EG has a mix of accurate & lure items
EG_LABELS = {0: "Exposure Group 0", 1: "Exposure Group 1"}
EG_FILES  = {
    0: DATA_DIR / "80 statements_EG0.csv",
    1: DATA_DIR / "80 statements_EG1.csv",
}
