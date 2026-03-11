"""
Shared utilities for all experiments.
Data loading, confidence elicitation, response coding, loss computation,
logging, and visualization helpers.
"""
import sys
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# ── Path setup (allows import from src/) ─────────────────────────────────
_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_DIR.parent))

from config import *
from src import *

import anthropic


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(experiment_name: str) -> logging.Logger:
    """Configure dual logging: file + terminal."""
    log_dir = LOGS_DIR / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_statements() -> pd.DataFrame:
    """Load the 80-item statement template (questions + correct answers)."""
    df = pd.read_csv(DATA_DIR / "80 statements.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Item #": "item_num",
        "Fill_in_Statements": "question",
        "Accurate - Statement": "accurate_statement",
        "Inaccurate - Statement": "inaccurate_statement",
    })
    df = df[["item_num", "question", "accurate_statement"]].dropna(subset=["question"])
    df["item_num"] = df["item_num"].astype(int)
    df["correct_answer"] = df["accurate_statement"].apply(
        lambda s: str(s).strip().rstrip(".").split()[-1].lower()
    )
    df["item_id"] = "Item" + df["item_num"].astype(str)
    return df


def load_items(eg: int) -> pd.DataFrame:
    """Load items for a specific exposure group.

    Each EG file is counterbalanced: it contains both accurate and lure
    statements across different item subsets (e.g., items 1-20 accurate
    in EG0 are lure in EG1, and vice versa).
    """
    statements = load_statements()
    eg_path = EG_FILES[eg]
    eg_df = pd.read_csv(eg_path, header=None, names=["item_num", "exposure_statement"])
    eg_df["item_num"] = eg_df["item_num"].astype(int)
    items = statements.merge(eg_df, on="item_num")
    items["exposure_group"] = eg
    return items


def load_human_data() -> pd.DataFrame:
    """Load the full human behavioral dataset."""
    return pd.read_csv(DATA_DIR / "Uncertaintydata_long_share.csv")


def get_participant_split(human_df: pd.DataFrame) -> tuple[list, list]:
    """Reproducible split into calibration and test participants.

    Returns (calib_ids, test_ids). Saves/loads from results/participant_split.json.
    """
    split_path = RESULTS_DIR / "participant_split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        return split["calib"], split["test"]

    rng = np.random.default_rng(RANDOM_SEED)
    all_ids = human_df["ID_1"].unique()
    shuffled = rng.permutation(all_ids).tolist()
    calib = shuffled[:N_CALIB]
    test = shuffled[N_CALIB:]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"calib": calib, "test": test}, f)

    return calib, test


def compute_human_norms(
    human_df: pd.DataFrame,
    participant_ids: list,
    exposure_group: Optional[int] = None,
) -> pd.DataFrame:
    """Compute per-item human confidence and accuracy norms.

    If exposure_group is specified, filter to that exposure condition.
    Returns DataFrame: item_id, human_conf_norm, human_mean_correct, validity.
    """
    subset = human_df[human_df["ID_1"].isin(participant_ids)]
    if exposure_group is not None:
        subset = subset[subset["Exposure"] == exposure_group]

    if len(subset) == 0:
        return pd.DataFrame(columns=["item_id", "human_conf_norm", "human_mean_correct"])

    norms = (
        subset.groupby("Item")
        .agg(
            human_conf_norm=("Confidence", lambda x: (x.mean() - 1) / 3),
            human_mean_correct=("Correct", "mean"),
            validity=("Validity", "first"),
            n_responses=("Confidence", "count"),
        )
        .reset_index()
        .rename(columns={"Item": "item_id"})
    )
    return norms


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def build_prompt(exposure_stmt: str, question: str) -> str:
    """Build the fill-in-the-blank completion prompt."""
    stem = question.replace("____", "").strip().rstrip(".")
    return (
        f"The following fact was recently shared with you:\n{exposure_stmt}\n\n"
        f"Now complete this sentence with ONE word or short phrase:\n{stem} is/are"
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONFIDENCE ELICITATION (Logit-based)
# ═══════════════════════════════════════════════════════════════════════════

def get_client() -> anthropic.Anthropic:
    """Create an Anthropic client."""
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def get_confidence(
    model, prompt: str, completion: str, client: anthropic.Anthropic = None
) -> tuple[float, dict]:
    """Logit-based confidence elicitation from Gemma.

    Prompts Gemma with the completion and a 1-4 confidence scale, then
    extracts logits over tokens ["1","2","3","4"] from the next-token
    distribution. Returns (normalized_score [0,1], logit_details_dict).

    No Claude involved — confidence comes directly from the model's
    internal state. The client param is kept for API compatibility but unused.
    """
    import torch

    # Clean completion text: take first line only, strip formatting junk
    answer_text = completion.strip().split("\n")[0].strip()

    conf_prompt = (
        f"{prompt} {answer_text}\n\n"
        f"Your answer was: {answer_text}\n"
        f"On a scale of 1 to 4, how confident are you that your answer "
        f"is correct?\n"
        f"1 = not at all confident\n"
        f"2 = somewhat not confident\n"
        f"3 = somewhat confident\n"
        f"4 = very confident\n\n"
        f"Reply with ONLY the single digit 1, 2, 3, or 4.\n"
    )

    # Forward pass — get logits at the last token position
    tokens = model.to_tokens(conf_prompt)
    with torch.no_grad():
        logits = model(tokens)  # (batch, seq, vocab)
    last_logits = logits[0, -1, :]  # (vocab,)

    # Get token IDs for "1", "2", "3", "4"
    scale_tokens = {}
    for digit in ["1", "2", "3", "4"]:
        tids = model.to_tokens(digit, prepend_bos=False).squeeze().tolist()
        # Handle both scalar and list returns
        tid = tids if isinstance(tids, int) else tids[0]
        scale_tokens[digit] = tid

    # Extract logits for the 4 scale tokens and softmax
    digit_logits = torch.tensor([last_logits[scale_tokens[d]] for d in ["1", "2", "3", "4"]])
    probs = torch.softmax(digit_logits, dim=0).cpu().numpy()

    # Expected value: sum(prob_i * i) for i in 1..4, then normalize to [0,1]
    expected_val = float(np.sum(probs * np.array([1, 2, 3, 4])))
    normalized = (expected_val - 1) / 3  # map [1,4] -> [0,1]

    logit_details = {
        "probs": {str(i + 1): float(probs[i]) for i in range(4)},
        "expected_value": expected_val,
    }

    return normalized, logit_details


def code_response(
    exposure_stmt: str, question: str, correct_answer: str,
    completion: str, client: anthropic.Anthropic,
) -> tuple[str, int]:
    """Claude codes response as Correct/Lure/Unsure/Other (researcher eval analog).

    First tries a fast lowercase substring match for Correct/Lure.
    Falls back to Claude only when the match is ambiguous.
    """
    clean = completion.strip().lower()
    correct_lower = correct_answer.strip().lower()

    # Extract lure answer from exposure statement (last word)
    lure_answer = exposure_stmt.strip().rstrip(".").split()[-1].lower()

    # Fast deterministic check: does the completion contain the answer word?
    has_correct = correct_lower in clean
    has_lure = lure_answer in clean and lure_answer != correct_lower

    if has_correct and not has_lure:
        return "Correct", 1
    if has_lure and not has_correct:
        return "Lure", 0

    # Ambiguous or neither — fall back to Claude
    prompt = (
        f'A model was shown: "{exposure_stmt}"\n'
        f'Then asked to complete: "{question}"\n'
        f'Raw model output: "{completion.strip()}"\n'
        f'Correct answer: "{correct_answer}"\n'
        f'Lure (false) answer: "{lure_answer}"\n\n'
        f"Ignore formatting artifacts (underscores, markdown, extra text). "
        f"Focus on whether the actual answer word matches the correct or lure answer.\n\n"
        f"Classify as exactly one of: Correct, Lure, Unsure, Other.\n"
        f"Reply with ONLY one word."
    )
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    cat = msg.content[0].text.strip().capitalize()
    if cat not in ("Correct", "Lure", "Unsure", "Other"):
        cat = "Other"
    return cat, int(cat == "Correct")


# ═══════════════════════════════════════════════════════════════════════════
# LOSS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(
    model_confs: list[float],
    human_confs: list[float],
    model_accs: list[float],
    l1: float = LAMBDA1,
    l2: float = LAMBDA2,
) -> tuple[float, float, float]:
    """L(alpha) = l1 * ECE + l2 * (1 - corr(confidence, accuracy)).

    Returns (loss, ece, corr).
    """
    mc = np.array(model_confs)
    hc = np.array(human_confs)
    ma = np.array(model_accs)
    ece = float(np.mean(np.abs(mc - hc)))
    if np.std(mc) > 1e-8 and np.std(ma) > 1e-8:
        corr = float(np.corrcoef(mc, ma)[0, 1])
    else:
        corr = 0.0
    loss = l1 * ece + l2 * (1 - corr)
    return loss, ece, corr


# ═══════════════════════════════════════════════════════════════════════════
# INTERVENTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def make_target_interventions(
    saes: dict, unsure_coeff: float, refusal_coeff: float
) -> list:
    """Create steering interventions for the target latents."""
    ivs = []
    for layer, idx in UNSURE_LATENTS:
        ivs.append(steer(saes[layer], idx, coeff=unsure_coeff, label=f"unsure_L{layer}"))
    for layer, idx in REFUSAL_LATENTS:
        ivs.append(steer(saes[layer], idx, coeff=refusal_coeff, label=f"refusal_L{layer}"))
    return ivs


def make_control_interventions(saes: dict, name: str, coeff: float) -> list:
    """Create a single control latent intervention."""
    layer, idx = CONTROL_LATENTS[name]
    return [steer(saes[layer], idx, coeff=coeff, label=name)]


# ═══════════════════════════════════════════════════════════════════════════
# RUN CONDITION
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(
    model,
    items: pd.DataFrame,
    interventions: list | None,
    client: anthropic.Anthropic,
    n_samples: int = N_SAMPLES,
    logger: Optional[logging.Logger] = None,
    checkpoint_path: Optional[Path] = None,
    existing_results: Optional[dict] = None,
) -> dict:
    """Run all items under a single condition. Returns per-item results dict.

    Supports resuming from existing_results and checkpointing.
    """
    results = dict(existing_results) if existing_results else {}

    for _, row in tqdm(items.iterrows(), total=len(items), desc="Items"):
        iid = row["item_id"]
        if iid in results:
            continue

        prompt = build_prompt(row["exposure_statement"], row["question"])

        raw_outputs = generate(
            model, prompt, interventions=interventions,
            max_new_tokens=MAX_NEW_TOKENS, n=n_samples, temperature=TEMPERATURE,
        )
        # Strip prompt prefix — model.generate() returns prompt + completion
        completions = [
            out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
            for out in raw_outputs
        ]
        conf_data = [get_confidence(model, prompt, c, client) for c in completions]
        conf_scores = [cd[0] for cd in conf_data]
        conf_logit_details = [cd[1] for cd in conf_data]
        coded = [
            code_response(
                row["exposure_statement"], row["question"],
                row["correct_answer"], c, client,
            )
            for c in completions
        ]
        categories = [cat for cat, _ in coded]
        acc_scores = [acc for _, acc in coded]

        results[iid] = {
            "prompt": prompt,
            "completions": completions,
            "confidence_logits": conf_logit_details,
            "confidence_scores": conf_scores,
            "response_categories": categories,
            "accuracy_scores": acc_scores,
            "mean_confidence": float(np.mean(conf_scores)),
            "mean_accuracy": float(np.mean(acc_scores)),
        }

        if logger:
            logger.info(
                f"  {iid}: conf={np.mean(conf_scores):.2f} "
                f"acc={np.mean(acc_scores):.2f} cats={categories}"
            )

        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=2)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

EG_COLORS      = {0: "#4C72B0", 1: "#DD8452"}
BASELINE_COLOR = "#4C72B0"
OPTIMAL_COLOR  = "#C44E52"
CONTROL_COLOR  = "#8DA0CB"
HUMAN_COLOR    = "#55A868"


def setup_viz():
    """Set consistent visualization defaults."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
    })


def save_plot(fig: plt.Figure, plot_dir: Path, name: str):
    """Save figure to plot directory as PNG."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def align_model_human(
    model_results: dict, human_norms: pd.DataFrame
) -> pd.DataFrame:
    """Merge model per-item results with human norms into a single DataFrame."""
    model_df = pd.DataFrame([
        {
            "item_id": k,
            "model_conf": v["mean_confidence"],
            "model_acc": v["mean_accuracy"],
        }
        for k, v in model_results.items()
    ])
    merged = model_df.merge(human_norms, on="item_id", how="inner")
    return merged


def collect_response_categories(results: dict) -> pd.DataFrame:
    """Flatten per-item response categories into a long-form DataFrame."""
    rows = []
    for iid, v in results.items():
        for cat in v["response_categories"]:
            rows.append({"item_id": iid, "category": cat})
    return pd.DataFrame(rows)
