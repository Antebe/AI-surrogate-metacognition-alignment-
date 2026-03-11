"""
SAE Steering Toolkit
====================
Modular helpers for steering & ablating SAE latents in TransformerLens models.

Quick start::

    from src import *

    model = load_model(hf_token="hf_...")
    sae_20 = load_sae(layer=20)

    result = compare(
        model, "When I look in the mirror, I see",
        interventions=[steer(sae_20, latent_idx=12082, coeff=240)],
    )
    print(result)
"""

from .config import DEVICE, DEFAULT_MODEL_NAME, DEFAULT_SAE_RELEASE
from .loading import load_model, load_sae, load_saes
from .hooks import (
    Intervention,
    make_hooks,
    steer,
    ablate,
    clamp,
)
from .generate import (
    generate,
    compare,
    sweep_coefficients,
    grid_experiment,
    CompareResult,
    SweepResult,
)
from .viz import plot_sweep, print_grid
