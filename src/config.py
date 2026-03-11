"""
Global configuration: device selection, default model/SAE settings.
"""

import torch


def get_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    # elif torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


# ── Defaults (override in your notebook / script) ────────────────────────
# DEFAULT_MODEL_NAME = "google/gemma-2-2b"
# DEFAULT_SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
# DEFAULT_SAE_WIDTH = "width_16k/canonical"

DEFAULT_MODEL_NAME = "gemma-2-9b-it"
DEFAULT_SAE_RELEASE = "gemma-scope-9b-it-res-canonical"
DEFAULT_SAE_WIDTH = "width_16k/canonical"



DEVICE = get_device()
torch.set_grad_enabled(False)
