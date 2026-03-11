"""
Loading the HookedSAETransformer and SAE weights.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

from sae_lens import HookedSAETransformer, SAE

from .config import DEVICE, DEFAULT_MODEL_NAME, DEFAULT_SAE_RELEASE, DEFAULT_SAE_WIDTH


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEVICE,
    hf_token: Optional[str] = None,
) -> HookedSAETransformer:
    """Load a HookedSAETransformer model (downloads on first call)."""
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    model = HookedSAETransformer.from_pretrained(model_name).to(device)
    return model


def load_sae(
    layer: int,
    release: str = DEFAULT_SAE_RELEASE,
    width: str = DEFAULT_SAE_WIDTH,
    device: str = DEVICE,
) -> SAE:
    """Load a single SAE for a given layer. Returns the SAE object."""
    sae_id = f"layer_{layer}/{width}"
    sae, _cfg, _sparsity = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device
    )
    return sae


def load_saes(
    layers: list[int],
    release: str = DEFAULT_SAE_RELEASE,
    width: str = DEFAULT_SAE_WIDTH,
    device: str = DEVICE,
) -> Dict[int, SAE]:
    """Load SAEs for multiple layers. Returns {layer: SAE}."""
    return {layer: load_sae(layer, release, width, device) for layer in layers}
