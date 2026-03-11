"""
Steering & ablation hooks for SAE latents.

Key concepts
------------
- **steer**: add a scaled decoder direction to the residual stream.
- **ablate / clamp**: force a specific SAE latent activation to a fixed value
  (0 = ablate, any other value = clamp).

All public helpers return a list of `(hook_name, hook_fn)` tuples that you can
concatenate and pass straight into `model.hooks(fwd_hooks=...)`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE


# ── Low-level hook functions ─────────────────────────────────────────────

def _steer_hook(
    activations: Tensor,
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    coeff: float,
) -> Tensor:
    """Add `coeff * sae.W_dec[latent_idx]` to every sequence position."""
    return activations + coeff * sae.W_dec[latent_idx]


def _steer_multi_hook(
    activations: Tensor,
    hook: HookPoint,
    sae: SAE,
    latent_coeffs: dict[int, float],
) -> Tensor:
    """Steer multiple latents at once (single SAE / single hook point)."""
    delta = torch.zeros_like(activations[0, 0])  # (d_model,)
    for idx, c in latent_coeffs.items():
        delta = delta + c * sae.W_dec[idx]
    return activations + delta


def _ablate_hook(
    activations: Tensor,
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    clamp_value: float = 0.0,
) -> Tensor:
    """
    Zero-ablate (or clamp) a single SAE latent.

    This encodes → sets the latent → decodes, replacing the original
    activation with the modified reconstruction.
    """
    # Encode
    z = sae.encode(activations)               # (..., n_latents)
    z[..., latent_idx] = clamp_value           # clamp / ablate
    # Decode back
    return sae.decode(z)


def _get_hook_name(sae: SAE) -> str:
    """Get hook name from SAE configuration, handling different SAE versions."""
    if hasattr(sae.cfg, 'hook_name'):
        return sae.cfg.hook_name
    elif hasattr(sae.cfg, 'metadata') and 'hook_name' in sae.cfg.metadata:
        return sae.cfg.metadata['hook_name']
    else:
        raise AttributeError("Cannot find hook_name in SAE configuration")


# ── Dataclass for readable experiment specs ──────────────────────────────

@dataclass
class Intervention:
    """One atomic steering / ablation intervention."""
    sae: SAE
    latent_idx: int
    coeff: float = 0.0
    mode: Literal["steer", "ablate"] = "steer"
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"L{self._guess_layer()}_{self.mode}_{self.latent_idx}_c{self.coeff}"

    def _guess_layer(self) -> str:
        # e.g. hook_name = "blocks.20.hook_resid_post"
        try:
            hook_name = _get_hook_name(self.sae)
        except AttributeError:
            return "?"
        
        parts = hook_name.split(".")
        for p in parts:
            if p.isdigit():
                return p
        return "?"


# ── Public API ───────────────────────────────────────────────────────────

HookList = List[Tuple[str, partial]]


def make_hooks(interventions: Sequence[Intervention]) -> HookList:
    """
    Turn a list of :class:`Intervention` specs into TransformerLens hooks.

    Interventions that share the same SAE *and* are both `steer` mode are
    batched into a single hook call for efficiency.
    """
    hooks: HookList = []

    # Group steer interventions by hook_name
    steer_groups: dict[str, dict[int, float]] = {}
    steer_saes: dict[str, SAE] = {}

    for iv in interventions:
        if iv.mode == "steer":
            hname = _get_hook_name(iv.sae)
            steer_groups.setdefault(hname, {})[iv.latent_idx] = iv.coeff
            steer_saes[hname] = iv.sae
        elif iv.mode == "ablate":
            hook_fn = partial(
                _ablate_hook,
                sae=iv.sae,
                latent_idx=iv.latent_idx,
                clamp_value=iv.coeff,  # 0 → ablate, nonzero → clamp
            )
            hooks.append((_get_hook_name(iv.sae), hook_fn))

    # Emit one batched steer hook per hook point
    for hname, lc in steer_groups.items():
        hook_fn = partial(
            _steer_multi_hook,
            sae=steer_saes[hname],
            latent_coeffs=lc,
        )
        hooks.append((hname, hook_fn))

    return hooks


# ── Convenience shortcuts ────────────────────────────────────────────────

def steer(sae: SAE, latent_idx: int, coeff: float, label: str = "") -> Intervention:
    """Shortcut to create a steer intervention."""
    return Intervention(sae=sae, latent_idx=latent_idx, coeff=coeff, mode="steer", label=label)


def ablate(sae: SAE, latent_idx: int, label: str = "") -> Intervention:
    """Shortcut to create an ablation (zero-clamp) intervention."""
    return Intervention(sae=sae, latent_idx=latent_idx, coeff=0.0, mode="ablate", label=label)


def clamp(sae: SAE, latent_idx: int, value: float, label: str = "") -> Intervention:
    """Shortcut to clamp a latent to a fixed nonzero value."""
    return Intervention(sae=sae, latent_idx=latent_idx, coeff=value, mode="ablate", label=label)
