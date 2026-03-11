"""
Generation helpers: run prompts with and without interventions,
sweep over coefficients, compare multiple intervention sets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from sae_lens import HookedSAETransformer

from .hooks import HookList, Intervention, make_hooks


# ── Single generation ────────────────────────────────────────────────────

def generate(
    model: HookedSAETransformer,
    prompt: str,
    interventions: Sequence[Intervention] | None = None,
    max_new_tokens: int = 40,
    n: int = 1,
    temperature: float = 0.5,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    verbose: bool = False,
) -> list[str]:
    """Generate *n* completions, optionally with interventions."""
    kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature, verbose=verbose)
    if top_k is not None:
        kwargs["top_k"] = top_k
    if top_p is not None:
        kwargs["top_p"] = top_p

    outputs: list[str] = []
    if interventions:
        hooks = make_hooks(interventions)
        with model.hooks(fwd_hooks=hooks):
            for _ in range(n):
                outputs.append(model.generate(prompt, **kwargs))
    else:
        for _ in range(n):
            outputs.append(model.generate(prompt, **kwargs))
    return outputs


# ── Compare baseline vs. steered ─────────────────────────────────────────

@dataclass
class CompareResult:
    prompt: str
    baseline: list[str]
    steered: list[str]
    label: str = ""

    def __repr__(self):
        sep = "\n  "
        base = sep.join(self.baseline)
        steer = sep.join(self.steered)
        return (
            f"═══ {self.label or 'Comparison'} ═══\n"
            f"Prompt: {self.prompt!r}\n"
            f"── Baseline ──\n  {base}\n"
            f"── Steered ──\n  {steer}\n"
        )


def compare(
    model: HookedSAETransformer,
    prompt: str,
    interventions: Sequence[Intervention],
    n_baseline: int = 1,
    n_steered: int = 5,
    label: str = "",
    verbose: bool = False,
    **gen_kwargs,
) -> CompareResult:
    """Generate baseline and steered completions side by side."""
    baseline = generate(model, prompt, n=n_baseline, verbose=verbose, **gen_kwargs)
    steered = generate(model, prompt, interventions=interventions, n=n_steered, verbose=verbose, **gen_kwargs)
    return CompareResult(prompt=prompt, baseline=baseline, steered=steered, label=label)


# ── Coefficient sweep ────────────────────────────────────────────────────

@dataclass
class SweepResult:
    prompt: str
    latent_idx: int
    coefficients: list[float]
    generations: Dict[float, list[str]]  # coeff → list of outputs

    def __repr__(self):
        lines = [f"═══ Sweep latent {self.latent_idx} | prompt: {self.prompt!r} ═══"]
        for c, gens in self.generations.items():
            lines.append(f"  coeff={c}")
            for g in gens:
                lines.append(f"    {g}")
        return "\n".join(lines)


def sweep_coefficients(
    model: HookedSAETransformer,
    prompt: str,
    base_interventions: Sequence[Intervention],
    sweep_index: int = 0,
    coefficients: Sequence[float] = (0, 25, 50, 100, 200, 400),
    n: int = 3,
    verbose: bool = False,
    **gen_kwargs,
) -> SweepResult:
    """
    Take a list of interventions, pick the one at *sweep_index*, and sweep
    its coefficient through *coefficients*, generating *n* samples each.

    All other interventions are kept fixed.
    """
    target = base_interventions[sweep_index]
    results: Dict[float, list[str]] = {}

    for c in coefficients:
        # Rebuild the intervention list with the new coeff
        modified = []
        for i, iv in enumerate(base_interventions):
            if i == sweep_index:
                modified.append(Intervention(
                    sae=iv.sae,
                    latent_idx=iv.latent_idx,
                    coeff=c,
                    mode=iv.mode,
                    label=iv.label,
                ))
            else:
                modified.append(iv)
        results[c] = generate(model, prompt, interventions=modified, n=n, verbose=verbose, **gen_kwargs)

    return SweepResult(
        prompt=prompt,
        latent_idx=target.latent_idx,
        coefficients=list(coefficients),
        generations=results,
    )


# ── Multi-intervention grid ─────────────────────────────────────────────

def grid_experiment(
    model: HookedSAETransformer,
    prompts: Sequence[str],
    intervention_sets: Dict[str, Sequence[Intervention]],
    n: int = 3,
    verbose: bool = False,
    **gen_kwargs,
) -> Dict[str, Dict[str, list[str]]]:
    """
    Run a grid of (prompt × intervention_set).

    Returns  {set_label: {prompt: [completions]}}
    """
    results: Dict[str, Dict[str, list[str]]] = {}
    for label, ivs in intervention_sets.items():
        results[label] = {}
        for prompt in prompts:
            results[label][prompt] = generate(
                model, prompt, interventions=ivs, n=n, verbose=verbose, **gen_kwargs
            )
    return results
