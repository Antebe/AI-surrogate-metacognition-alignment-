"""
Quick plotting helpers for sweep results and comparisons.
"""

from __future__ import annotations
from typing import Dict, Sequence

import plotly.graph_objects as go

from .generate import SweepResult


def plot_sweep(
    result: SweepResult,
    title: str | None = None,
) -> go.Figure:
    """
    Show sweep generations as an annotated table figure.
    Each row = one coefficient, columns = samples.
    """
    coeffs = result.coefficients
    max_n = max(len(v) for v in result.generations.values())

    header = ["coeff"] + [f"sample {i+1}" for i in range(max_n)]
    rows = []
    for c in coeffs:
        gens = result.generations[c]
        # Truncate long outputs for readability
        row = [str(c)] + [g[:120] + "…" if len(g) > 120 else g for g in gens]
        # Pad if fewer samples
        row += [""] * (max_n + 1 - len(row))
        rows.append(row)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header, align="left"),
                cells=dict(
                    values=list(zip(*rows)),  # transpose
                    align="left",
                    height=30,
                ),
            )
        ]
    )
    fig.update_layout(
        title=title or f"Coefficient sweep — latent {result.latent_idx}",
        height=120 + 40 * len(coeffs),
    )
    return fig


def print_grid(grid: Dict[str, Dict[str, list[str]]]) -> None:
    """Pretty-print the output of `grid_experiment`."""
    for label, prompts in grid.items():
        print(f"\n{'═'*60}")
        print(f"  {label}")
        print(f"{'═'*60}")
        for prompt, gens in prompts.items():
            print(f"  prompt: {prompt!r}")
            for g in gens:
                print(f"    → {g}")
            print()
