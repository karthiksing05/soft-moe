#!/usr/bin/env python
"""Convergence figures from reports/convergence/conv_results.json.

Three figures (persona panel appears once its block is filled in):
  convergence_curves.png : held-out metric vs training steps, for generic SFT / expert-token SFT /
                           EM Phase-A (backbone) — "how fast does each converge".
  split_sweep.png        : final metric vs Phase-B budget fraction at fixed total steps — best A/B split.
  split_ratios.png       : every Phase A/B split as a stacked-bar budget allocation, annotated with its
                           resulting metric — an at-a-glance view of all split ratios (★ = best).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE                                                # figures live alongside this script
DATA = HERE / "conv_results.json"
COL = {"generic SFT": "#8c8c8c", "expert-token joint SFT": "#2a6fdb", "EM Phase-A (backbone)": "#5aa469"}
A_COL, B_COL = "#5aa469", "#e08a1e"                         # Phase A (backbone) / Phase B (token-only)


def _best_idx(ys, better):
    return (min if better == "down" else max)(range(len(ys)), key=lambda i: ys[i])


def curves_panel(ax, block):
    for name, pts in block["curves"].items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", ms=4, lw=1.8, color=COL.get(name, None), label=name)
    if block.get("ceiling") is not None:
        ax.axhline(block["ceiling"], color="#c0504d", lw=0.9, ls="--")
        ax.text(0.98, block["ceiling"], block.get("ceiling_label", "ceiling"), transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", color="#c0504d", fontsize=7.5)
    ax.set_xlabel("training steps"); ax.set_ylabel(block["metric"])
    arrow = "↑ better" if block["better"] == "up" else "↓ better"
    ax.set_title(f"{block['label']}\nconvergence ({arrow})")
    ax.legend(frameon=False, fontsize=8, loc="best")


def split_panel(ax, block):
    xs, ys = zip(*block["split"])                          # (phaseB_steps, metric)
    T = block["split_total"]
    frac = [100.0 * b / T for b in xs]
    best = _best_idx(ys, block["better"])
    ax.plot(frac, ys, "-o", ms=6, lw=1.8, color="#5aa469")
    ax.plot(frac[best], ys[best], "*", ms=16, color="#d1a000", zorder=5)
    for f, y, b in zip(frac, ys, xs):
        ax.annotate(f"A{T-b}/B{b}", (f, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel(f"Phase-B (token-only) share of the {T}-step budget (%)")
    ax.set_ylabel(block["metric"])
    ax.set_title(f"{block['label']}\nbest Phase A/B split (★)  —  B=0 is all-Phase-A")


def ratios_panel(ax, block):
    """Every split as a stacked A/B budget bar, annotated with its resulting metric."""
    splits = sorted(block["split"], key=lambda p: p[0])    # by Phase-B steps
    T = block["split_total"]
    ys = [m for _, m in splits]
    best = _best_idx(ys, block["better"])
    pos = np.arange(len(splits))
    for i, (b, m) in enumerate(splits):
        A = T - b
        ax.barh(i, A, color=A_COL, edgecolor="white")
        if b > 0:
            ax.barh(i, b, left=A, color=B_COL, edgecolor="white")
        ax.text(A / 2, i, f"A={A}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        if b > 0:
            ax.text(A + b / 2, i, f"B={b}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        star = "  ★ best" if i == best else ""
        ax.text(T * 1.02, i, f"{m:.1f}{'%' if block['better']=='up' else ' ppl'}{star}",
                va="center", fontsize=9, fontweight="bold" if i == best else "normal")
    ax.set_yticks(pos); ax.set_yticklabels([f"{int(100*(1-b/T))}/{int(100*b/T)}" for b, _ in splits])
    ax.set_ylabel("Phase A / Phase B  (% of budget)")
    ax.set_xlabel(f"training steps (total budget = {T})")
    ax.set_xlim(0, T * 1.28)
    ax.set_title(f"{block['label']}\nall Phase A/B split ratios")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=A_COL), plt.Rectangle((0, 0), 1, 1, color=B_COL)],
              labels=["Phase A (backbone, tokens frozen)", "Phase B (token-only, backbone frozen)"],
              frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)


def _grid(blocks, panel, fname, figh=4.4):
    fig, ax = plt.subplots(1, len(blocks), figsize=(6.4 * len(blocks), figh), squeeze=False)
    for i, (_, b) in enumerate(blocks):
        panel(ax[0][i], b)
    fig.tight_layout(); fig.savefig(FIGS / fname, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {fname}")


def main():
    data = json.loads(DATA.read_text())
    blocks = [(k, data[k]) for k in ("knowledge", "persona") if data.get(k)]
    _grid(blocks, curves_panel, "convergence_curves.png")
    _grid(blocks, split_panel, "split_sweep.png")
    _grid(blocks, ratios_panel, "split_ratios.png")


if __name__ == "__main__":
    main()
