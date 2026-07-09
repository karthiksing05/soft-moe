#!/usr/bin/env python
"""2-D PCA of the token space: where do the learned expert vectors land vs vocab & their content centroids."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
TASKS = [("persona", "Persona / style"), ("domain", "Domain-QA")]


def panel(ax, d, title):
    V = np.array(d["proj_vocab"])
    ax.scatter(V[:, 0], V[:, 1], s=4, color="#cccccc", alpha=0.5, label="vocab (sample)")
    cmap = plt.get_cmap("tab10")
    for k, (x, y, name) in enumerate(d["proj_expert"]):
        cx, cy, _ = d["proj_centroid"][k]
        c = cmap(k % 10)
        ax.plot([x, cx], [y, cy], color=c, lw=0.7, alpha=0.5)
        ax.scatter([x], [y], s=140, color=c, marker="*", edgecolor="k", lw=0.4, zorder=5)
        ax.scatter([cx], [cy], s=45, color=c, marker="o", edgecolor="k", lw=0.4, zorder=4)
        ax.annotate(name, (x, y), fontsize=7.5, fontweight="bold", ha="center", va="bottom")
    ax.set_xticks([]); ax.set_yticks([])
    nm = d["norms"]
    ax.set_title(f"{title}\n★ expert vector   ● its content centroid   ·  grey = vocabulary\n"
                 f"(expert norms ≫ vocab: p50 {nm['vocab_p50']:.1f}, p99 {nm['vocab_p99']:.1f})", fontsize=9)


def main():
    blocks = [(t, json.loads((HERE.parent / "figs" / f"{t}_tokspace.json").read_text()), lab)
              for t, lab in TASKS if (HERE.parent / "figs" / f"{t}_tokspace.json").exists()]
    if not blocks:
        print("no tokspace JSONs yet"); return
    fig, ax = plt.subplots(1, len(blocks), figsize=(7 * len(blocks), 6))
    if len(blocks) == 1:
        ax = [ax]
    for a, (_, d, lab) in zip(ax, blocks):
        panel(a, d, lab)
    fig.suptitle("Learned expert vectors in the token-embedding space (2-D PCA)", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "token_space.png", dpi=140, bbox_inches="tight")
    print("wrote token_space.png")


if __name__ == "__main__":
    main()
