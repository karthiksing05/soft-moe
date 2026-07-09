#!/usr/bin/env python
"""Linear-separability figure — 2-D LDA projection of the persona-induced representation shift, per scheme."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SEP = HERE.parent / "figs" / "sep"
ARMS = [("sft", "SFT (joint)"), ("frozen", "frozen-random"), ("em", "EM (alternation)")]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False, sharey=False)
for i, (key, label) in enumerate(ARMS):
    d = json.loads((SEP / f"{key}_sep.json").read_text())
    P = np.array([[p[0], p[1]] for p in d["proj2d"]]); y = np.array([p[2] for p in d["proj2d"]])
    names = d["names"]; K = len(names)
    cmap = plt.get_cmap("tab10")
    for k in range(K):
        m = y == k
        ax[i].scatter(P[m, 0], P[m, 1], s=22, color=cmap(k % 10), label=names[k], alpha=0.8)
        cx, cy = P[m, 0].mean(), P[m, 1].mean()
        ax[i].annotate(names[k], (cx, cy), fontsize=7.5, fontweight="bold", ha="center", va="center")
    ax[i].set_xticks([]); ax[i].set_yticks([])
    ax[i].set_title(f"{label}\nprobe acc {d['probe_acc']:.0%} (chance {d['chance']:.0%})  ·  "
                    f"Fisher {d['fisher_ratio']:.2f}  ·  silhouette {d['silhouette']:.2f}", fontsize=9)
fig.suptitle("The expert token induces a linearly separable persona space — 2-D LDA of the token-induced "
             "representation shift Δ\n(8 personas perfectly separable in every scheme; on balanced data the "
             "alternation doesn't sharpen it — the token does the work)", y=1.06, fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(HERE.parent / "figs" / "separability.png", dpi=140, bbox_inches="tight")
print("wrote separability.png")
