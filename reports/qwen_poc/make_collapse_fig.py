#!/usr/bin/env python
"""Persona-embedding collapse figure from figs/collapse.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "figs" / "collapse.json").read_text())
runs = d["runs"]; x = np.arange(len(runs))
COLS = ["#8c8c8c", "#2a6fdb", "#5aa469"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].bar(x, d["mean_cos"], 0.55, color=COLS)
ax[0].set_xticks(x); ax[0].set_xticklabels(runs)
ax[0].set_ylabel("mean pairwise cosine similarity")
ax[0].set_title("Embedding collapse\n(→1 = collapsed; lower = more distinct personas)")
for xi, v in zip(x, d["mean_cos"]):
    ax[0].annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontweight="bold")

ax[1].bar(x, d["mean_l2"], 0.55, color=COLS)
ax[1].set_xticks(x); ax[1].set_xticklabels(runs)
ax[1].set_ylabel("mean pairwise L2 distance")
ax[1].set_title("Embedding separation\n(higher = personas farther apart)")
for xi, v in zip(x, d["mean_l2"]):
    ax[1].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontweight="bold")
fig.suptitle("Persona-embedding collapse metric — EM keeps embeddings far more distinct than joint SFT",
             y=1.03, fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(HERE / "figs" / "collapse.png", dpi=140, bbox_inches="tight")
print("wrote collapse.png")
