#!/usr/bin/env python
"""Synthetic-syntax figure: exact-match accuracy (control / EM-right / EM-swap) + per-transform swap ppl."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "figs" / "synsyntax.json").read_text())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))

labels = list(d["accuracy"]); vals = [d["accuracy"][k] for k in labels]
cols = ["#8c8c8c", "#2a6fdb", "#c0504d"]
ax[0].bar(range(len(labels)), vals, 0.6, color=cols)
ax[0].axhline(d["chance"], color="k", lw=0.9, ls="--")
ax[0].text(len(labels) - 0.5, d["chance"] + 1.5, f"chance (1/8 = {d['chance']:.1f}%)", ha="right", fontsize=8)
for i, v in enumerate(vals):
    ax[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels, fontsize=9)
ax[0].set_ylabel("exact-match generation accuracy (%)"); ax[0].set_ylim(0, 108)
ax[0].set_title("Reproducing an *impossible* transform\nonly the right token works — a generic token can't")

sp = d["swap_ppl"]; names = list(sp); x = np.arange(len(names))
ax[1].bar(x, [sp[n] for n in names], 0.6, color="#c0504d")
ax[1].axhline(d["right_ppl"], color="#2a6fdb", lw=1.2, ls="--")
ax[1].text(len(names) - 0.5, d["right_ppl"] + 0.4, f"right-token ppl ≈ {d['right_ppl']:.2f}", ha="right", color="#2a6fdb", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax[1].set_ylabel("wrong-token (swap) held-out ppl")
ax[1].set_title(f"Swap test per transform — wrong token → wrong syntax\n(mean swap-ratio ×{d['swap_mean']:.1f}; "
                "right-token ppl ≈ 1.0)")

fig.suptitle("Synthetic-syntax personas — the token is *fully* load-bearing when the output is impossible from priors",
             y=1.02, fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "synsyntax.png", dpi=140, bbox_inches="tight")
print("wrote synsyntax.png")
