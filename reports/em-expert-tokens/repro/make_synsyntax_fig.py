#!/usr/bin/env python
"""Synthetic-syntax figure: exact-match accuracy (control / SFT / EM, right vs swap) + per-transform swap ppl."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "figs" / "synsyntax.json").read_text())

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))

labels = list(d["accuracy"]); vals = [d["accuracy"][k] for k in labels]
cols = ["#8c8c8c", "#2a6fdb", "#c0504d", "#2e8b57", "#c0504d"]
bars = ax[0].bar(range(len(labels)), vals, 0.62, color=cols)
ax[0].axhline(d["chance"], color="k", lw=0.9, ls="--")
ax[0].text(len(labels) - 0.5, d["chance"] + 2, f"chance (1/8 = {d['chance']:.1f}%)", ha="right", fontsize=8)
for i, v in enumerate(vals):
    ax[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels, fontsize=9)
ax[0].set_ylabel("exact-match generation accuracy (%)"); ax[0].set_ylim(0, 112)
ax[0].set_title("Reproducing an *impossible* transform\nboth SFT & EM two-phase nail it; a generic token can't; wrong token → 0")

sp1, sp2 = d["swap_ppl_sft"], d["swap_ppl_em"]; names = list(sp1); x = np.arange(len(names)); w = 0.4
ax[1].bar(x - w/2, [sp1[n] for n in names], w, color="#2a6fdb", label=f"SFT (joint)  mean ×{d['swap_mean_sft']:.1f}")
ax[1].bar(x + w/2, [sp2[n] for n in names], w, color="#2e8b57", label=f"EM two-phase  mean ×{d['swap_mean_em']:.1f}")
ax[1].axhline(1.0, color="k", lw=0.8, ls="--"); ax[1].text(len(names)-0.5, 1.06, "right-token ppl ≈ 1.0", ha="right", fontsize=8)
ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax[1].set_ylabel("wrong-token (swap) held-out ppl  [log]"); ax[1].legend(fontsize=8, loc="upper right")
ax[1].set_title("Swap test per transform — wrong token → wrong syntax\n(both fully load-bearing; SFT keys the token sharper)")

fig.suptitle("Synthetic-syntax personas — the token is *fully* load-bearing when the output is impossible from priors",
             y=1.02, fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "synsyntax.png", dpi=140, bbox_inches="tight")
print("wrote synsyntax.png")
