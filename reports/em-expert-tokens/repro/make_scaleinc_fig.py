#!/usr/bin/env python
"""Incremental scaling figure: new-persona ppl + token load-bearing-ness vs number of base personas K."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "figs" / "scaleinc.json").read_text())
K = d["K"]; ppl = d["new_ppl"]; swap = d["new_swap"]

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax2 = ax.twinx()
l1, = ax.plot(K, ppl, "-o", ms=7, color="#2a6fdb", label="new-persona ppl (↓ better)")
for k, p in zip(K, ppl):
    ax.annotate(f"{p:.1f}", (k, p), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color="#2a6fdb")
l2, = ax2.plot(K, swap, "--s", ms=6, color="#5aa469", alpha=0.85, label="new-token swap-ratio (↑ = load-bearing)")
ax.set_xscale("log"); ax.set_xticks(K); ax.set_xticklabels([str(v) for v in K])
ax.set_xlabel("# base personas the backbone was trained on (K, log)")
ax.set_ylabel("new-persona held-out ppl (token-only add)", color="#2a6fdb")
ax2.set_ylabel("new-token swap-ratio", color="#5aa469")
ax.set_title("Incremental training scales with persona diversity\n"
             "adding a held-out persona (token-only, fixed budget) gets easier — and the new token becomes\n"
             "genuinely load-bearing — once the backbone has seen ~32+ distinct personas", fontsize=10.5)
ax.legend(handles=[l1, l2], frameon=False, fontsize=9, loc="center left")
fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "scaleinc.png", dpi=140, bbox_inches="tight")
print("wrote scaleinc.png")
