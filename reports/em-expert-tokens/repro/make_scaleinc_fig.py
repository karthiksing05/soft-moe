#!/usr/bin/env python
"""Incremental scaling figure: new-persona ppl (added via token-only) vs number of base personas K."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "figs" / "scaleinc.json").read_text())
K = d["K"]; ppl = d["new_ppl"]

fig, ax = plt.subplots(figsize=(7.6, 5.0))
ax.plot(K, ppl, "-o", ms=7, color="#2a6fdb")
for k, p in zip(K, ppl):
    ax.annotate(f"{p:.2f}", (k, p), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
ax.set_xscale("log")
ax.set_xticks(K); ax.set_xticklabels([str(v) for v in K])
ax.set_xlabel("# base personas the backbone was trained on (K, log)")
ax.set_ylabel("new-persona held-out ppl after token-only add (lower = better)")
ax.set_title("Incremental training scales with persona diversity\n"
             "adding a held-out persona (token-only, fixed budget) gets EASIER as the backbone\n"
             "has seen more distinct personas — a more general conditioning backbone", fontsize=11)
fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "scaleinc.png", dpi=140, bbox_inches="tight")
print("wrote scaleinc.png")
