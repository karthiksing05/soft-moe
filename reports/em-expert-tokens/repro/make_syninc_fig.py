#!/usr/bin/env python
"""Incremental synthetic-syntax figure: adding a NOVEL impossible transform three ways.
Panel A: new-transform accuracy vs steps (token-only plateaus; SFT/EM reach 100%).
Panel B: the trade-off — new-transform vs base retention (token-only preserves base but partial;
         SFT/EM fully learn but catastrophically forget)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "figs" / "syninc.json").read_text())
arms = d["arms"]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.9))

# Panel A: new-transform accuracy vs steps
for name, a in arms.items():
    ax[0].plot(a["steps"], a["new"], "-o", color=a["color"], lw=2, label=name.replace("\n", " "))
ax[0].axhline(100, color="k", lw=0.7, ls=":")
ax[0].set_xlabel("optimizer steps on the new transform"); ax[0].set_ylabel("new-transform exact-match (%)")
ax[0].set_ylim(-4, 108); ax[0].legend(fontsize=8.5, loc="center right")
ax[0].set_title("Learning the *novel* transform (len_prefix)\ntoken-only plateaus ~42%; backbone updates reach 100%")

# Panel B: trade-off — new vs base retention
names = list(arms); x = np.arange(len(names)); w = 0.38
new = [arms[n]["final_new"] for n in names]; base = [arms[n]["final_base"] for n in names]
ax[1].bar(x - w/2, new, w, color="#4472c4", label="new transform (len_prefix)")
ax[1].bar(x + w/2, base, w, color="#ed7d31", label="base retention (7 old transforms)")
ax[1].axhline(d["base_before"]["base"], color="#ed7d31", lw=1, ls="--")
ax[1].text(len(names)-0.5, d["base_before"]["base"] - 7, "base before adding (99.8%)", ha="right", color="#c55a11", fontsize=8)
for i, (nv, bv) in enumerate(zip(new, base)):
    ax[1].text(i - w/2, nv + 2, f"{nv:.0f}", ha="center", fontsize=8.5, fontweight="bold")
    ax[1].text(i + w/2, bv + 2, f"{bv:.0f}", ha="center", fontsize=8.5, fontweight="bold")
ax[1].set_xticks(x); ax[1].set_xticklabels([n.replace("\n", "\n") for n in names], fontsize=8.5)
ax[1].set_ylabel("exact-match (%)"); ax[1].set_ylim(0, 112); ax[1].legend(fontsize=8.5, loc="upper center")
ax[1].set_title("No free lunch when the capability is NOVEL\ntoken-only keeps the base but is partial; SFT/EM learn it but forget")

fig.suptitle("Incrementally adding a genuinely novel impossible transform — token-only can't install new "
             "computation into a frozen backbone", y=1.02, fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "syninc.png", dpi=140, bbox_inches="tight")
print("wrote syninc.png")
