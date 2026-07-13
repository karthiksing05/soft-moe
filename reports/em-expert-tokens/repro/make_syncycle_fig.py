#!/usr/bin/env python
"""Cyclic-EM incremental figure. Panel A: seed-averaged (n=3) new vs base retention for the key treatments
— token-FIRST recovers retention that A-first EM (== full-SFT) throws away. Panel B: the token-first
Pareto frontier vs backbone budget (single seed, shape), sitting up-right of the A-first / full-SFT corner."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "figs" / "syncycle.json").read_text())

fig, ax = plt.subplots(1, 2, figsize=(14, 5.0))

# Panel A: seed-averaged new + base retention
order = d["seed_order"]; sd = d["seeded"]
mean = lambda v: float(np.mean(v)); std = lambda v: float(np.std(v))
x = np.arange(len(order)); w = 0.38
new_m = [mean(sd[k]["new"]) for k in order];  new_s = [std(sd[k]["new"]) for k in order]
bas_m = [mean(sd[k]["base"]) for k in order]; bas_s = [std(sd[k]["base"]) for k in order]
ax[0].bar(x - w/2, new_m, w, yerr=new_s, capsize=3, color="#4472c4", label="new transform (len_prefix)")
ax[0].bar(x + w/2, bas_m, w, yerr=bas_s, capsize=3, color="#ed7d31", label="base retention (7 old)")
for i in range(len(order)):
    ax[0].text(i - w/2, new_m[i] + 2, f"{new_m[i]:.0f}", ha="center", fontsize=8)
    ax[0].text(i + w/2, bas_m[i] + bas_s[i] + 2, f"{bas_m[i]:.0f}", ha="center", fontsize=8, fontweight="bold")
ax[0].set_xticks(x); ax[0].set_xticklabels([o.replace(" ", "\n", 1) for o in order], fontsize=8.5)
ax[0].set_ylabel("exact-match (%)"); ax[0].set_ylim(0, 116); ax[0].legend(fontsize=8.5, loc="upper center", ncol=2)
ax[0].set_title("Same 150-step backbone budget — only the ORDER differs (n=3, ±sd)\n"
                "A-first EM forgets like full-SFT (base ≈15%); token-FIRST keeps ≈59%")

# Panel B: token-first Pareto frontier vs backbone budget a
fr = d["frontier"]; rf = d["frontier_refs"]
ax[1].axhspan(90, 101, xmin=0.62, xmax=1.0, color="#e8f4e8", zorder=0)
ax[1].text(97, 93, "goal:\nkeep base +\nlearn new", ha="right", va="center", fontsize=8, color="#2e7d32")
ax[1].plot(fr["base"], fr["new"], "-o", color="#2e8b57", lw=2, zorder=3, label="token-first frontier (vary backbone budget a)")
for a, b, n in zip(fr["a"], fr["base"], fr["new"]):
    ax[1].annotate(f"a={a}", (b, n), textcoords="offset points", xytext=(5, -9), fontsize=7.5, color="#2e8b57")
mk = {"token-only": ("s", "#8c8c8c"), "A-first EM": ("X", "#c0504d"), "full-SFT": ("P", "#d99694"), "replay": ("*", "#8064a2")}
for k, (m, c) in mk.items():
    ax[1].scatter(rf[k]["base"], rf[k]["new"], marker=m, s=150 if m == "*" else 90, color=c, zorder=4, label=k, edgecolor="k", linewidth=0.4)
ax[1].annotate("", xy=(fr["base"][-1], fr["new"][-1]), xytext=(rf["A-first EM"]["base"], rf["A-first EM"]["new"]),
               arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.6, ls="--"))
ax[1].text(50, 88, "reorder\nB before A", fontsize=8, color="#2e8b57", ha="center")
ax[1].set_xlabel("base retention (%)  →  more"); ax[1].set_ylabel("new-transform accuracy (%)  →  more")
ax[1].set_xlim(20, 100); ax[1].set_ylim(0, 108); ax[1].legend(fontsize=7.8, loc="lower left")
ax[1].set_title("Pareto frontier (single seed) — token-first pushes toward the\nreplay corner without any base data")

fig.suptitle("Making EM incremental: fit the token FIRST, then a backbone burst — recovers forgetting for free",
             y=1.02, fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "syncycle.png", dpi=140, bbox_inches="tight")
print("wrote syncycle.png")
