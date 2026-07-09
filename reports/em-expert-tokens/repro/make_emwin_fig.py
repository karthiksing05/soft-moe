#!/usr/bin/env python
"""Figure for the many-personas / few-episodes study — when does EM beat joint SFT?

Reads ../figs/emwin.json (3B, K=64, sweep over episodes-per-persona n). Panel A: mean held-out ppl vs n
(log-y) for joint SFT / frozen-random / EM two-phase. Panel B: EM's % improvement over joint and over
frozen vs n — the advantage grows as data per persona shrinks.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "figs" / "emwin.json").read_text())
n = d["n"]; joint = d["joint"]; frozen = d["frozen"]; em = d["em"]
COL = {"joint": "#c0504d", "frozen": "#8c8c8c", "em": "#2a6fdb"}

fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))

ax[0].plot(n, joint, "-o", ms=6, color=COL["joint"], label="joint SFT (token, naive)")
ax[0].plot(n, frozen, "-o", ms=6, color=COL["frozen"], label="frozen-random tokens")
ax[0].plot(n, em, "-o", ms=6, color=COL["em"], label="EM two-phase")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xticks(n); ax[0].set_xticklabels([str(v) for v in n])
ax[0].set_xlabel(f"episodes per persona (n),  K={d['K']} personas  [{d['model']}]")
ax[0].set_ylabel("mean held-out ppl (log, lower = better)")
ax[0].set_title("All three tie at ample data; EM pulls ahead as data per persona shrinks")
ax[0].legend(frameon=False, fontsize=9)

emj = [100 * (j - e) / j for j, e in zip(joint, em)]
emf = [100 * (f - e) / f for f, e in zip(frozen, em)]
ax[1].axhline(0, color="k", lw=0.8)
ax[1].plot(n, emj, "-o", ms=6, color=COL["em"], label="EM vs joint SFT")
ax[1].plot(n, emf, "-s", ms=6, color="#5aa469", label="EM vs frozen-random")
ax[1].set_xscale("log"); ax[1].set_xticks(n); ax[1].set_xticklabels([str(v) for v in n])
ax[1].set_xlabel("episodes per persona (n)  (log)")
ax[1].set_ylabel("EM improvement (%)")
ax[1].set_title("EM's advantage grows as episodes-per-persona shrink\n(>0 = EM better)")
ax[1].legend(frameon=False, fontsize=9)
for x, g in zip(n, emj):
    ax[1].annotate(f"{g:+.0f}%", (x, g), textcoords="offset points", xytext=(0, 8), ha="center",
                   fontsize=8, fontweight="bold", color=COL["em"])
fig.suptitle("When does EM beat standard SFT? — many personas (K=64) × few episodes each (Qwen2.5-3B)",
             y=1.02, fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "emwin.png", dpi=140, bbox_inches="tight")
print("wrote emwin.png")
