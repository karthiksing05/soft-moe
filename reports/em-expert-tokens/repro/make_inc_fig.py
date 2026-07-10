#!/usr/bin/env python
"""Incremental-training figure: how cheaply can a NEW persona be added — token-only vs full SFT vs EM.

Reads ../figs/inc.json:
  {"arms":[{"name","color","steps":[...],"robot":[...],"base":[...],"phaseB_from"?}], "n":[...],"tokn_robot":[...]}
Panel A: new-persona ppl (solid) and retained base-persona ppl (dashed) vs incremental steps, per mechanism.
Panel B: new-persona ppl vs #new examples (token-only).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "figs" / "inc.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.9))

allsteps = set()
for arm in d["arms"]:
    c = arm["color"]; s = arm["steps"]; allsteps |= set(s)
    ax[0].plot(s, arm["robot"], "-o", color=c, ms=5, label=f"{arm['name']}: new persona")
    ax[0].plot(s, arm["base"], "--o", color=c, ms=4, alpha=0.55, label=f"{arm['name']}: base (retained)")
    if arm.get("phaseB_from") is not None:
        ax[0].axvline(arm["phaseB_from"], color=c, lw=0.8, ls=":", alpha=0.6)
        ax[0].text(arm["phaseB_from"], ax[0].get_ylim()[1], " EM: Phase A→B", color=c, fontsize=7, va="top")
ax[0].set_xscale("symlog"); ax[0].set_yscale("log")
xt = sorted(allsteps); ax[0].set_xticks(xt); ax[0].set_xticklabels([str(v) for v in xt], fontsize=7)
ax[0].set_xlabel("incremental training steps (0 = before adding)")
ax[0].set_ylabel("held-out ppl (log, lower = better)")
ax[0].set_title("Adding a new persona — solid = new persona, dashed = base personas retained\n"
                "token-only (fit ~1 embedding) · full SFT (fit 3B) · EM (Phase A backbone → Phase B tokens)")
ax[0].legend(frameon=False, fontsize=6.8, ncol=1, loc="upper right")

ax[1].plot(d["n"], d["tokn_robot"], "-o", color="#2a6fdb")
ax[1].set_xscale("log"); ax[1].set_xticks(d["n"]); ax[1].set_xticklabels([str(v) for v in d["n"]])
ax[1].set_xlabel("# examples of the new persona (log)")
ax[1].set_ylabel("new-persona held-out ppl")
ax[1].set_title("Data efficiency of adding a persona\n(token-only, 100 steps)")

fig.suptitle("Incremental training — how easy is it to add a new persona vector? (token-only vs full SFT vs EM)",
             y=1.02, fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "incremental.png", dpi=140, bbox_inches="tight")
print("wrote incremental.png")
