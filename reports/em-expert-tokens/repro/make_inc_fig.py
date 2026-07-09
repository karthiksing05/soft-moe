#!/usr/bin/env python
"""Incremental-training figure: how cheaply can a NEW persona be added?

Reads ../figs/inc.json:
  {"steps":[0,10,30,100,300], "tok_robot":[...],"tok_base":[...],"full_robot":[...],"full_base":[...],
   "n":[5,25,100], "tokn_robot":[...]}
Panel A: new-persona ppl (solid) and retained base-persona ppl (dashed) vs incremental steps, for
token-only (fit 1 embedding, backbone frozen) vs full fine-tune. Panel B: new-persona ppl vs #new examples.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE.parent / "figs" / "inc.json").read_text())
s = d["steps"]
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.7))

ax[0].plot(s, d["tok_robot"], "-o", color="#2a6fdb", label="token-only: new persona")
ax[0].plot(s, d["tok_base"], "--o", color="#2a6fdb", alpha=0.6, label="token-only: base personas (retained)")
ax[0].plot(s, d["full_robot"], "-s", color="#c0504d", label="full fine-tune: new persona")
ax[0].plot(s, d["full_base"], "--s", color="#c0504d", alpha=0.6, label="full fine-tune: base personas (forgetting)")
ax[0].set_xscale("symlog"); ax[0].set_yscale("log")
ax[0].set_xticks(s); ax[0].set_xticklabels([str(v) for v in s])
ax[0].set_xlabel("incremental training steps (0 = before adding)")
ax[0].set_ylabel("held-out ppl (log, lower = better)")
ax[0].set_title("Adding a new persona: token-only (fit ~1 embedding, backbone frozen)\nvs full fine-tune "
                "(fit 3B) — solid = new persona, dashed = base retained")
ax[0].legend(frameon=False, fontsize=7.5)

ax[1].plot(d["n"], d["tokn_robot"], "-o", color="#2a6fdb")
ax[1].set_xscale("log"); ax[1].set_xticks(d["n"]); ax[1].set_xticklabels([str(v) for v in d["n"]])
ax[1].set_xlabel("# examples of the new persona (log)")
ax[1].set_ylabel("new-persona held-out ppl")
ax[1].set_title("Data efficiency of adding a persona\n(token-only, 100 steps)")

fig.suptitle("Incremental training — how easy is it to add a new persona vector?", y=1.02,
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "incremental.png", dpi=140, bbox_inches="tight")
print("wrote incremental.png")
