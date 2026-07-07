#!/usr/bin/env python
"""Catastrophic-forgetting figure across methods ordered by how much the backbone is updated on Task B.

Reads reports/qwen_poc/figs/cf.json:
  {personas_A, personas_B, methods:[{key,label,color}],
   <key>: {afterA:{p:ppl}, afterB:{p:ppl}}, ...}
Panel 1: mean Task-A ppl after learning A vs after learning B (rise = forgetting).
Panel 2: mean Task-B ppl after learning B (did the method still learn the new task?).
The retention/plasticity trade-off runs left→right as backbone updates on B increase.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "figs" / "cf.json"
BEFORE, AFTER = "#5aa469", "#c0504d"


def gmean(d, keys):
    return float(np.exp(np.mean([np.log(d[k]) for k in keys])))


def main():
    d = json.loads(DATA.read_text())
    A, B = d["personas_A"], d["personas_B"]
    methods = d["methods"]
    labels = [m["label"] for m in methods]
    x = np.arange(len(methods)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(6 + 2.2 * len(methods), 4.7))

    aA = [gmean(d[m["key"]]["afterA"], A) for m in methods]
    aB = [gmean(d[m["key"]]["afterB"], A) for m in methods]
    ax[0].bar(x - w/2, aA, w, label="Task-A ppl after learning A", color=BEFORE)
    ax[0].bar(x + w/2, aB, w, label="Task-A ppl after learning B", color=AFTER)
    for xi, b, a2 in zip(x, aA, aB):
        ax[0].annotate(f"{b:.2f}", (xi - w/2, b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax[0].annotate(f"{a2:.2f}\n(+{100*(a2-b)/b:.0f}%)", (xi + w/2, a2), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=8, fontweight="bold")
    if d.get("joint_ref"):
        ax[0].axhline(d["joint_ref"]["A"], color="k", lw=0.9, ls=":")
        ax[0].text(len(methods) - 0.5, d["joint_ref"]["A"], " joint-trained ceiling", fontsize=7.5, va="bottom", ha="right")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("Task-A held-out ppl (lower = better)")
    ax[0].set_title("Forgetting of Task A\n(rise after learning Task B = catastrophic forgetting)")
    ax[0].legend(frameon=False, fontsize=8.5)

    bB = [gmean(d[m["key"]]["afterB"], B) for m in methods]
    cols = [m.get("color", "#2a6fdb") for m in methods]
    ax[1].bar(x, bB, 0.5, color=cols)
    for xi, v in zip(x, bB):
        ax[1].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, fontweight="bold")
    if d.get("joint_ref"):
        ax[1].axhline(d["joint_ref"]["B"], color="k", lw=0.9, ls=":")
        ax[1].text(len(methods) - 0.5, d["joint_ref"]["B"], " joint-trained ceiling", fontsize=7.5, va="bottom", ha="right")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("Task-B held-out ppl (lower = better)")
    ax[1].set_title("Task B learned (after learning B)\n(did the method still acquire the new task?)")
    fig.suptitle("Catastrophic forgetting: sequential full-FT vs EM (cycling / token-only) on Task B",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE.parent / "figs" / "catastrophic_forgetting.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    for m, b, a2, bb in zip(methods, aA, aB, bB):
        print(f"{m['key']:10} A-forget +{100*(a2-b)/b:5.0f}%   B-learned {bb:.2f}")


if __name__ == "__main__":
    main()
