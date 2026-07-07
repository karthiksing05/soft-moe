#!/usr/bin/env python
"""Catastrophic-forgetting figure: does EM-continual forget Task A less than sequential full-FT?

Reads reports/qwen_poc/figs/cf.json:
  {personas_A, personas_B,
   fullft: {afterA: {p:ppl}, afterB: {p:ppl}},
   em:     {afterA: {p:ppl}, afterB: {p:ppl}}}
Panel 1: mean Task-A ppl after learning A vs after learning B (forgetting = the rise).
Panel 2: mean Task-B ppl after learning B (did the method still learn the new task?).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "figs" / "cf.json"
BEFORE, AFTER = "#5aa469", "#c0504d"


def gmean(d, keys):
    return float(np.exp(np.mean([np.log(d[k]) for k in keys])))


def main():
    d = json.loads(DATA.read_text())
    A, B = d["personas_A"], d["personas_B"]
    methods = [("fullft", "sequential full-FT"), ("em", "EM-continual\n(token-only on B)")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    x = np.arange(len(methods)); w = 0.38
    aA = [gmean(d[m]["afterA"], A) for m, _ in methods]
    aB = [gmean(d[m]["afterB"], A) for m, _ in methods]
    ax[0].bar(x - w/2, aA, w, label="Task-A ppl after learning A", color=BEFORE)
    ax[0].bar(x + w/2, aB, w, label="Task-A ppl after learning B", color=AFTER)
    for xi, b, a2 in zip(x, aA, aB):
        ax[0].annotate(f"{b:.2f}", (xi - w/2, b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax[0].annotate(f"{a2:.2f}\n(+{100*(a2-b)/b:.0f}%)", (xi + w/2, a2), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=8, fontweight="bold")
    ax[0].set_xticks(x); ax[0].set_xticklabels([lbl for _, lbl in methods])
    ax[0].set_ylabel("Task-A held-out ppl (lower = better)")
    ax[0].set_title("Forgetting of Task A\n(rise after learning Task B = catastrophic forgetting)")
    ax[0].legend(frameon=False, fontsize=8.5)

    bB = [gmean(d[m]["afterB"], B) for m, _ in methods]
    ax[1].bar(x, bB, 0.5, color="#2a6fdb")
    for xi, v in zip(x, bB):
        ax[1].annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, fontweight="bold")
    ax[1].set_xticks(x); ax[1].set_xticklabels([lbl for _, lbl in methods])
    ax[1].set_ylabel("Task-B held-out ppl (lower = better)")
    ax[1].set_title("Task B learned (after learning B)\n(did the method still acquire the new task?)")
    fig.suptitle("Catastrophic forgetting: sequential full-FT vs EM-continual (persona A → B)", y=1.02,
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "figs" / "catastrophic_forgetting.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote catastrophic_forgetting.png | A-forget: fullft +{100*(aB[0]-aA[0])/aA[0]:.0f}%  em +{100*(aB[1]-aA[1])/aA[1]:.0f}%")


if __name__ == "__main__":
    main()
