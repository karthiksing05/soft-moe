#!/usr/bin/env python
"""Cold-start / long-tail figure: does EM beat joint SFT *more* as a persona's train volume shrinks?

Reads reports/qwen_poc/figs/coldstart.json: {personas, volumes, control, joint, em_std, em_bheavy}
(per-persona held-out ppl). Panel A: per-persona joint-vs-EM ppl (sorted by volume). Panel B: the
EM-over-joint improvement (%) vs train volume on a log axis — the thesis predicts it rises for the tail.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
DATA = FIGS / "coldstart.json"
CTL, JOINT, EM = "#8c8c8c", "#c0504d", "#2a6fdb"


def main():
    d = json.loads(DATA.read_text())
    order = sorted(d["personas"], key=lambda p: -d["volumes"][p])
    vols = [d["volumes"][p] for p in order]
    joint = [d["joint"][p] for p in order]
    em = [d["em_std"][p] for p in order]
    gap = [100 * (j - e) / j for j, e in zip(joint, em)]
    x = np.arange(len(order))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    w = 0.4
    ax[0].bar(x - w/2, joint, w, label="joint SFT (naive)", color=JOINT)
    ax[0].bar(x + w/2, em, w, label="EM two-phase", color=EM)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{p}\n({v})" for p, v in zip(order, vols)], fontsize=8)
    ax[0].set_ylabel("held-out ppl (lower = better)")
    ax[0].set_title("Per-persona held-out ppl by train volume\n(persona (n train examples), high → low)")
    ax[0].legend(frameon=False, fontsize=9)

    ax[1].axhline(0, color="k", lw=0.8)
    cols = [EM if g > 0 else JOINT for g in gap]
    ax[1].scatter(vols, gap, c=cols, s=70, zorder=3)
    for v, g, p in zip(vols, gap, order):
        ax[1].annotate(p, (v, g), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    if "em_bheavy" in d:
        gb = [100 * (d["joint"][p] - d["em_bheavy"][p]) / d["joint"][p] for p in order]
        ax[1].plot(vols, gb, "x", ms=8, color="#5aa469", label="EM Phase-B-heavy vs joint")
    z = np.polyfit(np.log10(vols), gap, 1)
    xs = np.linspace(min(vols), max(vols), 50)
    ax[1].plot(xs, np.polyval(z, np.log10(xs)), "--", color=EM, lw=1, label="trend")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("train examples for this persona (log)")
    ax[1].set_ylabel("EM improvement over joint SFT (%)")
    ax[1].set_title("Does EM help MORE in the low-data tail?\n(>0 = EM better; thesis predicts rise → left)")
    ax[1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(FIGS / "coldstart.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {FIGS/'coldstart.png'}   mean EM gap {np.mean(gap):+.1f}%   "
          f"tail(<=32) {np.mean([g for v,g in zip(vols,gap) if v<=32]):+.1f}%   "
          f"head(>=128) {np.mean([g for v,g in zip(vols,gap) if v>=128]):+.1f}%")


if __name__ == "__main__":
    main()
