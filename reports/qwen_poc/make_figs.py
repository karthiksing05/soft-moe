#!/usr/bin/env python
"""Render the persona/domain EM-vs-control figures from reports/qwen_poc/figs/results.json.

Two per-test panels (per-item held-out PPL control-vs-EM, and the swap-ratio bars) plus a headline
contrast figure (per-item EM gain % and mean swap-ratio, persona vs domain). Regenerate after the
domain run fills in the `domain` block. No cluster needed — pure matplotlib.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
DATA = FIGS / "results.json"

CTL, EM = "#8c8c8c", "#2a6fdb"     # control grey, EM blue


def _gain(ctl, em):
    return [100.0 * (c - e) / c for c, e in zip(ctl, em)]   # % PPL reduction (>0 = EM better)


def per_test_fig(block, name):
    items, ctl, em, swap = block["items"], block["control"], block["em"], block["swap"]
    order = sorted(range(len(items)), key=lambda i: swap[i], reverse=True)
    items = [items[i] for i in order]; ctl = [ctl[i] for i in order]
    em = [em[i] for i in order]; swap = [swap[i] for i in order]
    x = np.arange(len(items)); w = 0.4
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(x - w/2, ctl, w, label="control (generic token)", color=CTL)
    ax[0].bar(x + w/2, em, w, label="EM (per-expert token)", color=EM)
    ax[0].set_xticks(x); ax[0].set_xticklabels(items, rotation=35, ha="right")
    ax[0].set_ylabel("held-out PPL  (lower = better)")
    ax[0].set_title(f"{block['label']}\nper-item PPL: control vs EM")
    ax[0].legend(frameon=False, fontsize=9)
    cols = [EM if s >= 1.15 else CTL for s in swap]
    ax[1].bar(x, swap, 0.6, color=cols)
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_xticks(x); ax[1].set_xticklabels(items, rotation=35, ha="right")
    ax[1].set_ylabel("swap-ratio  (wrong-token PPL / right)")
    ax[1].set_title("token load-bearing-ness\n(>1 = wrong token hurts = real signal)")
    mg = float(np.mean(_gain(ctl, em))); ms = float(np.mean(swap))
    fig.suptitle(f"mean per-item EM gain {mg:+.1f}%    mean swap-ratio ×{ms:.2f}", y=1.02, fontsize=11)
    fig.tight_layout()
    p = FIGS / f"{name}_results.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p, mg, ms


def contrast_fig(data):
    labels, gains, swaps = [], [], []
    for key, short in [("domain", "domain-QA\n(identity in prompt)"), ("persona", "persona/style\n(identity hidden)")]:
        b = data.get(key)
        if not b:
            continue
        labels.append(short)
        gains.append(float(np.mean(_gain(b["control"], b["em"]))))
        swaps.append(float(np.mean(b["swap"])))
    if len(labels) < 2:
        return None
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    c = ["#c0504d" if g < 0 else EM for g in gains]
    ax[0].bar(x, gains, 0.5, color=c)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("mean per-item EM gain over control (%)")
    ax[0].set_title("Does the per-expert token help?")
    for xi, g in zip(x, gains):
        ax[0].text(xi, g + (0.3 if g >= 0 else -0.3), f"{g:+.1f}%", ha="center",
                   va="bottom" if g >= 0 else "top", fontsize=10, fontweight="bold")
    ax[1].bar(x, swaps, 0.5, color=EM)
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("mean swap-ratio (token load-bearing-ness)")
    ax[1].set_title("Does the token carry causal signal?")
    for xi, s in zip(x, swaps):
        ax[1].text(xi, s + 0.02, f"×{s:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.suptitle("EM expert-token pays off only when identity is hidden", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = FIGS / "contrast.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    data = json.loads(DATA.read_text())
    for key in ("persona", "domain"):
        if data.get(key):
            p, mg, ms = per_test_fig(data[key], key)
            print(f"wrote {p}  (mean gain {mg:+.1f}%, swap ×{ms:.2f})")
    c = contrast_fig(data)
    print(f"wrote {c}" if c else "contrast: need both persona and domain blocks")


if __name__ == "__main__":
    main()
