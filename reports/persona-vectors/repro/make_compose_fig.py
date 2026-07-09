#!/usr/bin/env python
"""Composition (interpolation) curves: held-out ppl of A's and B's data vs α for α·e_A + (1−α)·e_B."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
TASKS = [("persona", "Persona"), ("domain", "Domain")]


def main():
    blocks = [(t, json.loads((FIGS / f"{t}_compose.json").read_text()), lab)
              for t, lab in TASKS if (FIGS / f"{t}_compose.json").exists()]
    if not blocks:
        print("no compose JSONs yet"); return
    ncol = max(len(d["results"]) for _, d, _ in blocks)
    fig, ax = plt.subplots(len(blocks), ncol, figsize=(4.2 * ncol, 3.8 * len(blocks)), squeeze=False)
    for ri, (_, d, lab) in enumerate(blocks):
        for ci in range(ncol):
            a = ax[ri][ci]
            if ci >= len(d["results"]):
                a.axis("off"); continue
            r = d["results"][ci]; al = r["alphas"]
            a.plot(al, r["pplA"], "-o", color="#2a6fdb", label=f"{r['A']}'s data")
            a.plot(al, r["pplB"], "-s", color="#c0504d", label=f"{r['B']}'s data")
            a.set_xlabel(f"α  (α·{r['A']} + (1−α)·{r['B']})"); a.set_ylabel("held-out ppl")
            a.set_title(f"{lab}: {r['A']} ⊕ {r['B']}", fontsize=9)
            a.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Composing expert vectors — interpolating α·e_A + (1−α)·e_B and scoring each persona's held-out data",
                 y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "composition.png", dpi=140, bbox_inches="tight")
    print("wrote composition.png")


if __name__ == "__main__":
    main()
