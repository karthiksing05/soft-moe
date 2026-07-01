#!/usr/bin/env python
"""Generate comparison figures for the scaled experiment into reports/scaled/comparison_figures/.

Reads reports/scaled/{results.csv, per_domain_ppl.md}; the MoE arm's *learned* domain routing-NMI
and the expert-token *cross-routing* degradation (×worse) come from routing_analysis.md and are
kept here as small dicts (they aren't in results.csv). Run: python scripts/make_comparison_figures.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "reports" / "scaled"
OUT = SC / "comparison_figures"
OUT.mkdir(parents=True, exist_ok=True)

FAMILY = {  # method -> (family, pretty label)
    "dense_1x": ("dense", "Dense-1×"), "dense_ceiling": ("dense", "Dense-ceiling"),
    "moe_g1": ("moe", "MoE-G1 (coarse)"), "moe_g2": ("moe", "MoE-G2 (fine)"),
    "moe_oracle": ("moe", "MoE-oracle (DEMix)"), "cbtm": ("cbtm", "c-BTM"),
    "ours_sup_alt": ("ours", "ours sup+alt"), "ours_sup_seq": ("ours", "ours sup+seq"),
    "ours_unsup_alt": ("ours", "ours unsup+alt"), "ours_unsup_seq": ("ours", "ours unsup+seq"),
}
FAM_COLOR = {"dense": "#888888", "moe": "#d62728", "ours": "#1f77b4", "cbtm": "#2ca02c"}

# domain routing-NMI: MoE = its *learned* routing (routing_analysis.md); ours = main-table routing-NMI
DOMAIN_NMI = {"moe_g1": 0.009, "moe_g2": 0.462, "moe_oracle": 1.000,
              "ours_sup_alt": 1.000, "ours_sup_seq": 1.000,
              "ours_unsup_alt": 0.621, "ours_unsup_seq": 0.613}
# cross-routing ×worse-through-wrong-expert (routing_analysis.md), expert methods only
XWORSE = {"cbtm": 3.05, "ours_sup_alt": 1.48, "ours_unsup_alt": 1.06,
          "ours_sup_seq": 1.02, "ours_unsup_seq": 1.02}


def load_results() -> dict:
    rows = {}
    with open(SC / "results.csv") as fh:
        for r in csv.DictReader(fh):
            rows[r["method"]] = {k: (float(v) if v not in ("", "nan") else float("nan"))
                                 for k, v in r.items() if k != "method"}
    return rows


def load_per_domain() -> tuple[list[str], dict]:
    text = (SC / "per_domain_ppl.md").read_text().splitlines()
    header = next(l for l in text if l.startswith("| method"))
    domains = [c.strip() for c in header.split("|")[2:-2]]      # between method and macro
    data = {}
    for l in text:
        if l.startswith("| ") and not l.startswith("| method") and "---" not in l:
            cells = [c.strip() for c in l.split("|")[1:-1]]
            if len(cells) >= len(domains) + 2:
                try:
                    data[cells[0]] = [float(x) for x in cells[1:1 + len(domains)]]
                except ValueError:
                    pass
    return domains, data


R = load_results()
DENSE = R["dense_1x"]["macro_ppl"]


def _order(keys):  # sort by macro-ppl
    return sorted(keys, key=lambda k: R[k]["macro_ppl"])


# --- 1. macro-ppl bar (all models) ---
def fig_macro_bar():
    ks = _order(R.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([FAMILY[k][1] for k in ks], [R[k]["macro_ppl"] for k in ks],
            color=[FAM_COLOR[FAMILY[k][0]] for k in ks])
    ax.axvline(DENSE, color="k", ls="--", lw=0.9, label="Dense-1×")
    ax.set_xlabel("macro-perplexity ↓"); ax.set_title("Scaled comparison — macro-ppl (all 10 models)")
    ax.invert_yaxis(); ax.legend(); ax.set_xlim(2.3, 3.3)
    for i, k in enumerate(ks):
        ax.text(R[k]["macro_ppl"] + 0.01, i, f"{R[k]['macro_ppl']:.3f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "1_macro_ppl_all.png", dpi=130); plt.close(fig)


# --- 2/3. Axis A (active) and Axis B (total) ---
def fig_axis(param_key, fname, title, xlabel):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in R:
        fam = FAMILY[k][0]
        ax.scatter(R[k][param_key] / 1e6, R[k]["macro_ppl"], color=FAM_COLOR[fam], s=60,
                   edgecolor="k", lw=0.4, zorder=3)
        ax.annotate(FAMILY[k][1], (R[k][param_key] / 1e6, R[k]["macro_ppl"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(DENSE, color="#888888", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("macro-ppl ↓"); ax.set_title(title)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAM_COLOR.items()]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=130); plt.close(fig)


# --- 4. granularity / routing story ---
def fig_granularity():
    ks = ["dense_1x", "moe_g1", "moe_g2", "moe_oracle"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([FAMILY[k][1] for k in ks], [R[k]["macro_ppl"] for k in ks],
                  color=["#888888", "#f4a582", "#d62728", "#b2182b"])
    ax.axhline(DENSE, color="k", ls="--", lw=0.9, label="Dense-1×")
    ax.set_ylabel("macro-ppl ↓"); ax.set_ylim(2.3, 2.65)
    ax.set_title("Granularity is decisive (H1/H2)\ncoarse MoE > dense > fine MoE")
    for b, k in zip(bars, ks):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{R[k]['macro_ppl']:.3f}", ha="center", fontsize=8)
    plt.xticks(rotation=15, ha="right"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "4_granularity_effect.png", dpi=130); plt.close(fig)


# --- 5. domain routing-NMI: MoE (learned) vs ours (explicit) ---
def fig_domain_nmi():
    ks = ["moe_g1", "moe_g2", "moe_oracle", "ours_unsup_seq", "ours_sup_seq"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([FAMILY[k][1] for k in ks], [DOMAIN_NMI[k] for k in ks],
           color=[FAM_COLOR[FAMILY[k][0]] for k in ks])
    ax.set_ylabel("domain routing-NMI ↑"); ax.set_ylim(0, 1.05)
    ax.set_title("Domain specialization: MoE routes tokens (balanced/emergent)\nvs ours = explicit domain experts")
    for i, k in enumerate(ks):
        ax.text(i, DOMAIN_NMI[k] + 0.02, f"{DOMAIN_NMI[k]:.2f}", ha="center", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(OUT / "5_domain_routing_nmi.png", dpi=130); plt.close(fig)


# --- 6. specialization (×worse) vs quality ---
def fig_spec_vs_quality():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, xw in XWORSE.items():
        ax.scatter(xw, R[k]["macro_ppl"], color=FAM_COLOR[FAMILY[k][0]], s=70, edgecolor="k", lw=0.4)
        ax.annotate(FAMILY[k][1], (xw, R[k]["macro_ppl"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.axhline(DENSE, color="#888888", ls="--", lw=0.8, label="Dense-1×")
    ax.set_xlabel("expert specialization (× worse through wrong expert →)")
    ax.set_ylabel("macro-ppl ↓"); ax.set_title("Specialization vs quality (expert methods)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "6_specialization_vs_quality.png", dpi=130); plt.close(fig)


# --- 7. per-domain ppl heatmap ---
def fig_per_domain():
    domains, data = load_per_domain()
    ks = [k for k in _order(R.keys()) if k in data]
    M = np.array([data[k] for k in ks])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([FAMILY[k][1] for k in ks])
    for i in range(len(ks)):
        for j in range(len(domains)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=6)
    fig.colorbar(im, label="perplexity ↓"); ax.set_title("Per-domain perplexity (every model)")
    fig.tight_layout(); fig.savefig(OUT / "7_per_domain_heatmap.png", dpi=130); plt.close(fig)


# --- 8. efficiency: quality vs total params (memory), families ---
def fig_efficiency():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in R:
        ax.scatter(R[k]["total_params"] / 1e6, R[k]["macro_ppl"], color=FAM_COLOR[FAMILY[k][0]],
                   s=60, edgecolor="k", lw=0.4)
        ax.annotate(FAMILY[k][1], (R[k]["total_params"] / 1e6, R[k]["macro_ppl"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("total params (M, log ≈ memory)"); ax.set_ylabel("macro-ppl ↓")
    ax.set_title("Efficiency: quality vs total params (memory axis)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAM_COLOR.items()]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "8_quality_vs_total_params.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    fig_macro_bar()
    fig_axis("active_params", "2_axisA_quality_vs_active.png",
             "Axis A (iso-active-FLOPs): quality vs active params", "active params/token (M, log)")
    fig_axis("total_params", "3_axisB_quality_vs_total.png",
             "Axis B (iso-total-params): quality vs total params", "total params (M, log)")
    fig_granularity()
    fig_domain_nmi()
    fig_spec_vs_quality()
    fig_per_domain()
    fig_efficiency()
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")
