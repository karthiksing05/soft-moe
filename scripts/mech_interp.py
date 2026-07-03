#!/usr/bin/env python
"""Mechanistic-interp studies on an expert-token run. Writes reports/mech_interp/{report.md,*.png}.

Usage: python scripts/mech_interp.py --run <run_dir> --data-root <dir> [--out reports/mech_interp]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from softmoe.data.dataset import load_dataset_split
from softmoe.eval.harness import load_run_model
from softmoe.eval.mech_interp import activation_shift, gate_signatures, latent_domain_separation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="reports/mech_interp")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    model, cfg, paths = load_run_model(a.run, a.data_root)
    test = load_dataset_split(paths.root, "test")
    pad = int(cfg.get_path("data.pad_token_id", 0) or 0)
    name = cfg.get_path("meta.regime", Path(a.run).name)

    shift = activation_shift(model, test, dev, pad=pad)
    gates = gate_signatures(model)
    latent = latent_domain_separation(model, test, dev, pad=pad)

    # --- fig 1: where the expert acts (per-layer residual shift) ---
    rs = shift["per_layer_rel_shift"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(rs) + 1), rs, "o-")
    ax.set_xlabel("layer"); ax.set_ylabel("relative L2 shift  ||h_expert−h_dense|| / ||h_dense||")
    ax.set_title(f"Where the expert token acts ({name})")
    fig.tight_layout(); fig.savefig(out / "1_activation_shift.png", dpi=130); plt.close(fig)

    # --- fig 2: what subspace each expert governs (spectral gates) ---
    if gates is not None:
        g = np.asarray(gates["gates"])                     # [K, L, r]
        K, L, r = g.shape
        mid = L - 1
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        im = a1.imshow(g[:, mid, :], aspect="auto", cmap="RdBu_r", vmin=0, vmax=2)
        a1.set_xlabel(f"governed direction (r={r})"); a1.set_ylabel("expert (domain)")
        a1.set_title(f"Gate signature @ layer {mid+1}\n(>1 amplify, <1 suppress; 1=pass-through)")
        fig.colorbar(im, ax=a1)
        a2.plot(range(1, L + 1), gates["cross_expert_cosine_by_layer"], "s-")
        a2.axhline(0, color="gray", ls="--", lw=0.8)
        a2.set_xlabel("layer"); a2.set_ylabel("mean cross-expert gate cosine")
        a2.set_title("Do domains govern distinct subspaces?\n(0 = orthogonal/distinct, 1 = identical)")
        fig.tight_layout(); fig.savefig(out / "2_gate_signatures.png", dpi=130); plt.close(fig)

    # --- fig 3: latent-space domain separation (with vs without the token) ---
    P = np.asarray(latent["pca_with"]); Y = np.asarray(latent["labels"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = a1.scatter(P[:, 0], P[:, 1], c=Y, cmap="tab10", s=8, alpha=0.7)
    a1.set_title("Pooled last-hidden (PCA), colored by domain\n[with expert token]")
    a1.set_xlabel("PC1"); a1.set_ylabel("PC2"); fig.colorbar(sc, ax=a1, label="domain")
    aw, ad = latent["probe_acc_with_expert"], latent["probe_acc_dense"]
    a2.bar(["dense\n(no token)", "with\nexpert token"], [ad, aw], color=["#888", "#1f77b4"])
    a2.set_ylim(0, 1); a2.set_ylabel("domain linear-probe accuracy ↑")
    a2.set_title(f"Does the token add domain-linear-separability?\n{ad:.3f} → {aw:.3f}")
    for i, v in enumerate([ad, aw]):
        a2.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(out / "3_latent_separation.png", dpi=130); plt.close(fig)

    # --- report ---
    L = [f"# Mechanistic interp — {name}", "",
         "How the EM expert token changes the network, measured as *expert-on vs governor-off* (the",
         "only difference is the conditioning). Figures 1–3.", "",
         "## 1. Where the expert acts (per-layer residual shift)",
         "Relative L2 change in the residual stream, per layer (`1_activation_shift.png`):", "",
         "| layer | rel. shift |", "|---|---|"]
    L += [f"| {i+1} | {v:.3f} |" for i, v in enumerate(rs)]
    if gates is not None:
        L += ["", "## 2. What subspace each expert governs (`2_gate_signatures.png`)",
              "Each expert's per-layer gate over the r governed directions (>1 amplify, <1 suppress).",
              "Cross-expert gate cosine (0 = distinct subspaces per domain, 1 = identical):", "",
              "| layer | cross-expert cosine |", "|---|---|"]
        L += [f"| {i+1} | {v:.3f} |" for i, v in enumerate(gates["cross_expert_cosine_by_layer"])]
    L += ["", "## 3. Latent-space domain separation (`3_latent_separation.png`)",
          f"- domain linear-probe accuracy **dense {latent['probe_acc_dense']:.3f} → "
          f"with-expert {latent['probe_acc_with_expert']:.3f}** ({latent['n_domains']} domains).",
          "- Higher with the token ⇒ the expert makes domains more linearly separable in the latent space."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))
    print(f"\nwrote {out}/report.md + 3 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
