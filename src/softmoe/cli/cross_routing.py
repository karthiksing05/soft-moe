"""``cross-routing`` — expert×domain perplexity matrix for one or more trained runs.

Routes every domain's test set through every expert and reports the [domain × expert] perplexity
matrix + specialization summary. Only meaningful for runs with per-document experts (SoftMoE /
c-BTM); dense/hard_moe are skipped with a note.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from softmoe.data.dataset import load_dataset_split
from softmoe.eval.cross_routing import expert_domain_matrix, render_markdown
from softmoe.eval.harness import load_run_model
from softmoe.models.soft_moe import SoftMoE
from softmoe.utils.logging import get_logger

logger = get_logger()


def analyze_run(run_dir: str, data_root: str, device: str) -> tuple[str | None, str]:
    """Return (markdown_section, regime). Dispatches to the right per-model domain analysis."""
    from softmoe.models.baselines.hard_moe import HardMoE
    from softmoe.eval.moe_analysis import moe_domain_utilization, render_markdown as moe_render

    model, cfg, paths = load_run_model(run_dir, data_root)
    regime = cfg.get_path("meta.regime") or cfg.get_path("model.method", "run")
    with open(paths.domains) as fh:
        domain_names = list(json.load(fh)["domain_to_id"].keys())
    test = load_dataset_split(paths.root, "test")
    pad = int(cfg.get_path("data.pad_token_id", 0) or 0)
    bs = int(cfg.get_path("eval.batch_size", 8))

    if isinstance(model, SoftMoE):
        rep = expert_domain_matrix(model, test, device=device, batch_size=bs, pad_token_id=pad)
        s = rep["summary"]
        logger.info("[routing] %-14s gap=%.2fx best-is-self=%.0f%%", regime,
                    s["specialization_gap_ratio"], s["frac_domains_best_is_self"] * 100)
        return render_markdown(rep, domain_names, regime), regime
    if isinstance(model, HardMoE):
        rep = moe_domain_utilization(model, test, device=device, batch_size=bs, pad_token_id=pad)
        logger.info("[routing] %-14s MoE routing-NMI=%.3f dead=%d/%d", regime,
                    rep["routing_nmi"], rep["dead_experts"], rep["n_routed"])
        return moe_render(rep, domain_names, regime), regime
    logger.info("[routing] %s has no experts — per-domain ppl is in the main table.", regime)
    return None, regime


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Expert x domain cross-routing matrix.")
    ap.add_argument("--runs", nargs="+", required=True, help="one or more experiments/<run> dirs")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="markdown output path (also writes .json)")
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    md_sections = ["# Domain routing analysis (every model)\n",
                   "Per-model expert↔domain specialization: our expert-token methods show an "
                   "expert×domain **perplexity** matrix (swap test); the MoE arm shows its learned "
                   "expert×domain **token-routing** distribution + routing-NMI.\n"]
    for run in args.runs:
        section, regime = analyze_run(run, args.data_root, device)
        if section is not None:
            md_sections.append(section)

    md = "\n".join(md_sections)
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"wrote {out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
