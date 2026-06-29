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
from softmoe.models.baselines.cbtm import CBTM
from softmoe.models.soft_moe import SoftMoE
from softmoe.utils.logging import get_logger

logger = get_logger()


def analyze_run(run_dir: str, data_root: str, device: str) -> tuple[dict | None, list[str], str]:
    model, cfg, paths = load_run_model(run_dir, data_root)
    regime = cfg.get_path("meta.regime") or cfg.get_path("model.method", "run")
    if not isinstance(model, (SoftMoE, CBTM)):
        logger.info("[cross-routing] %s has no per-document experts — skipping.", regime)
        return None, [], regime
    with open(paths.domains) as fh:
        domain_names = list(json.load(fh)["domain_to_id"].keys())
    test = load_dataset_split(paths.root, "test")
    pad = int(cfg.get_path("data.pad_token_id", 0) or 0)
    bs = int(cfg.get_path("eval.batch_size", 8))
    report = expert_domain_matrix(model, test, device=device, batch_size=bs, pad_token_id=pad)
    return report, domain_names, regime


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Expert x domain cross-routing matrix.")
    ap.add_argument("--runs", nargs="+", required=True, help="one or more experiments/<run> dirs")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="markdown output path (also writes .json)")
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    md_sections = ["# Expert × domain cross-routing\n"]
    all_json = {}
    for run in args.runs:
        report, names, regime = analyze_run(run, args.data_root, device)
        if report is None:
            continue
        md_sections.append(render_markdown(report, names, regime))
        all_json[regime] = report
        s = report["summary"]
        logger.info("[cross-routing] %-12s gap=%.2fx best-is-self=%.0f%%", regime,
                    s["specialization_gap_ratio"], s["frac_domains_best_is_self"] * 100)

    md = "\n".join(md_sections)
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        out.with_suffix(".json").write_text(json.dumps(all_json, indent=2))
        print(f"wrote {out} and {out.with_suffix('.json')}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
