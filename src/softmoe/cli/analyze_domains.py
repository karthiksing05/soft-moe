"""``analyze-domains`` — emit a domain-separability report for a built corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from softmoe.data.build import CorpusPaths
from softmoe.eval.domain_analysis import analyze_domains, render_markdown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze domain separability of a built corpus.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--processed-dir", help="path to data/processed/<recipe>")
    g.add_argument("--recipe", help="recipe name (with --data-root)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=None, help="markdown output path (also writes .json next to it)")
    args = ap.parse_args(argv)

    pdir = Path(args.processed_dir) if args.processed_dir else CorpusPaths.for_recipe(Path(args.data_root), args.recipe).root
    report = analyze_domains(pdir)
    md = render_markdown(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        out.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(f"wrote {out} and {out.with_suffix('.json')}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
