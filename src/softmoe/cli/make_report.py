"""``make_report`` — aggregate runs' metrics into the comparison table + CSV + figures. (M6)"""

from __future__ import annotations

import argparse

from softmoe.eval.report import make_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate runs into a report.")
    ap.add_argument("--runs", default="experiments", help="dir of run subdirs with metrics.json")
    ap.add_argument("--out", default="reports/latest", help="output report dir")
    args = ap.parse_args(argv)

    result = make_report(args.runs, args.out)
    print(result["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
