"""``build_data`` — raw -> tokenized/sharded corpus + cluster assignments. (Milestone M1)"""

from __future__ import annotations

import argparse

from softmoe.data.build import build_corpus
from softmoe.utils.config import load_config


def _data_subconfig(cfg):
    data = cfg["data"] if "data" in cfg else cfg
    data["seed"] = cfg.get("seed", data.get("seed", 0))
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a multi-domain tokenized corpus.")
    ap.add_argument("--config", required=True, help="path to a data or experiment yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    paths = build_corpus(_data_subconfig(cfg), data_root=args.data_root, force=args.force)
    print(f"built corpus at {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
