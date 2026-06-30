"""``train`` — config-driven training for every method (ours + all baselines). (M2–M5)

A run is fully described by one ``configs/experiment/*.yaml``. Builds data if missing, constructs
the model from config, runs the EM trainer, and (unless ``--no-eval``) evaluates immediately.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from softmoe.data.build import CorpusPaths, build_corpus
from softmoe.data.dataset import build_tokenizer, load_dataset_split, tokenizer_vocab_size
from softmoe.models.factory import build_model
from softmoe.training.em_trainer import EMTrainer
from softmoe.utils.config import load_config
from softmoe.utils.logging import get_logger

logger = get_logger()


def _data_subconfig(cfg):
    data = cfg["data"] if "data" in cfg else cfg
    data["seed"] = cfg.get("seed", data.get("seed", 0))
    return data


def _load_centroids(paths: CorpusPaths):
    try:
        with open(paths.clusterer, "rb") as fh:
            clusterer = pickle.load(fh)
        return getattr(clusterer, "centroids", None)
    except (FileNotFoundError, AttributeError):  # pragma: no cover
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train a soft-MoE method or baseline.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--run-dir", default=None, help="output dir (default experiments/<name>)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force-data", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--init-backbone-from", default=None,
                    help="run dir (or checkpoint) to load backbone weights from before training "
                         "(for the sequential 'train LLM, then EM tokens' regime)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    name = cfg.get_path("meta.name") or Path(args.config).stem
    seed = cfg.get("seed", 0)
    run_dir = Path(args.run_dir or f"experiments/{name}_seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- data ------------------------------------------------------------------------
    data_cfg = _data_subconfig(cfg)
    paths = build_corpus(data_cfg, data_root=args.data_root, force=args.force_data)
    cfg["data"]["pad_token_id"] = _pad_id(data_cfg.get("tokenizer", "gpt2"))

    train_ds = load_dataset_split(paths.root, "train")
    val_ds = load_dataset_split(paths.root, "val")
    tokenizer = build_tokenizer(data_cfg.get("tokenizer", "gpt2"))
    vocab = tokenizer_vocab_size(tokenizer)
    with open(paths.domains) as fh:
        n_clusters = int(json.load(fh)["n_clusters"])

    # --- model -----------------------------------------------------------------------
    centroids = None
    if cfg.get_path("model.expert_tokens.init") == "from_cluster_centroids":
        centroids = _load_centroids(paths)
    model = build_model(cfg, vocab_size=vocab, data_n_experts=n_clusters, centroids=centroids)

    # --- optional: load a pretrained backbone (sequential "train LLM, then EM tokens") ----
    init_from = args.init_backbone_from or cfg.get_path("model.init_backbone_from")
    if init_from:
        _load_backbone_weights(model, init_from)

    # --- train -----------------------------------------------------------------------
    trainer = EMTrainer(cfg, run_dir, device=args.device)
    trainer.fit(model, train_ds, val_ds)
    logger.info("[train] done -> %s", run_dir)

    if not args.no_eval:
        from softmoe.eval.harness import evaluate_run

        evaluate_run(run_dir, data_root=args.data_root, device=args.device)
    print(f"run dir: {run_dir}")
    return 0


def _pad_id(tokenizer_name: str) -> int:
    tok = build_tokenizer(tokenizer_name)
    pad = getattr(tok, "pad_token_id", None)
    return int(pad) if pad is not None else 0


def _load_backbone_weights(model, init_from: str) -> None:
    """Copy ``backbone.*`` weights from a trained run into ``model`` (sequential regime).

    ``init_from`` is a run dir (uses ``checkpoints/best.pt``) or a direct .pt path. Both Dense and
    SoftMoE store the LM under ``self.backbone``, so the backbone keys line up.
    """
    import torch

    path = Path(init_from)
    ckpt = path / "checkpoints" / "best.pt" if path.is_dir() else path
    if not ckpt.exists():
        raise FileNotFoundError(f"--init-backbone-from: no checkpoint at {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    bb = {k: v for k, v in state.items() if k.startswith("backbone.")}
    if not bb:
        raise ValueError(f"No 'backbone.*' weights found in {ckpt}.")
    missing, unexpected = model.load_state_dict(bb, strict=False)
    loaded = len(bb) - len([k for k in bb if k in unexpected])
    logger.info("[init] warm-started %d backbone tensors from %s", loaded, ckpt)


if __name__ == "__main__":
    raise SystemExit(main())
