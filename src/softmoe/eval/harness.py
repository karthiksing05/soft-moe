"""Eval harness: run a trained model over the metric suite and write ``metrics.json``.

Rebuilds the model from the run's ``resolved_config.yaml``, loads ``checkpoints/best.pt``, then
computes every LM + specialization metric in [03]. ``make_report`` aggregates many such jsons.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from softmoe.data.build import CorpusPaths
from softmoe.data.dataset import build_tokenizer, load_dataset_split, tokenizer_vocab_size
from softmoe.eval.perplexity import collect_routing, per_domain_perplexity
from softmoe.eval.specialization import (
    contingency_matrix,
    routing_metrics,
    swap_test,
    token_separation,
    utilization_metrics,
)
from softmoe.models.factory import build_model
from softmoe.models.soft_moe import SoftMoE
from softmoe.utils.config import Config, load_config
from softmoe.utils.logging import get_logger

logger = get_logger()


def _load_run_config(run_dir: Path) -> Config:
    resolved = run_dir / "resolved_config.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"{resolved} not found — was this run trained?")
    import yaml

    with open(resolved) as fh:
        return Config(yaml.safe_load(fh))


def run_eval(model, processed_dir, cfg, device: str = "cpu") -> dict:
    pad = int(cfg.get_path("data.pad_token_id", 0) or 0)
    test = load_dataset_split(processed_dir, "test")
    bs = int(cfg.get_path("eval.batch_size", 8))
    method = cfg.get_path("model.method", "softmoe")

    metrics: dict = {"method": method, "regime": cfg.get_path("meta.regime", method)}

    # --- LM quality: learned + oracle routing ----------------------------------------
    learned = per_domain_perplexity(model, test, device, "learned", bs, pad)
    metrics["lm_learned"] = learned
    if isinstance(model, SoftMoE):
        oracle = per_domain_perplexity(model, test, device, "oracle", bs, pad)
        metrics["lm_oracle"] = oracle
        metrics["oracle_routed_gap"] = learned["macro_ppl"] - oracle["macro_ppl"]

    # --- specialization ---------------------------------------------------------------
    pred, dom, clu = collect_routing(model, test, device, bs, pad)
    n_experts = int(pred.max()) + 1 if len(pred) else 1
    if isinstance(model, SoftMoE):
        n_experts = model.tokens.n_experts
    metrics["routing_vs_domain"] = routing_metrics(pred, dom)
    metrics["utilization"] = utilization_metrics(pred, n_experts)
    metrics["contingency_expert_by_domain"] = contingency_matrix(
        pred, dom, n_experts, test.n_domains
    ).tolist()

    if isinstance(model, SoftMoE):
        metrics["token_separation"] = token_separation(model.tokens.embeddings)
        metrics["swap_test"] = swap_test(model, test, device, bs, pad)
        metrics["added_trainable_params"] = int(model.num_added_trainable_params())
    else:
        metrics["added_trainable_params"] = int(getattr(model, "num_added_trainable_params", lambda: 0)())

    metrics["total_params"] = int(sum(p.numel() for p in model.parameters()))
    return metrics


def load_run_model(run_dir: str | Path, data_root: str = "data"):
    """Rebuild a run's model from its resolved config and load ``checkpoints/best.pt``.

    Returns ``(model, cfg, CorpusPaths)``. Shared by ``evaluate_run`` and the cross-routing CLI.
    """
    run_dir = Path(run_dir)
    cfg = _load_run_config(run_dir)
    recipe = cfg.get_path("data.recipe")
    paths = CorpusPaths.for_recipe(Path(data_root), recipe)
    if not paths.meta.exists():
        raise FileNotFoundError(f"Processed data for recipe '{recipe}' not found at {paths.root}.")

    tokenizer = build_tokenizer(cfg.get_path("data.tokenizer", "gpt2"))
    vocab = tokenizer_vocab_size(tokenizer)
    with open(paths.domains) as fh:
        n_clusters = int(json.load(fh)["n_clusters"])

    model = build_model(cfg, vocab_size=vocab, data_n_experts=n_clusters)
    best = run_dir / "checkpoints" / "best.pt"
    if best.exists():
        state = torch.load(best, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        logger.info("[eval] loaded %s (step %s)", best, state.get("step"))
    else:
        logger.warning("[eval] no checkpoint at %s — using a randomly-initialized model.", best)
    return model, cfg, paths


def evaluate_run(run_dir: str | Path, data_root: str = "data", device: str | None = None) -> dict:
    run_dir = Path(run_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, paths = load_run_model(run_dir, data_root)
    metrics = run_eval(model, paths.root, cfg, device)
    out = run_dir / "metrics.json"
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("[eval] wrote %s", out)
    return metrics
