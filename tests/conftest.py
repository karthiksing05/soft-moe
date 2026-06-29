"""Shared fixtures. Everything here is hermetic (offline): the synthetic data recipe + a tiny
GPT-2 built locally + the byte tokenizer. No network, no model downloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from softmoe.data.build import build_corpus
from softmoe.utils.config import Config


SYNTH_DATA = {
    "recipe": "synth",
    "block_size": 32,
    "tokenizer": "bytes",
    "embedding_method": "tfidf",
    "clusterer": {"kind": "kmeans", "n_clusters": 3},
    "domains": [
        {"name": "english", "source": "synthetic", "generator": "english", "max_docs": 60, "min_chars": 1},
        {"name": "code", "source": "synthetic", "generator": "code", "max_docs": 60, "min_chars": 1},
        {"name": "arithmetic", "source": "synthetic", "generator": "arithmetic", "max_docs": 60, "min_chars": 1},
    ],
    "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
    "seed": 0,
}


@pytest.fixture(scope="session")
def synth_corpus(tmp_path_factory) -> Path:
    data_root = tmp_path_factory.mktemp("data")
    paths = build_corpus(Config(SYNTH_DATA), data_root=data_root, force=True)
    return paths.root


@pytest.fixture(scope="session")
def data_root_with_synth(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dataroot")
    build_corpus(Config(SYNTH_DATA), data_root=root, force=True)
    return root


def tiny_experiment_cfg(recipe_data: dict) -> Config:
    """A minimal SoftMoE experiment config over the synth recipe + tiny backbone."""
    return Config(
        {
            "seed": 0,
            "meta": {"name": "test_ours", "regime": "ours_unsup"},
            "data": dict(recipe_data),
            "model": {
                "method": "softmoe",
                "backbone": {"kind": "tiny", "tiny": {"n_embd": 32, "n_layer": 2, "n_head": 2, "n_inner": 64, "n_positions": 128}},
                "backbone_mode": "frozen",
                "injection": "prefix",
                "expert_tokens": {"n_experts": "auto", "tokens_per_expert": 1, "init": "random", "trainable": True},
                "router": {"kind": "soft", "top_k": 1, "pool": "meanhidden"},
                "router_supervise_with": "cluster",
            },
            "train": {
                "max_steps": 6,
                "batch_size": 4,
                "eval_every": 6,
                "log_every": 3,
                "em": {"mode": "soft", "reassign_every": 0},
                "lambdas": {"sep": 1.0, "balance": 0.1, "route": 0.5},
                "optimizer": {"name": "adamw", "lr": 1.0e-3, "wd": 0.0},
                "scheduler": "cosine",
                "wandb": {"enabled": False},
            },
            "eval": {"batch_size": 4},
        }
    )
