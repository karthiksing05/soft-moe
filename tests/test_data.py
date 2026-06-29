"""M1 acceptance: corpus shapes, every domain in every split, dense ids, cluster count."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from softmoe.data.dataset import SoftMoEDataset


def test_build_outputs_exist(synth_corpus):
    for f in ["tokens.npy", "meta.parquet", "clusterer.pkl", "domains.json", "cluster_stats.json"]:
        assert (synth_corpus / f).exists(), f"missing {f}"


def test_token_shapes(synth_corpus):
    tokens = np.load(synth_corpus / "tokens.npy")
    assert tokens.ndim == 2 and tokens.shape[1] == 32
    assert tokens.dtype == np.int64


def test_every_domain_in_every_split(synth_corpus):
    meta = pd.read_parquet(synth_corpus / "meta.parquet")
    domains = sorted(meta["domain_id"].unique())
    for split_id in (0, 1, 2):
        present = sorted(meta[meta["split"] == split_id]["domain_id"].unique())
        assert present == domains, f"split {split_id} missing domains: {set(domains) - set(present)}"


def test_dense_integer_ids(synth_corpus):
    meta = pd.read_parquet(synth_corpus / "meta.parquet")
    for col in ("domain_id", "cluster_id"):
        vals = sorted(meta[col].unique())
        assert vals == list(range(len(vals))), f"{col} not dense: {vals}"


def test_cluster_count_matches_config(synth_corpus):
    with open(synth_corpus / "domains.json") as fh:
        info = json.load(fh)
    assert info["n_clusters"] == 3
    assert set(info["domain_to_id"]) == {"english", "code", "arithmetic"}


def test_dataset_contract(synth_corpus):
    ds = SoftMoEDataset(synth_corpus, "train")
    item = ds[0]
    assert set(item) >= {"input_ids", "labels", "domain_id", "cluster_id", "cluster_path"}
    assert item["input_ids"].shape == item["labels"].shape == (32,)
    assert isinstance(item["cluster_path"], list)


def test_cluster_stats_present(synth_corpus):
    with open(synth_corpus / "cluster_stats.json") as fh:
        stats = json.load(fh)
    assert "nmi" in stats and "contingency_cluster_by_domain" in stats
    assert len(stats["contingency_cluster_by_domain"]) == stats["n_clusters"]
