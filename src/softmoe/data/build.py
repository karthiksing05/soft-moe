"""End-to-end data build: raw -> embeddings -> clusters -> tokenized/sharded corpus + splits.

Linear, resumable, cached. Each stage writes under ``data/processed/<recipe>/`` and is skipped
if its output exists (unless ``force``). Emits a sanity report and ``cluster_stats.json``
(cluster↔domain purity/NMI + contingency) — itself a result that feeds the unsupervised eval.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from softmoe.data.clustering import embed_documents, make_clusterer
from softmoe.data.dataset import (
    build_tokenizer,
    make_splits,
    tokenize_and_shard,
    tokenizer_vocab_size,
)
from softmoe.data.download import download_domain
from softmoe.utils.config import Config
from softmoe.utils.logging import get_logger
from softmoe.utils.seeding import seed_everything

logger = get_logger()


@dataclass
class CorpusPaths:
    root: Path
    raw_dir: Path
    tokens: Path
    meta: Path
    clusterer: Path
    domains: Path
    cluster_stats: Path

    @classmethod
    def for_recipe(cls, data_root: Path, recipe: str) -> "CorpusPaths":
        root = data_root / "processed" / recipe
        return cls(
            root=root,
            raw_dir=root / "raw",
            tokens=root / "tokens.npy",
            meta=root / "meta.parquet",
            clusterer=root / "clusterer.pkl",
            domains=root / "domains.json",
            cluster_stats=root / "cluster_stats.json",
        )


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_corpus(cfg: Config, data_root: str | Path = "data", force: bool = False) -> CorpusPaths:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    recipe = cfg["recipe"]
    paths = CorpusPaths.for_recipe(Path(data_root), recipe)
    paths.root.mkdir(parents=True, exist_ok=True)

    if paths.meta.exists() and not force:
        logger.info("[build] %s already built at %s (use force=True to rebuild).", recipe, paths.root)
        return paths

    # --- Stage 1: download/generate raw docs -----------------------------------------
    domains = cfg["domains"]
    domain_names = [d["name"] for d in domains]
    domain_to_id = {name: i for i, name in enumerate(domain_names)}
    all_docs: list[dict] = []
    for d in domains:
        jsonl = download_domain(d, paths.raw_dir, seed=seed, force=force)
        all_docs.extend(_read_jsonl(jsonl))
    logger.info("[build] %d total docs across %d domains.", len(all_docs), len(domains))

    # --- Stage 2: embed ----------------------------------------------------------------
    texts = [d["text"] for d in all_docs]
    emb_method = cfg.get("embedding_method", "tfidf")
    emb = embed_documents(
        texts,
        method=emb_method,
        model_name=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        seed=seed,
    )
    np.save(paths.root / "embeddings.npy", emb)

    # --- Stage 3: cluster --------------------------------------------------------------
    clusterer = make_clusterer(cfg.get("clusterer", {"kind": "kmeans", "n_clusters": len(domains)}), seed=seed)
    clusterer.fit(emb)
    cluster_ids = clusterer.predict(emb)
    cluster_paths = clusterer.predict_path(emb)
    with open(paths.clusterer, "wb") as fh:
        pickle.dump(clusterer, fh)

    # --- Stage 4: tokenize + shard -----------------------------------------------------
    tokenizer = build_tokenizer(cfg.get("tokenizer", "gpt2"))
    block_size = int(cfg.get("block_size", 1024))
    sharded = tokenize_and_shard(
        all_docs, tokenizer, block_size, domain_to_id, cluster_ids, cluster_paths
    )
    tokens = sharded["tokens"]
    if len(tokens) == 0:
        raise RuntimeError("Tokenization produced 0 blocks; reduce block_size or add data.")

    # --- Stage 5: split ----------------------------------------------------------------
    split = make_splits(sharded["domain_id"], cfg.get("splits", {"train": 0.9, "val": 0.05, "test": 0.05}), seed)

    np.save(paths.tokens, tokens)
    _write_meta(paths.meta, sharded, split)
    with open(paths.domains, "w") as fh:
        json.dump({"domain_to_id": domain_to_id, "n_clusters": int(clusterer.n_clusters)}, fh, indent=2)

    # --- Report + cluster stats --------------------------------------------------------
    stats = _cluster_stats(sharded["domain_id"], sharded["cluster_id"], domain_names)
    with open(paths.cluster_stats, "w") as fh:
        json.dump(stats, fh, indent=2)
    _print_report(recipe, sharded, split, domain_names, tokenizer, block_size, stats)
    return paths


def _write_meta(meta_path: Path, sharded: dict, split: np.ndarray) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "block_idx": np.arange(len(sharded["domain_id"])),
            "domain_id": sharded["domain_id"],
            "cluster_id": sharded["cluster_id"],
            "cluster_path": [json.dumps(p) for p in sharded["cluster_path"]],
            "split": split,
        }
    )
    df.to_parquet(meta_path, index=False)


def _cluster_stats(domain_id: np.ndarray, cluster_id: np.ndarray, domain_names: list[str]) -> dict:
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    n_dom = len(domain_names)
    n_clu = int(cluster_id.max()) + 1
    contingency = np.zeros((n_clu, n_dom), dtype=int)
    for c, d in zip(cluster_id, domain_id):
        contingency[c, d] += 1
    purity = float(contingency.max(axis=1).sum() / max(1, contingency.sum()))
    return {
        "nmi": float(normalized_mutual_info_score(domain_id, cluster_id)),
        "ari": float(adjusted_rand_score(domain_id, cluster_id)),
        "purity": purity,
        "n_domains": n_dom,
        "n_clusters": n_clu,
        "domain_names": domain_names,
        "contingency_cluster_by_domain": contingency.tolist(),
    }


def _print_report(recipe, sharded, split, domain_names, tokenizer, block_size, stats) -> None:
    dom = sharded["domain_id"]
    lines = [
        f"\n===== data build report: {recipe} =====",
        f"tokenizer vocab: {tokenizer_vocab_size(tokenizer)}  block_size: {block_size}",
        f"total blocks: {len(dom)}  (train/val/test = "
        f"{int((split==0).sum())}/{int((split==1).sum())}/{int((split==2).sum())})",
        "blocks per domain:",
    ]
    for i, name in enumerate(domain_names):
        lines.append(f"  {name:>12}: {int((dom == i).sum())}")
    lines.append(f"cluster↔domain  NMI={stats['nmi']:.3f}  ARI={stats['ari']:.3f}  purity={stats['purity']:.3f}")
    lines.append("contingency [cluster x domain]:")
    for c, row in enumerate(stats["contingency_cluster_by_domain"]):
        lines.append(f"  c{c}: {row}")
    lines.append("=" * 40)
    logger.info("\n".join(lines))
