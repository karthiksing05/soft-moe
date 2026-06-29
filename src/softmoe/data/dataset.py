"""Stage 4-5 + the trainer-facing Dataset/collator/sampler contract.

Tokenization packs documents into fixed ``block_size`` blocks; each block records its majority
``domain_id`` and ``cluster_id`` (and ``cluster_path``). The Dataset returns everything the EM
loop needs, so the E-step can either use the labels directly (supervised) or ignore them and
route via the learned router (unsupervised), while eval always has ground truth for scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# --------------------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------------------
class ByteTokenizer:
    """A trivial, fully-offline byte-level tokenizer (vocab 256 + specials).

    Used for hermetic tests and the offline ``synth`` recipe so nothing depends on a network
    download. Mirrors the slice of the HF tokenizer API the pipeline needs.
    """

    def __init__(self):
        self.eos_token_id = 256
        self.pad_token_id = 256
        self.bos_token_id = 256
        self.vocab_size = 257

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def __call__(self, text: str):
        return {"input_ids": self.encode(text)}

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")

    def __len__(self) -> int:
        return self.vocab_size


def build_tokenizer(name: str):
    """Build a tokenizer: ``bytes`` (offline) or any HF tokenizer name."""
    if name == "bytes":
        return ByteTokenizer()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenizer_vocab_size(tokenizer) -> int:
    if isinstance(tokenizer, ByteTokenizer):
        return tokenizer.vocab_size
    return len(tokenizer)


def _eos_id(tokenizer) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    return int(eos) if eos is not None else 0


# --------------------------------------------------------------------------------------
# Tokenize + pack into blocks
# --------------------------------------------------------------------------------------
def tokenize_and_shard(
    docs: list[dict],
    tokenizer,
    block_size: int,
    domain_to_id: dict[str, int],
    cluster_ids: np.ndarray,
    cluster_paths: list[list[int]],
) -> dict[str, np.ndarray | list]:
    """Pack tokenized docs into ``block_size`` blocks with per-block majority labels.

    ``docs[i]`` aligns with ``cluster_ids[i]`` / ``cluster_paths[i]`` (global doc index). Returns
    arrays for tokens + per-block ``domain_id``/``cluster_id`` plus the per-block ``cluster_path``.
    """
    eos = _eos_id(tokenizer)
    token_blocks: list[list[int]] = []
    block_domains: list[int] = []
    block_clusters: list[int] = []
    block_paths: list[list[int]] = []

    buf: list[int] = []
    buf_domain: list[int] = []
    buf_cluster: list[int] = []
    buf_path: list[list[int]] = []

    def flush_block():
        # Take exactly block_size tokens; majority labels over the contributing docs.
        chunk = buf[:block_size]
        dom = _majority(buf_domain[:block_size])
        clu = _majority(buf_cluster[:block_size])
        # path of the majority cluster among the block's tokens
        path = buf_path[0] if buf_path else [clu]
        token_blocks.append(chunk)
        block_domains.append(dom)
        block_clusters.append(clu)
        block_paths.append(path)

    for i, doc in enumerate(docs):
        ids = tokenizer.encode(doc["text"])
        ids = ids + [eos]
        dom_id = domain_to_id[doc["domain"]]
        clu_id = int(cluster_ids[i])
        path = [int(x) for x in cluster_paths[i]]
        for tid in ids:
            buf.append(tid)
            buf_domain.append(dom_id)
            buf_cluster.append(clu_id)
            buf_path.append(path)
            if len(buf) >= block_size:
                flush_block()
                buf = buf[block_size:]
                buf_domain = buf_domain[block_size:]
                buf_cluster = buf_cluster[block_size:]
                buf_path = buf_path[block_size:]
    # tail: pad the final partial block so no domain is silently dropped on tiny corpora
    if buf:
        pad_id = getattr(tokenizer, "pad_token_id", eos) or eos
        while len(buf) < block_size:
            buf.append(pad_id)
            buf_domain.append(buf_domain[-1] if buf_domain else 0)
            buf_cluster.append(buf_cluster[-1] if buf_cluster else 0)
            buf_path.append(buf_path[-1] if buf_path else [0])
        flush_block()

    return {
        "tokens": np.asarray(token_blocks, dtype=np.int64),
        "domain_id": np.asarray(block_domains, dtype=np.int64),
        "cluster_id": np.asarray(block_clusters, dtype=np.int64),
        "cluster_path": block_paths,
    }


def _majority(xs: list[int]) -> int:
    if not xs:
        return 0
    vals, counts = np.unique(np.asarray(xs), return_counts=True)
    return int(vals[int(np.argmax(counts))])


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
def make_splits(
    domain_ids: np.ndarray, fractions: dict[str, float], seed: int
) -> np.ndarray:
    """Deterministic per-domain train/val/test split; every domain appears in every split.

    Returns an int array (0=train,1=val,2=test) aligned to blocks.
    """
    rng = np.random.default_rng(seed)
    split = np.zeros(len(domain_ids), dtype=np.int64)
    train_f = fractions.get("train", 0.9)
    val_f = fractions.get("val", 0.05)
    for d in np.unique(domain_ids):
        idx = np.where(domain_ids == d)[0]
        perm = rng.permutation(idx)
        n = len(perm)
        n_train = max(1, int(round(train_f * n)))
        n_val = max(1, int(round(val_f * n))) if n - n_train >= 2 else 0
        # guarantee at least one test block per domain when possible
        if n - n_train - n_val < 1 and n - n_train >= 1:
            n_val = max(0, n - n_train - 1)
        split[perm[:n_train]] = 0
        split[perm[n_train : n_train + n_val]] = 1
        split[perm[n_train + n_val :]] = 2
    return split


# --------------------------------------------------------------------------------------
# Dataset / collator / sampler
# --------------------------------------------------------------------------------------
class SoftMoEDataset(Dataset):
    """Returns ``{input_ids, labels, domain_id, cluster_id, cluster_path}`` per block."""

    def __init__(self, processed_dir: str | Path, split: str = "train"):
        self.dir = Path(processed_dir)
        self.split = split
        split_id = {"train": 0, "val": 1, "test": 2}[split]
        meta = _load_meta(self.dir)
        self.tokens = np.load(self.dir / "tokens.npy", mmap_mode="r")
        sel = np.where(meta["split"].to_numpy() == split_id)[0]
        self.index = sel
        self.domain_id = meta["domain_id"].to_numpy()
        self.cluster_id = meta["cluster_id"].to_numpy()
        self.paths = meta["cluster_path"].tolist()
        self.n_domains = int(meta["domain_id"].max()) + 1
        self.n_clusters = int(meta["cluster_id"].max()) + 1

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        b = int(self.index[i])
        ids = torch.from_numpy(np.array(self.tokens[b], dtype=np.int64))
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "domain_id": int(self.domain_id[b]),
            "cluster_id": int(self.cluster_id[b]),
            "cluster_path": self._parse_path(self.paths[b]),
        }

    @staticmethod
    def _parse_path(p) -> list[int]:
        if isinstance(p, str):
            return [int(x) for x in json.loads(p)]
        if isinstance(p, (list, np.ndarray)):
            return [int(x) for x in p]
        return [int(p)]


@dataclass
class Collator:
    pad_token_id: int = 0

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = (input_ids != self.pad_token_id).long()
        # never mask out everything; keep at least causal LM well-defined
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "domain_id": torch.tensor([b["domain_id"] for b in batch], dtype=torch.long),
            "cluster_id": torch.tensor([b["cluster_id"] for b in batch], dtype=torch.long),
            "cluster_path": [b["cluster_path"] for b in batch],
        }


class MixedDomainSampler(Sampler[int]):
    """Sample blocks honoring per-domain ``weight``; ``balanced`` forces uniform domain usage.

    The balanced mode lets the load-balance regularizer (training/02) be measured against a
    controlled prior rather than the natural domain frequencies.
    """

    def __init__(
        self,
        domain_ids: np.ndarray,
        weights: dict[int, float] | None = None,
        balanced: bool = False,
        num_samples: int | None = None,
        seed: int = 0,
    ):
        self.domain_ids = np.asarray(domain_ids)
        self.unique = np.unique(self.domain_ids)
        self.balanced = balanced
        self.num_samples = num_samples or len(self.domain_ids)
        self.rng = np.random.default_rng(seed)
        if balanced or weights is None:
            self.weights = {int(d): 1.0 for d in self.unique}
        else:
            self.weights = {int(d): float(weights.get(int(d), 1.0)) for d in self.unique}
        self._by_domain = {int(d): np.where(self.domain_ids == d)[0] for d in self.unique}

    def __iter__(self) -> Iterator[int]:
        probs = np.array([self.weights[int(d)] for d in self.unique], dtype=np.float64)
        probs = probs / probs.sum()
        for _ in range(self.num_samples):
            d = self.rng.choice(self.unique, p=probs)
            pool = self._by_domain[int(d)]
            yield int(self.rng.choice(pool))

    def __len__(self) -> int:
        return self.num_samples


def load_dataset_split(processed_dir: str | Path, split: str) -> SoftMoEDataset:
    return SoftMoEDataset(processed_dir, split)


def _load_meta(processed_dir: Path):
    import pandas as pd

    return pd.read_parquet(processed_dir / "meta.parquet")
