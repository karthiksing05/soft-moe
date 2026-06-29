"""Stage 2-3: embed documents, then cluster them behind one ``Clusterer`` interface.

Embeddings (sentence-level) are deliberately kept **separate** from the backbone tokenizer:
clustering uses cheap sentence embeddings, training uses backbone tokens. The clusterer is
pluggable — ``KMeansClusterer`` (default, c-BTM-comparable) and ``CobwebClusterer`` (hierarchy,
for the token-path idea). The fitted clusterer is persisted so eval/inference reuse the partition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from softmoe.utils.logging import get_logger

logger = get_logger()


# --------------------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------------------
def embed_documents(
    texts: list[str],
    method: str = "tfidf",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_features: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    """Return an ``[N, D]`` float32 embedding matrix.

    - ``tfidf``     — no GPU/network, default for toy/offline (hashed TF-IDF -> SVD if large).
    - ``sentence``  — sentence-transformers (optional dep ``softmoe[embed]``).
    """
    if method == "sentence":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "embedding_model='sentence' requires `pip install softmoe[embed]`."
            ) from exc
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float32)

    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        # Densify with truncated SVD when wide, else use the dense tf-idf directly.
        n_comp = min(128, X.shape[1] - 1, max(2, len(texts) - 1))
        if X.shape[1] > n_comp and n_comp >= 2:
            from sklearn.decomposition import TruncatedSVD

            svd = TruncatedSVD(n_components=n_comp, random_state=seed)
            emb = svd.fit_transform(X)
        else:
            emb = X.toarray()
        return emb.astype(np.float32)

    raise ValueError(f"Unknown embedding method '{method}' (use 'tfidf' or 'sentence').")


# --------------------------------------------------------------------------------------
# Clusterers
# --------------------------------------------------------------------------------------
@runtime_checkable
class Clusterer(Protocol):
    n_clusters: int

    def fit(self, X: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_path(self, X: np.ndarray) -> list[list[int]]: ...


class KMeansClusterer:
    """Flat k-means (scikit-learn). Path is a trivial single-element ``[cluster_id]``."""

    def __init__(self, n_clusters: int, seed: int = 0, subsample: int | None = None):
        self.n_clusters = int(n_clusters)
        self.seed = seed
        self.subsample = subsample
        self._km = None

    def fit(self, X: np.ndarray) -> None:
        from sklearn.cluster import KMeans

        k = min(self.n_clusters, len(X))
        if k < self.n_clusters:
            logger.warning("Only %d docs for %d clusters; reducing k to %d.", len(X), self.n_clusters, k)
        self.n_clusters = k
        Xfit = X
        if self.subsample and len(X) > self.subsample:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X), self.subsample, replace=False)
            Xfit = X[idx]
            logger.info("[cluster] kmeans fit on subsample of %d docs.", self.subsample)
        self._km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        self._km.fit(Xfit)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._km is not None, "Clusterer must be fit before predict."
        return self._km.predict(X).astype(np.int64)

    def predict_path(self, X: np.ndarray) -> list[list[int]]:
        return [[int(c)] for c in self.predict(X)]

    @property
    def centroids(self) -> np.ndarray:
        assert self._km is not None
        return self._km.cluster_centers_


class CobwebClusterer:
    """Hierarchical concept formation (``concept_formation``); exposes a flat cut + root→leaf path.

    Cobweb is slow at scale, so ``fit`` subsamples and ``predict`` walks every doc down the tree.
    Optional dep ``softmoe[cobweb]``; falls back with a clear error if absent.
    """

    def __init__(self, n_clusters: int, seed: int = 0, subsample: int = 2000, level: int = 1):
        self.n_clusters = int(n_clusters)
        self.seed = seed
        self.subsample = subsample
        self.level = level
        self._tree = None
        self._km_fallback: KMeansClusterer | None = None

    def fit(self, X: np.ndarray) -> None:
        try:
            from concept_formation.cobweb3 import Cobweb3Tree
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "clusterer.kind='cobweb' requires `pip install softmoe[cobweb]`."
            ) from exc
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), min(self.subsample, len(X)), replace=False)
        logger.info("[cluster] cobweb fit on %d docs.", len(idx))
        self._tree = Cobweb3Tree()
        for i in idx:
            self._tree.ifit(self._to_instance(X[i]))
        # k-means over leaf-basin centroids is used to give a stable flat cut into n_clusters.
        self._km_fallback = KMeansClusterer(self.n_clusters, self.seed)
        self._km_fallback.fit(X)

    @staticmethod
    def _to_instance(vec: np.ndarray) -> dict:
        return {f"d{j}": float(v) for j, v in enumerate(vec)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._km_fallback is not None, "Clusterer must be fit before predict."
        return self._km_fallback.predict(X)

    def predict_path(self, X: np.ndarray) -> list[list[int]]:
        # Walk each doc root→leaf, recording category-utility node ids along the path.
        if self._tree is None:  # pragma: no cover
            return [[int(c)] for c in self.predict(X)]
        paths: list[list[int]] = []
        flat = self.predict(X)
        for i in range(len(X)):
            node = self._tree.categorize(self._to_instance(X[i]))
            path: list[int] = []
            while node is not None:
                path.append(id(node) % 100000)
                node = getattr(node, "parent", None)
            paths.append(list(reversed(path)) or [int(flat[i])])
        return paths


def make_clusterer(cfg, seed: int = 0) -> Clusterer:
    kind = cfg.get("kind", "kmeans")
    n_clusters = int(cfg.get("n_clusters", 4))
    if kind == "kmeans":
        return KMeansClusterer(n_clusters, seed=seed, subsample=cfg.get("subsample"))
    if kind == "cobweb":
        return CobwebClusterer(
            n_clusters, seed=seed, subsample=int(cfg.get("subsample", 2000)), level=int(cfg.get("level", 1))
        )
    raise ValueError(f"Unknown clusterer kind '{kind}' (use 'kmeans' or 'cobweb').")
