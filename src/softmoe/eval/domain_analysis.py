"""Domain-separability analysis for a built corpus.

The whole hypothesis needs *distinct, relevant* domains (THEORY §E.5). This module quantifies
that for a processed corpus and renders a thorough markdown report, so "the data really has
separable domains" is evidence, not an assertion. Metrics:

- per-domain document/block counts and mean length,
- **silhouette score** of the document embeddings under the ground-truth domain labels
  (how cleanly domains separate in embedding space; >0.1 meaningful, >0.25 strong),
- inter-domain **centroid cosine-distance** matrix (which domains are near/far),
- clusterer↔domain agreement (NMI / ARI / purity + contingency), i.e. can an unsupervised
  k-means recover the domains — the premise the unsupervised expert router relies on,
- a couple of **sample documents per domain** (so a human can eyeball the registers differ).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from softmoe.utils.logging import get_logger

logger = get_logger()


def _per_doc_domain_labels(processed_dir: Path, domain_to_id: dict[str, int]) -> tuple[np.ndarray, dict]:
    """Reconstruct per-document domain labels aligned to embeddings.npy (built in domain order)."""
    labels: list[int] = []
    samples: dict[str, list[str]] = {}
    lengths: dict[str, list[int]] = {}
    for name, did in domain_to_id.items():
        path = processed_dir / "raw" / f"{name}.jsonl"
        n = 0
        samples[name] = []
        lengths[name] = []
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)
                labels.append(did)
                lengths[name].append(len(doc["text"]))
                if len(samples[name]) < 2:
                    samples[name].append(doc["text"][:240].replace("\n", " "))
                n += 1
    return np.asarray(labels), {"samples": samples, "lengths": lengths}


def analyze_domains(processed_dir: str | Path, max_silhouette: int = 6000, seed: int = 0) -> dict:
    processed_dir = Path(processed_dir)
    emb = np.load(processed_dir / "embeddings.npy")
    with open(processed_dir / "domains.json") as fh:
        info = json.load(fh)
    domain_to_id = info["domain_to_id"]
    id_to_name = {v: k for k, v in domain_to_id.items()}

    labels, extra = _per_doc_domain_labels(processed_dir, domain_to_id)
    if len(labels) != len(emb):
        logger.warning("doc labels (%d) != embeddings (%d); truncating to min.", len(labels), len(emb))
        m = min(len(labels), len(emb))
        labels, emb = labels[:m], emb[:m]

    # --- silhouette (subsampled for tractability) ---
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    idx = np.arange(len(emb))
    if len(emb) > max_silhouette:
        idx = rng.choice(len(emb), max_silhouette, replace=False)
    try:
        sil = float(silhouette_score(emb[idx], labels[idx]))
    except ValueError:
        sil = float("nan")

    # --- inter-domain centroid cosine-distance matrix ---
    names = [id_to_name[i] for i in range(len(domain_to_id))]
    cents = np.stack([emb[labels == i].mean(0) for i in range(len(names))])
    cn = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
    cos_dist = (1.0 - cn @ cn.T)

    # --- clusterer vs domain agreement (re-uses the built cluster ids) ---
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    meta = pd.read_parquet(processed_dir / "meta.parquet")
    dom_b, clu_b = meta["domain_id"].to_numpy(), meta["cluster_id"].to_numpy()
    n_clu = int(clu_b.max()) + 1
    contingency = np.zeros((n_clu, len(names)), dtype=int)
    for c, d in zip(clu_b, dom_b):
        contingency[c, d] += 1
    purity = float(contingency.max(axis=1).sum() / max(1, contingency.sum()))

    # --- per-domain counts ---
    per_domain = {}
    for i, name in enumerate(names):
        n_docs = int((labels == i).sum())
        n_blocks = int((dom_b == i).sum())
        mean_chars = float(np.mean(extra["lengths"][name])) if extra["lengths"][name] else 0.0
        per_domain[name] = {"docs": n_docs, "blocks": n_blocks, "mean_chars": round(mean_chars, 1)}

    return {
        "recipe": processed_dir.name,
        "n_domains": len(names),
        "domain_names": names,
        "n_docs": int(len(labels)),
        "n_blocks": int(len(meta)),
        "silhouette_domains": sil,
        "silhouette_n_sampled": int(len(idx)),
        "centroid_cosine_distance": cos_dist.round(3).tolist(),
        "clusterer_vs_domain": {
            "nmi": float(normalized_mutual_info_score(dom_b, clu_b)),
            "ari": float(adjusted_rand_score(dom_b, clu_b)),
            "purity": purity,
            "n_clusters": n_clu,
            "contingency_cluster_by_domain": contingency.tolist(),
        },
        "per_domain": per_domain,
        "samples": extra["samples"],
    }


def render_markdown(report: dict) -> str:
    names = report["domain_names"]
    L = [
        f"# Domain-separability report — `{report['recipe']}` corpus",
        "",
        f"- **Domains:** {report['n_domains']} ({', '.join(names)})",
        f"- **Documents:** {report['n_docs']:,}  |  **Blocks:** {report['n_blocks']:,}",
        f"- **Silhouette (domain labels, n={report['silhouette_n_sampled']}):** "
        f"**{report['silhouette_domains']:.3f}**  "
        f"_(>0.1 meaningful, >0.25 strong separation in embedding space)_",
        "",
        "## Per-domain volume",
        "",
        "| domain | docs | blocks | mean chars |",
        "|---|---|---|---|",
    ]
    for name in names:
        d = report["per_domain"][name]
        L.append(f"| {name} | {d['docs']:,} | {d['blocks']:,} | {d['mean_chars']:.0f} |")

    L += ["", "## Unsupervised recovery (k-means clusters vs true domains)", ""]
    cd = report["clusterer_vs_domain"]
    L.append(f"NMI **{cd['nmi']:.3f}** · ARI **{cd['ari']:.3f}** · purity **{cd['purity']:.3f}** "
             f"({cd['n_clusters']} clusters). A high score means an unsupervised clusterer recovers the "
             "domains — the premise the unsupervised expert router exploits.")
    L += ["", "Contingency `[cluster × domain]` (rows=clusters, cols=" + ", ".join(names) + "):", "", "```"]
    for c, row in enumerate(cd["contingency_cluster_by_domain"]):
        L.append(f"c{c}: {row}")
    L += ["```", "", "## Inter-domain distance (centroid cosine distance)", "",
          "Larger = more distinct. `0`=identical direction, `1`=orthogonal.", "",
          "| | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for i, name in enumerate(names):
        L.append(f"| **{name}** | " + " | ".join(f"{v:.2f}" for v in report["centroid_cosine_distance"][i]) + " |")

    L += ["", "## Sample documents per domain", ""]
    for name in names:
        L.append(f"**{name}**")
        for s in report["samples"][name]:
            L.append(f"> {s}…")
        L.append("")
    return "\n".join(L)
