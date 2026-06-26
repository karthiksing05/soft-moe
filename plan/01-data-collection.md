# 01 — Data Collection

> Owns: `src/softmoe/data/`, `configs/data/`, `scripts/build_data.py`, `data/`.
> Goal: produce a **multi-domain, tokenized, sharded corpus** plus a **cluster assignment**
> for every document, so the EM loop has both ground-truth domain labels (supervised) and a
> clusterer-induced partition (unsupervised / Cobweb).

## 1. Why data is the crux here

The whole hypothesis is that *distinct sub-domains* can be hosted as expert tokens in one
backbone. So the corpus must have **clearly separable domains** with enough data per domain to
specialize, and **held-out splits per domain** so we can measure specialization, not just
average loss. We need three things per document:
1. token ids (for LM training),
2. a **ground-truth domain label** (for the supervised regime + for scoring routing accuracy),
3. a **cluster id** from an unsupervised clusterer (k-means and Cobweb) over document embeddings.

## 2. Datasets to use

Pick a small, medium, and large recipe. All are config-selectable.

| Recipe | Domains | Source | Use |
|--------|---------|--------|-----|
| `toy` | 3 synthetic/tiny domains (e.g. tiny slices of wikitext, a code snippet set, and a math/arithmetic set) | bundled / `datasets` streaming, ~1–5 MB each | CPU smoke tests, M1–M4 |
| `dev` | 4–6 domains from **M2D2** *or* hand-picked HF datasets: `wikipedia`, `github-code`/`codeparrot`, `pubmed_abstracts` (Pile subset), `legal` (e.g. `pile-of-law` slice), `arxiv`, `openwebtext` | HuggingFace `datasets` | iterate, M5–M6 |
| `main` | 8–16 domains, c-BTM-style. Mirror the **c-BTM** setup: clusters of **C4** and/or **S2ORC** | HF `datasets`, streamed + cached | headline run, M7 |

> **Anchor on c-BTM** ([Gururangan et al., 2023], *Scaling Expert LMs with Unsupervised Domain
> Discovery*) because it is the closest baseline: they k-means-cluster C4/S2ORC embeddings into
> domains and train one expert model per cluster. We reuse their clustering recipe but learn
> tokens in one model instead of N models. Keeping the corpus comparable makes the baseline fair.

Each domain config entry specifies: `name`, `hf_path`, `hf_config`, `split`, `text_field`,
`max_docs`, `min_chars`, and `weight` (sampling weight for the mixed stream).

## 3. Pipeline stages (`scripts/build_data.py` → `src/softmoe/data/build.py`)

Implement as a linear, resumable, cached pipeline. Each stage writes to `data/processed/<recipe>/`
and is skipped if its output exists (unless `--force`).

1. **Download / stream** (`download.py`)
   - Pull each domain via `datasets.load_dataset(..., streaming=True)`, take up to `max_docs`,
     drop docs shorter than `min_chars`, normalize whitespace.
   - Write per-domain `jsonl` shards: `{ "text": ..., "domain": <name> }`.

2. **Embed for clustering** (`clustering.py::embed`)
   - Embed each document with a cheap encoder: default `sentence-transformers/all-MiniLM-L6-v2`
     (mean-pooled). Provide a `tfidf` fallback (no GPU, for `toy`).
   - Cache embeddings as a memmap `embeddings.npy` aligned to a global doc index.

3. **Cluster** (`clustering.py`) — pluggable behind one interface:
   ```python
   class Clusterer(Protocol):
       def fit(self, X: np.ndarray) -> None: ...
       def predict(self, X: np.ndarray) -> np.ndarray: ...   # cluster id per doc
       n_clusters: int
   ```
   - `KMeansClusterer` (scikit-learn). `n_clusters` is a config knob; default = number of
     domains for `dev`, and 16–64 for `main` (c-BTM uses up to 128 — keep it configurable).
   - `CobwebClusterer` — wrap `concept_formation`'s Cobweb (or a local impl). Cobweb yields a
     **hierarchy**; expose both a flat cut (`predict` → leaf/level-k node id) and the path of
     node ids root→leaf (this enables the "hierarchical token path" idea in the brainstorm).
   - Persist the fitted clusterer (`clusterer.pkl`) so eval/inference reuse the same partition.

4. **Tokenize + shard** (`dataset.py::tokenize_and_shard`)
   - Tokenize with the **backbone's** tokenizer (config: `model.tokenizer`). Pack into fixed
     `block_size` sequences (default 1024) with document separators; record, per block, the
     **majority domain label** and **majority cluster id**.
   - Write Arrow/`datasets` shards or `.npy` token arrays + a parallel `meta.parquet` with
     columns `[block_idx, domain_id, cluster_id, cluster_path]`.

5. **Split** — deterministic per-domain `train/val/test` (e.g. 90/5/5) by hashing doc id, so
   every domain is represented in every split. Save split indices, not copies.

## 4. The Dataset / collator contract (`dataset.py`)

The trainer consumes batches shaped for the EM loop. Implement:
```python
class SoftMoEDataset(torch.utils.data.Dataset):
    # returns dict: input_ids[block], labels[block], domain_id, cluster_id, cluster_path
class MixedDomainSampler:
    # samples blocks honoring per-domain `weight`; supports a per-batch "balanced" mode
    # so the load-balance regularizer (see 02) is measured against a controlled prior.
```
The collator must expose `domain_id` and `cluster_id` tensors so the **E-step** can either use
them directly (supervised) or ignore them and route via the learned router (unsupervised),
while eval always has ground truth for scoring routing accuracy.

## 5. Config example (`configs/data/dev.yaml`)

```yaml
recipe: dev
block_size: 1024
tokenizer: gpt2
embedding_model: sentence-transformers/all-MiniLM-L6-v2
clusterer:
  kind: kmeans          # kmeans | cobweb
  n_clusters: 6
domains:
  - { name: wiki,    hf_path: wikipedia,      hf_config: 20220301.en, text_field: text, max_docs: 50000, weight: 1.0 }
  - { name: code,    hf_path: codeparrot/github-code, text_field: code, max_docs: 50000, weight: 1.0 }
  - { name: pubmed,  hf_path: pile-pubmed,    text_field: text, max_docs: 50000, weight: 1.0 }
  - { name: legal,   hf_path: pile-of-law,    hf_config: ...,   text_field: text, max_docs: 50000, weight: 1.0 }
  - { name: arxiv,   hf_path: arxiv_dataset,  text_field: abstract, max_docs: 50000, weight: 1.0 }
  - { name: web,     hf_path: openwebtext,    text_field: text, max_docs: 50000, weight: 1.0 }
splits: { train: 0.9, val: 0.05, test: 0.05 }
seed: 0
```

## 6. Deliverables / acceptance (Milestone M1)

- `python scripts/build_data.py --config configs/data/toy.yaml` runs end-to-end on CPU and
  writes `data/processed/toy/{train,val,test}` + `clusterer.pkl` + `meta.parquet`.
- `tests/test_data.py` asserts: shard token shapes, every domain present in every split,
  `cluster_id` and `domain_id` are dense integer ids, cluster count matches config.
- A one-page sanity report (printed by the script): docs/blocks per domain, cluster↔domain
  contingency table (so we can eyeball how well unsupervised clusters recover true domains —
  this *is* a result, log it).

## 7. Notes & gotchas for the agent

- Stream, don't download whole Pile/C4. Cap with `max_docs` and cache aggressively.
- Keep the **embedding** step independent of the **tokenizer** step — clustering uses sentence
  embeddings, training uses backbone tokens; do not conflate them.
- Cobweb is slow on large N; subsample for `fit`, then `predict` all. Log the subsample size.
- The cluster↔domain alignment (purity / NMI) collected here feeds directly into the
  unsupervised evaluation in [03](03-evaluation.md) — emit it as `data/processed/<recipe>/cluster_stats.json`.
