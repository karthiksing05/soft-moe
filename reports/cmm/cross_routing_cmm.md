# Expert × domain cross-routing

### ours_alt

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | 4.2 | 4.1 | 4.2 | **4.0*** | 4.2 | 4.0 | 4.2 | 1.04× |
| news | 4.9 | **4.5*** | 4.9 | 4.7 | 5.0 | 4.5 | 4.9 | 1.08× |
| reviews | 4.2 | 4.2 | 4.2 | 4.0 | **3.9*** | 3.9 | 4.1 | 1.07× |
| arxiv | **3.0*** | 3.6 | 3.1 | 3.5 | 3.7 | 3.0 | 3.5 | 1.15× |
| pubmed | 3.5 | 3.9 | **3.3*** | 3.8 | 4.2 | 3.3 | 3.8 | 1.15× |

**Summary:** mean self-expert ppl 3.8 vs mean other-expert ppl 4.1 → **1.09× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### ours_seq

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | 4.0 | 4.0 | 4.0 | **4.0*** | 4.0 | 4.0 | 4.0 | 1.01× |
| news | 4.6 | 4.6* | 4.6 | **4.6** | 4.6 | 4.6 | 4.6 | 1.00× |
| reviews | 3.9 | 3.8 | 3.9 | 3.8 | **3.8*** | 3.8 | 3.9 | 1.01× |
| arxiv | **3.0*** | 3.1 | 3.0 | 3.1 | 3.1 | 3.0 | 3.1 | 1.02× |
| pubmed | 3.3 | 3.4 | **3.3*** | 3.4 | 3.4 | 3.3 | 3.4 | 1.02× |

**Summary:** mean self-expert ppl 3.7 vs mean other-expert ppl 3.8 → **1.01× worse** through the wrong expert; lowest-ppl expert is the matched one for **80%** of domains.
