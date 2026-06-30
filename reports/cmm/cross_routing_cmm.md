# Expert × domain cross-routing

### ours_sup_alt

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | **4.1*** | 5.0 | 5.1 | 5.5 | 5.2 | 4.1 | 5.2 | 1.27× |
| news | 5.9 | **4.7*** | 5.5 | 7.6 | 7.2 | 4.7 | 6.5 | 1.39× |
| reviews | 5.0 | 4.5 | **3.9*** | 6.2 | 6.2 | 3.9 | 5.5 | 1.39× |
| arxiv | 4.3 | 4.7 | 4.7 | **3.1*** | 3.4 | 3.1 | 4.3 | 1.38× |
| pubmed | 4.2 | 4.9 | 5.0 | 3.9 | **3.4*** | 3.4 | 4.5 | 1.31× |

**Summary:** mean self-expert ppl 3.8 vs mean other-expert ppl 5.2 → **1.35× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### ours_unsup_alt

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | 4.3 | 4.3 | 4.3 | **4.1*** | 4.3 | 4.1 | 4.3 | 1.04× |
| news | 5.1 | **4.8*** | 5.1 | 4.9 | 5.2 | 4.8 | 5.1 | 1.07× |
| reviews | 4.4 | 4.3 | 4.4 | 4.1 | **4.0*** | 4.0 | 4.3 | 1.08× |
| arxiv | **3.1*** | 3.7 | 3.2 | 3.5 | 3.8 | 3.1 | 3.6 | 1.14× |
| pubmed | 3.6 | 4.1 | **3.5*** | 3.9 | 4.3 | 3.5 | 3.9 | 1.14× |

**Summary:** mean self-expert ppl 3.9 vs mean other-expert ppl 4.2 → **1.09× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### ours_unsup_alt20k

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | 4.0 | 4.0 | 4.0 | **3.9*** | 4.1 | 3.9 | 4.1 | 1.04× |
| news | 4.9 | **4.5*** | 4.8 | 4.7 | 4.9 | 4.5 | 4.8 | 1.08× |
| reviews | 4.2 | 4.1 | 4.2 | 3.9 | **3.8*** | 3.8 | 4.1 | 1.07× |
| arxiv | **3.0*** | 3.4 | 3.1 | 3.4 | 3.6 | 3.0 | 3.4 | 1.13× |
| pubmed | 3.4 | 3.8 | **3.2*** | 3.6 | 4.1 | 3.2 | 3.7 | 1.15× |

**Summary:** mean self-expert ppl 3.7 vs mean other-expert ppl 4.0 → **1.09× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.
