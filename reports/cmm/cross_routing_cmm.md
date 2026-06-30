# Expert × domain cross-routing

### ours_unsup_alt

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | 4.2 | 4.1 | 4.1 | **4.0*** | 4.1 | 4.0 | 4.1 | 1.04× |
| news | 4.9 | **4.5*** | 4.9 | 4.7 | 5.0 | 4.5 | 4.9 | 1.08× |
| reviews | 4.2 | 4.2 | 4.2 | 4.0 | **3.9*** | 3.9 | 4.2 | 1.07× |
| arxiv | **3.0*** | 3.7 | 3.1 | 3.5 | 3.7 | 3.0 | 3.5 | 1.15× |
| pubmed | 3.5 | 4.0 | **3.3*** | 3.7 | 4.2 | 3.3 | 3.8 | 1.16× |

**Summary:** mean self-expert ppl 3.8 vs mean other-expert ppl 4.1 → **1.09× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### ours_unsup_seq

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | 4.0 | 3.9 | 4.0 | **3.9*** | 4.0 | 3.9 | 4.0 | 1.02× |
| news | 4.6 | **4.5*** | 4.6 | 4.5 | 4.6 | 4.5 | 4.6 | 1.02× |
| reviews | 3.9 | 3.9 | 3.9 | 3.8 | **3.8*** | 3.8 | 3.9 | 1.02× |
| arxiv | **3.0*** | 3.3 | 3.0 | 3.2 | 3.3 | 3.0 | 3.2 | 1.06× |
| pubmed | 3.3 | 3.6 | **3.3*** | 3.5 | 3.7 | 3.3 | 3.5 | 1.08× |

**Summary:** mean self-expert ppl 3.7 vs mean other-expert ppl 3.8 → **1.04× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### ours_sup_alt

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | **3.9*** | 4.9 | 5.0 | 4.9 | 4.7 | 3.9 | 4.9 | 1.25× |
| news | 5.7 | **4.4*** | 5.2 | 6.6 | 6.3 | 4.4 | 6.0 | 1.36× |
| reviews | 4.9 | 4.3 | **3.8*** | 5.7 | 5.7 | 3.8 | 5.2 | 1.36× |
| arxiv | 3.9 | 4.6 | 4.5 | **3.0*** | 3.2 | 3.0 | 4.0 | 1.36× |
| pubmed | 3.9 | 4.8 | 4.9 | 3.6 | **3.2*** | 3.2 | 4.3 | 1.32× |

**Summary:** mean self-expert ppl 3.7 vs mean other-expert ppl 4.9 → **1.33× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### ours_sup_seq

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | **3.9*** | 4.3 | 4.3 | 4.0 | 4.0 | 3.9 | 4.1 | 1.06× |
| news | 4.5 | **4.5*** | 4.6 | 4.7 | 4.6 | 4.5 | 4.6 | 1.04× |
| reviews | 3.9 | 3.9 | **3.8*** | 4.0 | 3.9 | 3.8 | 3.9 | 1.04× |
| arxiv | 3.3 | 3.5 | 3.5 | **3.0*** | 3.1 | 3.0 | 3.3 | 1.12× |
| pubmed | 3.6 | 3.9 | 4.1 | 3.4 | **3.3*** | 3.3 | 3.7 | 1.15× |

**Summary:** mean self-expert ppl 3.7 vs mean other-expert ppl 3.9 → **1.08× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.
