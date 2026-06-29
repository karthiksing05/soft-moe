# Expert × domain cross-routing

### ours_sup

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | **32.6*** | 38.8 | 41.0 | 37.7 | 40.1 | 32.6 | 39.4 | 1.21× |
| news | **50.5*** | 59.4 | 67.3 | 68.3 | 67.8 | 50.5 | 65.7 | 1.30× |
| reviews | 41.7 | **38.6*** | 50.2 | 48.5 | 50.7 | 38.6 | 47.8 | 1.24× |
| arxiv | 36.9 | 44.6 | **30.3*** | 36.9 | 32.3 | 30.3 | 37.7 | 1.24× |
| pubmed | 31.1 | 35.4 | 29.1 | **25.1*** | 28.8 | 25.1 | 31.1 | 1.24× |

**Summary:** mean self-expert ppl 35.4 vs mean other-expert ppl 44.3 → **1.25× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### ours_unsup

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | **31.5*** | 39.2 | 40.3 | 41.4 | 40.3 | 31.5 | 40.3 | 1.28× |
| news | **49.7*** | 59.2 | 65.8 | 73.7 | 66.6 | 49.7 | 66.3 | 1.33× |
| reviews | 42.2 | **38.6*** | 48.7 | 54.4 | 48.3 | 38.6 | 48.4 | 1.25× |
| arxiv | 37.2 | 45.0 | **30.3*** | 36.1 | 33.0 | 30.3 | 37.8 | 1.25× |
| pubmed | 31.0 | 35.3 | 28.9 | **24.8*** | 29.9 | 24.8 | 31.3 | 1.26× |

**Summary:** mean self-expert ppl 35.0 vs mean other-expert ppl 44.8 → **1.28× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### ours_unsup_alt

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | **28.0*** | 29.0 | 29.4 | 29.2 | 29.4 | 28.0 | 29.2 | 1.05× |
| news | **35.6*** | 37.1 | 37.4 | 37.1 | 37.3 | 35.6 | 37.2 | 1.05× |
| reviews | 39.9 | **38.9*** | 41.3 | 41.0 | 41.1 | 38.9 | 40.8 | 1.05× |
| arxiv | 21.4 | 22.8 | **20.9*** | 21.7 | 21.1 | 20.9 | 21.8 | 1.04× |
| pubmed | 19.5 | 20.3 | 19.6 | **19.1*** | 19.7 | 19.1 | 19.8 | 1.04× |

**Summary:** mean self-expert ppl 28.5 vs mean other-expert ppl 29.8 → **1.05× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### cbtm

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|
| wiki | **31.0*** | 46.8 | 118.1 | 55.0 | 110.1 | 31.0 | 82.5 | 2.67× |
| news | **34.3*** | 136.9 | 639.5 | 342.5 | 619.5 | 34.3 | 434.6 | 12.68× |
| reviews | 56.5 | **40.2*** | 252.4 | 142.2 | 305.1 | 40.2 | 189.0 | 4.70× |
| arxiv | 31.3 | 92.4 | **23.4*** | 37.7 | 27.6 | 23.4 | 47.3 | 2.02× |
| pubmed | 33.0 | 80.5 | 39.5 | **20.3*** | 63.2 | 20.3 | 54.1 | 2.66× |

**Summary:** mean self-expert ppl 29.8 vs mean other-expert ppl 161.5 → **5.41× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.
