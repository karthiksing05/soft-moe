# Domain routing analysis (every model)

Per-model expert↔domain specialization: our expert-token methods show an expert×domain **perplexity** matrix (swap test); the MoE arm shows its learned expert×domain **token-routing** distribution + routing-NMI.

### moe

Learned MoE token-routing — fraction of each **domain**'s tokens sent to each **expert** (top-1, summed over 6 layers). **Routing-NMI vs domain: 0.518** · usage-entropy 1.00 · dead experts 0/16.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | e10 | e11 | e12 | e13 | e14 | e15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki | 0.06 | 0.06 | 0.06 | 0.07 | 0.06 | 0.06 | 0.06 | 0.07 | 0.06 | 0.07 | 0.06 | 0.06 | 0.07 | 0.05 | 0.07 | 0.06 |
| news | 0.07 | 0.07 | 0.06 | 0.08 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.06 | 0.05 | 0.07 | 0.07 |
| reviews | 0.06 | 0.07 | 0.05 | 0.06 | 0.06 | 0.07 | 0.06 | 0.05 | 0.06 | 0.06 | 0.07 | 0.07 | 0.06 | 0.04 | 0.07 | 0.08 |
| arxiv | 0.06 | 0.05 | 0.08 | 0.05 | 0.06 | 0.06 | 0.06 | 0.08 | 0.07 | 0.06 | 0.06 | 0.04 | 0.07 | 0.09 | 0.06 | 0.05 |
| pubmed | 0.06 | 0.05 | 0.08 | 0.05 | 0.07 | 0.06 | 0.06 | 0.08 | 0.07 | 0.06 | 0.05 | 0.04 | 0.06 | 0.09 | 0.06 | 0.06 |
| math | 0.06 | 0.06 | 0.05 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.07 | 0.08 | 0.07 | 0.05 | 0.06 | 0.08 |
| legal | 0.06 | 0.08 | 0.06 | 0.08 | 0.07 | 0.06 | 0.06 | 0.05 | 0.06 | 0.08 | 0.05 | 0.05 | 0.07 | 0.06 | 0.06 | 0.05 |
| stories | 0.07 | 0.05 | 0.05 | 0.05 | 0.04 | 0.07 | 0.08 | 0.04 | 0.05 | 0.05 | 0.09 | 0.12 | 0.06 | 0.04 | 0.07 | 0.08 |

### em_prefix

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki | **3.5*** | 4.5 | 4.4 | 4.8 | 4.5 | 5.0 | 4.6 | 6.0 | 3.5 | 4.8 | 1.39× |
| news | 4.5 | **3.5*** | 4.3 | 6.5 | 5.9 | 4.6 | 4.7 | 5.7 | 3.5 | 5.2 | 1.50× |
| reviews | 4.5 | 4.1 | **3.5*** | 6.3 | 5.9 | 4.4 | 4.7 | 4.7 | 3.5 | 4.9 | 1.40× |
| arxiv | 3.6 | 4.6 | 4.4 | **2.8*** | 3.2 | 5.0 | 4.5 | 5.8 | 2.8 | 4.4 | 1.57× |
| pubmed | 3.6 | 4.5 | 4.4 | 3.4 | **2.9*** | 4.8 | 4.4 | 5.8 | 2.9 | 4.4 | 1.49× |
| math | 4.1 | 3.8 | 3.8 | 5.5 | 5.1 | **2.9*** | 3.9 | 4.1 | 2.9 | 4.3 | 1.50× |
| legal | 3.3 | 3.2 | 3.4 | 4.4 | 4.0 | 3.4 | **2.1*** | 4.3 | 2.1 | 3.7 | 1.75× |
| stories | 3.7 | 3.3 | 3.1 | 5.5 | 5.3 | 3.0 | 3.6 | **2.3*** | 2.3 | 3.9 | 1.72× |

**Summary:** mean self-expert ppl 2.9 vs mean other-expert ppl 4.5 → **1.52× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### em_gov

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki | **3.4*** | 4.4 | 4.4 | 5.2 | 4.9 | 5.1 | 5.1 | 6.2 | 3.4 | 5.0 | 1.47× |
| news | 4.4 | **3.4*** | 4.3 | 7.5 | 6.9 | 4.5 | 4.8 | 6.2 | 3.4 | 5.5 | 1.62× |
| reviews | 4.5 | 4.1 | **3.5*** | 6.3 | 6.4 | 4.4 | 5.1 | 4.6 | 3.5 | 5.1 | 1.46× |
| arxiv | 3.7 | 4.8 | 4.6 | **2.8*** | 3.2 | 5.3 | 4.6 | 6.2 | 2.8 | 4.6 | 1.65× |
| pubmed | 3.6 | 4.5 | 4.5 | 3.4 | **2.9*** | 5.2 | 4.5 | 6.3 | 2.9 | 4.6 | 1.57× |
| math | 4.2 | 3.7 | 3.7 | 5.8 | 5.8 | **2.8*** | 4.2 | 4.0 | 2.8 | 4.5 | 1.59× |
| legal | 3.6 | 3.3 | 3.5 | 4.6 | 4.3 | 3.5 | **2.1*** | 4.7 | 2.1 | 3.9 | 1.87× |
| stories | 3.7 | 3.3 | 3.0 | 6.0 | 6.2 | 2.9 | 4.2 | **2.2*** | 2.2 | 4.2 | 1.89× |

**Summary:** mean self-expert ppl 2.9 vs mean other-expert ppl 4.7 → **1.62× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.
