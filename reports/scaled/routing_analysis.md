# Domain routing analysis (every model)

Per-model expert↔domain specialization: our expert-token methods show an expert×domain **perplexity** matrix (swap test); the MoE arm shows its learned expert×domain **token-routing** distribution + routing-NMI.

### moe_g1

Learned MoE token-routing — fraction of each **domain**'s tokens sent to each **expert** (top-1, summed over 8 layers). **Routing-NMI vs domain: 0.006** · usage-entropy 1.00 · dead experts 0/8.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
|---|---|---|---|---|---|---|---|---|
| wiki | 0.13 | 0.12 | 0.13 | 0.13 | 0.13 | 0.12 | 0.13 | 0.12 |
| news | 0.13 | 0.13 | 0.12 | 0.13 | 0.12 | 0.12 | 0.13 | 0.13 |
| reviews | 0.13 | 0.12 | 0.13 | 0.12 | 0.13 | 0.13 | 0.12 | 0.12 |
| arxiv | 0.13 | 0.13 | 0.12 | 0.12 | 0.12 | 0.13 | 0.12 | 0.12 |
| pubmed | 0.13 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.13 | 0.13 |
| math | 0.12 | 0.13 | 0.13 | 0.12 | 0.13 | 0.12 | 0.13 | 0.12 |
| legal | 0.12 | 0.12 | 0.13 | 0.12 | 0.13 | 0.13 | 0.12 | 0.13 |
| stories | 0.12 | 0.13 | 0.12 | 0.13 | 0.12 | 0.13 | 0.13 | 0.12 |

### moe_g2

Learned MoE token-routing — fraction of each **domain**'s tokens sent to each **expert** (top-1, summed over 8 layers). **Routing-NMI vs domain: 0.467** · usage-entropy 1.00 · dead experts 0/16.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | e10 | e11 | e12 | e13 | e14 | e15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki | 0.06 | 0.07 | 0.05 | 0.05 | 0.06 | 0.06 | 0.06 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 | 0.07 |
| news | 0.07 | 0.07 | 0.06 | 0.08 | 0.06 | 0.06 | 0.06 | 0.07 | 0.06 | 0.07 | 0.06 | 0.06 | 0.07 | 0.06 | 0.05 | 0.06 |
| reviews | 0.06 | 0.06 | 0.06 | 0.07 | 0.06 | 0.06 | 0.06 | 0.07 | 0.06 | 0.05 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.08 |
| arxiv | 0.06 | 0.06 | 0.08 | 0.04 | 0.07 | 0.07 | 0.07 | 0.06 | 0.06 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.07 |
| pubmed | 0.06 | 0.06 | 0.08 | 0.04 | 0.07 | 0.07 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 | 0.06 | 0.06 | 0.05 | 0.07 |
| math | 0.07 | 0.06 | 0.06 | 0.08 | 0.07 | 0.07 | 0.05 | 0.07 | 0.06 | 0.05 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 |
| legal | 0.06 | 0.06 | 0.05 | 0.07 | 0.06 | 0.06 | 0.06 | 0.07 | 0.06 | 0.07 | 0.07 | 0.06 | 0.06 | 0.06 | 0.07 | 0.05 |
| stories | 0.07 | 0.06 | 0.07 | 0.08 | 0.06 | 0.07 | 0.05 | 0.07 | 0.07 | 0.05 | 0.06 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 |

### moe_oracle

Learned MoE token-routing — fraction of each **domain**'s tokens sent to each **expert** (top-1, summed over 8 layers). **Routing-NMI vs domain: 1.000** · usage-entropy 0.96 · dead experts 0/8.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
|---|---|---|---|---|---|---|---|---|
| wiki | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| news | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| reviews | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| arxiv | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pubmed | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| math | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| legal | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| stories | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

### cbtm

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki | 15.1 | 8.6 | 5.6 | 13.0 | **3.1*** | 14.9 | 5.6 | 4.2 | 3.1 | 9.6 | 3.10× |
| news | 12.5 | 5.6 | 6.8 | 16.4 | 4.5 | 13.5 | 8.9 | **3.5*** | 3.5 | 9.7 | 2.79× |
| reviews | 15.3 | 8.7 | **3.4*** | 13.0 | 3.9 | 15.2 | 5.5 | 4.8 | 3.4 | 9.5 | 2.78× |
| arxiv | 16.2 | 10.9 | 7.1 | 16.0 | **2.5*** | 16.3 | 8.3 | 4.2 | 2.5 | 11.3 | 4.58× |
| pubmed | 15.0 | 10.1 | 6.9 | 15.1 | **2.5*** | 14.7 | 8.0 | 4.2 | 2.5 | 10.6 | 4.25× |
| math | 13.3 | 7.4 | 5.1 | 6.3* | **3.4** | 13.4 | 4.8 | 4.7 | 6.3 | 7.4 | 1.18× |
| legal | 12.5 | 7.8 | 5.6 | 16.8 | **1.8*** | 15.9 | 7.9 | 4.5 | 1.8 | 10.2 | 5.67× |
| stories | 14.0 | 7.7 | 4.1 | 8.9 | 3.6 | 13.9 | **2.0*** | 4.3 | 2.0 | 8.1 | 4.02× |

**Summary:** mean self-expert ppl 3.1 vs mean other-expert ppl 9.5 → **3.05× worse** through the wrong expert; lowest-ppl expert is the matched one for **88%** of domains.

### ours_sup_seq

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki | **2.9*** | 3.8 | 3.6 | 4.0 | 3.8 | 4.2 | 4.0 | 5.1 | 2.9 | 4.1 | 1.40× |
| news | 3.9 | **2.8*** | 3.6 | 5.2 | 4.8 | 4.0 | 3.9 | 5.0 | 2.8 | 4.4 | 1.57× |
| reviews | 3.9 | 3.5 | **3.0*** | 5.0 | 4.9 | 3.9 | 4.2 | 4.3 | 3.0 | 4.2 | 1.40× |
| arxiv | 3.1 | 3.8 | 3.6 | **2.4*** | 2.8 | 4.1 | 3.8 | 4.9 | 2.4 | 3.7 | 1.53× |
| pubmed | 3.0 | 3.8 | 3.7 | 2.9 | **2.5*** | 3.9 | 3.5 | 4.9 | 2.5 | 3.7 | 1.49× |
| math | 3.4 | 3.2 | 3.1 | 4.2 | 3.8 | **2.3*** | 3.3 | 3.3 | 2.3 | 3.5 | 1.49× |
| legal | 2.6 | 2.5 | 2.7 | 3.2 | 2.6 | 2.7 | **1.8*** | 3.2 | 1.8 | 2.8 | 1.56× |
| stories | 3.0 | 2.7 | 2.6 | 3.9 | 3.7 | 2.5 | 3.0 | **1.9*** | 1.9 | 3.0 | 1.59× |

**Summary:** mean self-expert ppl 2.5 vs mean other-expert ppl 3.7 → **1.49× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.

### ours_unsup_seq

Perplexity of each **domain** (row) routed through each **expert** (column). `*` marks the domain's own (routed/cluster) expert; **bold** marks the lowest-perplexity expert in the row. Specialization ⇒ the `*` and **bold** coincide and the diagonal is well below the rest.

| domain ↓ / expert → | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | self-ppl | other-mean | ×worse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki | 5.6 | 5.2 | 3.4 | 3.9 | **3.2*** | 18.7 | 4.7 | 3.4 | 3.2 | 6.4 | 2.01× |
| news | 5.8 | 5.3 | 3.6 | 3.9 | 3.7 | 19.6 | 4.7 | **2.8*** | 2.8 | 6.7 | 2.39× |
| reviews | 5.6 | 5.3 | **3.0*** | 3.8 | 3.8 | 20.2 | 4.1 | 3.4 | 3.0 | 6.6 | 2.17× |
| arxiv | 4.6 | 4.4 | 3.5 | 3.8 | **2.7*** | 15.2 | 4.5 | 3.4 | 2.7 | 5.6 | 2.08× |
| pubmed | 4.5 | 4.3 | 3.5 | 3.6 | **2.7*** | 15.1 | 4.5 | 3.4 | 2.7 | 5.6 | 2.09× |
| math | 4.8 | 4.6 | 3.1 | **2.4*** | 3.1 | 15.5 | 3.2 | 3.1 | 2.4 | 5.4 | 2.26× |
| legal | 3.3 | 3.3 | 2.7 | 2.6 | **1.9*** | 15.3 | 3.0 | 2.4 | 1.9 | 4.6 | 2.44× |
| stories | 4.4 | 4.2 | 2.6 | 2.5 | 2.8 | 16.7 | **1.9*** | 2.7 | 1.9 | 5.1 | 2.66× |

**Summary:** mean self-expert ppl 2.6 vs mean other-expert ppl 5.7 → **2.23× worse** through the wrong expert; lowest-ppl expert is the matched one for **100%** of domains.
