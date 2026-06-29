# Soft-MoE — comparison table

| method | macro-ppl ↓ | micro-ppl ↓ | routing-NMI ↑ | routing-acc ↑ | util-entropy ↑ | sep ↑ | swap-ratio ↑ | +params ↓ |
|---|---|---|---|---|---|---|---|---|
| cbtm | 30.219 | 28.316 | 0.760 | 0.804 | 0.970 | nan | nan | 649125888 |
| dense | 26.989 | 25.031 | 0.760 | 0.804 | 0.970 | nan | nan | 0 |
| hard_moe | 27.968 | 25.945 | 0.760 | 0.804 | 0.970 | nan | nan | 226722816 |
| ours_sup | 35.658 | 32.460 | 0.760 | 0.804 | 0.970 | 1.248 | 1.180 | 15360 |
| ours_unsup | 35.198 | 32.217 | 0.830 | 0.828 | 0.974 | 1.249 | 1.217 | 609797 |
