# Soft-MoE — comparison table

| method | macro-ppl ↓ | micro-ppl ↓ | routing-NMI ↑ | routing-acc ↑ | util-entropy ↑ | sep ↑ | swap-ratio ↑ | +params (over dense) ↓ | total-trainable ↓ | active/token ↓ | total ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 2.965 | 2.841 | 0.626 | 0.564 | 0.649 | nan | nan | 0 | 4952320 | 4952320 | 4952320 |
| em_gov | 2.894 | 2.778 | 1.000 | 1.000 | 0.962 | 1.142 | 1.618 | 2048 | 2048 | 5124800 | 5126592 |
| em_prefix | 2.940 | 2.820 | 1.000 | 1.000 | 0.962 | 1.143 | 1.530 | 2048 | 2048 | 4952576 | 4954368 |
| moe | 2.660 | 2.557 | 0.626 | 0.564 | 0.649 | nan | nan | 23686656 | 27063040 | 4978432 | 27063040 |
