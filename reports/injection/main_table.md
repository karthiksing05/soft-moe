# Soft-MoE — comparison table

| method | macro-ppl ↓ | micro-ppl ↓ | routing-NMI ↑ | routing-acc ↑ | util-entropy ↑ | sep ↑ | swap-ratio ↑ | +params (over dense) ↓ | total-trainable ↓ | active/token ↓ | total ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_d256 | 2.949 | 2.828 | 1.000 | 1.000 | 0.962 | 1.143 | 1.532 | 2048 | 2048 | 4952576 | 4954368 |
| inj_token_d256 | 2.961 | 2.838 | 1.000 | 1.000 | 0.962 | 1.143 | 1.512 | 2048 | 2048 | 4952576 | 4954368 |
| govern_film_d256 | 2.920 | 2.800 | 1.000 | 1.000 | 0.962 | 1.143 | 1.436 | 2048 | 2048 | 6531584 | 6533376 |
| govern_spectral_d256 | 2.915 | 2.796 | 1.000 | 1.000 | 0.962 | 1.142 | 1.575 | 2048 | 2048 | 5075552 | 5077344 |
