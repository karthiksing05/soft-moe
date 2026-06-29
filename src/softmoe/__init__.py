"""softmoe — soft-MoE-style partitioning within a single LLM via EM-discovered expert tokens.

The package is organized into four subsystems, mirroring the research loop:

- ``softmoe.data``     — build a multi-domain, tokenized, sharded corpus + cluster assignments.
- ``softmoe.models``   — the ``ExpertTokenBank`` + ``SoftMoE`` wrapper + router, and all baselines.
- ``softmoe.training`` — the EM training loop and its losses/regularizers.
- ``softmoe.eval``     — perplexity + specialization metrics, harness, and report.

Everything is config-driven (``softmoe.utils.config``); no hyperparameters live in code.
"""

import os as _os
import sys as _sys

# macOS/conda commonly ships duplicate OpenMP runtimes (torch's libomp + sklearn/scipy's), whose
# collision segfaults when torch LAPACK runs alongside sklearn. Set mitigations before torch is
# imported anywhere. On macOS we also pin a single OpenMP thread (the reliable fix); Linux/cluster
# runs are left untouched for full multi-thread performance. Override any of these via the env.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
if _sys.platform == "darwin":
    _os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
