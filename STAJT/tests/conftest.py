"""Keep test runs light on shared machines (Zaratan login nodes have 128 cores)."""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")
