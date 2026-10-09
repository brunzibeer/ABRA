"""[Historical] Drives the ORIGINAL main.py: run it from commit 4c07a60 (original code + golden scripts).
Run the unmodified ``main.py`` with global seeding and a per-step loss trace.

``main.py`` has no seeding and logs training losses only to wandb. This wrapper seeds python/numpy/torch,
replaces the ``wandb`` handle *inside engine.py only* with a recorder that writes every logged dict to
``$GOLDEN_TRACE`` (jsonl), and then executes ``main.py`` as ``__main__`` with the remaining argv.

    GOLDEN_SEED=0 GOLDEN_TRACE=trace.jsonl python tests/golden/seeded_main.py <main.py args>
"""
import json
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

seed = int(os.environ.get("GOLDEN_SEED", "0"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

trace_path = os.environ.get("GOLDEN_TRACE")
if trace_path:
    import engine

    class _Recorder:
        run = True

        def __init__(self, path):
            self._f = open(path, "w")

        def log(self, data, step=None):
            self._f.write(json.dumps({"step": step, **{k: float(v) for k, v in data.items()}}) + "\n")
            self._f.flush()

    engine.wandb = _Recorder(trace_path)

sys.argv = [str(REPO / "main.py")] + sys.argv[1:]
runpy.run_path(str(REPO / "main.py"), run_name="__main__")
