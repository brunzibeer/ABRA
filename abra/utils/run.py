"""Run directory, logging and seeding.

Every command writes into its ``--output_root``:
    run.log        complete stdout/stderr of the run (tee'd from the console)
    metrics.jsonl  one JSON object per logged step / evaluation
    config.json    resolved arguments, git commit and environment
wandb mirrors the metrics only when ``--wandb_project`` is given.
"""
import datetime
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


class _Tee:
    """Duplicate a text stream into a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except OSError:
        return None


def environment():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.node(),
    }


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class RunLogger:
    """Owns the run directory: log file, metrics and the optional wandb mirror."""

    def __init__(self, out_dir, args, command, tee=True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = open(self.out_dir / "metrics.jsonl", "a")
        if tee:
            self._log = open(self.out_dir / "run.log", "a")
            sys.stdout = _Tee(sys.__stdout__, self._log)
            sys.stderr = _Tee(sys.__stderr__, self._log)
        started = datetime.datetime.now().isoformat(timespec="seconds")
        self.config = {"command": command, "started": started, "git_commit": git_commit(),
                       "argv": sys.argv, "environment": environment(), "args": _jsonable(vars(args))}
        (self.out_dir / "config.json").write_text(json.dumps(self.config, indent=2))
        print(f"[run] {command} started {started}, commit {self.config['git_commit']}, output {self.out_dir}")

        self.wandb = None
        if getattr(args, "wandb_project", None):
            import wandb
            name = f"{args.exp_name}_{os.getpid()}"
            wandb.init(project=args.wandb_project, entity=getattr(args, "wandb_entity", None),
                       config=self.config["args"], name=name)
            self.wandb = wandb

    def log(self, metrics, step=None):
        """Write one metrics record (and mirror it to wandb if enabled)."""
        record = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "step": step, **_jsonable(metrics)}
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def log_table(self, key, columns, rows):
        self.log({key: {"columns": columns, "rows": [list(r) for r in rows]}})
        if self.wandb is not None:
            self.wandb.log({key: self.wandb.Table(data=[list(r) for r in rows], columns=columns)})

    def close(self):
        if self.wandb is not None:
            self.wandb.finish()
        self._metrics.close()
        if isinstance(sys.stdout, _Tee):
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            self._log.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """Give every DataLoader worker its own, reproducible python/numpy seed."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
