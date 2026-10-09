from dataclasses import dataclass, field

import torch

from abra.weights.svd import SVDCache

TRANSPORTS = {}


def register(name):
    def wrap(fn):
        TRANSPORTS[name] = fn
        return fn
    return wrap


@dataclass
class TransportContext:
    """Weights a transport may use, restricted to the adapted layers (layer -> weight matrix)."""
    target: dict                       # theta_T
    source: dict = None                # theta_S (parent of the residual); optional for most methods
    device: str = "cpu"                # device for methods that must reproduce legacy GPU behaviour
    target_svd: SVDCache = field(init=False)
    source_svd: SVDCache = field(init=False)

    def __post_init__(self):
        self.target_svd = SVDCache(self.target)
        self.source_svd = SVDCache(self.source) if self.source is not None else None


def transport(name, residual, ctx, **options):
    if name not in TRANSPORTS:
        raise ValueError(f"unknown transport {name!r}; available: {sorted(TRANSPORTS)}")
    out = TRANSPORTS[name](residual, ctx, **options)
    missing = set(residual.layers) - set(out)
    assert not missing, f"transport {name} dropped layers {sorted(missing)[:3]}"
    return {layer: r.to(torch.float32) for layer, r in out.items()}
