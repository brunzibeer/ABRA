"""On-disk formats of domain experts and class residuals.

Domain expert (``kind = domain_expert``)
    safetensors with the trained tensors (``<module>.weight`` / ``.bias`` / ``in_proj_*``), as in the
    original ``best.pth``.

Class residual (``kind = residual``), the class-specific update tau = W_expert - W_parent of every
adapted layer, in one of two encodings:
    ``svft_band``  band of tau expressed in the canonical SVD basis of the parent domain expert
                   (``<layer>.band``, row-major band order). ~0.3 MB instead of 93 MB. Needs the
                   parent to be decoded; the parent's content hash is checked on load.
    ``dense``      tau itself (``<layer>.delta``), basis-free.
Both store ``<layer>.sketch``: fixed random probes y^T tau x of the true residual, re-checked after
decoding so that a wrong parent or an SVD that drifted is caught instead of silently used.

Metadata (string -> string in the safetensors header) holds ``kind``, ``encoding`` and a JSON blob
``abra`` with provenance (class, domain, parent path and hash, SVFT band width, seed, commit, ...).

Legacy ``.pth`` files of the original code are read transparently: domain experts are plain state
dicts; class experts hold dense ``svft_layer.{u,s_pre,s,v}`` (tau = u s v).
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, safe_open, save_file

from abra.weights.svd import band_indices, canonical_svd

N_PROBES = 8
SKETCH_TOL = 5e-3


# ---------------------------------------------------------------------------- helpers

def content_hash(tensors):
    """Format-independent sha256 of a dict of tensors (names, dtypes, shapes and values)."""
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().to("cpu").contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _stable_seed(name):
    h = 0
    for c in name.encode():
        h = (h * 131 + c) % (2**31 - 1)
    return h


def sketch(delta, layer):
    """``N_PROBES`` bilinear probes y^T delta x with vectors fixed by the layer name."""
    g = torch.Generator().manual_seed(_stable_seed(layer))
    x = torch.randn(N_PROBES, delta.shape[1], generator=g, dtype=torch.float64)
    y = torch.randn(N_PROBES, delta.shape[0], generator=g, dtype=torch.float64)
    return torch.einsum("pi,ij,pj->p", y, delta.double(), x)


def _sketch_error(delta, layer, stored):
    """Probe error normalised so that it is <= ||error||_F / ||delta||_F (Cauchy-Schwarz)."""
    g = torch.Generator().manual_seed(_stable_seed(layer))
    x = torch.randn(N_PROBES, delta.shape[1], generator=g, dtype=torch.float64)
    y = torch.randn(N_PROBES, delta.shape[0], generator=g, dtype=torch.float64)
    scale = delta.double().norm() * x.norm(dim=1) * y.norm(dim=1) + 1e-30
    return ((sketch(delta, layer) - stored.double()).abs() / scale).max().item()


def _read(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        tensors = load_file(str(path))
        info = json.loads(meta.get("abra", "{}"))
        info.update({k: v for k, v in meta.items() if k != "abra"})
        return tensors, info
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    return state, {"kind": "legacy"}


def _write(path, tensors, kind, info, encoding=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"kind": kind, "abra": json.dumps(info, default=str)}
    if encoding:
        meta["encoding"] = encoding
    save_file({k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}, str(path), metadata=meta)


# ---------------------------------------------------------------------------- domain experts

def save_domain_expert(path, tensors, info):
    info = {**info, "sha256": content_hash(tensors)}
    _write(path, tensors, "domain_expert", info)
    return info


def load_domain_expert(path):
    """Return ``(tensors, info)``; ``info['sha256']`` is the content hash used to link residuals."""
    tensors, info = _read(path)
    if info.get("kind") not in ("domain_expert", "legacy"):
        raise ValueError(f"{path} is a {info.get('kind')}, not a domain expert")
    if any(".svft_layer." in k for k in tensors):
        raise ValueError(f"{path} is a (legacy) class expert, not a domain expert")
    info.setdefault("sha256", content_hash(tensors))
    info["path"] = str(path)
    return tensors, info


def layer_weights(tensors, layers):
    """Weight matrices of the adapted layers (``<layer>.weight``) from a domain-expert state dict."""
    return {layer: tensors[f"{layer}.weight"] for layer in layers}


# ---------------------------------------------------------------------------- residuals

@dataclass
class Residual:
    """Class residual: dense tau per adapted layer, plus the SVFT coefficients it came from."""
    delta: dict                      # layer -> tau (float32, out x in)
    info: dict = field(default_factory=dict)
    coefficients: dict = None        # layer -> k x k coefficients of tau in `basis` (SVFT residuals)
    basis: dict = None               # layer -> (U, Vh) the coefficients refer to
    source_sigma: dict = None        # layer -> singular values of the parent (source) domain expert

    @property
    def layers(self):
        return sorted(self.delta)


def residual_from_svft(trained_delta, parent_weights, off_diag, info):
    """Encode an SVFT residual in the canonical basis of its parent (band only).

    ``trained_delta``: layer -> tau computed in the training basis (exact). Returns the tensors to save
    and the per-layer fraction of tau's energy outside the band (should be ~0).
    """
    tensors, off_band = {}, {}
    for layer, delta in trained_delta.items():
        u, _, vh = canonical_svd(parent_weights[layer])
        coeffs = u.T @ delta.double() @ vh.T
        rows, cols = band_indices(coeffs.shape[0], off_diag)
        band = coeffs[rows, cols]
        off_band[layer] = 1.0 - (band.norm() ** 2 / (coeffs.norm() ** 2 + 1e-30)).item()
        tensors[f"{layer}.band"] = band.float()
        tensors[f"{layer}.sketch"] = sketch(delta, layer)
    info = {**info, "off_diag": off_diag, "max_off_band_energy": max(off_band.values())}
    return tensors, info


def save_residual(path, trained_delta, parent_weights, parent_info, off_diag, info, encoding="svft_band"):
    info = {**info, "parent": parent_info.get("path"), "parent_sha256": parent_info["sha256"]}
    if encoding == "svft_band":
        tensors, info = residual_from_svft(trained_delta, parent_weights, off_diag, info)
    elif encoding == "dense":
        tensors = {f"{l}.delta": d.float() for l, d in trained_delta.items()}
        tensors.update({f"{l}.sketch": sketch(d, l) for l, d in trained_delta.items()})
    else:
        raise ValueError(encoding)
    _write(path, tensors, "residual", info, encoding)
    return info


def load_residual(path, parent=None):
    """Load a class residual as dense tau per layer.

    ``parent``: path of the parent domain expert; needed for ``svft_band`` residuals when the path
    stored in the file is not valid on this machine. Its content hash must match.
    """
    tensors, info = _read(path)
    info["path"] = str(path)
    if info["kind"] == "legacy":
        return _load_legacy_expert(tensors, info)
    if info["kind"] != "residual":
        raise ValueError(f"{path} is a {info['kind']}, not a class residual")

    layers = sorted({k.rsplit(".", 1)[0] for k in tensors if k.endswith(".sketch")})
    if info["encoding"] == "dense":
        residual = Residual({l: tensors[f"{l}.delta"] for l in layers}, info)
    else:
        parent_path = parent or info.get("parent")
        if parent_path is None or not Path(parent_path).exists():
            raise FileNotFoundError(f"{path}: parent domain expert {parent_path!r} not found; pass it explicitly")
        parent_tensors, parent_info = load_domain_expert(parent_path)
        if parent_info["sha256"] != info["parent_sha256"]:
            raise ValueError(f"{path}: parent {parent_path} has hash {parent_info['sha256'][:12]}, "
                             f"expected {info['parent_sha256'][:12]} (wrong domain expert?)")
        delta, coefficients, basis, sigma = {}, {}, {}, {}
        for l in layers:
            u, s, vh = canonical_svd(parent_tensors[f"{l}.weight"])
            k = u.shape[1]
            rows, cols = band_indices(k, info["off_diag"])
            c = torch.zeros(k, k, dtype=torch.float64)
            c[rows, cols] = tensors[f"{l}.band"].double()
            coefficients[l], basis[l], sigma[l] = c, (u, vh), s
            delta[l] = (u @ c @ vh).float()
        residual = Residual(delta, info, coefficients, basis, sigma)

    worst = max(_sketch_error(residual.delta[l], l, tensors[f"{l}.sketch"]) for l in layers)
    residual.info["sketch_error"] = worst
    if worst > SKETCH_TOL:
        raise ValueError(f"{path}: decoded residual does not match its checksum (error {worst:.2e} > {SKETCH_TOL})")
    return residual


def _load_legacy_expert(tensors, info):
    """Original class-expert checkpoint: dense u, s_pre, s, v per layer (tau = u s v)."""
    layers = sorted({k.split(".svft_layer.")[0] for k in tensors if ".svft_layer." in k})
    if not layers:
        raise ValueError(f"{info['path']}: no svft_layer tensors; is this a domain expert?")
    delta, coefficients, basis, sigma = {}, {}, {}, {}
    for l in layers:
        u, s, v = (tensors[f"{l}.svft_layer.{x}"].double() for x in ("u", "s", "v"))
        delta[l] = (u @ s @ v).float()
        coefficients[l], basis[l] = tensors[f"{l}.svft_layer.s"], (u, v)
        sigma[l] = torch.diagonal(tensors[f"{l}.svft_layer.s_pre"]).double()
    info.update({"encoding": "legacy_svft_dense"})
    return Residual(delta, info, coefficients, basis, sigma)
