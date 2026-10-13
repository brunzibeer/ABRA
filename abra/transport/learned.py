"""Transport learned from the available classes (Option 1 of the research plan, options.pdf §4.1).

For every adapted layer a map is fitted by ridge regression towards the identity on pairs of residuals of the
*available* classes a: tau_S^a (SVFT on theta_S with source data) and tau_T^a (SVFT on theta_T with target data).
The map is then applied to the residuals of the unavailable classes (``transport("learned", ..., operator=op)``).

Variants (``variant``):
    weight_left        tau_T ~ A tau_S                      A: m x m              lam -> inf: task addition
    weight_full        tau_T ~ A tau_S B (alternating)      A: m x m, B: n x n    lam -> inf: task addition
    weight_lowrank<r>  A = I + rank-r correction, B = I     (rank-r truncation of the weight_left solution)
    spectral_full      c_T ~ L c_S B,  c = U_T^T tau V_T     L, B: k x k           lam -> inf: ABRA
    spectral_band<w>   as spectral_full with L, B banded (|i - j| <= w; w = 0: one scale per singular direction)
U_T, V_T are the canonical singular vectors of theta_T (k = min(m, n)). ``band_out`` masks the predicted target
coefficients to the SVFT band (residuals trained with SVFT on theta_T are banded in its basis).

Ridge: min_A sum_a ||A M_a - T_a||^2 + lam_A ||A - I||^2 (and the same for B), solved in closed form; the
two-sided variants alternate A and B (the objective decreases at every step). ``lam`` is relative to the data:
lam_A = lam * sum_a ||S_a||^2 / rows and lam_B = lam * sum_a ||S_a||^2 / cols, fixed for the whole fit.
Encoder layer 0 is never trained (docs/MIGRATION.md §4): its residuals are unrelated random SVFT initialisations, so
those layers are not fitted (identity) unless ``keep_layer0``.
"""
import re
from dataclasses import dataclass, field

import torch

from abra.weights.io import _read, _write, content_hash
from abra.weights.svd import band_mask

LAYER0 = re.compile(r"\.(fusion_layers|layers)\.0\.")
_VARIANT = re.compile(r"^(weight_left|weight_full|weight_lowrank(\d+)|spectral_full|spectral_band(\d+))$")


def parse_variant(name):
    """-> (coordinates 'weight' | 'spectral', form 'left' | 'full' | 'lowrank' | 'band', rank or band width)."""
    m = _VARIANT.match(name)
    if not m:
        raise ValueError(f"unknown learned-transport variant {name!r}: weight_left, weight_full, weight_lowrank<r>, "
                         "spectral_full, spectral_band<w>")
    coords = "spectral" if name.startswith("spectral") else "weight"
    if name == "weight_left":
        return coords, "left", 0
    if m.group(2):
        return coords, "lowrank", int(m.group(2))
    if m.group(3):
        return coords, "band", int(m.group(3))
    return coords, "full", 0


def is_layer0(layer):
    return bool(LAYER0.search(layer))


@dataclass
class Operator:
    """Fitted transport: per layer a left map (A or L) and/or a right map (B); a missing map is the identity."""
    variant: str
    lam: float
    left: dict
    right: dict
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- coordinates

def to_coords(coords, layer, tau, ctx):
    tau = tau.double()
    if coords == "weight":
        return tau
    u, _, vh = ctx.target_svd(layer)
    return u.T @ tau @ vh.T


def from_coords(coords, layer, x, ctx):
    if coords == "weight":
        return x
    u, _, vh = ctx.target_svd(layer)
    return u @ x @ vh


def coordinates(pairs, ctx, coords, layers):
    """{layer: {class: (S, T)}} in the given coordinates, computed once (reused by every fold and lambda)."""
    return {layer: {c: (to_coords(coords, layer, ts.delta[layer], ctx), to_coords(coords, layer, tt.delta[layer], ctx))
                    for c, (ts, tt) in pairs.items()} for layer in layers}


def target_hash(ctx, layers):
    return content_hash({layer: ctx.target[layer] for layer in layers})


# ---------------------------------------------------------------------------- ridge solvers

def _banded_rows(G, H, lam, w):
    """Rows of X with support |i - j| <= w minimising sum ||X M - T||^2 + lam ||X - I||^2, given G = sum M M^T and
    H = sum T M^T: per row, (G[J, J] + lam I) x_J = H[i, J] + lam e_i[J]; all rows solved in one batch."""
    p = G.shape[0]
    rows = torch.arange(p)[:, None]
    idx = rows + torch.arange(-w, w + 1)[None, :]                      # p x (2w+1)
    valid = (idx >= 0) & (idx < p)
    idxc = idx.clamp(0, p - 1)
    system = G[idxc[:, :, None], idxc[:, None, :]]
    system = torch.where(valid[:, :, None] & valid[:, None, :], system, torch.zeros_like(system))
    system = system + lam * torch.eye(2 * w + 1, dtype=G.dtype)          # invalid entries decouple: lam * x = 0
    rhs = H[rows.expand_as(idx), idxc] + lam * (idx == rows).to(G.dtype)
    rhs = torch.where(valid, rhs, torch.zeros_like(rhs))
    sol = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(G)
    out[rows.expand_as(idx)[valid], idx[valid]] = sol[valid]
    return out


def _solve_left(S, T, B, lam, band):
    """A minimising sum ||A (S B) - T||^2 + lam ||A - I||^2."""
    M = [s if B is None else s @ B for s in S]
    G = sum(m @ m.T for m in M)
    H = sum(t @ m.T for t, m in zip(T, M))
    if band is not None:
        return _banded_rows(G, H, lam, band)
    eye = torch.eye(G.shape[0], dtype=G.dtype)
    return torch.linalg.solve(G + lam * eye, (H + lam * eye).T).T     # A (G + lam I) = H + lam I, G symmetric


def _solve_right(S, T, A, lam, band):
    """B minimising sum ||(A S) B - T||^2 + lam ||B - I||^2."""
    N = [A @ s for s in S]
    G = sum(n.T @ n for n in N)
    H = sum(n.T @ t for n, t in zip(N, T))
    if band is not None:
        return _banded_rows(G, H.T, lam, band).T                       # columns of B = rows of B^T
    eye = torch.eye(G.shape[0], dtype=G.dtype)
    return torch.linalg.solve(G + lam * eye, H + lam * eye)


def fit_layer(S, T, form, param, lam, iters=5, tol=1e-6):
    """Maps (A, B) of one layer from lists of source / target coordinates; None = identity."""
    total = sum(float((s ** 2).sum()) for s in S)
    if total == 0:
        return None, None
    p, q = S[0].shape
    lam_a, lam_b = max(lam * total / p, 1e-30), max(lam * total / q, 1e-30)
    if form in ("left", "lowrank"):
        A = _solve_left(S, T, None, lam_a, None)
        if form == "lowrank":
            eye = torch.eye(p, dtype=A.dtype)
            u, s, vh = torch.linalg.svd(A - eye, full_matrices=False)
            A = eye + (u[:, :param] * s[:param]) @ vh[:param]
        return A, None
    band = param if form == "band" else None
    A, B = None, torch.eye(q, dtype=S[0].dtype)
    for step in range(iters):
        A_new = _solve_left(S, T, B, lam_a, band)
        B_new = _solve_right(S, T, A_new, lam_b, band)
        converged = step > 0 and max(float((A_new - A).norm() / (A.norm() + 1e-30)),
                                     float((B_new - B).norm() / (B.norm() + 1e-30))) < tol
        A, B = A_new, B_new
        if converged:
            break
    return A, B


# ---------------------------------------------------------------------------- operator

def fit_operator(coords_cache, classes, variant, lam, iters=5, band_out=None, keep_layer0=False, target_sha=None):
    """Fit on ``classes`` (keys of the pairs) from the cache of ``coordinates``."""
    coords, form, param = parse_variant(variant)
    left, right, skipped = {}, {}, []
    for layer, per_class in coords_cache.items():
        if is_layer0(layer) and not keep_layer0:
            skipped.append(layer)
            continue
        S = [per_class[c][0] for c in classes]
        T = [per_class[c][1] for c in classes]
        A, B = fit_layer(S, T, form, param, lam, iters)
        if A is not None:
            left[layer] = A
        if B is not None:
            right[layer] = B
    info = {"variant": variant, "lam": lam, "coords": coords, "iters": iters, "band_out": band_out,
            "classes": list(classes), "skipped": skipped, "layers": sorted(coords_cache), "target_sha256": target_sha}
    return Operator(variant, lam, left, right, info)


def predict(op, residual, ctx, coords_cache=None, cls=None):
    """Transported residual R (layer -> tensor, float64). ``coords_cache``/``cls``: reuse precomputed coordinates."""
    coords, _, _ = parse_variant(op.variant)
    band_out, skipped = op.info.get("band_out"), set(op.info.get("skipped", []))
    out = {}
    for layer in residual.layers:
        x = coords_cache[layer][cls][0] if coords_cache is not None else to_coords(coords, layer, residual.delta[layer], ctx)
        if layer in op.left:
            x = op.left[layer].double() @ x
        if layer in op.right:
            x = x @ op.right[layer].double()
        if band_out is not None and coords == "spectral" and layer not in skipped:
            x = x * band_mask(x.shape[0], band_out)
        out[layer] = from_coords(coords, layer, x, ctx)
    return out


def apply_operator(op, residual, ctx):
    """Transport with a saved operator; refuses an operator fitted on a different theta_T."""
    expected = op.info.get("target_sha256")
    if expected is not None:
        layers = op.info.get("layers") or residual.layers
        found = target_hash(ctx, layers)
        if found != expected:
            raise ValueError(f"the operator was fitted on another target domain expert (hash {expected[:12]}, "
                             f"this one {found[:12]})")
    missing = set(residual.layers) - set(op.info.get("layers", residual.layers))
    if missing:
        raise ValueError(f"the operator has no map for {len(missing)} layers of the residual, e.g. {sorted(missing)[:2]}")
    return predict(op, residual, ctx)


def deviation(op):
    """Mean ||A - I||_F / ||I||_F (and for B) per layer type: how much the learned map departs from the identity."""
    out = {}
    for side, maps in (("left", op.left), ("right", op.right)):
        for layer, m in maps.items():
            kind = re.sub(r"\.\d+\.", ".N.", layer)
            eye = torch.eye(m.shape[0], dtype=torch.float64)
            out.setdefault(kind, {}).setdefault(side, []).append(float((m.double() - eye).norm() / eye.norm()))
    return {k: {s: sum(v) / len(v) for s, v in sides.items()} for k, sides in sorted(out.items())}


def save_operator(path, op):
    tensors = {f"{layer}.left": a.float() for layer, a in op.left.items()}
    tensors.update({f"{layer}.right": b.float() for layer, b in op.right.items()})
    _write(path, tensors, "transport_operator", {**op.info, "variant": op.variant, "lam": op.lam})


def load_operator(path):
    tensors, info = _read(path)
    if info.get("kind") != "transport_operator":
        raise ValueError(f"{path} is a {info.get('kind')}, not a learned transport operator")
    left = {k[: -len(".left")]: v for k, v in tensors.items() if k.endswith(".left")}
    right = {k[: -len(".right")]: v for k, v in tensors.items() if k.endswith(".right")}
    return Operator(info["variant"], float(info["lam"]), left, right, info)
