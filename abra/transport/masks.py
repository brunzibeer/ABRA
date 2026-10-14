"""Masks for the ABRA coefficients C = U_T^T tau V_T (k x k, target spectral basis): R = U_T (W * C) V_T^T.

Without a mask ABRA is the projection of tau onto the target singular subspaces (task addition on square layers);
any W != 1 changes the residual in weight space. A mask is given as a string (``--abra_mask``):

  plain               diagonal only (= band:0)
  banded              the SVFT band of the residual (same pattern as its source coefficients)
  band:W[:G]          1 on |i - j| <= W, G outside (G = 0: hard band; band:0:G softens everything off the diagonal)
  block:R[:G]         1 on the leading R x R block (top-R target directions), G outside
  gauss:S             soft, exp(-(i - j)^2 / (2 S^2)): off-diagonal entries fade with their distance from the diagonal
  exp:S               soft, exp(-|i - j| / S)
  score[:T]           SCORE trim (Chaves et al. 2026, Eq. 4): keep the diagonal and the off-diagonal entries with
                      |c - mean_off| < T std_off (per layer, default T = 1.96), zero the outliers
  topk:P              keep the fraction P of the largest |c| per layer (data-dependent hard mask)
"""
import torch

from abra.weights.svd import band_mask


def _distance(k, dtype):
    i = torch.arange(k, dtype=dtype)
    return (i[:, None] - i[None, :]).abs()


def _num(text, name, spec):
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"ABRA mask {spec!r}: {name} must be a number, got {text!r}") from None


def parse_mask(spec):
    """Validate a mask string and return (kind, params); raises ValueError on a malformed spec."""
    if spec is None:
        return None
    kind, *params = spec.split(":")
    arity = {"plain": (0, 0), "banded": (0, 0), "band": (1, 2), "block": (1, 2), "gauss": (1, 1), "exp": (1, 1),
             "score": (0, 1), "topk": (1, 1)}
    if kind not in arity:
        raise ValueError(f"unknown ABRA mask {spec!r}; kinds: {sorted(arity)}")
    lo, hi = arity[kind]
    if not lo <= len(params) <= hi:
        raise ValueError(f"ABRA mask {spec!r}: {kind} takes {lo}-{hi} parameters")
    values = [_num(p, f"parameter {n + 1}", spec) for n, p in enumerate(params)]
    if kind in ("band", "block") and len(values) == 2 and not 0 <= values[1] <= 1:
        raise ValueError(f"ABRA mask {spec!r}: the outside weight G must be in [0, 1]")
    if kind in ("gauss", "exp") and values[0] <= 0:
        raise ValueError(f"ABRA mask {spec!r}: the width must be > 0")
    if kind == "topk" and not 0 < values[0] <= 1:
        raise ValueError(f"ABRA mask {spec!r}: P must be in (0, 1]")
    return kind, values


def mask_weights(spec, coeffs, svft_pattern=None, off_diag=None):
    """Weights W (same shape as ``coeffs``) of the mask ``spec`` for one layer.

    ``svft_pattern``: boolean pattern of the residual's SVFT coefficients (``banded``); ``off_diag``: its band
    half-width, used when the pattern is not stored."""
    kind, p = parse_mask(spec)
    k, dtype = coeffs.shape[0], coeffs.dtype
    if kind == "plain":
        return torch.eye(k, dtype=dtype)
    if kind == "banded":
        if svft_pattern is not None:
            return (svft_pattern != 0).to(dtype)
        return band_mask(k, off_diag, dtype=dtype)
    if kind in ("band", "block"):
        g = p[1] if len(p) == 2 else 0.0
        if kind == "band":
            inside = _distance(k, dtype) <= p[0]
        else:
            idx = torch.arange(k)
            inside = (idx[:, None] < p[0]) & (idx[None, :] < p[0])
        return torch.where(inside, torch.ones((), dtype=dtype), torch.full((), g, dtype=dtype))
    if kind == "gauss":
        return torch.exp(-_distance(k, dtype) ** 2 / (2 * p[0] ** 2))
    if kind == "exp":
        return torch.exp(-_distance(k, dtype) / p[0])
    if kind == "score":
        t = p[0] if p else 1.96
        off = ~torch.eye(k, dtype=torch.bool)
        vals = coeffs[off]
        keep = (coeffs - vals.mean()).abs() < t * vals.std()
        return (keep | ~off).to(dtype)
    if kind == "topk":
        n = max(1, round(p[0] * coeffs.numel()))
        thr = coeffs.abs().flatten().kthvalue(coeffs.numel() - n + 1).values
        return (coeffs.abs() >= thr).to(dtype)
    raise AssertionError(kind)
