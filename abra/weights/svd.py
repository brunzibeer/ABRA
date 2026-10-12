"""Canonical-sign SVD and the SVFT band layout.

The sign (and, for near-degenerate singular values, the rotation) of singular vectors returned by
``torch.linalg.svd`` depends on device, library build and run. Anything stored in a singular basis
must therefore be expressed in a reproducible basis: we compute the SVD in float64 on CPU and fix
the sign of every left singular vector so that its largest-magnitude entry is positive (the paired
right singular vector gets the same sign, which leaves U diag(S) Vh unchanged).
"""
import torch


def canonical_svd(weight):
    """Return ``(U, S, Vh)`` of ``weight`` (float64, CPU) with canonical signs."""
    u, s, vh = torch.linalg.svd(weight.detach().to("cpu", torch.float64), full_matrices=False)
    rows = u.abs().argmax(dim=0)
    signs = torch.sign(u[rows, torch.arange(u.shape[1])])
    signs[signs == 0] = 1
    return u * signs, s, vh * signs[:, None]


def band_indices(k, off_diag):
    """Row-major (row, col) indices of a k x k band with |row - col| <= off_diag.

    The order matches ``torch.sparse.spdiags(...).coalesce().indices()`` used by the original SVFT
    code, so parameter vectors can be exchanged between the two implementations.
    """
    i = torch.arange(k)
    rows, cols = torch.meshgrid(i, i, indexing="ij")
    keep = (rows - cols).abs() <= off_diag
    return rows[keep], cols[keep]


def band_mask(k, off_diag, device=None, dtype=torch.float64):
    i = torch.arange(k, device=device)
    return ((i[:, None] - i[None, :]).abs() <= off_diag).to(dtype)


class SVDCache:
    """Canonical SVDs of the adapted layers of one set of weights, computed lazily once."""

    def __init__(self, weights):
        self.weights = weights
        self._cache = {}

    def __call__(self, layer):
        if layer not in self._cache:
            self._cache[layer] = canonical_svd(self.weights[layer])
        return self._cache[layer]
