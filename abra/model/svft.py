"""SVFT (Singular Vectors guided Fine-Tuning) for training class experts.

A frozen linear layer W = U diag(sigma) Vh is adapted as W' = U (diag(sigma) + M) Vh, where M is a learnable
banded k x k matrix (|i - j| <= off_diag), scaled by sigmoid(gate). Only used for *training*; evaluation
works on merged weights (theta + residual) with plain nn.Linear layers.

Numerically equivalent to the original torch-sparse implementation (``groundingdino/models/svft.py`` at
tag ``official``): same SVD call, same parameter layout and order, same random initialisation stream.
"""
import torch
import torch.nn.functional as F
from torch import nn

from abra.weights.svd import band_indices


class SVFTLinear(nn.Module):
    def __init__(self, linear, off_diag):
        super().__init__()
        u, sigma, vh = torch.linalg.svd(linear.weight.detach(), full_matrices=False)
        k = sigma.shape[0]
        rows, cols = band_indices(k, off_diag)
        self.register_buffer("u", u.contiguous())
        self.register_buffer("vh", vh.contiguous())
        self.register_buffer("sigma", sigma.contiguous())
        self.register_buffer("rows", rows.to(u.device))
        self.register_buffer("cols", cols.to(u.device))
        self.k = k
        self.bias = linear.bias
        # The original implementation drew a (2*off_diag+1, k) random matrix to build the band indices;
        # consuming it keeps the random initialisation identical for a given seed.
        # Both draws use the CPU generator, as the original parameters were created on CPU.
        torch.randn([2 * off_diag + 1, k])
        s = torch.zeros(rows.shape[0])
        nn.init.kaiming_normal_(s[None, :])
        self.s = nn.Parameter(s.to(u.device))
        self.gate = nn.Parameter(torch.tensor([0.0], device=u.device))

    def coefficients(self):
        """Dense k x k residual coefficients M (gate applied)."""
        m = torch.zeros(self.k, self.k, device=self.s.device, dtype=self.s.dtype)
        return m.index_put((self.rows, self.cols), self.s * torch.sigmoid(self.gate))

    @property
    def weight(self):
        # nn.MultiheadAttention reads out_proj.weight directly, so this must behave like Linear.weight
        return self.u @ (torch.diag(self.sigma) + self.coefficients()) @ self.vh

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def residual(self):
        """tau = U M Vh, the class-specific update of this layer (detached, float64, CPU)."""
        with torch.no_grad():
            return (self.u.double() @ self.coefficients().double() @ self.vh.double()).cpu()


def apply_svft(model, layer_names, off_diag):
    """Replace the given nn.Linear layers by SVFTLinear (in reverse order, as the original code did)."""
    for name in reversed(layer_names):
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, SVFTLinear(model.get_submodule(name), off_diag))


def svft_residuals(model, layer_names):
    return {name: model.get_submodule(name).residual() for name in layer_names}
