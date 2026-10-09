"""The torch-sparse-free SVFT layer must match the original implementation (tag ``official``)."""
import pytest
import torch

from abra.model.svft import SVFTLinear
from abra.weights.svd import band_indices

torch_sparse = pytest.importorskip("torch_sparse")
legacy_svft = pytest.importorskip("legacy_svft")


@pytest.mark.parametrize("k,d", [(8, 0), (8, 2), (16, 3)])
def test_band_order_matches_spdiags(k, d):
    offsets = torch.cat([torch.arange(0, d + 1), torch.arange(-1, -d - 1, -1)])
    idx = torch.sparse.spdiags(torch.randn(2 * d + 1, k), offsets, (k, k)).coalesce().indices()
    rows, cols = band_indices(k, d)
    assert torch.equal(idx[0], rows) and torch.equal(idx[1], cols)


@pytest.mark.parametrize("shape", [(32, 32), (16, 48), (48, 16)])
def test_svft_matches_legacy(shape):
    linear = torch.nn.Linear(shape[1], shape[0])
    x = torch.randn(5, shape[1])

    torch.manual_seed(0)
    old = legacy_svft.LinearWithSVFT(linear, off_diag=2, pattern="banded")
    torch.manual_seed(0)
    new = SVFTLinear(linear, off_diag=2)

    assert torch.equal(old.svft_layer.s.detach(), new.s.detach()), "random init differs"
    with torch.no_grad():  # give the residual a non-trivial value
        for p_old, p_new in ((old.svft_layer.s, new.s), (old.svft_layer.gate, new.gate)):
            p_old.add_(0.3)
            p_new.add_(0.3)
    y_old, y_new = old(x), new(x)
    torch.testing.assert_close(y_new, y_old, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new.weight, old.weight, rtol=1e-5, atol=1e-5)

    y_old.square().sum().backward()
    y_new.square().sum().backward()
    torch.testing.assert_close(new.s.grad, old.svft_layer.s.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(new.gate.grad, old.svft_layer.gate.grad, rtol=1e-4, atol=1e-5)

    # the residual W' - W equals U M Vh with M the gated band
    tau = new.residual()
    torch.testing.assert_close(tau.float(), (new.weight - linear.weight).detach(), rtol=1e-4, atol=1e-5)
