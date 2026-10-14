"""ABRA coefficient masks (abra/transport/masks.py) on synthetic layers: shapes, limits and equivalences."""
from types import SimpleNamespace

import pytest
import torch

from abra.transport import TransportContext, transport
from abra.transport.masks import mask_weights, parse_mask


def _setup(m=48, n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = {"a": torch.randn(m, n, generator=g, dtype=torch.float64),
              "b": torch.randn(24, 24, generator=g, dtype=torch.float64)}
    delta = {k: 1e-2 * torch.randn(*w.shape, generator=g, dtype=torch.float64) for k, w in target.items()}
    residual = SimpleNamespace(delta=delta, layers=list(delta), coefficients=None, info={"off_diag": 2},
                               source_sigma=None)
    return residual, TransportContext(target=target)


def _coeffs(residual, ctx, layer):
    u, _, vh = ctx.target_svd(layer)
    return u.T @ residual.delta[layer] @ vh.T


@pytest.mark.parametrize("spec", ["plain:1", "band", "band:1:2", "block:4:-0.1", "gauss:0", "exp:-1", "topk:0",
                                  "topk:1.5", "score:1:2", "nope", "band:x"])
def test_malformed_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        parse_mask(spec)


def test_hard_band_and_plain_match_the_band_mask():
    c = torch.randn(10, 10, dtype=torch.float64)
    torch.testing.assert_close(mask_weights("band:0", c), torch.eye(10, dtype=torch.float64))
    torch.testing.assert_close(mask_weights("plain", c), mask_weights("band:0", c))
    w = mask_weights("band:2", c)
    i = torch.arange(10)
    assert torch.equal(w.bool(), (i[:, None] - i[None, :]).abs() <= 2)
    torch.testing.assert_close(mask_weights("banded", c, off_diag=2), w)


def test_soft_masks_keep_the_diagonal_and_soften_the_rest():
    c = torch.randn(12, 12, dtype=torch.float64)
    for spec in ("band:0:0.3", "gauss:2", "exp:1.5", "block:5:0.2"):
        w = mask_weights(spec, c)
        assert torch.all(w.diagonal()[:5] == 1) and torch.all((w >= 0) & (w <= 1)), spec
    w = mask_weights("band:0:0.3", c)
    assert torch.allclose(w[~torch.eye(12, dtype=torch.bool)], torch.tensor(0.3, dtype=torch.float64))
    g = mask_weights("gauss:2", c)
    assert g[0, 1] > g[0, 2] > g[0, 5] > 0                       # fades with the distance from the diagonal
    b = mask_weights("block:5:0.2", c)
    assert b[4, 4] == 1 and b[5, 5] == 0.2 and b[0, 6] == 0.2


def test_score_trim_keeps_diagonal_and_drops_off_diagonal_outliers():
    c = 0.01 * torch.randn(16, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    c[3, 9], c[0, 0] = 5.0, -7.0                                   # one off-diagonal outlier, one big diagonal entry
    w = mask_weights("score", c)
    assert w[3, 9] == 0 and w[0, 0] == 1 and torch.all(w.diagonal() == 1)
    assert w.sum() > 0.8 * c.numel()                               # most off-diagonal entries survive
    assert torch.all(mask_weights("score:1e9", c) == 1)


def test_topk_keeps_the_requested_fraction():
    c = torch.randn(20, 20, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    w = mask_weights("topk:0.1", c)
    assert w.sum() == 40 and (c.abs()[w.bool()].min() >= c.abs()[~w.bool()].max())


@pytest.mark.parametrize("spec", ["band:1000", "band:0:1", "block:1000", "score:1e9", "topk:1", "gauss:1e6"])
def test_masks_that_keep_everything_equal_unmasked_abra(spec):
    residual, ctx = _setup()
    ref = transport("abra", residual, ctx)
    out = transport("abra", residual, ctx, mask=spec)
    for layer in ref:
        torch.testing.assert_close(out[layer], ref[layer], rtol=1e-5, atol=1e-7)


def test_masked_abra_is_no_longer_task_addition_on_square_layers():
    residual, ctx = _setup()
    unmasked = transport("abra", residual, ctx)
    torch.testing.assert_close(unmasked["b"], residual.delta["b"].float(), rtol=1e-4, atol=1e-7)
    for spec in ("plain", "band:0:0.5", "gauss:1", "score"):
        out = transport("abra", residual, ctx, mask=spec)
        rel = (out["b"].double() - residual.delta["b"]).norm() / residual.delta["b"].norm()
        assert rel > 1e-2, spec


def test_mask_is_applied_in_the_target_basis():
    residual, ctx = _setup()
    out = transport("abra", residual, ctx, mask="band:0:0.5")
    u, _, vh = ctx.target_svd("a")
    c = _coeffs(residual, ctx, "a")
    expected = u @ (c * mask_weights("band:0:0.5", c)) @ vh
    torch.testing.assert_close(out["a"], expected.float())


def test_bad_spec_fails_before_any_work():
    residual, ctx = _setup()
    with pytest.raises(ValueError, match="unknown ABRA mask"):
        transport("abra", residual, ctx, mask="diagonal")
