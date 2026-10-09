"""Weight-space diagnostics on synthetic residuals (no checkpoints needed)."""
import torch

from abra.diagnostics.weight_space import (d1_residual_similarity, d2_leave_one_out, d3_domain_shift,
                                           fit_layer_scales)
from abra.weights.io import Residual

LAYERS = {"a.weight_layer": (12, 8), "b.weight_layer": (8, 8)}


def _pairs(scale, noise=0.0, n=3):
    g = torch.Generator().manual_seed(0)
    pairs = {}
    for c in range(n):
        src = {l: torch.randn(*shape, generator=g) for l, shape in LAYERS.items()}
        tgt = {l: scale * s + noise * torch.randn(*s.shape, generator=g) for l, s in src.items()}
        pairs[f"class{c}"] = (Residual(src), Residual(tgt))
    return pairs


def test_layer_scales_recover_a_global_scale():
    alphas = fit_layer_scales(_pairs(1.3))
    assert all(abs(a - 1.3) < 1e-6 for a in alphas.values())
    assert all(abs(a - 1.0) < 1e-12 for a in fit_layer_scales(_pairs(1.3), gamma=1.0).values())


def test_d1_identical_residuals():
    d1 = d1_residual_similarity(_pairs(1.0))
    assert all(abs(r["mean_cos"] - 1) < 1e-9 and r["mean_rel_dist"] < 1e-9 for r in d1.values())


def test_d2_layer_scaling_beats_task_addition_when_target_is_scaled():
    pairs = _pairs(1.5, noise=0.05)
    theta_t = {l: torch.randn(*shape) for l, shape in LAYERS.items()}
    d2 = d2_leave_one_out(pairs, theta_t, transports=("task_addition",))
    assert abs(d2["mean"]["task_addition"] - 1.0) < 1e-9  # task addition is the reference by definition
    assert d2["mean"]["layer_scale_gamma0"] < 0.2


def test_d3_zero_shift():
    theta = {l: torch.randn(*shape) for l, shape in LAYERS.items()}
    d3 = d3_domain_shift(theta, theta, top_k=4)
    assert all(r["rel_shift"] == 0 and r["max_principal_angle_deg"] < 1e-3 for r in d3.values())
