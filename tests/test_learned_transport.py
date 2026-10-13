"""Option 1 (learned transport from the available classes) on synthetic residuals: no checkpoints needed."""
import pytest
import torch

from abra.diagnostics.weight_space import d2_learned, d2_leave_one_out
from abra.transport import TransportContext, transport
from abra.transport.learned import (coordinates, fit_operator, load_operator, predict, save_operator,
                                    target_hash)
from abra.weights.io import Residual

LAYERS = {"enc.layers.1.proj": (12, 8), "enc.layers.1.sq": (8, 8), "enc.layers.0.untrained": (8, 8)}
VARIANTS = ["weight_left", "weight_full", "weight_lowrank2", "spectral_full", "spectral_band0", "spectral_band1"]


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _theta(seed=0):
    g = _gen(seed)
    return {l: torch.randn(*s, generator=g, dtype=torch.float64) for l, s in LAYERS.items()}


def _residual(g):
    return Residual({l: torch.randn(*s, generator=g, dtype=torch.float64) for l, s in LAYERS.items()})


def _pairs(n, target_fn, seed=1, noise=0.0):
    g = _gen(seed)
    pairs = {}
    for c in range(n):
        src = _residual(g)
        tgt = Residual({l: target_fn(l, d) + noise * torch.randn(*d.shape, generator=g, dtype=d.dtype)
                        for l, d in src.delta.items()})
        pairs[f"class{c}"] = (src, tgt)
    return pairs


def _fit(pairs, ctx, variant, lam, iters=5, **kw):
    layers = sorted(LAYERS)
    cache = coordinates(pairs, ctx, "spectral" if variant.startswith("spectral") else "weight", layers)
    return fit_operator(cache, list(pairs), variant, lam, iters, target_sha=target_hash(ctx, layers), **kw)


def _rel(a, b):
    return max(float((a[l] - b[l]).norm() / b[l].norm()) for l in b)


@pytest.mark.parametrize("variant", VARIANTS)
def test_strong_ridge_gives_task_addition_or_abra(variant):
    ctx = TransportContext(target=_theta())
    pairs = _pairs(3, lambda l, d: 1.7 * d + 0.3)
    op = _fit(pairs, ctx, variant, lam=1e12)
    new = _residual(_gen(9))
    reference = transport("abra" if variant.startswith("spectral") else "task_addition", new, ctx)
    assert _rel(predict(op, new, ctx), {l: r.double() for l, r in reference.items()}) < 1e-6


def test_recovers_a_two_sided_map():
    g = _gen(3)
    a_true = {l: torch.eye(s[0], dtype=torch.float64) + 0.1 * torch.randn(s[0], s[0], generator=g, dtype=torch.float64)
              for l, s in LAYERS.items()}
    b_true = {l: torch.eye(s[1], dtype=torch.float64) + 0.1 * torch.randn(s[1], s[1], generator=g, dtype=torch.float64)
              for l, s in LAYERS.items()}
    pairs = _pairs(30, lambda l, d: a_true[l] @ d @ b_true[l])
    ctx = TransportContext(target=_theta())
    op = _fit(pairs, ctx, "weight_full", lam=1e-9, iters=100, keep_layer0=True)
    new = _residual(_gen(11))
    truth = {l: a_true[l] @ d @ b_true[l] for l, d in new.delta.items()}
    assert _rel(predict(op, new, ctx), truth) < 1e-3


def test_banded_map_generalises_from_two_classes_better_than_full():
    theta = _theta()
    ctx = TransportContext(target=theta)
    g = _gen(5)
    scales = {l: 1 + 0.5 * torch.randn(min(s), generator=g, dtype=torch.float64) for l, s in LAYERS.items()}

    def target(l, d):  # truth: one scale per target singular direction (spectral_band0 family)
        u, _, vh = ctx.target_svd(l)
        return u @ (scales[l][:, None] * (u.T @ d @ vh.T)) @ vh

    pairs = _pairs(3, target, noise=0.02)
    out, _, _ = d2_learned(pairs, theta, ["spectral_band0", "spectral_full"], [1e-6, 1e-3, 1e-1])
    assert out["spectral_band0"]["best_error"] < 0.3
    assert out["spectral_band0"]["best_error"] < out["spectral_full"]["best_error"]


def test_layer0_is_left_at_the_identity():
    ctx = TransportContext(target=_theta())
    pairs = _pairs(3, lambda l, d: 2.0 * d)
    op = _fit(pairs, ctx, "weight_full", lam=1e-6)
    assert "enc.layers.0.untrained" in op.info["skipped"] and "enc.layers.0.untrained" not in op.left
    new = _residual(_gen(7))
    pred = predict(op, new, ctx)
    assert torch.equal(pred["enc.layers.0.untrained"], new.delta["enc.layers.0.untrained"])  # task addition there
    assert _rel({"x": pred["enc.layers.1.sq"]}, {"x": 2.0 * new.delta["enc.layers.1.sq"]}) < 1e-3


def test_lowrank_correction_has_the_requested_rank():
    ctx = TransportContext(target=_theta())
    op = _fit(_pairs(4, lambda l, d: d + 0.1 * d.T @ d if d.shape[0] == d.shape[1] else 1.2 * d), ctx,
              "weight_lowrank2", lam=1e-4)
    for layer, a in op.left.items():
        assert torch.linalg.matrix_rank(a - torch.eye(a.shape[0], dtype=a.dtype), tol=1e-9) <= 2
    assert not op.right


def test_save_load_registry_and_target_check(tmp_path):
    ctx = TransportContext(target=_theta())
    op = _fit(_pairs(3, lambda l, d: 1.3 * d), ctx, "spectral_band1", lam=1e-3)
    save_operator(tmp_path / "op.safetensors", op)
    loaded = load_operator(tmp_path / "op.safetensors")
    assert loaded.variant == "spectral_band1" and loaded.info["skipped"] == ["enc.layers.0.untrained"]
    new = _residual(_gen(13))
    out = transport("learned", new, ctx, operator=loaded)
    assert set(out) == set(LAYERS) and all(r.dtype == torch.float32 for r in out.values())
    assert _rel({l: r.double() for l, r in out.items()}, predict(op, new, ctx)) < 1e-5  # float32 storage
    with pytest.raises(ValueError, match="another target domain expert"):
        transport("learned", new, TransportContext(target=_theta(seed=99)), operator=loaded)
    with pytest.raises(ValueError, match="needs an operator"):
        transport("learned", new, ctx)


def test_d2_excludes_layer0_and_reports_learned_variants():
    theta = _theta()
    ctx = TransportContext(target=theta)
    g = _gen(21)

    def target(l, d):
        if ".layers.0." in l:  # as in the real experts: unrelated to the source residual (random initialisation)
            return torch.randn(*d.shape, generator=g, dtype=d.dtype)
        u, _, vh = ctx.target_svd(l)  # SVFT residuals trained on theta_T lie in its singular subspaces
        return 1.5 * u @ (u.T @ d @ vh.T) @ vh

    pairs = _pairs(3, target, noise=0.01)
    d2 = d2_leave_one_out(pairs, theta, transports=("task_addition",))
    assert abs(d2["mean"]["task_addition"] - 1.0) < 1e-6   # transports return float32
    with_layer0 = d2_leave_one_out(pairs, theta, transports=("task_addition",), keep_layer0=True)
    assert d2["mean"]["layer_scale_gamma0"] < with_layer0["mean"]["layer_scale_gamma0"]  # layer 0 only adds noise
    out, caches, _ = d2_learned(pairs, theta, ["weight_left", "spectral_band0"], [1e-4, 1.0])
    assert set(out) == {"weight_left", "spectral_band0"} and set(caches) == {"weight", "spectral"}
    assert all(r["best_lambda"] in (1e-4, 1.0) and r["best_error"] < 1 for r in out.values())
