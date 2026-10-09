"""Weight formats and transports on the released checkpoints.

Needs ``$ABRA_WEIGHTS`` (default ``../weights``) with cityscape_normal/, cityscape_fog/ and
cityscapes_experts/exp1_<cls>/best.pth; skipped otherwise.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from abra.transport import TransportContext, transport
from abra.weights.io import (layer_weights, load_domain_expert, load_residual, save_domain_expert, save_residual)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("ABRA_WEIGHTS", ROOT.parent / "weights"))
NORMAL, FOG = WEIGHTS / "cityscape_normal/best.pth", WEIGHTS / "cityscape_fog/best.pth"
EXPERT = WEIGHTS / "cityscapes_experts/exp1_train/best.pth"
GOLDEN = ROOT / "tests/golden/offline_golden.json"

pytestmark = pytest.mark.skipif(not EXPERT.exists(), reason=f"released weights not found in {WEIGHTS}")


@pytest.fixture(scope="module")
def legacy():
    return load_residual(EXPERT)


@pytest.fixture(scope="module")
def migrated(tmp_path_factory, legacy):
    parent, parent_info = load_domain_expert(NORMAL)
    out = tmp_path_factory.mktemp("w") / "train.safetensors"
    save_residual(out, legacy.delta, layer_weights(parent, legacy.layers), parent_info, 2, {"class": "train"})
    return out


def test_domain_expert_roundtrip(tmp_path):
    tensors, info = load_domain_expert(NORMAL)
    out = tmp_path / "normal.safetensors"
    save_domain_expert(out, tensors, {"domain": "normal"})
    again, again_info = load_domain_expert(out)
    assert again_info["sha256"] == info["sha256"]
    assert all(torch.equal(again[k], tensors[k]) for k in tensors)


def test_band_residual_is_small_and_faithful(migrated, legacy):
    assert migrated.stat().st_size < 1_000_000  # vs 93 MB for the original checkpoint
    decoded = load_residual(migrated, parent=NORMAL)
    for layer in legacy.layers:
        err = (decoded.delta[layer] - legacy.delta[layer]).norm() / legacy.delta[layer].norm()
        assert err < 5e-3, (layer, err.item())


def test_band_residual_rejects_wrong_parent(migrated):
    with pytest.raises(ValueError, match="hash"):
        load_residual(migrated, parent=FOG)


def _fingerprint_close(residual_out, golden, rtol):
    """Compare dense residuals with the fingerprints of tests/golden/offline_golden.json."""
    from abra.weights.io import _stable_seed
    worst = 0.0
    for layer, fp in golden.items():
        dw = residual_out[layer].double()
        g = torch.Generator().manual_seed(_stable_seed(layer))
        x = torch.randn(8, dw.shape[1], generator=g, dtype=torch.float64)
        y = torch.randn(8, dw.shape[0], generator=g, dtype=torch.float64)
        probes = torch.einsum("pi,ij,pj->p", y, dw, x)
        worst = max(worst, abs(dw.norm().item() - fp["fro"]) / fp["fro"],
                    *((probes - torch.tensor(fp["probes"], dtype=torch.float64)).abs() / fp["fro"]).tolist())
    assert worst < rtol, worst
    return worst


@pytest.mark.parametrize("method,options,golden_key,rtol", [
    ("abra", {}, "abra_none", 1e-4),
    ("abra", {"mask": "banded"}, "abra_banded", 5e-3),
    ("abra", {"mask": "plain"}, "abra_plain", 5e-3),
])
def test_transports_match_original(legacy, method, options, golden_key, rtol):
    golden = json.loads(GOLDEN.read_text())["experts"]["exp1_train"][golden_key]
    fog, _ = load_domain_expert(FOG)
    ctx = TransportContext(target=layer_weights(fog, legacy.layers))
    _fingerprint_close(transport(method, legacy, ctx, **options), golden, rtol)


def test_abra_without_mask_is_projection(legacy):
    """ABRA (no mask) == P_U tau P_V; equals task addition on square layers."""
    fog, _ = load_domain_expert(FOG)
    ctx = TransportContext(target=layer_weights(fog, legacy.layers))
    abra = transport("abra", legacy, ctx)
    for layer, tau in legacy.delta.items():
        if tau.shape[0] == tau.shape[1]:
            torch.testing.assert_close(abra[layer], tau, rtol=1e-4, atol=1e-6)


def test_task_addition_differs_from_original_only_by_reconstruction_error(legacy):
    """The original 'analogy' added U(S_pre + S)V - theta_S = tau + (U S_pre V - theta_S): the class residual
    plus the float32 SVD reconstruction error of theta_S (~1% of ||tau||). The new task addition adds tau."""
    golden = json.loads(GOLDEN.read_text())["experts"]["exp1_train"]["task_addition"]
    normal, _ = load_domain_expert(NORMAL)
    raw = torch.load(EXPERT, map_location="cpu")
    original = {}
    for layer in legacy.layers:
        u, s_pre, s, v = (raw[f"{layer}.svft_layer.{x}"] for x in ("u", "s_pre", "s", "v"))
        original[layer] = u @ (s_pre + s) @ v - normal[f"{layer}.weight"]
    _fingerprint_close(original, golden, 1e-4)
    fog, _ = load_domain_expert(FOG)
    new = transport("task_addition", legacy, TransportContext(target=layer_weights(fog, legacy.layers)))
    assert _fingerprint_close(new, golden, 3e-2) > 1e-3  # the reconstruction error is gone
