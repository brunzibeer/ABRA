"""Command helpers without the detector: theta_0 as SVFT parent, alpha search metric, merged transport output."""
from types import SimpleNamespace

import pytest
import torch

from abra import commands
from abra.weights.io import content_hash, load_domain_expert


def test_theta0_parent_hash_matches_the_checkpoint_file(tmp_path):
    state = {"module.a.weight": torch.randn(4, 3), "b.bias": torch.randn(4)}
    path = tmp_path / "base.pth"
    torch.save({"model": state}, path)
    theta_0 = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}  # as load_base_weights
    _, info = commands.svft_parent(SimpleNamespace(domain=None, weights=path), model=None, theta_0=theta_0)
    # residuals store this hash; decoding them later loads the parent from the path and must find the same hash
    assert info["sha256"] == load_domain_expert(path)[1]["sha256"] == content_hash(theta_0)
    assert info["path"] == str(path) and info["kind"] == "base"


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2, bias=False)


@pytest.mark.parametrize("metric,best", [("AP", 1.5), ("AP50", 0.7), ("AP75", 2.2)])
def test_alpha_search_maximises_the_chosen_metric(monkeypatch, metric, best):
    model = Tiny()
    w0 = model.lin.weight.detach().clone()
    residual = {"lin": torch.ones(2, 2)}

    def fake_eval(model, *a, **k):  # each metric peaks at a different alpha
        alpha = (model.lin.weight.detach() - w0).mean().item()
        return {"AP": -(alpha - 1.5) ** 2, "AP50": -(alpha - 0.7) ** 2, "AP75": -(alpha - 2.2) ** 2}

    monkeypatch.setattr("abra.eval.evaluate.evaluate", fake_eval)
    args = SimpleNamespace(alpha_annotations=None, min_alpha=0.1, max_alpha=3.0, alpha_step=0.1, alpha_patience=5,
                           alpha_metric=metric)
    out = commands._alpha_search(args, model, None, residual, "cpu", SimpleNamespace(log=lambda *a, **k: None), None, None)
    assert out["alpha"] == pytest.approx(best) and out["alpha_metric"] == metric and out["oracle"]
    assert all(set(r) == {"alpha", "AP", "AP50", "AP75"} for r in out["alpha_curve"])   # every alpha, all metrics
    assert max(out["alpha_curve"], key=lambda r: r[metric])["alpha"] == pytest.approx(best)
    assert (model.lin.weight - w0).mean().item() == pytest.approx(best, abs=1e-5)  # model left at the best alpha


def test_merge_matches_evaluation_arithmetic():
    from abra.model.compose import add_residual
    model = Tiny()
    target = {"lin.weight": model.lin.weight.detach().clone(), "other.bias": torch.randn(3)}
    r = {"lin": torch.randn(2, 2)}
    before = target["lin.weight"].clone()
    merged = commands.merge(target, r, 1.3)
    with torch.no_grad():
        add_residual(model, r, 1.3)
    assert torch.equal(merged["lin.weight"], model.lin.weight.detach())  # bit-identical to eval --alpha 1.3
    assert torch.equal(target["lin.weight"], before)  # the target tensors are not modified
    assert merged["other.bias"] is target["other.bias"]  # untouched tensors are kept
