"""Phase-0 reference for the refactor: weight-space outputs of the *current* transporters.

Runs on CPU, needs no dataset and no compiled ops. For every class expert and every method it
computes the class residual that ends up in the target model, dW = W_eval - theta_T restricted to the
method part (U_T s_T V_T for SVFT-based transports), and stores a compact fingerprint per layer:
Frobenius norm and 8 fixed random bilinear probes y^T dW x. The refactored code must reproduce these.

Usage (from the repo root):
    python tests/golden/make_offline_golden.py \
        --source_domain ../weights/cityscape_normal/best.pth \
        --target_domain ../weights/cityscape_fog/best.pth \
        --experts ../weights/cityscapes_experts/exp1_*/best.pth \
        --out tests/golden/offline_golden.json

Notes:
- ABRA outputs are invariant to SVD sign choices, so CPU vs GPU SVD does not matter for them.
- 'simple' (Param Delta) copies the source band into the *target* singular basis without alignment,
  so its output depends on the arbitrary SVD signs of both experts. Its fingerprint is recorded for
  reference only and is not a valid regression target.
"""
import argparse
import glob
import json
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# The current transporter module imports wandb only for logging; stub it if not installed.
try:
    import wandb  # noqa: F401
except ImportError:
    sys.modules["wandb"] = types.SimpleNamespace(run=None)

from utils.transporter import RotationTransporter, SimpleTransporter  # noqa: E402
from utils.transporter_functions import find_unique_layers, merge_svft_layers  # noqa: E402

N_PROBES = 8


def svft_eval_state(domain_sd, layers):
    """Mimic the state dict of SVFT layers in eval/transporter mode built on a domain expert."""
    sd = {}
    for layer in layers:
        u, s, vh = torch.linalg.svd(domain_sd[f"{layer}.weight"].float(), full_matrices=False)
        sd[f"{layer}.svft_layer.u"] = u
        sd[f"{layer}.svft_layer.s_pre"] = torch.diag(s)
        sd[f"{layer}.svft_layer.s"] = torch.zeros_like(torch.diag(s))
        sd[f"{layer}.svft_layer.v"] = vh
    return sd


def fingerprint(dw, layer):
    g = torch.Generator().manual_seed(_stable_seed(layer))
    x = torch.randn(N_PROBES, dw.shape[1], generator=g, dtype=torch.float64)
    y = torch.randn(N_PROBES, dw.shape[0], generator=g, dtype=torch.float64)
    dw = dw.double()
    probes = torch.einsum("pi,ij,pj->p", y, dw, x)
    return {"fro": dw.norm().item(), "probes": probes.tolist()}


def _stable_seed(name):
    h = 0
    for c in name.encode():
        h = (h * 131 + c) % (2**31 - 1)
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_domain", type=Path, required=True)
    parser.add_argument("--target_domain", type=Path, required=True)
    parser.add_argument("--experts", nargs="+", required=True, help="class expert checkpoints (globs allowed)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    experts = sorted({p for e in args.experts for p in glob.glob(str(e))})
    theta_s = torch.load(args.source_domain, map_location="cpu")
    theta_t = torch.load(args.target_domain, map_location="cpu")

    out = {"source_domain": str(args.source_domain), "target_domain": str(args.target_domain),
           "n_probes": N_PROBES, "experts": {}}
    for path in experts:
        name = Path(path).parent.name
        w_source = torch.load(path, map_location="cpu")
        layers = sorted(find_unique_layers(w_source))
        w_target = svft_eval_state(theta_t, layers)

        def residual_from_s(s_dict):
            return {l: w_target[f"{l}.svft_layer.u"] @ s_dict[f"{l}.svft_layer.s"] @ w_target[f"{l}.svft_layer.v"]
                    for l in layers}

        merged = merge_svft_layers(w_source, layers, "cpu")
        methods = {
            "abra_none": residual_from_s(RotationTransporter(w_source, dict(w_target), mask=None).transport()),
            "abra_banded": residual_from_s(RotationTransporter(w_source, dict(w_target), mask="banded").transport()),
            "abra_plain": residual_from_s(RotationTransporter(w_source, dict(w_target), mask="plain").transport()),
            "simple_signdependent": residual_from_s(SimpleTransporter(w_source, dict(w_target)).transport()),
            # task addition (what the 'analogy' transporter computes): theta_T + (merged_source - theta_S)
            "task_addition": {l: merged[f"{l}.weight"] - theta_s[f"{l}.weight"] for l in layers},
        }
        out["experts"][name] = {m: {l: fingerprint(dw, l) for l, dw in res.items()} for m, res in methods.items()}
        print(f"{name}: " + ", ".join(
            f"{m}={sum(v['fro'] ** 2 for v in fp.values()) ** 0.5:.3f}" for m, fp in out["experts"][name].items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
