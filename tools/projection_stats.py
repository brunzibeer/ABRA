"""Projection statistics of the ABRA transport (revision E9), CPU only.

For every adapted layer and every class residual tau (source class expert, basis (U_S, V_S) of theta_S):

    rel_change = ||P_U tau P_V - tau||_F / ||tau||_F,  P_U = U_T U_T^T,  P_V = V_T V_T^T
    L_orth_err = ||L^T L - I||_F,  L = U_T^T U_S
    R_orth_err = ||R^T R - I||_F,  R = V_T^T V_S

(U_T, V_T: canonical float64 SVD of theta_T, as in ``abra eval --method abra``; full_matrices=False.)

    python tools/projection_stats.py --weights <theta_0> --target_domain <theta_T> \
        --expert bus=<class expert> --expert train=<class expert> ... --out projection_stats.csv

Writes one row per layer x class and prints min / median / max per layer shape.
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abra.weights.io import load_domain_expert, load_residual  # noqa: E402
from abra.weights.svd import SVDCache  # noqa: E402

COLUMNS = ["layer_name", "shape", "class", "rel_change", "L_orth_err", "R_orth_err"]


def target_weights(base_path, target_path, layers):
    """Adapted-layer weights of theta_T: the domain expert's tensors over theta_0 (what eval loads)."""
    base = torch.load(base_path, map_location="cpu")["model"]
    base = {k[7:] if k.startswith("module.") else k: v for k, v in base.items()}
    tensors, _ = load_domain_expert(target_path)
    return {l: tensors.get(f"{l}.weight", base[f"{l}.weight"]) for l in layers}


def layer_stats(tau, u_t, vh_t, u_s, vh_s):
    tau = tau.double()
    projected = u_t @ (u_t.T @ tau @ vh_t.T) @ vh_t
    rel = ((projected - tau).norm() / tau.norm()).item()
    lmat, rmat = u_t.T @ u_s.double(), vh_t @ vh_s.double().T
    eye = torch.eye(lmat.shape[1], dtype=torch.float64)
    return rel, (lmat.T @ lmat - eye).norm().item(), (rmat.T @ rmat - eye).norm().item()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True, help="base Grounding DINO checkpoint (theta_0)")
    p.add_argument("--target_domain", type=Path, required=True, help="target domain expert theta_T")
    p.add_argument("--expert", action="append", required=True, metavar="CLASS=PATH", help="source class expert")
    p.add_argument("--source_domain", type=Path, default=None,
                   help="parent of new-format residuals, if the path stored in them is not valid here")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    experts = dict(e.split("=", 1) for e in args.expert)
    rows, svd = [], None
    for cls, path in experts.items():
        residual = load_residual(path, parent=args.source_domain)
        if residual.basis is None:
            raise SystemExit(f"{path}: dense residual without a source basis")
        if svd is None:
            svd = SVDCache(target_weights(args.weights, args.target_domain, residual.layers))
        for layer in residual.layers:
            u_t, _, vh_t = svd(layer)
            u_s, vh_s = residual.basis[layer]
            rel, l_err, r_err = layer_stats(residual.delta[layer], u_t, vh_t, u_s, vh_s)
            shape = "x".join(str(n) for n in residual.delta[layer].shape)
            rows.append({"layer_name": layer, "shape": shape, "class": cls, "rel_change": rel,
                         "L_orth_err": l_err, "R_orth_err": r_err})
        print(f"[projection] {cls}: {len(residual.layers)} layers", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    by_shape = defaultdict(list)
    for r in rows:
        by_shape[r["shape"]].append(r)
    print("| shape | layers | metric | min | median | max |\n|---|---|---|---|---|---|")
    for shape, rs in sorted(by_shape.items()):
        n_layers = len({r["layer_name"] for r in rs})
        for m in ("rel_change", "L_orth_err", "R_orth_err"):
            v = [r[m] for r in rs]
            print(f"| {shape} | {n_layers} | {m} | {min(v):.3e} | {statistics.median(v):.3e} | {max(v):.3e} |")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
