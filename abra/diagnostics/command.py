"""``abra diagnose``: weight-space diagnostics D1-D3 and the Option-2 layer scales (no data, CPU)."""
import json

from abra.diagnostics.weight_space import (d1_residual_similarity, d2_leave_one_out, d3_domain_shift,
                                           fit_layer_scales, summarize)
from abra.weights.io import layer_weights, load_domain_expert, load_residual


def add_diagnose_args(p):
    p.add_argument("--source_domain", required=True, help="source domain expert theta_S")
    p.add_argument("--target_domain", required=True, help="target domain expert theta_T")
    p.add_argument("--pair", nargs=3, action="append", default=[], metavar=("CLASS", "SOURCE_RESIDUAL", "TARGET_RESIDUAL"),
                   help="available class with its residual trained on theta_S and on theta_T (repeat per class)")
    p.add_argument("--transports", nargs="+", default=["task_addition", "abra"], help="transports scored by D2")
    p.add_argument("--gamma", type=float, default=0.0, help="shrinkage towards 1 of the fitted layer scales")
    p.add_argument("--out", required=True, help="report json; with --pair also writes <out>.alpha.json (layer scales)")


def diagnose(args):
    theta_s, _ = load_domain_expert(args.source_domain)
    theta_t, _ = load_domain_expert(args.target_domain)
    pairs = {cls: (load_residual(s, parent=args.source_domain), load_residual(t, parent=args.target_domain))
             for cls, s, t in args.pair}
    layers = next(iter(pairs.values()))[0].layers if pairs else sorted(
        k[:-len(".weight")] for k in theta_s if k.endswith(".weight") and theta_s[k].dim() == 2)
    report = {"d3": d3_domain_shift(layer_weights(theta_s, layers), layer_weights(theta_t, layers))}
    if pairs:
        report["d1"] = d1_residual_similarity(pairs)
        report["d2"] = d2_leave_one_out(pairs, layer_weights(theta_t, layers), args.transports)
        alphas = fit_layer_scales(pairs, args.gamma)
        with open(f"{args.out}.alpha.json", "w") as f:
            json.dump(alphas, f, indent=1)
        report["layer_scales"] = {"file": f"{args.out}.alpha.json", "gamma": args.gamma,
                                  "mean": sum(alphas.values()) / len(alphas)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(summarize(report))
    print(f"[diagnose] report -> {args.out}")
    return report
