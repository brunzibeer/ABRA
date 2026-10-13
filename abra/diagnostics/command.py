"""``abra diagnose``: weight-space diagnostics D1-D3, the Option-2 layer scales and the Option-1 learned transports
(no data, CPU)."""
import json

from abra.diagnostics.weight_space import (d1_residual_similarity, d2_learned, d2_leave_one_out, d3_domain_shift,
                                           fit_layer_scales, summarize)
from abra.transport.learned import deviation, fit_operator, save_operator, target_hash
from abra.weights.io import layer_weights, load_domain_expert, load_residual


def add_diagnose_args(p):
    p.add_argument("--source_domain", required=True, help="source domain expert theta_S")
    p.add_argument("--target_domain", required=True, help="target domain expert theta_T")
    p.add_argument("--pair", nargs=3, action="append", default=[], metavar=("CLASS", "SOURCE_RESIDUAL", "TARGET_RESIDUAL"),
                   help="available class with its residual trained on theta_S and on theta_T (repeat per class)")
    p.add_argument("--transports", nargs="+", default=["task_addition", "abra"], help="transports scored by D2")
    p.add_argument("--gamma", type=float, default=0.0, help="shrinkage towards 1 of the fitted layer scales")
    p.add_argument("--learned", nargs="*", default=[],
                   help="Option 1 variants scored by D2 (weight_left, weight_full, weight_lowrank<r>, spectral_full, "
                        "spectral_band<w>); each writes <out>.<variant>.operator.safetensors fitted on every --pair at "
                        "its best lambda")
    p.add_argument("--lambdas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
                   help="relative ridge strengths tried for --learned (selected by the leave-one-class-out error)")
    p.add_argument("--iters", type=int, default=5, help="alternating steps of the two-sided variants")
    p.add_argument("--band_out", type=int, default=None,
                   help="spectral variants: keep only the predicted target coefficients within this band")
    from abra.cli import str2bool
    p.add_argument("--keep_layer0", type=str2bool, default=False,
                   help="also fit and score encoder layer 0 (never trained: its residuals are random initialisations)")
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
        report["d2"] = d2_leave_one_out(pairs, layer_weights(theta_t, layers), args.transports,
                                        keep_layer0=args.keep_layer0)
        if args.learned:
            learned, caches, ctx = d2_learned(pairs, layer_weights(theta_t, layers), args.learned, args.lambdas,
                                              args.iters, args.band_out, args.keep_layer0)
            sha = target_hash(ctx, layers)
            for variant, r in learned.items():
                if r["best_lambda"] is None:
                    continue
                coords = "spectral" if variant.startswith("spectral") else "weight"
                op = fit_operator(caches[coords], list(pairs), variant, r["best_lambda"], args.iters, args.band_out,
                                  args.keep_layer0, sha)
                op.info.update({"source_domain": str(args.source_domain), "target_domain": str(args.target_domain),
                                "loo_error": r["best_error"]})
                path = f"{args.out}.{variant}.operator.safetensors"
                save_operator(path, op)
                r.update(operator=path, deviation=deviation(op))
            report["learned"] = learned
        alphas = fit_layer_scales(pairs, args.gamma, args.keep_layer0)
        with open(f"{args.out}.alpha.json", "w") as f:
            json.dump(alphas, f, indent=1)
        report["layer_scales"] = {"file": f"{args.out}.alpha.json", "gamma": args.gamma,
                                  "mean": sum(alphas.values()) / len(alphas)}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(summarize(report))
    print(f"[diagnose] report -> {args.out}")
    return report
