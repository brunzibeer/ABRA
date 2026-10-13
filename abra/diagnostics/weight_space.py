"""Weight-space diagnostics D1-D3 and the layer-wise scaling of Option 2 (research plan, options.pdf).

Inputs are weights only: the source/target domain experts theta_S, theta_T and, for the *available* classes a,
pairs of residuals (tau_S^a trained on theta_S, tau_T^a trained on theta_T with target data).

D1  per class and layer: cos(tau_S, tau_T) and ||tau_T - tau_S|| / ||tau_T||
D2  leave-one-class-out error of a transport T fitted without class a':
        e(T) = sum_l ||T(tau_S^a') - tau_T^a'|| / sum_l ||tau_S^a' - tau_T^a'||     (< 1: better than task addition)
D3  per layer: ||theta_T - theta_S|| / ||theta_S|| and principal angles between their top singular subspaces
Option 2: alpha_l = sum_a <tau_S^a, tau_T^a> / sum_a ||tau_S^a||^2, shrunk towards 1 by gamma.
Option 1: learned per-layer maps (``abra/transport/learned.py``), scored by D2 for every variant and lambda.
Encoder layer 0 is never trained (its residuals are random SVFT initialisations): by default it is left out of the D2
errors and of the fits (alpha = 1, identity maps).
"""
import torch

from abra.transport import TransportContext, transport
from abra.transport.learned import coordinates, fit_operator, is_layer0, parse_variant, predict


def _f(t):
    return t.double()


def d1_residual_similarity(pairs):
    out = {}
    for cls, (tau_s, tau_t) in pairs.items():
        rows = {}
        for layer in tau_s.layers:
            s, t = _f(tau_s.delta[layer]).flatten(), _f(tau_t.delta[layer]).flatten()
            rows[layer] = {"cos": (s @ t / (s.norm() * t.norm() + 1e-30)).item(),
                           "rel_dist": ((t - s).norm() / (t.norm() + 1e-30)).item()}
        out[cls] = {"layers": rows,
                    "mean_cos": sum(r["cos"] for r in rows.values()) / len(rows),
                    "mean_rel_dist": sum(r["rel_dist"] for r in rows.values()) / len(rows)}
    return out


def d3_domain_shift(theta_s, theta_t, top_k=32):
    out = {}
    for layer in theta_s:
        s, t = _f(theta_s[layer]), _f(theta_t[layer])
        u_s = torch.linalg.svd(s, full_matrices=False)[0][:, :top_k]
        u_t = torch.linalg.svd(t, full_matrices=False)[0][:, :top_k]
        cosines = torch.linalg.svdvals(u_s.T @ u_t).clamp(max=1.0)
        angles = torch.rad2deg(torch.arccos(cosines))
        out[layer] = {"rel_shift": ((t - s).norm() / s.norm()).item(),
                      "max_principal_angle_deg": angles.max().item(),
                      "mean_principal_angle_deg": angles.mean().item()}
    return out


def fit_layer_scales(pairs, gamma=0.0, keep_layer0=False):
    """Option 2: least-squares scale per layer from available classes, shrunk towards 1 (task addition)."""
    layers = next(iter(pairs.values()))[0].layers
    alphas = {}
    for layer in layers:
        if is_layer0(layer) and not keep_layer0:
            alphas[layer] = 1.0
            continue
        num = sum((_f(s.delta[layer]) * _f(t.delta[layer])).sum() for s, t in pairs.values())
        den = sum((_f(s.delta[layer]) ** 2).sum() for s, _ in pairs.values())
        alpha = (num / den).item() if den > 0 else 1.0
        alphas[layer] = (1 - gamma) * alpha + gamma
    return alphas


def scored_layers(layers, keep_layer0=False):
    return [l for l in layers if keep_layer0 or not is_layer0(l)]


def _error(pred, tau_s, tau_t, layers=None):
    layers = tau_s.layers if layers is None else layers
    num = sum((_f(pred[l]) - _f(tau_t.delta[l])).norm() for l in layers)
    den = sum((_f(tau_s.delta[l]) - _f(tau_t.delta[l])).norm() for l in layers)
    return (num / den).item()


def _mean_row(out):
    methods = sorted({k for row in out.values() for k in row})
    return {m: sum(r[m] for r in out.values() if m in r) / sum(1 for r in out.values() if m in r) for m in methods}


def d2_leave_one_out(pairs, theta_t, transports=("task_addition", "abra"), gammas=(0.0, 0.5, 1.0), keep_layer0=False):
    """Leave-one-class-out error of fixed transports and of Option 2 for each shrinkage gamma."""
    ctx = TransportContext(target=theta_t)
    out = {}
    for held_out, (tau_s, tau_t) in pairs.items():
        layers = scored_layers(tau_s.layers, keep_layer0)
        row = {name: _error(transport(name, tau_s, ctx), tau_s, tau_t, layers) for name in transports}
        rest = {c: p for c, p in pairs.items() if c != held_out}
        for gamma in gammas if rest else ():
            alphas = fit_layer_scales(rest, gamma, keep_layer0)
            pred = {l: alphas[l] * _f(tau_s.delta[l]) for l in tau_s.layers}
            row[f"layer_scale_gamma{gamma:g}"] = _error(pred, tau_s, tau_t, layers)
        out[held_out] = row
    out["mean"] = _mean_row(out)
    return out


def d2_learned(pairs, theta_t, variants, lambdas, iters=5, band_out=None, keep_layer0=False):
    """Option 1: leave-one-class-out error of every learned variant at every lambda (fit on the other classes).
    Returns {variant: {"errors": {lambda: {class: e, "mean": e}}, "best_lambda", "best_error"}}."""
    ctx = TransportContext(target=theta_t)
    layers = next(iter(pairs.values()))[0].layers
    scored = scored_layers(layers, keep_layer0)
    caches = {}
    out = {}
    for variant in variants:
        coords = parse_variant(variant)[0]
        cache = caches.setdefault(coords, coordinates(pairs, ctx, coords, layers))
        errors = {}
        for lam in lambdas:
            row = {}
            for held_out, (tau_s, tau_t) in pairs.items():
                rest = [c for c in pairs if c != held_out]
                if not rest:
                    continue
                op = fit_operator(cache, rest, variant, lam, iters, band_out, keep_layer0)
                row[held_out] = _error(predict(op, tau_s, ctx, cache, held_out), tau_s, tau_t, scored)
            if row:
                row["mean"] = sum(row.values()) / len(row)
                errors[lam] = row
        best = min(errors, key=lambda lam: errors[lam]["mean"]) if errors else None
        out[variant] = {"errors": errors, "best_lambda": best, "best_error": errors[best]["mean"] if errors else None}
    return out, caches, ctx


def summarize(report):
    lines = []
    if "d1" in report:
        lines.append("D1 residual similarity (source vs target residual of the same class)")
        for cls, r in report["d1"].items():
            lines.append(f"  {cls:12s} mean cos {r['mean_cos']:.3f}   mean ||tau_T - tau_S||/||tau_T|| {r['mean_rel_dist']:.3f}")
    if "d2" in report:
        lines.append("D2 leave-one-class-out error (< 1 beats task addition)")
        for cls, r in report["d2"].items():
            lines.append(f"  {cls:12s} " + "  ".join(f"{m} {v:.3f}" for m, v in sorted(r.items())))
    if "learned" in report:
        lines.append("Option 1 learned transport: mean leave-one-class-out error per lambda (< 1 beats task addition; "
                     "the best lambda is chosen on these errors, so its error is slightly optimistic)")
        for variant, r in report["learned"].items():
            curve = "  ".join(f"{lam:g}:{e['mean']:.3f}" for lam, e in r["errors"].items())
            best = f"best lambda {r['best_lambda']:g} -> {r['best_error']:.3f}" if r["best_lambda"] is not None else "-"
            lines.append(f"  {variant:18s} {best}   [{curve}]")
    if "d3" in report:
        shifts = [r["rel_shift"] for r in report["d3"].values()]
        angles = [r["max_principal_angle_deg"] for r in report["d3"].values()]
        lines.append(f"D3 domain shift: mean ||theta_T - theta_S||/||theta_S|| {sum(shifts) / len(shifts):.4f} "
                     f"(max {max(shifts):.4f}); max principal angle {max(angles):.1f} deg")
    return "\n".join(lines)

