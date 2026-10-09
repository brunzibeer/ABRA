"""Implementation of the CLI commands (see ``abra/cli.py`` for the options)."""
import json
from pathlib import Path

import numpy as np
import torch

from abra.data import datasets as data
from abra.model import build
from abra.model.compose import add_residual, restore, snapshot
from abra.model.layers import (adapted_linear_layers, freeze_all, parameter_groups, selected_modules,
                               trainable_keys, unfreeze_modules)
from abra.model.svft import apply_svft, svft_residuals
from abra.train.loop import fit
from abra.train.optim import build_optimizer, build_scheduler
from abra.transport import TransportContext, transport
from abra.utils.checkpoint import load_checked
from abra.utils.run import RunLogger, seed_everything
from abra.weights.io import (layer_weights, load_domain_expert, load_residual, save_domain_expert, save_residual)


def _setup(args, command):
    if "cuda" in args.device and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available: pass --device cpu or check the CUDA installation.")
    logger = RunLogger(args.output_root, args, command)
    seed_everything(args.seed)
    # PostProcess builds its label -> token map from the categories of the evaluation json
    args.coco_val_path = getattr(args, "test_annotations", None) or getattr(args, "target_test_annotations", None)
    build.merge_model_config(args)
    return logger, torch.device(args.device)


def _model(args, device):
    model, postprocessors = build.build_model(args)
    model.to(device)
    theta_0 = build.load_base_weights(model, args.weights)
    return model, postprocessors, theta_0


def _test(args):
    dataset = data.test_dataset(args)
    return data.test_loader(dataset, args), data.coco_api(dataset)


def _training(args, device, model, postprocessors, logger, save):
    criterion = build.build_criterion(args)
    train_set = data.train_dataset(args)
    loaders = {"train": data.train_loader(train_set, args)}
    loaders["test"], loaders["test_gt"] = _test(args)
    if args.val_annotations:
        val_set = data.test_dataset(args, annotations=args.val_annotations)
        loaders["val"], loaders["val_gt"] = data.test_loader(val_set, args), data.coco_api(val_set)
    optimizer = build_optimizer(args, parameter_groups(model, args))
    scheduler = build_scheduler(args, optimizer, len(loaders["train"]) // args.accumulation_steps)
    final = fit(model, criterion, postprocessors, optimizer, scheduler, loaders, args, device, logger, save)
    logger.close()
    return final


def _provenance(args, extra=None):
    return {"exp_name": args.exp_name, "train_annotations": args.train_annotations, "label_map": args.label_map,
            "root": args.root, "base_weights": args.weights, "trainable": args.trainable, "seed": args.seed,
            **(extra or {})}


# ---------------------------------------------------------------------------- training

def train_domain(args):
    logger, device = _setup(args, "train-domain")
    model, postprocessors, _ = _model(args, device)
    if args.init:
        tensors, _ = load_domain_expert(args.init)
        load_checked(model, tensors, name=f"init {args.init}")
    freeze_all(model)
    unfreeze_modules(model, selected_modules(model, args.trainable))
    keys = trainable_keys(model)
    print(f"[train] {len(keys)} trainable tensors ({args.trainable})")

    def save(name, ap):
        state = model.state_dict()
        info = _provenance(args, {"objectify": args.objectify, "select_AP": ap, "checkpoint": name})
        save_domain_expert(args.output_root / f"{name}.safetensors", {k: state[k] for k in keys}, info)

    return _training(args, device, model, postprocessors, logger, save)


def train_expert(args):
    args.objectify = False  # class experts train on the class captions of the label map
    logger, device = _setup(args, "train-expert")
    model, postprocessors, _ = _model(args, device)
    parent, parent_info = load_domain_expert(args.domain)
    load_checked(model, parent, name=f"parent domain expert {args.domain}")
    freeze_all(model)
    layers = adapted_linear_layers(model, args.trainable)
    parent_weights = layer_weights(parent, layers)
    apply_svft(model, layers, args.svft_diag, zero_init=args.svft_zero_init)
    print(f"[train] SVFT on {len(layers)} layers (band half-width {args.svft_diag})")

    def save(name, ap):
        info = _provenance(args, {"select_AP": ap, "checkpoint": name})
        info = save_residual(args.output_root / f"{name}.safetensors", svft_residuals(model, layers), parent_weights,
                             parent_info, args.svft_diag, info, encoding=args.residual_encoding)
        if info.get("max_off_band_energy", 0) > 1e-3:
            print(f"[save] WARNING: {info['max_off_band_energy']:.2e} of a residual's energy is outside the band")

    return _training(args, device, model, postprocessors, logger, save)


# ---------------------------------------------------------------------------- evaluation

def _apply(model, r, alpha, args, device):
    """theta_T + alpha * R on the adapted layers (alpha: float or layer -> float)."""
    add_residual(model, r, alpha)


def _transported(args, model, device, residual):
    layers = residual.layers
    target = {l: model.get_submodule(l).weight.detach().cpu() for l in layers}
    source = None
    if args.alpha_estimate and residual.source_sigma is None:
        source = layer_weights(load_domain_expert(args.source_domain or residual.info["parent"])[0], layers)
    ctx = TransportContext(target=target, source=source, device=str(device))
    options = {"mask": args.abra_mask, "norm_preserving": args.alpha_estimate} if args.method == "abra" else {}
    return transport(args.method, residual, ctx, **options)


def evaluate(args):
    from abra.eval.evaluate import evaluate as run_eval

    logger, device = _setup(args, "eval")
    model, postprocessors, _ = _model(args, device)
    if args.domain:
        tensors, _ = load_domain_expert(args.domain)
        load_checked(model, tensors, name=f"domain expert {args.domain}")
    loader, coco_gt = _test(args)

    result = {"method": "zero_shot" if not args.domain else "domain_expert", "alpha": None, "oracle": False}
    if args.residual:
        residual = load_residual(args.residual, parent=args.source_domain)
        check = residual.info.get("sketch_error")
        print(f"[residual] {args.residual}: encoding {residual.info.get('encoding')}"
              + (f", checksum error {check:.1e}" if check is not None else " (original checkpoint, no checksum)"))
        r = _transported(args, model, device, residual)
        result.update(method=args.method, alpha=args.alpha)
        if args.alpha_file:
            alpha = json.loads(Path(args.alpha_file).read_text())
            missing = set(r) - set(alpha)
            if missing:
                raise SystemExit(f"--alpha_file has no scale for {len(missing)} layers, e.g. {sorted(missing)[:2]}")
            result.update(alpha=f"per-layer ({args.alpha_file})")
            _apply(model, r, alpha, args, device)
        elif args.alpha_search:
            result.update(_alpha_search(args, model, postprocessors, r, device, logger, loader, coco_gt))
        else:
            _apply(model, r, args.alpha, args, device)

    stats = run_eval(model, postprocessors, loader, coco_gt, args, device, logger, epoch=0, max_epoch=0, tag="TEST")
    result.update({k: stats[k] for k in ("AP", "AP50", "AP75", "APs", "APm", "APl", "per_category")})
    (args.output_root / "results.json").write_text(json.dumps(result, indent=2, default=str))
    logger.log({"result": result})
    print(f"[result] {result['method']}: AP {stats['AP']:.3f}  AP50 {stats['AP50']:.3f}  AP75 {stats['AP75']:.3f}"
          + (f"  (alpha {result['alpha']}{', ORACLE' if result['oracle'] else ''})" if result["alpha"] else ""))
    logger.close()
    return result


def _alpha_search(args, model, postprocessors, residual, device, logger, test_loader, test_gt):
    """ABRA-alpha: evaluate alpha in [min, max) and keep the best (same procedure as the original code)."""
    from abra.eval.evaluate import evaluate as run_eval

    if args.alpha_annotations:
        dataset = data.test_dataset(args, annotations=args.alpha_annotations)
        loader, gt, oracle = data.test_loader(dataset, args), data.coco_api(dataset), False
    else:
        loader, gt, oracle = test_loader, test_gt, True
        print("[alpha] WARNING: alpha is selected on the TEST split (oracle); pass --alpha_annotations to avoid it")
    base = snapshot(model, residual)
    alphas = np.arange(args.min_alpha, args.max_alpha, args.alpha_step).tolist()
    best_alpha, best_ap, last_ap, worse = alphas[0], -float("inf"), -float("inf"), 0
    for alpha in alphas:
        restore(model, base)
        _apply(model, residual, alpha, args, device)
        ap = run_eval(model, postprocessors, loader, gt, args, device, logger=None)["AP"]
        logger.log({"alpha/alpha": alpha, "alpha/AP": ap})
        if ap > best_ap:
            best_alpha, best_ap, worse = alpha, ap, 0
        elif ap < last_ap:
            worse += 1
        last_ap = ap
        print(f"[alpha] alpha={alpha:.3f} AP={ap:.3f}" + (" <- best" if best_alpha == alpha else ""))
        if worse >= args.alpha_patience:
            print(f"[alpha] early stop: AP decreased {worse} times since the best")
            break
    restore(model, base)
    _apply(model, residual, best_alpha, args, device)
    return {"alpha": best_alpha, "oracle": oracle, "alpha_select_AP": best_ap}


# ---------------------------------------------------------------------------- weights utilities

def transport_offline(args):
    from abra.weights.io import save_residual as _save
    residual = load_residual(args.residual, parent=args.source_domain)
    target, target_info = load_domain_expert(args.target_domain)
    weights = layer_weights(target, residual.layers)
    source = None
    if args.alpha_estimate and residual.source_sigma is None:
        source = layer_weights(load_domain_expert(args.source_domain or residual.info["parent"])[0], residual.layers)
    ctx = TransportContext(target=weights, source=source, device=args.device)
    options = {"mask": args.abra_mask, "norm_preserving": args.alpha_estimate} if args.method == "abra" else {}
    r = transport(args.method, residual, ctx, **options)
    info = {"transport": args.method, "abra_mask": args.abra_mask, "alpha_estimate": args.alpha_estimate,
            "source_residual": str(args.residual), "source_residual_info": residual.info}
    _save(args.out, r, weights, target_info, off_diag=None, info=info, encoding="dense")
    print(f"[transport] {args.method}: {len(r)} layers -> {args.out} (dense residual on {args.target_domain})")


def migrate(args):
    from abra.weights.io import _read
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.domain:
        tensors, info = load_domain_expert(path)
        out = args.out_dir / f"{Path(path).parent.name}_{Path(path).stem}.safetensors"
        save_domain_expert(out, tensors, {"migrated_from": str(path)})
        print(f"[migrate] domain expert {path} -> {out} (sha256 {info['sha256'][:12]})")
    if args.expert:
        if args.parent is None:
            raise SystemExit("--expert needs --parent (the domain expert the experts were trained from)")
        parent, parent_info = load_domain_expert(args.parent)
        for path in args.expert:
            legacy = load_residual(path)
            raw, _ = _read(path)
            off_diag = int(max((i - j).abs().max().item() for i, j in
                               (torch.nonzero(raw[f"{l}.svft_layer.s"], as_tuple=True) for l in legacy.layers)))
            out = args.out_dir / f"{Path(path).parent.name}_{Path(path).stem}.safetensors"
            info = save_residual(out, legacy.delta, layer_weights(parent, legacy.layers), parent_info, off_diag,
                                 {"migrated_from": str(path)})
            decoded = load_residual(out, parent=args.parent)
            err = max(((decoded.delta[l] - legacy.delta[l]).norm() / legacy.delta[l].norm()).item() for l in legacy.layers)
            print(f"[migrate] class expert {path} -> {out}: off-band energy {info['max_off_band_energy']:.1e}, "
                  f"max relative reconstruction error {err:.1e}")


def fda(args):
    from abra.baselines.fda import run_fda
    logger, device = _setup(args, "fda")
    model, postprocessors, _ = _model(args, device)
    result = run_fda(args, model, postprocessors, device, logger)
    logger.close()
    return result
