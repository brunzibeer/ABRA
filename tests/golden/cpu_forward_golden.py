"""CPU regression harness: fingerprints of raw model outputs on synthetic images.

Creates a tiny synthetic dataset, runs every setting through the ``abra`` CLI on CPU (6 evaluations and
two 2-step trainings), records model outputs (pred_logits / pred_boxes) and training loss traces, and
compares them with ``cpu_forward_golden.json``, which was produced by the ORIGINAL ``main.py`` (tag
``official`` + Phase-0 scripts, commit 6a34a0d; regenerate it from that commit if ever needed).

Deformable attention uses the pure-PyTorch path on CPU, so the compiled op is not needed. Run from the
repo root:

    python tests/golden/cpu_forward_golden.py --weights_dir ../weights

``--weights_dir`` must contain groundingdino_swint_ogc.pth, cityscape_normal/best.pth,
cityscape_fog/best.pth and cityscapes_experts/exp1_train/best.pth.

Expected differences, by design (see REFACTOR_PLAN.md): evaluations with a class residual use the exact
target weights theta_T + R instead of the float32 SVD reconstruction U_T (S_T + s_T) V_T, which moves
outputs by ~1e-4 relative; everything else must be identical.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "tests" / "golden" / "cpu_forward_golden.json"
N_PROBES = 64
# Settings that are only expected to be *close* to the original (docs/MIGRATION.md §3.1 and §3.3), with the allowed
# deviation of the sorted top-100 detection scores. Everything else must be identical.
CLOSE_TOLERANCE = {
    "abra": 2e-3, "abra_banded": 2e-3, "param_delta": 2e-3, "task_addition": 3e-3,  # exact theta_T instead of U S V
    "train_expert": 1e-2,  # dense vs sparse SVFT rounding (~1e-6), amplified by Adam's first step
}
CLASS = "train"


def settings(w, data):
    expert = w / "cityscapes_experts" / f"exp1_{CLASS}" / "best.pth"
    fog, normal = w / "cityscape_fog" / "best.pth", w / "cityscape_normal" / "best.pth"
    transport = ["--domain", fog, "--residual", expert, "--source_domain", normal]
    train = ["--train_annotations", data / "train.jsonl", "--label_map", data / "label_map.json",
             "--epochs", 1, "--batch_size", 1, "--optim", "adamw", "--trainable", "encoder_attn",
             "--param_type", "default"]
    return {
        "zero_shot": ["eval"],
        "target_domain": ["eval", "--domain", fog],
        "abra": ["eval", *transport, "--method", "abra"],
        "abra_banded": ["eval", *transport, "--method", "abra", "--abra_mask", "banded"],
        "param_delta": ["eval", *transport, "--method", "param_delta"],
        "task_addition": ["eval", *transport, "--method", "task_addition"],
        "train_domain": ["train-domain", *train, "--objectify", 1, "--lr", "1e-4", "--lp_lr", "1e-5"],
        "train_expert": ["train-expert", *train, "--domain", normal, "--svft_diag", 2, "--lr", "1e-2",
                         "--lp_lr", "1e-3", "--weight_decay", 0.0],
    }


def settings_old(w, data):
    expert = w / "cityscapes_experts" / f"exp1_{CLASS}" / "best.pth"
    transport = ["--finetuning", w / "cityscape_fog" / "best.pth", "--source_class", expert]
    train = ["--train_annotations", data / "train.jsonl", "--label_map", data / "label_map.json",
             "--epochs", 1, "--batch_size", 1, "--optim", "adamw", "--trainable", "encoder_attn",
             "--param_type", "default"]
    return {
        "zero_shot": ["--test", 1],
        "target_domain": ["--test", 1, "--finetuning", w / "cityscape_fog" / "best.pth"],
        "abra": ["--test", 1, *transport, "--svft", 1, "--transporter", "abra"],
        "abra_banded": ["--test", 1, *transport, "--svft", 1, "--transporter", "abra", "--abra_mask", "banded"],
        "param_delta": ["--test", 1, *transport, "--svft", 1, "--transporter", "simple"],
        "task_addition": ["--test", 1, *transport, "--svft", 0, "--transporter", "analogy",
                          "--source_domain", w / "cityscape_normal" / "best.pth",
                          "--target_domain", w / "cityscape_fog" / "best.pth"],
        "train_domain": [*train, "--objectify", 1, "--lr", "1e-4", "--lp_lr", "1e-5"],
        "train_expert": [*train, "--finetuning", w / "cityscape_normal" / "best.pth", "--svft", 1,
                         "--svft_pattern", "banded", "--svft_diag", 2, "--lr", "1e-2", "--lp_lr", "1e-3",
                         "--weight_decay", 0.0],
    }


def make_data(root):
    """Two deterministic random 'images' with one box each, as COCO (val) and ODVG (train)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    images, annotations, odvg = [], [], []
    for i in range(2):
        name = f"synth/img_{i}.png"
        for split in ("train", "val"):
            path = root / "cityscapes_synth" / split / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (256, 512, 3), dtype=np.uint8)).save(path)
        box = [100 + 40 * i, 60, 260 + 40 * i, 180]
        images.append({"id": i + 1, "file_name": name, "height": 256, "width": 512})
        annotations.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "iscrowd": 0,
                            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                            "area": (box[2] - box[0]) * (box[3] - box[1])})
        odvg.append({"filename": name, "height": 256, "width": 512,
                     "detection": {"instances": [{"bbox": box, "label": 1, "category": CLASS}]}})
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": CLASS, "supercategory": "object"}]}
    (root / "val.json").write_text(json.dumps(coco))
    (root / "train.jsonl").write_text("\n".join(json.dumps(o) for o in odvg) + "\n")
    (root / "label_map.json").write_text(json.dumps({"1": CLASS}))


def worker(out_path, cli_args):
    """Run the abra CLI in-process with output capture."""
    import torch

    torch.set_num_threads(4)
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    import abra.model.build as build
    from abra.cli import main as cli_main

    captured = {"outputs": [], "trace": []}
    build_model = build.build_model

    def build_model_with_capture(args):
        model, post = build_model(args)

        def hook(module, inputs, output):
            if not module.training:
                captured["outputs"].append({k: output[k].detach().clone() for k in ("pred_logits", "pred_boxes")})
        model.register_forward_hook(hook)
        return model, post

    build.build_model = build_model_with_capture
    cli_main(cli_args)
    out_dir = Path(cli_args[cli_args.index("--output_root") + 1])
    for line in open(out_dir / "metrics.jsonl"):
        record = json.loads(line)
        if "TRAIN/total_loss" in record:
            captured["trace"].append({k: v for k, v in record.items() if k.startswith("TRAIN/")})
    torch.save(captured, out_path)


def worker_old(out_path, main_args):
    """Run the ORIGINAL main.py in-process with seeding, an op stub and output capture (--cli old)."""
    import random
    import runpy
    import types

    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    sys.modules.setdefault("MultiScaleDeformableAttention", types.ModuleType("MultiScaleDeformableAttention"))
    # checkpoints saved from GPU runs would otherwise be restored onto CUDA
    torch_load = torch.load
    torch.load = lambda *a, **kw: torch_load(*a, **{"map_location": "cpu", **kw})
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)

    import engine
    import utils.builders as builders

    captured = {"outputs": [], "trace": []}

    class Recorder:
        run = True

        @staticmethod
        def log(data, step=None):
            captured["trace"].append({k: float(v) for k, v in data.items()})

    engine.wandb = Recorder()
    build_model = builders.build_model

    def build_model_with_capture(args):
        model, post = build_model(args)

        def hook(module, inputs, output):
            if not module.training:
                captured["outputs"].append({k: output[k].detach().clone() for k in ("pred_logits", "pred_boxes")})
        model.register_forward_hook(hook)
        return model, post

    builders.build_model = build_model_with_capture
    sys.argv = [str(REPO / "main.py")] + main_args
    runpy.run_path(str(REPO / "main.py"), run_name="__main__")
    torch.save(captured, out_path)


def fingerprint(captured):
    import torch

    def probes(t):
        flat = t.double().flatten()
        finite = flat[torch.isfinite(flat)]  # pred_logits are -inf at padded text positions
        g = torch.Generator().manual_seed(finite.numel())
        idx = torch.randint(finite.numel(), (N_PROBES,), generator=g)
        return {"n_finite": finite.numel(), "sum": finite.sum().item(), "abs_max": finite.abs().max().item(),
                "probes": finite[idx].tolist()}

    def top_scores(o):
        """Sorted top-100 query scores (max sigmoid over text tokens): invariant to query order."""
        scores = o["pred_logits"].double().sigmoid().amax(-1).flatten()
        return scores.sort(descending=True).values[:100].tolist()

    return {
        "outputs": [{**{k: probes(v) for k, v in o.items()}, "top_scores": top_scores(o)} for o in captured["outputs"]],
        "trace": [{k: v for k, v in s.items() if "loss" in k} for s in captured["trace"]],
    }


def compare(ref, new, rtol=1e-4, atol=1e-5):
    """Return a list of human-readable differences (empty if equal within tolerance)."""
    diffs = []

    def close(a, b):
        if a == b:  # also covers the -inf entries of pred_logits at padded text positions
            return True
        return abs(a - b) <= atol + rtol * abs(b)

    if len(ref["outputs"]) != len(new["outputs"]):
        return [f"number of forward passes {len(new['outputs'])} != {len(ref['outputs'])}"]
    for i, (r, n) in enumerate(zip(ref["outputs"], new["outputs"])):
        for k in ("pred_logits", "pred_boxes"):
            bad = [j for j, (a, b) in enumerate(zip(n[k]["probes"], r[k]["probes"])) if not close(a, b)]
            if n[k]["n_finite"] != r[k]["n_finite"]:
                diffs.append(f"forward {i} {k}: {n[k]['n_finite']} finite values vs {r[k]['n_finite']}")
            elif bad or not close(n[k]["sum"], r[k]["sum"]):
                diffs.append(f"forward {i} {k}: {len(bad)} probes differ, sum {n[k]['sum']:.6g} vs {r[k]['sum']:.6g}")
    for i, (r, n) in enumerate(zip(ref["trace"], new["trace"])):
        for k in r:
            if k not in n or not close(n[k], r[k]):
                diffs.append(f"train step {i} {k}: {n.get(k)} vs {r[k]}")
    if len(ref["trace"]) != len(new["trace"]):
        diffs.append(f"trace length {len(new['trace'])} != {len(ref['trace'])}")
    return diffs


def top_score_deviation(ref, new):
    """Largest difference of the sorted top-100 scores (order-independent closeness)."""
    worst = 0.0
    for r, n in zip(ref["outputs"], new["outputs"]):
        if "top_scores" in r:
            worst = max([worst] + [abs(a - b) for a, b in zip(n["top_scores"], r["top_scores"])])
    return worst


def max_deviation(ref, new):
    """Largest probe / loss difference, relative to the tensor's largest magnitude (or the loss)."""
    worst = 0.0
    for r, n in zip(ref["outputs"], new["outputs"]):
        for k in ("pred_logits", "pred_boxes"):
            scale = r[k]["abs_max"] or 1.0
            worst = max([worst] + [abs(a - b) / scale for a, b in zip(n[k]["probes"], r[k]["probes"]) if a != b])
    for r, n in zip(ref["trace"], new["trace"]):
        worst = max([worst] + [abs(n.get(k, float("nan")) - v) / (abs(v) or 1.0) for k, v in r.items()])
    return worst


def report(name, ref, new, args):
    diffs = compare(ref, new)
    dev, score_dev = max_deviation(ref, new), top_score_deviation(ref, new)
    if not diffs:
        print(f"[{name}] IDENTICAL (max deviation {dev:.1e})")
        return True
    if name in args.allow_close and score_dev <= CLOSE_TOLERANCE.get(name, 0.0):
        print(f"[{name}] CLOSE: top-100 scores within {score_dev:.1e} (allowed for this setting; "
              f"per-query deviation {dev:.1e} from reordered queries)")
        return True
    print(f"[{name}] DIFFERS (top-100 score deviation {score_dev:.1e}, per-query {dev:.1e}):\n  " + "\n  ".join(diffs[:6]))
    return False


def compare_files(ref_path, new_path, args):
    ref, new = json.loads(ref_path.read_text()), json.loads(new_path.read_text())
    ok = [report(name, ref[name], new[name], args) for name in ref if name in new]
    sys.exit(0 if all(ok) else 1)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        return worker(sys.argv[2], sys.argv[3:])
    if len(sys.argv) > 2 and sys.argv[1] == "--worker-old":
        return worker_old(sys.argv[2], sys.argv[3:])

    p = argparse.ArgumentParser()
    p.add_argument("--weights_dir", type=Path, required=True)
    p.add_argument("--only", nargs="+", default=None, help="subset of settings")
    p.add_argument("--write", action="store_true", help="write the reference instead of comparing")
    p.add_argument("--cli", default="new", choices=["new", "old"], help="old: drive the original main.py")
    p.add_argument("--allow_close", nargs="*", default=list(CLOSE_TOLERANCE),
                   help="settings that may differ per query if their sorted top-100 scores match (see CLOSE_TOLERANCE)")
    p.add_argument("--reference", type=Path, default=REFERENCE, help="reference json to write / compare against")
    p.add_argument("--compare_files", type=Path, nargs=2, metavar=("REFERENCE", "NEW"),
                   help="only compare two result files written with --write")
    args = p.parse_args()
    if args.compare_files:
        return compare_files(*args.compare_files, args)
    w = args.weights_dir.resolve()

    results, failures = {}, []
    reference = json.loads(args.reference.read_text()) if args.reference.exists() and not args.write else None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_data(tmp)
        for name, extra in (settings_old if args.cli == "old" else settings)(w, tmp).items():
            if args.only and name not in args.only:
                continue
            common = ["--device", "cpu", "--weights", w / "groundingdino_swint_ogc.pth", "--exp_name", name,
                      "--output_root", tmp / "out" / name, "--root", tmp / "cityscapes_synth",
                      "--test_annotations", tmp / "val.json", "--num_workers", 0]
            if args.cli == "old":
                main_args = ["--model_cfg", REPO / "groundingdino/config/cfg_odvg.py", *common, *extra]
            else:
                main_args = [extra[0], *common, "--seed", 0, *extra[1:]]
            out = tmp / f"{name}.pt"
            log = tmp / f"{name}.log"
            with open(log, "w") as f:
                worker_flag = "--worker-old" if args.cli == "old" else "--worker"
                proc = subprocess.run([sys.executable, __file__, worker_flag, str(out)] + [str(a) for a in main_args],
                                      stdout=f, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                failures.append(name)
                print(f"[{name}] FAILED:\n" + "".join(log.read_text().splitlines(True)[-25:]))
                continue
            import torch
            results[name] = fingerprint(torch.load(out))
            if reference is not None:
                if name not in reference:
                    print(f"[{name}] no reference")
                    continue
                if not report(name, reference[name], results[name], args):
                    failures.append(name)
            else:
                print(f"[{name}] ok ({len(results[name]['outputs'])} forward passes, {len(results[name]['trace'])} train steps)")

    if args.write:
        merged = json.loads(args.reference.read_text()) if args.reference.exists() and args.only else {}
        merged.update(results)
        args.reference.write_text(json.dumps(merged, indent=1))
        print(f"wrote {args.reference}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
