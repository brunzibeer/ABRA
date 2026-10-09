"""CPU regression harness: fingerprints of the raw outputs of ``main.py`` on synthetic images.

Complements the GPU golden (real AP) with something that runs on any machine in a few minutes: it
creates a tiny synthetic dataset, runs ``main.py --device cpu`` for every setting (evaluation-only
settings and two 2-step training runs), records model outputs (pred_logits / pred_boxes) and training
loss traces, and either writes a reference or compares against one.

Deformable attention uses the pure-PyTorch path on CPU, so the compiled op is not needed (a stub
module satisfies the import). Run from the repo root:

    python tests/golden/cpu_forward_golden.py --weights_dir ../weights --write   # create reference
    python tests/golden/cpu_forward_golden.py --weights_dir ../weights           # compare (exit 1 on mismatch)

``--weights_dir`` must contain groundingdino_swint_ogc.pth, cityscape_normal/best.pth,
cityscape_fog/best.pth and cityscapes_experts/exp1_train/best.pth.
"""
import argparse
import json
import os
import random
import runpy
import subprocess
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "tests" / "golden" / "cpu_forward_golden.json"
N_PROBES = 64
CLASS = "train"


def settings(w, data):
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


def worker(out_path, main_args):
    """Run main.py in-process with seeding, an op stub and output capture."""
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

    return {
        "outputs": [{k: probes(v) for k, v in o.items()} for o in captured["outputs"]],
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
        for k in r:
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


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        return worker(sys.argv[2], sys.argv[3:])

    p = argparse.ArgumentParser()
    p.add_argument("--weights_dir", type=Path, required=True)
    p.add_argument("--only", nargs="+", default=None, help="subset of settings")
    p.add_argument("--write", action="store_true", help="write the reference instead of comparing")
    args = p.parse_args()
    w = args.weights_dir.resolve()

    results, failures = {}, []
    reference = json.loads(REFERENCE.read_text()) if REFERENCE.exists() and not args.write else None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_data(tmp)
        for name, extra in settings(w, tmp).items():
            if args.only and name not in args.only:
                continue
            main_args = ["--device", "cpu", "--model_cfg", REPO / "groundingdino/config/cfg_odvg.py",
                         "--weights", w / "groundingdino_swint_ogc.pth", "--exp_name", name,
                         "--output_root", tmp / "out" / name, "--root", tmp / "cityscapes_synth",
                         "--test_annotations", tmp / "val.json", "--num_workers", 0, *extra]
            out = tmp / f"{name}.pt"
            log = tmp / f"{name}.log"
            with open(log, "w") as f:
                proc = subprocess.run([sys.executable, __file__, "--worker", str(out)] + [str(a) for a in main_args],
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
                diffs = compare(reference[name], results[name])
                print(f"[{name}] " + ("identical within tolerance" if not diffs else "DIFFERS:\n  " + "\n  ".join(diffs)))
                failures += [name] if diffs else []
            else:
                print(f"[{name}] ok ({len(results[name]['outputs'])} forward passes, {len(results[name]['trace'])} train steps)")

    if args.write:
        merged = json.loads(REFERENCE.read_text()) if REFERENCE.exists() and args.only else {}
        merged.update(results)
        REFERENCE.write_text(json.dumps(merged, indent=1))
        print(f"wrote {REFERENCE}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
