"""Phase-0 reference for the refactor: detection results of the *current* code (needs a CUDA GPU).

Runs the unmodified ``main.py`` (through ``seeded_main.py``) for a set of evaluation settings per class
and two short training smoke runs, and collects AP / AP50 / AP75 plus loss traces into one JSON file.
The refactored code is accepted only if it reproduces these results (tolerance: 0.05 AP).

Example (from the repo root; ``{cls}`` in a path is replaced by each class name):
    python tests/golden/run_gpu_golden.py \
        --gdino_weights weights/groundingdino_swint_ogc.pth \
        --fog_root datasets/cityscapes/fog/images \
        --normal_root datasets/cityscapes/normal/images \
        --fog_val_ann 'datasets/cityscapes/fog/annotations/experts/{cls}/annotations_val.json' \
        --normal_ann_dir 'datasets/cityscapes/normal/annotations/experts/{cls}' \
        --source_domain path/to/cityscape_normal/best.pth \
        --target_domain path/to/cityscape_fog/best.pth \
        --expert 'path/to/cityscapes_experts/exp1_{cls}/best.pth'

``--normal_ann_dir`` must contain ``annotations_train.jsonl`` and ``annotations_val.json`` (training smoke only).
Each run's full stdout goes to ``<work_dir>/<run>.log``.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEEDED_MAIN = REPO / "tests" / "golden" / "seeded_main.py"
MODEL_CFG = REPO / "groundingdino" / "config" / "cfg_odvg.py"

TABLE_HEADER = "Evaluation results for bbox:"


def parse_ap(log_text):
    """Return AP, AP50, AP75, APs, APm, APl (in %, 3 decimals) of the last eval table in the log.

    ``CocoEvaluator.summarize`` prints a pipe table after the header line; the pycocotools summary
    itself only has 0.1-AP resolution, which is too coarse for the 0.05 tolerance.
    """
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    idx = log_text.rfind(TABLE_HEADER)
    if idx < 0:
        return dict.fromkeys(keys)
    rows = [l for l in log_text[idx:].splitlines()[1:] if l.strip().startswith("|")]
    values = [float(v) for v in rows[2].strip().strip("|").split("|")]
    return dict(zip(keys, values))


def preflight(args):
    """Fail fast, before any GPU work, if a weight, annotation or image file is missing."""
    problems = []
    need = [args.gdino_weights, args.source_domain, args.target_domain]
    for cls in args.classes:
        need += [Path(args.expert.format(cls=cls)), Path(args.fog_val_ann.format(cls=cls))]
    if not args.skip_train:
        train_dir = Path(args.normal_ann_dir.format(cls=args.classes[0]))
        need += [train_dir / "annotations_train.jsonl", train_dir / "annotations_val.json"]
    problems += [f"missing file: {f}" for f in need if not Path(f).exists()]
    for root in (args.fog_root, args.normal_root):
        if "cityscapes" not in str(root):
            problems.append(f"image root must contain 'cityscapes' (main.py appends /val only then): {root}")
    for cls in args.classes:
        ann = Path(args.fog_val_ann.format(cls=cls))
        if ann.exists():
            names = [img["file_name"] for img in json.load(open(ann))["images"]]
            missing = [n for n in names if not (args.fog_root / "val" / n).exists()]
            if missing:
                problems.append(f"{cls}: {len(missing)}/{len(names)} images missing, e.g. {args.fog_root / 'val' / missing[0]}")
    if problems:
        sys.exit("Pre-flight check failed:\n  " + "\n  ".join(problems))
    print(f"Pre-flight check passed ({len(args.classes)} classes).", flush=True)


def run(name, main_args, work_dir, seed=0, trace=False):
    log_path = work_dir / f"{name}.log"
    env = dict(os.environ, GOLDEN_SEED=str(seed))
    if trace:
        env["GOLDEN_TRACE"] = str(work_dir / f"{name}.trace.jsonl")
    cmd = [sys.executable, str(SEEDED_MAIN)] + [str(a) for a in main_args]
    print(f"[{name}] {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    text = log_path.read_text()
    result = {"returncode": proc.returncode, "seed": seed, "log": str(log_path), **parse_ap(text)}
    if trace and proc.returncode == 0:
        result["trace"] = [json.loads(l) for l in open(work_dir / f"{name}.trace.jsonl")]
    status = "ok" if proc.returncode == 0 else f"FAILED (see {log_path})"
    print(f"[{name}] {status} AP={result['AP']}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gdino_weights", type=Path, required=True)
    p.add_argument("--fog_root", type=Path, required=True, help="Foggy Cityscapes images root (contains val/)")
    p.add_argument("--normal_root", type=Path, required=True, help="Cityscapes images root (contains train/, val/)")
    p.add_argument("--fog_val_ann", required=True, help="fog val COCO json per class, with {cls}")
    p.add_argument("--normal_ann_dir", required=True, help="dir with normal annotations_train.jsonl/annotations_val.json, with {cls}")
    p.add_argument("--source_domain", type=Path, required=True)
    p.add_argument("--target_domain", type=Path, required=True)
    p.add_argument("--expert", required=True, help="class expert checkpoint, with {cls}")
    p.add_argument("--classes", nargs="+", default=["bus", "motorcycle", "rider", "train", "truck"])
    p.add_argument("--alpha_search", action="store_true", help="also run ABRA-alpha (slow)")
    p.add_argument("--with_banded_ta", action="store_true", help="also run ABRA banded mask and task addition")
    p.add_argument("--skip_train", action="store_true", help="skip the two training smoke runs")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--work_dir", type=Path, default=REPO / "outputs_golden")
    p.add_argument("--out", type=Path, default=REPO / "tests" / "golden" / "gpu_golden.json")
    p.add_argument("--check_only", action="store_true", help="only run the pre-flight file check")
    args = p.parse_args()
    preflight(args)
    if args.check_only:
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)

    results = {"args": {k: str(v) for k, v in vars(args).items()}, "eval": {}, "train": {}}

    def common(name, test_ann, root):
        return ["--model_cfg", MODEL_CFG, "--weights", args.gdino_weights, "--exp_name", name,
                "--output_root", args.work_dir / name, "--root", root, "--test_annotations", test_ann,
                "--num_workers", args.num_workers]

    for cls in args.classes:
        expert = Path(args.expert.format(cls=cls))
        fog_val = Path(args.fog_val_ann.format(cls=cls))
        base = lambda name: common(f"{cls}_{name}", fog_val, args.fog_root) + ["--test", 1]
        transport = ["--finetuning", args.target_domain, "--source_class", expert]
        settings = {
            "zero_shot": [],
            "target_domain": ["--finetuning", args.target_domain],
            "abra": transport + ["--svft", 1, "--transporter", "abra"],
        }
        if args.with_banded_ta:  # needs a main.py with --abra_mask and the 'analogy' transporter
            settings["abra_banded"] = transport + ["--svft", 1, "--transporter", "abra", "--abra_mask", "banded"]
            settings["task_addition"] = transport + ["--svft", 0, "--transporter", "analogy",
                                                     "--source_domain", args.source_domain,
                                                     "--target_domain", args.target_domain]
        if args.alpha_search:
            settings["abra_alpha_oracle"] = transport + ["--svft", 1, "--transporter", "abra", "--alpha_search", 1]
        res = {name: run(f"{cls}_{name}", base(name) + extra, args.work_dir) for name, extra in settings.items()}
        # Param Delta depends on arbitrary GPU SVD signs: run it twice to measure that variance.
        for rep in range(2):
            res[f"param_delta_rep{rep}"] = run(f"{cls}_param_delta_rep{rep}",
                                               base(f"param_delta_rep{rep}") + transport + ["--svft", 1, "--transporter", "simple"],
                                               args.work_dir)
        results["eval"][cls] = res

    if not args.skip_train:
        cls = args.classes[0]
        train_dir = Path(args.normal_ann_dir.format(cls=cls))
        label = json.loads(open(train_dir / "annotations_train.jsonl").readline())["detection"]["instances"][0]
        label_map = args.work_dir / f"label_map_{cls}.json"
        label_map.write_text(json.dumps({str(label["label"]): label["category"]}))
        train_common = lambda name: common(name, train_dir / "annotations_val.json", args.normal_root) + [
            "--train_annotations", train_dir / "annotations_train.jsonl", "--label_map", label_map,
            "--epochs", 1, "--batch_size", 2, "--debug", 1, "--optim", "adamw", "--trainable", "encoder_attn",
            "--param_type", "default", "--num_workers", 0]
        results["train"]["domain_smoke"] = run(
            "train_domain_smoke", train_common("train_domain_smoke") + ["--objectify", 1, "--lr", "1e-4", "--lp_lr", "1e-5"],
            args.work_dir, trace=True)
        results["train"]["expert_smoke"] = run(
            "train_expert_smoke", train_common("train_expert_smoke") + [
                "--finetuning", args.source_domain, "--svft", 1, "--svft_pattern", "banded", "--svft_diag", 2,
                "--lr", "1e-2", "--lp_lr", "1e-3", "--weight_decay", 0.0],
            args.work_dir, trace=True)

    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
