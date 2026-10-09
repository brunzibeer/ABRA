"""Check the refactored ``abra`` CLI against the GPU golden of the original code (``gpu_golden.json``).

Runs, per class, zero-shot / target domain expert / ABRA / Param Delta through ``python -m abra eval``
(with the ORIGINAL .pth checkpoints, read transparently) and the two seeded 5-step training smoke runs,
then prints a table and exits 1 if any AP differs by more than ``--tol`` (default 0.05).

Same arguments as ``run_gpu_golden.py``; run from the repo root on the GPU machine:

    python tests/golden/check_gpu_new_cli.py \
        --gdino_weights weights/groundingdino_swint_ogc.pth \
        --fog_root datasets/cityscapes/fog/images --normal_root datasets/cityscapes/normal/images \
        --fog_val_ann 'datasets/cityscapes/fog/annotations/experts/{cls}/annotations_val.json' \
        --normal_ann_dir 'datasets/cityscapes/normal/annotations/experts/{cls}' \
        --source_domain /path/to/cityscape_normal/best.pth --target_domain /path/to/cityscape_fog/best.pth \
        --expert '/path/to/cityscapes_experts/exp1_{cls}/best.pth'
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "golden" / "gpu_golden.json"


def run(name, cli_args, work_dir):
    out = work_dir / name
    cmd = [sys.executable, "-m", "abra", *[str(a) for a in cli_args], "--exp_name", name, "--output_root", out]
    print(f"[{name}] running", flush=True)
    with open(work_dir / f"{name}.log", "w") as log:
        proc = subprocess.run([str(c) for c in cmd], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[{name}] FAILED, see {work_dir / (name + '.log')}")
        return None
    return out


def main():
    p = argparse.ArgumentParser()
    for a in ("--gdino_weights", "--fog_root", "--normal_root", "--source_domain", "--target_domain"):
        p.add_argument(a, type=Path, required=True)
    p.add_argument("--fog_val_ann", required=True)
    p.add_argument("--normal_ann_dir", required=True)
    p.add_argument("--expert", required=True)
    p.add_argument("--classes", nargs="+", default=["bus", "motorcycle", "rider", "train", "truck"])
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--work_dir", type=Path, default=REPO / "outputs_golden_new")
    args = p.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    golden = json.loads(GOLDEN.read_text())

    rows, failed = [], False
    for cls in args.classes:
        common = ["--weights", args.gdino_weights, "--root", args.fog_root, "--num_workers", args.num_workers,
                  "--test_annotations", args.fog_val_ann.format(cls=cls)]
        transport = ["--domain", args.target_domain, "--residual", args.expert.format(cls=cls),
                     "--source_domain", args.source_domain]
        settings = {"zero_shot": [], "target_domain": ["--domain", args.target_domain],
                    "abra": [*transport, "--method", "abra"], "param_delta_rep0": [*transport, "--method", "param_delta"]}
        for setting, extra in settings.items():
            out = run(f"{cls}_{setting}", ["eval", *common, *extra], args.work_dir)
            new = json.loads((out / "results.json").read_text())["AP"] if out else float("nan")
            old = golden["eval"].get(cls, {}).get(setting, {}).get("AP")
            ok = old is not None and abs(new - old) <= args.tol
            failed |= not ok
            rows.append((cls, setting, old, new, ok))

    if not args.skip_train:
        cls = args.classes[0] if "train" not in args.classes else "train"
        train_dir = Path(args.normal_ann_dir.format(cls=cls))
        label = json.loads(open(train_dir / "annotations_train.jsonl").readline())["detection"]["instances"][0]
        label_map = args.work_dir / f"label_map_{cls}.json"
        label_map.write_text(json.dumps({str(label["label"]): label["category"]}))
        common = ["--weights", args.gdino_weights, "--root", args.normal_root, "--num_workers", 0, "--seed", 0,
                  "--test_annotations", train_dir / "annotations_val.json",
                  "--train_annotations", train_dir / "annotations_train.jsonl", "--label_map", label_map,
                  "--epochs", 1, "--batch_size", 2, "--debug", 1, "--optim", "adamw", "--trainable", "encoder_attn",
                  "--param_type", "default"]
        smoke = {"domain_smoke": ["train-domain", *common, "--objectify", 1, "--lr", "1e-4", "--lp_lr", "1e-5"],
                 "expert_smoke": ["train-expert", *common, "--domain", args.source_domain, "--svft_diag", 2,
                                  "--lr", "1e-2", "--lp_lr", "1e-3", "--weight_decay", 0.0]}
        for name, cli in smoke.items():
            out = run(name, cli, args.work_dir)
            losses = [json.loads(l)["TRAIN/total_loss"] for l in open(out / "metrics.jsonl")
                      if "TRAIN/total_loss" in l] if out else []
            ref = [r["TRAIN/total_loss"] for r in golden["train"][name]["trace"]]
            dev = max((abs(a - b) / abs(b) for a, b in zip(losses, ref)), default=float("nan"))
            print(f"[{name}] loss trace relative deviation {dev:.1e} (new {losses}, original {ref})")

    print(f"\n{'class':12s} {'setting':18s} {'original':>9s} {'new':>9s}  ok (|diff| <= {args.tol})")
    for cls, setting, old, new, ok in rows:
        print(f"{cls:12s} {setting:18s} {old if old is not None else float('nan'):9.3f} {new:9.3f}  {'yes' if ok else 'NO'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
