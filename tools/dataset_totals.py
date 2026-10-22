"""Dataset totals per domain and split (revision round 2, T5), CPU only, from the annotation files the evaluation
code reads (train: ``annotations_train.jsonl``; test: Cityscapes ``annotations_val.json``, SDGOD ``annotations_test.json``).

    python tools/dataset_totals.py --cityscapes datasets/cityscapes --sdgod datasets/sdgod --out dataset_stats_round2.csv

Three rows per domain and split, in the round-1 format of dataset_stats.csv (class, n_instances, n_images empty):
``total_images`` (every image of the file), ``images_with_annotations`` (>= 1 box), ``total_instances``. The totals
are then compared with the values printed in the paper (Supp. F); mismatches are printed, the paper values are never
changed here.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abra.data.stats import load  # noqa: E402
from tools.revision_stats import BENCHMARKS, COLUMNS  # noqa: E402

# Supp. F of the paper (train images, test images); Cityscapes and Foggy Cityscapes share their images
PAPER = {"cityscapes/normal": (2975, 500), "cityscapes/fog": (2975, 500), "sdgod/daytime_sunny": (20757, 6951),
         "sdgod/daytime_foggy": (2812, 963), "sdgod/dusk_rainy": (2611, 890), "sdgod/night_sunny": (19599, 6559),
         "sdgod/night_rainy": (1864, 630)}


def totals(images):
    return {"total_images": len(images), "images_with_annotations": sum(1 for _, b in images if b),
            "total_instances": sum(len(b) for _, b in images)}


def benchmark_rows(name, root):
    spec, rows = BENCHMARKS[name], []
    for domain in spec["domains"]:
        ann = Path(root) / domain / "annotations"
        for split, file in (("train", "annotations_train.jsonl"), ("test", spec["test"])):
            rows += [{"domain": f"{name}/{domain}", "split": split, "stat": k, "value": v}
                     for k, v in totals(load(ann / file)).items()]
    return rows


def check(rows):
    """-> lines comparing total_images with the paper; a mismatch is marked MISMATCH."""
    got = {(r["domain"], r["split"]): r["value"] for r in rows if r["stat"] == "total_images"}
    lines = []
    for domain, (train, test) in PAPER.items():
        for split, paper in (("train", train), ("test", test)):
            value = got.get((domain, split))
            mark = "ok" if value == paper else "MISMATCH"
            lines.append(f"{domain:22s} {split:5s} paper {paper:6d}  annotation file {value}  {mark}")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cityscapes", type=Path, required=True)
    p.add_argument("--sdgod", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rows = benchmark_rows("cityscapes", args.cityscapes) + benchmark_rows("sdgod", args.sdgod)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")
    print("\n".join(check(rows)))


if __name__ == "__main__":
    main()
