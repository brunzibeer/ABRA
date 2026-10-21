"""Dataset statistics of the revision (E8), CPU only: one csv for every domain, split and class.

    python tools/revision_stats.py --cityscapes datasets/cityscapes --sdgod datasets/sdgod --out dataset_stats.csv

Rows (columns domain, split, class, n_instances, n_images, stat, value):
- per class (stat empty): instances and images with >= 1 instance, for every domain and split (train, test; the
  Cityscapes test split is ``annotations_val.json``). Test rows = the images of the per-class protocol.
- per training split (class empty): ``domain_expert_images`` (images with >= 1 available instance = the
  Objectification set), ``domain_expert_images_with_unavailable``, ``domain_expert_available_box_share`` (% of the
  boxes of those images that belong to available classes);
- per training split: ``top{k}_coverage`` (% of the boxes covered by the k most frequent classes, k = 1..4; class =
  those classes, most frequent first).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abra.data.stats import class_counts, domain_expert_images, load, pct  # noqa: E402

BENCHMARKS = {
    "cityscapes": {"domains": ["normal", "fog"], "test": "annotations_val.json",
                   "available": ["car", "person", "bicycle"]},
    "sdgod": {"domains": ["daytime_sunny", "daytime_foggy", "dusk_rainy", "night_rainy", "night_sunny"],
              "test": "annotations_test.json", "available": ["car", "person", "truck"]},
}
COLUMNS = ["domain", "split", "class", "n_instances", "n_images", "stat", "value"]


def benchmark_rows(name, root):
    spec, rows = BENCHMARKS[name], []
    for domain in spec["domains"]:
        ann = Path(root) / domain / "annotations"
        splits = {"train": load(ann / "annotations_train.jsonl"), "test": load(ann / spec["test"])}
        train_inst, _ = class_counts(splits["train"])
        classes = sorted(train_inst, key=lambda c: -train_inst[c])
        for split, images in splits.items():
            inst, imgs = class_counts(images)
            rows += [{"domain": f"{name}/{domain}", "split": split, "class": c, "n_instances": inst[c],
                      "n_images": imgs[c]} for c in classes]
        train = splits["train"]
        unavailable = [c for c in classes if c not in spec["available"]]
        dom = domain_expert_images(train, spec["available"], unavailable)
        boxes = [c for _, b in train if any(x in spec["available"] for x, _ in b) for c, _ in b]
        share = pct(sum(c in spec["available"] for c in boxes), len(boxes))
        summary = {"domain_expert_images": dom["images"],
                   "domain_expert_images_with_unavailable": dom["images_any_unavailable"],
                   "domain_expert_available_box_share": share}
        rows += [{"domain": f"{name}/{domain}", "split": "train", "stat": k, "value": v} for k, v in summary.items()]
        total = sum(train_inst.values())
        for k in range(1, 5):
            top = classes[:k]
            rows.append({"domain": f"{name}/{domain}", "split": "train", "class": " ".join(top),
                         "stat": f"top{k}_coverage", "value": pct(sum(train_inst[c] for c in top), total)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cityscapes", type=Path, required=True, help="Cityscapes root (normal/, fog/)")
    p.add_argument("--sdgod", type=Path, required=True, help="SDGOD root (<domain>/)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rows = benchmark_rows("cityscapes", args.cityscapes) + benchmark_rows("sdgod", args.sdgod)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r.get("stat"):
            v = r["value"]
            print(f"{r['domain']:28s} {r['stat']:40s} {v:.4f}" if isinstance(v, float) else
                  f"{r['domain']:28s} {r['stat']:40s} {v}", f"({r['class']})" if r.get("class") else "")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
