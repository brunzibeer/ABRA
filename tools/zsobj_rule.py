"""Rule of the Fig. 4 "Zero shot w/ Obj." domain experts (revision round 2, T1), CPU only.

The script that built their training files is lost; the files are not (wandb CS_NORMAL_ZS / CS_FOG_ZS trained
``domains/cityscapes/{normal,fog}/dom3`` on ``annotations/domains/annotations_train_domain_zs_030.jsonl``). This
compares each of them with the Objectification file of the same domain (``annotations_train_domain.jsonl``, the
training set of dom1) and reports what was added:

    python tools/zsobj_rule.py --cityscapes datasets/cityscapes \
        --zs annotations/domains/annotations_train_domain_zs_030.jsonl --out zsobj_rule.json

Per domain: images only in one file, objectification boxes kept or dropped, the boxes added (count, categories,
extra fields such as a score, per-image counts), and the largest IoU of every added box with the objectification
boxes of its image (a duplicate-removal threshold shows as an empty band below 1).
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def records(path):
    return {r["filename"]: r for r in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def iou(a, b):
    """IoU of two [x1, y1, x2, y2] boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box(inst):
    return tuple(round(float(x), 2) for x in inst["bbox"])


def compare(base_path, zs_path):
    """Boxes matched by coordinates (the zero-shot file renames categories, so names are compared separately)."""
    base, zs = records(base_path), records(zs_path)
    out = {"base_file": str(base_path), "zs_file": str(zs_path), "base_images": len(base), "zs_images": len(zs),
           "images_only_in_base": len(base.keys() - zs.keys()), "images_only_in_zs": len(zs.keys() - base.keys())}
    kept = dropped = 0
    renamed, added_cats, labels, fields, per_image, best_iou, added = Counter(), Counter(), Counter(), Counter(), Counter(), [], []
    for name, rec in zs.items():
        old = base.get(name, {"detection": {"instances": []}})["detection"]["instances"]
        new = rec["detection"]["instances"]
        pool = {}
        for o in old:
            pool.setdefault(box(o), []).append(o["category"])
        n = 0
        for inst in new:
            labels[(inst.get("label"), inst.get("category"))] += 1
            fields.update(f for f in inst if f not in ("bbox", "category", "label"))
            if pool.get(box(inst)):
                renamed[(pool[box(inst)].pop(), inst.get("category"))] += 1
                kept += 1
            else:
                n += 1
                added_cats[inst.get("category")] += 1
                best_iou.append(max((iou(inst["bbox"], o["bbox"]) for o in old), default=0.0))
                added.append(inst)
        dropped += sum(len(v) for v in pool.values())
        per_image[n] += 1
    bands = Counter(min(int(v * 10), 9) / 10 for v in best_iou)   # [0.0, 0.1), ..., [0.9, 1.0]
    out.update({"base_boxes_kept (same coordinates)": kept, "base_boxes_dropped": dropped,
                "base_category -> zs_category of kept boxes": {f"{a} -> {b}": v for (a, b), v in renamed.most_common()},
                "added_boxes": len(added), "added_categories": dict(added_cats.most_common()),
                "zs_label_id -> category": {f"{a} -> {b}": v for (a, b), v in sorted(labels.items(), key=str)},
                "extra_fields": dict(fields), "added_per_image_median": sorted(per_image.elements())[len(zs) // 2] if zs else None,
                "added_max_iou_with_base_by_band": {f"{b:.1f}": bands[b] for b in sorted(bands)},
                "example_added": added[:3]})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cityscapes", type=Path, required=True)
    p.add_argument("--zs", default="annotations/domains/annotations_train_domain_zs_030.jsonl")
    p.add_argument("--base", default="annotations/domains/annotations_train_domain.jsonl")
    p.add_argument("--domains", nargs="+", default=["normal", "fog"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    report = {}
    for d in args.domains:   # a lost file is reported, not fatal (Foggy's zero-shot file is not on the server)
        base, zs = args.cityscapes / d / args.base, args.cityscapes / d / args.zs
        missing = [str(f) for f in (base, zs) if not f.exists()]
        report[d] = {"missing": missing} if missing else compare(base, zs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
