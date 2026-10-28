"""``abra convert-coco``: one COCO json of a dataset split -> the master file of one domain, in the layout read by
``prepare-data --benchmark masters`` (``<domain>/annotations/annotations_train.jsonl`` or
``annotations_{val,test}.json``). Used for FLIR ADAS v2, which has one COCO json per modality and split.

The output categories are given (``--classes``: ids 1..n in that order), so every modality and split gets the same
ids whatever the ids of the input file. ``--rename src=dst`` maps dataset names to the class names used as prompts
(several dataset names may map to one class). Boxes of other categories are dropped and counted; boxes thinner than
1 px are dropped, as in the Cityscapes conversion. Images without a kept box stay (negatives of the standard protocol)
unless ``--drop_empty 1``. Standard library only.
"""
import json
from collections import Counter
from pathlib import Path

from abra.data.prepare import coco_to_odvg, now, write_json, write_jsonl


def add_coco_args(p):
    from abra.cli import str2bool
    p.add_argument("--coco", type=Path, required=True, help="input COCO json (e.g. FLIR images_thermal_train/coco.json)")
    p.add_argument("--split", required=True, choices=["train", "val", "test"],
                   help="train -> ODVG jsonl; val / test -> COCO json")
    p.add_argument("--out_dir", type=Path, required=True, help="<masters_root>/<domain>, e.g. masters/thermal")
    p.add_argument("--classes", nargs="+", required=True,
                   help="output classes, in id order (1..n); the same list for every modality and split")
    p.add_argument("--rename", nargs="+", default=[], metavar="SRC=DST",
                   help="dataset category name -> class name, e.g. bike=bicycle 'light=traffic light'")
    p.add_argument("--file_prefix", default="",
                   help="prepended to every file_name, so that one image root serves every split "
                        "(e.g. images_thermal_train/)")
    p.add_argument("--drop_empty", type=str2bool, default=False, help="drop images without a kept box")
    p.add_argument("--date_created", default=None, help="info.date_created of a COCO output (default: now)")


def parse_rename(pairs):
    rename = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--rename expects SRC=DST, got {pair!r}")
        src, dst = pair.split("=", 1)
        rename[src] = dst
    return rename


def convert(coco, classes, rename, file_prefix="", drop_empty=False, date_created=None):
    """-> (COCO dict with categories 1..n = ``classes``, report dict)."""
    if len(set(classes)) != len(classes):
        raise SystemExit(f"--classes has duplicates: {classes}")
    source = {c["id"]: c["name"] for c in coco["categories"]}
    unknown = sorted(set(rename) - set(source.values()))
    if unknown:
        raise SystemExit(f"--rename names not in the input categories: {unknown} (input: {sorted(source.values())})")
    new_id = {name: i for i, name in enumerate(classes, start=1)}
    mapped = {cid: new_id.get(rename.get(name, name)) for cid, name in source.items()}
    missing = sorted(set(classes) - {rename.get(name, name) for name in source.values()})
    if missing:
        raise SystemExit(f"--classes not produced by any input category: {missing} (input: {sorted(source.values())}; "
                         f"use --rename)")

    dropped, thin, kept = Counter(), 0, {}
    for ann in coco["annotations"]:
        cid = mapped.get(ann["category_id"])
        if cid is None:
            dropped[source.get(ann["category_id"], ann["category_id"])] += 1
            continue
        x, y, w, h = ann["bbox"]
        if w < 1 or h < 1:
            thin += 1
            continue
        kept.setdefault(ann["image_id"], []).append({**ann, "category_id": cid})

    out = {"info": {"description": f"{coco.get('info', {}).get('description', '')} -> ABRA master".strip(),
                    "date_created": date_created or now()},
           "licenses": coco.get("licenses", []), "images": [], "annotations": [],
           "categories": [{"id": i, "name": name, "supercategory": "object"} for name, i in new_id.items()]}
    empty = 0
    for img in coco["images"]:
        anns = kept.get(img["id"], [])
        if not anns:
            empty += 1
            if drop_empty:
                continue
        out["images"].append({"id": img["id"], "file_name": file_prefix + img["file_name"],
                              "height": img["height"], "width": img["width"]})
        for ann in anns:
            x, y, w, h = ann["bbox"]
            out["annotations"].append({"id": len(out["annotations"]) + 1, "image_id": img["id"],
                                       "category_id": ann["category_id"], "bbox": [x, y, w, h],
                                       "area": ann.get("area", w * h), "iscrowd": ann.get("iscrowd", 0)})
    per_class = Counter(a["category_id"] for a in out["annotations"])
    report = {"images": len(out["images"]), "empty_images": empty, "empty_dropped": drop_empty,
              "boxes": {name: per_class[i] for name, i in new_id.items()},
              "dropped_categories": dict(dropped.most_common()), "dropped_thin": thin}
    return out, report


def convert_coco(args):
    coco = json.loads(args.coco.read_text())
    out, report = convert(coco, args.classes, parse_rename(args.rename), args.file_prefix, args.drop_empty,
                          args.date_created)
    ann = args.out_dir / "annotations"
    if args.split == "train":
        path = ann / "annotations_train.jsonl"
        write_jsonl(coco_to_odvg(out), path)
    else:
        path = ann / f"annotations_{args.split}.json"
        write_json(out, path)
    print(f"{report['images']} images ({report['empty_images']} without a kept box"
          f"{', dropped' if args.drop_empty else ''}) -> {path}")
    print("boxes: " + ", ".join(f"{name} {n}" for name, n in report["boxes"].items()))
    if report["dropped_categories"]:
        print("dropped (not in --classes): " + ", ".join(f"{k} {v}" for k, v in report["dropped_categories"].items()))
    if report["dropped_thin"]:
        print(f"dropped: {report['dropped_thin']} boxes thinner than 1 px")
    for name, n in report["boxes"].items():
        if n == 0:
            print(f"WARNING: no {name} box in this split")
    return report
