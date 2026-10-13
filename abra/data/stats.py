"""``abra data-stats``: dataset statistics for the class partition of a benchmark (rebuttal / supplementary tables).

Inputs are master files with every class (ODVG jsonl or COCO json, detected from the content): the source train
split, optionally the target train split (default: the source one, as for Cityscapes -> Foggy Cityscapes, whose
images are the same scenes) and the target evaluation split. Tables (markdown on stdout, everything in ``--out``):

T1  class partition: images / instances per class and split, % of classes, % of instances and % of images that
    are available (objectified) or unavailable;
T2  target domain-expert (Objectification) images that contain unavailable-class instances, overall and per class;
T3  what each --exclude_unavailable mode of ``prepare-data`` leaves for the target domain expert;
T4  evaluation protocol: images evaluated per class (images containing it) vs. all evaluation images;
T5  classes present in the images of any other annotation file (``--describe``, e.g. the target images used by
    FDA), looked up in the target masters;
plus exact duplicate boxes per split.
"""
import json
from collections import Counter
from pathlib import Path


def load(path):
    """-> list of (file_name, [(category, bbox), ...]) for ODVG jsonl or COCO json."""
    path = Path(path)
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and "images" in data:
        names = {c["id"]: c["name"] for c in data["categories"]}
        boxes = {img["id"]: [] for img in data["images"]}
        for ann in data["annotations"]:
            boxes[ann["image_id"]].append((names[ann["category_id"]], tuple(ann["bbox"])))
        return [(img["file_name"], boxes[img["id"]]) for img in data["images"]]
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [(r["filename"], [(i["category"], tuple(i["bbox"])) for i in r["detection"]["instances"]]) for r in records]


def class_counts(images):
    inst, imgs = Counter(), Counter()
    for _, boxes in images:
        inst.update(c for c, _ in boxes)
        imgs.update({c for c, _ in boxes})
    return inst, imgs


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def partition(images, available, unavailable):
    inst, imgs = class_counts(images)
    n_inst = sum(inst.values())
    cats = [{c for c, _ in boxes} for _, boxes in images]
    only_avail = sum(1 for s in cats if s and s <= set(available))
    any_unav = sum(1 for s in cats if s & set(unavailable))
    n_classes = len(available) + len(unavailable)
    return {
        "images": len(images), "instances": n_inst, "images_without_instances": sum(1 for s in cats if not s),
        "classes": {c: {"role": "available" if c in available else "unavailable", "instances": inst[c],
                        "images": imgs[c], "pct_instances": pct(inst[c], n_inst), "pct_images": pct(imgs[c], len(images))}
                    for c in list(available) + list(unavailable)},
        "pct_classes_available": pct(len(available), n_classes),
        "pct_instances_available": pct(sum(inst[c] for c in available), n_inst),
        "pct_instances_unavailable": pct(sum(inst[c] for c in unavailable), n_inst),
        "images_only_available": only_avail, "pct_images_only_available": pct(only_avail, len(images)),
        "images_any_unavailable": any_unav, "pct_images_any_unavailable": pct(any_unav, len(images)),
    }


def domain_expert_images(images, available, unavailable):
    """T2 + T3 on the target train split: the Objectification set = images with >= 1 available instance."""
    dom = [(f, boxes) for f, boxes in images if any(c in available for c, _ in boxes)]
    cats = [{c for c, _ in boxes} for _, boxes in dom]
    avail_inst = lambda keep: sum(1 for (_, boxes), k in zip(dom, keep) if k for c, _ in boxes if c in available)
    per_class, strict = {}, {"none": {"images": len(dom), "pct": 100.0, "available_instances": avail_inst([True] * len(dom))}}
    for c in unavailable:
        has = [c in s for s in cats]
        n_total = sum(1 for _, boxes in images if any(x == c for x, _ in boxes))
        per_class[c] = {"images": sum(has), "pct_images": pct(sum(has), len(dom)),
                        "instances": sum(1 for (_, boxes), h in zip(dom, has) if h for x, _ in boxes if x == c),
                        "images_with_class_outside_set": n_total - sum(has)}
        keep = [not h for h in has]
        strict[f"class:{c}"] = {"images": sum(keep), "pct": pct(sum(keep), len(dom)), "available_instances": avail_inst(keep)}
    keep = [not (s & set(unavailable)) for s in cats]
    strict["all"] = {"images": sum(keep), "pct": pct(sum(keep), len(dom)), "available_instances": avail_inst(keep)}
    any_unav = sum(1 for s in cats if s & set(unavailable))
    return {"images": len(dom), "images_any_unavailable": any_unav, "pct_any_unavailable": pct(any_unav, len(dom)),
            "per_class": per_class, "strict": strict}


def eval_protocol(images, classes):
    """T4: per class, images evaluated by the per-class protocol vs. all evaluation images."""
    inst, imgs = class_counts(images)
    out = {}
    for c in classes:
        sel = [{x for x, _ in boxes} for _, boxes in images if any(x == c for x, _ in boxes)]
        out[c] = {"eval_images": imgs[c], "instances": inst[c], "all_images": len(images),
                  "negative_images_excluded": len(images) - imgs[c],
                  "eval_images_with_other_unavailable": sum(1 for s in sel if s - {c} & set(classes))}
    return out


def duplicates(images):
    n = 0
    for _, boxes in images:
        n += sum(k - 1 for k in Counter(boxes).values() if k > 1)
    return n


def add_stats_args(p):
    p.add_argument("--source_train", type=Path, required=True, help="source train master (ODVG jsonl or COCO json)")
    p.add_argument("--target_train", type=Path, default=None, help="target train master (default: --source_train)")
    p.add_argument("--target_eval", type=Path, required=True, help="target evaluation master (all images, all classes)")
    p.add_argument("--available", nargs="+", default=None,
                   help="available (objectified) classes (default: the --top_k most frequent source train classes)")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--unavailable", nargs="+", default=None, help="default: every other class")
    p.add_argument("--describe", type=Path, nargs="*", default=[], help="other annotation files to describe (T5)")
    p.add_argument("--name", default="benchmark", help="title of the tables")
    p.add_argument("--out", type=Path, default=None, help="write all numbers as json")


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def data_stats(args):
    splits = {"source train": load(args.source_train)}
    splits["target train"] = load(args.target_train) if args.target_train else splits["source train"]
    splits["target eval"] = load(args.target_eval)
    inst, _ = class_counts(splits["source train"])
    available = args.available or [c for c, _ in inst.most_common(args.top_k)]
    all_classes = sorted({c for images in splits.values() for _, boxes in images for c, _ in boxes})
    unavailable = args.unavailable or [c for c in sorted(inst, key=lambda c: -inst[c]) if c not in available] + \
        [c for c in all_classes if c not in inst and c not in available]

    result = {"name": args.name, "available": available, "unavailable": unavailable,
              "partition": {s: partition(images, available, unavailable) for s, images in splits.items()},
              "domain_expert": domain_expert_images(splits["target train"], available, unavailable),
              "eval_protocol": eval_protocol(splits["target eval"], unavailable),
              "duplicate_boxes": {s: duplicates(images) for s, images in splits.items()},
              "describe": {}}
    # T5: the classes present in an image come from the masters (a split file keeps only some of its boxes)
    master = {f: boxes for images in (splits["target train"], splits["target eval"]) for f, boxes in images}
    for path in args.describe:
        images = load(path)
        found = [(f, master[f]) for f, _ in images if f in master]
        result["describe"][str(path)] = {"images": len(images), "found_in_masters": len(found),
                                         "instances_in_file": sum(len(b) for _, b in images),
                                         "partition": partition(found, available, unavailable)}

    f = lambda x: f"{x:.1f}"
    print(f"# {args.name}\n\nAvailable (objectified): {', '.join(available)}. Unavailable: {', '.join(unavailable)}.\n")
    print("## T1 Class partition\n")
    rows = []
    for c in available + unavailable:
        role = "available" if c in available else "unavailable"
        rows.append([c, role] + [f"{p['classes'][c]['images']} / {p['classes'][c]['instances']}"
                                 for p in result["partition"].values()])
    print(_table(["class", "role"] + [f"{s} img / inst" for s in splits], rows))
    print()
    rows = []
    for s, p in result["partition"].items():
        rows.append([s, p["images"], p["instances"], f(p["pct_classes_available"]), f(p["pct_instances_available"]),
                     f(p["pct_images_only_available"]), f(p["pct_images_any_unavailable"]), p["images_without_instances"]])
    print(_table(["split", "images", "instances", "% classes avail.", "% instances avail.", "% images only avail.",
                  "% images >=1 unavail.", "images w/o boxes"], rows))
    d = result["domain_expert"]
    print(f"\n## T2 Target domain-expert images with unavailable classes\n\n{d['images']} images contain >= 1 "
          f"available instance; {d['images_any_unavailable']} ({f(d['pct_any_unavailable'])}%) also contain >= 1 "
          f"unavailable instance (unlabelled during Objectification).\n")
    print(_table(["class", "images", "% of set", "instances", "images with the class outside the set"],
                 [[c, v["images"], f(v["pct_images"]), v["instances"], v["images_with_class_outside_set"]]
                  for c, v in d["per_class"].items()]))
    print("\n## T3 Strict filter (prepare-data --exclude_unavailable)\n")
    print(_table(["mode", "images left", "% of set", "available instances left"],
                 [[m, v["images"], f(v["pct"]), v["available_instances"]] for m, v in d["strict"].items()]))
    print("\n## T4 Evaluation protocol (target eval split)\n")
    print(_table(["class", "evaluated images", "instances", "all images", "negative images excluded",
                  "evaluated images with another unavailable class"],
                 [[c, v["eval_images"], v["instances"], v["all_images"], v["negative_images_excluded"],
                   v["eval_images_with_other_unavailable"]] for c, v in result["eval_protocol"].items()]))
    if result["describe"]:
        print("\n## T5 Other files\n")
        print("Classes present in the images of each file, from the target masters (the file itself may keep only "
              "some boxes).\n")
        rows = []
        for path, v in result["describe"].items():
            p = v["partition"]
            rows.append(["/".join(Path(path).parts[-3:]), v["images"], v["found_in_masters"], v["instances_in_file"],
                         ", ".join(f"{c} {p['classes'][c]['images']}" for c in available + unavailable if p["classes"][c]["images"]),
                         f(p["pct_images_any_unavailable"])])
        print(_table(["file", "images", "found in masters", "boxes in file", "images containing each class",
                      "% images >=1 unavail."], rows))
    print("\nExact duplicate boxes (same image, class and box): " +
          ", ".join(f"{s} {n}" for s, n in result["duplicate_boxes"].items()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    return result
