"""``abra convert-m3fd``: M3FD (``M3FD_Detection.zip``: ``Annotation/*.xml`` Pascal VOC, one per registered pair, and
``Vis/`` / ``Ir/`` PNG) -> the master files of the ``rgb`` and ``thermal`` domains, in the layout read by
``prepare-data --benchmark masters``. M3FD has no official split: ``--split_file`` (written by ``tools/m3fd_split.py``,
e.g. ``abra/data/splits/m3fd_split_v2.json``) lists the images of rgb_train, ir_train, val and test.

- ``rgb``: train = rgb_train (Vis images); val and test = the Vis images of val and test.
- ``thermal``: train = ir_train (Ir images); val and test = the Ir images of val and test.
The two training sets come from disjoint scenes (PLAN D2). Val and test are the same pairs in both modalities, with the
same boxes (one xml per registered pair).

Class names are the prompts: People -> person, Car -> car, Bus -> bus, Motorcycle -> motorcycle, Truck -> truck, and
Lamp -> ``--lamp``, which has no obvious prompt ("street lamp" or "traffic light": both are kept as variants, PLAN §3).
Boxes: VOC corners -> COCO [x, y, w, h], clipped to the image; boxes thinner than 1 px are dropped (as convert-coco).
Image ids are the M3FD number + 1 (00000 -> 1) in every file. File names are relative to the unpacked folder
(``Vis/00000.png``), which is then the image root of every command (``--root <raw> --add_split 0``).
Standard library only.
"""
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from abra.data.coco_domain import convert
from abra.data.prepare import coco_to_odvg, write_json, write_jsonl

M3FD_NAMES = ("People", "Car", "Bus", "Motorcycle", "Lamp", "Truck")
PROMPTS = {"People": "person", "Car": "car", "Bus": "bus", "Motorcycle": "motorcycle", "Truck": "truck"}
SPLITS = ("rgb_train", "ir_train", "val", "test")
DOMAINS = (("rgb", "Vis", "rgb_train"), ("thermal", "Ir", "ir_train"))


def add_m3fd_args(p):
    p.add_argument("--raw", type=Path, required=True, help="unpacked M3FD_Detection.zip (Annotation/, Vis/, Ir/)")
    p.add_argument("--split_file", type=Path, required=True, help="split json of tools/m3fd_split.py")
    p.add_argument("--out_dir", type=Path, required=True, help="masters root: writes rgb/ and thermal/ in it")
    p.add_argument("--lamp", required=True, help="class name (prompt) of the M3FD Lamp class, e.g. 'street lamp'")
    p.add_argument("--date_created", default=None, help="info.date_created of the COCO outputs (default: now)")


def read_annotations(raw):
    """-> {image number (str): {"width", "height", "objects": [(name, x1, y1, x2, y2)]}}."""
    files = sorted((Path(raw) / "Annotation").glob("*.xml"))
    if not files:
        raise SystemExit(f"no xml in {Path(raw) / 'Annotation'}")
    anns = {}
    for f in files:
        r = ET.parse(f).getroot()
        objects = []
        for o in r.findall("object"):
            box = [float(o.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
            objects.append((o.findtext("name"), *box))
        anns[f.stem] = {"width": int(float(r.findtext("size/width"))), "height": int(float(r.findtext("size/height"))),
                        "objects": objects}
    return anns


def read_split(path, anns):
    """-> (image numbers per split, images in no split, of which left out on purpose by the split file)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    split = data["splits"]
    missing = [s for s in SPLITS if s not in split]
    if missing:
        raise SystemExit(f"{path}: no {missing} in 'splits'")
    seen = Counter(i for s in SPLITS for i in split[s])
    twice = sorted(i for i, k in seen.items() if k > 1)
    if twice:
        raise SystemExit(f"{path}: {len(twice)} images in more than one split, e.g. {twice[:5]}")
    unknown = sorted(set(seen) - set(anns))
    if unknown:
        raise SystemExit(f"{path}: {len(unknown)} images without an xml, e.g. {unknown[:5]}")
    left_out = set(data.get("excluded_by_guard", [])) | set(data.get("dropped", []))
    return {s: split[s] for s in SPLITS}, sorted(set(anns) - set(seen)), left_out


def to_coco(anns, images, folder, description):
    """M3FD images -> COCO dict with the dataset class names (ids 1..6 in M3FD_NAMES order)."""
    ids = {name: i for i, name in enumerate(M3FD_NAMES, start=1)}
    coco = {"info": {"description": description}, "licenses": [], "images": [], "annotations": [],
            "categories": [{"id": i, "name": name, "supercategory": "object"} for name, i in ids.items()]}
    for image in images:
        a = anns[image]
        w, h = a["width"], a["height"]
        coco["images"].append({"id": int(image) + 1, "file_name": f"{folder}/{image}.png", "width": w, "height": h})
        for name, x1, y1, x2, y2 in a["objects"]:
            if name not in ids:
                raise SystemExit(f"{image}.xml: class {name!r} is not an M3FD class {M3FD_NAMES}")
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": int(image) + 1,
                                        "category_id": ids[name], "bbox": [x1, y1, x2 - x1, y2 - y1],
                                        "area": max(0.0, x2 - x1) * max(0.0, y2 - y1), "iscrowd": 0})
    return coco


def convert_m3fd(args):
    anns = read_annotations(args.raw)
    split, unassigned, left_out = read_split(args.split_file, anns)
    unexpected = [i for i in unassigned if i not in left_out]
    print(f"{len(unassigned) - len(unexpected)} images left out by the split file (guard band, broken pairs)")
    if unexpected:
        print(f"WARNING: {len(unexpected)} more images in no split, e.g. {unexpected[:5]}")
    classes = ["person", "car", "bus", "motorcycle", args.lamp, "truck"]
    rename = {**PROMPTS, "Lamp": args.lamp}
    reports = {}
    for domain, folder, train in DOMAINS:
        for part, key in (("train", train), ("val", "val"), ("test", "test")):
            coco = to_coco(anns, split[key], folder, f"M3FD {folder} {key} ({args.split_file.name})")
            out, report = convert(coco, classes, rename, date_created=args.date_created)
            ann = args.out_dir / domain / "annotations"
            if part == "train":
                path = ann / "annotations_train.jsonl"
                write_jsonl(coco_to_odvg(out), path)
            else:
                path = ann / f"annotations_{part}.json"
                write_json(out, path)
            reports[f"{domain} {part}"] = report
            print(f"{domain} {part} ({key}): {report['images']} images, "
                  + ", ".join(f"{name} {n}" for name, n in report["boxes"].items())
                  + (f"; dropped {report['dropped_thin']} boxes thinner than 1 px" if report["dropped_thin"] else "")
                  + f" -> {path}")
    write_json({"split_file": str(args.split_file), "lamp": args.lamp, "classes": classes, "reports": reports},
               args.out_dir / "conversion.json")
    return reports
