"""``abra prepare-data``: build the annotation splits of a benchmark (Cityscapes -> Foggy Cityscapes, SDGOD).

Port of the original ``cs_to_coco`` scripts (``main.py``, ``coco_to_jsonl.py``, ``filter_classes.py``,
``extract_single_classes.py``, ``normal_to_foggy.py``, ``extract_single_class_few*.py``). Every file is written in
the byte format of the original scripts, so the released splits are regenerated identically (COCO json files differ
only in ``info.date_created``, see ``--date_created``). Standard library only.

Pipeline (one in-memory pass)::

    gtFine polygons (or the master files of the release)
      -> <domain>/annotations/annotations_{train.jsonl,val.json}       all classes (val = standard protocol)
      -> <domain>/annotations/domains/annotations_train_domain.jsonl   Objectification: top-k classes
      -> <domain>/annotations/experts/<cls>/annotations_{train.jsonl,val.json}
      -> label_maps/{standard_map,top_3,uda_map}.json, label_maps/experts/<cls>.json
    fog = normal with ``_foggy_beta_<beta>`` appended to every image name (same scenes, synthetic fog).

SDGOD (``--benchmark sdgod``): the masters are per weather domain (``<domain>/annotations/annotations_train.jsonl``,
``annotations_test.json``: the random train/test split of the release, kept as is); the domain, per-class and
``multi/<cls>/annotations_test.json`` files (images containing a class with all their boxes,
``extract_multi_classes.py``) are rebuilt from them, with the top-k classes of ``--source_domain`` for every domain.

Options for the rebuttal experiments: ``--exclude_unavailable`` (domain files of the target without images that
contain unavailable classes) and ``--few_shot`` (target class files with N images or N instances).
"""
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Cityscapes label -> id in gtFine, in the category order of the original conversion (COCO ids 1..8)
CITYSCAPES_CLASSES = {"person": 24, "rider": 25, "car": 26, "truck": 27, "bus": 28, "train": 31, "motorcycle": 32,
                      "bicycle": 33}
DOMAINS = ("normal", "fog")


# ---------------------------------------------------------------------------------------------- readers / writers

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def write_json(data, path, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------------------------- conversion

def _polygon_area(polygon):
    x, y = [p[0] for p in polygon], [p[1] for p in polygon]
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))


def cityscapes_to_coco(cityscapes_root, split, date_created, classes=tuple(CITYSCAPES_CLASSES)):
    """gtFine polygons -> COCO dict, as the original ``main.py``: box = polygon extent, area = polygon area,
    polygons with < 3 points or boxes thinner than 1 px are skipped. Images missing from ``leftImg8bit`` are
    skipped when that folder exists."""
    root = Path(cityscapes_root)
    ann_dir, img_dir = root / "gtFine" / split, root / "leftImg8bit" / split
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"annotation directory not found: {ann_dir}")
    coco = {"info": {"description": "Cityscapes Dataset converted to COCO format", "url": "https://www.cityscapes-dataset.com/",
                     "version": "1.0", "year": 2024, "contributor": "Cityscapes Dataset", "date_created": date_created},
            "licenses": [], "images": [], "annotations": [], "categories": []}
    category_map = {}
    for idx, name in enumerate(classes, start=1):
        coco["categories"].append({"id": idx, "name": name, "supercategory": "object"})
        category_map[name] = idx
    image_id = annotation_id = 1
    for city in sorted(d for d in ann_dir.iterdir() if d.is_dir()):
        for ann_file in sorted(city.glob("*_gtFine_polygons.json")):
            base = ann_file.stem.replace("_gtFine_polygons", "")
            if img_dir.is_dir() and not (img_dir / city.name / f"{base}_leftImg8bit.png").exists():
                print(f"Warning: image not found for {ann_file}")
                continue
            data = json.loads(ann_file.read_text())
            coco["images"].append({"id": image_id, "file_name": f"{city.name}/{base}_leftImg8bit.png",
                                   "height": data["imgHeight"], "width": data["imgWidth"], "license": 0,
                                   "flickr_url": "", "coco_url": "", "date_captured": ""})
            for obj in data["objects"]:
                if obj["label"] not in category_map or len(obj["polygon"]) < 3:
                    continue
                xs, ys = [p[0] for p in obj["polygon"]], [p[1] for p in obj["polygon"]]
                bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                if bbox[2] < 1 or bbox[3] < 1:
                    continue
                coco["annotations"].append({"id": annotation_id, "image_id": image_id,
                                            "category_id": category_map[obj["label"]], "bbox": bbox,
                                            "area": _polygon_area(obj["polygon"]),
                                            "segmentation": [[c for point in obj["polygon"] for c in point]],
                                            "iscrowd": 0})
                annotation_id += 1
            image_id += 1
    return coco


def coco_to_odvg(coco):
    """COCO dict -> ODVG records (xyxy boxes, category ids kept), as ``coco_to_jsonl.py --idmap identity``."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    records = []
    for img in coco["images"]:
        instances = []
        for ann in by_image[img["id"]]:
            x, y, w, h = ann["bbox"]
            instances.append({"bbox": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
                              "label": ann["category_id"], "category": names[ann["category_id"]]})
        records.append({"filename": img["file_name"], "height": img["height"], "width": img["width"],
                        "detection": {"instances": instances}})
    return records


def foggy_name(file_name, beta):
    path = Path(file_name)
    return str(path.parent / f"{path.stem}_foggy_beta_{beta}{path.suffix}")


def foggy_odvg(records, beta):
    return [{**r, "filename": foggy_name(r["filename"], beta)} for r in records]


def foggy_coco(coco, beta):
    """As ``normal_to_foggy.py``: only image names (and the info text) change."""
    fog = copy.deepcopy(coco)
    fog["info"]["description"] = f"Cityscapes Foggy Dataset (beta={beta}) converted to COCO format"
    fog["info"]["version"] = f"foggy_beta_{beta}"
    for img in fog["images"]:
        img["file_name"] = foggy_name(img["file_name"], beta)
    return fog


# ---------------------------------------------------------------------------------------------- selection

def odvg_categories(records):
    """label -> name, in order of first appearance (the order the original scripts iterate in)."""
    cats = {}
    for r in records:
        for inst in r["detection"]["instances"]:
            cats.setdefault(inst["label"], inst["category"])
    return cats


def odvg_select(records, classes):
    """Images with >= 1 instance of ``classes``; only those instances, relabelled 1..k by original label
    (``filter_classes.py``). Instances are grouped by class in order of first appearance in the input."""
    cats = odvg_categories(records)
    selected = [label for label, name in cats.items() if name in classes]
    missing = set(classes) - {cats[label] for label in selected}
    if missing:
        raise ValueError(f"classes not found in the annotations: {sorted(missing)}")
    new_label = {label: i for i, label in enumerate(sorted(selected), start=1)}
    out = []
    for r in records:
        instances = [{"bbox": inst["bbox"], "label": new_label[label], "category": cats[label]}
                     for label in selected for inst in r["detection"]["instances"] if inst["label"] == label]
        if instances:
            out.append({"filename": r["filename"], "height": r["height"], "width": r["width"],
                        "detection": {"instances": instances}})
    return out


def odvg_class(records, name):
    """Images with >= 1 instance of ``name``, only those instances, label 1 (``extract_single_classes.py``)."""
    out = []
    for r in records:
        instances = [{"bbox": i["bbox"], "label": 1, "category": name} for i in r["detection"]["instances"]
                     if i["category"] == name]
        if instances:
            out.append({"filename": r["filename"], "height": r["height"], "width": r["width"],
                        "detection": {"instances": instances}})
    return out


def odvg_without(records, classes, full):
    """Drop from ``records`` every image that contains an instance of ``classes`` in ``full`` (the annotations with
    every class: a domain file no longer has the boxes of the unavailable classes). Strict Objectification data."""
    classes = set(classes)
    drop = {r["filename"] for r in full if any(i["category"] in classes for i in r["detection"]["instances"])}
    return [r for r in records if r["filename"] not in drop]


def coco_class(coco, name, date_created):
    """Images with >= 1 instance of ``name``, only those annotations, category 1 (``extract_single_classes.py``
    on a COCO json)."""
    cat = next(c for c in coco["categories"] if c["name"] == name)
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] == cat["id"]:
            by_image[ann["image_id"]].append(ann)
    out = {"info": {"description": f"Filtered annotations for class: {name}", "url": "", "version": "1.0",
                    "year": 2024, "contributor": "", "date_created": date_created},
           "licenses": coco.get("licenses", []), "images": [], "annotations": [],
           "categories": [{"id": 1, "name": name, "supercategory": cat.get("supercategory", "object")}]}
    new_image_id = annotation_id = 1
    for img in coco["images"]:
        if img["id"] not in by_image:
            continue
        out["images"].append({**img, "id": new_image_id})
        for ann in by_image[img["id"]]:
            out["annotations"].append({**ann, "id": annotation_id, "image_id": new_image_id, "category_id": 1})
            annotation_id += 1
        new_image_id += 1
    return out


def few_shot(records, n, unit, seed, name):
    """``n`` images (all their instances of the class, ``extract_single_class_few_images.py``) or ``n`` instances
    (``extract_single_class_few.py``) of one single-class ODVG file. Seeded per class, and nested: the n-shot set
    contains every smaller-shot set of the same seed, so the shot curve adds data instead of resampling."""
    rng = random.Random(f"{seed}:{name}")
    if unit == "images":
        order = list(range(len(records)))
        rng.shuffle(order)
        keep = set(order[:n])
        return [r for i, r in enumerate(records) if i in keep]
    pool = [(i, j) for i, r in enumerate(records) for j in range(len(r["detection"]["instances"]))]
    rng.shuffle(pool)
    keep = set(pool[:n])
    out = []
    for i, r in enumerate(records):
        instances = [inst for j, inst in enumerate(r["detection"]["instances"]) if (i, j) in keep]
        if instances:
            out.append({**r, "detection": {"instances": instances}})
    return out


def coco_multi(coco, name, date_created):
    """Images with >= 1 instance of ``name`` (by image id), with ALL their annotations; categories renumbered 1..K
    by name (``extract_multi_classes.py``)."""
    old = {c["id"]: c for c in coco["categories"]}
    new_id = {cid: i for i, (cid, _) in enumerate(sorted(old.items(), key=lambda x: x[1]["name"]), start=1)}
    cat = next(c["id"] for c in coco["categories"] if c["name"] == name)
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    images = {img["id"]: img for img in coco["images"]}
    selected = sorted({ann["image_id"] for ann in coco["annotations"] if ann["category_id"] == cat})
    info = coco.get("info", {})
    out = {"info": {"description": f"Images containing class: {name} (with all their annotations)",
                    "url": info.get("url", ""), "version": "1.0", "year": info.get("year", 2024),
                    "contributor": info.get("contributor", ""), "date_created": date_created},
           "licenses": coco.get("licenses", []), "images": [], "annotations": [],
           "categories": [{**old[cid], "id": i} for cid, i in sorted(new_id.items(), key=lambda x: x[1])]}
    for i, img_id in enumerate(selected, start=1):
        out["images"].append({**images[img_id], "id": i})
    annotation_id = 1
    for i, img_id in enumerate(selected, start=1):
        for ann in by_image[img_id]:
            out["annotations"].append({**ann, "id": annotation_id, "image_id": i, "category_id": new_id[ann["category_id"]]})
            annotation_id += 1
    return out


def top_classes(records, k):
    counts = Counter(i["category"] for r in records for i in r["detection"]["instances"])
    return [name for name, _ in counts.most_common(k)]


# ---------------------------------------------------------------------------------------------- command

def add_prepare_args(p):
    p.add_argument("--benchmark", default="cityscapes", choices=["cityscapes", "sdgod"])
    g = p.add_argument_group("input: Cityscapes (gtFine or the master files of the release)")
    g.add_argument("--cityscapes_root", type=Path, default=None,
                   help="Cityscapes root with gtFine/{train,val} (and leftImg8bit/, used to skip missing images)")
    g.add_argument("--train_master", type=Path, default=None,
                   help="instead of gtFine/train: ODVG jsonl of all train images and classes")
    g.add_argument("--val_master", type=Path, default=None, help="instead of gtFine/val: COCO json of all val images")
    g.add_argument("--fog_beta", default="0.02", help="Foggy Cityscapes beta in the image names")
    g = p.add_argument_group("input: SDGOD")
    g.add_argument("--sdgod_root", type=Path, default=None,
                   help="folder with <domain>/annotations/annotations_{train.jsonl,test.json} for every domain")
    g.add_argument("--source_domain", default="daytime_sunny", help="SDGOD source domain (Day-Clear)")
    g = p.add_argument_group("output")
    g.add_argument("--out_dir", type=Path, required=True, help="e.g. datasets/cityscapes or datasets/sdgod")
    g.add_argument("--objectify_classes", nargs="+", default=None,
                   help="classes of the domain files (default: the --top_k most frequent source train classes)")
    g.add_argument("--top_k", type=int, default=3)
    g.add_argument("--expert_classes", nargs="+", default=None, help="per-class files (default: every class)")
    g.add_argument("--exclude_unavailable", nargs="+", default=["none"], choices=["none", "class", "all"],
                   help="extra target domain files: 'class' drops the images containing one unavailable class "
                        "(one file per class), 'all' drops the images containing any unavailable class")
    g.add_argument("--few_shot", type=int, nargs="+", default=[], help="target class files with N shots, e.g. 1 5 10 20 30")
    g.add_argument("--few_shot_unit", default="images", choices=["images", "instances"])
    g.add_argument("--few_shot_classes", nargs="+", default=None, help="default: the unavailable classes")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--date_created", default=None, help="info.date_created of the COCO files (default: now)")
    g.add_argument("--check_against", type=Path, default=None,
                   help="directory with the same layout (e.g. the release): compare every file written")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_file(mine, reference):
    """jsonl / label maps: identical bytes; COCO json: identical apart from info.date_created."""
    if mine.read_bytes() == reference.read_bytes():
        return True
    if mine.suffix != ".json":
        return False
    a, b = json.loads(mine.read_text()), json.loads(reference.read_text())
    for d in (a, b):
        if isinstance(d, dict) and isinstance(d.get("info"), dict):
            d["info"].pop("date_created", None)
    return a == b


def _inputs(args, date):
    """-> {domain: (train ODVG records, eval COCO dict)}, source domain, eval split name."""
    if args.benchmark == "sdgod":
        if not args.sdgod_root:
            raise SystemExit("pass --sdgod_root")
        domains = {}
        for d in sorted(p.name for p in Path(args.sdgod_root).iterdir() if (p / "annotations").is_dir()):
            ann = Path(args.sdgod_root) / d / "annotations"
            domains[d] = (read_jsonl(ann / "annotations_train.jsonl"), json.loads((ann / "annotations_test.json").read_text()))
        if args.source_domain not in domains:
            raise SystemExit(f"source domain {args.source_domain} not in {sorted(domains)}")
        return domains, args.source_domain, "test"
    if args.train_master:
        train = read_jsonl(args.train_master)
    elif args.cityscapes_root:
        train = coco_to_odvg(cityscapes_to_coco(args.cityscapes_root, "train", date))
    else:
        raise SystemExit("pass --cityscapes_root or --train_master")
    if args.val_master:
        val = json.loads(Path(args.val_master).read_text())
    elif args.cityscapes_root:
        val = cityscapes_to_coco(args.cityscapes_root, "val", date)
    else:
        raise SystemExit("pass --cityscapes_root or --val_master")
    # Foggy Cityscapes renders the same scenes: the fog files are the normal ones with the foggy image names
    return {"normal": (train, val), "fog": (foggy_odvg(train, args.fog_beta), foggy_coco(val, args.fog_beta))}, "normal", "val"


def prepare(args):
    date = args.date_created or now()
    domains, source, split = _inputs(args, date)
    sdgod = args.benchmark == "sdgod"
    train = domains[source][0]
    labels = odvg_categories(train)
    all_classes = [labels[k] for k in sorted(labels)]
    objectified = args.objectify_classes or top_classes(train, args.top_k)
    unavailable = [c for c in all_classes if c not in objectified]
    experts = args.expert_classes or sorted(all_classes)
    print(f"domains: {list(domains)} (source {source})\nclasses: {all_classes}\n"
          f"objectified (available): {objectified}\nunavailable: {unavailable}")

    out = Path(args.out_dir)
    written = []

    def emit(path, writer, data, **kw):
        writer(data, path, **kw)
        written.append(path)

    maps = out / "label_maps"
    top_map = maps / "domains" / f"top_{len(objectified)}.json" if sdgod else maps / f"top_{len(objectified)}.json"
    emit(maps / "standard_map.json", write_json, {str(k): labels[k] for k in sorted(labels)}, indent=4)
    emit(top_map, write_json,
         {str(i): labels[k] for i, k in enumerate(sorted(k for k in labels if labels[k] in objectified), start=1)}, indent=4)
    if not sdgod:
        emit(maps / "uda_map.json", write_json,
             {str(i): labels[k] for i, k in enumerate(sorted(k for k in labels if labels[k] in unavailable), start=1)}, indent=4)
    for name in experts:
        emit(maps / "experts" / f"{name}.json", write_json, {"1": name}, indent=4)

    for domain, (d_train, d_eval) in domains.items():
        target = domain != source
        ann = out / domain / "annotations"
        emit(ann / "annotations_train.jsonl", write_jsonl, d_train)
        emit(ann / f"annotations_{split}.json", write_json, d_eval)
        domain_records = odvg_select(d_train, objectified)
        emit(ann / "domains" / "annotations_train_domain.jsonl", write_jsonl, domain_records)
        if target and "class" in args.exclude_unavailable:
            for name in unavailable:
                emit(ann / "domains" / f"annotations_train_domain_no_{name}.jsonl", write_jsonl,
                     odvg_without(domain_records, [name], d_train))
        if target and "all" in args.exclude_unavailable:
            emit(ann / "domains" / "annotations_train_domain_no_unavailable.jsonl", write_jsonl,
                 odvg_without(domain_records, unavailable, d_train))
        for name in experts:
            class_records = odvg_class(d_train, name)
            emit(ann / "experts" / name / "annotations_train.jsonl", write_jsonl, class_records)
            emit(ann / "experts" / name / f"annotations_{split}.json", write_json, coco_class(d_eval, name, date))
            if sdgod:
                emit(ann / "multi" / name / f"annotations_{split}.json", write_json, coco_multi(d_eval, name, date))
            if target and name in (args.few_shot_classes or unavailable):
                for n in args.few_shot:
                    emit(ann / "experts" / name / f"annotations_train_{n}.jsonl", write_jsonl,
                         few_shot(class_records, n, args.few_shot_unit, args.seed, name))

    manifest = {"benchmark": args.benchmark, "source": source, "objectified": objectified, "unavailable": unavailable,
                "fog_beta": None if sdgod else args.fog_beta, "exclude_unavailable": args.exclude_unavailable,
                "few_shot": args.few_shot, "few_shot_unit": args.few_shot_unit, "seed": args.seed,
                "files": {str(p.relative_to(out)): {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in written}}
    write_json(manifest, out / "manifest.json")
    print(f"wrote {len(written)} files under {out} (list and checksums in manifest.json)")

    if args.check_against:
        ref = Path(args.check_against)
        result = {"identical": [], "different": [], "missing": []}
        for p in written:
            rel = p.relative_to(out)
            key = "missing" if not (ref / rel).exists() else ("identical" if same_file(p, ref / rel) else "different")
            result[key].append(str(rel))
        for key in ("different", "missing"):
            for rel in result[key]:
                print(f"  {key}: {rel}")
        print(f"check against {ref}: {len(result['identical'])} identical, {len(result['different'])} different, "
              f"{len(result['missing'])} not in the reference")
        if result["different"]:
            raise SystemExit(1)
        return result
    return manifest
