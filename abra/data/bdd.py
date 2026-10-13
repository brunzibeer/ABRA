"""``abra convert-bdd``: BDD-format label files (one json per image/sequence) -> one COCO json (SDGOD splits).

Port of the original ``bdd_to_coco.py``. Same output, with these changes:
- files are read in sorted order (the original used the file-system order, so image ids and category ids could
  change between machines);
- the time-of-day filter is an option (``--timeofday daytime`` was hard-wired; pass ``--timeofday all`` to keep all);
- a file with several frames used to give every frame the same ``file_name``; such files are now reported, and
  their frames are named ``<name>_<timestamp>.jpg`` only with ``--frame_names timestamp``.
Only objects with a ``box2d`` are converted; frames without any selected class are dropped.
"""
import json
from datetime import datetime
from pathlib import Path

DEFAULT_CLASSES = ["car", "person", "truck", "bus", "bicycle", "bike", "motorcycle", "motor", "rider"]


def add_bdd_args(p):
    p.add_argument("--input_dir", type=Path, required=True, help="folder of BDD-format json files")
    p.add_argument("--output", type=Path, required=True, help="output COCO json")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="BDD categories to keep ('all' = every one)")
    p.add_argument("--timeofday", default="daytime", help="keep only files with this attributes.timeofday ('all' = no filter)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--frame_names", default="name", choices=["name", "timestamp"],
                   help="file_name of a frame: <name>.jpg (original) or <name>_<timestamp>.jpg")


def convert(input_dir, classes, timeofday, width, height, frame_names="name"):
    selected = None if classes in (None, ["all"]) else set(classes)
    images, raw, cat_order = [], [], {}
    skipped = {"timeofday": 0, "no_selected_class": 0, "no_box2d": 0}
    multi_frame = []
    for path in sorted(Path(input_dir).glob("*.json")):
        data = json.loads(path.read_text())
        if timeofday != "all" and data.get("attributes", {}).get("timeofday") != timeofday:
            skipped["timeofday"] += 1
            continue
        name = data.get("name", path.stem)
        frames = data.get("frames", [])
        if len(frames) > 1:
            multi_frame.append(name)
        for frame in frames:
            objects = []
            for obj in frame.get("objects", []):
                if "box2d" not in obj:
                    skipped["no_box2d"] += 1
                elif selected is None or obj["category"] in selected:
                    objects.append(obj)
            if not objects:
                skipped["no_selected_class"] += 1
                continue
            image_id = len(images) + 1
            stem = f"{name}_{frame.get('timestamp', '')}" if frame_names == "timestamp" else name
            images.append({"id": image_id, "file_name": f"{stem}.jpg", "width": width, "height": height})
            for obj in objects:
                cat_order.setdefault(obj["category"], len(cat_order) + 1)
                b = obj["box2d"]
                bbox = [b["x1"], b["y1"], b["x2"] - b["x1"], b["y2"] - b["y1"]]
                raw.append((image_id, obj["category"], [round(v, 4) for v in bbox], round(bbox[2] * bbox[3], 4)))
    coco = {"info": {"description": f"BDD -> COCO ({timeofday})", "date_created": datetime.now().strftime("%Y/%m/%d"),
                     "classes_filter": sorted(selected) if selected else "all"},
            "licenses": [], "images": images,
            "annotations": [{"id": i, "image_id": img, "category_id": cat_order[cat], "bbox": bbox, "area": area,
                             "iscrowd": 0} for i, (img, cat, bbox, area) in enumerate(raw, start=1)],
            "categories": [{"id": i, "name": c, "supercategory": c.split("/")[0]} for c, i in cat_order.items()]}
    return coco, skipped, multi_frame


def convert_bdd(args):
    coco, skipped, multi_frame = convert(args.input_dir, args.classes, args.timeofday, args.width, args.height,
                                         args.frame_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, indent=2))
    print(f"{len(coco['images'])} images, {len(coco['annotations'])} boxes, categories "
          f"{[c['name'] for c in coco['categories']]} -> {args.output}")
    print(f"skipped: {skipped['timeofday']} files (timeofday), {skipped['no_selected_class']} frames without a "
          f"selected class, {skipped['no_box2d']} objects without box2d")
    if multi_frame and args.frame_names == "name":
        print(f"WARNING: {len(multi_frame)} files have several frames, which share one file_name "
              f"(e.g. {multi_frame[0]}); use --frame_names timestamp if the images are stored per frame")
    return coco
