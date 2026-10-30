"""convert-coco and prepare-data --benchmark masters on a synthetic two-modality benchmark (FLIR-like: one COCO json
per modality and split, with different category ids in each file). No images or downloads needed."""
import json
from pathlib import Path

import pytest

from abra.cli import build_parser, main
from abra.data import coco_domain, prepare

CLASSES = ["person", "bicycle", "car", "traffic light", "dog"]
RENAME = ["bike=bicycle", "light=traffic light"]

# image -> boxes (dataset names); the same scenes in both modalities
IMAGES = {
    "a.jpg": [("car", [0, 0, 10, 10]), ("car", [20, 20, 10, 10]), ("person", [5, 5, 4, 12]), ("light", [50, 5, 3, 8]),
              ("bike", [70, 10, 8, 8])],
    "b.jpg": [("bike", [1, 1, 8, 8]), ("person", [3, 3, 4, 10]), ("dog", [30, 30, 15, 10]), ("car", [60, 40, 10, 5])],
    "c.jpg": [("car", [1, 1, 0.5, 9])],  # thinner than 1 px: dropped, the image stays as a negative
}
# FLIR files number their categories independently: ids differ between modalities
IDS = {"rgb": {"person": 1, "bike": 2, "car": 3, "light": 10, "dog": 12},
       "thermal": {"car": 1, "person": 2, "bike": 3, "light": 4, "dog": 5, "skateboard": 6}}


def dataset_coco(modality, extra=()):
    ids = IDS[modality]
    coco = {"info": {"description": f"flir {modality}"}, "licenses": [], "images": [], "annotations": [],
            "categories": [{"id": i, "name": n, "supercategory": "unknown"} for n, i in ids.items()]}
    for image_id, (name, boxes) in enumerate(IMAGES.items(), start=1):
        coco["images"].append({"id": image_id, "file_name": f"data/{name}", "height": 80, "width": 100})
        for cat, box in list(boxes) + [b for img, b in extra if img == name]:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                                        "category_id": ids[cat], "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
    return coco


@pytest.fixture(scope="module")
def masters(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("flir")
    reports = {}
    for modality in ("rgb", "thermal"):
        extra = [("b.jpg", ("skateboard", [40, 40, 5, 5]))] if modality == "thermal" else []
        src = tmp / f"{modality}.json"
        src.write_text(json.dumps(dataset_coco(modality, extra)))
        for split in ("train", "val", "test"):
            reports[modality, split] = main(["convert-coco", "--coco", str(src), "--split", split,
                                             "--out_dir", str(tmp / "masters" / modality), "--classes", *CLASSES,
                                             "--rename", *RENAME, "--file_prefix", f"images_{modality}_{split}/",
                                             "--date_created", "2026-01-01 00:00:00"])
    return tmp / "masters", reports


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_same_ids_in_every_modality(masters):
    root, reports = masters
    for modality in ("rgb", "thermal"):
        test = json.loads((root / modality / "annotations/annotations_test.json").read_text())
        assert [(c["id"], c["name"]) for c in test["categories"]] == list(enumerate(CLASSES, start=1))
        assert test["images"][0]["file_name"] == f"images_{modality}_test/data/a.jpg"
        train = jsonl(root / modality / "annotations/annotations_train.jsonl")
        labels = {(i["label"], i["category"]) for r in train for i in r["detection"]["instances"]}
        assert labels == {(1, "person"), (2, "bicycle"), (3, "car"), (4, "traffic light"), (5, "dog")}
    rgb, thermal = reports["rgb", "train"], reports["thermal", "train"]
    assert rgb["boxes"] == thermal["boxes"] == {"person": 2, "bicycle": 2, "car": 3, "traffic light": 1, "dog": 1}
    assert rgb["dropped_thin"] == 1 and rgb["empty_images"] == 1 and rgb["images"] == 3  # c.jpg kept as a negative
    assert thermal["dropped_categories"] == {"skateboard": 1} and rgb["dropped_categories"] == {}


def test_drop_empty_and_input_errors():
    coco = dataset_coco("rgb")
    out, report = coco_domain.convert(coco, CLASSES, coco_domain.parse_rename(RENAME), drop_empty=True)
    assert [i["file_name"] for i in out["images"]] == ["data/a.jpg", "data/b.jpg"] and report["empty_images"] == 1
    with pytest.raises(SystemExit, match="not produced by any input category"):
        coco_domain.convert(coco, CLASSES, {})  # bicycle / traffic light need --rename
    with pytest.raises(SystemExit, match="not in the input categories"):
        coco_domain.convert(coco, CLASSES, {"bicycle": "bike"})
    with pytest.raises(SystemExit, match="SRC=DST"):
        coco_domain.parse_rename(["bike"])


def run(argv):
    return prepare.prepare(build_parser().parse_args(["prepare-data"] + argv))


def test_prepare_masters_with_val(masters, tmp_path):
    root, _ = masters
    out = tmp_path / "out"
    manifest = run(["--benchmark", "masters", "--masters_root", str(root), "--source_domain", "rgb",
                    "--out_dir", str(out), "--exclude_unavailable", "all", "--date_created", "2026-01-01 00:00:00"])
    assert manifest["objectified"] == ["car", "person", "bicycle"]
    assert manifest["unavailable"] == ["traffic light", "dog"] and manifest["val_domains"] == ["rgb", "thermal"]
    maps = out / "label_maps"
    assert json.loads((maps / "domains/top_3.json").read_text()) == {"1": "person", "2": "bicycle", "3": "car"}
    assert json.loads((maps / "uda_map.json").read_text()) == {"1": "traffic light", "2": "dog"}
    ann = out / "thermal/annotations"
    for split in ("val", "test"):
        dog = json.loads((ann / f"experts/dog/annotations_{split}.json").read_text())
        assert len(dog["images"]) == 1 and dog["categories"][0]["name"] == "dog"
        assert (ann / f"multi/dog/annotations_{split}.json").exists()
    assert (ann / "annotations_val.json").exists()
    assert (ann / "domains/annotations_train_domain_no_unavailable.jsonl").exists()
    assert not (out / "rgb/annotations/domains/annotations_train_domain_no_unavailable.jsonl").exists()


# ------------------------------------------------------------------------------------------------ M3FD

M3FD = {  # image number -> (size, boxes as VOC corners)
    "00000": ((100, 80), [("People", 5, 5, 9, 17), ("Car", 0, 0, 10, 10), ("Lamp", 50, 5, 53, 13)]),
    "00001": ((100, 80), [("Car", 20, 20, 30, 30), ("Bus", 40, 30, 70, 50), ("Truck", 60, 10, 95, 40)]),
    "00002": ((100, 80), [("People", 1, 1, 5, 11), ("Motorcycle", 30, 30, 40, 42), ("Car", 90, 70, 110, 90)]),
    "00003": ((100, 80), [("Car", 1, 1, 9, 9), ("Lamp", 70, 2, 72, 9), ("Lamp", 3, 3, 3.5, 9)]),  # 0.5 px wide
    "00004": ((60, 40), [("Bus", 1, 1, 30, 20), ("People", 40, 5, 44, 15), ("Truck", 2, 20, 20, 39)]),
    "00005": ((100, 80), [("Motorcycle", 10, 10, 20, 25), ("Car", 50, 50, 60, 60), ("People", 70, 30, 74, 42)]),
}
M3FD_SPLIT = {"rgb_train": ["00000", "00001"], "ir_train": ["00002", "00003"], "val": ["00004"], "test": ["00005"]}


def m3fd_raw(root):
    (root / "Annotation").mkdir(parents=True)
    for image, ((w, h), boxes) in M3FD.items():
        objects = "".join(f"<object><name>{n}</name><pose>Unspecified</pose><truncated>0</truncated>"
                          f"<difficult>0</difficult><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax>"
                          f"<ymax>{y2}</ymax></bndbox></object>" for n, x1, y1, x2, y2 in boxes)
        (root / "Annotation" / f"{image}.xml").write_text(
            f"<annotation><folder>F</folder><filename>{image}.png</filename><size><width>{w}</width>"
            f"<height>{h}</height><depth>3</depth></size>{objects}</annotation>")
    return root


@pytest.fixture(scope="module")
def m3fd(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m3fd")
    split = tmp / "split.json"
    split.write_text(json.dumps({"splits": M3FD_SPLIT}))
    reports = main(["convert-m3fd", "--raw", str(m3fd_raw(tmp / "raw")), "--split_file", str(split),
                    "--out_dir", str(tmp / "masters"), "--lamp", "street lamp", "--date_created", "2026-01-01 00:00:00"])
    return tmp, reports


def test_m3fd_domains_and_splits(m3fd):
    tmp, reports = m3fd
    root = tmp / "masters"
    names = ["person", "car", "bus", "motorcycle", "street lamp", "truck"]
    for domain, folder, train in (("rgb", "Vis", ["00000", "00001"]), ("thermal", "Ir", ["00002", "00003"])):
        records = jsonl(root / domain / "annotations/annotations_train.jsonl")
        assert [r["filename"] for r in records] == [f"{folder}/{i}.png" for i in train]
        for part, image in (("val", "00004"), ("test", "00005")):
            coco = json.loads((root / domain / f"annotations/annotations_{part}.json").read_text())
            assert [c["name"] for c in coco["categories"]] == names
            assert [(i["id"], i["file_name"]) for i in coco["images"]] == [(int(image) + 1, f"{folder}/{image}.png")]
    val = json.loads((root / "thermal/annotations/annotations_val.json").read_text())
    assert [(a["category_id"], a["bbox"]) for a in val["annotations"]] == [(3, [1, 1, 29, 19]), (1, [40, 5, 4, 10]),
                                                                           (6, [2, 20, 18, 19])]
    assert reports["thermal train"]["dropped_thin"] == 1 and reports["thermal train"]["boxes"]["street lamp"] == 1
    car = [a for a in jsonl(root / "thermal/annotations/annotations_train.jsonl")[0]["detection"]["instances"]
           if a["category"] == "car"]
    assert car[0]["bbox"] == [90, 70, 100, 80]  # clipped to the 100x80 image
    assert json.loads((root / "conversion.json").read_text())["lamp"] == "street lamp"


def test_m3fd_split_errors(tmp_path):
    raw = m3fd_raw(tmp_path / "raw")
    for bad, message in (({**M3FD_SPLIT, "test": ["00005", "00000"]}, "more than one split"),
                         ({**M3FD_SPLIT, "test": ["00009"]}, "without an xml"),
                         ({k: v for k, v in M3FD_SPLIT.items() if k != "val"}, "no \\['val'\\]")):
        split = tmp_path / "split.json"
        split.write_text(json.dumps({"splits": bad}))
        with pytest.raises(SystemExit, match=message):
            main(["convert-m3fd", "--raw", str(raw), "--split_file", str(split), "--out_dir", str(tmp_path / "out"),
                  "--lamp", "lamp"])


def test_m3fd_prepare_masters(m3fd, tmp_path):
    tmp, _ = m3fd
    manifest = run(["--benchmark", "masters", "--masters_root", str(tmp / "masters"), "--source_domain", "rgb",
                    "--out_dir", str(tmp_path / "out"), "--date_created", "2026-01-01 00:00:00"])
    assert manifest["val_domains"] == ["rgb", "thermal"] and len(manifest["objectified"]) == 3


def test_prepare_masters_rejects_different_categories(masters, tmp_path):
    root, _ = masters
    bad = tmp_path / "masters"
    for modality in ("rgb", "thermal"):
        for f in (root / modality / "annotations").glob("annotations_*"):
            dst = bad / modality / "annotations" / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(f.read_bytes())
    test = bad / "thermal/annotations/annotations_test.json"
    coco = json.loads(test.read_text())
    coco["categories"][0], coco["categories"][1] = ({**coco["categories"][0], "name": "bicycle"},
                                                    {**coco["categories"][1], "name": "person"})
    test.write_text(json.dumps(coco))
    with pytest.raises(SystemExit, match="categories of thermal test differ"):
        run(["--benchmark", "masters", "--masters_root", str(bad), "--source_domain", "rgb", "--out_dir",
             str(tmp_path / "out")])
