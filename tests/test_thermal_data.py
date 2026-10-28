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
