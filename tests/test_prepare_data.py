"""prepare-data / data-stats / convert-bdd on a synthetic Cityscapes tree (no images or downloads needed).

With ``ABRA_RELEASE_ANNOTATIONS`` pointing to the released files in the ``datasets/cityscapes`` layout (including
``normal/annotations/annotations_{train.jsonl,val.json}``), the release is also rebuilt and compared."""
import json
import os
from pathlib import Path

import pytest

from abra.cli import build_parser
from abra.data import bdd, prepare, stats

# (city, image) -> objects; polygons are rectangles unless given as a list of points
SCENES = {
    ("aachen", "aachen_000000_000019"): [("car", 0, 0, 10, 10), ("car", 0, 0, 10, 10), ("person", 5, 5, 8, 20),
                                         ("bus", 20, 20, 60, 50), ("cargroup", 0, 0, 5, 5)],
    ("aachen", "aachen_000001_000019"): [("person", 1, 1, 4, 9), ("bicycle", 2, 2, 9, 9), ("rider", 3, 3, 6, 9)],
    ("bochum", "bochum_000000_000313"): [("train", 0, 10, 90, 40), ("truck", 1, 1, 30, 20), ("car", 40, 40, 41, 48)],
    ("bochum", "bochum_000001_000313"): [("bus", 1, 1, 3, 1), ("motorcycle", 1, 1, 2, 2)],  # bus box is 0 px tall
    ("bochum", "bochum_000002_000313"): [("car", 5, 5, 25, 15), ("car", 30, 5, 50, 15), ("bicycle", 0, 0, 3, 3),
                                         ("person", [[1, 1], [2, 2]])],  # degenerate polygon: skipped
}


def rect(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def make_cityscapes(root, splits=("train", "val")):
    for split in splits:
        for (city, name), objects in SCENES.items():
            objs = [{"label": o[0], "polygon": o[1] if len(o) == 2 else rect(*o[1:])} for o in objects]
            ann = root / "gtFine" / split / city / f"{name}_gtFine_polygons.json"
            ann.parent.mkdir(parents=True, exist_ok=True)
            ann.write_text(json.dumps({"imgHeight": 64, "imgWidth": 128, "objects": objs}))
    return root


def run(argv):
    return prepare.prepare(build_parser().parse_args(["prepare-data"] + argv))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cs")
    make_cityscapes(tmp / "cityscapes")
    out = tmp / "out"
    run(["--cityscapes_root", str(tmp / "cityscapes"), "--out_dir", str(out), "--exclude_unavailable", "class", "all",
         "--few_shot", "1", "2", "--top_k", "3", "--top_k_variants", "1", "2", "--date_created", "2026-01-01 00:00:00"])
    return out


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_conversion_matches_original_rules(built):
    coco = json.loads((built / "normal/annotations/annotations_val.json").read_text())
    assert [c["name"] for c in coco["categories"]] == list(prepare.CITYSCAPES_CLASSES)
    assert len(coco["images"]) == 5
    # cargroup (not a class), the 0-px bus and the 2-point person are dropped; duplicates are kept
    assert len(coco["annotations"]) == 14
    first = coco["annotations"][0]
    assert first["bbox"] == [0, 0, 10, 10] and first["area"] == 100.0 and len(first["segmentation"][0]) == 8
    train = jsonl(built / "normal/annotations/annotations_train.jsonl")
    assert train[0]["detection"]["instances"][0] == {"bbox": [0, 0, 10, 10], "label": 3, "category": "car"}


def test_domain_file_and_label_maps(built):
    maps = built / "label_maps"
    assert json.loads((maps / "top_3.json").read_text()) == {"1": "person", "2": "car", "3": "bicycle"}
    assert json.loads((maps / "uda_map.json").read_text()) == {"1": "rider", "2": "truck", "3": "bus", "4": "train",
                                                                "5": "motorcycle"}
    assert (maps / "experts" / "bus.json").read_text() == '{\n    "1": "bus"\n}'
    domain = jsonl(built / "normal/annotations/domains/annotations_train_domain.jsonl")
    assert len(domain) == 4  # bochum_000001 has no car/person/bicycle
    labels = [(i["category"], i["label"]) for i in domain[0]["detection"]["instances"]]
    assert labels == [("car", 2), ("car", 2), ("person", 1)]  # grouped by first appearance, relabelled by original id


def test_top_k_variants_and_multi(built):
    maps = built / "label_maps"
    assert json.loads((maps / "top_1.json").read_text()) == {"1": "car"}
    assert json.loads((maps / "top_2.json").read_text()) == {"1": "person", "2": "car"}
    ann = built / "fog/annotations"
    # k = 1 equals the class file of the most frequent class (the original k = 1 runs used experts/car)
    assert (ann / "domains/annotations_train_domain_top1.jsonl").read_bytes() == \
        (ann / "experts/car/annotations_train.jsonl").read_bytes()
    top2 = jsonl(ann / "domains/annotations_train_domain_top2.jsonl")
    assert {i["category"] for r in top2 for i in r["detection"]["instances"]} == {"car", "person"}
    multi = json.loads((ann / "multi/rider/annotations_val.json").read_text())
    assert len(multi["images"]) == 1 and len(multi["annotations"]) == 3  # rider image with its person and bicycle


def test_fog_is_normal_with_renamed_images(built):
    normal = jsonl(built / "normal/annotations/experts/car/annotations_train.jsonl")
    fog = jsonl(built / "fog/annotations/experts/car/annotations_train.jsonl")
    assert fog[0]["filename"] == "aachen/aachen_000000_000019_leftImg8bit_foggy_beta_0.02.png"
    assert [{**r, "filename": None} for r in normal] == [{**r, "filename": None} for r in fog]
    val = json.loads((built / "fog/annotations/experts/bus/annotations_val.json").read_text())
    assert val["info"]["date_created"] == "2026-01-01 00:00:00" and val["categories"][0]["id"] == 1


def test_strict_filters(built):
    domains = built / "fog/annotations/domains"
    assert len(jsonl(domains / "annotations_train_domain_no_bus.jsonl")) == 3
    strict = jsonl(domains / "annotations_train_domain_no_unavailable.jsonl")
    assert [r["filename"] for r in strict] == ["bochum/bochum_000002_000313_leftImg8bit_foggy_beta_0.02.png"]
    assert not (built / "normal/annotations/domains/annotations_train_domain_no_bus.jsonl").exists()  # target only


def test_few_shot_is_seeded_and_nested(built):
    files = built / "fog/annotations/experts/rider"
    assert not (built / "fog/annotations/experts/car/annotations_train_1.jsonl").exists()  # available: no few-shot
    one, two = jsonl(files / "annotations_train_1.jsonl"), jsonl(files / "annotations_train_2.jsonl")
    assert len(one) == 1 and len(two) == 1  # only one rider image exists
    records = [{"filename": str(i), "height": 1, "width": 1, "detection": {"instances": [{"bbox": [0, 0, 1, 1]}] * 2}}
               for i in range(10)]
    small, large = (prepare.few_shot(records, n, "images", 0, "x") for n in (3, 6))
    assert len(small) == 3 and len(large) == 6 and all(r in large for r in small)
    assert prepare.few_shot(records, 3, "images", 0, "x") == small != prepare.few_shot(records, 3, "images", 1, "x")
    inst = prepare.few_shot(records, 5, "instances", 0, "x")
    assert sum(len(r["detection"]["instances"]) for r in inst) == 5


def test_check_against_itself(built, tmp_path):
    result = run(["--train_master", str(built / "normal/annotations/annotations_train.jsonl"),
                  "--val_master", str(built / "normal/annotations/annotations_val.json"), "--out_dir", str(tmp_path),
                  "--check_against", str(built)])
    assert not result["different"] and not result["missing"] and len(result["identical"]) == 65


@pytest.mark.skipif(not os.environ.get("ABRA_RELEASE_ANNOTATIONS"), reason="set ABRA_RELEASE_ANNOTATIONS")
def test_release_is_regenerated(tmp_path):
    ref = Path(os.environ["ABRA_RELEASE_ANNOTATIONS"])
    result = run(["--train_master", str(ref / "normal/annotations/annotations_train.jsonl"),
                  "--val_master", str(ref / "normal/annotations/annotations_val.json"), "--out_dir", str(tmp_path),
                  "--check_against", str(ref)])
    assert not result["different"] and len(result["identical"]) >= 29


def test_sdgod_layout(built, tmp_path):
    root = tmp_path / "sdgod"
    for domain, cs in (("daytime_sunny", "normal"), ("night_rainy", "fog")):
        ann = root / domain / "annotations"
        ann.mkdir(parents=True)
        (ann / "annotations_train.jsonl").write_bytes((built / cs / "annotations/annotations_train.jsonl").read_bytes())
        (ann / "annotations_test.json").write_bytes((built / cs / "annotations/annotations_val.json").read_bytes())
    out = tmp_path / "out"
    run(["--benchmark", "sdgod", "--sdgod_root", str(root), "--out_dir", str(out), "--exclude_unavailable", "all",
         "--date_created", "2026-01-01 00:00:00"])
    assert json.loads((out / "label_maps/domains/top_3.json").read_text()) == {"1": "person", "2": "car", "3": "bicycle"}
    assert not (out / "label_maps/uda_map.json").exists()
    assert (out / "night_rainy/annotations/domains/annotations_train_domain_no_unavailable.jsonl").exists()
    assert not (out / "daytime_sunny/annotations/domains/annotations_train_domain_no_unavailable.jsonl").exists()
    multi = json.loads((out / "daytime_sunny/annotations/multi/bus/annotations_test.json").read_text())
    assert [c["name"] for c in multi["categories"]] == sorted(prepare.CITYSCAPES_CLASSES)  # renumbered by name
    assert len(multi["images"]) == 1 and len(multi["annotations"]) == 4  # the bus image keeps its cars and person
    assert [a["category_id"] for a in multi["annotations"]] == [3, 3, 5, 2]  # car, car, person, bus


def test_stats(built):
    a = build_parser().parse_args(["data-stats", "--source_train", str(built / "normal/annotations/annotations_train.jsonl"),
                                   "--target_eval", str(built / "fog/annotations/annotations_val.json"),
                                   "--describe", str(built / "fog/annotations/experts/bus/annotations_train.jsonl")])
    r = stats.data_stats(a)
    assert r["available"] == ["car", "person", "bicycle"]
    p = r["partition"]["source train"]
    assert p["images"] == 5 and p["instances"] == 14 and p["pct_classes_available"] == 37.5
    assert p["images_only_available"] == 1 and p["images_any_unavailable"] == 4
    d = r["domain_expert"]
    assert d["images"] == 4 and d["images_any_unavailable"] == 3 and d["strict"]["all"]["images"] == 1
    assert d["per_class"]["motorcycle"]["images_with_class_outside_set"] == 1
    assert r["eval_protocol"]["bus"]["eval_images"] == 1 and r["eval_protocol"]["bus"]["negative_images_excluded"] == 4
    assert r["duplicate_boxes"]["source train"] == 1
    described = next(iter(r["describe"].values()))  # bus file keeps 1 box; its image also has cars and a person
    assert described["found_in_masters"] == 1 and described["instances_in_file"] == 1
    assert described["partition"]["classes"]["car"]["images"] == 1 and described["partition"]["instances"] == 4


def test_convert_bdd(tmp_path):
    src = tmp_path / "bdd"
    src.mkdir()
    box = lambda x1, y1, x2, y2: {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    files = {"b": {"name": "b", "attributes": {"timeofday": "daytime"},
                   "frames": [{"timestamp": 1, "objects": [{"category": "car", "box2d": box(0, 0, 2, 3)},
                                                           {"category": "lane", "poly2d": []}]}]},
             "a": {"name": "a", "attributes": {"timeofday": "daytime"},
                   "frames": [{"timestamp": 1, "objects": [{"category": "bus", "box2d": box(1, 1, 5, 5)}]},
                              {"timestamp": 2, "objects": [{"category": "traffic sign", "box2d": box(1, 1, 2, 2)}]}]},
             "c": {"name": "c", "attributes": {"timeofday": "night"},
                   "frames": [{"timestamp": 1, "objects": [{"category": "car", "box2d": box(0, 0, 1, 1)}]}]}}
    for name, data in files.items():
        (src / f"{name}.json").write_text(json.dumps(data))
    coco, skipped, multi = bdd.convert(src, bdd.DEFAULT_CLASSES, "daytime", 1280, 720)
    assert [i["file_name"] for i in coco["images"]] == ["a.jpg", "b.jpg"]  # sorted files; a's 2nd frame has no class
    assert [c["name"] for c in coco["categories"]] == ["bus", "car"]
    assert coco["annotations"][1]["bbox"] == [0, 0, 2, 3] and coco["annotations"][1]["area"] == 6
    assert skipped == {"timeofday": 1, "no_selected_class": 1, "no_box2d": 1} and multi == ["a"]
    coco, *_ = bdd.convert(src, ["all"], "all", 1280, 720, frame_names="timestamp")
    assert [i["file_name"] for i in coco["images"]] == ["a_1.jpg", "a_2.jpg", "b_1.jpg", "c_1.jpg"]
