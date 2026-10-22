"""Revision tools (tools/revision_*.py, tools/projection_stats.py) on synthetic run folders, datasets and layers."""
import csv
import json

import pytest
import torch

from tools import projection_stats, revision_collect, revision_stats


def _coco(path, images, boxes):
    """COCO json with ``images`` file names and ``boxes`` = [(image index, class)]."""
    names = sorted({c for _, c in boxes} | {"bus"})
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    data = {"images": [{"id": i, "file_name": f, "width": 10, "height": 10} for i, f in enumerate(images)],
            "categories": cats,
            "annotations": [{"id": k, "image_id": i, "category_id": names.index(c) + 1, "bbox": [0, 0, 1, 1 + k],
                             "area": 1, "iscrowd": 0} for k, (i, c) in enumerate(boxes)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _run(root, name, test, result, eval_classes=None, residual=None):
    d = root / "eval" / name
    d.mkdir(parents=True)
    args = {"exp_name": name, "test_annotations": str(test), "eval_classes": eval_classes, "domain": None,
            "residual": str(residual) if residual else None}
    (d / "config.json").write_text(json.dumps({"git_commit": "abc1234", "environment": {"gpu": "G", "torch": "2.1"},
                                               "args": args}))
    (d / "results.json").write_text(json.dumps(result))


def test_parse_run_and_alpha_tie_break():
    assert revision_collect.parse_run("sdgod_daytime_foggy_ft_srcft2_bike") == ("sdgod", "daytime_foggy", "ft_srcft2", "bike")
    assert revision_collect.parse_run("cityscapes_fog_abra_mask_band-0-0.5_exp1_bus")[2] == "abra_mask_band-0-0.5_exp1"
    curve = [{"alpha": 0.8, "AP": 40.0}, {"alpha": 1.3, "AP": 41.0}, {"alpha": 0.9, "AP": 41.0}]
    assert revision_collect.select_alpha(curve)["alpha"] == 0.9   # tie on mAP -> closest to 1.0


def test_collect_writes_unrounded_rows_counts_and_alpha_curve(tmp_path):
    per_class = _coco(tmp_path / "experts/bus/annotations_val.json", ["a", "b"], [(0, "bus"), (1, "bus"), (1, "bus")])
    standard = _coco(tmp_path / "annotations_val.json", ["a", "b", "c"], [(0, "bus"), (2, "car")])
    residual = tmp_path / "res.pth"
    residual.write_bytes(b"residual")
    runs = tmp_path / "revision"
    _run(runs, "cityscapes_fog_zero_shot_exp1_bus", per_class, {"AP": 48.623456, "AP50": 70.1, "AP75": 50.2})
    _run(runs, "cityscapes_fog_zero_shot_std_exp1_bus", standard, {"AP": 34.5, "AP50": 50.0, "AP75": 30.0},
         eval_classes=["bus"])
    curve = [{"alpha": 1.0, "AP": 57.2, "AP50": 80.0, "AP75": 60.0}, {"alpha": 1.1, "AP": 57.3, "AP50": 79.0, "AP75": 61.0}]
    _run(runs, "cityscapes_fog_abra_alpha_exp1_bus", per_class,
         {"AP": 57.3, "AP50": 79.0, "AP75": 61.0, "alpha": 1.1, "alpha_curve": curve}, residual=residual)
    _run(runs, "cityscapes_fog_task_add_exp9_bus", per_class, {"AP": 1.0, "AP50": 1.0, "AP75": 1.0})   # no method
    rebuttal = tmp_path / "refactor"
    _run(rebuttal, "cityscapes_fog_abra_mask_topk-0.05_exp1_car", per_class, {"AP": 50.8, "AP50": 1.0, "AP75": 2.0})
    _run(rebuttal, "cityscapes_fog_task_add_exp1_bus", per_class, {"AP": 57.27, "AP50": 1.0, "AP75": 2.0})

    out = tmp_path / "out"
    revision_collect.main(["collect", "--runs", str(runs), "--rebuttal", str(rebuttal), "--out", str(out)])
    rows = {(r["method"], r["protocol"]): r for r in csv.DictReader(open(out / "results.csv"))}
    assert set(rows) == {("zero_shot", "per_class"), ("zero_shot", "standard"), ("abra_alpha", "per_class")}
    zs = rows["zero_shot", "per_class"]
    assert (zs["exp_id"], zs["mAP"], zs["n_images"], zs["n_gt"]) == ("E1", "48.623456", "2", "3")
    std = rows["zero_shot", "standard"]
    assert (std["exp_id"], std["n_images"], std["n_gt"]) == ("E3", "3", "1")
    alpha = rows["abra_alpha", "per_class"]
    assert (alpha["alpha"], alpha["mAP"], alpha["AP50"]) == ("1.1", "57.3", "79.0")
    assert alpha["checkpoint_id"].startswith(f"residual={residual}#")
    search = list(csv.DictReader(open(out / "alpha_search.csv")))
    assert [r["selected"] for r in search] == ["False", "True"]
    masks = {r["mask_id"]: r for r in csv.DictReader(open(out / "masks.csv"))}
    assert masks["topk:0.05"]["class_set"] == "avail" and masks["direct addition"]["class_set"] == "unav"
    report = (out / "sanity_report.md").read_text()
    assert "| cityscapes | fog | zero_shot | bus | 48.62 | 48.6235 | +0.0035 | yes |" in report
    assert "FAILED" in report   # the other E0 values are missing
    assert "abc1234" in (out / "metadata.yaml").read_text()


def test_alpha_map_reads_no_checkpoint(tmp_path, capsys):
    per_class = _coco(tmp_path / "experts/bus/annotations_val.json", ["a"], [(0, "bus")])
    curve = [{"alpha": 0.9, "AP": 30.0, "AP50": 1.0, "AP75": 1.0}, {"alpha": 1.4, "AP": 31.0, "AP50": 1.0, "AP75": 1.0}]
    _run(tmp_path, "cityscapes_fog_abra_alpha_exp1_train", per_class, {"AP": 31.0, "AP50": 1.0, "AP75": 1.0,
         "alpha": 1.4, "alpha_curve": curve}, residual=tmp_path / "missing.pth")   # would fail if hashed
    revision_collect.main(["alpha-map", str(tmp_path / "eval"), "cityscapes", "fog"])
    assert capsys.readouterr().out.strip() == "train=1.4"


def test_dataset_stats_rows(tmp_path):
    for name, root, avail in (("cityscapes", tmp_path / "cs", "car"), ("sdgod", tmp_path / "sd", "car")):
        for domain in revision_stats.BENCHMARKS[name]["domains"]:
            ann = root / domain / "annotations"
            ann.mkdir(parents=True)
            lines = [{"filename": "a", "detection": {"instances": [{"category": avail, "bbox": [0, 0, 1, 1]},
                                                                    {"category": avail, "bbox": [0, 0, 2, 2]},
                                                                    {"category": "bus", "bbox": [0, 0, 1, 1]}]}},
                     {"filename": "b", "detection": {"instances": [{"category": "bus", "bbox": [0, 0, 1, 1]}]}}]
            (ann / "annotations_train.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
            _coco(ann / revision_stats.BENCHMARKS[name]["test"], ["a"], [(0, "bus")])
    out = tmp_path / "stats.csv"
    revision_stats.main(["--cityscapes", str(tmp_path / "cs"), "--sdgod", str(tmp_path / "sd"), "--out", str(out)])
    rows = list(csv.DictReader(open(out)))
    fog = {r["stat"] or (r["split"], r["class"]): r for r in rows if r["domain"] == "cityscapes/fog"}
    assert (fog["train", "bus"]["n_instances"], fog["train", "bus"]["n_images"]) == ("2", "2")
    assert fog["domain_expert_images"]["value"] == "1" and fog["domain_expert_images_with_unavailable"]["value"] == "1"
    assert float(fog["domain_expert_available_box_share"]["value"]) == pytest.approx(200 / 3)
    assert float(fog["top1_coverage"]["value"]) == pytest.approx(50.0)


def test_projection_is_identity_on_square_layers():
    torch.manual_seed(0)
    u_t, _, vh_t = torch.linalg.svd(torch.randn(6, 6, dtype=torch.float64))
    u_s, _, vh_s = torch.linalg.svd(torch.randn(6, 6, dtype=torch.float64))
    rel, l_err, r_err = projection_stats.layer_stats(torch.randn(6, 6), u_t, vh_t, u_s, vh_s)
    assert rel < 1e-12 and l_err < 1e-12 and r_err < 1e-12
    u_t, _, vh_t = torch.linalg.svd(torch.randn(4, 8, dtype=torch.float64), full_matrices=False)
    u_s, _, vh_s = torch.linalg.svd(torch.randn(4, 8, dtype=torch.float64), full_matrices=False)
    rel, l_err, r_err = projection_stats.layer_stats(torch.randn(4, 8), u_t, vh_t, u_s, vh_s)
    assert rel > 0.1 and l_err < 1e-12 and r_err > 0.1   # row space of a 4 x 8 layer: only R is not orthogonal
