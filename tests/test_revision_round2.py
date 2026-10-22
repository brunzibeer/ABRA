"""Revision round 2 tools (collect round2, zsobj_rule, dataset_totals, recipes_round2, plot_fig4/5) on synthetic data."""
import csv
import io
import json
import tarfile

import pytest

from tools import dataset_totals, plot_fig4, plot_fig5, recipes_round2, revision_collect, zsobj_rule

CLASSES = ["bus", "motorcycle", "rider", "train", "truck"]


def _coco(path, n_images=2, cls="bus"):
    data = {"images": [{"id": i, "file_name": f"{i}.png", "width": 10, "height": 10} for i in range(n_images)],
            "categories": [{"id": 1, "name": cls}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _run(root, name, test, result, gpu="Quadro RTX 5000", commit=None, residual=None):
    d = root / "eval" / name
    d.mkdir(parents=True)
    config = {"started": "2026-10-22T10:00:00", "git_commit": commit, "environment": {"gpu": gpu, "host": "node1"},
              "args": {"exp_name": name, "test_annotations": str(test), "eval_classes": None, "domain": None,
                       "residual": str(residual) if residual else None}}
    (d / "config.json").write_text(json.dumps(config))
    (d / "results.json").write_text(json.dumps(result))


def _res(ap):
    return {"AP": ap, "AP50": ap + 20, "AP75": ap - 1}


def test_round2_collect_fills_commit_and_provenance(tmp_path):
    test = _coco(tmp_path / "experts/bus/annotations_val.json")
    runs, reb, r1, out = tmp_path / "revision2", tmp_path / "refactor", tmp_path / "out", tmp_path / "out_round2"
    _run(runs, "cityscapes_fog_abra_zsobj_exp3_bus", test, _res(35.12345))
    _run(runs, "cityscapes_fog_abra_merge_exp3_bus", test, _res(30.0), gpu="NVIDIA L40S", commit="abc1234")
    _run(reb, "cityscapes_fog_abra_mask_plain_exp1_bus", test, _res(17.0), commit=None)
    _run(reb, "cityscapes_fog_target_exp1_bus", test, _res(8.0), gpu="NVIDIA RTX A5000", commit="52cbd15")
    _run(reb, "cityscapes_fog_abra_k1_exptop1_train", test, _res(30.0))
    r1.mkdir()
    revision_collect.write_csv(r1 / "masks.csv", revision_collect.MASK_COLUMNS,
                               revision_collect.collect_masks(reb / "eval"))
    revision_collect.main(["round2", "--runs", str(runs), "--rebuttal", str(reb), "--round1", str(r1), "--out", str(out),
                           "--code_commit", "fff0000"])
    rows = {r["method"]: r for r in csv.DictReader(open(out / "results_round2.csv"))}
    assert (rows["fig4_zs_obj"]["exp_id"], rows["fig4_zs_obj"]["code_commit"], rows["fig4_zs_obj"]["mAP"]) == \
        ("T1", "fff0000", "35.12345")
    assert (rows["merge"]["exp_id"], rows["merge"]["gpu"], rows["merge"]["code_commit"]) == ("T2", "NVIDIA L40S", "abc1234")
    mask = next(csv.DictReader(open(out / "masks_round2.csv")))
    assert (mask["mask_id"], mask["gpu"], mask["code_commit"], mask["run_date"]) == \
        ("plain", "Quadro RTX 5000", "unknown", "2026-10-22")
    other = list(csv.DictReader(open(out / "provenance_other.csv")))
    assert [(r["table"], r["setting"], r["gpu"]) for r in other] == \
        [("Table 3", "theta_T alone", "NVIDIA RTX A5000"), ("k-ablation", "k = 1", "Quadro RTX 5000")]
    assert "| Table 3 | theta_T alone | 1 | NVIDIA RTX A5000: 1 | yes | - |" in (out / "provenance_summary.md").read_text()


def test_round2_collect_refuses_mask_rows_that_differ_from_round1(tmp_path):
    test = _coco(tmp_path / "t.json")
    reb, r1 = tmp_path / "refactor", tmp_path / "out"
    (tmp_path / "revision2" / "eval").mkdir(parents=True)
    _run(reb, "cityscapes_fog_abra_mask_plain_exp1_bus", test, _res(17.0))
    r1.mkdir()
    (r1 / "masks.csv").write_text("benchmark,target_domain,class,class_set,mask_id,mAP,AP50,AP75,origin\n"
                                  "cityscapes,fog,bus,unav,plain,17.5,1,1,log\n")
    with pytest.raises(SystemExit, match="differ"):
        revision_collect.main(["round2", "--runs", str(tmp_path / "revision2"), "--rebuttal", str(reb),
                               "--round1", str(r1), "--out", str(tmp_path / "o")])


def _jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"filename": f, "height": 10, "width": 10, "detection": {"instances": inst}})
                              for f, inst in records) + "\n")


def test_zsobj_rule_reports_added_boxes(tmp_path):
    gt = {"bbox": [0, 0, 10, 10], "label": 1, "category": "car"}
    renamed = {"bbox": [0, 0, 10, 10], "label": 3, "category": "truck"}   # the same box under another name
    zs = {"bbox": [20, 20, 30, 30], "label": 2, "category": "bus"}
    near = {"bbox": [0, 0, 10, 9], "label": 2, "category": "bus"}
    for d in ("normal", "fog"):
        _jsonl(tmp_path / d / "base.jsonl", [("a.png", [gt]), ("b.png", [gt])])
        _jsonl(tmp_path / d / "zs.jsonl", [("a.png", [renamed, zs]), ("b.png", [near]), ("c.png", [zs])])
    zsobj_rule.main(["--cityscapes", str(tmp_path), "--zs", "zs.jsonl", "--base", "base.jsonl", "--domains", "fog", "normal",
                     "--out", str(tmp_path / "r.json")])
    rep = json.loads((tmp_path / "r.json").read_text())["fog"]
    assert (rep["images_only_in_zs"], rep["base_boxes_kept (same coordinates)"], rep["base_boxes_dropped"],
            rep["added_boxes"]) == (1, 1, 1, 3)
    assert rep["base_category -> zs_category of kept boxes"] == {"car -> truck": 1}
    assert rep["added_categories"] == {"bus": 3}
    assert rep["added_max_iou_with_base_by_band"] == {"0.0": 2, "0.9": 1}
    zsobj_rule.main(["--cityscapes", str(tmp_path), "--zs", "missing.jsonl", "--base", "base.jsonl", "--domains", "fog",
                     "--out", str(tmp_path / "m.json")])
    assert "missing" in json.loads((tmp_path / "m.json").read_text())["fog"]


def test_dataset_totals_rows_and_paper_check(tmp_path):
    for bench, spec in dataset_totals.BENCHMARKS.items():
        for d in spec["domains"]:
            ann = tmp_path / bench / d / "annotations"
            _jsonl(ann / "annotations_train.jsonl", [("a.png", [{"bbox": [0, 0, 1, 1], "category": "car"}]), ("b.png", [])])
            _coco(ann / spec["test"], n_images=3)
    out = tmp_path / "totals.csv"
    dataset_totals.main(["--cityscapes", str(tmp_path / "cityscapes"), "--sdgod", str(tmp_path / "sdgod"), "--out", str(out)])
    rows = {(r["domain"], r["split"], r["stat"]): r["value"] for r in csv.DictReader(open(out))}
    assert rows["cityscapes/fog", "train", "total_images"] == "2"
    assert rows["cityscapes/fog", "train", "images_with_annotations"] == "1"
    assert rows["sdgod/night_sunny", "test", "total_instances"] == "1"
    lines = dataset_totals.check([{"domain": "cityscapes/fog", "split": "test", "stat": "total_images", "value": 500}])
    assert any("cityscapes/fog" in x and "test" in x and x.endswith("ok") for x in lines)
    assert any("MISMATCH" in x for x in lines)


def _results(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "target_domain", "method", "class", "protocol", "mAP", "AP50"])
        w.writeheader()
        w.writerows({"benchmark": "cityscapes", "target_domain": "fog", "method": m, "class": c, "protocol": "per_class",
                     "mAP": v, "AP50": v + 20} for m, v in rows for c in CLASSES)
    return path


def test_plots_size_fonts_and_determinism(tmp_path):
    r1 = _results(tmp_path / "r1.csv", [("fig4_supervised", 37.6), ("abra", 40.5)])
    r2 = _results(tmp_path / "r2.csv", [("fig4_zs_obj", 35.0), ("merge", 30.0)])
    pdfs = []
    for i in range(2):
        pdf = tmp_path / f"f4_{i}.pdf"
        plot_fig4.main(["--round1", str(r1), "--round2", str(r2), "--out", str(pdf)])
        pdfs.append(pdf.read_bytes())
    assert pdfs[0] == pdfs[1]
    assert b"/MediaBox [ 0 0 351.3 170.3 ]" in pdfs[0] and b"/CIDFontType2" in pdfs[0]
    plot_fig4.main(["--round1", str(r1), "--two_bar", "--out", str(tmp_path / "two.pdf")])
    with pytest.raises(SystemExit):
        plot_fig4.main(["--round1", str(r1), "--out", str(tmp_path / "x.pdf")])   # the first bar needs --round2
    from PIL import Image
    png = tmp_path / "f5.png"
    plot_fig5.main(["--round1", str(r1), "--round2", str(r2), "--out", str(png), "--pdf", str(tmp_path / "f5.pdf")])
    assert Image.open(png).size == (1600, 600)


def test_recipes_wandb_first_then_launch_script(tmp_path):
    run = {"name": "CS_FOG_FT4_bus_1234abcd", "id": "w1", "state": "finished", "created": "2026-03-02T00:00:00Z",
           "summary": {"EVAL/AP": 40.0}, "config": {"lr": 0.01, "epochs": 12, "finetuning": "transported/bus/merged.pth"}}
    fda = {"name": "UDA_FDA_ABRA_bus_00000000", "id": "w2", "state": "finished", "created": "2026-03-02T00:00:00Z",
           "summary": {"EVAL/AP": 41.0}, "config": {"lr": 1e-6, "lp_lr": 1e-7, "fda_lb": 0.01}}
    (tmp_path / "runs.jsonl").write_text(json.dumps(run) + "\n" + json.dumps(fda) + "\n")
    (tmp_path / "hist.jsonl").write_text(json.dumps({"name": fda["name"], "history": [{"EVAL/AP": 40.0}, {"EVAL/AP": 41.5},
                                                                                      {"EVAL/AP": 41.0}]}) + "\n")
    script = ('OUT=/x\nargs=(\n    "--exp_name CS_FOG_FT4_bus --lr 0.02 --output_root $OUT/ft4_bus\n'
              '    --lp_names ref_point_head sampling_offsets --step_milestones 7 9"\n)\n')
    with tarfile.open(tmp_path / "sb.tar.gz", "w:gz") as tar:
        data = script.encode()
        info = tarfile.TarInfo("./sb_ft.sh")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    recipes_round2.main(["--wandb_runs", str(tmp_path / "runs.jsonl"), "--wandb_hist", str(tmp_path / "hist.jsonl"),
                         "--sbatch", str(tmp_path / "sb.tar.gz"), "--out", str(tmp_path / "r.yaml"),
                         "--grid", str(tmp_path / "g.csv")])
    text = (tmp_path / "r.yaml").read_text()
    ft4 = text[text.index('"svft_from_abra"'):text.index('"fda_from_abra"')]
    assert '"value": 0.01\n    "source": "wandb w1"\n    "launch_script_disagrees":' in ft4   # wandb wins
    assert '"value": ["ref_point_head", "sampling_offsets"]\n    "source": "sbatch sb_ft.sh (archived July 2026)"' in ft4
    assert '"seed":\n    "value": "not recorded"' in ft4
    grid = next(csv.DictReader(open(tmp_path / "g.csv")))
    assert (grid["run"], grid["mAP_best"], grid["mAP_last"], grid["AP50"]) == (fda["name"], "41.5", "41.0", "not recorded")
