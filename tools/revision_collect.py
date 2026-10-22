"""Collect the revision experiments (spec of 21 Oct 2026, E1-E10) into the files the manuscript is filled from.

    python tools/revision_collect.py provenance <run roots...>          # commit / GPU / torch of existing runs
    python tools/revision_collect.py alpha-map <eval root> <benchmark> <target>   # ALPHA_MAP of the E1/E6 searches
    python tools/revision_collect.py collect --runs <revision RUNS> --rebuttal <rebuttal RUNS> --out <dir>

``collect`` writes results.csv, alpha_search.csv, masks.csv, sanity_report.md and metadata.yaml (dataset_stats.csv
and projection_stats.csv are written by tools/revision_stats.py and tools/projection_stats.py). Every float is
written unrounded (0-100 scale); no averages are written.

Run names are ``<benchmark>_<target>_<label>_<class>`` (slurm/eval.sh); LABELS maps a label to the method of the
spec. ABRA-alpha: alpha selected on mAP among the evaluated alphas, ties -> the alpha closest to 1.0; AP50 and AP75
at that alpha, from the same pass.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

SD_DOMAINS = ["daytime_sunny", "daytime_foggy", "dusk_rainy", "night_rainy", "night_sunny"]
LABELS = {   # label (without _std) -> spec method id
    "zero_shot_exp1": "zero_shot", "source_exp9": "source", "ft_srcft2": "source", "ft_ft2": "ref_target_ft",
    "ft_fda": "fda", "ft_fdabest": "fda_best", "param_delta_exp1": "param_delta", "abra_exp1": "abra",
    "abra_alpha_exp1": "abra_alpha", "abra_oracle_exp1": "abra_alpha", "ft_ft4": "abra_init_svft",
    "ft_fda_abra": "abra_init_fda", "ft_fda_abrabest": "abra_init_fda_best", "abra_fig4sup_exp2": "fig4_supervised",
    # Fig. 4 "Zero shot w/ Obj.": candidate readings (slurm/revision.sh fig4)
    "zero_shot_obj_exp1": "fig4_theta0_object_prompt", "target_obj_exp1": "fig4_thetaT_object_prompt",
    "zero_shot_multi_exp1": "fig4_theta0_multi",
}
RESULT_COLUMNS = ["exp_id", "benchmark", "target_domain", "method", "class", "protocol", "alpha", "n_images", "n_gt",
                  "mAP", "AP50", "AP75", "checkpoint_id"]
ALPHA_COLUMNS = ["benchmark", "target_domain", "class", "alpha", "mAP", "AP50", "AP75", "selected"]
MASK_COLUMNS = ["benchmark", "target_domain", "class", "class_set", "mask_id", "mAP", "AP50", "AP75", "origin"]
AVAILABLE = {"cityscapes": {"car", "person", "bicycle"}, "sdgod": {"car", "person", "truck"}}
ABRAOUT = "/work/horizon_store/abraout/"
# E0 (sanity gate): mAP per class, per-class protocol, from the rebuttal (Tables R1, R4) and the Day-Foggy masks run
EXPECTED = {
    ("cityscapes", "fog", "zero_shot"): {"bus": 48.62, "motorcycle": 23.13, "rider": 18.97, "train": 16.31, "truck": 31.20},
    ("cityscapes", "fog", "source"): {"bus": 54.12, "motorcycle": 28.95, "rider": 42.49, "train": 33.40, "truck": 36.78},
    ("cityscapes", "fog", "abra"): {"bus": 57.22, "motorcycle": 30.01, "rider": 42.29, "train": 34.97, "truck": 38.12},
    ("sdgod", "daytime_foggy", "abra"): {"bike": 23.37, "bus": 46.75, "motorcycle": 25.74, "rider": 33.51},
}


def parse_run(name):
    """``<benchmark>_<target>_<label>_<class>`` -> (benchmark, target, label, class)."""
    bench, rest = name.split("_", 1)
    targets = SD_DOMAINS if bench == "sdgod" else ["normal", "fog"]
    target = next(t for t in targets if rest.startswith(t + "_"))
    label, cls = rest[len(target) + 1:].rsplit("_", 1)
    return bench, target, label, cls


def runs(root):
    """Every evaluation under ``root``: (run name, results.json, config.json)."""
    for path in sorted(Path(root).rglob("results.json")):
        config = json.loads((path.parent / "config.json").read_text())
        yield config["args"]["exp_name"], json.loads(path.read_text()), config


def select_alpha(curve):
    """Highest mAP; on a tie the alpha closest to 1.0."""
    return max(curve, key=lambda r: (r["AP"], -abs(r["alpha"] - 1.0)))


@lru_cache(maxsize=None)
def counts(test_json, cls):
    """(images, instances of ``cls``) of a COCO json: the images and ground truth an evaluation uses."""
    data = json.loads(Path(test_json).read_text())
    ids = {c["id"] for c in data["categories"] if c["name"] == cls}
    if not ids and len(data["categories"]) == 1:   # per-class file with its category renamed ("object")
        ids = {data["categories"][0]["id"]}
    return len(data["images"]), sum(1 for a in data["annotations"] if a["category_id"] in ids)


@lru_cache(maxsize=None)
def sha12(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def checkpoint_id(args):
    parts = []
    if args.get("domain"):
        parts.append("domain=" + args["domain"].replace(ABRAOUT, ""))
    if args.get("residual"):
        parts.append(f"residual={args['residual'].replace(ABRAOUT, '')}#{sha12(args['residual'])}")
    return ";".join(parts) or "theta_0"


def exp_id(bench, method, protocol):
    if bench == "sdgod":
        return {"fda": "E4", "fda_best": "E4", "param_delta": "E5"}.get(method, "E6")
    if protocol == "standard":
        return "E3"
    if method.startswith("fig4"):
        return "E10"
    return {"abra_init_svft": "E2", "abra_init_fda": "E2", "abra_init_fda_best": "E2"}.get(method, "E1")


def collect_results(root, ids=True):
    """Rows of results.csv and alpha_search.csv; ``ids=False`` skips checkpoint_id (it hashes every checkpoint, too
    slow for the login node, where alpha-map runs)."""
    results, alphas = [], []
    for name, res, config in runs(root):
        bench, target, label, cls = parse_run(name)
        args = config["args"]
        protocol = "standard" if args.get("eval_classes") else "multi" if "_multi_" in label else "per_class"
        method = LABELS.get(label.replace("_std_", "_"))
        if method is None:
            print(f"skipped (no method for label {label!r}): {name}", file=sys.stderr)
            continue
        metrics, alpha = {k: res[k] for k in ("AP", "AP50", "AP75")}, None
        if method.startswith("abra") and not method.startswith("abra_init"):
            alpha = res.get("alpha")
        if res.get("alpha_curve"):
            best = select_alpha(res["alpha_curve"])
            metrics, alpha = {k: best[k] for k in ("AP", "AP50", "AP75")}, best["alpha"]
            alphas += [{"benchmark": bench, "target_domain": target, "class": cls, "alpha": r["alpha"], "mAP": r["AP"],
                        "AP50": r["AP50"], "AP75": r["AP75"], "selected": r is best} for r in res["alpha_curve"]]
        n_images, n_gt = counts(args["test_annotations"], cls)
        results.append({"exp_id": exp_id(bench, method, protocol), "benchmark": bench, "target_domain": target,
                        "method": method, "class": cls, "protocol": protocol, "alpha": alpha, "n_images": n_images,
                        "n_gt": n_gt, "mAP": metrics["AP"], "AP50": metrics["AP50"], "AP75": metrics["AP75"],
                        "checkpoint_id": checkpoint_id(args) if ids else None, "_config": config})
    return results, alphas


def collect_masks(root):
    rows = []
    for name, res, config in runs(root):
        bench, target, label, cls = parse_run(name)
        if label.startswith("abra_mask_"):
            mask = label[len("abra_mask_"):].rsplit("_", 1)[0].replace("-", ":")
        elif label in ("abra_exp1", "task_add_exp1"):
            mask = "none (ABRA)" if label == "abra_exp1" else "direct addition"
        else:
            continue
        if config["args"].get("eval_classes") or bench == "sdgod" and target != "daytime_foggy":
            continue
        rows.append({"benchmark": bench, "target_domain": target, "class": cls,
                     "class_set": "avail" if cls in AVAILABLE[bench] else "unav", "mask_id": mask, "mAP": res["AP"],
                     "AP50": res["AP50"], "AP75": res["AP75"], "origin": "log"})
    return rows


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {path}")


def provenance(roots):
    seen = Counter()
    for root in roots:
        for _, _, config in runs(root):
            env = config.get("environment", {})
            seen[(config.get("git_commit"), env.get("gpu"), env.get("torch"), env.get("cuda"))] += 1
    return seen


def sanity_report(results, out):
    lines = ["# E0 sanity report", "", "Expected = rebuttal values (mAP, per-class protocol); tolerance 0.01.", "",
             "| benchmark | target | method | class | expected | obtained | diff | ok |", "|---|---|---|---|---|---|---|---|"]
    misses = 0
    for (bench, target, method), exp in EXPECTED.items():
        for cls, value in exp.items():
            got = [r["mAP"] for r in results if (r["benchmark"], r["target_domain"], r["method"], r["class"],
                                                 r["protocol"]) == (bench, target, method, cls, "per_class")]
            if not got:
                lines.append(f"| {bench} | {target} | {method} | {cls} | {value:.2f} | missing | | NO |")
                misses += 1
                continue
            ok = abs(got[0] - value) <= 0.01 + 1e-9   # expected values are rounded to 2 decimals
            misses += not ok
            lines.append(f"| {bench} | {target} | {method} | {cls} | {value:.2f} | {got[0]:.4f} | "
                         f"{got[0] - value:+.4f} | {'yes' if ok else 'NO'} |")
    env = Counter((r["_config"].get("git_commit"), r["_config"]["environment"].get("gpu")) for r in results)
    lines += ["", f"**{'PASSED' if not misses else f'FAILED: {misses} value(s)'}**", "",
              "Commits / GPUs of the collected evaluations: " + "; ".join(f"{c} on {g}: {n}" for (c, g), n in env.items())]
    (out / "sanity_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def yaml_value(v):
    return json.dumps(v)   # JSON scalars and lists are valid YAML


def metadata(results, alphas, out, extra):
    from importlib.metadata import version

    import numpy
    import torch
    commits = sorted({r["_config"].get("git_commit") for r in results} - {None})
    envs = sorted({json.dumps(r["_config"]["environment"], sort_keys=True) for r in results})
    families = {}
    for r in results:
        families.setdefault(f"{r['benchmark']}/{r['target_domain']}/{r['method']}", set()).add(r["checkpoint_id"])
    selected = {}
    for a in alphas:
        if a["selected"]:
            selected.setdefault(f"{a['benchmark']}/{a['target_domain']}", {})[a["class"]] = a["alpha"]
    pycoco_version = version("pycocotools")
    lines = [
        "# metadata of the revision experiments (tools/revision_collect.py)",
        f"code_commits: {yaml_value(commits)}",
        f"rebuttal_code_commits: {yaml_value(extra['rebuttal_commits'])}",
        f"collected_with: {yaml_value({'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': numpy.__version__, 'pycocotools': pycoco_version})}",
        "run_environments:", *[f"  - {e}" for e in envs],
        "svd:",
        "  abra: 'torch.linalg.svd(theta_T layer, full_matrices=False), float64, CPU, canonical signs (largest |entry| of each left vector > 0); abra/weights/svd.py'",
        "  param_delta_legacy: 'torch.linalg.svd(theta_T layer, full_matrices=False), float32, on the evaluation device (cuda); abra/transport/methods.py param_delta'",
        "prompt_template: '\" . \".join(category names of the test json) + \" .\"; per-class protocol: \"<class> .\"; standard protocol: --eval_classes <class>'",
        "evaluation: 'eval mode, seed 0, batch size 1 (sequential), Grounding DINO val transforms (no fix_size)'",
        "alpha_grid: '[0.1, 10) step 0.1, early stop after 10 decreases of mAP (paper and rebuttal grid)'",
        "alpha_rule: 'max mAP over the evaluated alphas; tie -> alpha closest to 1.0; AP50/AP75 at that alpha, same pass'",
        f"alpha_selected: {yaml_value(selected)}",
        "checkpoint_rules:",
        "  class_experts: 'best epoch by mAP on the source test split of the class (source, exp1/exp9 residuals)'",
        "  domain_experts: 'best epoch by mAP over ALL categories of the domain test split (annotations_val/test.json, caption = every category): includes the unavailable classes'",
        "  ref_target_ft: 'best epoch by mAP on the target test split of the class (reference, uses target labels)'",
        "  abra_init_svft: 'best epoch by mAP on the target test split of the class (reference, uses target labels)'",
        "  fda: 'last epoch (12 epochs, no selection); fda_best = best epoch on the target test split, reported for comparison only'",
        "  abra_init_fda: 'last epoch (12 epochs); abra_init_fda_best = best epoch on the target test split'",
        "checkpoint_ids:", *[f"  {k}: {yaml_value(sorted(v))}" for k, v in sorted(families.items())],
        "fda_recipe: 'abra fda (slurm/fda.sh): SVFT band half-width 2 on encoder_attn, init theta_0 (fda) or the original ABRA-transported model (fda_abra), FDA L = 0.01, no entropy loss, 12 epochs, batch 4, AdamW, lr 1e-2 / lp_lr 1e-3 (fda_abra: 1e-6 / 1e-7) on ref_point_head + sampling_offsets, weight decay 0, no scheduler, seed 0; source = labelled source train images of the class (other boxes removed); style pool = target train images containing the class (labels unused); images resized to 1024 x 512; an epoch = min(#source, #style) batches (source and style loaders zipped)'",
        "ref_target_ft_recipe: 'original runs (fts/<benchmark>/<domain>/ft2): SVFT band half-width 2 on theta_0, target train images of the class, 12 epochs, batch 4, AdamW lr 1e-2 / lp_lr 1e-3, weight decay 0, multi-step 7 9'",
        "masks:",
        "  definition: 'R = U_T (W * C) V_T^T with C = U_T^T tau V_T (k x k, k = min(out, in)); thresholds per layer; W = 1 is ABRA (abra/transport/masks.py)'",
        "  plain: 'diagonal only (= band:0)'",
        "  banded: 'the SVFT band of the residual (|i - j| <= 2)'",
        "  band:W: '1 on |i - j| <= W, 0 outside'",
        "  band:W:G: '1 on |i - j| <= W, G outside'",
        "  block:R: '1 on the leading R x R block, 0 outside; no-op where k <= R (block:128 on the 128 x 256 layers)'",
        "  gauss:S: 'exp(-(i - j)^2 / (2 S^2))'",
        "  exp:S: 'exp(-|i - j| / S)'",
        "  score: 'SCORE trim: keep the diagonal and the off-diagonal entries with |c - mean_off| < 1.96 std_off (per layer)'",
        "  topk:P: 'keep the fraction P of the largest |c| per layer'",
    ]
    (out / "metadata.yaml").write_text("\n".join(lines) + "\n")
    print(f"metadata -> {out / 'metadata.yaml'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("provenance")
    s.add_argument("roots", nargs="+", type=Path)
    s = sub.add_parser("alpha-map")
    s.add_argument("root", type=Path)
    s.add_argument("benchmark")
    s.add_argument("target")
    s = sub.add_parser("collect")
    s.add_argument("--runs", type=Path, required=True, help="RUNS of the revision (eval/ below it)")
    s.add_argument("--rebuttal", type=Path, required=True, help="RUNS of the rebuttal (mask runs, E7)")
    s.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    if args.command == "provenance":
        for (commit, gpu, torch_v, cuda), n in provenance(args.roots).most_common():
            print(f"{n:5d}  commit {commit}  gpu {gpu}  torch {torch_v}  cuda {cuda}")
        return
    if args.command == "alpha-map":
        results, _ = collect_results(args.root, ids=False)
        pick = {r["class"]: r["alpha"] for r in results if r["method"] == "abra_alpha" and r["protocol"] == "per_class"
                and (r["benchmark"], r["target_domain"]) == (args.benchmark, args.target)}
        print(",".join(f"{c}={a:g}" for c, a in sorted(pick.items())))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    results, alphas = collect_results(args.runs / "eval")
    write_csv(args.out / "results.csv", RESULT_COLUMNS, results)
    write_csv(args.out / "alpha_search.csv", ALPHA_COLUMNS, alphas)
    write_csv(args.out / "masks.csv", MASK_COLUMNS, collect_masks(args.rebuttal / "eval"))
    sanity_report(results, args.out)
    rebuttal_commits = sorted({c for c, *_ in provenance([args.rebuttal / "eval"])} - {None})
    metadata(results, alphas, args.out, {"rebuttal_commits": rebuttal_commits})


if __name__ == "__main__":
    main()
