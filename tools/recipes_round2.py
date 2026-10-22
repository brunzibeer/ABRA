"""Training recipes of the model families the revision must state (round 2, T6, plus the T1/T2 models), CPU only.

    python tools/recipes_round2.py --wandb_runs wandb_runs.jsonl --wandb_hist wandb_hist.jsonl \
        --sbatch sbatch_files.tar.gz --out recipes_round2.yaml --grid fda_abra_lr_grid.csv

Sources, recorded per field: the wandb run config (``wandb <run id>``; the dump keeps scalar values only) and, for
what wandb lacks, the original launch script with the same ``--exp_name`` (``sbatch <file>``; the archived scripts
date from July 2026, after the runs, so a disagreement with wandb is reported and wandb is kept). Fields in neither
are ``not recorded``; nothing is filled from defaults. The checkpoint rule is the training code's (``code``): the
best epoch by EVAL/AP on ``test_annotations`` (epoch 0 excluded), plus ``last``.
"""
import argparse
import csv
import json
import re
import shlex
import tarfile
from collections import defaultdict
from pathlib import Path

FIELDS = {   # recipe field -> (wandb config key, launch-script option)
    "initialization": ("finetuning", "finetuning"), "train_annotations": ("train_annotations", "train_annotations"),
    "test_annotations (selection)": ("test_annotations", "test_annotations"), "label_map": ("label_map", "label_map"),
    "objectify": ("objectify", "objectify"), "trained_layers": ("trainable", "trainable"), "svft": ("svft", "svft"),
    "svft_pattern": ("svft_pattern", "svft_pattern"), "off_diag": ("svft_diag", "svft_diag"),
    "epochs": ("epochs", "epochs"), "batch_size": ("batch_size", "batch_size"), "optimizer": ("optim", "optim"),
    "lr": ("lr", "lr"), "lp_names": (None, "lp_names"), "lp_lr (lp_names)": ("lp_lr", "lp_lr"),
    "weight_decay": ("weight_decay", "weight_decay"), "scheduler": ("scheduler", "scheduler"),
    "step_milestones": (None, "step_milestones"), "seed": ("seed", "seed"),
    "fda_beta (fda_lb)": ("fda_lb", None), "uda": ("uda", None),
    "source_train_annotations (FDA)": ("source_train_annotations", "source_train_annotations"),
    "target_train_annotations (FDA, style pool)": ("target_train_annotations", "target_train_annotations"),
}
FDA_ONLY = {"fda_beta (fda_lb)", "uda", "source_train_annotations (FDA)", "target_train_annotations (FDA, style pool)"}
CHECKPOINT_RULE = "best epoch by EVAL/AP on test_annotations (evaluated every epoch, epoch 0 excluded), and last"
C5 = "(bus|motorcycle|rider|train|truck)"
FAMILIES = [   # (key, description, wandb run name, launch-script exp_name): regular expressions
    ("svft_from_abra", "Table 2 SVFT from ABRA (fts/cityscapes/fog/ft4_<class>)", rf"CS_FOG_FT4_{C5}_[0-9a-f]{{8}}$",
     rf"CS_FOG_FT4_{C5}$"),
    ("fda_from_abra", "Table 2 FDA from ABRA (uda/abra/<class>)", rf"UDA_FDA_ABRA_{C5}_[0-9a-f]{{8}}$", rf"UDA_FDA_ABRA_{C5}$"),
    ("fda_from_theta0", "Table 1/2 FDA from theta_0 (uda/<class>)", rf"UDA_FDA_{C5}_[0-9a-f]{{8}}$", rf"UDA_FDA_{C5}$"),
    ("few_shot_theta0", "Fig. 6 few-shot from theta_0", rf"FEWIMG_FOG_\d+_{C5}_[0-9a-f]{{8}}$", rf"FEWIMG_FOG_\d+_{C5}$"),
    ("few_shot_abra", "Fig. 6 few-shot from ABRA", rf"FEWIMG_ABRA_FOG_\d+_{C5}_[0-9a-f]{{8}}$", rf"FEWIMG_ABRA_FOG_\d+_{C5}$"),
    ("supervised_domain_expert", "Fig. 4 Supervised: theta_T (domains/cityscapes/fog/dom2)", r"CS_FOG_DOM2_[0-9a-f]{8}$",
     r"CS_FOG_DOM2$"),
    ("supervised_source_domain_expert", "Fig. 4 Supervised: theta_S (domains/cityscapes/normal/dom2)",
     r"CS_NORMAL_DOM2_[0-9a-f]{8}$", r"CS_NORMAL_DOM2$"),
    ("supervised_class_experts", "Fig. 4 Supervised: class experts (experts/cityscapes/normal/exp2_<class>)",
     rf"CS_NORMAL_EXP2_{C5}_[0-9a-f]{{8}}$", rf"CS_NORMAL_EXP2_{C5}$"),
    ("zs_obj_target_domain_expert", "Fig. 4 Zero shot w/ Obj.: theta_T (domains/cityscapes/fog/dom3)", r"CS_FOG_ZS_b28fc8f2$",
     r"CS_FOG_ZS$"),
    ("zs_obj_source_domain_expert", "Fig. 4 Zero shot w/ Obj.: theta_S (domains/cityscapes/normal/dom3)",
     r"CS_NORMAL_ZS_680f6cbc$", r"CS_NORMAL_ZS$"),
    ("zs_obj_class_experts", "Fig. 4 Zero shot w/ Obj.: class experts (experts/cityscapes/normal/exp3_<class>)",
     rf"CS_NORMAL_EXP3_{C5}_[0-9a-f]{{8}}$", rf"CS_NORMAL_EXP3_{C5}$"),
    ("merge_expert", "Fig. 5 Merge (experts/cityscapes/normal/exp3)", r"CS_NORMAL_EXP3_e76c3739$", r"CS_NORMAL_EXP3$"),
]


def launch_scripts(tar_path):
    """{exp_name: (script, {option: value})} from the archived sbatch files (variables substituted)."""
    out = {}
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".sh") or ".venv" in m.name:
                continue
            text = tar.extractfile(m).read().decode()
            env = dict(re.findall(r"^([A-Z_]+)=(\S+)\s*$", text, re.M))
            block = re.search(r"^args=\((.*?)^\)", text, re.S | re.M)
            for chunk in re.findall(r'"([^"]*)"', block.group(1) if block else ""):
                chunk = re.sub(r"\$\{?([A-Z_]+)\}?", lambda v: env.get(v.group(1), v.group(0)), chunk)
                opts, key = {}, None
                for tok in shlex.split(chunk):
                    if tok.startswith("--"):
                        key = tok[2:]
                        opts[key] = []
                    elif key:
                        opts[key].append(tok)
                opts = {k: (v[0] if len(v) == 1 else v) for k, v in opts.items()}
                if "exp_name" in opts:
                    out[opts["exp_name"]] = (Path(m.name).name, opts)
    return out


def number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def field(run, script, wkey, skey):
    """{"value", "source"} of one field of one run: wandb first, the launch script for what wandb lacks."""
    cfg = run["config"] if run else {}
    opts = script[1] if script else {}
    if wkey and wkey in cfg:
        rec = {"value": cfg[wkey], "source": f"wandb {run['id']}"}
        if skey in opts and number(opts[skey]) != number(cfg[wkey]):
            rec["launch_script_disagrees"] = f"{opts[skey]} ({script[0]})"
        return rec
    if skey and skey in opts:
        return {"value": opts[skey], "source": f"sbatch {script[0]} (archived July 2026)"}
    if script and skey:
        return {"value": "not recorded", "source": f"option not passed in sbatch {script[0]}"}
    return {"value": "not recorded", "source": None}


def strip(name):
    return re.sub(r"_[0-9a-f]{8}$", "", name)


def family(runs, scripts, wandb_re, script_re, fda=False):
    members = sorted((r for r in runs if re.match(wandb_re, r["name"])), key=lambda r: r["name"])
    entries = [(r["name"], r, scripts.get(strip(r["name"]))) for r in members]
    seen = {strip(r["name"]) for r in members}
    entries += [(n, None, scripts[n]) for n in sorted(scripts) if re.match(script_re, n) and n not in seen]
    rec = {"runs": [{"name": n, "wandb_id": r["id"] if r else None, "state": r["state"] if r else "not in wandb",
                     "created": r["created"][:10] if r else None, "launch_script": s[0] if s else None}
                    for n, r, s in entries]}
    for name, (wkey, skey) in FIELDS.items():
        if name in FDA_ONLY and not fda:
            continue
        per_run = {n: field(r, s, wkey, skey) for n, r, s in entries}
        if not per_run:
            continue
        sources = "; ".join(sorted({v["source"] or "-" for v in per_run.values()}))
        if len({json.dumps(v["value"]) for v in per_run.values()}) == 1:
            rec[name] = {"value": next(iter(per_run.values()))["value"], "source": sources}
        else:
            rec[name] = {"per_run": {n: v["value"] for n, v in per_run.items()}, "source": sources}
        dis = {n: v["launch_script_disagrees"] for n, v in per_run.items() if "launch_script_disagrees" in v}
        if dis:
            rec[name]["launch_script_disagrees"] = dis
    if rec.get("seed", {}).get("value") == "not recorded":
        rec["seed"]["note"] = "the release training code never sets a seed (init_seeds is not called)"
    if rec.get("initialization", {}).get("value") == "not recorded" and entries:
        rec["initialization"]["note"] = "no --finetuning: the run starts from --weights (theta_0)"
    rec["checkpoint_rule"] = {"value": CHECKPOINT_RULE, "source": "code (main.py of the release)"}
    return rec, entries


def history_ap(hist):
    """{run name: [EVAL/AP per evaluated epoch]} from the wandb history dump."""
    out = {}
    for line in open(hist):
        r = json.loads(line)
        out[r["name"]] = [h["EVAL/AP"] for h in r.get("history", []) if isinstance(h.get("EVAL/AP"), (int, float))]
    return out


def yaml(obj, indent=0):
    pad = "  " * indent
    lines = []
    for k, v in obj.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{json.dumps(str(k))}:")
            lines += yaml(v, indent + 1)
        else:
            lines.append(f"{pad}{json.dumps(str(k))}: {json.dumps(v)}")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wandb_runs", type=Path, required=True)
    p.add_argument("--wandb_hist", type=Path, required=True)
    p.add_argument("--sbatch", type=Path, required=True, help="archive of the original launch scripts")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True, help="csv of the FDA runs (T6.2)")
    args = p.parse_args(argv)
    runs = [json.loads(line) for line in open(args.wandb_runs)]
    scripts = launch_scripts(args.sbatch)
    hist = history_ap(args.wandb_hist)
    recipes, grid, fam_entries = {}, [], {}
    for key, desc, wandb_re, script_re in FAMILIES:
        rec, entries = family(runs, scripts, wandb_re, script_re, fda=key.startswith("fda"))
        fam_entries[key] = entries
        recipes[key] = {"description": desc, **rec}
        if key.startswith("fda"):
            for name, r, _ in entries:
                ap = hist.get(name, [])
                cls = re.search(r"_(bus|motorcycle|rider|train|truck)_", name).group(1)
                grid.append({"family": key, "run": name, "wandb_id": r["id"] if r else None, "class": cls,
                             "lr": r["config"].get("lr") if r else None, "lp_lr": r["config"].get("lp_lr") if r else None,
                             "epochs_logged": len(ap), "mAP_best": max(ap) if ap else None,
                             "mAP_last": ap[-1] if ap else (r["summary"].get("EVAL/AP") if r else None),
                             "AP50": "not recorded", "state": r["state"] if r else None})
    points = defaultdict(list)   # (init, k, class) -> runs
    for fam in ("few_shot_theta0", "few_shot_abra"):
        for name, r, _ in fam_entries[fam]:
            k, cls = re.search(r"FOG_(\d+)_([a-z]+)", name).groups()
            points[(fam, int(k), cls)].append(r["config"] if r else {})
    ks = sorted({k for _, k, _ in points})
    lr_by_k = {f"{fam} k={k}": sorted({c.get("lr") for (f, kk, _), cs in points.items() if f == fam and kk == k for c in cs})
               for fam in ("few_shot_theta0", "few_shot_abra") for k in ks}
    same = all({c.get("train_annotations") for c in points.get(("few_shot_theta0", k, cls), [])}
               == {c.get("train_annotations") for c in points.get(("few_shot_abra", k, cls), [])}
               for _, k, cls in points)
    recipes["few_shot_summary"] = {
        "k_values (images per class)": ks, "lr_by_k": lr_by_k,
        "runs_per_point": sorted({len(v) for v in points.values()}),
        "same_subset_files_for_both_initializations": same,
        "k_0": "no few-shot run exists for k = 0 (the k = 0 points are the untrained models)",
        "subset_drawing": {"value": "not recorded", "source": None}}
    args.out.write_text("# Recipes of the revision model families (tools/recipes_round2.py)\n" + "\n".join(yaml(recipes)) + "\n")
    with open(args.grid, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    print(f"{len(recipes)} families -> {args.out}; {len(grid)} FDA runs -> {args.grid}")
    print(f"launch scripts parsed: {len(scripts)} exp_names")


if __name__ == "__main__":
    main()
