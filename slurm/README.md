# ABRA on SLURM: complete guide

This guide is in the repository; the scripts it describes (`slurm/*.sh`) are cluster-specific and git-ignored: they
are copied to the server by hand.
They replace the original `sb_*.sh` (conda + `main.py`) with `python -m abra` in a uv environment, with the
original hyper-parameters.

Contents

1. [How the scripts work](#1-how-the-scripts-work)
2. [One-time setup on the server](#2-one-time-setup-on-the-server)
3. [Paths and environment variables](#3-paths-and-environment-variables)
4. [Array indices: classes and domains](#4-array-indices-classes-and-domains)
5. [Where the results go](#5-where-the-results-go)
6. [Data preparation: `data.sh`](#6-data-preparation-datash)
7. [Domain experts: `train_domain.sh`](#7-domain-experts-train_domainsh)
8. [Class experts and fine-tuning references: `train_expert.sh`](#8-class-experts-and-fine-tuning-references-train_expertsh)
9. [Full fine-tuning: `train_full.sh`](#9-full-fine-tuning-train_fullsh)
10. [Transported models: `transport.sh`](#10-transported-models-transportsh)
11. [Evaluation: `eval.sh`](#11-evaluation-evalsh)
12. [FDA baseline: `fda.sh`](#12-fda-baseline-fdash)
13. [Recipes: paper tables, rebuttal experiments, Option 1](#13-recipes-paper-tables-and-rebuttal-experiments)
14. [Old script -> new call](#14-old-script---new-call)
15. [What changed with respect to the old scripts](#15-what-changed-with-respect-to-the-old-scripts)
16. [Monitoring and troubleshooting](#16-monitoring-and-troubleshooting)

---

## 1. How the scripts work

| Script | Job | GPU | Default `--time` | Array over |
|---|---|---|---|---|
| `data.sh` | annotation files + statistics (run once, first) | no | 1 h | - |
| `train_domain.sh` | domain experts (Objectification) | yes | 6 h (SDGOD: pass `--time=24:00:00`) | domains |
| `train_expert.sh` | SVFT class residuals, fine-tuning references, few-shot | yes | 10 h | classes |
| `train_full.sh` | full fine-tuning on one class (no SVFT) | yes | 1.5 h | classes |
| `transport.sh` | ABRA-transported models as checkpoint files | no | 20 min | classes |
| `eval.sh` | every evaluation (zero-shot ... ABRA-α) | yes | 30 min | classes |
| `fda.sh` | FDA baseline (Table 2) | yes | 2.5 h, 50 GB | classes |
| `diagnose.sh` | weight-space diagnostics, Option 2 α, Option 1 learned transports | no | 3 h, 16 GB, 8 CPUs (about 30 min) | - |
| `common.sh` | shared settings, sourced by every script (not a job) | | | |

Principles:

- **One script per stage**, configured by positional arguments after the script name:
  `sbatch [sbatch options] slurm/<script>.sh <benchmark> <variant/setting> [<domain>] [...]`.
- **The array index selects the class or domain** (section 4). The `#SBATCH --array` in each file is only a default:
  pass `--array=...` on the command line to choose (`--array=3` runs only the fourth item).
- **Every job prints the exact command** it runs (`+ python -m abra ...` at the top of the `.err` log), so a run can
  always be repeated by hand.
- **Original runs are never overwritten**: checkpoints are read from `CKPT` (default: the original runs in
  `/work/horizon_store/abraout`) and every new run is written under `RUNS` (default `abraout/refactor`).
- `sbatch` options given on the command line override the `#SBATCH` lines (`--time`, `--mem`, `--constraint`, ...).
- The scripts run with `set -euo pipefail`: a missing argument or a failing command stops the job immediately.

---

## 2. One-time setup on the server

The repository is `/homes/mbernardi/git/ABRA` (git clone of branch `refactor`; `main` is the paper release and is
not used here). The scripts `slurm/*.sh` are copied in by hand (they are not in git; this guide is).

```bash
cd /homes/mbernardi/git/ABRA
git checkout refactor && git pull                          # code updates
tar -xzf ../slurm_ABRA.tar.gz                              # the scripts (only when they changed)

# data and base weights: links in the repository root (see section 3 to use other paths instead)
ln -s <folder with cityscapes/ and sdgod/> datasets
ln -s <folder with groundingdino_swint_ogc.pth> weights
mkdir -p LOGS                                              # SLURM does not create the log folder

# Python environment (once, and after changes to pyproject.toml / uv.lock)
module load cuda/11.8
uv sync                                                    # .venv with Python 3.10, torch 2.1.0 cu118, from uv.lock
source .venv/bin/activate
cd groundingdino/models/GroundingDINO/ops && python setup.py build install && cd -   # deformable attention op
python -m abra -h                                          # the CLI loads
```

Check the compiled op on a GPU node (`srun --gres=gpu:1 -p all_usr_prod --account=farm4trade_phd --pty bash`, then
`python groundingdino/models/GroundingDINO/ops/test.py`). Without the op the code still runs, on a slow PyTorch path.

Then: `sbatch slurm/data.sh` (section 6) and a first check, e.g. ABRA on Cityscapes -> Foggy (section 13.1).

---

## 3. Paths and environment variables

All defined in `common.sh`; any of them can be overridden for one submission by putting it in front of `sbatch`
(`VAR=value sbatch slurm/...`). SLURM passes the submitting environment to the job.

| Variable | Default | Meaning |
|---|---|---|
| `REPO` | `/homes/mbernardi/git/ABRA` | the repository; the scripts source `$REPO/slurm/common.sh` |
| `VENV` | `$REPO/.venv` | uv environment to activate |
| `ABRAOUT` | `/work/horizon_store/abraout` | root of the original runs (read only) |
| `CKPT` | `$ABRAOUT` | where domain experts (`domains/`), class experts (`experts/`) and fine-tuning runs (`fts/`) are read |
| `CKPT_FILE` | `best.pth` | checkpoint file name inside each run folder: `best.pth` (original) or `best.safetensors` (new) |
| `RUNS` | `$ABRAOUT/refactor` | where every new run writes |
| `WEIGHTS` | `$REPO/weights/groundingdino_swint_ogc.pth` | base Grounding DINO (θ0) |
| `CS` | `$REPO/datasets/cityscapes` | Cityscapes / Foggy Cityscapes (`normal/`, `fog/`, `label_maps/`) |
| `SD` | `$REPO/datasets/sdgod` | SDGOD (`<domain>/`, `label_maps/`) |
| `EXTRA` | empty | extra options appended to the `abra` command (e.g. `EXTRA="--fix_layer0_grad 1"`) |
| `WANDB_PROJECT`, `WANDB_ENTITY` | unset | also log to wandb (metrics are always in `metrics.jsonl` and `run.log`) |
| `MERGED_TEMPLATE` | `$ABRAOUT/transported/%s/merged.pth` | transported model per class (`%s` = class), parent of `ft4`, `few<N>_abra`, `fda_abra` |
| `TARGET_DOMAIN` | `$CKPT/domains/<b>/<target>/dom1/$CKPT_FILE` | `eval.sh`: target domain expert (`%s` = class) |
| `ALPHA_ANN` | unset | `eval.sh`: COCO json (`%s` = class) on which α is selected (otherwise the test split = oracle) |
| `CLASS_SET` | `unavailable` | `eval.sh`: `available` evaluates the transported available classes |
| `FT_DOMAIN` | target domain | `eval.sh ... ft`: domain the FT residual was trained on |

**Original vs new checkpoints.** By default everything reads the original checkpoints, so new evaluations reproduce
the paper setting. After retraining with the new code, point `CKPT` to the new runs:

```bash
CKPT=/work/horizon_store/abraout/refactor CKPT_FILE=best.safetensors sbatch slurm/eval.sh cityscapes fog abra
```

Mixed cases (e.g. original class experts with a newly trained strict target expert) use `TARGET_DOMAIN` (13.4).

**Original class experts need no `--source_domain`**: their files carry their own SVD basis and singular values.
New-format residuals record the path and hash of their parent and find it by themselves on the same server.

---

## 4. Array indices: classes and domains

The index of an array task (`--array=...`) picks one item of these lists (defined in `common.sh`):

| List | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Cityscapes all classes | bicycle | bus | car | motorcycle | person | rider | train | truck |
| Cityscapes unavailable | bus | motorcycle | rider | train | truck | | | |
| Cityscapes available | bicycle | car | person | | | | | |
| SDGOD all classes | bike | bus | car | motorcycle | person | rider | truck | |
| SDGOD unavailable | bike | bus | motorcycle | rider | | | | |
| SDGOD available | car | person | truck | | | | | |
| Cityscapes domains | normal | fog | | | | | | |
| SDGOD domains | daytime_sunny (source, Day-Clear) | daytime_foggy | dusk_rainy | night_rainy | night_sunny | | | |

An index past the end of the list stops the job with `array index N out of range`. The order of the old SDGOD eval
arrays (bus, motorcycle, rider, bike) is not kept: the new order is the same in every script.

---

## 5. Where the results go

```
$RUNS/                                   (default /work/horizon_store/abraout/refactor)
├── domains/<benchmark>/<domain>/<variant>/          train_domain.sh
├── experts/<benchmark>/<domain>/<variant>_<class>/  train_expert.sh (exp*), train_full.sh exp1full
├── fts/<benchmark>/<domain>/<variant>_<class>/      train_expert.sh ft2/ft4/few*, train_full.sh ft5
├── transported/<benchmark>/<target>/<class>/merged.safetensors   transport.sh
├── eval/<benchmark>/<target>/<setting>[_multi]_<expert variant>_<class>/   eval.sh
├── uda/cityscapes/fog/<variant>_<class>/            fda.sh
├── datasets/{cityscapes,sdgod}/                     data.sh (generated copy, compared with datasets/)
└── stats/                                           data.sh (data-stats tables)
```

Every run folder contains `run.log` (complete console output), `metrics.jsonl` (every loss and AP, one JSON per line),
`config.json` (all arguments, git commit, environment), and:

- training: `best.safetensors` (selected by AP on `--test_annotations`, as in the paper) and `last.safetensors`;
  domain experts store the trained tensors, class experts the SVFT residual (~0.3 MB, band in the canonical basis of
  the parent, with the parent's path and hash);
- evaluation: `results.json` (AP, AP50, AP75, APs/m/l, per-class AP, method, α, `oracle` flag, α metric).

Tables: `python -m abra report $RUNS/eval/cityscapes/fog [--csv results.csv]` (with the `.venv` active) collects
every `results.json` below a folder.

---

## 6. Data preparation: `data.sh`

```bash
sbatch slurm/data.sh
```

Run once before the other scripts (CPU only, a few minutes). Steps:

1. `prepare-data` rebuilds both benchmarks from their master files into `$RUNS/datasets/` and compares every file
   with the existing `datasets/` tree (`--check_against`). **If an existing file differs, the job stops** before
   touching anything (COCO json files may differ only in `info.date_created`).
2. The files that are **missing** in `datasets/` are copied there (`rsync --ignore-existing`: nothing existing is
   overwritten): Foggy class train files (`fog/annotations/experts/<cls>/annotations_train.jsonl`, lost), few-shot
   files (`experts/<cls>/annotations_train_{1,5,10,20,30}.jsonl`, images, seeded and nested), strict domain files
   (`domains/annotations_train_domain_no_<cls>.jsonl`, `..._no_unavailable.jsonl`, target domains), top-k domain files
   (`domains/annotations_train_domain_top{1,2}.jsonl`, label maps `top_{1,2}.json`) and the Cityscapes `multi/<cls>/`
   evaluation files. The log lists every copied file.
3. Checks that `label_maps/domains/top_3.json` (used by the original DOM1/DOM2) equals `label_maps/top_3.json`
   (used now) and prints a warning otherwise.
4. `data-stats` tables (class partition, θT images with unavailable classes, strict filters, evaluation protocol)
   in `$RUNS/stats/*.md` and `*.json`.

Note: if `datasets/` is a link, the missing files are added where the link points.

---

## 7. Domain experts: `train_domain.sh`

```bash
sbatch [--array=..] slurm/train_domain.sh <cityscapes|sdgod> <variant>
```

Array over domains (Cityscapes 0 normal, 1 fog; SDGOD 0-4, section 4). Common settings: 10 epochs, batch 4, AdamW,
lr 1e-4, weight decay 1e-4, multi-step milestones 7 9, `ref_point_head`/`sampling_offsets` at lr 1e-5, selection on
`annotations_<val|test>.json` of the domain.

| Variant | Training file / label map | Objectify | Trainable | Notes |
|---|---|---|---|---|
| `dom1` | `domains/annotations_train_domain.jsonl` / `top_3.json` | yes | encoder_attn | **paper** |
| `dom2` | same | no (class labels) | encoder_attn | Fig. 4 (semantic supervision) |
| `dom5` | `annotations_train.jsonl` (all classes) / `standard_map.json` | yes | encoder_attn | originally normal only (`--array=0`) |
| `dom6` | as dom1 | yes | encoder | |
| `dom7` | as dom6, `--seed 1` | yes | encoder | the original dom7 had the same arguments as dom6 (unseeded code) |
| `dom8` | `annotations_train_domain_top1.jsonl` / `top_1.json` (= the car file used originally) | yes | encoder | k = 1 |
| `top1`, `top2` | `..._top1/2.jsonl` / `top_1/2.json` | yes | encoder_attn | k ablation, same settings as dom1 |
| `no_<class>` | `..._no_<class>.jsonl` | yes | encoder_attn | strict: no target image containing `<class>` (target domains only) |
| `no_unavailable` | `..._no_unavailable.jsonl` | yes | encoder_attn | strict: no target image with any unavailable class |

Examples:

```bash
sbatch --array=0-1 slurm/train_domain.sh cityscapes dom1                  # normal and fog
sbatch --array=0-4 --time=24:00:00 slurm/train_domain.sh sdgod dom1       # all 5 SDGOD domains
sbatch --array=1 slurm/train_domain.sh cityscapes no_bus                  # strict fog expert for bus
sbatch --array=0-1 slurm/train_domain.sh cityscapes top2                  # k = 2
```

---

## 8. Class experts and fine-tuning references: `train_expert.sh`

```bash
sbatch [--array=..] slurm/train_expert.sh <cityscapes|sdgod> <variant> <domain> [all|unavailable|available]
```

Array over classes of the chosen set (default `all`; `unavailable` for `few*`). SVFT band half-width 2, 12 epochs,
batch 4, AdamW, weight decay 0, `sampling_offsets` SVFT parameters at `lp_lr`. The **parent** is the model the
residual is trained on and later added to.

| Variant | Parent | lr / lp_lr | Milestones | Output |
|---|---|---|---|---|
| `exp1`, `exp2` | `dom1` / `dom2` of `<domain>` | 1e-2 / 1e-3 | 7 9 | `experts/` |
| `exp5`, `exp6`, `exp7`, `exp8` | `dom5` ... `dom8` | 1e-2 / 1e-3 | 8 10 | `experts/` |
| `exptop1`, `exptop2` | `top1` / `top2` | 1e-2 / 1e-3 | 7 9 | `experts/` |
| `exp9` | θ0 (no domain expert) | 1e-2 / 1e-3 | 7 9 | `experts/` (θ0-residual ablation) |
| `ft2` | θ0, data of `<domain>` | 1e-2 / 1e-3 | 7 9 | `fts/` (fine-tuning reference) |
| `ft4` | transported model (`MERGED_TEMPLATE`) | 1e-2 / 1e-3 | 7 9 | `fts/` |
| `few<N>` (N = 1 5 10 20 30) | θ0, `annotations_train_<N>.jsonl` | 1e-4/1e-5 (N ≤ 10), 1e-3/1e-4 (20), 5e-3/1e-4 (30) | 7 9 | `fts/` |
| `few<N>_abra` | transported model | as `few<N>` | 7 9 | `fts/` |

Parents are read from `CKPT` (`$CKPT/domains/<b>/<domain>/<dom variant>/$CKPT_FILE`): by default the original
domain experts. To train on new domain experts: `CKPT=$RUNS CKPT_FILE=best.safetensors`.

Examples:

```bash
sbatch --array=0-7 slurm/train_expert.sh cityscapes exp1 normal             # 8 source class experts
sbatch --array=0-4 slurm/train_expert.sh cityscapes exp1 normal unavailable # only bus..truck
sbatch --array=0-3 slurm/train_expert.sh sdgod exp1 daytime_sunny unavailable
sbatch --array=0-2 slurm/train_expert.sh sdgod exp1 dusk_rainy available    # car person truck on Dusk-Rainy
sbatch --array=0-4 slurm/train_expert.sh cityscapes ft2 fog unavailable     # target fine-tuning reference
sbatch --array=0-4 --time=00:30:00 slurm/train_expert.sh cityscapes few10 fog
```

---

## 9. Full fine-tuning: `train_full.sh`

```bash
sbatch [--array=..] slurm/train_full.sh cityscapes <exp1full|ft5> <domain>
```

Fine-tunes the whole `encoder_attn` modules on one class (class labels, no SVFT): 12 epochs, batch 4, lr 1e-4,
weight decay 1e-4, lp_lr 1e-5. The result is a full checkpoint (like a domain expert), evaluated with
`eval --domain <run>/best.safetensors`.

| Variant | Starts from | Classes | Milestones | Output |
|---|---|---|---|---|
| `exp1full` | `dom1` of `<domain>` | unavailable | 8 10 | `experts/<b>/<domain>/exp1full_<cls>` |
| `ft5` | θ0 | all | 7 9 | `fts/<b>/<domain>/ft5_<cls>` |

---

## 10. Transported models: `transport.sh`

```bash
sbatch [--array=0-4] slurm/transport.sh <cityscapes|sdgod> <target domain> [alpha, default 1]
```

Writes θT + α·R_ABRA (source `exp1` expert of each unavailable class transported onto the target `dom1`) as a
domain-expert file `$RUNS/transported/<b>/<target>/<cls>/merged.safetensors`. CPU only, seconds per class. The
weights are bit-identical to what `eval --method abra --alpha α` evaluates. These files are the parents of `ft4`,
`few<N>_abra` and `fda_abra`; use them instead of the original `transported/<cls>/merged.pth` with

```bash
MERGED_TEMPLATE=/work/horizon_store/abraout/refactor/transported/cityscapes/fog/%s/merged.safetensors \
  sbatch --array=0-4 slurm/train_expert.sh cityscapes few10_abra fog
```

The original `merged.pth` were built by the original code (layers rebuilt from the float32 SVD of θT); the new ones
use the exact θT (docs/MIGRATION.md §3.1).

---

## 11. Evaluation: `eval.sh`

```bash
sbatch [--array=..] slurm/eval.sh <cityscapes|sdgod> <target domain> <setting> [expert variant, default exp1]
```

Array over the unavailable classes (Cityscapes 0-4, SDGOD 0-3; `CLASS_SET=available` for the available ones). The
source domain is automatic: Cityscapes = the other of `normal`/`fog`, SDGOD = `daytime_sunny`. The class expert is
`$CKPT/experts/<b>/<source>/<expert variant>_<cls>/$CKPT_FILE`, the target domain expert
`$CKPT/domains/<b>/<target>/dom1/$CKPT_FILE` (or `TARGET_DOMAIN`).

| Setting | Model evaluated | Paper row / old EVA |
|---|---|---|
| `zero_shot` | θ0 | Zero shot |
| `target` | θT | target domain expert alone |
| `source` | θS + τ (source class expert) | Source (EVA*B); see section 15 |
| `task_add` | θT + τ | direct addition (rebuttal R1-Q1); old `analogy`, EVA1B n2f |
| `param_delta` | Param Δ | Param Δ |
| `abra` | θT + R_ABRA, α = 1 | **ABRA** (EVA1D) |
| `abra_alpha` | ABRA-α: α ∈ [0.1, 10), step 0.1, patience 10, max AP | **ABRA-α** (EVA1F) |
| `abra_alpha_ap50`, `abra_alpha_ap75` | same, α maximising AP50 / AP75 | EVA1I/J, SDGOD G/H, I/L |
| `ft` | θ0 + an FT residual (`fts/<b>/<FT_DOMAIN>/<variant>_<cls>`, variant default `ft2`) | Fine-tuning reference; EVA8A |
| `learned` | θT + learned transport of τ (Option 1; `LEARNED=<variant>`, operator from `diagnose.sh` or `OPERATOR=<file>`) | new (13.9) |
| `<setting>_multi` | same model on `multi/<cls>/` | EVA*A / EVA*C |

**Test files.** Per class (default): `<target>/annotations/experts/<cls>/annotations_<val|test>.json`, the images
containing the class with only its boxes (paper protocol). `_multi`: `multi/<cls>/annotations_<val|test>.json`,
the same images with the boxes of every class and every class in the prompt (AP averaged over the classes, as the
original code did). Standard all-images protocol for one class: `EXTRA="--test_annotations <...>/annotations_val.json --eval_classes <cls>"`
(the last `--test_annotations` wins).

**α selection.** Without `ALPHA_ANN`, α is selected on the test split and `results.json` has `"oracle": true`
(as in the paper). With `ALPHA_ANN=<json, %s = class>` it is selected there (non-oracle). The file must use images
of the target root (`--root` is the target images folder).

**Expert variants.** `exp1` (paper), `exp8`, `exptop1/2` (their parent domain expert is used for `source`), `exp9`
(θ0 residual: `source` then evaluates θ0 + τ, `task_add` evaluates θT + τ).

Examples:

```bash
sbatch --array=0-4 slurm/eval.sh cityscapes fog abra                      # ABRA, Cityscapes -> Foggy
sbatch --array=0-4 slurm/eval.sh cityscapes fog abra_alpha                # ABRA-α (oracle, as in the paper)
sbatch --array=0-3 slurm/eval.sh sdgod night_rainy abra
sbatch --array=0-4 slurm/eval.sh cityscapes normal abra_multi exp8        # Foggy -> Cityscapes, k = 1 experts, multi
FT_DOMAIN=daytime_sunny sbatch --array=0-3 slurm/eval.sh sdgod daytime_foggy ft_multi ft2
```

---

## 12. FDA baseline: `fda.sh`

```bash
sbatch [--array=0-4] slurm/fda.sh <fda|fda_abra>
```

Cityscapes -> Foggy, one task per unavailable class: labelled source images of the class, restyled with the
amplitude spectrum of the unlabelled foggy images of the class (`fog/experts/<cls>/annotations_train.jsonl`, as in
the original runs), SVFT (`--svft 1`), 12 epochs, batch 4, weight decay 0, 50 GB memory.

| Variant | Starts from | lr / lp_lr |
|---|---|---|
| `fda` | θ0 | 1e-2 / 1e-3 |
| `fda_abra` | transported model (`MERGED_TEMPLATE`) | 1e-6 / 1e-7 |

The target test split is evaluated before training and after every epoch; `best.safetensors` is a residual of the
starting model. The original FDA built no learning-rate scheduler; `abra fda` does the same. Evaluate a result with
`eval --domain <start> --residual <run>/best.safetensors --method task_addition` (start = nothing for `fda`).

---

## 13. Recipes: paper tables and rebuttal experiments

### 13.1 First check (reproduces the GPU verification)

```bash
sbatch --array=0-4 slurm/eval.sh cityscapes fog abra
python -m abra report /work/horizon_store/abraout/refactor/eval/cityscapes/fog
```

Expected on a Quadro RTX 5000 (new code, original checkpoints): bus 57.223, motorcycle 30.008, rider 42.286,
train 34.969, truck 38.119 (mean 40.521). Other GPU types: a few hundredths of AP.

### 13.2 Table 1, Cityscapes -> Foggy, from the original checkpoints

```bash
for s in zero_shot source param_delta abra abra_alpha; do sbatch --array=0-4 slurm/eval.sh cityscapes fog $s; done
sbatch --array=0-4 slurm/eval.sh cityscapes fog ft ft2        # needs fts/cityscapes/fog/ft2_<cls> (train_expert.sh ft2)
```

SDGOD: the same per target domain with `--array=0-3 ... sdgod <domain> <setting>`.

### 13.3 Direct addition and the θ0-residual ablation (rebuttal R1-Q1)

```bash
sbatch --array=0-4 slurm/eval.sh cityscapes fog task_add                  # θT + τ_S (τ learned on θS)
sbatch --array=0-4 slurm/train_expert.sh cityscapes exp9 normal unavailable   # τ learned on θ0 (if not trained yet)
sbatch --array=0-4 slurm/eval.sh cityscapes fog task_add exp9             # θT + τ(θ0)
sbatch --array=0-4 slurm/eval.sh cityscapes fog source exp9               # θ0 + τ(θ0)
sbatch --array=0-4 slurm/eval.sh cityscapes fog target                    # θT alone
```

Note: `exp9` read from `CKPT`: original EXP9 runs exist for the classes trained by `sb_cs_train_exp_normal_zero`;
new ones go to `$RUNS` (then `CKPT=$RUNS CKPT_FILE=best.safetensors`).

### 13.4 Strict target domain expert (rebuttal R1-Q2 / R4-Q1)

```bash
sbatch --array=1 slurm/train_domain.sh cityscapes no_bus                  # one strict fog expert per class ...
sbatch --array=1 slurm/train_domain.sh cityscapes no_unavailable          # ... or one without all 5 classes
TARGET_DOMAIN=/work/horizon_store/abraout/refactor/domains/cityscapes/fog/no_%s/best.safetensors \
  sbatch --array=0 slurm/eval.sh cityscapes fog abra                      # bus with its strict expert (%s = class)
TARGET_DOMAIN=/work/horizon_store/abraout/refactor/domains/cityscapes/fog/no_unavailable/best.safetensors \
  sbatch --array=0-4 slurm/eval.sh cityscapes fog abra
```

The class experts stay the original ones (they only depend on the source domain). Image counts left by each filter:
`$RUNS/stats/cityscapes.md` (T3).

### 13.5 Objectification with k = 1, 2, 3 (rebuttal R4-Q6)

```bash
sbatch --array=0-1 slurm/train_domain.sh cityscapes top1                 # and top2 (k = 3 is dom1)
sbatch --array=0-4 slurm/train_expert.sh cityscapes exptop1 normal unavailable
CKPT=/work/horizon_store/abraout/refactor CKPT_FILE=best.safetensors \
TARGET_DOMAIN=/work/horizon_store/abraout/refactor/domains/cityscapes/fog/top1/best.safetensors \
  sbatch --array=0-4 slurm/eval.sh cityscapes fog abra exptop1           # source and target experts with k = 1
```

`eval.sh` uses the target `dom1` by default, so the k-ablation needs `TARGET_DOMAIN` (target expert with the same k).
The original k = 1
run (`dom8`, `exp8`) trained the whole encoder (`--trainable encoder`), unlike dom1 (`encoder_attn`); `top1`/`top2`
use encoder_attn so that only k changes.

### 13.6 Setting-compliant ABRA-α (rebuttal R1-Q3 / R4-Q2)

Option A, one global α from the available classes (no signal from unavailable target classes):

```bash
CLASS_SET=available sbatch --array=0-2 slurm/eval.sh cityscapes fog abra_alpha   # α per available class
python -m abra report /work/horizon_store/abraout/refactor/eval/cityscapes/fog     # read their α, take e.g. the mean
EXTRA="--alpha 1.2" sbatch --array=0-4 slurm/eval.sh cityscapes fog abra          # apply it to the unavailable classes
```

(the available-class experts `exp1_{bicycle,car,person}` of the source domain exist from the original 8-class runs).
Option B, α selected on another split of the class: `ALPHA_ANN=<json with %s>`; the images must be in the target
image root.

### 13.7 Few-shot (Fig. 6)

```bash
for n in 1 5 10 20 30; do
  sbatch --array=0-4 --time=00:30:00 slurm/train_expert.sh cityscapes few$n fog        # from θ0
  sbatch --array=0-4 --time=00:30:00 slurm/train_expert.sh cityscapes few${n}_abra fog # from ABRA
done
```

The few-shot files are seeded (`--seed 0` in `data.sh`) and nested (the 10-shot images contain the 5-shot images);
the original files were unseeded and are lost, so the draws differ from Fig. 6.

### 13.8 Setting-compliant model selection and opt-in fixes

These change results; do not mix them with paper numbers.

```bash
EXTRA="--val_annotations <coco json> --eval_after_epoch 1" sbatch ... slurm/train_expert.sh ...
EXTRA="--fix_layer0_grad 1" sbatch ... slurm/train_domain.sh ...            # also train encoder layer 0
EXTRA="--svft_zero_init 1" sbatch ... slurm/train_expert.sh ...             # residual starts at zero
```

### 13.9 Option 1: transport learned from the available classes (options.pdf §4.1)

The available classes have a residual on both domain experts: τ_S on θS (`experts/<b>/<source>/exp1_<cls>`) and
τ_T on θT (`experts/<b>/<target>/exp1_<cls>`, trained with target data, which the setting allows). For every layer a
map is fitted by ridge regression towards the identity so that it turns τ_S into τ_T, then applied to the unavailable
classes. Variants:

| Variant | Map per layer | Strong ridge gives |
|---|---|---|
| `weight_left` | τ_T ≈ A τ_S (A up to 1024×1024) | task addition |
| `weight_full` | τ_T ≈ A τ_S B (alternating closed forms) | task addition |
| `weight_lowrank<r>` | A = I + rank-r correction | task addition |
| `spectral_full` | c_T ≈ L c_S B in the target SVD coordinates (c = U_Tᵀ τ V_T, at most 256×256) | ABRA |
| `spectral_band<w>` | as spectral_full with banded L, B (w = 0: one scale per singular direction) | ABRA |

The ridge strength λ (relative to the data) is chosen per variant by the leave-one-class-out error D2 (fit on 2 of
the 3 available classes, predict the third; e < 1 = better than task addition). Encoder layer 0 is never trained
(random residuals): it keeps the identity map and is left out of the error.

1. Fit and score (CPU, no images):

   ```bash
   sbatch slurm/diagnose.sh cityscapes fog          # pairs bicycle car person, every variant
   sbatch slurm/diagnose.sh sdgod night_rainy       # pairs car person truck
   ```

   The `.out` log prints D1-D3, Option 2 and, per variant, the error for every λ and the best one. Files in
   `$RUNS/diagnose/`: `<b>_<source>_to_<target>.json` (report, incl. ||A - I|| per layer type),
   `.alpha.json` (Option 2) and `.<variant>.operator.safetensors` (Option 1, fitted on all classes at the best λ).
   Options through `EXTRA`: `--lambdas ...`, `--iters 10`, `--band_out 2` (keep only the SVFT band of the predicted
   target coefficients), `--keep_layer0 1`.

2. Evaluate only the variants with e < 1 on the unavailable classes:

   ```bash
   LEARNED=spectral_band0 sbatch --array=0-4 slurm/eval.sh cityscapes fog learned
   EXTRA="--alpha_file $RUNS/diagnose/cityscapes_normal_to_fog.json.alpha.json" \
     sbatch --array=0-4 slurm/eval.sh cityscapes fog task_add        # Option 2 (theta_T + alpha_l tau) for comparison
   python -m abra report $RUNS/eval/cityscapes/fog                  # with task_add, abra, abra_alpha
   ```

   The operator stores the hash of the target domain expert it was fitted on and refuses any other θT. The same
   operator also works offline: `python -m abra transport --method learned --operator <file> --residual <τ_S>
   --target_domain <θT> --merged_out <file>`.

---

## 14. Old script -> new call

| Old script (array) | New call |
|---|---|
| `sb_cs_train_dom` 0-1 DOM1, 2-3 DOM2, 4 DOM5, 5-6 DOM6, 7-8 DOM7, 9-10 DOM8 | `train_domain.sh cityscapes dom1` (`dom2`, `dom5` with `--array=0`, `dom6`, `dom7`, `dom8`) |
| `sb_sd_train_dom` 0-4 | `--array=0-4 --time=24:00:00 train_domain.sh sdgod dom1` |
| `sb_cs_train_exp_normal` 0-7 EXP1, 8-15 EXP2, 16-23 EXP5, 24-31 EXP6, 32-39 EXP7, 40-47 EXP8 | `--array=0-7 train_expert.sh cityscapes exp1 normal` (`exp2`, `exp5`, `exp6`, `exp7`, `exp8`) |
| `sb_cs_train_exp_fog` 0-7 | `--array=0-7 train_expert.sh cityscapes exp1 fog` |
| `sb_cs_train_exp_normal_zero` 0-7 EXP9 | `--array=0-7 train_expert.sh cityscapes exp9 normal` |
| `sb_cs_train_exp_normal_full` 0-4 | `--array=0-4 train_full.sh cityscapes exp1full normal` |
| `sb_cs_train_ft_normal` 0-7 FT2, 8-15 FT4 | `--array=0-7 train_expert.sh cityscapes ft2 normal` / `ft4 normal` |
| `sb_cs_train_ft_fog` 0-7 FT4, 8-15 FT5 | `--array=0-7 train_expert.sh cityscapes ft4 fog` / `--array=0-7 train_full.sh cityscapes ft5 fog` |
| `sb_sd_train_exp_ds` 0-3 | `--array=0-3 train_expert.sh sdgod exp1 daytime_sunny unavailable` |
| `sb_sd_train_exp_seen` 0-14 (3 per domain) | per domain: `--array=0-2 train_expert.sh sdgod exp1 <domain> available` |
| `sb_sd_train_ft` 0-19 (4 per domain) | per domain: `--array=0-3 train_expert.sh sdgod ft2 <domain> unavailable` |
| `sb_cs_eval_n2f` EVA1A, B, C, D, F, I, J | `eval.sh cityscapes fog` + `source_multi`, `task_add`, `abra_multi`, `abra`, `abra_alpha`, `abra_alpha_ap50`, `abra_alpha_ap75` |
| `sb_cs_eval_f2n` EVA1A/B/C/D, EVA8A/B/C/D | `eval.sh cityscapes normal` + `source_multi`, `source`, `abra_multi`, `abra` (`... exp8` for EVA8) |
| `sb_sd_eval_ds2df` / `ds2dr` A, B, C, D, α, α50, α75 | `--array=0-3 eval.sh sdgod <domain>` + `source_multi`, `source`, `abra_multi`, `abra`, `abra_alpha`, `abra_alpha_ap50`, `abra_alpha_ap75` |
| `sb_sd_eval_ds2nr` / `ds2ns` (α, α50, α75) | `--array=0-3 eval.sh sdgod <night_rainy/night_sunny> abra_alpha` (`_ap50`, `_ap75`) |
| `sb_sd_eval_analogy_all` | per domain: `--array=0-3 eval.sh sdgod <domain> task_add` |
| `sb_sd_eval_exp8a` | per domain: `FT_DOMAIN=daytime_sunny ... eval.sh sdgod <domain> ft_multi ft2` |
| `sb_uda_fda` / `sb_uda_fda_abra` | `fda.sh fda` / `fda.sh fda_abra` |
| `sb_uda_few_imgs` (N blocks of 5) / `_abra` | `--array=0-4 train_expert.sh cityscapes few<N> fog` / `few<N>_abra` |
| `sb_cs_eval_n2f` EVA1G (`abraplus`) | not converted: transporter removed (only Param Δ, ABRA(-α), task addition are kept) |
| `sb_bd_*` (BDD100K) | not converted (not used) |

All calls are `sbatch [--array=..] slurm/<script>`. The equivalence was checked by expanding the 429 original jobs
and comparing hyper-parameters, data files and checkpoints with the new commands (one task per configuration).

---

## 15. What changed with respect to the old scripts

Results:

- **Evaluation weights**: ABRA / Param Δ evaluate θT + α·R with the exact θT; per-class AP moves by up to a few
  tenths (GPU-checked: zero-shot and θT identical, ABRA ≤ 0.28 AP on single classes, ABRA-α same α). MIGRATION §3.1.
- **`source`**: the old evaluation of a class expert alone (`--svft_ckpt 1`) loaded only the SVFT tensors on θ0, so
  biases and text cross-attention in-projections were θ0's, not θS's. `source` now evaluates the trained θS + τ,
  so the Source rows change. MIGRATION §4.6.
- **Seeds**: runs are seeded (`--seed 0`; `dom7` uses 1); the original code was unseeded. GPU training is still not
  bit-reproducible (atomic adds in the deformable attention backward).

Scripts:

- conda -> uv `.venv`; `module load cuda/11.8` kept (compiled op).
- One log per array task (`%x_%A_%a`); the old `%x_%A` made all tasks of an array write the same file.
- Unused or default options removed (`--train_annotations` in evaluations, `--svft 1`, `--svft_pattern banded`,
  `--test 1`, `--model_cfg`, `--param_type default`, `--accumulation_steps 1`).
- `sb_cs_train_exp_normal_full` wrote into `exp5_<cls>` (the EXP5 folder): now `exp1full_<cls>`.
- `sb_sd_eval_exp8a` read `fts/.../ft2/<cls>/best.pth`, the training wrote `ft2_<cls>`: fixed.
- DOM1/DOM2 used `label_maps/domains/top_3.json`, the rest `label_maps/top_3.json`: now always the latter (`data.sh`
  checks they are equal). DOM8 used `experts/car` + `experts/car.json`: now the top-1 domain files, byte-identical.
- `--alpha_target ap50/ap75` (only in a later working copy of the original code) is `--alpha_metric AP50/AP75`.
- Few-shot files were in `fewimages/<cls>/`; now `experts/<cls>/annotations_train_<N>.jsonl` (regenerated).
- Consistent run names: `<benchmark>_<domain>_<variant>_<class>`; outputs under `$RUNS` (section 5).

---

## 16. Monitoring and troubleshooting

```bash
squeue -u $USER                                   # running / pending jobs
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,NodeList
tail -f LOGS/ABRA_EVAL_<jobid>_<task>.err         # the command and the progress bars
grep -h "\[result\]" LOGS/ABRA_EVAL_<jobid>_*.out # one result line per evaluation
scancel <jobid>                                   # cancel (add _<task> for one task)
```

| Problem | Cause / fix |
|---|---|
| job ends at once, no log | `LOGS/` missing: `mkdir -p /homes/mbernardi/git/ABRA/LOGS` |
| `No such file .../slurm/common.sh` | the repository is not at `REPO`: `REPO=<path> sbatch ...` (and edit the `#SBATCH --output` paths) |
| `array index N out of range` | `--array` larger than the class/domain list (section 4) |
| `FileNotFoundError` on an annotation file | run `slurm/data.sh` (few-shot, strict, top-k, multi, Foggy train files) |
| `... not found` on a checkpoint | check `CKPT`/`CKPT_FILE`; new runs are in `$RUNS` with `best.safetensors` |
| `MultiScaleDeformableAttention op not compiled` | compile the op in the `.venv` (section 2); otherwise very slow |
| CUDA error / out of memory on small GPUs | restrict the GPU type: `--constraint=gpu_A40_45G` |
| `prepare-data` stops with `different:` | an existing annotation file differs from the regenerated one: check before going on |
| results to compare with the paper | default `CKPT` (original checkpoints) and no `EXTRA` options; ABRA-α without `ALPHA_ANN` is the paper's oracle setting |
