# [PRL] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection

<!-- TODO: add Project Page and arXiv badges once the links exist -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the PRL paper (under review): **"ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection"**.

> **Abstract:** *Although recent Open-Vocabulary Object Detection architectures, such as GDINO, demonstrate strong zero-shot capabilities, their performance degrades significantly under domain shifts. Moreover, many domains of practical interest, such as nighttime or foggy scenes, lack large annotated datasets, preventing direct fine-tuning. In this paper, we introduce Aligned Basis Relocation for Adaptation (ABRA), a method that transfers class-specific detection knowledge from a labeled source domain to a target domain where no training images containing these classes are accessible. ABRA formulates this adaptation as a geometric transport problem in the weight space of a pretrained detector, aligning source and target domain experts to transport class-specific knowledge. Extensive experiments across challenging domain shifts demonstrate that ABRA successfully teleports class-level specialization under multiple adverse conditions.*

![Teaser Image](assets/method.jpg)
*(Caption: Overview of the transfer pipeline. Starting from domain experts obtained through Objectification, class-specific residuals learned via SVFT in the source domain are analytically transported to unseen target domains.)*

---

## 🛠️ Installation

Both options use Python 3.10, PyTorch 2.1.0 and CUDA 11.8.

**Option 1: uv (recommended, uses the lock file)**

```bash
uv sync
source .venv/bin/activate
```

On Linux, uv installs the CUDA 11.8 PyTorch wheels. On other platforms, check that the resolved PyTorch matches your hardware.

**Option 2: conda**

```bash
conda create -n abra python=3.10
conda activate abra
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Then, for both options:**

1. Compile the deformable attention CUDA op (needs `nvcc`):
   ```bash
   cd groundingdino/models/GroundingDINO/ops && python setup.py build install && python test.py && cd -
   ```
   Without it, the code still imports and runs on CPU with a slow PyTorch fallback (useful for tests, not for training).
2. Download the base Grounding DINO weights to `weights/`:
   ```bash
   mkdir -p weights && wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   ```

Everything runs through one entry point, from the repository root: `python -m abra <command> [options]`
(`python -m abra <command> -h` lists the options). Every run writes to `--output_root` (required, together with
`--exp_name`): checkpoints, `run.log` (complete console output), `metrics.jsonl` (every loss / AP) and `config.json`
(arguments, git commit, environment). Metrics go to wandb only with `--wandb_project`. Runs are seeded with `--seed`
(default 0).

---

## 📦 Model Weights and Dataset Splits

| Resource | Link | Contents |
| --- | --- | --- |
| Pretrained Grounding DINO weights | [GroundingDINO releases](https://github.com/IDEA-Research/GroundingDINO/releases) | Base detector checkpoint (`groundingdino_swint_ogc.pth`) |
| ABRA release | [GitHub Releases](https://github.com/brunzibeer/ABRA/releases) | Domain and class expert weights, dataset splits (annotations) and label maps |

Annotations are in ODVG format for training (jsonl, see [data_format.md](data_format.md)) and COCO format for
evaluation (json). The original `best.pth` checkpoints are read directly by every command.

---

## 📂 Data Preparation

Two benchmarks: **Cityscapes → Foggy Cityscapes** (β = 0.02) and **SDGOD** (source Day-Clear; targets Day-Foggy,
Dusk-Rainy, Night-Clear, Night-Rainy). Download the images from the official websites. The annotation files are built
with `python -m abra prepare-data` (CPU, under a minute), which regenerates the released splits identically (COCO json
files differ only in `info.date_created`).

**Objectification and class partition.** Domain-expert files keep the boxes of the 3 most frequent classes of the
source training set (Cityscapes: car, person, bicycle; SDGOD: car, person, truck), the *available* classes. Every other
class is *unavailable* in the target domain. Per-class files keep the images that contain a class and only its boxes.

### Cityscapes → Foggy Cityscapes

```bash
# from the Cityscapes gtFine polygons (<cityscapes_root>/gtFine/{train,val})
python -m abra prepare-data --cityscapes_root /path/to/cityscapes --out_dir datasets/cityscapes
# or from the two master files of the release
python -m abra prepare-data --out_dir datasets/cityscapes \
  --train_master /path/to/release/normal/annotations/annotations_train.jsonl \
  --val_master /path/to/release/normal/annotations/annotations_val.json
```

Boxes are the extent of the gtFine polygons of the 8 Cityscapes classes. Foggy Cityscapes renders the same scenes, so
the fog files are the normal ones with `_foggy_beta_0.02` in the image names. Result, with the images placed as shown:

```text
datasets/cityscapes/
├── normal/
│   ├── annotations/
│   │   ├── annotations_train.jsonl, annotations_val.json      # all images and classes
│   │   ├── domains/annotations_train_domain.jsonl             # Objectification (available classes)
│   │   └── experts/<class>/{annotations_train.jsonl, annotations_val.json}
│   └── images/{train,val}/<city>/*.png
├── fog/
│   ├── annotations/                                           # same structure as normal/
│   └── images/{train,val}/<city>/*_foggy_beta_0.02.png
└── label_maps/
    ├── top_3.json, standard_map.json
    └── experts/<class>.json
```

### SDGOD

The release provides, for each domain, the train/test split as `annotations/annotations_train.jsonl` and
`annotations/annotations_test.json` (all classes). From these, `prepare-data` builds the rest:

```bash
python -m abra prepare-data --benchmark sdgod --sdgod_root /path/to/release/sdgod --out_dir datasets/sdgod
```

```text
datasets/sdgod/
├── <domain>/                     # daytime_sunny (source), daytime_foggy, dusk_rainy, night_sunny, night_rainy
│   ├── annotations/
│   │   ├── annotations_train.jsonl, annotations_test.json
│   │   ├── domains/annotations_train_domain.jsonl
│   │   ├── experts/<class>/{annotations_train.jsonl, annotations_test.json}
│   │   └── multi/<class>/annotations_test.json             # images containing the class, with all their boxes
│   └── images/
└── label_maps/
    ├── standard_map.json, domains/top_3.json
    └── experts/<class>.json
```

`train/` or `val/` is appended to `--root` automatically when the path contains `cityscapes`, `AODRaw` or `bdd`;
otherwise pass `--add_split 1`.

### Optional files

| Option of `prepare-data` | Files (target domains only) | Use |
| --- | --- | --- |
| `--exclude_unavailable class` | `domains/annotations_train_domain_no_<class>.jsonl` | target domain expert without the images that contain `<class>` |
| `--exclude_unavailable all` | `domains/annotations_train_domain_no_unavailable.jsonl` | target domain expert without the images that contain any unavailable class |
| `--few_shot 1 5 10 20 30` (`--few_shot_unit images\|instances`, `--seed`) | `experts/<class>/annotations_train_<N>.jsonl` | few-shot training; seeded and nested (the 10-shot set contains the 5-shot set) |

`--check_against <dir>` compares every written file with an existing tree (e.g. the release).

### Statistics

`data-stats` prints, per benchmark, the images and boxes of every class and split, the share of available classes,
boxes and images, how many target domain-expert images contain unavailable classes, what each strict filter leaves, and
how many images the per-class evaluation uses:

```bash
python -m abra data-stats --out stats_cityscapes.json \
  --source_train datasets/cityscapes/normal/annotations/annotations_train.jsonl \
  --target_train datasets/cityscapes/fog/annotations/annotations_train.jsonl \
  --target_eval datasets/cityscapes/fog/annotations/annotations_val.json
```

For SDGOD, use the Day-Clear train file as `--source_train` and the train / test files of one target domain.

---

## 🏋️‍♂️ Domain Expert Training

The first step of the ABRA pipeline is **Objectification**: Domain Experts are trained on the source and target domains with class-agnostic boxes (every box is an `object`), to learn object localization for each condition.

```bash
python -m abra train-domain \
  --exp_name dom_normal --output_root outputs/domains/cityscapes/normal \
  --weights weights/groundingdino_swint_ogc.pth \
  --root datasets/cityscapes/normal/images \
  --train_annotations datasets/cityscapes/normal/annotations/domains/annotations_train_domain.jsonl \
  --test_annotations datasets/cityscapes/normal/annotations/annotations_val.json \
  --label_map datasets/cityscapes/label_maps/top_3.json \
  --trainable encoder_attn --epochs 10 --batch_size 2 \
  --optim adamw --lr 1e-4 --weight_decay 1e-4 --scheduler multi --step_milestones 7 9 \
  --lp_names ref_point_head sampling_offsets --lp_lr 1e-5
```

For the target domain expert (Foggy Cityscapes), use the same command with `fog` instead of `normal` in `--exp_name`,
`--output_root`, `--root` and the annotation paths. For a strict target domain expert, pass one of the
`annotations_train_domain_no_*.jsonl` files as `--train_annotations`.

---

## 🎯 Class Expert Training

The Source Domain Expert is fine-tuned on one source-domain class with **Singular Value Fine-Tuning (SVFT)**, which isolates the class-specific residual (e.g. *bus*). The residual is stored in the singular basis of its parent domain expert (only the SVFT band is saved, ~0.3 MB), together with the parent's hash and a checksum.

```bash
python -m abra train-expert \
  --exp_name exp1_bus --output_root outputs/experts/cityscapes/normal/bus \
  --weights weights/groundingdino_swint_ogc.pth \
  --domain outputs/domains/cityscapes/normal/best.safetensors \
  --root datasets/cityscapes/normal/images \
  --train_annotations datasets/cityscapes/normal/annotations/experts/bus/annotations_train.jsonl \
  --test_annotations datasets/cityscapes/normal/annotations/experts/bus/annotations_val.json \
  --label_map datasets/cityscapes/label_maps/experts/bus.json \
  --svft_diag 2 --trainable encoder_attn --epochs 12 --batch_size 4 \
  --optim adamw --lr 1e-2 --weight_decay 0.0 --scheduler multi --step_milestones 7 9 \
  --lp_names ref_point_head sampling_offsets --lp_lr 1e-3
```

The same command with `--domain` set to the *target* domain expert and target-domain data trains the target-domain
fine-tuning reference (or residuals of available target classes). For few-shot runs, pass a target
`experts/<class>/annotations_train_<N>.jsonl` file as `--train_annotations`.

Model selection uses `--test_annotations` by default, as in the paper; `--val_annotations <coco json>` selects on
another split instead, and `--eval_after_epoch 1` lets the last epoch be selected. Two opt-in fixes change results and
are off by default to reproduce the paper: `--fix_layer0_grad 1` (encoder layer 0 is otherwise never trained, see
[docs/MIGRATION.md](docs/MIGRATION.md) §4) for both training commands, and `--svft_zero_init 1` (class experts start
exactly at their parent).

---

## 🚀 ABRA

ABRA transports the source class residual onto the **target** domain expert (`--domain`) and evaluates on the target domain, without any target-domain labels of that class:

```bash
python -m abra eval \
  --exp_name abra_bus_fog --output_root outputs/eval/abra/bus \
  --weights weights/groundingdino_swint_ogc.pth \
  --domain outputs/domains/cityscapes/fog/best.safetensors \
  --residual outputs/experts/cityscapes/normal/bus/best.safetensors \
  --method abra \
  --root datasets/cityscapes/fog/images \
  --test_annotations datasets/cityscapes/fog/annotations/experts/bus/annotations_val.json
```

| Setting | Options |
| --- | --- |
| ABRA | `--method abra` (optionally `--abra_mask banded` or `plain`) |
| ABRA-α | add `--alpha_search 1`. α is chosen on `--alpha_annotations`; without it, α is chosen on the test split and the result is marked `"oracle": true` |
| Fixed α | `--alpha 1.3` |
| Param Δ baseline | `--method param_delta` |
| Task addition (θ_T + τ) | `--method task_addition` |
| Zero-shot Grounding DINO | no `--domain`, no `--residual` |
| Target domain expert only | `--domain` without `--residual` |
| Source class expert on the target domain | `--domain` = source domain expert, `--residual` = its class expert, `--method task_addition` |
| Standard all-images protocol for one class | `--test_annotations datasets/cityscapes/fog/annotations/annotations_val.json --eval_classes bus` |

`results.json` holds AP, AP50, AP75, APs/m/l, per-class AP, method, α and the oracle flag;
`python -m abra report outputs/eval --csv results.csv` collects all runs into one table.

The released `best.pth` checkpoints work as `--domain` and `--residual`. Evaluation uses θ_T + α·R with the exact
target weights, while the original code rebuilt each layer from the float32 SVD of θ_T; per-class AP moves by up to a
few tenths (see [docs/MIGRATION.md](docs/MIGRATION.md) §3.1).

Other commands (offline `transport`, `migrate` of the original checkpoints, weight-space `diagnose`) are listed by
`python -m abra -h` and described in [docs/MIGRATION.md](docs/MIGRATION.md).
