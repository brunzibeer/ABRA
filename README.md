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

---

## 📦 Model Weights and Dataset Splits

| Resource | Link | Contents |
| --- | --- | --- |
| Pretrained Grounding DINO weights | [GroundingDINO releases](https://github.com/IDEA-Research/GroundingDINO/releases) | Base detector checkpoint (`groundingdino_swint_ogc.pth`) |
| ABRA release | [GitHub Releases](https://github.com/brunzibeer/ABRA/releases) | Domain and class expert weights, dataset splits (annotations) and label maps |

The ODVG annotation format is described in [data_format.md](data_format.md).
The splits can also be regenerated with `python -m abra prepare-data` (see [Dataset Preparation](#-dataset-preparation)).

---

## 📂 Dataset Preparation

We use **Cityscapes** (source) and **Foggy Cityscapes** (target, β = 0.02). Download both from the official website and arrange them as below, with the annotation files from the ABRA release:

```text
datasets/
└── cityscapes/
    ├── normal/
    │   ├── annotations/
    │   │   ├── annotations_{train.jsonl,val.json}       # all images and classes
    │   │   ├── domains/annotations_train_domain.jsonl
    │   │   └── experts/<class>/{annotations_train.jsonl, annotations_val.json}
    │   └── images/{train,val}/<city>/*.png
    ├── fog/
    │   ├── annotations/  (same structure as normal/)
    │   └── images/{train,val}/<city>/*_foggy_beta_0.02.png
    └── label_maps/
        ├── top_3.json, standard_map.json, uda_map.json
        └── experts/<class>.json
```

**Building the splits.** `prepare-data` writes the whole `annotations/` and `label_maps/` tree (CPU, seconds), either
from the Cityscapes `gtFine` polygons or from the two master files of the release:

```bash
python -m abra prepare-data --cityscapes_root /path/to/cityscapes --out_dir datasets/cityscapes
python -m abra prepare-data --train_master normal/annotations/annotations_train.jsonl \
    --val_master normal/annotations/annotations_val.json --out_dir datasets/cityscapes
```

Boxes are the extent of the gtFine polygons of the 8 Cityscapes classes. The domain files keep the 3 most frequent
train classes (car, person, bicycle: Objectification); the per-class files keep the images containing a class and only
its boxes. Foggy Cityscapes renders the same scenes, so its files are the normal ones with `_foggy_beta_0.02` in the
image names. `--check_against <dir>` compares every file with an existing tree (the released files are regenerated
identically; COCO files differ only in `info.date_created`). Options for further experiments:

- `--exclude_unavailable class all`: additional target domain files without the images that contain one
  (`annotations_train_domain_no_<class>.jsonl`) or any (`..._no_unavailable.jsonl`) unavailable class;
- `--few_shot 1 5 10 20 30` (`--few_shot_unit images|instances`, `--seed`): target class files with N shots,
  `experts/<class>/annotations_train_<N>.jsonl`, nested (the 10-shot set contains the 5-shot set).

**SDGOD.** The release ships, per weather domain, `annotations/annotations_{train.jsonl,test.json}` (the random
train/test split used in the paper). `python -m abra prepare-data --benchmark sdgod --sdgod_root <dir> --out_dir
datasets/sdgod` rebuilds the domain (top-3 classes of Day-Clear: car, person, truck), per-class and `multi/<class>/`
files (images containing a class, with all their boxes) of every domain, plus `label_maps/`.

`python -m abra data-stats` prints the class partition of a benchmark (images and boxes per class and split, share of
available classes / boxes / images), how many domain-expert images contain unavailable classes, what each strict
filter leaves, and how many images the per-class evaluation uses. `python -m abra convert-bdd` converts BDD-format
label files (SDGOD) to COCO json.

`train/` or `val/` is appended to `--root` automatically when the path contains `cityscapes`, `AODRaw` or `bdd`; otherwise pass `--add_split 1`.

---

## ⌨️ Command line

Everything runs through one entry point, from the repository root:

```bash
python -m abra <command> [options]      # python -m abra <command> -h lists the options
```

| Command | What it does | Main output |
| --- | --- | --- |
| `train-domain` | Domain expert (Objectification) | `best.safetensors`, `last.safetensors` |
| `train-expert` | Class expert: SVFT on a domain expert | class residual `best.safetensors` (~0.3 MB) |
| `eval` | Zero-shot, a domain expert, or a transported class residual | `results.json` |
| `transport` | Transport a residual offline (CPU, seconds, no data) | dense residual `.safetensors` |
| `migrate` | Convert original `.pth` checkpoints to the new formats | `.safetensors` |
| `fda` | FDA baseline (Table 2) | `best.safetensors` |
| `diagnose` | Weight-space diagnostics D1–D3, per-layer α fitted on available classes | report `.json`, `.alpha.json` |
| `report` | Collect all `results.json` into one table | markdown / csv |

Every run writes to `--output_root` (required, together with `--exp_name`):

- `run.log`: the complete console output;
- `metrics.jsonl`: every logged loss / AP as one JSON line;
- `config.json`: the resolved arguments, git commit and environment.

Metrics are mirrored to wandb only if you pass `--wandb_project` (and optionally `--wandb_entity`). Runs are seeded with `--seed` (default 0).

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

For the target domain expert (Foggy Cityscapes), use the same command with `fog` instead of `normal` in `--exp_name`, `--output_root`, `--root` and the annotation paths.

---

## 🎯 Class Expert Training

The Source Domain Expert is fine-tuned on one source-domain class with **Singular Value Fine-Tuning (SVFT)**, which isolates the class-specific residual (e.g. *bus*). The residual is stored in the singular basis of its parent domain expert (only the SVFT band is saved), together with the parent's hash and a checksum.

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

Two opt-in fixes change results and are off by default to reproduce the paper: `--fix_layer0_grad 1` (encoder layer 0 is otherwise never trained, see [docs/MIGRATION.md](docs/MIGRATION.md) §4) for both training commands, and `--svft_zero_init 1` (class experts start exactly at their parent).

The same command with `--domain` set to the *target* domain expert and target-domain data trains the target-domain fine-tuning reference (or residuals of available target classes).

---

## 🚀 Method Evaluation

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
| Param Δ baseline | `--method param_delta` |
| Task addition (θ_T + τ) | `--method task_addition` |
| ABRA-α | add `--alpha_search 1`. α is chosen on `--alpha_annotations`; without it, α is chosen on the test split and the result is marked `"oracle": true` |
| Fixed α | `--alpha 1.3` |
| Per-layer α (e.g. fitted by `diagnose` on available classes, no oracle) | `--alpha_file report.json.alpha.json` |
| Standard all-images protocol for one class | multi-class `--test_annotations` + `--eval_classes bus` |
| Zero-shot Grounding DINO | no `--domain`, no `--residual` |
| Target domain expert only | `--domain` without `--residual` |
| Source class expert on the target domain | `--domain` = source domain expert, `--residual` = its class expert, `--method task_addition` |

`results.json` holds AP, AP50, AP75, APs/m/l, per-class AP, method, α and the oracle flag.

**Original checkpoints.** The original `best.pth` files (domain and class experts) are read directly. Evaluation uses θ_T + α·R, while the original code rebuilt each layer from the float32 SVD of θ_T; the difference is that SVD reconstruction error (top detection scores agree within ~3e-4 in our CPU check, see [docs/MIGRATION.md](docs/MIGRATION.md)).

**Offline transport.** Transports only need weights, so they can be computed and saved without data:

```bash
python -m abra transport --method abra \
  --residual outputs/experts/cityscapes/normal/bus/best.safetensors \
  --target_domain outputs/domains/cityscapes/fog/best.safetensors \
  --out outputs/transported/abra_bus_fog.safetensors
```

**Converting original checkpoints** (93 MB class experts → ~0.3 MB):

```bash
python -m abra migrate --out_dir weights_new \
  --domain weights/cityscape_normal/best.pth weights/cityscape_fog/best.pth \
  --expert weights/cityscapes_experts/exp1_*/best.pth --parent weights/cityscape_normal/best.pth
```

**Diagnostics** (weights only). Given residuals of *available* classes trained on both domain experts:

```bash
python -m abra diagnose --out outputs/diagnostics/cityscapes.json \
  --source_domain outputs/domains/cityscapes/normal/best.safetensors \
  --target_domain outputs/domains/cityscapes/fog/best.safetensors \
  --pair car outputs/experts/cityscapes/normal/car/best.safetensors outputs/experts/cityscapes/fog/car/best.safetensors \
  --pair person outputs/experts/cityscapes/normal/person/best.safetensors outputs/experts/cityscapes/fog/person/best.safetensors
```

It prints D1 (source vs target residual similarity), D2 (leave-one-class-out error of each transport and of the per-layer α; < 1 beats task addition) and D3 (domain shift per layer), and writes the fitted per-layer α for `eval --alpha_file`.

**Tables**: `python -m abra report outputs/eval --csv results.csv`.

---

## 🌫️ FDA Baseline (Table 2)

Unsupervised domain adaptation with Fourier Domain Adaptation, trained on labelled source and unlabelled target images:

```bash
python -m abra fda \
  --exp_name fda_bus --output_root outputs/fda/bus \
  --weights weights/groundingdino_swint_ogc.pth \
  --source_root datasets/cityscapes/normal/images \
  --target_root datasets/cityscapes/fog/images \
  --source_train_annotations datasets/cityscapes/normal/annotations/experts/bus/annotations_train.jsonl \
  --target_train_annotations datasets/cityscapes/fog/annotations/experts/bus/annotations_train.jsonl \
  --target_test_annotations datasets/cityscapes/fog/annotations/experts/bus/annotations_val.json \
  --label_map datasets/cityscapes/label_maps/experts/bus.json \
  --epochs 12 --batch_size 2 --optim adamw --lr 1e-4 --weight_decay 1e-4
```

---

## 🧪 Tests

```bash
pytest tests/                                                   # unit tests (CPU; weight tests need ../weights)
python tests/golden/cpu_forward_golden.py --weights_dir weights # end-to-end CPU regression vs the original code
```

`tests/golden/` holds reference outputs of the original code and the scripts that produce and check them, including the GPU check of real AP (see [tests/golden/README.md](tests/golden/README.md)).
