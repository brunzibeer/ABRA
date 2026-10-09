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
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
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
<!-- TODO: the split-generation script will be added as tools/prepare_data.py -->

---

## 📂 Dataset Preparation

We use **Cityscapes** (source) and **Foggy Cityscapes** (target, β = 0.02). Download both from the official website and arrange them as below, with the annotation files from the ABRA release:

```text
datasets/
└── cityscapes/
    ├── normal/
    │   ├── annotations/
    │   │   ├── domains/annotations_train_domain.jsonl
    │   │   └── experts/<class>/{annotations_train.jsonl, annotations_val.json}
    │   └── images/{train,val}/<city>/*.png
    ├── fog/
    │   ├── annotations/  (same structure as normal/)
    │   └── images/{train,val}/<city>/*_foggy_beta_0.02.png
    └── label_maps/
        ├── top_3.json
        └── experts/<class>.json
```

`main.py` appends `train/` or `val/` to `--root` automatically when the path contains `cityscapes`, `AODRaw` or `bdd`; otherwise pass `--add_split 1`.

All commands below are run from the repository root. `--exp_name` is required; outputs (checkpoints, visualizations) go to `--output_root`.

---

## 🏋️‍♂️ Domain Expert Training

The first step of the ABRA pipeline is **Objectification**. We train Domain Experts on the source and target domains to learn class-agnostic object localization for each condition (every box is labelled `object`).

### 1. Source Domain Expert (Cityscapes)

```bash
python main.py \
  --exp_name dom_normal \
  --output_root outputs/domains/cityscapes/normal \
  --root datasets/cityscapes/normal/images \
  --train_annotations datasets/cityscapes/normal/annotations/domains/annotations_train_domain.jsonl \
  --test_annotations datasets/cityscapes/normal/annotations/annotations_val.json \
  --label_map datasets/cityscapes/label_maps/top_3.json \
  --objectify 1 \
  --model_cfg groundingdino/config/cfg_odvg.py \
  --weights weights/groundingdino_swint_ogc.pth \
  --trainable encoder_attn --param_type default \
  --epochs 10 --batch_size 2 --accumulation_steps 1 \
  --optim adamw --lr 1e-4 --weight_decay 1e-4 --scheduler multi --step_milestones 7 9 \
  --lp_names ref_point_head sampling_offsets --lp_lr 1e-5
```

### 2. Target Domain Expert (Foggy Cityscapes)

Same command, with the target-domain data:

```bash
python main.py \
  --exp_name dom_fog \
  --output_root outputs/domains/cityscapes/fog \
  --root datasets/cityscapes/fog/images \
  --train_annotations datasets/cityscapes/fog/annotations/domains/annotations_train_domain.jsonl \
  --test_annotations datasets/cityscapes/fog/annotations/annotations_val.json \
  --label_map datasets/cityscapes/label_maps/top_3.json \
  --objectify 1 \
  --model_cfg groundingdino/config/cfg_odvg.py \
  --weights weights/groundingdino_swint_ogc.pth \
  --trainable encoder_attn --param_type default \
  --epochs 10 --batch_size 2 --accumulation_steps 1 \
  --optim adamw --lr 1e-4 --weight_decay 1e-4 --scheduler multi --step_milestones 7 9 \
  --lp_names ref_point_head sampling_offsets --lp_lr 1e-5
```

---

## 🎯 Class Expert Training

We fine-tune the Source Domain Expert on one source-domain class with **Singular Value Fine-Tuning (SVFT)**, which isolates the class-specific residual (e.g. *bus*):

```bash
python main.py \
  --exp_name exp1_bus \
  --output_root outputs/experts/cityscapes/normal/exp1_bus \
  --root datasets/cityscapes/normal/images \
  --train_annotations datasets/cityscapes/normal/annotations/experts/bus/annotations_train.jsonl \
  --test_annotations datasets/cityscapes/normal/annotations/experts/bus/annotations_val.json \
  --label_map datasets/cityscapes/label_maps/experts/bus.json \
  --model_cfg groundingdino/config/cfg_odvg.py \
  --weights weights/groundingdino_swint_ogc.pth \
  --finetuning outputs/domains/cityscapes/normal/best.pth \
  --trainable encoder_attn --param_type default \
  --epochs 12 --batch_size 4 --accumulation_steps 1 \
  --optim adamw --lr 1e-2 --weight_decay 0.0 --scheduler multi --step_milestones 7 9 \
  --lp_names ref_point_head sampling_offsets --lp_lr 1e-3 \
  --svft 1 --svft_pattern banded --svft_diag 2
```

Repeat for the other classes by changing the class name in `--exp_name`, `--output_root`, the annotation paths and `--label_map`.

---

## 🚀 Method Evaluation

ABRA transports the source class residual onto the **target** domain expert (`--finetuning`) and evaluates on the target domain, without any target-domain labels of that class:

```bash
python main.py \
  --exp_name abra_bus_fog \
  --output_root outputs/eval/abra/bus \
  --root datasets/cityscapes/fog/images \
  --test_annotations datasets/cityscapes/fog/annotations/experts/bus/annotations_val.json \
  --model_cfg groundingdino/config/cfg_odvg.py \
  --weights weights/groundingdino_swint_ogc.pth \
  --finetuning outputs/domains/cityscapes/fog/best.pth \
  --source_class outputs/experts/cityscapes/normal/exp1_bus/best.pth \
  --svft 1 --svft_pattern banded --svft_diag 2 \
  --transporter abra \
  --test 1
```

Variants of the same command:

| Method | Change |
| --- | --- |
| Param Δ baseline | `--transporter simple` |
| ABRA-α | add `--alpha_search 1` (α in `[--min_alpha, --max_alpha)` with step `--alpha_step`). **Note:** α is selected on `--test_annotations`. |
| Zero-shot Grounding DINO | remove `--finetuning`, `--source_class`, `--svft`, `--transporter` |
| Target domain expert only | remove `--source_class`, `--svft`, `--transporter` |

---

## 🌫️ FDA Baseline (Table 2)

Unsupervised domain adaptation with Fourier Domain Adaptation, trained on labelled source and unlabelled target images:

```bash
python main.py \
  --exp_name fda_bus \
  --output_root outputs/fda/bus \
  --uda fda \
  --source_root datasets/cityscapes/normal/images \
  --target_root datasets/cityscapes/fog/images \
  --source_train_annotations datasets/cityscapes/normal/annotations/experts/bus/annotations_train.jsonl \
  --target_train_annotations datasets/cityscapes/fog/annotations/experts/bus/annotations_train.jsonl \
  --target_test_annotations datasets/cityscapes/fog/annotations/experts/bus/annotations_val.json \
  --label_map datasets/cityscapes/label_maps/experts/bus.json \
  --model_cfg groundingdino/config/cfg_odvg.py \
  --weights weights/groundingdino_swint_ogc.pth \
  --trainable encoder_attn --param_type default \
  --epochs 12 --batch_size 2 --optim adamw --lr 1e-4 --weight_decay 1e-4
```
<!-- TODO(berna): replace with the exact Table 2 hyper-parameters. Note: the FDA band size is currently fixed at L=0.01 (--fda_lb is not read) and --fda_ent/--fda_ita have no effect. -->

---

## 🧪 Regression tests (refactor)

`tests/golden/` holds reference outputs of the released code and the scripts that produce and check them (see [tests/golden/README.md](tests/golden/README.md)). The CPU check needs no GPU or dataset:

```bash
python tests/golden/cpu_forward_golden.py --weights_dir weights
```
