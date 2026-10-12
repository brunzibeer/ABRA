# Migration notes: original code (tag `official`) → `abra` package

This file lists every behaviour change of the refactor and the findings about the original code that matter
for interpreting results. The original code is available at tag `official` (commit `47a7b73`).

## 1. Commands

| Original `main.py` | New |
| --- | --- |
| `--objectify 1` (domain expert) | `python -m abra train-domain` |
| `--svft 1 --finetuning <domain>` (class expert) | `python -m abra train-expert --domain <domain>` |
| `--test 1 --finetuning <target> --source_class <expert> --svft 1 --transporter abra` | `python -m abra eval --domain <target> --residual <expert> --method abra` |
| `--transporter simple` | `--method param_delta` |
| `--transporter analogy` | `--method task_addition` (see §3.2) |
| `--transporter merged --merged_alpha a` | not kept: equals `a`·ABRA + (1−`a`)·task addition |
| `--abra_mask`, `--alpha_search`, `--min_alpha/--max_alpha/--alpha_step/--alpha_patience`, `--alpha_estimate` | same names on `abra eval` |
| `--uda fda` | `python -m abra fda` |
| `--mse` / `--mse_gt` | `python -m abra diagnose` (D1/D2 compare transported and target-trained residuals in weight space) |
| — | new: `eval --eval_classes` (standard all-images protocol), `eval --alpha_file` (per-layer α), `report` |
| `--output_root`, `--exp_name`, wandb flags | same; every run also writes `run.log`, `metrics.jsonl`, `config.json` |
| `--model_cfg` | optional (defaults to `groundingdino/config/cfg_odvg.py`), read without `SLConfig` |

Removed as dead, broken or deprecated: `abrag` (crashed), `analogy_full`, `abraplus`, calibration (hard-coded
cluster paths), weighted Procrustes, `--hierarchical_training`, `--log_seen_classes`, `--frcnn*`, `--resume`,
`--dynamic_prompts`, SVFT patterns other than `banded`, ConfMix (never implemented).

## 2. Weight formats

| | Original | New |
| --- | --- | --- |
| Domain expert | `best.pth`: trained tensors, 47 MB, no metadata | `best.safetensors`: same tensors + provenance and content hash |
| Class expert | `best.pth`: dense `u, s_pre, s, v` per layer, 93 MB; `u, s_pre, v` duplicate the parent's SVD | `best.safetensors`: SVFT band in the canonical SVD basis of the parent (~0.3 MB), parent hash, checksum |

Original files are still read directly (`load_domain_expert`, `load_residual`), and `abra migrate` converts them.

Why "canonical": the sign of each singular vector returned by `torch.linalg.svd` is arbitrary and can change
between runs, devices and library builds (two original class experts trained from the *same* domain expert
share `U` in only 42 of 66 layers). A band stored in one basis and read in another is wrong, which is very
likely why storing only the band failed before. The new format recomputes the parent SVD in float64 with
fixed signs; the checksum catches a wrong parent or SVD drift. Measured on the released experts:
reconstruction error ≤ 1e-3 relative to ‖τ‖.

## 3. Behaviour changes (intentional)

### 3.1 Evaluation weights
The original evaluation replaced every adapted layer by `U_T (S_T + s_T) V_T` built from the float32 SVD of θ_T,
i.e. θ_T plus its SVD reconstruction error (max 6.6e-5 absolute per entry on the released experts). The new code
evaluates `θ_T + α·R` with the exact θ_T. On the CPU regression the sorted top-100 detection scores agree with the
original within ~3e-4 (ABRA, banded ABRA, Param Δ) and ~1e-3 (task addition, see §3.2); per-query outputs differ
more because a tiny change can reorder the two-stage query selection. Rebuilding the weights exactly as the original
code (same SVD, same op order) was tried and gave deviations of the same size, so it was not kept: bit-identical
outputs are not reachable without the original sparse kernels. The real-AP check is
`tests/golden/check_gpu_new_cli.py` (tolerance 0.05 AP).

Measured on the GPU (2026-10-12, original and new code in the same job on one Quadro RTX 5000; the original code
reproduces `gpu_golden.json` within 0.019 AP there): zero-shot and target domain expert are **identical** for all
five classes; ABRA and Param Δ change by up to a few tenths of AP on single classes, in both directions, because
the exact θ_T is used. The exact θ_T was kept as the reference (it is the correct evaluation, not an approximation).

| AP (Cityscapes → Foggy) | bus | motorcycle | rider | train | truck | mean |
| --- | --- | --- | --- | --- | --- | --- |
| ABRA, original code | 57.237 | 30.012 | 42.270 | 35.092 | 38.098 | 40.542 |
| ABRA, new code | 57.223 | 30.008 | 42.286 | 34.969 | 38.119 | 40.521 |
| ABRA, new code + migrated weights | 57.209 | 29.979 | 42.316 | 34.926 | 38.113 | 40.509 |
| Param Δ, original code | 37.497 | 7.210 | 9.256 | 8.057 | 13.756 | 15.155 |
| Param Δ, new code | 37.422 | 7.289 | 9.277 | 7.780 | 13.742 | 15.102 |

The largest changes are on `train`, which has only 23 boxes in its Foggy val split: one box crossing one IoU
threshold moves its AP by ~0.3.

### 3.2 Task addition
The original `analogy` transporter added `U_S (S_pre + s) V_S − θ_S`, i.e. τ plus the float32 reconstruction error
of θ_S (about 1% of ‖τ‖); `--method task_addition` adds exactly τ (verified in `tests/test_weights_transport.py`).
With `--svft 1` the original `analogy` loaded **none** of its weights (0/66 tensors, silently) and evaluated the
bare target domain expert.

### 3.3 Training
- `--scheduler cosine|sequential` are stepped every optimizer step (they were built in iterations but stepped
  once per epoch); `step`/`multi` are unchanged.
- `--param_type adapt` matches parameter names by substring (the original matched any single character of
  `"backbone"`/`"transformer"`).
- New, opt-in: `--val_annotations` (model selection on a validation split instead of the test split) and
  `--eval_after_epoch 1` (select after each epoch, so the last epoch can be selected). Defaults reproduce the
  original procedure and print a warning that selection uses the test split.
- SVFT is implemented with dense band matrices instead of `torch-sparse` (no compiled dependency). Initialisation,
  parameter layout and gradients are the same; the effective weights differ by float rounding (≤1.3e-6). Over a
  training run this behaves like any other float-level difference: Adam's first step (≈ ±lr·sign(grad)) flips the
  sign of a few near-zero gradient entries (0.16% in our check), so trajectories are statistically equivalent but
  not bit-identical. (Training on GPU was not bit-reproducible before either: the deformable-attention backward uses
  atomic adds.) Domain-expert training is bit-identical to the original on CPU.
- `--seed` (default 0) seeds python, numpy, torch and the data loader workers; the original code was unseeded.
- Default `--optim` is `adamw` (the original default `sgd` was never used in the documented commands).

### 3.4 ABRA-α
`--alpha_annotations` selects α on a given split. Without it α is selected on the test split, as before, and
`results.json` records `"oracle": true`.

### 3.5 FDA
`--fda_lb` is used (the original always used L = 0.01, which is the default). The original `--fda_ent` /
`--fda_ita` never reached the loss and are not exposed. A `last` checkpoint is also saved.

## 4. Findings about the original code (relevant for the paper)

1. **Encoder layer 0 never trains.** `torch.utils.checkpoint` (reentrant, the default in torch 2.1) returns
   outputs that do not require grad when its inputs don't. The inputs of `fusion_layers.0` and `layers.0` come
   from the frozen backbone/BERT, so these layers get no gradient: domain experts keep the pretrained weights
   there (fog and normal experts are bit-identical on all 20 layer-0 tensors), and class experts keep the
   random SVFT initialisation there (≈2% of the weight norm), which ABRA then transports. Fix, opt-in because it
   changes results: `--fix_layer0_grad 1` (non-reentrant checkpointing; verified in `tests/test_layer0_grad.py`).
   Related opt-in: `--svft_zero_init 1` starts class experts exactly at their parent (no random residual).
2. **Model selection on the test split**: `best.pth` (domain and class experts) and ABRA-α are selected by AP on
   `--test_annotations`. With the default order the model after the last epoch is never a `best` candidate.
3. **Param Δ depends on arbitrary SVD signs**: it reuses source coefficients in the target basis without
   alignment. Its numbers are reproducible on one machine but changed completely between two torch builds in our
   tests. With canonical bases (new-format residuals) it becomes deterministic but different from the paper.
4. **ABRA without mask equals the projection P_U τ P_V**, which is task addition on square layers (24/66). The mean
   ABRA AP of the GPU golden (5 classes, Cityscapes→Foggy) is 40.54, the task-addition figure in the research plan.
5. **FDA**: L fixed at 0.01; entropy minimisation never applied (see §3.5).
