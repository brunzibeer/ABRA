# Golden reference (refactor Phase 0)

Outputs of the code at tag `official` (commit `47a7b73`), used to prove that the refactor does not change results.
Acceptance tolerance: **0.05 AP** (identical results preferred).

| File | Produced by | Needs |
|---|---|---|
| `offline_golden.json` | `make_offline_golden.py` | CPU only; domain experts + class experts |
| `gpu_golden.json` | `run_gpu_golden.py` | CUDA GPU, compiled deformable attention, Cityscapes/Foggy images, `annotations/`, base GDINO weights |

`offline_golden.json` fingerprints, per expert, method and layer, the class residual added to the target domain
expert: Frobenius norm and 8 fixed random probes `yᵀ ΔW x`. `simple_signdependent` (Param Δ) depends on
arbitrary SVD signs, so it is recorded for reference only.

`gpu_golden.json` holds AP/AP50/AP75/APs/APm/APl (3 decimals) for zero-shot, target domain expert, ABRA (mask
none; banded and task addition only with `--with_banded_ta`) and Param Δ (run twice, to measure its run-to-run variance from GPU SVD signs) per
class, plus loss traces of two 5-step `--debug` training runs (domain and class expert) with seed 0. Optional:
`--alpha_search` adds ABRA-α (selected on the test split, i.e. the current oracle behaviour).

**Environment matters.** References must be compared in the pinned environment (torch 2.1.0, numpy 1.24,
transformers 4.28; `uv sync`). SVD results depend on the torch/LAPACK build: with torch 2.2 instead of 2.1,
ABRA without mask and task addition still agree to ~1e-5, but banded/plain masks differ up to ~1e-3 (near-degenerate
singular vectors) and Param Δ changes completely (sign-dependent).

`cpu_forward_golden.json` holds the outputs of the ORIGINAL `main.py` (commit 6a34a0d, `--cli old`) on synthetic images
for 8 settings (6 evaluations, 2 two-step trainings): per-query probes of the raw outputs, the sorted top-100 scores and
loss traces. `cpu_forward_golden.py` runs the same settings through the `abra` CLI on CPU (~16 min) and compares.
Settings listed in `--allow_close` may differ per query if their sorted top-100 scores match within `--score_tol`;
see `docs/MIGRATION.md` §3.1 for why the transport settings are only close. `check_gpu_new_cli.py` runs the new CLI
on the GPU server and compares real AP with `gpu_golden.json` (tolerance 0.05).

Helpers:
- `make_image_subset.py`: copies only the ~1,000 images these runs read (362 foggy val, 362 normal val, the
  training images of one class) from full Cityscapes / Foggy Cityscapes, so a small server needs ~2 GB instead of ~40 GB.
- `check_env.py`: one-minute OK/FAIL check of GPU, package versions, compiled deformable attention (compared
  with the PyTorch reference), repo imports, BERT availability, weights and images. Run it first.

Run from the repo root on the GPU machine (`{cls}` is replaced by each class name; the runner first checks that
every weight, annotation and image file exists, and `--check_only` stops after that check):

```bash
python tests/golden/check_env.py
python tests/golden/run_gpu_golden.py \
    --gdino_weights /path/to/groundingdino_swint_ogc.pth \
    --fog_root /path/to/cityscapes/fog/images \
    --normal_root /path/to/cityscapes/normal/images \
    --fog_val_ann '/path/to/fog/annotations/{cls}/annotations_val.json' \
    --normal_ann_dir '/path/to/normal/annotations/{cls}' \
    --source_domain /path/to/cityscape_normal/best.pth \
    --target_domain /path/to/cityscape_fog/best.pth \
    --expert '/path/to/cityscapes_experts/exp1_{cls}/best.pth'
```

Logs of every run go to `outputs_golden/<run>.log`; commit only `gpu_golden.json`.
