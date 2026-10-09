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
none and banded), task addition and Param Δ (run twice, to measure its run-to-run variance from GPU SVD signs) per
class, plus loss traces of two 5-step `--debug` training runs (domain and class expert) with seed 0. Optional:
`--alpha_search` adds ABRA-α (selected on the test split, i.e. the current oracle behaviour).

Run from the repo root on the GPU machine:

```bash
python tests/golden/run_gpu_golden.py \
    --gdino_weights /path/to/groundingdino_swint_ogc.pth \
    --fog_root /path/to/cityscapes/fog/images \
    --normal_root /path/to/cityscapes/normal/images \
    --annotations annotations \
    --source_domain /path/to/weights/cityscape_normal/best.pth \
    --target_domain /path/to/weights/cityscape_fog/best.pth \
    --experts_dir /path/to/weights/cityscapes_experts \
    --classes bus motorcycle rider train truck
```

Logs of every run go to `outputs_golden/<run>.log`; commit only `gpu_golden.json`.
