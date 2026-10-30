#!/bin/bash
#SBATCH -p all_usr_prod
#SBATCH --account=farm4trade_phd
#SBATCH --job-name=M3FD_ZS
#SBATCH --output=LOGS/%x_%A_%a.out
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --array=0-3
#
# Phase 2 of the RGB -> thermal extension (PLAN §4, the go / no-go gate): zero-shot Grounding DINO (theta_0) per class
# on the M3FD test split, in both modalities. One array task per split version x modality:
#   0 = v1 thermal, 1 = v1 rgb, 2 = v2 thermal, 3 = v2 rgb
# Each task evaluates the 6 classes with the per-class protocol (experts/<class>/annotations_test.json: the images with
# the class, prompt "<class> ."), Lamp under both prompts ("traffic light", "street lamp"). Thermal predictions with
# score >= 0.3 are saved to look at later. From the repository folder (abra-thermal), with the .venv active:
#
#   sbatch tools/m3fd_zero_shot.sh            the 4 tasks (GPU, ~10 min each)
#   bash tools/m3fd_zero_shot.sh report       afterwards, on the login node: one table of all results (+ csv)
#
# Runs: $RUNS/zero_shot/m3fd/<v1|v2>/<thermal|rgb>/<class>/ (results.json, config.json, predictions.json).
# Logs: LOGS/M3FD_ZS_<job>_<task>.out.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
REPO=$PWD
M=${M:-/work/horizon_store/chiara/data/m3fd}
RUNS=${RUNS:-/work/horizon_store/abraout/thermal}
OUT=$RUNS/zero_shot/m3fd
WEIGHTS=${WEIGHTS:-$REPO/weights/groundingdino_swint_ogc.pth}
export PYTHONUNBUFFERED=1

set +u   # module and activate scripts read unset variables
if [[ ${1:-} != report ]] && command -v module >/dev/null 2>&1; then
    module unload cuda >/dev/null 2>&1 || true
    module load cuda/11.8
fi
[[ -n ${VIRTUAL_ENV:-} ]] || source .venv/bin/activate
set -u

if [[ ${1:-} == report ]]; then
    python -m abra report "$OUT" --csv "$OUT/zero_shot.csv"
    exit
fi

TASKS=("v1 thermal" "v1 rgb" "v2 thermal" "v2 rgb")
read -r v mod <<< "${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}"
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
echo "+ $(date '+%F %T') on $(hostname) (${gpu:-no GPU}): $v $mod"
for cls in person car bus motorcycle truck "traffic light" "street lamp"; do
    variant=${v}_traffic_light
    [[ $cls == "street lamp" ]] && variant=${v}_street_lamp
    tag=${cls// /_}
    run=$OUT/$v/$mod/$tag
    extra=()
    [[ $mod == thermal ]] && extra=(--save_coco_format "$run/predictions.json" --zs_th 0.3)
    echo "+ $(date +%T) $cls"
    python -m abra eval --exp_name "m3fd_${v}_${mod}_zero_shot_$tag" --output_root "$run" --weights "$WEIGHTS" \
        --root "$M/raw" --add_split 0 \
        --test_annotations "$M/$variant/$mod/annotations/experts/$cls/annotations_test.json" ${extra[@]+"${extra[@]}"}
done
python -m abra report "$OUT/$v/$mod"
echo "+ $(date '+%F %T') done"
