#!/bin/bash
#SBATCH -p all_usr_prod
#SBATCH --account=farm4trade_phd
#SBATCH --job-name=M3FD_CONVERT
#SBATCH --output=LOGS/%x_%j.out
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#
# M3FD -> the four data variants of the RGB -> thermal extension (PLAN §3): split v1 / v2 x Lamp prompt "street lamp" /
# "traffic light". CPU only. From the repository folder (abra-thermal), with the .venv active:
#
#   sbatch tools/m3fd_convert.sh
#
# Writes, under M (default /work/horizon_store/chiara/data/m3fd):
#   lamp_crops.png            Lamp boxes, Vis next to Ir (to choose the prompt)
#   masters/<tag>/            convert-m3fd output (+ conversion.json), tag = v1_street_lamp, v1_traffic_light, ...
#   <tag>/                    prepare-data output (manifest.json, label_maps/, annotations) and stats.json
# The log is LOGS/M3FD_CONVERT_<job id>.out. Existing outputs are rewritten.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[[ -n ${VIRTUAL_ENV:-} ]] || source .venv/bin/activate
M=${M:-/work/horizon_store/chiara/data/m3fd}

echo "+ $(date '+%F %T') on $(hostname), code $(git rev-parse --short HEAD)"
python tools/m3fd_crops.py "$M/raw" --cls Lamp --out "$M/lamp_crops.png"
for v in v1 v2; do
    for lamp in "street lamp" "traffic light"; do
        tag=${v}_${lamp// /_}
        echo "+ $tag"
        python -m abra convert-m3fd --raw "$M/raw" --split_file "abra/data/splits/m3fd_split_$v.json" \
            --out_dir "$M/masters/$tag" --lamp "$lamp"
        python -m abra prepare-data --benchmark masters --masters_root "$M/masters/$tag" --source_domain rgb \
            --out_dir "$M/$tag" --exclude_unavailable class all
        python -m abra data-stats --name "M3FD $tag" --out "$M/$tag/stats.json" \
            --source_train "$M/$tag/rgb/annotations/annotations_train.jsonl" \
            --target_train "$M/$tag/thermal/annotations/annotations_train.jsonl" \
            --target_eval "$M/$tag/thermal/annotations/annotations_test.json"
    done
done
echo "+ $(date '+%F %T') done"
