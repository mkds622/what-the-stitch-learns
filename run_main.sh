#!/bin/bash
# The main grid: four depths, two objectives, five seeds each.
#
# Runs are sequential and each one is skipped if its configuration has already
# been trained, so the script can be stopped and started again without losing
# or repeating work.
#
#   ./run_main.sh            # run it
#   ./run_main.sh --dry-run  # print the commands and stop
#   GPU=1 ./run_main.sh      # use a different device
set -euo pipefail

SEEDS=0,1,2,3,4
BLOCKS=(add_4 add_12 add_20 add_24)
GPU=${GPU:-2}
THREADS=${THREADS:-8}

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="echo"

mkdir -p grids logs

for block in "${BLOCKS[@]}"; do
  for objective in case1 case2; do

    # One entry reads and reconstructs the same layer. Two read the first and
    # reconstruct the second, which is the other model at the same depth.
    if [[ "$objective" == "case1" ]]; then
      layers="[\"A@${block}\"]"
    else
      layers="[\"A@${block}\", \"B@${block}\"]"
    fi

    config="configs/${objective}_main.yaml"
    out="grids/main_${objective}_${block}.json"
    logfile="logs/main_${objective}_${block}.log"

    echo "=== ${objective} at ${block} -> ${out}"

    $DRY env OMP_NUM_THREADS="$THREADS" CUDA_VISIBLE_DEVICES="$GPU" \
      nice -n 10 python scripts/grid.py \
        --config "$config" \
        --axis "train.seed=${SEEDS}" \
        --set "activations.layers=${layers}" \
        --out "$out" \
      2>&1 | tee -a "$logfile"
  done
done

echo
echo "all eight configurations finished"