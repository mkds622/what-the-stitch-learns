#!/bin/bash
# The full case 1 and case 2 programme: train, tabulate, draw, compare.
#
# Four depths, two objectives, five seeds, then a dictionary comparison at each
# depth. Every stage is skipped if its output already exists, so the script can
# be stopped and started again without losing or repeating work.
#
#   ./run_full.sh case1              # train case 1 only
#   ./run_full.sh case2              # train case 2 only
#   ./run_full.sh report             # tables and per-arm figures
#   ./run_full.sh compare            # the dictionary comparisons
#   ./run_full.sh all                # everything, in order
#   ./run_full.sh all --dry-run      # print the commands and stop
#
# Case 1 and case 2 are separate stages so they can be run at the same time on
# two cards, which halves the wall clock:
#
#   GPU=2 ./run_full.sh case1 &
#   GPU=3 ./run_full.sh case2 &
#
# The comparison stage needs both arms finished at a depth before it can read
# that depth.
set -euo pipefail

STAGE="${1:-all}"
DRY=""
[[ "${2:-}" == "--dry-run" ]] && DRY="echo"

SEEDS=0,1,2,3,4
BLOCKS=(add_4 add_12 add_20 add_24)
GPU=${GPU:-2}
THREADS=${THREADS:-8}

# The metrics the tables are written for. One table per metric per grid, each
# showing the five seeds and their spread.
METRICS=(
  eval_explained_variance
  eval_l0
  eval_l0_frac
  eval_effective_expansion
  eval_effective_features
  eval_frequency_gini
  eval_max_feature_frequency
  eval_dead_or_dormant
  eval_dead_or_dormant_wider
  train_dead
  baseline_ratio
  agreement
  gate_passed
)

mkdir -p grids logs comparisons/main2
mkdir -p figures/main2/case1 figures/main2/case2 figures/main2/both figures/main2/comparison

layers_for() {
  # One entry reads and reconstructs the same layer. Two read the first and
  # reconstruct the second, which is the other model at the same depth.
  if [[ "$1" == "case1" ]]; then
    echo "[\"A@${2}\"]"
  else
    echo "[\"A@${2}\", \"B@${2}\"]"
  fi
}

# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------

train_arm() {
  local objective="$1"
  for block in "${BLOCKS[@]}"; do
    local out="grids/main2_${objective}_${block}.json"
    local logfile="logs/main2_${objective}_${block}.log"

    echo "=== train ${objective} at ${block} -> ${out}"

    $DRY env OMP_NUM_THREADS="$THREADS" CUDA_VISIBLE_DEVICES="$GPU" \
      nice -n 10 python scripts/grid.py \
        --config "configs/${objective}_main.yaml" \
        --axis "train.seed=${SEEDS}" \
        --set "activations.layers=$(layers_for "$objective" "$block")" \
        --out "$out" \
      2>&1 | tee -a "$logfile"
  done
}

# ----------------------------------------------------------------------
# tables and per-arm figures
# ----------------------------------------------------------------------

report_arm() {
  local objective="$1"
  local dir="figures/main2/${objective}"

  for block in "${BLOCKS[@]}"; do
    local grid="grids/main2_${objective}_${block}.json"
    [[ -f "$grid" ]] || { echo "skipping ${grid}, not trained yet"; continue; }

    # One file per depth holding every metric's table, so a depth can be read
    # in one place.
    local tables="${dir}/tables_${block}.txt"
    echo "=== tables ${objective} ${block} -> ${tables}"
    $DRY rm -f "$tables"
    for metric in "${METRICS[@]}"; do
      $DRY sh -c "echo '--- ${metric}' >> '${tables}'"
      $DRY sh -c "python scripts/grid.py --results '${grid}' --metric '${metric}' >> '${tables}' 2>&1"
    done

    # Reconstruction against sparsity, one point per seed. Neither number
    # means anything alone, so the pair is what gets looked at.
    echo "=== tradeoff ${objective} ${block}"
    $DRY python scripts/plot.py \
      --results "$grid" --tradeoff \
      --out "${dir}/tradeoff_${block}.png"
  done

  local grids=()
  for block in "${BLOCKS[@]}"; do
    [[ -f "grids/main2_${objective}_${block}.json" ]] \
      && grids+=("grids/main2_${objective}_${block}.json")
  done
  [[ ${#grids[@]} -gt 0 ]] || return

  # The four depths on one pair of axes. Everything above is about seeds at
  # one depth; these are the figures the four-depth design exists for. One
  # per metric, four points, each carrying its five seeds as the spread.
  for metric in "${METRICS[@]}"; do
    echo "=== ${metric} across depths, ${objective}"
    $DRY python scripts/plot.py \
      --results "${grids[@]}" \
      --x depth --metric "$metric" --linear-x \
      --out "${dir}/by_depth_${metric}.png"
  done

  # Coloured by depth, so the four blocks are told apart rather than drawn as
  # one undifferentiated cloud.
  echo "=== tradeoff across depths, ${objective}"
  $DRY python scripts/plot.py \
    --results "${grids[@]}" --tradeoff --label-by depth \
    --out "${dir}/tradeoff_all_depths.png"
}

# ----------------------------------------------------------------------
# both arms on one pair of axes
# ----------------------------------------------------------------------

report_both() {
  local dir="figures/main2/both"
  local grids=()
  for objective in case1 case2; do
    for block in "${BLOCKS[@]}"; do
      [[ -f "grids/main2_${objective}_${block}.json" ]] \
        && grids+=("grids/main2_${objective}_${block}.json")
    done
  done
  [[ ${#grids[@]} -gt 0 ]] || return

  # One line per objective, each carrying its own seed spread. Everything
  # above draws the two arms into separate files, where the distance between
  # them is the one thing that cannot be read. This is where it is drawn.
  for metric in "${METRICS[@]}"; do
    echo "=== ${metric} across depths, both arms"
    $DRY python scripts/plot.py \
      --results "${grids[@]}" \
      --x depth --metric "$metric" --linear-x --split-by objective \
      --out "${dir}/by_depth_${metric}.png"
  done

  echo "=== tradeoff, both arms"
  $DRY python scripts/plot.py \
    --results "${grids[@]}" --tradeoff --label-by objective \
    --out "${dir}/tradeoff_all.png"
}

# ----------------------------------------------------------------------
# the dictionary comparison
# ----------------------------------------------------------------------

compare_depth() {
  local block="$1"
  local one="grids/main2_case1_${block}.json"
  local two="grids/main2_case2_${block}.json"

  if [[ ! -f "$one" || ! -f "$two" ]]; then
    echo "skipping ${block}, both arms must be trained first"
    return
  fi

  # The configuration supplies the dataset and the backbone. The depth comes
  # from the runs themselves, so one configuration serves every depth and
  # there is no way to compare at the wrong one.
  local out="comparisons/main2/${block}.json"
  local figures="figures/main2/comparison/${block}"

  echo "=== compare ${block} -> ${out}"
  $DRY env CUDA_VISIBLE_DEVICES="$GPU" python scripts/compare.py \
    --from-grid "$one" "$two" \
    --out "$out" \
    --reading encoder \
    --firing \
    --config configs/case1_main.yaml \
    --device cuda \
    --figures "$figures" \
    --title "block ${block}" \
    2>&1 | tee -a "logs/main2_compare_${block}.log"
}

# ----------------------------------------------------------------------

case "$STAGE" in
  case1)   train_arm case1 ;;
  case2)   train_arm case2 ;;
  report)  report_arm case1; report_arm case2; report_both ;;
  compare) for block in "${BLOCKS[@]}"; do compare_depth "$block"; done ;;
  all)
    train_arm case1
    train_arm case2
    report_arm case1
    report_arm case2
    report_both
    for block in "${BLOCKS[@]}"; do compare_depth "$block"; done
    ;;
  *)
    echo "usage: $0 {case1|case2|report|compare|all} [--dry-run]" >&2
    exit 2
    ;;
esac

echo
echo "stage ${STAGE} finished"