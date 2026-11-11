#!/usr/bin/env bash
# Reference-beta ("beta-fixed") variant of the frequency-oracle FT-Transformer, on the lab server.
# Method: attacker_detector/reference_beta.py. Train and validate only on complete runs at one attacker
# share (source beta), pick one threshold on validation F1 re-weighted to that share, freeze it, and test
# on the unchanged held-out runs (every share of the sweep), with F1 and IGR.
#
#   nohup bash scripts/run_beta_fixed.sh all > /dev/null 2>&1 &
#   tail -f results/beta_fixed_logs/run_beta_fixed.log
#
# Experiment matrix (results root <- data root, source beta = threshold reference beta):
#   oat_v3_n500k_bfix          <- output/oat_v3_n500k            0.05  (MGA-A main sweep; test beta 0.01-0.20)
#   oat_apa_n500k_bfix         <- output/oat_apa_n500k           0.05  (APA main sweep; test beta 0.01-0.20)
#   oat_b010_mga-a_n500k_bfix  <- output/oat_b010_mga-a_n500k    0.10
#   oat_b015_mga-a_n500k_bfix  <- output/oat_b015_mga-a_n500k    0.15
# Every protocol found under the data root, every dataset, in-distribution and cross-dataset (train on the
# other two). Nothing existing is overwritten: new validation runs go to output/<data root>_val/, models and
# results to results/<root>_bfix/, logs to results/beta_fixed_logs/.
#
# Utility-selected thresholds for the trained models (no retraining; attacker_detector/utility_threshold.py):
#   nohup bash scripts/run_beta_fixed.sh utility > /dev/null 2>&1 &      # = check uval uselect ueval igr pack
#
# Stages (`all` = check val train igr pack; `utility` = check uval uselect ueval igr pack):
#   check  unit tests of the variant and of IGR (CPU, a few minutes)
#   val    generate the validation source runs: one new complete run (n = 500,000) per eligible config,
#          at the source beta, replicate indices from 1000 on (no seed shared with the main dataset)
#   train  one model per (root, protocol, dataset, in_dist|cross), JOBS at a time spread over the GPUs;
#          a finished model is skipped, a half-finished one is redone in the same folder
#   igr    IGR (and F1) of every model on re-simulated held-out runs at its frozen threshold, plus no
#          defence, perfect removal and Diffstats (scripts/igr/run_igr_server.sh run collect)
#   pack   results/beta_fixed_results.tar.gz: CSVs, thresholds, manifests, checkpoints (no predictions)
#   uval     utility-validation runs: UVAL_REPS new complete runs per eligible config at the source beta
#            (replicates 2000+), stored compactly in results/<root>/utility_val/ (scripts/igr/run_igr.py uval)
#   uselect  per model: the threshold with the lowest mean IGR^2 on its utility-validation runs (source
#            datasets only) -> threshold_utility.json next to threshold.json (which is kept)
#   ueval    F1 sweep + 60,000-row test at threshold_utility.json from the stored held-out scores ->
#            test_results_utility.csv, sensitivity_test_results_utility.csv (no retraining, no features)
#   igr      (as above) now scores both frozen thresholds on the same re-simulated users; the previous
#            igr_runs.csv is kept as igr_runs_f1threshold.csv. IGR_CACHE=all (default) keeps the held-out
#            runs (~35 GB) so later thresholds can be scored without simulating again
#
# Env: PY, JOBS (parallel trainings / evaluations, default 6), GPUS (default: all, e.g. "0 1 2"),
#      SHARDS (simulation processes, default 16), PROC (threads each, default 4), UVAL_REPS (default 1),
#      IGR_CACHE (all|none, default all), ROOTS (subset of the four, space-separated), SKIP_TESTS=1.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CONDA_BASE=${CONDA_BASE:-/home/boby/miniconda3}
PY=${PY:-$CONDA_BASE/envs/attacker-detector/bin/python}
JOBS=${JOBS:-6}
SHARDS=${SHARDS:-16}
PROC=${PROC:-4}
UVAL_REPS=${UVAL_REPS:-1}
IGR_CACHE=${IGR_CACHE:-all}
LOGS=results/beta_fixed_logs
mkdir -p "$LOGS"
exec > >(tee -a "$LOGS/run_beta_fixed.log") 2>&1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl_$USER} PYTHONDONTWRITEBYTECODE=1
say () { echo "[$(date '+%F %T')] $*"; }

declare -A DATA=( [oat_v3_n500k_bfix]=output/oat_v3_n500k [oat_apa_n500k_bfix]=output/oat_apa_n500k
                  [oat_b010_mga-a_n500k_bfix]=output/oat_b010_mga-a_n500k
                  [oat_b015_mga-a_n500k_bfix]=output/oat_b015_mga-a_n500k )
declare -A BETA=( [oat_v3_n500k_bfix]=0.05 [oat_apa_n500k_bfix]=0.05
                  [oat_b010_mga-a_n500k_bfix]=0.10 [oat_b015_mga-a_n500k_bfix]=0.15 )
ROOTS=(${ROOTS:-oat_v3_n500k_bfix oat_apa_n500k_bfix oat_b010_mga-a_n500k_bfix oat_b015_mga-a_n500k_bfix})
DATASETS=(zipf emoji fire)
if [[ -z ${GPUS:-} ]]; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
fi
GPU_LIST=($GPUS)

# the original runs' FT-Transformer settings (scripts/run_extra_eps.sh), feature set v3, seed 42
FT=(-m ft_transformer --train-samples 500000 --test-samples 60000 --val-size 0.10 --balanced-batches
    --k-folds 0 --epochs 50 --patience 10 --batch-size 256 --lr 0.0005 --d-token 64 --n-heads 8
    --n-layers 3 --ffn-d-multiplier 1.33 --attention-dropout 0.1 --residual-dropout 0.0
    --seed 42 --no-plot --save-predictions --feature-set v3)

protocols_of () { local d; for d in "$1"/*/; do [[ -f $d/design.json ]] && basename "$d"; done; }

stage_check () {
  [[ ${SKIP_TESTS:-0} == 1 ]] && { say "check: skipped (SKIP_TESTS=1)"; return 0; }
  say "check: unit tests"
  "$PY" -m unittest discover -s tests/reference_beta -t . > "$LOGS/tests_reference_beta.log" 2>&1 \
    && "$PY" -m unittest discover -s tests/igr -t . > "$LOGS/tests_igr.log" 2>&1 \
    || { tail -30 "$LOGS"/tests_*.log; say "tests FAILED -- see $LOGS/tests_*.log"; return 1; }
  grep -h "^Ran\|^OK" "$LOGS"/tests_*.log
}

stage_val () {
  local W W_OLH root data P D V args WP OLH
  W=$(( $(nproc) / 2 )); (( W > 24 )) && W=24; (( W < 1 )) && W=1
  W_OLH=$(( W > 16 ? 16 : W ))
  for root in "${ROOTS[@]}"; do
    data=${DATA[$root]}
    for P in $(protocols_of "$data"); do
      D=$data/$P; V=${data}_val/$P
      if [[ -f $V/features.npy && -f $V/config.npy && ! -f $V/config.bin ]]; then say "val: skip $V"; continue; fi
      args=$("$PY" -m attacker_detector.reference_beta gen-args "$D" "${BETA[$root]}") || return 1
      WP=$W; OLH=(); [[ $P == OLH* ]] && { WP=$W_OLH; OLH=(--olh-parallel); }
      mkdir -p "$V"; rm -f "$V"/*.bin
      say "val: $V  (beta ${BETA[$root]}, workers $WP)"
      # shellcheck disable=SC2086
      "$PY" generate_dataset.py -o "$V" --protocols "$P" $args --workers "$WP" --inner-processors 2 \
          --tasks-per-worker 2 "${OLH[@]}" --save-every 10 > "$LOGS/val_$(basename "$data")_$P.log" 2>&1 \
        || { say "val FAILED: $V -- see $LOGS/val_$(basename "$data")_$P.log"; return 1; }
      grep -h "Successful runs\|Failed runs\|Total users" "$LOGS/val_$(basename "$data")_$P.log" | sed 's/^/    /'
      grep -q "Failed runs: 0" "$LOGS/val_$(basename "$data")_$P.log" || { say "val: failed runs in $V"; return 1; }
    done
  done
}

job_list () {   # one line per model: out_dir | log | main.py arguments
  local root data P D Dc TR X
  for root in "${ROOTS[@]}"; do
    data=${DATA[$root]}
    for P in $(protocols_of "$data"); do
      for D in "${DATASETS[@]}"; do
        Dc=${D^}
        TR=(); for X in "${DATASETS[@]}"; do [[ $X != "$D" ]] && TR+=("$X"); done
        echo "results/$root/in_dist/${Dc}_$P|$LOGS/${root}_in_${Dc}_$P.log|-d $data/$P --val-data ${data}_val/$P --source-beta ${BETA[$root]} --protocol $P --dataset $D"
        echo "results/$root/cross/${P}_${Dc}_test|$LOGS/${root}_cross_${P}_$Dc.log|-d $data/$P --val-data ${data}_val/$P --source-beta ${BETA[$root]} --protocol $P --training-method cross --train-dataset ${TR[*]} --test-dataset $D"
      done
    done
  done
}

train_one () {   # $1 = job line, $2 = GPU
  local out log rest
  IFS='|' read -r out log rest <<< "$1"
  if grep -qs '"status": "complete"' "$out/variant.json"; then echo "  skip $out"; return 0; fi
  echo "  [$(date '+%T')] train $out (GPU ${2:-cpu})"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$2 "$PY" main.py $rest "${FT[@]}" -o "$out" > "$log" 2>&1 \
    || { echo "  FAILED $out -- see $log"; return 1; }
  grep -h "Frozen threshold" "$log" | sed "s|^|    $out: |" | cut -c1-220
}

stage_train () {
  local jobs n i bad=0 pids=() line
  mapfile -t jobs < <(job_list)
  n=${#jobs[@]}
  say "train: $n models, $JOBS at a time, GPUs '${GPU_LIST[*]:-cpu}'"
  for (( i = 0; i < n; i++ )); do
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || bad=1; done
    line=${jobs[$i]}
    gpu=${GPU_LIST[$(( i % (${#GPU_LIST[@]} > 0 ? ${#GPU_LIST[@]} : 1) ))]:-}
    train_one "$line" "$gpu" &
  done
  while (( $(jobs -rp | wc -l) > 0 )); do wait -n || bad=1; done
  local done_n
  done_n=$(grep -ls '"status": "complete"' results/*_bfix/*/*/variant.json 2>/dev/null | wc -l)
  say "train: $done_n complete variant models under results/*_bfix"
  (( bad )) && say "train: some models FAILED -- grep FAILED $LOGS/run_beta_fixed.log; rerun 'train' to retry them"
  return $bad
}

stage_igr () {
  local root
  for root in "${ROOTS[@]}"; do      # keep the F1-threshold-only IGR table before runs are scored again
    if ls results/$root/*/*/threshold_utility.json > /dev/null 2>&1 && [[ -f results/$root/igr/igr_runs.csv \
          && ! -f results/$root/igr/igr_runs_f1threshold.csv ]]; then
      cp results/$root/igr/igr_runs.csv results/$root/igr/igr_runs_f1threshold.csv
    fi
  done
  say "igr: frozen-threshold models + no defence + perfect removal + Diffstats on re-simulated held-out runs"
  PY=$PY ROOTS="${ROOTS[*]}" BASELINES=diffstats SHARDS=$SHARDS PROC=$PROC SKIP_TESTS=1 IGR_ARGS="--cache $IGR_CACHE" \
    bash scripts/igr/run_igr_server.sh plan run collect
}

stage_uval () {
  local i pids=() bad=0
  say "uval: $UVAL_REPS utility-validation run(s) per eligible config, $SHARDS shards x $PROC threads"
  for (( i = 0; i < SHARDS; i++ )); do
    "$PY" scripts/igr/run_igr.py uval --roots "${ROOTS[@]}" --uval-replicates "$UVAL_REPS" --shard "$i" \
        --num-shards "$SHARDS" --processors "$PROC" > "$LOGS/uval_shard$i.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { bad=1; say "  uval shard $i had failures -- grep -A5 FAILED $LOGS/uval_shard$i.log"; }
  done
  say "uval: $(cat "$LOGS"/uval_shard*.log | grep -c 'no-defence IGR') runs simulated in this pass"
  return $bad
}

stage_uselect () {
  say "uselect: mean-IGR^2 threshold per model on its utility-validation runs"
  CUDA_VISIBLE_DEVICES=${GPU_LIST[0]:-} "$PY" scripts/igr/run_igr.py uselect --roots "${ROOTS[@]}" \
      > "$LOGS/uselect.log" 2>&1
  local rc=$?
  grep -E "utility threshold|FAILED|uselect:" "$LOGS/uselect.log" | cut -c1-230
  return $rc
}

eval_one () {   # $1 = model dir
  local d=$1
  if [[ -f $d/sensitivity_test_results_utility.csv && $d/sensitivity_test_results_utility.csv -nt $d/threshold_utility.json ]]; then
    echo "  skip $d"; return 0
  fi
  "$PY" -m attacker_detector.reference_beta evaluate "$d" threshold_utility.json utility \
      > "$LOGS/ueval_$(echo "$d" | tr '/' '_').log" 2>&1 || { echo "  FAILED $d"; return 1; }
  tail -1 "$LOGS/ueval_$(echo "$d" | tr '/' '_').log" | cut -c1-200
}

stage_ueval () {
  local d bad=0 dirs=() root
  for root in "${ROOTS[@]}"; do
    for d in results/$root/in_dist/* results/$root/cross/*; do
      [[ -f $d/threshold_utility.json ]] && dirs+=("$d")
    done
  done
  say "ueval: ${#dirs[@]} models at threshold_utility.json, $JOBS at a time"
  for d in "${dirs[@]}"; do
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || bad=1; done
    eval_one "$d" &
  done
  while (( $(jobs -rp | wc -l) > 0 )); do wait -n || bad=1; done
  return $bad
}

stage_pack () {
  local files
  files=$(find results/*_bfix -type f ! -name 'test_predictions.npz' ! -path '*/igr/cache/*' \
              ! -path '*/igr/runs/*' ! -name '*.lock' ! -path '*/utility_val/*.npz' ! -name '*.tmp*' \
              2>/dev/null)
  [[ -z $files ]] && { say "pack: nothing to pack"; return 1; }
  # shellcheck disable=SC2086
  tar czf results/beta_fixed_results.tar.gz $files $LOGS/*.log
  say "pack: results/beta_fixed_results.tar.gz ($(du -h results/beta_fixed_results.tar.gz | cut -f1)) -- on your machine:"
  say "  scp <server>:$(pwd)/results/beta_fixed_results.tar.gz results/ && tar xzf results/beta_fixed_results.tar.gz"
  say "  python scripts/plots/make_beta_fixed_figures.py"
}

STAGES=("$@")
(( ${#STAGES[@]} )) || { sed -n '2,48p' "$0"; exit 1; }
[[ ${STAGES[*]} == all ]] && STAGES=(check val train igr pack)
[[ ${STAGES[*]} == utility ]] && STAGES=(check uval uselect ueval igr pack)
for s in "${STAGES[@]}"; do
  case $s in check|val|train|igr|pack|uval|uselect|ueval) ;; *) echo "unknown stage: $s"; exit 1 ;; esac
done
say "=== run_beta_fixed.sh ${STAGES[*]}  (PY=$PY, roots: ${ROOTS[*]})"
for s in "${STAGES[@]}"; do
  t0=$SECONDS
  "stage_$s" || { say "stage $s FAILED"; exit 1; }
  say "stage $s done in $(( (SECONDS - t0) / 60 )) min"
done
say "=== done"
