#!/usr/bin/env bash
# Threshold and scaling study on the lab server (method: attacker_detector/threshold_scaling_study.py).
#   1. pooled-beta checkpoints (reused) with a utility-selected threshold
#   2. fixed beta: four features (control), three features, three features without per-run median/IQR,
#      four features without per-run median/IQR (excess_share kept)
#      pooled beta trained here: four features (control) and four features without per-run median/IQR
#      small hyperparameter test: the two unscaled four-feature models with dropout 0.1 + batch 64 and with
#      token size 128, on OUE and OLH-Server, zipf, in-distribution only (16 models)
#   3. every model and threshold scored on the same simulated held-out users -> comparison tables
#
#   bash scripts/run_threshold_scaling_study.sh all           # foreground; stage progress is printed and logged
#   tail -3 results/threshold_scaling_study_runs.log          # the run directory of this invocation (or why it did not start)
#   tail -f <run directory>/logs/*_shard*.log                 # per-run and per-model progress (second terminal)
#
# Every invocation writes to a NEW directory results/<RUN_NAME> (default threshold_and_scaling_study; no
# timestamp) and stops if it exists: give a later invocation another RUN_NAME. Existing result folders are only read.
# An interrupted invocation is continued with RESUME=<its run directory> and the remaining stages
# (finished runs, models and thresholds are skipped):
#   RESUME=results/threshold_and_scaling_study bash scripts/run_threshold_scaling_study.sh simulate train select score table pack
# Model variants added to the code later are added to a finished run directory with the extend stage: only
# the runs and models they need are simulated and trained (42 models per variant for both roots; the
# pooled-beta variants need 273 more runs), every model is scored again, tables and the tarball are redone:
#   RESUME=results/threshold_and_scaling_study bash scripts/run_threshold_scaling_study.sh extend simulate train select score table pack
#
# Stages (`all` = check init plan simulate train select score table pack):
#   check     unit tests of the study, of the fixed-beta variant and of IGR (CPU, a few minutes)
#   init      create the run directory; find and verify the reused checkpoints (sha256 against
#             scripts/threshold_scaling_study/reference/reused.json); nothing is created if one is missing
#   extend    (RESUME only) add the trained variants the run directory does not have yet (VARIANTS limits
#             them); stops if a simulated run is missing
#   plan      runs to simulate per root, role and protocol
#   simulate  every training, validation, utility-validation and held-out run, once (966 runs for both
#             roots, 1239 with the pooled-beta variants); SHARDS processes x PROC threads; Diffstats is
#             flagged on the held-out runs
#   train     the fixed-beta feature variants, one model per root, protocol, dataset and kind
#             (42 models per variant for both roots); JOBS processes on the GPUs in GPUS
#   select    utility threshold (mean IGR^2 on utility-validation runs) of the pooled checkpoints and of the
#             new models
#   score     every model at every threshold on the held-out runs
#   table     <run directory>/comparison/{side_by_side,in_dist,cross}/
#   pack      results/<RUN_NAME>.tar.gz without the simulated runs
#
# Env: PY, GPUS (default "2"), JOBS (default 4), SHARDS (default 8), PROC (default 4),
#      ROOTS (default "oat_v3_n500k oat_apa_n500k"), BASELINES (default "diffstats"; "diffstats fiad" adds
#      FIAD on the held-out runs, slow), RUN_NAME, RESUME, VARIANTS (extend), SKIP_TESTS=1.
#      Tests only: STUDY_RESULTS, STUDY_CONFIG.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CONDA_BASE=${CONDA_BASE:-/home/boby/miniconda3}
PY=${PY:-$CONDA_BASE/envs/attacker-detector/bin/python}
GPUS=${GPUS-2}                       # GPUS= (empty) runs on the CPU
JOBS=${JOBS:-4}
SHARDS=${SHARDS:-8}
PROC=${PROC:-4}
BASELINES=${BASELINES-diffstats}
GPU_LIST=($GPUS)
RES=${STUDY_RESULTS:-results}       # where existing results live and the run directory is made (tests override it)
RUNS_LOG=$RES/threshold_scaling_study_runs.log
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl_$USER} PYTHONDONTWRITEBYTECODE=1
say () { echo "[$(date '+%F %T')] $*"; }
study () { "$PY" -m attacker_detector.threshold_scaling_study "$@" --run-dir "$RUN_DIR"; }

STAGES=("$@")
(( ${#STAGES[@]} )) || { sed -n '2,46p' "$0"; exit 1; }
if [[ -n ${RESUME:-} ]]; then
  RUN_DIR=${RESUME%/}
  [[ -f $RUN_DIR/study_config.json ]] || { echo "RESUME=$RUN_DIR is not a run directory of this study"; exit 1; }
  [[ ${STAGES[*]} == all ]] && STAGES=(plan simulate train select score table pack)
  for s in "${STAGES[@]}"; do [[ $s == init ]] && { echo "init creates a new directory; drop it when resuming"; exit 1; }; done
else
  RUN_DIR=$RES/${RUN_NAME:-threshold_and_scaling_study}
  [[ -e $RUN_DIR ]] && { echo "$RUN_DIR already exists; choose another RUN_NAME or continue it with RESUME=$RUN_DIR"; exit 1; }
  [[ ${STAGES[*]} == all ]] && STAGES=(check init plan simulate train select score table pack)
  [[ " ${STAGES[*]} " == *" init "* ]] || { echo "a new invocation needs the init stage (or RESUME=<run directory>)"; exit 1; }
  [[ " ${STAGES[*]} " == *" extend "* ]] && { echo "extend adds to an existing run directory: RESUME=<run directory>"; exit 1; }
fi
for s in "${STAGES[@]}"; do
  case $s in check|init|extend|plan|simulate|train|select|score|table|pack) ;; *) echo "unknown stage: $s"; exit 1 ;; esac
done
TARBALL=$RES/$(basename "$RUN_DIR").tar.gz

failures () {   # the failed tests of a unittest log with their tracebacks (they stand before the stages' own
                # output, so the end of the log does not show them); the end of the log if there are none
  awk '/^(FAIL|ERROR): /{p=1} /^Ran /{p=0} p' "$1" | grep . || tail -40 "$1"
  grep -h "^Ran\|^FAILED" "$1"
}

stage_check () {
  [[ ${SKIP_TESTS:-0} == 1 ]] && { say "check: skipped (SKIP_TESTS=1)"; return 0; }
  local t log; log=$(mktemp)
  for t in tests/threshold_scaling tests/reference_beta tests/igr; do
    say "check: $t"
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]:-} "$PY" -m unittest discover -s "$t" -t . > "$log" 2>&1 || { failures "$log"; say "tests FAILED ($t)"; return 1; }
    grep -h "^Ran\|^OK" "$log"
  done
  rm -f "$log"
}

stage_init () {
  # shellcheck disable=SC2086
  study init --results "$RES" ${ROOTS:+--roots $ROOTS} ${STUDY_CONFIG:+--config $STUDY_CONFIG} --baselines $BASELINES || return 1
  echo "$(date '+%F %T') run directory: $RUN_DIR" >> "$RUNS_LOG"
}

# shellcheck disable=SC2086
stage_extend () { study extend ${VARIANTS:+--variants $VARIANTS}; }

stage_plan () { study plan; }

shards () {   # $1 = stage, $2 = processes, $3 = "gpu" to spread them over GPUS, rest = extra arguments
  local stage=$1 k=$2 gpu=$3 i pids=() bad=0; shift 3
  for (( i = 0; i < k; i++ )); do
    if [[ $gpu == gpu ]]; then
      CUDA_VISIBLE_DEVICES=${GPU_LIST[$(( i % (${#GPU_LIST[@]} > 0 ? ${#GPU_LIST[@]} : 1) ))]:-} \
        study "$stage" --shard "$i" --num-shards "$k" "$@" > "$RUN_DIR/logs/${stage}_shard$i.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES= study "$stage" --shard "$i" --num-shards "$k" "$@" > "$RUN_DIR/logs/${stage}_shard$i.log" 2>&1 &
    fi
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { bad=1; say "  $stage shard $i had failures -- grep -A8 FAILED $RUN_DIR/logs/${stage}_shard$i.log"; }
  done
  return $bad
}

stage_simulate () {
  say "simulate: $SHARDS shards x $PROC threads (logs: $RUN_DIR/logs/simulate_shard*.log)"
  shards simulate "$SHARDS" cpu --processors "$PROC"; local rc=$?
  say "simulate: $(find "$RUN_DIR/simulated_runs" -name '*.npz' ! -name '*.tmp.npz' | wc -l) runs stored"
  (( rc )) && say "simulate: some runs failed; run the stage again (RESUME=$RUN_DIR), finished runs are skipped"
  return $rc
}

stage_train () {
  say "train: $JOBS processes on GPU(s) '${GPU_LIST[*]:-cpu}' (logs: $RUN_DIR/logs/train_shard*.log)"
  shards train "$JOBS" gpu; local rc=$?
  say "train: $(grep -ls '"status": "complete"' "$RUN_DIR"/variants/*/*/*/*/variant.json 2>/dev/null | wc -l) models complete"
  return $rc
}

stage_select () {
  say "select: utility thresholds on the utility-validation runs"
  shards select "$JOBS" gpu; local rc=$?
  say "select: $(find "$RUN_DIR/variants" -name threshold_utility.json | wc -l) utility thresholds"
  return $rc
}

stage_score () {
  say "score: every model and threshold on the held-out runs"
  shards score "$JOBS" gpu
}

stage_table () { study table 2>&1 | tee "$RUN_DIR/logs/table.log"; return "${PIPESTATUS[0]}"; }

stage_pack () {
  tar czf "$TARBALL.tmp" -C "$RES" --exclude="$(basename "$RUN_DIR")/simulated_runs" "$(basename "$RUN_DIR")" \
    && mv "$TARBALL.tmp" "$TARBALL" || return 1
  say "pack: $TARBALL ($(du -h "$TARBALL" | cut -f1)) -- on your machine, from the repository root:"
  say "  scp <server>:$(cd "$RES" && pwd)/$(basename "$TARBALL") results/ && tar xzf results/$(basename "$TARBALL") -C results"
}

echo "=== run_threshold_scaling_study.sh ${STAGES[*]}  run directory: $RUN_DIR"
PRE=$(mktemp)                              # output of check and init, until the run directory exists
for s in "${STAGES[@]}"; do
  t0=$SECONDS
  if [[ -d $RUN_DIR/logs ]]; then
    { say "--- stage $s"; "stage_$s"; } 2>&1 | tee -a "$RUN_DIR/logs/launcher.log"; rc=${PIPESTATUS[0]}
  else
    { say "--- stage $s"; "stage_$s"; } 2>&1 | tee -a "$PRE"; rc=${PIPESTATUS[0]}
    [[ -d $RUN_DIR/logs ]] && { cat "$PRE" >> "$RUN_DIR/logs/launcher.log"; : > "$PRE"; }
  fi
  if (( rc )); then
    if [[ -d $RUN_DIR/logs ]]; then
      say "stage $s FAILED" | tee -a "$RUN_DIR/logs/launcher.log"
    else                                   # nothing was created: keep the reason where it can be found
      mkdir -p "$RES"; { echo "$(date '+%F %T') stage $s FAILED before a run directory existed:"; cat "$PRE"; } >> "$RUNS_LOG"
    fi
    rm -f "$PRE"; echo "stage $s FAILED (run directory: $RUN_DIR; see $RUNS_LOG or $RUN_DIR/logs/launcher.log)"; exit 1
  fi
  [[ -d $RUN_DIR/logs ]] && say "stage $s done in $(( (SECONDS - t0) / 60 )) min" | tee -a "$RUN_DIR/logs/launcher.log"
done
rm -f "$PRE"
[[ -d $RUN_DIR/logs ]] && say "=== done: $RUN_DIR" >> "$RUN_DIR/logs/launcher.log"
echo "=== done: $RUN_DIR"
