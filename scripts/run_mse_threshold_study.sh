#!/usr/bin/env bash
# MSE threshold study on the lab server (method: attacker_detector/mse_threshold_study.py).
# The cut-off of a detector is chosen on the mean squared error of the target items instead of on IGR.
# No model is trained: the unscaled models of a finished threshold and scaling study are read (SOURCE):
#   fixed beta 4 features, fixed beta 3 features, pooled beta 4 features -- none with per-run median/IQR.
# Run it after that study has those models (its extend ... train stages).
#
#   SOURCE=results/threshold_and_scaling_study bash scripts/run_mse_threshold_study.sh all     # foreground
#   tail -3 results/mse_threshold_study_runs.log              # the run directory of this invocation (or why it did not start)
#   tail -f <run directory>/logs/*_shard*.log                 # per-run and per-model progress (second terminal)
#
# Every invocation writes to a NEW directory results/<RUN_NAME> (default mse_threshold_study; no timestamp)
# and stops if it exists: give a later invocation another RUN_NAME. SOURCE and older result folders are
# only read. An interrupted invocation is continued with RESUME=<its run directory> and the remaining
# stages (finished runs and thresholds are skipped):
#   RESUME=results/mse_threshold_study bash scripts/run_mse_threshold_study.sh simulate select score table pack
#
# Stages (`all` = check init plan simulate select score table pack):
#   check     unit tests of this study and of IGR (CPU, a few minutes)
#   init      create the run directory; check that SOURCE holds every model (sha256 against its
#             variant.json); nothing is created if one is missing
#   plan      runs to simulate per root, role and protocol
#   simulate  the utility-validation and held-out runs, once (588 runs for both roots), stored with each
#             user's support on every target item; SHARDS processes x PROC threads; Diffstats is flagged
#             on the held-out runs
#   select    per model: the MSE threshold and the IGR threshold on the same utility-validation runs
#   score     every model of SOURCE at its frozen thresholds and at the new ones, on the held-out runs
#   table     <run directory>/comparison/{side_by_side,in_dist,cross}/
#   pack      results/<RUN_NAME>.tar.gz without the simulated runs
#
# Env: SOURCE (required for init), PY, GPUS (default "2"), JOBS (default 4), SHARDS (default 8),
#      PROC (default 4), BASELINES (default "diffstats"; "diffstats fiad" adds FIAD on the held-out runs,
#      slow), RUN_NAME, RESUME, SKIP_TESTS=1. Tests only: STUDY_RESULTS.
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
RES=${STUDY_RESULTS:-results}       # where the run directory is made (tests override it)
RUNS_LOG=$RES/mse_threshold_study_runs.log
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl_$USER} PYTHONDONTWRITEBYTECODE=1
say () { echo "[$(date '+%F %T')] $*"; }
study () { "$PY" -m attacker_detector.mse_threshold_study "$@" --run-dir "$RUN_DIR"; }

STAGES=("$@")
(( ${#STAGES[@]} )) || { sed -n '2,33p' "$0"; exit 1; }
if [[ -n ${RESUME:-} ]]; then
  RUN_DIR=${RESUME%/}
  [[ -f $RUN_DIR/study_config.json ]] || { echo "RESUME=$RUN_DIR is not a run directory of this study"; exit 1; }
  [[ ${STAGES[*]} == all ]] && STAGES=(plan simulate select score table pack)
  for s in "${STAGES[@]}"; do [[ $s == init ]] && { echo "init creates a new directory; drop it when resuming"; exit 1; }; done
else
  RUN_DIR=$RES/${RUN_NAME:-mse_threshold_study}
  [[ -e $RUN_DIR ]] && { echo "$RUN_DIR already exists; choose another RUN_NAME or continue it with RESUME=$RUN_DIR"; exit 1; }
  [[ ${STAGES[*]} == all ]] && STAGES=(check init plan simulate select score table pack)
  [[ " ${STAGES[*]} " == *" init "* ]] || { echo "a new invocation needs the init stage (or RESUME=<run directory>)"; exit 1; }
  [[ -n ${SOURCE:-} ]] || { echo "set SOURCE=<run directory of the threshold and scaling study>"; exit 1; }
  [[ -f ${SOURCE%/}/study_config.json ]] || { echo "SOURCE=$SOURCE is not a run directory of the threshold and scaling study"; exit 1; }
fi
for s in "${STAGES[@]}"; do
  case $s in check|init|plan|simulate|select|score|table|pack) ;; *) echo "unknown stage: $s"; exit 1 ;; esac
done
TARBALL=$RES/$(basename "$RUN_DIR").tar.gz

stage_check () {
  [[ ${SKIP_TESTS:-0} == 1 ]] && { say "check: skipped (SKIP_TESTS=1)"; return 0; }
  local t log; log=$(mktemp)
  for t in tests/mse_threshold tests/igr; do
    say "check: $t"
    "$PY" -m unittest discover -s "$t" -t . > "$log" 2>&1 || { tail -40 "$log"; say "tests FAILED ($t)"; return 1; }
    grep -h "^Ran\|^OK" "$log"
  done
  rm -f "$log"
}

stage_init () {
  # shellcheck disable=SC2086
  study init --results "$RES" --source "${SOURCE%/}" --baselines $BASELINES || return 1
  echo "$(date '+%F %T') run directory: $RUN_DIR (models from ${SOURCE%/})" >> "$RUNS_LOG"
}

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

stage_select () {
  say "select: MSE and IGR thresholds on the utility-validation runs (logs: $RUN_DIR/logs/select_shard*.log)"
  shards select "$JOBS" gpu; local rc=$?
  say "select: $(find "$RUN_DIR/thresholds" -name threshold_mse.json | wc -l) MSE thresholds"
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

echo "=== run_mse_threshold_study.sh ${STAGES[*]}  run directory: $RUN_DIR"
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
