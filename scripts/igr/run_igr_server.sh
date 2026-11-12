#!/usr/bin/env bash
# IGR (item gain ratio) comparison on the lab server: our v3 FT-Transformer vs FIAD vs Diffstats, each run
# as "remove the users it flags, then re-estimate" (method: attacker_detector/igr.py). Every stage is
# resumable: re-run the same command after a crash or kill and finished runs are skipped. Existing results
# and datasets are only read; everything new goes to results/<root>/igr/ and results/figures/frequency_oracle/igr/.
#
#   nohup bash scripts/igr/run_igr_server.sh all > /dev/null 2>&1 &     # now: original models, FIAD, Diffstats
#   tail -f results/igr_logs/run_igr_server.log
#   # after Codex (run_wsp_server.sh import_codex, plus its baselines stage for fine-tune / random):
#   bash scripts/igr/run_igr_server.sh wsp collect figures pack          # WSP models on the cached users
#
# Stages (`all` = check plan norms run collect figures pack):
#   check    IGR unit tests (CPU, about a minute)
#   plan     how many LDP runs each root / protocol / dataset has
#   norms    v3 z-score of each training data dir (loads output/<data root>/<PROTOCOL> once, serial)
#   run      re-simulate every held-out run and score every detector on it; SHARDS processes in parallel,
#            PROC threads each (simulation, FIAD, torch). FIAD dominates the time.
#   wsp      WSP Stage-1 (seeds 42-44) and edited / fine-tune / random checkpoints on the cached runs of
#            the cells the WSP runs test on (no simulation, no FIAD; minutes)
#   collect  merge per-run rows -> results/<root>/igr/igr_runs.csv; F1 on the re-simulated runs vs the
#            stored held-out predictions -> igr_f1_check.csv
#   figures  scripts/igr/igr_figures.py -> results/figures/frequency_oracle/igr/
#   pack     results/igr_results.tar.gz (CSVs, designs, figures; no caches) to copy to your machine
#
# Env overrides: PY, SHARDS (default 8), PROC (default 4), IGR_GPU (GPU(s) for our model's forward pass,
#   space-separated, shards spread round-robin, e.g. "0 1 2"; default 0; empty = CPU), ROOTS (space-separated results roots, default the five *_v3f roots),
#   REPLICATES (runs per config; default the datasets' held-out count, 1), IGR_ARGS (extra run_igr.py args),
#   BASELINES (baselines scored next to ours, default "fiad diffstats"; BASELINES=diffstats skips FIAD,
#   which takes most of the run time; BASELINES= scores ours only; a later run that asks for a baseline
#   a run lacks re-simulates that run, so all detectors share users),
#   SKIP_TESTS=1.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
missing=()
for f in attacker_detector/igr.py scripts/igr/run_igr.py scripts/igr/igr_figures.py tests/igr/test_igr.py \
         attacker_detector/baselines.py; do
  [[ -e $f ]] || missing+=("$f")
done
if (( ${#missing[@]} )); then
  echo "run_igr_server.sh: checkout at $(pwd) is missing files:"; printf '  %s\n' "${missing[@]}"
  echo "fix: commit + push them from your machine, then 'git pull' here"; exit 1
fi

CONDA_BASE=${CONDA_BASE:-/home/boby/miniconda3}
if [[ -z ${PY:-} ]]; then
  PY=$CONDA_BASE/envs/attacker-detector-wsp/bin/python
  [[ -x $PY ]] || PY=$CONDA_BASE/envs/attacker-detector/bin/python
fi
SHARDS=${SHARDS:-8}
PROC=${PROC:-4}
IGR_GPU=${IGR_GPU-0}
LOGS=results/igr_logs
mkdir -p "$LOGS"
exec > >(tee -a "$LOGS/run_igr_server.log") 2>&1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl_$USER} PYTHONDONTWRITEBYTECODE=1

ARGS=()
[[ -n ${ROOTS:-} ]] && ARGS+=(--roots $ROOTS)
[[ -n ${REPLICATES:-} ]] && ARGS+=(--replicates "$REPLICATES")
[[ -n ${BASELINES+set} ]] && ARGS+=(--baselines $BASELINES)
[[ -n ${IGR_ARGS:-} ]] && ARGS+=($IGR_ARGS)

say () { echo "[$(date '+%F %T')] $*"; }
igr () { "$PY" scripts/igr/run_igr.py "$@" ${ARGS[@]+"${ARGS[@]}"}; }

stage_check () {
  [[ ${SKIP_TESTS:-0} == 1 ]] && { say "check: tests skipped (SKIP_TESTS=1)"; return 0; }
  say "check: IGR unit tests ($PY)"
  "$PY" -m unittest discover -s tests/igr -t . > "$LOGS/tests.log" 2>&1 \
    || { tail -30 "$LOGS/tests.log"; say "tests FAILED -- see $LOGS/tests.log"; return 1; }
  tail -3 "$LOGS/tests.log"
}

stage_plan () { igr plan; }

stage_norms () {
  say "norms: v3 z-score per training data dir (serial; each loads one dataset dir)"
  igr norms
}

stage_run () {
  local free; free=$(free -g | awk '/^Mem:/{print $7}')
  local gpus=($IGR_GPU)          # shards are spread round-robin over the listed GPUs
  say "run: $SHARDS shards x $PROC threads, GPU(s) '${IGR_GPU:-cpu}', ${free} GB RAM available now"
  local pids=() i
  for (( i = 0; i < SHARDS; i++ )); do
    CUDA_VISIBLE_DEVICES=${gpus[$(( i % (${#gpus[@]} > 0 ? ${#gpus[@]} : 1) ))]:-} \
      nohup "$PY" scripts/igr/run_igr.py run --shard "$i" --num-shards "$SHARDS" \
      --processors "$PROC" ${ARGS[@]+"${ARGS[@]}"} > "$LOGS/run_shard$i.log" 2>&1 &
    pids+=($!)
  done
  say "  shard logs: $LOGS/run_shard*.log   (progress: grep -c 'sim ' $LOGS/run_shard*.log)"
  local bad=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { bad=1; say "  shard $i had failed runs -- grep -A5 FAILED $LOGS/run_shard$i.log"; }
  done
  (( bad )) && say "run: some runs failed; re-run 'run' to retry them (finished runs are skipped)"
  return $bad
}

stage_wsp () {
  say "wsp: WSP models on the cached runs"
  CUDA_VISIBLE_DEVICES=${IGR_GPU%% *} "$PY" scripts/igr/run_igr.py wsp --processors "$PROC" 2>&1 | tee "$LOGS/wsp.log" | tail -25
}

stage_collect () { igr collect; }

stage_figures () {
  say "figures -> results/figures/frequency_oracle/igr/"
  "$PY" scripts/igr/igr_figures.py ${ROOTS:+--roots $ROOTS}
}

stage_pack () {
  local files
  files=$(ls results/*/igr/igr_runs.csv results/*/igr/igr_f1_check.csv results/*/igr/design_*.json \
             results/*/igr/v3_norm_*.json 2>/dev/null)
  [[ -d results/figures/frequency_oracle/igr ]] && files="$files results/figures/frequency_oracle/igr"
  [[ -z $files ]] && { say "pack: nothing collected yet"; return 1; }
  tar czf results/igr_results.tar.gz $files
  say "pack: results/igr_results.tar.gz ($(du -h results/igr_results.tar.gz | cut -f1)) -- on your machine:"
  say "  scp <server>:$(pwd)/results/igr_results.tar.gz . && tar xzf igr_results.tar.gz"
}

STAGES=("$@")
(( ${#STAGES[@]} )) || { sed -n '2,32p' "$0"; exit 1; }
[[ ${STAGES[*]} == all ]] && STAGES=(check plan norms run collect figures pack)
for s in "${STAGES[@]}"; do
  case $s in
    check|plan|norms|run|wsp|collect|figures|pack) ;;
    *) echo "unknown stage: $s"; exit 1 ;;
  esac
done
say "=== run_igr_server.sh ${STAGES[*]}  (PY=$PY)"
for s in "${STAGES[@]}"; do
  t0=$SECONDS
  "stage_$s" || { say "stage $s FAILED"; exit 1; }
  say "stage $s done in $(( (SECONDS - t0) / 60 )) min"
done
say "=== done"
