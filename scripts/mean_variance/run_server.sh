#!/usr/bin/env bash
# Full SR/PM mean-variance benchmark on the lab server. Every step is resumable:
# re-run the same command after a crash or kill and finished work is skipped.
#
#   nohup bash scripts/mean_variance/run_server.sh all > /dev/null 2>&1 &
#   tail -f results/mean_variance_server/run_server.log          # progress
#   bash scripts/mean_variance/run_server.sh preflight         # one stage
#   bash scripts/mean_variance/run_server.sh paper kmeans      # several stages, in order
#
# Stages (`all` order):
#   preflight  packages, CUDA check (stage `cuda`), dataset files, unit tests, data audit (warms the source cache)
#   cuda       only the CUDA check: GPUs visible, kernels run, benchmark methods pick and train on cuda,
#              one GPU per shard (scripts/mean_variance/check_cuda.py)
#   smoke      small end-to-end run of every method on GPU (pipeline check, not a result)
#   paper      attack validation at paper scale (7 configs), feasibility maps, budget curves  [CPU]
#   kmeans     paper Sec. 8.1 k-means verification: rate, beta, mean, variance sweeps         [CPU]
#   budget     budget ladder (train + evaluate), then below-budget frozen evaluation          [GPU]
#   breadth    main grid: generate, evaluate sharded over GPUs, report                        [GPU]
#   mech       optional, after breadth: MLP/FT with mech features on breadth's PM rounds, then
#              compare_mech.py (base vs mech, paired; logs/mech_compare.log)                  [GPU]
#   setfeat    optional, after breadth: Deep Sets / Set Transformer with richer observation-only inputs
#              (feature views 'context', PM+SR; 'context_report', PM) on breadth's rounds, then
#              compare_setfeat.py (paired vs the 'minimal' breadth models; logs/setfeat_compare.log) [GPU]
#   asr        after budget and breadth: attack success before / after each defence, kept share of hits,
#              MSE and utility gain, from per_round files (attack_success.py ->
#              <experiment>/attack_success.csv.gz, a few MB; copy these to plot locally)    [CPU]
# `all` = preflight, smoke, then the CPU track (paper, kmeans) next to the GPU track (budget, breadth), then asr.
#
# Env overrides:
#   PY          python of the conda env (default: /home/boby/miniconda3/envs/attacker-detector/bin/python)
#   SHARDS      evaluate processes for budget/breadth (default 2 per GPU, capped by RAM/32 GB and CPUs/8)
#   THREADS     BLAS/torch threads per shard (default CPUs / (2 * SHARDS))
#   EPS_LIST    eps values for feasibility maps and budget curves (default "0.1 1 2 4 6")
#   RESULTS     output folder (default results/mean_variance_server; kept apart from local results/mean_variance)
#   SKIP_TESTS=1, ALLOW_CPU=1 (run even if torch cannot see the GPUs)
# Everything this script produces goes under $RESULTS:
#   run_server.log                  this script's console output (one line per step)
#   logs/<step>.log                 full output of each step
#   <experiment>/                   one folder per config (rounds, checkpoints, per_round, aggregate.csv, figures)
#   feasibility/, budget_curves/, provenance_<source>.json
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-/home/boby/miniconda3/envs/attacker-detector/bin/python}
[[ -x $PY ]] || PY=$(command -v python)
R=${RESULTS:-results/mean_variance_server}
LOGS=$R/logs
C=configs/mean_variance
EPS_LIST=${EPS_LIST:-"0.1 1 2 4 6"}
mkdir -p "$LOGS"
exec > >(tee -a "$R/run_server.log") 2>&1       # console output also kept inside $R

NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
NCPU=$(nproc)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [[ -z ${SHARDS:-} ]]; then
  SHARDS=$(( NGPU > 0 ? 2 * NGPU : 4 ))
  (( SHARDS > RAM_GB / 32 )) && SHARDS=$(( RAM_GB / 32 ))
  (( SHARDS > NCPU / 8 )) && SHARDS=$(( NCPU / 8 ))
  (( SHARDS < 1 )) && SHARDS=1
fi
THREADS=${THREADS:-$(( NCPU / (2 * SHARDS) ))}
(( THREADS < 1 )) && THREADS=1

say () { echo "[$(date '+%F %T')] $*"; }
run () { "$PY" scripts/mean_variance/run.py "$@" --results-root "$R"; }

# job <name> <cmd...>: run quietly into $LOGS/<name>.log (appended), one status line
job () {
  local name=$1; shift
  say "start  $name"
  if "$@" >> "$LOGS/$name.log" 2>&1; then say "done   $name"
  else say "FAILED $name -- see $LOGS/$name.log"; return 1; fi
}

# evaluate <config> in $SHARDS processes, round-robin over GPUs
eval_sharded () {
  local cfg=$1 name=$2 n=$SHARDS i gpu pids=() fail=0
  say "start  ${name}_evaluate ($n shards, $NGPU GPUs, $THREADS threads each)"
  for (( i = 0; i < n; i++ )); do
    gpu=""; (( NGPU > 0 )) && gpu=$(( i % NGPU ))
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
      "$PY" scripts/mean_variance/run.py evaluate --config "$cfg" --shard "$i/$n" --results-root "$R" \
      >> "$LOGS/${name}_evaluate_shard$i.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { say "FAILED ${name}_evaluate shard $i/$n -- see $LOGS/${name}_evaluate_shard$i.log"; fail=1; }
  done
  (( fail == 0 )) && say "done   ${name}_evaluate"
  return $fail
}

# generate -> sharded evaluate -> report
grid () {
  local cfg=$1 name=$2
  job "${name}_generate" run generate --config "$cfg" || return 1
  eval_sharded "$cfg" "$name" || return 1
  job "${name}_report" run report --config "$cfg"
}

# wait for every background pid given; fail if any failed
wait_all () { local p fail=0; for p in "$@"; do wait "$p" || fail=1; done; return $fail; }

# ------------------------------------------------------------------ stages
stage_preflight () {
  say "python $PY | GPUs $NGPU | CPUs $NCPU | RAM ${RAM_GB} GB | shards $SHARDS x $THREADS threads"
  (( NGPU > 0 )) && nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | sed 's/^/    /'
  "$PY" - <<'EOF' || return 1
import importlib.util, sys
need = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'torch', 'yaml', 'psutil', 'pyarrow', 'catboost']
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"    missing packages: {missing}\n    install: {sys.executable} -m pip install "
             + ' '.join({'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}.get(m, m) for m in missing))
print("    packages ok")
EOF
  if [[ ${ALLOW_CPU:-0} == 1 ]]; then
    stage_cuda || say "WARNING: CUDA check failed; ALLOW_CPU=1, so neural methods will run on CPU"
  else
    stage_cuda || return 1
  fi
  local f bad=0
  for f in datasets/mean_variance/taxi/green_tripdata_2020-12.parquet \
           datasets/mean_variance/income/psam_pusa.csv datasets/mean_variance/income/psam_pusb.csv \
           datasets/mean_variance/retirement/employee-compensation.csv; do
    [[ -s $f ]] || { say "MISSING $f"; bad=1; }
  done
  (( bad == 0 )) || return 1
  if [[ ${SKIP_TESTS:-0} != 1 ]]; then
    job tests "$PY" -m unittest discover -s tests/mean_variance -t . || return 1
  fi
  job audit_data run audit-data     # also builds datasets/mean_variance/.cache before parallel jobs start
  grep -h '^{' "$LOGS/audit_data.log" | tail -3 | cut -c1-160 | sed 's/^/    /'
}

# CUDA detection by the benchmark code itself (see check_cuda.py); results on screen, warnings in the log
stage_cuda () {
  say "start  cuda_check"
  "$PY" scripts/mean_variance/check_cuda.py --config "$C/breadth.yaml" 2>> "$LOGS/cuda_check.log" \
    | tee -a "$LOGS/cuda_check.log" | while IFS= read -r l; do echo "    $l"; done   # unbuffered, unlike sed
  local s=${PIPESTATUS[0]}
  (( s == 0 )) && say "done   cuda_check" || say "FAILED cuda_check -- see $LOGS/cuda_check.log"
  return "$s"
}

stage_smoke () { grid "$C/smoke.yaml" smoke; }

stage_paper () {
  local c e pids=()
  for c in paper_attack_default paper_attack_eps paper_attack_table4_taxi paper_attack_beta_taxi \
           paper_attack_S1_taxi paper_attack_S2_taxi paper_attack_ne_taxi; do
    job "attack_$c" run attack-validate --config "$C/$c.yaml" & pids+=($!)
  done
  (
    for e in $EPS_LIST; do
      [[ -f $R/feasibility/feasibility_eps${e}.csv ]] || job "feasibility_eps$e" run feasibility-maps --epsilon "$e" || exit 1
      [[ -f $R/budget_curves/budget_curves_eps${e}.csv ]] || job "budget_curves_eps$e" run budget-curves --epsilon "$e" || exit 1
    done
  ) & pids+=($!)
  wait_all "${pids[@]}"
}

stage_kmeans () {
  # beta / mean / variance sweeps use r = 0.6 (the rate-sweep optimum found locally);
  # check verify_kmeans_rate_taxi/kmeans_verification_summary.csv still agrees.
  local c pids=()
  for c in rate beta mean var; do
    job "kmeans_$c" run verify-kmeans --config "$C/verify_kmeans_${c}_taxi.yaml" & pids+=($!)
  done
  wait_all "${pids[@]}"
}

stage_budget () {
  grid "$C/budget_sweep.yaml" budget_sweep || return 1
  job below_budget_generate run generate --config "$C/below_budget.yaml" --splits test || return 1
  job below_budget_frozen run evaluate-frozen --config "$C/below_budget.yaml" \
      --checkpoints "$R/budget_sweep/checkpoints"
}

stage_breadth () { grid "$C/breadth.yaml" breadth; }

# mech-feature ablation on breadth's PM rounds (needs breadth generate), then the paired comparison
stage_mech () {
  eval_sharded "$C/breadth_mech.yaml" breadth_mech || return 1
  job mech_compare "$PY" scripts/mean_variance/compare_mech.py --results-root "$R" || return 1
  sed -n '/^== by view and attack/,/^$/p' "$LOGS/mech_compare.log" | sed 's/^/    /'
}

# feature-view ablation for the set models on breadth's rounds (needs breadth generate + evaluate),
# then the paired comparison; new method names, so breadth's own results and checkpoints stay as they are
stage_setfeat () {
  eval_sharded "$C/breadth_setfeat.yaml" breadth_setfeat || return 1
  eval_sharded "$C/breadth_setfeat_report.yaml" breadth_setfeat_report || return 1
  job setfeat_compare "$PY" scripts/mean_variance/compare_setfeat.py --results-root "$R" || return 1
  sed -n '/^== verdicts/,/^$/p' "$LOGS/setfeat_compare.log" | sed 's/^/    /'
}

# attack success rate before / after defence, paired per round; a few MB per experiment
stage_asr () {
  local jobs=$(( NCPU < 32 ? NCPU : 32 ))
  job asr "$PY" scripts/mean_variance/attack_success.py --results-root "$R" --jobs "$jobs" || return 1
  ls -lh "$R"/*/attack_success.csv.gz | awk '{print "    " $5 "  " $NF}'
}

stage_all () {
  stage_preflight || return 1
  stage_smoke || return 1
  ( stage_paper; p=$?; stage_kmeans; exit $(( p || $? )) ) & local cpu=$!
  ( stage_budget; b=$?; stage_breadth; exit $(( b || $? )) ) & local gpu=$!
  wait_all $cpu $gpu || return 1
  stage_asr
}

# ------------------------------------------------------------------ main
(( $# )) || { awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 1; }
status=0
for s in "$@"; do
  declare -F "stage_$s" > /dev/null || { echo "unknown stage: $s"; exit 1; }
  say "=== stage $s"
  "stage_$s" || { status=1; say "=== stage $s FAILED"; [[ $s == preflight ]] && exit 1; }
done
(( status == 0 )) && say "ALL DONE -- results in $R" || say "finished with failures; fix and re-run (finished work is skipped)"
exit $status
