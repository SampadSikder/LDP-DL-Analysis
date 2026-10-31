#!/usr/bin/env bash
# Full SR/PM mean-variance benchmark on the lab server. Every step is resumable:
# re-run the same command after a crash or kill and finished work is skipped.
#
#   nohup bash scripts/mean_variance/run_server.sh all > run_mv_all.log 2>&1 &
#   bash scripts/mean_variance/run_server.sh preflight         # one stage
#   bash scripts/mean_variance/run_server.sh paper kmeans      # several stages, in order
#
# Stages (`all` order):
#   preflight  env + GPU check, dataset files, unit tests, data audit (warms the source cache)
#   smoke      small end-to-end run of every method on GPU (pipeline check, not a result)
#   paper      attack validation at paper scale (7 configs), feasibility maps, budget curves  [CPU]
#   kmeans     paper Sec. 8.1 k-means verification: rate, beta, mean, variance sweeps         [CPU]
#   budget     budget ladder (train + evaluate), then below-budget frozen evaluation          [GPU]
#   breadth    main grid: generate, evaluate sharded over GPUs, report                        [GPU]
# `all` = preflight, smoke, then the CPU track (paper, kmeans) next to the GPU track (budget, breadth).
#
# Env overrides:
#   PY          python of the conda env (default: /home/boby/miniconda3/envs/attacker-detector/bin/python)
#   SHARDS      evaluate processes for budget/breadth (default 2 per GPU, capped by RAM/32 GB and CPUs/8)
#   THREADS     BLAS/torch threads per shard (default CPUs / (2 * SHARDS))
#   EPS_LIST    eps values for feasibility maps and budget curves (default "0.1 1 2 4 6")
#   SKIP_TESTS=1, ALLOW_CPU=1 (run even if torch cannot see the GPUs)
# Logs: results/mean_variance/logs/<step>.log; this script prints one line per step.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-/home/boby/miniconda3/envs/attacker-detector/bin/python}
[[ -x $PY ]] || PY=$(command -v python)
R=results/mean_variance
LOGS=$R/logs
C=configs/mean_variance
EPS_LIST=${EPS_LIST:-"0.1 1 2 4 6"}
mkdir -p "$LOGS"

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
run () { "$PY" scripts/mean_variance/run.py "$@"; }

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
      "$PY" scripts/mean_variance/run.py evaluate --config "$cfg" --shard "$i/$n" \
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
  NGPU=$NGPU ALLOW_CPU=${ALLOW_CPU:-0} "$PY" - <<'EOF' || return 1
import importlib.util, os, sys
need = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'torch', 'yaml', 'psutil', 'pyarrow', 'catboost']
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"    missing packages: {missing}\n    install: {sys.executable} -m pip install "
             + ' '.join({'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}.get(m, m) for m in missing))
import torch
from attacker_detector.mean_variance.methods.removal import device_for
print(f"    torch {torch.__version__}, cuda {torch.version.cuda}, visible GPUs {torch.cuda.device_count()}, "
      f"auto device -> {device_for('auto')}")
if int(os.environ['NGPU']) > 0 and not torch.cuda.is_available() and os.environ['ALLOW_CPU'] != '1':
    sys.exit("    nvidia-smi sees GPUs but torch does not (CPU-only torch build?). Set ALLOW_CPU=1 to run anyway.")
EOF
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

stage_all () {
  stage_preflight || return 1
  stage_smoke || return 1
  ( stage_paper; p=$?; stage_kmeans; exit $(( p || $? )) ) & local cpu=$!
  ( stage_budget; b=$?; stage_breadth; exit $(( b || $? )) ) & local gpu=$!
  wait_all $cpu $gpu
}

# ------------------------------------------------------------------ main
(( $# )) || { sed -n '2,24p' "$0"; exit 1; }
status=0
for s in "$@"; do
  declare -F "stage_$s" > /dev/null || { echo "unknown stage: $s"; exit 1; }
  say "=== stage $s"
  "stage_$s" || { status=1; say "=== stage $s FAILED"; [[ $s == preflight ]] && exit 1; }
done
(( status == 0 )) && say "ALL DONE -- results in $R" || say "finished with failures; fix and re-run (finished work is skipped)"
exit $status
