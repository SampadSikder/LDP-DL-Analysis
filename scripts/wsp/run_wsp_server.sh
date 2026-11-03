#!/usr/bin/env bash
# Weight-space programming (WSP) add-on on the lab server. Every stage is resumable: re-run the same command
# after a crash or kill and finished work is skipped. Existing code and results are only read.
#
#   bash scripts/wsp/run_wsp_server.sh setup                 # once: clone the conda env, install LangChain packages
#   nohup bash scripts/wsp/run_wsp_server.sh all > /dev/null 2>&1 &      # everything up to the pilot, then stops
#   tail -f results/wsp/run_wsp_server.log
#   bash scripts/wsp/run_wsp_server.sh wsp eval report       # after you approve the pilot numbers
#
# Stages (`all` = preflight stage1_mv stage1_freq baselines pilot):
#   setup        conda env clone ($ENV_NAME) of attacker-detector + requirements-wsp.txt (original env untouched)
#   preflight    .env present and git-ignored, packages, WSP unit tests, one tiny API call (check_api.py)
#   stage1_mv    mean/variance Stage 1: existing run.py evaluate, configs/wsp/stage1_mv.yaml (<= 50 epochs),
#                on breadth's rounds (linked read-only), sharded over GPUs                              [GPU]
#   stage1_freq  frequency-oracle Stage 1: main.py (unchanged) for seeds 43 / 44 (seed 42 = existing runs),
#                then verify every run reproduces its saved test F1 and cache train/val arrays          [GPU]
#   baselines    gated gradient fine-tune + gated random edit search on tier A (no tokens)            [GPU]
#   pilot        the 2 pilot LLM runs, then cost_report.py --pilot: tokens/run, projected total vs cap  [API]
#   wsp          every tier-A LLM run (WSP_CONCURRENCY in parallel; stops launching at 90% of the cap) [API]
#   eval         test evaluation of Stage 1 + every finished run (mean/var sharded, freq per run/seed),
#                then the existing attack_success.py on results/mean_variance_wsp/eval              [GPU]
#   report       cost_report.py, compare_wsp.py, wsp_figures.py -> results/figures/wsp/             [CPU]
#   tier_b       later week: baselines + LLM runs for income / retirement (after recalibrating the cap)
#
# Env overrides: PY (python of the WSP env), ENV_NAME (default attacker-detector-wsp), BASE_ENV
#   (default attacker-detector), SHARDS, THREADS, WSP_CONCURRENCY (parallel LLM runs, default 4),
#   WSP_GPUS (e.g. 0,1,2), SKIP_TESTS=1, SKIP_API_CHECK=1.
# Secrets: .env at the repo root (OPENAI_API_KEY, WSP_MODEL=gpt-6-astra, WSP_REASONING_EFFORT=xhigh,
#   WSP_TOKEN_CAP=633000000). It is read by Python only; this script never prints it.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

CONDA_BASE=${CONDA_BASE:-/home/boby/miniconda3}
BASE_ENV=${BASE_ENV:-attacker-detector}
ENV_NAME=${ENV_NAME:-attacker-detector-wsp}
PY=${PY:-$CONDA_BASE/envs/$ENV_NAME/bin/python}
W=results/wsp
LOGS=$W/logs
MVR=results/mean_variance_wsp/stage1
mkdir -p "$LOGS"
exec > >(tee -a "$W/run_wsp_server.log") 2>&1

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
export WSP_CONCURRENCY=${WSP_CONCURRENCY:-4}
GPUS_CSV=${WSP_GPUS:-$(seq -s, 0 $(( NGPU > 0 ? NGPU - 1 : 0 )))}
(( NGPU == 0 )) && GPUS_CSV=""
export WSP_GPUS=$GPUS_CSV

say () { echo "[$(date '+%F %T')] $*"; }
job () {
  local name=$1; shift
  say "start  $name"
  if "$@" >> "$LOGS/$name.log" 2>&1; then say "done   $name"
  else say "FAILED $name -- see $LOGS/$name.log"; return 1; fi
}
need_py () { [[ -x $PY ]] || { say "no python at $PY -- run: bash scripts/wsp/run_wsp_server.sh setup"; return 1; }; }

# ------------------------------------------------------------------ stages
stage_setup () {
  local conda=$CONDA_BASE/bin/conda
  if [[ ! -x $CONDA_BASE/envs/$ENV_NAME/bin/python ]]; then
    say "cloning conda env $BASE_ENV -> $ENV_NAME (the original env is not modified)"
    job setup_clone "$conda" create -y --clone "$BASE_ENV" -n "$ENV_NAME" || return 1
  fi
  job setup_pip "$CONDA_BASE/envs/$ENV_NAME/bin/pip" install -r requirements-wsp.txt || return 1
  "$CONDA_BASE/envs/$ENV_NAME/bin/python" -c "import langgraph, langchain_openai, dotenv, torch; \
print('langgraph ok; torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

stage_preflight () {
  need_py || return 1
  say "python $PY | GPUs $NGPU | CPUs $NCPU | RAM ${RAM_GB} GB | shards $SHARDS x $THREADS threads | LLM runs in parallel $WSP_CONCURRENCY"
  [[ -f .env ]] || { say ".env missing at $(pwd)/.env (OPENAI_API_KEY, WSP_MODEL, WSP_REASONING_EFFORT, WSP_TOKEN_CAP)"; return 1; }
  git check-ignore -q .env || { say ".env is NOT git-ignored -- refusing to run"; return 1; }
  chmod 600 .env
  "$PY" -c "import langgraph, langchain_openai, dotenv, yaml, psutil, pyarrow" || { say "packages missing: run setup"; return 1; }
  [[ -d results/mean_variance_server/breadth/rounds ]] || { say "breadth rounds missing (results/mean_variance_server/breadth/rounds)"; return 1; }
  if [[ ${SKIP_TESTS:-0} != 1 ]]; then job wsp_tests "$PY" -m unittest discover -s tests/wsp -t . || return 1; fi
  if [[ ${SKIP_API_CHECK:-0} != 1 ]]; then job check_api "$PY" scripts/wsp/check_api.py || return 1; tail -3 "$LOGS/check_api.log" | sed 's/^/    /'; fi
}

stage_stage1_mv () {
  need_py || return 1
  mkdir -p "$MVR/breadth"
  if [[ ! -e $MVR/breadth/rounds ]]; then
    ln -s "$(pwd)/results/mean_variance_server/breadth/rounds" "$MVR/breadth/rounds" || return 1
    say "linked $MVR/breadth/rounds -> results/mean_variance_server/breadth/rounds (read-only use)"
  fi
  local n=$SHARDS i gpu pids=() fail=0
  say "start  stage1_mv evaluate ($n shards, $NGPU GPUs, $THREADS threads each)"
  for (( i = 0; i < n; i++ )); do
    gpu=""; (( NGPU > 0 )) && gpu=$(( i % NGPU ))
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
      "$PY" scripts/mean_variance/run.py evaluate --config configs/wsp/stage1_mv.yaml --shard "$i/$n" \
      --results-root "$MVR" >> "$LOGS/stage1_mv_shard$i.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { say "FAILED stage1_mv shard $i/$n -- see $LOGS/stage1_mv_shard$i.log"; fail=1; }
  done
  (( fail == 0 )) || return 1
  say "done   stage1_mv evaluate"
  job stage1_mv_report "$PY" scripts/mean_variance/run.py report --config configs/wsp/stage1_mv.yaml --results-root "$MVR"
}

stage_stage1_freq () {
  need_py || return 1
  job stage1_freq_train "$PY" scripts/wsp/stage1_freq.py train --jobs "$(( NGPU > 0 ? NGPU : 1 ))" --gpus "$GPUS_CSV" || return 1
  CUDA_VISIBLE_DEVICES=${GPUS_CSV%%,*} job stage1_freq_verify "$PY" scripts/wsp/stage1_freq.py verify || return 1
  grep -E '^(OK|MISMATCH)' "$LOGS/stage1_freq_verify.log" | tail -15 | sed 's/^/    /'
}

stage_baselines () {
  need_py || return 1
  WSP_CONCURRENCY=$(( NGPU > 0 ? 2 * NGPU : 2 )) job baselines "$PY" scripts/wsp/sweep.py baselines
}

stage_pilot () {
  need_py || return 1
  job pilot "$PY" scripts/wsp/sweep.py pilot --concurrency 2 || return 1
  job pilot_cost "$PY" scripts/wsp/cost_report.py --pilot
  sed -n '/^ledger/p;/^pilot:/,$p' "$LOGS/pilot_cost.log" | tail -8 | sed 's/^/    /'
  say "PILOT DONE. Check the projection above and your dashboard %, then run:"
  say "    nohup bash scripts/wsp/run_wsp_server.sh wsp eval report > /dev/null 2>&1 &"
}

stage_wsp () {
  need_py || return 1
  job wsp_sweep "$PY" scripts/wsp/sweep.py wsp
  tail -5 "$LOGS/wsp_sweep.log" | sed 's/^/    /'
}

stage_eval () {
  need_py || return 1
  local n=$(( NGPU > 0 ? NGPU : 2 )) i gpu pids=() fail=0
  say "start  eval mean/variance ($n shards)"
  for (( i = 0; i < n; i++ )); do
    gpu=""; (( NGPU > 0 )) && gpu=$(( i % NGPU ))
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS "$PY" scripts/wsp/eval_wsp.py mv --shard "$i/$n" \
      >> "$LOGS/eval_mv_shard$i.log" 2>&1 &
    pids+=($!)
  done
  CUDA_VISIBLE_DEVICES=${GPUS_CSV%%,*} "$PY" scripts/wsp/eval_wsp.py freq >> "$LOGS/eval_freq.log" 2>&1 & local fp=$!
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { say "FAILED eval mv shard $i -- see $LOGS/eval_mv_shard$i.log"; fail=1; }; done
  wait $fp || { say "FAILED eval freq -- see $LOGS/eval_freq.log"; fail=1; }
  (( fail == 0 )) || return 1
  say "done   eval"
  job eval_asr "$PY" scripts/mean_variance/attack_success.py --results-root results/mean_variance_wsp \
    --experiments eval --jobs "$(( NCPU < 32 ? NCPU : 32 ))"
}

stage_report () {
  need_py || return 1
  job cost_report "$PY" scripts/wsp/cost_report.py || return 1
  job compare "$PY" scripts/wsp/compare_wsp.py || return 1
  job figures "$PY" scripts/wsp/wsp_figures.py || return 1
  sed -n '/^ledger/,$p' "$LOGS/cost_report.log" | tail -25 | sed 's/^/    /'
  say "copy back: results/figures/wsp/  results/wsp/cost_summary.csv  (and */transcript.jsonl if you want the logs)"
}

stage_tier_b () {
  need_py || return 1
  WSP_CONCURRENCY=$(( NGPU > 0 ? 2 * NGPU : 2 )) job tier_b_baselines "$PY" scripts/wsp/sweep.py baselines --set tier_b || return 1
  job tier_b_wsp "$PY" scripts/wsp/sweep.py wsp --set tier_b
}

stage_all () {
  stage_preflight || return 1
  stage_stage1_mv || return 1
  stage_stage1_freq || return 1
  stage_baselines || return 1
  stage_pilot
}

# ------------------------------------------------------------------ main
(( $# )) || { awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 1; }
status=0
for s in "$@"; do
  declare -F "stage_$s" > /dev/null || { echo "unknown stage: $s"; exit 1; }
  say "=== stage $s"
  if ! "stage_$s"; then say "=== stage $s FAILED"; status=1; break; fi
done
exit $status
