#!/usr/bin/env bash
# FIAD + Diffstats on every protocol / dataset of one data root (the
# <data_root>/<PROTOCOL> layout run_extra_eps.sh generates), written where
# scripts/plots/loader.load_baselines_oat reads them:
#   <results_root>/baselines/<Dataset>_<PROTOCOL>/{fiad,diffstats}/sensitivity_test_results.csv
#
#   nohup bash scripts/run_baselines.sh output/oat_apa_n500k results/oat_apa_n500k \
#       > run_baselines_apa.log 2>&1 &
#   python scripts/plots/make_figures_v6.py --root oat_apa_n500k --baselines measured
#
# Re-running skips finished cells; an interrupted cell resumes from its
# baseline_runs.csv. PY=/path/to/python overrides the interpreter.
set -uo pipefail

USAGE="usage: run_baselines.sh <data_root> <results_root> [processors]"
DATA=${1:?$USAGE}
RES=${2:?$USAGE}
PROC=${3:-4}
PY=${PY:-python}
say () { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$RES/logs"
for D in "$DATA"/*/; do
  D=${D%/}
  [[ -f $D/design.json ]] || continue
  P=$(basename "$D")
  DATASETS=$($PY -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['datasets']))" "$D/design.json")
  for DS in $DATASETS; do
    Dc=${DS^}
    OUT=$RES/baselines/${Dc}_$P
    LOG=$RES/logs/baselines_${Dc}_$P.log
    if [[ -f $OUT/fiad/sensitivity_test_results.csv && -f $OUT/diffstats/sensitivity_test_results.csv ]]; then
      say "skip $OUT"; continue
    fi
    say "baselines $P $DS"
    $PY main.py -d "$D" --protocol "$P" --dataset "$DS" --baseline fiad diffstats \
        --baseline-processors "$PROC" -o "$OUT" > "$LOG" 2>&1 \
      || { say "FAILED $OUT -- see $LOG"; exit 1; }
    grep -A2 "^fiad sensitivity\|^diffstats sensitivity" "$LOG" | grep overall | sed 's/^/    /'
  done
done
say "ALL DONE -- results in $RES/baselines"
