#!/usr/bin/env bash
# Extra sweeps at n = 500,000, v2 features, balanced attackers, r = 10, r' = 4 fixed.
#   b010     beta fixed at 0.10, eps 0.1-1.0 plus 1.5, 2, 2.5, 3
#   b015     beta fixed at 0.15, same eps list
#   higheps  beta 0.05 (as before), eps 1, 1.5, 2, 2.5, 3
#
#   nohup bash scripts/run_extra_eps.sh mga-a > run_extra_mga-a.log 2>&1 &
#   nohup bash scripts/run_extra_eps.sh apa   > run_extra_apa.log   2>&1 &
# Re-running skips finished datasets and models.
set -uo pipefail

ATTACK=${1:?usage: run_extra_eps.sh mga-a|apa}
PY=/home/boby/miniconda3/envs/attacker-detector/bin/python
case $ATTACK in
  mga-a) PROTOCOLS=(OUE OLH_Server HST_User HST_Server) ;;
  apa)   PROTOCOLS=(OUE HST_User) ;;          # OLH_Server waits for the S-APA probe
  *) echo "attack must be mga-a or apa"; exit 1 ;;
esac
DATASETS=(zipf emoji fire)
EPS_FULL="0.1 0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0"
EPS_HIGH="1.0 1.5 2.0 2.5 3.0"
# name | default beta | epsilons | attackers per config (= 500k train rows / 2 / #configs)
EXPS=(
  "b010|0.10|$EPS_FULL|25000"
  "b015|0.15|$EPS_FULL|25000"
  "higheps|0.05|$EPS_HIGH|50000"
)
W=$(( $(nproc) / 2 )); (( W > 24 )) && W=24; (( W < 1 )) && W=1
W_OLH=$(( W > 16 ? 16 : W ))
say () { echo "[$(date '+%F %T')] $*"; }

grep -q "each fake user promotes its own" attacker_detector/data/generators/attacks.py \
  || { echo "sync attacks.py first (HST_Server per-user subset fix missing)"; exit 1; }

FT=(-m ft_transformer --train-samples 500000 --test-samples 60000 --val-size 0.10 --balanced-batches
    --k-folds 0 --epochs 50 --patience 10 --batch-size 256 --lr 0.0005 --d-token 64 --n-heads 8
    --n-layers 3 --ffn-d-multiplier 1.33 --attention-dropout 0.1 --residual-dropout 0.0
    --seed 42 --no-plot --save-predictions)
train () {
  local out=$1 log=$2; shift 2
  [[ -f $out/sensitivity_test_results.csv ]] && { say "skip $out"; return; }
  say "train $out"
  $PY main.py "$@" -o "$out" > "$log" 2>&1 || { say "FAILED $out -- see $log"; exit 1; }
  grep -hE "^Train:|WARN" "$log" | sed 's/^/    /'
}

for E in "${EXPS[@]}"; do
  IFS='|' read -r NAME BETA EPS BAL <<< "$E"
  DATA=output/oat_${NAME}_${ATTACK}_n500k
  RES=results/oat_${NAME}_${ATTACK}_n500k
  mkdir -p "$DATA" "$RES/logs"
  say "=== $NAME: beta=$BETA eps=[$EPS] attack=$ATTACK"

  for P in "${PROTOCOLS[@]}"; do
    D=$DATA/$P
    if [[ -f $D/features.npy && -f $D/config.npy && ! -f $D/config.bin ]]; then say "skip generate $P"; continue; fi
    WP=$W; OLHFLAG=(); [[ $P == OLH_Server ]] && { WP=$W_OLH; OLHFLAG=(--olh-parallel); }
    say "generate $P (workers=$WP)"
    $PY generate_dataset.py -o "$D" --design oat --attack "$ATTACK" --protocols "$P" \
        --datasets "${DATASETS[@]}" \
        --epsilons $EPS --ratios "$BETA" --splits 4 --target-sizes 10 \
        --default-epsilon 1.0 --default-ratio "$BETA" --default-splits 4 --default-target-size 10 \
        --balance-attackers "$BAL" --holdout-replicates 1 --n 500000 \
        --feature-set v2 --block-candidates 16 \
        --workers "$WP" --inner-processors 2 --tasks-per-worker 2 "${OLHFLAG[@]}" --save-every 10 \
        > "$DATA/gen_$P.log" 2>&1 || { say "FAILED generate $P"; tail -20 "$DATA/gen_$P.log"; exit 1; }
    say "done generate $P"
  done

  say "verify $NAME"
  $PY - "$DATA" "$ATTACK" "${PROTOCOLS[@]}" <<'EOF' || exit 1
import sys, json, re, numpy as np
data, attack, protos = sys.argv[1], sys.argv[2], sys.argv[3:]
bad = False
for p in protos:
    F = np.load(f"{data}/{p}/features.npy", mmap_mode="r")
    des = json.load(open(f"{data}/{p}/design.json"))
    rows = des["n"] * sum(r["train"] + r["holdout"] for r in des["runs"])
    failed = re.search(r"Failed runs: (\d+)", open(f"{data}/gen_{p}.log").read())
    ok = F.shape == (rows, 14) and des.get("attack") == attack and failed and failed.group(1) == "0"
    bad |= not ok
    print(f"  {p:10} {F.shape} expected {rows} rows, attack={des.get('attack')} "
          f"failed={failed and failed.group(1)}  {'OK' if ok else 'PROBLEM'}")
sys.exit(1 if bad else 0)
EOF

  for P in "${PROTOCOLS[@]}"; do
    for D in "${DATASETS[@]}"; do
      Dc=${D^}
      TR=(); for X in "${DATASETS[@]}"; do [[ $X != "$D" ]] && TR+=("$X"); done
      train "$RES/in_dist/${Dc}_$P" "$RES/logs/in_${Dc}_$P.log" \
            -d "$DATA/$P" "${FT[@]}" --protocol "$P" --dataset "$D"
      train "$RES/cross/${P}_${Dc}_test" "$RES/logs/cross_${P}_${Dc}.log" \
            -d "$DATA/$P" "${FT[@]}" --protocol "$P" --training-method cross --train-dataset "${TR[@]}" --test-dataset "$D"
    done
  done
  say "done $NAME -- results in $RES"
done
say "ALL DONE ($ATTACK)"
