#!/usr/bin/env bash
# Full dataset generation for cross-dataset experiments: train {zipf, fire} -> test emoji.
#
# Domain size is left at each dataset's native value (zipf 1024, fire 296, emoji 1496).
# The 14 features are analytically scaled by (d, p, q), so they are dimensionless and
# comparable across domains without a uniform --domain and without fitted per-dataset stats.

set -euo pipefail

OUT="${1:-outputs/diffstats_style}"
LOG="${OUT}_gen.log"

if [[ -e "$OUT" ]]; then
    echo "ERROR: $OUT already exists." >&2
    echo "The generator appends to .bin staging files, so reusing a directory corrupts" >&2
    echo "the output. Remove it first:  rm -rf $OUT" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

python generate_dataset.py \
  -o "$OUT" \
  --protocols OUE \
  --epsilons 1.0 1.5 2.0 \
  --datasets zipf fire emoji \
  --ratios 0.1 0.2 0.3 0.4 \
  --target-sizes 8 10 12 14 \
  --splits 4 6 \
  --experiments 2 \
  --n 30000 \
  --seed 42 \
  --workers 6 \
  --inner-processors 2 \
  --save-every 10 \
  2>&1 | tee "$LOG"

echo
echo "Done. Train on zipf+fire, test on emoji with:"
echo
echo "  python main.py -d $OUT -m mlp \\"
echo "    --training-method cross --train-dataset zipf fire --test-dataset emoji \\"
echo "    --k-folds 3 --epochs 20 --patience 5 --batch-size 8192 \\"
echo "    --hp-hidden-sizes default --val-size 0.1 --seed 42 \\"
echo "    -o ${OUT}_run"
