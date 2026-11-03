#!/usr/bin/env python
"""Test evaluation of Stage-1 and every finished Stage-2 run (see attacker_detector/wsp/frozen_eval.py).

  python scripts/wsp/eval_wsp.py mv --shard 0/3      # mean/variance, one shard (run one per GPU)
  python scripts/wsp/eval_wsp.py freq                # frequency oracle (loads each dataset once per run/seed)
Then: python scripts/mean_variance/attack_success.py --results-root results/mean_variance_wsp --experiments eval
"""
import argparse
import sys

import _common  # noqa: F401  (sets sys.path / cwd)

from attacker_detector.wsp.frozen_eval import eval_freq, eval_mv
from attacker_detector.wsp.jobs import SWEEP, all_jobs, load_yaml, select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['mv', 'freq'])
    ap.add_argument('--set', nargs='+', default=['tier_a'])
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    import torch
    dev = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    sw = load_yaml(SWEEP)
    jobs = select(all_jobs(), args.set, kinds=[args.what])
    if args.what == 'mv':
        i, n = (int(v) for v in args.shard.split('/'))
        print(f'mean/variance eval shard {i}/{n}: {eval_mv(sw, jobs, (i, n), dev)} files written')
    else:
        df = eval_freq(sw, jobs, torch.device(dev))
        print(df.to_string() if len(df) else 'nothing new to evaluate')
    return 0


if __name__ == '__main__':
    sys.exit(main())
