#!/usr/bin/env python
"""Progress and finish-time estimate for a sharded `evaluate` run (e.g. breadth on the server).

  python scripts/mean_variance/eta.py                       # breadth, results/mean_variance_server
  python scripts/mean_variance/eta.py --config configs/mean_variance/budget_sweep.yaml

How the estimate works:
  * The unit list and its split over shards are rebuilt exactly as `run.py evaluate --shard i/n`
    builds them; a unit is done when its units/<key>.done.json exists.
  * Each unit gets a predicted cost from the Mac pilot (PM, eps 1, Taxi; completion_report.md
    section 4): fit time scaled by training rounds plus inference time per test round.
  * Finished units' logged wall times give the server's speed factor (observed / predicted),
    per method and protocol when available, else per method, else overall.
  * Remaining time per shard = factored cost of its unfinished units, minus time already spent on
    the unit in progress (time since the shard log was last written). Finish = slowest shard.
Read-only: it never touches results.
"""

import argparse
import datetime as dt
import glob
import itertools
import json
import os
import re
import sys
import time
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from attacker_detector.mean_variance.experiment import load_config  # noqa: E402

FIXED = ('standard', 'oracle_removal', 'constant', 'kmeans')        # one seed, as in run.py
# Mac pilot: fit seconds (3,260 train rounds) and inference seconds per test round
FIT = {'kmeans': 160, 'mlp_removal': 154, 'ft_removal': 377, 'deepsets_recovery': 51,
       'settransformer_recovery': 167, 'catboost_recovery': 30}
INF = {'kmeans': 0.03, 'mlp_removal': 0.0085, 'ft_removal': 0.036, 'deepsets_recovery': 0.0029,
       'settransformer_recovery': 0.0091, 'catboost_recovery': 0.003}
LINE = re.compile(r'^(\S+) eps=(\S+) (\S+) train=(\S+) (\S+) s(\d+): (\w+) \(([\d.]+)s\)')


def training_sets(cfg):
    from run import training_sets as ts                          # same split as evaluate
    return ts(cfg)


def unit_list(cfg):
    seeds = cfg.get('seeds', [0])
    return [(p, float(e), tag, tr, te, m, s)
            for p, e in itertools.product(cfg['protocols'], cfg['epsilon'])
            for tag, tr, te in training_sets(cfg)
            for m in cfg['methods']
            for s in (seeds if m not in FIXED else [0])]


def key(u):
    p, e, tag, tr, _, m, s = u
    return f"{tag}__{p}_eps{e:g}__{'+'.join(tr)}__{m}__s{s}"


GENERATED = {'feasible', 'generated_with_target_error', 'below_budget'}   # statuses that carry reports
PILOT_TRAIN = 3_260                                                       # pilot training rounds


def round_counts(res, exp):
    """{(split, protocol, epsilon, source): generated rounds}, from generate's coverage.csv."""
    import pandas as pd
    path = os.path.join(res, exp, 'coverage.csv')
    if not os.path.exists(path):
        sys.exit(f"{path} missing: run generate first")
    cov = pd.read_csv(path, usecols=['split', 'source', 'protocol', 'epsilon', 'status'])
    cov = cov[cov['status'].isin(GENERATED)]
    return {(s, p, float(e), src): int(c) for (s, p, e, src), c in
            cov.groupby(['split', 'protocol', 'epsilon', 'source']).size().items()}


def predicted(u, counts):
    """Pilot-speed seconds: fit scaled by training rounds (k-means: fixed 300-round rate search),
    plus inference per test round over all destinations."""
    p, e, _, tr, te, m, _ = u
    n_train = sum(counts.get(('train', p, e, s), 0) for s in tr)
    n_test = sum(counts.get(('test', p, e, s), 0) for s in te)
    fit = FIT.get(m, 0.0) * (1 if m == 'kmeans' else n_train / PILOT_TRAIN)
    return fit + INF.get(m, 2e-4) * n_test


def fmt(h):
    return f"{h:5.1f} h" if h >= 1 else f"{h * 60:4.0f} min"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/mean_variance/breadth.yaml')
    ap.add_argument('--results-root', default='results/mean_variance_server')
    ap.add_argument('--shards', type=int, help="number of evaluate shards (default: count of shard logs)")
    args = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config)
    res = os.path.join(ROOT, args.results_root) if not os.path.isabs(args.results_root) else args.results_root
    exp = cfg['experiment']
    logs = sorted(glob.glob(os.path.join(res, 'logs', f'{exp}_evaluate_shard*.log')),
                  key=lambda p: int(re.search(r'shard(\d+)\.log$', p).group(1)))
    n = args.shards or len(logs)
    if not n:
        sys.exit(f"no shard logs in {res}/logs for {exp}; pass --shards")
    units = unit_list(cfg)
    units_dir = os.path.join(res, exp, 'units')
    counts = round_counts(res, exp)

    status = {}
    for u in units:
        p = os.path.join(units_dir, key(u) + '.done.json')
        if os.path.exists(p):
            try:
                with open(p) as f:
                    status[key(u)] = json.load(f).get('status', '?')
            except (OSError, ValueError):
                status[key(u)] = '?'

    # observed wall seconds per finished unit, from the shard logs (latest non-skipped line wins)
    observed = {}
    for path in logs:
        with open(path) as f:
            for line in f:
                m = LINE.match(line.strip())
                if m and m.group(7) != 'skipped':
                    p, e, tag, tr, meth, s, st, sec = m.groups()
                    observed[f"{tag}__{p}_eps{float(e):g}__{tr}__{meth}__s{s}"] = float(sec)

    by_key = {key(u): u for u in units}
    ratios = defaultdict(list)
    for k, sec in observed.items():
        if k in by_key and status.get(k) == 'completed':
            u = by_key[k]
            pred = predicted(u, counts)
            if pred > 5:                                   # skip near-zero baselines (noise)
                ratios[(u[5], u[0])].append(sec / pred)
                ratios[u[5]].append(sec / pred)
                ratios['all'].append(sec / pred)

    def factor(u):
        for k in ((u[5], u[0]), u[5], 'all'):
            if len(ratios.get(k, [])) >= 2:
                return sum(ratios[k]) / len(ratios[k])
        return 1.0

    now = time.time()
    done_n = sum(1 for u in units if key(u) in status)
    print(f"{exp}: {done_n:,}/{len(units):,} units done "
          f"({sum(v == 'completed' for v in status.values())} completed, "
          f"{sum(v == 'failed' for v in status.values())} failed, "
          f"{sum(v == 'unavailable' for v in status.values())} unavailable), {n} shards")

    meths = sorted((m for m in {u[5] for u in units} if m in FIT),        # skip near-free baselines
                   key=lambda m: -predicted(next(x for x in units if x[5] == m), counts))
    print("speed vs Mac pilot (observed / predicted; <1 = server faster):")
    for m in meths:
        parts = [f"{p} {sum(r) / len(r):.2f} (n={len(r)})" for p in cfg['protocols'] if (r := ratios.get((m, p)))]
        print(f"  {m:24} " + (', '.join(parts) if parts else 'no finished units yet -> using overall factor'))

    worst, rows = 0.0, []
    for i in range(n):
        mine = units[i::n]
        left = [u for u in mine if key(u) not in status]
        rem = sum(predicted(u, counts) * factor(u) for u in left)
        if left and i < len(logs):
            rem = max(rem - (now - os.path.getmtime(logs[i])), 0.0)   # time already spent on current unit
        worst = max(worst, rem)
        cur = left[0] if left else None
        rows.append(f"  shard {i}: {len(mine) - len(left):3}/{len(mine)} done, ~{fmt(rem / 3600)} left"
                    + (f"  (now: {cur[0]} eps={cur[1]:g} {cur[2]} {'+'.join(cur[3])} {cur[5]} s{cur[6]})" if cur else ''))
    print('\n'.join(rows))
    eta = dt.datetime.now() + dt.timedelta(seconds=worst)
    print(f"estimated finish of evaluate: {eta:%a %d %b %H:%M} (in {fmt(worst / 3600).strip()}), then the report")
    if 'PM' in cfg['protocols'] and not any(k[1] == 'PM' for k in ratios if isinstance(k, tuple)):
        print("note: no PM units finished yet. Units run SR first; PM units are slower, so this is likely optimistic.")
    failed = [k for k, v in status.items() if v == 'failed']
    if failed:
        print(f"WARNING: {len(failed)} failed unit(s) are marked done and will be skipped on rerun:")
        for k in failed[:10]:
            print(f"  {units_dir}/{k}.done.json")
        print("  read the error inside, fix, delete those .done.json files, and rerun evaluate")


if __name__ == '__main__':
    main()
