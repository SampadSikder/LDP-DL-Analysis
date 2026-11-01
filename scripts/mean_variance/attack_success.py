#!/usr/bin/env python
"""Attack success before and after each defence, plus MSE and utility gain, from per-round files.

  python scripts/mean_variance/attack_success.py                  # breadth, budget_sweep, below_budget
  python scripts/mean_variance/attack_success.py --experiments breadth --jobs 32

Success (metrics.attack_success): a statistic the target moves is hit when the estimate lands within
tol of the intended shift, |est - target| <= tol |target - genuine|. A statistic the target preserves
(index 2 of an offset target) is not scored; the joint score needs every moved statistic hit, so m2v2
rounds have no ASR. Before = no defence: the projected all-reports estimate, which equals the
`standard` method's output, so every method's row carries it. After = the method's estimate.
oracle_removal's after-ASR is the genuine-only estimate: how often noise alone lands on the target.

Writes <experiment>/attack_success.csv.gz, one row per cell (partition, train_sources, source,
protocol, epsilon, train_epsilon, beta, budget_factor, attack, target, status, method). Columns are
sums, so cells pool by adding rows:
  n_rounds, sse_mean, sse_var             method's squared error to the genuine moments
  std_sse_mean, std_sse_var               the same for no defence, on the same rounds
  n_<s>_t<t>, before_<s>_t<t>, after_<s>_t<t>, kept_<s>_t<t>
      for statistic s in (mean, var, joint) and tol t in percent: rounds where s is scored, hit
      before, hit after, hit both before and after
Derived: ASR before = before / n, ASR after = after / n, kept = kept / before, MSE = sse / n_rounds,
utility gain AG = (std_sse - sse) / n_rounds (paper Sec. 8.1, against the projected baseline).
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from attacker_detector.mean_variance.metrics import attack_success, joint_success  # noqa: E402

KEYS = ['partition', 'train_sources', 'source', 'protocol', 'epsilon', 'train_epsilon', 'beta', 'budget_factor',
        'attack', 'target', 'status', 'method']
COLS = set(KEYS) | {'mean_hat', 'var_hat', 'genuine_mean', 'genuine_variance', 'standard_mean', 'standard_variance',
                    'target_mean', 'target_variance'}
TOLS = (0.25, 0.5)


def tag(t):
    return f't{round(t * 100):02d}'


def cell_sums(path, tols=TOLS):
    d = pd.read_csv(path, usecols=lambda c: c in COLS)
    for k in KEYS:
        if k not in d:
            d[k] = np.nan
    d['attack'] = d['attack'].fillna('none')
    d['target'] = d['target'].fillna('clean')
    # no defence = standard method = project_moments of the all-reports estimate
    bm = d.standard_mean.clip(-1, 1)
    bv = d.standard_variance.clip(lower=0).clip(upper=1 - bm ** 2)
    out = d[KEYS].copy()
    out['n_rounds'] = 1
    out['sse_mean'] = (d.mean_hat - d.genuine_mean) ** 2
    out['sse_var'] = (d.var_hat - d.genuine_variance) ** 2
    out['std_sse_mean'] = (bm - d.genuine_mean) ** 2
    out['std_sse_var'] = (bv - d.genuine_variance) ** 2
    for t in tols:
        before = {'mean': attack_success(bm, d.genuine_mean, d.target_mean, t),
                  'var': attack_success(bv, d.genuine_variance, d.target_variance, t)}
        after = {'mean': attack_success(d.mean_hat, d.genuine_mean, d.target_mean, t),
                 'var': attack_success(d.var_hat, d.genuine_variance, d.target_variance, t)}
        before['joint'] = joint_success(before['mean'], before['var'])
        after['joint'] = joint_success(after['mean'], after['var'])
        for s in ('mean', 'var', 'joint'):
            b, a = before[s], after[s]
            out[f'n_{s}_{tag(t)}'] = (~np.isnan(b)).astype(int)
            out[f'before_{s}_{tag(t)}'] = np.nan_to_num(b).astype(int)
            out[f'after_{s}_{tag(t)}'] = np.nan_to_num(a).astype(int)
            out[f'kept_{s}_{tag(t)}'] = (np.nan_to_num(b) * np.nan_to_num(a)).astype(int)
    return out.groupby(KEYS, dropna=False).sum().reset_index()


def run(exp_dir, jobs, tols):
    files = sorted(glob.glob(os.path.join(exp_dir, 'per_round', '*.csv')))
    if not files:
        print(f'skip {exp_dir}: no per_round files')
        return None
    with ProcessPoolExecutor(jobs) as ex:
        parts = list(ex.map(cell_sums, files, [tols] * len(files), chunksize=4))
    g = pd.concat(parts, ignore_index=True).groupby(KEYS, dropna=False).sum().reset_index()
    path = os.path.join(exp_dir, 'attack_success.csv.gz')
    g.to_csv(path, index=False, float_format='%.10g')
    print(f'{os.path.basename(exp_dir)}: {len(files)} files, {int(g.n_rounds.sum())} rows -> {len(g)} cells, {path}')
    show(g, tols)
    return g


def show(g, tols):
    """Within-source attacked rounds: ASR before / after and kept fraction per protocol x eps x method."""
    t = tag(0.5 if 0.5 in tols else tols[-1])
    w = g[(g.partition == 'single-source') & (g.train_sources == g.source) & (g.attack != 'none')]
    if w.empty:
        return
    s = w.groupby(['protocol', 'epsilon', 'method']).sum(numeric_only=True)
    r = pd.DataFrame(index=s.index)
    for st in ('joint', 'mean', 'var'):
        n = s[f'n_{st}_{t}'].replace(0, np.nan)
        r[f'before_{st}'] = s[f'before_{st}_{t}'] / n
        r[f'after_{st}'] = s[f'after_{st}_{t}'] / n
        r[f'kept_{st}'] = s[f'kept_{st}_{t}'] / s[f'before_{st}_{t}'].replace(0, np.nan)
    r['mse_ratio_mean'] = s.sse_mean / s.std_sse_mean
    r['mse_ratio_var'] = s.sse_var / s.std_sse_var
    print(f'== within-source attacked rounds, tol {t}: ASR before / after defence, kept share of before-hits, '
          f'MSE / no-defence MSE')
    print(r.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results/mean_variance_server')
    ap.add_argument('--experiments', nargs='+', default=['breadth', 'budget_sweep', 'below_budget'])
    ap.add_argument('--tolerances', nargs='+', type=float, default=list(TOLS))
    ap.add_argument('--jobs', type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    root = args.results_root if os.path.isabs(args.results_root) else os.path.join(ROOT, args.results_root)
    pd.set_option('display.width', 250)
    for e in args.experiments:
        run(os.path.join(root, e), args.jobs, tuple(args.tolerances))


if __name__ == '__main__':
    main()
