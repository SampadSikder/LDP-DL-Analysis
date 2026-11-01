#!/usr/bin/env python
"""Base vs mechanism-fit ('mech') features for MLP / FT removal, paired on the same PM test rounds.

  python scripts/mean_variance/compare_mech.py                     # results/mean_variance_server/breadth
  python scripts/mean_variance/compare_mech.py --results-root results/mean_variance --experiment smoke

d = mech - base squared error of the recovered mean / variance, paired per (training set, test round,
seed); shown as % of the base MSE with a 95% CI from resampling parent populations. d < 0: mech better.
Views: within (trained and tested on one source), cross (single-source model, other source),
multi (trained on two sources, tested on the third). Rows with attack 'none' are clean rounds:
'clean_damage' is the method's mean + variance MSE above the standard method's there. 'flagged' is
the share of reports the classifier flags; 'fallback' the share of rounds where a group would keep
< 16 reports and the estimate falls back to all of them (flagged 1.0 + fallback 1.0 = standard).
Writes <experiment>/mech_comparison.csv. Read-only otherwise.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from attacker_detector.mean_variance.evaluate import _cluster_bootstrap  # noqa: E402

KEY = ['partition', 'train_sources', 'round_id', 'seed']
PER = ['sq_err_mean', 'sq_err_var', 'removed_frac', 'fallback']
COLS = KEY + ['population_id', 'source', 'epsilon', 'attack'] + PER


def load(per_round, method):
    files = glob.glob(os.path.join(per_round, f'*__PM_eps*__{method}__s*.csv'))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in COLS) for f in files], ignore_index=True)
    if 'fallback' in df:
        df['fallback'] = df['fallback'].fillna('').astype(str) != ''
    return df


def paired(base, mech, std):
    m = base.merge(mech[KEY + PER], on=KEY, suffixes=('_base', '_mech'))
    # clean-round damage is measured against the standard method's own (projected) output
    m = m.merge(std[['partition', 'train_sources', 'round_id', 'sq_err_mean', 'sq_err_var']].rename(
        columns={'sq_err_mean': 'std_sq_err_mean', 'sq_err_var': 'std_sq_err_var'}),
        on=['partition', 'train_sources', 'round_id'], how='left')
    m['view'] = np.where(m['partition'] == 'multi-source', 'multi',
                         np.where(m['train_sources'] == m['source'], 'within', 'cross'))
    m['attack'] = m['attack'].fillna('none')
    for v in ('mean', 'var'):
        m[f'd_{v}'] = m[f'sq_err_{v}_mech'] - m[f'sq_err_{v}_base']
    return m


def summarize(g, n_boot):
    row = {'rounds': g['round_id'].nunique(), 'populations': g['population_id'].nunique()}
    for v in ('mean', 'var'):
        base = g[f'sq_err_{v}_base'].mean()
        lo, hi = _cluster_bootstrap(g, f'd_{v}', n_boot=n_boot)
        row.update({f'mse_{v}_base': base, f'mse_{v}_mech': g[f'sq_err_{v}_mech'].mean(),
                    f'd_{v}_%': 100 * g[f'd_{v}'].mean() / base,
                    f'd_{v}_ci_%': (100 * lo / base, 100 * hi / base),
                    f'{v}_verdict': 'mech better' if hi < 0 else 'mech worse' if lo > 0 else 'no clear diff'})
    for s in ('base', 'mech'):
        if (g['attack'] == 'none').all():
            row[f'clean_damage_{s}'] = (g[f'sq_err_mean_{s}'] + g[f'sq_err_var_{s}']
                                        - g['std_sq_err_mean'] - g['std_sq_err_var']).mean()
        row[f'flagged_{s}'] = g[f'removed_frac_{s}'].mean()
        row[f'fallback_{s}'] = g[f'fallback_{s}'].mean()
    return row


def table(m, by, n_boot):
    rows = [{**dict(zip(by, k if isinstance(k, tuple) else (k,))), **summarize(g, n_boot)}
            for k, g in m.groupby(by, sort=True)]
    return pd.DataFrame(rows)


def show(title, t):
    cols = [c for c in t.columns if not c.startswith('mse_') and c != 'populations']
    fmt = t[cols].copy()
    for c in fmt.columns:
        if c.endswith('_ci_%'):
            fmt[c] = fmt[c].map(lambda x: f"[{x[0]:+.1f}, {x[1]:+.1f}]")
        elif c.endswith('_%'):
            fmt[c] = fmt[c].map(lambda x: f"{x:+.1f}")
        elif c.startswith('clean_damage'):
            fmt[c] = fmt[c].map(lambda x: f"{x:+.2e}")
        elif c.startswith(('flagged', 'fallback')):
            fmt[c] = fmt[c].map(lambda x: f"{x:.3f}")
    print(f"\n== {title}\n{fmt.to_string(index=False)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results/mean_variance_server')
    ap.add_argument('--experiment', default='breadth')
    ap.add_argument('--n-boot', type=int, default=500)
    args = ap.parse_args()
    od = os.path.join(args.results_root if os.path.isabs(args.results_root) else os.path.join(ROOT, args.results_root),
                      args.experiment)
    per_round = os.path.join(od, 'per_round')
    std = load(per_round, 'standard')
    if std is None:
        sys.exit(f"standard results missing in {per_round}")
    parts = []
    for arch in ('mlp', 'ft'):
        base, mech = load(per_round, f'{arch}_removal'), load(per_round, f'{arch}_removal_mech')
        if base is None or mech is None:
            print(f"{arch}: {'base' if base is None else 'mech'} results missing in {per_round}; skipped")
            continue
        p = paired(base, mech, std)
        print(f"{arch}: {len(p):,} paired rows ({len(base):,} base, {len(mech):,} mech)")
        parts.append(p.assign(arch=arch))
    if not parts:
        sys.exit("nothing to compare: run the mech ablation (run_server.sh mech) first")
    m = pd.concat(parts, ignore_index=True)
    m['attacked'] = np.where(m['attack'] == 'none', 'clean', 'attacked')

    out = []
    for title, by, sub in (
            ("by view and attack", ['arch', 'view', 'attack'], m),
            ("clean rounds: damage vs standard", ['arch', 'view'], m[m['attack'] == 'none']),
            ("within-source, attacked rounds, by eps", ['arch', 'epsilon'], m[(m['view'] == 'within') & (m['attack'] != 'none')]),
            ("transfer pairs (train -> test), attacked rounds", ['arch', 'train_sources', 'source'], m[m['attack'] != 'none'])):
        if sub.empty:
            continue
        t = table(sub, by, args.n_boot)
        show(title, t)
        out.append(t.assign(table=title))
    path = os.path.join(od, 'mech_comparison.csv')
    pd.concat(out, ignore_index=True).to_csv(path, index=False)
    print(f"\nwritten: {path}")


if __name__ == '__main__':
    main()
