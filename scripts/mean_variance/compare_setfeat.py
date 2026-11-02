#!/usr/bin/env python
"""Feature-view ablation for the set models (Deep Sets, Set Transformer), paired on the same test rounds.

  python scripts/mean_variance/compare_setfeat.py                    # results/mean_variance_server/breadth
  python scripts/mean_variance/compare_setfeat.py --results-root results/mean_variance --experiment smoke

Comparisons, each paired per (training set, test round, seed), PM and SR shown separately:
  context        vs minimal   PM and SR
  context_report vs context   PM only (for SR, context_report is an alias of context: never compared)
Both sides share rounds, splits, seeds, token cap, rounds per batch, widths, optimizer and loss; only
the inputs (and so the first-layer sizes) differ.
d = B - A per round for the squared error of the mean, of the variance, and the combined utility loss
L_u = (err_mean / scale_mean)^2 + lambda_var (err_var / scale_var)^2, with the unit's training-round
scales read from checkpoints/<unit>/state.json (the same for A and B; NaN when checkpoints are absent).
Reported as % of A's mean, with a 95% CI from resampling parent populations; d < 0: B better.
Tables are split by rounds (clean / attacked, and IPA / OPA) and by view: within (trained and tested on
one source), cross (single-source model on another source), multi (trained on two sources, tested on
the third); also by eps and per transfer pair (e.g. taxi -> income). Nothing is pooled across clean
and attacked rounds or across views, so a gain in one cannot hide harm in another; the verdict list
names every cell where B is clearly worse. Costs: end-to-end inference time per round (feature
extraction + network, and each part where recorded), fit time per unit, parameter counts.
Writes <experiment>/setfeat_comparison.csv and setfeat_costs.csv. Read-only otherwise.
Attack-success rates for the new methods come from attack_success.py like every other method.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from attacker_detector.mean_variance.methods.set_models import SetRecovery  # noqa: E402

ARCHS = ('deepsets', 'settransformer')
COMPARISONS = (('context', 'minimal', ('PM', 'SR')), ('context_report', 'context', ('PM',)))
KEY = ['partition', 'train_sources', 'round_id', 'seed']
PER = ['sq_err_mean', 'sq_err_var', 'runtime', 't_features', 't_network']
COLS = KEY + ['population_id', 'source', 'protocol', 'epsilon', 'attack', 'fit_time'] + PER
METRICS = (('mean', 'MSE mean'), ('var', 'MSE variance'), ('loss', 'utility loss'))


def method_name(arch, view):
    return f'{arch}_recovery' + ('' if view == 'minimal' else f'_{view}')


def load(per_round, method, protocol):
    files = glob.glob(os.path.join(per_round, f'*__{protocol}_eps*__{method}__s*.csv'))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in COLS) for f in files], ignore_index=True)
    for c in PER:
        if c not in df:
            df[c] = np.nan
    return df


class Scales:
    """Training-round loss scales per unit, from the checkpoints' state.json (cached)."""

    def __init__(self, exp_dir):
        self.dir, self.cache = os.path.join(exp_dir, 'checkpoints'), {}

    def get(self, partition, protocol, eps, train, method, seed):
        key = f"{partition}__{protocol}_eps{eps:g}__{train}__{method}__s{seed}"
        if key not in self.cache:
            p = os.path.join(self.dir, key, 'state.json')
            sc = json.load(open(p))['scales'] if os.path.exists(p) else None
            self.cache[key] = (sc['scale_mean'], sc['scale_var'], sc['lambda_var']) if sc else (np.nan,) * 3
        return self.cache[key]


def paired(a, b, scales, ma, mb):
    m = a.merge(b[KEY + PER], on=KEY, suffixes=('_A', '_B'))
    m['view'] = np.where(m['partition'] == 'multi-source', 'multi',
                         np.where(m['train_sources'] == m['source'], 'within', 'cross'))
    m['attack'] = m['attack'].fillna('none')
    m['kind'] = np.where(m['attack'] == 'none', 'clean', 'attacked')
    units = m[['partition', 'protocol', 'epsilon', 'train_sources', 'seed']].drop_duplicates()
    sc = []
    for u in units.itertuples(index=False):
        s = scales.get(u.partition, u.protocol, u.epsilon, u.train_sources, ma, u.seed)
        if np.isnan(s[0]):
            s = scales.get(u.partition, u.protocol, u.epsilon, u.train_sources, mb, u.seed)
        sc.append((*u, *s))
    sc = pd.DataFrame(sc, columns=list(units.columns) + ['scale_mean', 'scale_var', 'lambda_var'])
    m = m.merge(sc, on=list(units.columns), how='left')
    for s in ('A', 'B'):
        m[f'sq_err_loss_{s}'] = (m[f'sq_err_mean_{s}'] / m.scale_mean ** 2
                                 + m.lambda_var * m[f'sq_err_var_{s}'] / m.scale_var ** 2)
    for v, _ in METRICS:
        m[f'd_{v}'] = m[f'sq_err_{v}_B'] - m[f'sq_err_{v}_A']
    return m


def ratio_ci(g, num, den, n_boot, seed=0):
    """95% CI of sum(num) / sum(den), resampling parent populations (rounds are not independent)."""
    s = g.groupby('population_id')[[num, den]].sum()
    if len(s) < 2:
        return np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    r = s[num].to_numpy()[idx].sum(1) / s[den].to_numpy()[idx].sum(1)
    return tuple(np.percentile(r, [2.5, 97.5]))


def summarize(g, n_boot):
    row = {'rounds': g['round_id'].nunique(), 'paired_rows': len(g), 'populations': g['population_id'].nunique()}
    for v, _ in METRICS:
        a = g[f'sq_err_{v}_A'].mean()
        if not np.isfinite(a) or a == 0:
            row.update({f'{v}_A': a, f'{v}_B': g[f'sq_err_{v}_B'].mean(), f'd_{v}_%': np.nan,
                        f'd_{v}_ci_%': (np.nan, np.nan), f'{v}_verdict': 'n/a'})
            continue
        lo, hi = ratio_ci(g, f'd_{v}', f'sq_err_{v}_A', n_boot)      # % change stays >= -100%
        lo, hi = 100 * lo, 100 * hi
        row.update({f'{v}_A': a, f'{v}_B': g[f'sq_err_{v}_B'].mean(), f'd_{v}_%': 100 * g[f'd_{v}'].mean() / a,
                    f'd_{v}_ci_%': (lo, hi),
                    f'{v}_verdict': 'B better' if hi < 0 else 'B worse' if lo > 0 else 'no clear diff'})
    return row


def table(m, by, n_boot):
    return pd.DataFrame([{**dict(zip(by, k if isinstance(k, tuple) else (k,))), **summarize(g, n_boot)}
                         for k, g in m.groupby(by, sort=True)])


def show(title, t):
    keep = [c for c in t.columns if not c.endswith(('_A', '_B')) and c not in ('populations', 'paired_rows')]
    f = t[keep].copy()
    for c in f.columns:
        if c.endswith('_ci_%'):
            f[c] = f[c].map(lambda x: f"[{x[0]:+.1f}, {x[1]:+.1f}]" if np.isfinite(x[0]) else '')
        elif c.endswith('_%'):
            f[c] = f[c].map(lambda x: f"{x:+.1f}")
    print(f"\n== {title}\n{f.to_string(index=False)}")


def n_params(arch, view, protocol):
    m = SetRecovery(arch, feature_view=view, device='cpu')
    m._set_schema(protocol)
    net = m._net()
    return int(sum(p.numel() for p in net.parameters())), m.d_tok, m.d_ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results/mean_variance_server')
    ap.add_argument('--experiment', default='breadth')
    ap.add_argument('--n-boot', type=int, default=500)
    args = ap.parse_args()
    od = os.path.join(args.results_root if os.path.isabs(args.results_root) else os.path.join(ROOT, args.results_root),
                      args.experiment)
    per_round = os.path.join(od, 'per_round')
    scales = Scales(od)
    pairs, costs = [], []
    for arch in ARCHS:
        for protocol in ('PM', 'SR'):
            loaded = {}
            for view in ('minimal', 'context', 'context_report'):
                if protocol == 'SR' and view == 'context_report':
                    continue                                    # alias of context: not separate evidence
                d = load(per_round, method_name(arch, view), protocol)
                if d is None:
                    continue
                loaded[view] = d
                fit = d.drop_duplicates(['partition', 'train_sources', 'epsilon', 'seed'])['fit_time']
                p, dt, dc = n_params(arch, view, protocol)
                costs.append({'protocol': protocol, 'arch': arch, 'view': view, 'd_tok': dt, 'd_ctx': dc,
                              'n_params': p, 'rounds': len(d), 'runtime_ms': 1e3 * d['runtime'].mean(),
                              't_features_ms': 1e3 * d['t_features'].mean(),
                              't_network_ms': 1e3 * d['t_network'].mean(), 'fit_time_s': fit.mean()})
            for vb, va, protos in COMPARISONS:
                if protocol not in protos:
                    continue
                if va not in loaded or vb not in loaded:
                    print(f"{arch} {protocol}: {vb} vs {va} skipped ({'/'.join(v for v in (va, vb) if v not in loaded)} "
                          f"results missing in {per_round})")
                    continue
                ma, mb = method_name(arch, va), method_name(arch, vb)
                p = paired(loaded[va], loaded[vb], scales, ma, mb)
                print(f"{arch} {protocol} {vb} vs {va}: {len(p):,} paired rows "
                      f"({len(loaded[va]):,} / {len(loaded[vb]):,}); loss scales for "
                      f"{p.scale_mean.notna().mean():.0%} of rows")
                pairs.append(p.assign(arch=arch, comparison=f'{vb} vs {va}'))
    if costs:
        c = pd.DataFrame(costs)
        print("\n== costs (per round: end-to-end runtime = feature extraction + network; fit time per unit)")
        print(c.round(3).to_string(index=False))
        c.to_csv(os.path.join(od, 'setfeat_costs.csv'), index=False)
    if not pairs:
        sys.exit("nothing to compare: run the ablation (run_server.sh setfeat) first")
    m = pd.concat(pairs, ignore_index=True)
    g = ['protocol', 'arch', 'comparison']
    out = []
    for title, by, sub in (
            ("by view, clean vs attacked", g + ['view', 'kind'], m),
            ("attacked rounds by attack", g + ['view', 'attack'], m[m['kind'] == 'attacked']),
            ("within-source by eps", g + ['epsilon', 'kind'], m[m['view'] == 'within']),
            ("transfer pairs (train -> test)", g + ['train_sources', 'source', 'kind'], m[m['view'] != 'within'])):
        if sub.empty:
            continue
        t = table(sub, by, args.n_boot)
        show(title, t)
        out.append(t.assign(table=title))
    res = pd.concat(out, ignore_index=True)
    tx = res[(res['table'] == 'transfer pairs (train -> test)') & (res['train_sources'] == 'taxi')
             & (res['source'] == 'income')]
    if len(tx):
        show("taxi -> income", tx.drop(columns=["table"]).dropna(axis=1, how="all"))
    print("\n== verdicts (by view, clean vs attacked): cells where B is clearly worse are listed")
    vt = res[res['table'] == 'by view, clean vs attacked']
    for key, grp in vt.groupby(g, sort=True):
        better = grp[(grp['loss_verdict'] == 'B better') | ((grp['loss_verdict'] == 'n/a')
                                                           & (grp['mean_verdict'] == 'B better'))]
        worse = grp[(grp[[f'{v}_verdict' for v, _ in METRICS]] == 'B worse').any(axis=1)]
        cells = lambda d: ', '.join(f"{r.view}/{r.kind}" for r in d.itertuples()) or 'none'  # noqa: E731
        print(f"  {' '.join(key)}: better (loss) in {cells(better)}; worse on some metric in {cells(worse)}")
    print()
    path = os.path.join(od, 'setfeat_comparison.csv')
    res.to_csv(path, index=False)
    print(f"written: {path}")


if __name__ == '__main__':
    main()
