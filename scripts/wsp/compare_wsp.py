#!/usr/bin/env python
"""Paired comparison: Stage 1 vs fine-tune vs random search vs WSP, on the same units, seeds and test rounds.

  python scripts/wsp/compare_wsp.py        # tables in results/figures/wsp/tables/, summary printed

Tables
  mv_cells.csv      per eval file (unit x variant) x view (within / cross) x status (clean / attacked):
                    rounds, squared-error sums of the method and of no defence (projected standard)
  mv_paired.csv     per unit x variant: within-source MSE ratio to Stage 1 and to no defence (mean, var)
  val_runs.csv      per Stage-2 run: validation metric at Stage 1 and after, accepted edits, stop reason
  freq_test.csv     per frequency run x seed x variant: test accuracy / precision / recall / F1
"""
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

from _common import ROOT

import numpy as np
import pandas as pd

from attacker_detector.wsp.jobs import FIG_DIR, SWEEP, all_jobs, load_yaml, rel

OUT = os.path.join(FIG_DIR, 'tables')
BASES = ('ft_removal', 'deepsets_recovery', 'settransformer_recovery')
COLS = ['method', 'seed', 'source', 'train_sources', 'protocol', 'epsilon', 'attack', 'n_fake', 'mean_hat', 'var_hat',
        'genuine_mean', 'genuine_variance', 'standard_mean', 'standard_variance', 'clf_pr_auc', 'clf_precision',
        'clf_recall', 'removed_frac']


def variant_parts(v):
    """'stage1' -> ('stage1', None); 'wsp' -> ('wsp', 25); 'ft_b10' -> ('ft', 10)."""
    if v == 'stage1':
        return 'stage1', None
    m = re.match(r'^(wsp|ft|rand)(?:_b(\d+))?$', str(v))
    return (m.group(1), int(m.group(2) or 25)) if m else (str(v), None)


def add_parts(df, col='variant'):
    if len(df):
        parts = df[col].map(variant_parts)
        df['vmode'], df['vbudget'] = parts.str[0], parts.str[1]
    return df


def split_method(name):
    for b in BASES:
        if name == b:
            return b, 'stage1'
        if name.startswith(b + '_'):
            return b, name[len(b) + 1:]
    return name, 'other'


def file_cells(path):
    d = pd.read_csv(path, usecols=lambda c: c in COLS)
    base, variant = split_method(str(d['method'].iloc[0]))
    bm = d.standard_mean.clip(-1, 1)
    bv = d.standard_variance.clip(lower=0).clip(upper=1 - bm ** 2)
    d['sse_mean'] = (d.mean_hat - d.genuine_mean) ** 2
    d['sse_var'] = (d.var_hat - d.genuine_variance) ** 2
    d['std_sse_mean'] = (bm - d.genuine_mean) ** 2
    d['std_sse_var'] = (bv - d.genuine_variance) ** 2
    d['view'] = np.where(d.source == d.train_sources, 'within', 'cross')
    d['status'] = np.where(d.n_fake.fillna(0) > 0, 'attacked', 'clean')
    agg = {'n': ('sse_mean', 'size'), 'sse_mean': ('sse_mean', 'sum'), 'sse_var': ('sse_var', 'sum'),
           'std_sse_mean': ('std_sse_mean', 'sum'), 'std_sse_var': ('std_sse_var', 'sum')}
    for c in ('clf_pr_auc', 'clf_precision', 'clf_recall', 'removed_frac'):
        if c in d:
            agg[c] = (c, 'mean')
    g = d.groupby(['view', 'status']).agg(**agg).reset_index()
    g['file'] = os.path.basename(path)
    g['base'], g['variant'] = base, variant
    g['vmode'], g['vbudget'] = variant_parts(variant)
    for c in ('seed', 'protocol', 'epsilon', 'train_sources'):
        g[c] = d[c].iloc[0]
    return g


def mv_cells(eval_dir, jobs_n=8):
    files = sorted(glob.glob(os.path.join(eval_dir, 'per_round', '*.csv')))
    if not files:
        return pd.DataFrame()
    with ProcessPoolExecutor(jobs_n) as ex:
        parts = list(ex.map(file_cells, files, chunksize=2))
    return pd.concat(parts, ignore_index=True)


def mv_paired(cells):
    w = cells[cells.view == 'within'].copy()
    rows = []
    keys = ['base', 'protocol', 'epsilon', 'train_sources', 'seed']
    for k, g in w.groupby(keys):
        ref = g[g.variant == 'stage1']
        for _, r in g.iterrows():
            for status in ('attacked', 'clean'):
                if r.status != status:
                    continue
                s1 = ref[ref.status == status]
                row = dict(zip(keys, k), variant=r.variant, vmode=r.vmode, vbudget=r.vbudget, status=status, n=r.n,
                           mse_mean=r.sse_mean / r.n, mse_var=r.sse_var / r.n,
                           ratio_std_mean=r.sse_mean / r.std_sse_mean, ratio_std_var=r.sse_var / r.std_sse_var)
                if len(s1):
                    row['ratio_stage1_mean'] = r.sse_mean / s1.sse_mean.iloc[0]
                    row['ratio_stage1_var'] = r.sse_var / s1.sse_var.iloc[0]
                rows.append(row)
    return pd.DataFrame(rows)


def val_runs(jobs):
    rows = []
    for j in jobs.values():
        p = os.path.join(j['run_dir'], 'wsp.json')
        if not os.path.exists(p):
            continue
        w = json.load(open(p))
        key = 'f1' if 'f1' in w['val_stage1'] else 'loss'
        rows.append({'job': j['id'], 'kind': j['kind'], 'base': j.get('method', 'freq_ft'), 'mode': j['mode'],
                     'budget': j['budget'], 'protocol': j.get('protocol', j['spec']['protocol'] if 'spec' in j else None),
                     'eps': j.get('eps'), 'run': j.get('run'), 'seed': j['seed'], 'metric': key,
                     'val_stage1': w['val_stage1'][key], 'val_final': w['val_final'][key],
                     'edits_submitted': w['edits_submitted'], 'edits_accepted': w['edits_accepted'], 'stop': w['stop']})
    return pd.DataFrame(rows)


def freq_test(fc):
    rows = []
    for p in glob.glob(os.path.join(ROOT, 'results', fc['out_root'], 'eval', '*', '*', '*', 'test_results.csv')):
        parts = p.split(os.sep)
        m = re.match(r'(.+)_s(\d+)$', parts[-4])
        if not m:
            continue
        r = pd.read_csv(p).iloc[0].to_dict()
        rows.append({'variant': m.group(1), 'seed': int(m.group(2)), 'run': f'{parts[-3]}/{parts[-2]}', **r})
    return add_parts(pd.DataFrame(rows))


def main():
    os.makedirs(OUT, exist_ok=True)
    sw = load_yaml(SWEEP)
    jobs = all_jobs()
    pd.set_option('display.width', 220)
    cells = mv_cells(os.path.join(rel(sw['mv']['out']), 'eval'))
    if len(cells):
        cells.to_csv(os.path.join(OUT, 'mv_cells.csv'), index=False)
        pr = mv_paired(cells)
        pr.to_csv(os.path.join(OUT, 'mv_paired.csv'), index=False)
        s = pr[pr.variant != 'stage1'].groupby(['base', 'variant', 'status'])[
            ['ratio_stage1_mean', 'ratio_stage1_var']].agg(['median', lambda x: (x < 1).mean()])
        s.columns = ['median_ratio_mean', 'share_better_mean', 'median_ratio_var', 'share_better_var']
        print('== mean/variance, within-source test, MSE ratio to Stage 1 (< 1 = better), across units x seeds')
        print(s.round(3).to_string())
    v = val_runs(jobs)
    if len(v):
        v.to_csv(os.path.join(OUT, 'val_runs.csv'), index=False)
        print('== validation (the gate\'s data): Stage 1 -> final, accepted edits')
        print(v.groupby(['base', 'mode', 'budget']).agg(runs=('job', 'count'), val_stage1=('val_stage1', 'mean'),
                                                        val_final=('val_final', 'mean'),
                                                        accepted=('edits_accepted', 'mean')).round(4).to_string())
    f = freq_test(sw['freq'])
    if len(f):
        f.to_csv(os.path.join(OUT, 'freq_test.csv'), index=False)
        print('== frequency oracle, test F1 (mean over seeds)')
        print(f.pivot_table(index='run', columns='variant', values='F1_Score', aggfunc='mean').round(4).to_string())
    print(f'tables: {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
