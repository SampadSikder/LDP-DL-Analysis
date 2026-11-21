"""Comparison tables of the threshold and scaling study (attacker_detector/threshold_scaling_study.py).

Reads <run dir>/scores/<root>/cells/*.csv (one row per held-out run, model and threshold rule; every row of
a run is on the same simulated users) and writes <run dir>/comparison/:
  side_by_side/side_by_side_by_test_beta.csv|.md   every variant next to the others, per attack, kind,
                                                   protocol and test beta
  side_by_side/paired_comparisons.csv|.md          the three paired comparisons, with a verdict per row
  side_by_side/replication_check.csv               retrained four-feature control vs the existing model
  side_by_side/limitations.md                      what differs between rows besides the named change
  in_dist/, cross/sweep_<parameter>.csv            the same metrics along each sweep (data of the figures)
Metrics are means over the held-out runs of a row (per-run F1, recall, precision, false-positive rate,
signed IGR and IGR^2); n_runs says how many.
"""

import glob
import os

import numpy as np
import pandas as pd

# (label, model, threshold rule); the first six are the side-by-side set that was asked for
VARIANTS = [
    ('Fixed beta + fixed threshold', 'fixed_beta_4f_scaled_existing', 'f1'),
    ('Pooled beta + fixed threshold', 'pooled_beta', 'fixed_p0.5'),
    ('Fixed beta + utility threshold', 'fixed_beta_4f_scaled_existing', 'utility'),
    ('Pooled beta + utility threshold', 'pooled_beta', 'utility'),
    ('Fixed beta, 3 features scaled + utility threshold', 'fixed_beta_3f_scaled', 'utility'),
    ('Fixed beta, 3 features unscaled + utility threshold', 'fixed_beta_3f_unscaled', 'utility'),
    ('Pooled beta + fixed threshold p > 1 - beta', 'pooled_beta', 'fixed_prior'),
    ('Pooled beta + utility threshold, all-beta calibration', 'pooled_beta', 'utility_all_beta'),
    ('Fixed beta, 4 features retrained (control) + fixed threshold', 'fixed_beta_4f_scaled_control', 'f1'),
    ('Fixed beta, 4 features retrained (control) + utility threshold', 'fixed_beta_4f_scaled_control', 'utility'),
    ('Fixed beta, 3 features scaled + fixed threshold', 'fixed_beta_3f_scaled', 'f1'),
    ('Fixed beta, 3 features unscaled + fixed threshold', 'fixed_beta_3f_unscaled', 'f1'),
    ('Diffstats', 'diffstats', ''),
    ('FIAD', 'fiad', ''),
    ('No removal', 'no_defense', ''),
    ('Perfect removal', 'oracle', ''),
]
MAIN = [v[0] for v in VARIANTS[:6]]
PAIRS = [
    ('threshold selection on the same pooled checkpoint', 'Pooled beta + fixed threshold',
     'Pooled beta + utility threshold'),
    ('dropping excess_share (fixed beta, utility threshold, same simulated runs)',
     'Fixed beta, 4 features retrained (control) + utility threshold',
     'Fixed beta, 3 features scaled + utility threshold'),
    ('removing the per-run median/IQR scaling (fixed beta, utility threshold, same simulated runs)',
     'Fixed beta, 3 features scaled + utility threshold', 'Fixed beta, 3 features unscaled + utility threshold'),
]
METRICS = ['f1', 'recall', 'precision', 'fpr', 'igr', 'igr2']
F1_TOL, IGR2_REL, IGR2_ABS = 0.01, 0.10, 0.001      # what counts as a change in the verdicts
GROUP = ['attack', 'kind', 'protocol']


def load(cfg):
    frames = []
    for root in cfg['roots']:
        files = sorted(glob.glob(os.path.join(cfg['run_dir'], 'scores', root, 'cells', '*.csv')))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=['nan']) for f in files], ignore_index=True)
        df.to_csv(os.path.join(cfg['run_dir'], 'scores', root, 'scored_runs.csv'), index=False)
        frames.append(df)
    if not frames:
        raise SystemExit('no scored runs; run the score stage first')
    df = pd.concat(frames, ignore_index=True)
    both = df[df.kind == 'both']           # no removal, perfect removal, baselines: the same for both kinds
    df = pd.concat([df[df.kind != 'both']] + [both.assign(kind=k) for k in ('in_dist', 'cross')], ignore_index=True)
    label = {(m, r): lab for lab, m, r in VARIANTS}
    df['variant'] = [label.get((m, r if isinstance(r, str) else '')) for m, r in zip(df.model, df.threshold_rule)]
    df = df[df.variant.notna()].copy()
    for c in METRICS + ['epsilon', 'attacker_ratio']:
        df[c] = pd.to_numeric(df[c], errors='coerce')      # empty = every user flagged, IGR undefined
    return df


def aggregate(df, by):
    g = df.groupby(GROUP + by + ['variant'], sort=False)
    out = g[METRICS].mean().join(g.size().rename('n_runs')).reset_index()
    # a run where a cut-off flags every user has no IGR; it must not vanish from the mean
    emptied = g['igr'].apply(lambda x: int(x.isna().sum())).rename('runs_all_flagged').reset_index()
    out = out.merge(emptied, on=GROUP + by + ['variant'])
    out.loc[out.runs_all_flagged > 0, ['igr', 'igr2']] = np.nan
    none = out[out.variant == 'No removal'][GROUP + by + ['igr', 'igr2']].rename(
        columns={'igr': 'no_removal_igr', 'igr2': 'no_removal_igr2'})
    out = out.merge(none, on=GROUP + by, how='left')
    order = {v[0]: i for i, v in enumerate(VARIANTS)}
    out['_o'] = out.variant.map(order)
    return out.sort_values(GROUP + by + ['_o']).drop(columns='_o').reset_index(drop=True)


def _verdict(b, a):
    df1 = a.f1 - b.f1
    f1 = 'helped' if df1 > F1_TOL else 'hurt' if df1 < -F1_TOL else 'same'
    d = a.igr2 - b.igr2
    big = abs(d) > max(IGR2_ABS, IGR2_REL * b.igr2)
    util = 'same' if not big or not np.isfinite(d) else 'helped' if d < 0 else 'hurt'
    both = {('helped', 'helped'): 'both', ('helped', 'same'): 'F1 only', ('same', 'helped'): 'utility only',
            ('same', 'same'): 'neither'}.get((f1, util), f'F1 {f1}, utility {util}')
    return f1, util, both


def paired(tab, by):
    rows = []
    key = GROUP + by
    for what, before, after in PAIRS:
        b = tab[tab.variant == before].set_index(key)
        a = tab[tab.variant == after].set_index(key)
        for k in b.index.intersection(a.index):
            rb, ra = b.loc[k], a.loc[k]
            f1, util, both = _verdict(rb, ra)
            rows.append({'comparison': what, 'before': before, 'after': after, **dict(zip(key, k)),
                         'n_runs': int(ra.n_runs), **{f'{m}_before': rb[m] for m in METRICS},
                         **{f'{m}_after': ra[m] for m in METRICS}, 'no_removal_igr': ra.no_removal_igr,
                         'no_removal_igr2': ra.no_removal_igr2, 'f1': f1, 'utility': util, 'helped': both})
    return pd.DataFrame(rows)


def _md(df, cols):
    def fmt(v):
        return f'{v:.3f}' if isinstance(v, float) else str(v)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in df[cols].itertuples(index=False)]
    return '\n'.join(lines)


def build(cfg):
    out = os.path.join(cfg['run_dir'], 'comparison')
    side = os.path.join(out, 'side_by_side')
    for d in (side, os.path.join(out, 'in_dist'), os.path.join(out, 'cross')):
        os.makedirs(d, exist_ok=True)
    df = load(cfg)
    df = df.rename(columns={'attacker_ratio': 'test_beta'})
    tab = aggregate(df, ['test_beta'])
    cols = GROUP + ['test_beta', 'variant', 'n_runs'] + METRICS + ['no_removal_igr', 'no_removal_igr2', 'runs_all_flagged']
    tab[cols].to_csv(os.path.join(side, 'side_by_side_by_test_beta.csv'), index=False)

    md = ['# Side by side, by test beta', '',
          'Means over the held-out runs of a row (n_runs). Every variant of a row was scored on the same simulated '
          'users. igr = signed IGR, igr2 = mean squared IGR (0 is best for both); fpr = share of genuine users '
          'flagged; no_removal_igr = signed IGR with every report kept.', '']
    main = tab[tab.variant.isin(MAIN)]
    for (attack, kind, protocol), g in main.groupby(GROUP, sort=False):
        md += [f'## {attack}, {kind}, {protocol}', '',
               _md(g, ['test_beta', 'variant', 'n_runs'] + METRICS + ['no_removal_igr']), '']
    with open(os.path.join(side, 'side_by_side_by_test_beta.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')

    pc = paired(tab, ['test_beta'])
    pc.to_csv(os.path.join(side, 'paired_comparisons.csv'), index=False)
    md = ['# Paired comparisons', '',
          f'Verdicts per attack, kind, protocol and test beta. F1 changed = more than {F1_TOL} either way; utility '
          f'changed = mean IGR^2 moved by more than {int(100 * IGR2_REL)}% of the "before" value and more than '
          f'{IGR2_ABS}. "helped" names what improved.', '']
    if len(pc):
        for what, g in pc.groupby('comparison', sort=False):
            md += [f'## {what}', '', f'{g.before.iloc[0]}  ->  {g.after.iloc[0]}', '',
                   'Rows by verdict: ' + ', '.join(f'{k}: {v}' for k, v in g.helped.value_counts().items()), '',
                   _md(g, GROUP + ['test_beta', 'f1_before', 'f1_after', 'igr_before', 'igr_after', 'igr2_before',
                                   'igr2_after', 'fpr_before', 'fpr_after', 'helped']), '']
    with open(os.path.join(side, 'paired_comparisons.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')

    # retrained control vs the existing fixed-beta model: how much a new simulation of the training runs moves
    rep = []
    for rule in ('fixed', 'utility'):
        a = tab[tab.variant == f'Fixed beta + {rule} threshold'].set_index(GROUP + ['test_beta'])
        b = tab[tab.variant == f'Fixed beta, 4 features retrained (control) + {rule} threshold'].set_index(GROUP + ['test_beta'])
        j = a[['f1', 'igr', 'igr2']].join(b[['f1', 'igr', 'igr2']], lsuffix='_existing', rsuffix='_control', how='inner')
        rep.append(j.assign(threshold=rule).reset_index())
    rep = pd.concat(rep, ignore_index=True) if rep else pd.DataFrame()
    if len(rep):
        rep['f1_diff'] = rep.f1_control - rep.f1_existing
        rep.to_csv(os.path.join(side, 'replication_check.csv'), index=False)

    for kind in ('in_dist', 'cross'):
        d = df[df.kind == kind]
        dflt = {'epsilon': 1.0, 'test_beta': float(next(iter(cfg['roots'].values()))['source_beta']), 'splits': 4}
        for par in ('epsilon', 'test_beta', 'splits'):
            others = [p for p in dflt if p != par]
            sub = d[np.logical_and.reduce([np.isclose(d[p].astype(float), dflt[p]) for p in others])]
            if len(sub):
                aggregate(sub, [par]).to_csv(os.path.join(out, kind, f'sweep_{par}.csv'), index=False)

    lim = ['# What limits the comparisons', '',
           f"- Code version of this run: {cfg.get('git', {}).get('commit')} "
           f"(uncommitted changes: {cfg.get('git', {}).get('uncommitted_changes')}). The reused checkpoints were "
           'trained by earlier commits.',
           '- All rows of a held-out run are on the same simulated users (one simulation per run, seeded like the '
           'dataset: same true items and targets as the dataset\'s held-out run, fresh LDP noise). Numbers therefore '
           'differ slightly from earlier tables, which used other simulations of the same runs.',
           '- Pooled beta and the existing fixed-beta model were trained on the stored datasets. The three new '
           'fixed-beta models were trained on new simulations of the same training runs (same configs, seeds, budget, '
           'quotas; fresh LDP noise), because the server no longer holds the datasets. The retrained four-feature '
           'control is the like-for-like partner of the two three-feature models; replication_check.csv shows how far '
           'it is from the existing model.',
           '- "Fixed beta + utility threshold" uses the existing threshold_utility.json, calibrated on an earlier '
           'simulation of the utility-validation runs. Every other utility threshold here was calibrated on this '
           "run's utility-validation simulations (same configs and seeds).",
           '- Pooled beta + utility threshold: calibration runs are at the source beta only (0.05), the population '
           'the fixed-beta models use, although the checkpoint was trained on six betas. The all-beta calibration row '
           'uses one run of every config of the sweep; runs count equally, and IGR is divided by beta * r, so the '
           'beta = 0.01 run weighs most in mean IGR^2.',
           '- Pooled beta vs fixed beta differ in training data (14 vs 9 configs), in the validation rows and in the '
           'global z-score (whole dataset vs optimization rows), not only in the betas seen.',
           '- The unscaled model still gets one global z-score fitted on its optimization rows (fixed constants, the '
           'same for every run); only the per-run median/IQR step is removed.',
           '- HST_Server has no APA: it is listed under MGA-A only.',
           f"- Baselines on the held-out runs: {', '.join(cfg.get('heldout_baselines', [])) or 'none'}. A baseline "
           'missing from a table failed on that run (see logs).', '']
    if len(rep):
        lim.insert(-1, f'- Replication check: |F1 control - F1 existing| has median {rep.f1_diff.abs().median():.3f} '
                       f'and maximum {rep.f1_diff.abs().max():.3f} over {len(rep)} rows.')
    with open(os.path.join(side, 'limitations.md'), 'w') as f:
        f.write('\n'.join(lim) + '\n')
    print(f'table: {len(df):,} scored rows -> {out}')
    if len(pc):
        print(pc.groupby(['comparison', 'helped']).size().unstack(fill_value=0).to_string())
    return 0
