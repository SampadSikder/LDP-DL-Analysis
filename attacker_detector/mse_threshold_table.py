"""Comparison tables of the MSE threshold study (attacker_detector/mse_threshold_study.py).

Reads <run dir>/scores/<root>/cells/*.csv (one row per held-out run, model and threshold rule; every row of
a run is on the same simulated users) and writes <run dir>/comparison/:
  side_by_side/side_by_side_by_test_beta.csv|.md   every version next to the others, per attack, kind,
                                                   protocol and test beta
  side_by_side/paired_comparisons.csv|.md          the paired comparisons (PAIRS), with a verdict per row
  side_by_side/best_igr_reference.csv              which version of the threshold and scaling study is the
                                                   reference row of each attack, kind and protocol
  side_by_side/limitations.md                      what differs between rows besides the named change
  in_dist/, cross/sweep_<parameter>.csv            the same metrics along each sweep (data of the figures)
Metrics are means over the held-out runs of a row (per-run F1, recall, precision, false-positive rate,
target-item MSE, signed IGR and IGR^2); n_runs says how many.
"""

import glob
import os

import numpy as np
import pandas as pd

from . import threshold_scaling_table as tb

# (label, model, threshold rule). Every model here has no per-run median/IQR scaling. The first six are
# the side-by-side set: the earlier study's six versions with the MSE cut-off as the dynamic threshold.
VARIANTS = [
    ('Fixed beta + fixed threshold', 'fixed_beta_4f_unscaled', 'f1'),
    ('Pooled beta + fixed threshold', 'pooled_beta_4f_unscaled', 'fixed_p0.5'),
    ('Fixed beta + MSE threshold', 'fixed_beta_4f_unscaled', 'mse'),
    ('Pooled beta + MSE threshold', 'pooled_beta_4f_unscaled', 'mse'),
    ('Fixed beta, 3 features + fixed threshold', 'fixed_beta_3f_unscaled', 'f1'),
    ('Fixed beta, 3 features + MSE threshold', 'fixed_beta_3f_unscaled', 'mse'),
    # the IGR cut-off of the same model, selected on the same utility-validation runs
    ('Fixed beta + IGR threshold', 'fixed_beta_4f_unscaled', 'igr'),
    ('Pooled beta + IGR threshold', 'pooled_beta_4f_unscaled', 'igr'),
    ('Fixed beta, 3 features + IGR threshold', 'fixed_beta_3f_unscaled', 'igr'),
    ('Pooled beta + MSE threshold, all-beta calibration', 'pooled_beta_4f_unscaled', 'mse_all_beta'),
    ('Pooled beta + IGR threshold, all-beta calibration', 'pooled_beta_4f_unscaled', 'igr_all_beta'),
    ('Pooled beta + validation-F1 threshold', 'pooled_beta_4f_unscaled', 'f1'),
    ('Diffstats', 'diffstats', ''),
    ('FIAD', 'fiad', ''),
    ('No removal', 'no_defense', ''),
    ('Perfect removal', 'oracle', ''),
]
MAIN = [v[0] for v in VARIANTS[:6]]
# every version of the threshold and scaling study with an IGR-selected threshold, at the threshold that
# study froze; the best of them per attack, kind and protocol is the reference row
SOURCE = ' (threshold and scaling study)'
SOURCE_IGR = [(lab + SOURCE, m, r) for lab, m, r in tb.VARIANTS if r in ('utility', 'utility_all_beta')]
BEST = 'Best IGR-cutoff version of the threshold and scaling study'
PAIRS = [
    ('threshold selection on the same pooled-beta checkpoint: fixed -> MSE', 'Pooled beta + fixed threshold',
     'Pooled beta + MSE threshold'),
    ('threshold selection on the same fixed-beta checkpoint: fixed -> MSE', 'Fixed beta + fixed threshold',
     'Fixed beta + MSE threshold'),
    ('dropping excess_share (fixed beta, MSE threshold)', 'Fixed beta + MSE threshold',
     'Fixed beta, 3 features + MSE threshold'),
    ('training on every beta instead of the source beta (MSE threshold)', 'Fixed beta + MSE threshold',
     'Pooled beta + MSE threshold'),
    ('objective of the threshold on the same pooled-beta checkpoint: IGR -> MSE', 'Pooled beta + IGR threshold',
     'Pooled beta + MSE threshold'),
    ('objective of the threshold on the same fixed-beta checkpoint: IGR -> MSE', 'Fixed beta + IGR threshold',
     'Fixed beta + MSE threshold'),
    ('objective of the threshold on the same three-feature checkpoint: IGR -> MSE',
     'Fixed beta, 3 features + IGR threshold', 'Fixed beta, 3 features + MSE threshold'),
    ('calibration runs of the pooled-beta MSE threshold: source beta -> every beta', 'Pooled beta + MSE threshold',
     'Pooled beta + MSE threshold, all-beta calibration'),
]
METRICS = ['f1', 'recall', 'precision', 'fpr', 'mse_target', 'igr', 'igr2']
F1_TOL, REL = 0.01, 0.10       # what counts as a change in the verdicts (relative for MSE and IGR^2)
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
    rule = df.threshold_rule.where(df.threshold_rule.notna(), '').astype(str)
    parts = []
    for table in (VARIANTS, SOURCE_IGR):   # a row can be a version of this study and of the source study
        label = {(m, r): lab for lab, m, r in table}
        v = pd.Series([label.get(k) for k in zip(df.model, rule)], index=df.index)
        parts.append(df[v.notna()].assign(variant=v[v.notna()]))
    df = pd.concat(parts, ignore_index=True)
    for c in METRICS + ['epsilon', 'attacker_ratio']:
        df[c] = pd.to_numeric(df[c], errors='coerce')      # empty = every user flagged, nothing to estimate
    return df


def best_reference(df):
    """Per attack, kind and protocol: the source-study version with the lowest mean squared IGR over all the
    held-out runs of the group (that study's own criterion). Chosen on test results, so it is the most
    favourable version of that study, not one that could have been picked in advance."""
    src = df[df.variant.str.endswith(SOURCE)]
    if src.empty:
        return pd.DataFrame(columns=GROUP + ['variant', 'mean_igr2', 'mean_mse_target', 'mean_f1', 'candidates'])
    g = src.groupby(GROUP + ['variant'], sort=False)
    stat = g.agg(mean_igr2=('igr2', lambda x: np.inf if x.isna().any() else x.mean()),
                 mean_mse_target=('mse_target', 'mean'), mean_f1=('f1', 'mean')).reset_index()
    stat['candidates'] = stat.groupby(GROUP).variant.transform('size')
    return stat.sort_values('mean_igr2', kind='stable').groupby(GROUP, sort=False).head(1).reset_index(drop=True)


def with_reference(df, best):
    """df plus the reference rows: the chosen source-study version of each group under the label BEST."""
    if best.empty:
        return df
    pick = df.merge(best[GROUP + ['variant']], on=GROUP + ['variant'])
    return pd.concat([df, pick.assign(reference_version=pick.variant, variant=BEST)], ignore_index=True)


def aggregate(df, by):
    order = {lab: i for i, lab in enumerate([v[0] for v in VARIANTS] + [BEST] + [v[0] for v in SOURCE_IGR])}
    g = df.groupby(GROUP + by + ['variant'], sort=False)
    out = g[METRICS].mean().join(g.size().rename('n_runs')).reset_index()
    # a run where a cut-off flags every user has no estimate; it must not vanish from the mean
    emptied = g['igr'].apply(lambda x: int(x.isna().sum())).rename('runs_all_flagged').reset_index()
    out = out.merge(emptied, on=GROUP + by + ['variant'])
    out.loc[out.runs_all_flagged > 0, ['igr', 'igr2', 'mse_target']] = np.nan
    if 'reference_version' in df.columns:
        ref = df[df.variant == BEST].groupby(GROUP).reference_version.first().reset_index()
        out = out.merge(ref, on=GROUP, how='left')
        out.loc[out.variant != BEST, 'reference_version'] = ''
    none = out[out.variant == 'No removal'][GROUP + by + ['igr', 'igr2', 'mse_target']].rename(
        columns={'igr': 'no_removal_igr', 'igr2': 'no_removal_igr2', 'mse_target': 'no_removal_mse_target'})
    out = out.merge(none, on=GROUP + by, how='left')
    out['_o'] = out.variant.map(order)
    return out.sort_values(GROUP + by + ['_o']).drop(columns='_o').reset_index(drop=True)


def _change(before, after):
    d = after - before
    if not np.isfinite(d) or abs(d) <= REL * before:
        return 'same'
    return 'helped' if d < 0 else 'hurt'


def paired(tab, by):
    rows, key = [], GROUP + by
    for what, before, after in PAIRS:
        b = tab[tab.variant == before].set_index(key)
        a = tab[tab.variant == after].set_index(key)
        for k in b.index.intersection(a.index):
            rb, ra = b.loc[k], a.loc[k]
            df1 = ra.f1 - rb.f1
            rows.append({'comparison': what, 'before': before, 'after': after, **dict(zip(key, k)),
                         'n_runs': int(ra.n_runs), **{f'{m}_before': rb[m] for m in METRICS},
                         **{f'{m}_after': ra[m] for m in METRICS}, 'no_removal_igr2': ra.no_removal_igr2,
                         'no_removal_mse_target': ra.no_removal_mse_target,
                         'f1': 'helped' if df1 > F1_TOL else 'hurt' if df1 < -F1_TOL else 'same',
                         'mse': _change(rb.mse_target, ra.mse_target), 'igr': _change(rb.igr2, ra.igr2)})
    return pd.DataFrame(rows)


def _md(df, cols):
    def fmt(v):
        if isinstance(v, float):
            return f'{v:.3e}' if 0 < abs(v) < 1e-3 else f'{v:.3f}'
        return str(v)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in df[cols].itertuples(index=False)]
    return '\n'.join(lines)


def build(cfg):
    out = os.path.join(cfg['run_dir'], 'comparison')
    side = os.path.join(out, 'side_by_side')
    for d in (side, os.path.join(out, 'in_dist'), os.path.join(out, 'cross')):
        os.makedirs(d, exist_ok=True)
    df = load(cfg).rename(columns={'attacker_ratio': 'test_beta'})
    best = best_reference(df)
    best.to_csv(os.path.join(side, 'best_igr_reference.csv'), index=False)
    df = with_reference(df, best)
    tab = aggregate(df, ['test_beta'])
    cols = GROUP + ['test_beta', 'variant', 'n_runs'] + METRICS + ['no_removal_mse_target', 'no_removal_igr',
                                                                  'no_removal_igr2', 'runs_all_flagged']
    cols += ['reference_version'] if 'reference_version' in tab.columns else []
    tab[cols].to_csv(os.path.join(side, 'side_by_side_by_test_beta.csv'), index=False)

    md = ['# Side by side, by test beta', '',
          'Means over the held-out runs of a row (n_runs). Every version of a row was scored on the same simulated '
          'users. mse_target = mean squared error of the target items after removal; igr = signed IGR, igr2 = mean '
          'squared IGR (0 is best for all three); fpr = share of genuine users flagged. No model has per-run '
          f'median/IQR scaling. "{BEST}" is named per group in best_igr_reference.csv.', '']
    main = tab[tab.variant.isin(MAIN + [BEST, 'Diffstats', 'No removal'])]
    for (attack, kind, protocol), g in main.groupby(GROUP, sort=False):
        md += [f'## {attack}, {kind}, {protocol}', '',
               _md(g, ['test_beta', 'variant', 'n_runs'] + METRICS), '']
    with open(os.path.join(side, 'side_by_side_by_test_beta.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')

    pc = paired(tab, ['test_beta'])
    pc.to_csv(os.path.join(side, 'paired_comparisons.csv'), index=False)
    md = ['# Paired comparisons', '',
          f'Verdicts per attack, kind, protocol and test beta. F1 changed = more than {F1_TOL} either way; MSE and '
          f'IGR changed = the mean target-item MSE / the mean IGR^2 moved by more than {int(100 * REL)}% of the '
          f'"before" value.', '']
    if len(pc):
        for what, g in pc.groupby('comparison', sort=False):
            md += [f'## {what}', '', f'{g.before.iloc[0]}  ->  {g.after.iloc[0]}', '',
                   'F1: ' + ', '.join(f'{k} {v}' for k, v in g.f1.value_counts().items())
                   + '; MSE: ' + ', '.join(f'{k} {v}' for k, v in g.mse.value_counts().items())
                   + '; IGR: ' + ', '.join(f'{k} {v}' for k, v in g.igr.value_counts().items()), '',
                   _md(g, GROUP + ['test_beta', 'f1_before', 'f1_after', 'mse_target_before', 'mse_target_after',
                                   'igr2_before', 'igr2_after', 'fpr_before', 'fpr_after', 'f1', 'mse', 'igr']), '']
    with open(os.path.join(side, 'paired_comparisons.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')

    for kind in ('in_dist', 'cross'):
        d = df[df.kind == kind]
        dflt = {'epsilon': 1.0, 'test_beta': float(next(iter(cfg['roots'].values()))['source_beta']), 'splits': 4}
        for par in ('epsilon', 'test_beta', 'splits'):
            others = [p for p in dflt if p != par]
            sub = d[np.logical_and.reduce([np.isclose(d[p].astype(float), dflt[p]) for p in others])]
            if len(sub):
                aggregate(sub, [par]).to_csv(os.path.join(out, kind, f'sweep_{par}.csv'), index=False)

    s = cfg.get('source_study', {})
    lim = ['# What limits the comparisons', '',
           f"- Code version of this run: {cfg.get('git', {}).get('commit')} (uncommitted changes: "
           f"{cfg.get('git', {}).get('uncommitted_changes')}). Models: {cfg.get('source_run_dir')}, created "
           f"{s.get('created')} by {(s.get('git') or {}).get('commit')}; nothing was trained here.",
           '- No model has per-run median/IQR scaling. Each keeps the one z-score fitted on its training rows.',
           '- All rows of a held-out run are on the same simulated users. These are new simulations of the held-out '
           'runs (same designs and seeds: same true items and targets, fresh LDP noise), because the earlier ones '
           'did not keep the per-target support. Numbers therefore differ slightly from the threshold and scaling '
           'study.',
           '- The MSE and the IGR threshold of a model were selected on the same new utility-validation runs '
           '(source beta; the pooled-beta model also on every config). The rows marked "(threshold and scaling '
           'study)" use the thresholds that study froze, calibrated on its own simulations of those runs.',
           f'- "{BEST}" is chosen per attack, kind and protocol on these held-out runs (lowest mean IGR^2 over all '
           'runs of the group), among every version of that study with an IGR-selected threshold, scaled models '
           'included. It is the most favourable version of that study, not one that could be picked in advance.',
           '- The target-item MSE is in frequency units and is not divided by beta. In a mean over runs, a run with '
           'a large beta or a small epsilon weighs more. IGR is divided by beta * r, so a run at beta = 0.01 weighs '
           'most there. With the source-beta calibration every run is at beta = 0.05 and the two objectives differ '
           'in how the items are combined (sum of the errors for IGR, each item squared for MSE).',
           '- The pooled-beta model was trained on six betas; its primary thresholds are calibrated at beta = 0.05 '
           'only, like the fixed-beta models. The all-beta rows use one run of every config.',
           '- HST_Server has no APA: it is listed under MGA-A only.',
           f"- Baselines on the held-out runs: {', '.join(cfg.get('heldout_baselines', [])) or 'none'}. A baseline "
           'missing from a table failed on that run (see logs).', '']
    with open(os.path.join(side, 'limitations.md'), 'w') as f:
        f.write('\n'.join(lim) + '\n')
    print(f'table: {len(df):,} scored rows -> {out}')
    if len(pc):
        print(pc.groupby(['comparison', 'mse']).size().unstack(fill_value=0).to_string())
    if len(best):
        print(best[GROUP + ['variant', 'mean_igr2', 'mean_mse_target']].to_string(index=False))
    return 0
