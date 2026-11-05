#!/usr/bin/env python
"""IGR figures and tables -> results/figures/frequency_oracle/igr/ (PNG) and .../igr/tables/ (CSV).

Reads results/<root>/igr/igr_runs.csv and design_<PROTOCOL>.json (run_igr.py collect; small files,
so they can be copied from the server and drawn locally):

  igr_<root>_<PROTOCOL>.png   IGR vs each swept parameter (one-at-a-time slices), one row per dataset
  igr_<root>_default.png      every protocol / dataset at the design's default point: IGR and MSE
  igr_wsp.png                 WSP runs: Stage 1 vs WSP-edited vs fine-tune / random (when scored)
  tables/igr_<root>_sweeps.csv, tables/igr_<root>_default.csv, tables/igr_wsp.csv

    python scripts/igr/igr_figures.py [--roots oat_b015_mga-a_n500k_v3f ...]

Reading IGR: 1/r = the baseline attack's level (fake users who just report a target honestly);
0 = the attack's gain on the targets is gone; below 0 = genuine supporters of the targets were removed
too. The y axis is symlog (linear inside +-1/r) because an undefended attack reaches IGR >> 1 at small
epsilon.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS = os.environ.get('IGR_RESULTS') or os.path.join(ROOT, 'results')
OUT = os.environ.get('IGR_FIG_DIR') or os.path.join(RESULTS, 'figures', 'frequency_oracle', 'igr')

# Same validated slots as scripts/plots/style.py (blue, orange, aqua, violet; all-pairs on the light
# surface: normal-vision dE >= 16.3, CVD dE >= 9.2). Reference series are grays with their own line
# style and marker, so no identity rests on colour alone.
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'

SERIES = {
    'ours_in_dist': dict(label='Ours (in-distribution)', color=BLUE, ls='-', marker='o', lw=2.0, ms=5, fill='full'),
    'ours_cross':   dict(label='Ours (cross-dataset)', color=BLUE, ls='--', marker='o', lw=1.6, ms=5, fill='none'),
    'fiad':         dict(label='FIAD', color=ORANGE, ls='--', marker='^', lw=1.4, ms=5, fill='none'),
    'diffstats':    dict(label='Diffstats', color=AQUA, ls=':', marker='s', lw=1.4, ms=4.5, fill='none'),
    'no_defense':   dict(label='No defence', color=MUTED, ls='-', marker='x', lw=1.2, ms=4.5, fill='full'),
    'oracle':       dict(label='Perfect removal (LDP noise only)', color=INK2, ls=(0, (1, 1.5)), marker=None,
                         lw=1.2, ms=0, fill='full'),
}
WSP_SERIES = {
    'stage1':   dict(label='Stage 1 (gradient)', color=BLUE, ls='-', marker='o', lw=2.0, ms=5, fill='full'),
    'wsp':      dict(label='WSP (Codex-edited)', color=VIOLET, ls='-', marker='D', lw=2.0, ms=4.5, fill='full'),
    'finetune': dict(label='Fine-tune baseline', color=MUTED, ls='--', marker='v', lw=1.2, ms=4.5, fill='none'),
    'random':   dict(label='Random-search baseline', color=MUTED, ls=':', marker='<', lw=1.2, ms=4.5, fill='none'),
    'fiad':     SERIES['fiad'], 'diffstats': SERIES['diffstats'], 'no_defense': SERIES['no_defense'],
}
PARAMS = ['epsilon', 'attacker_ratio', 'splits', 'target_set_size']
PARAM_LABEL = {'epsilon': r'$\epsilon$', 'attacker_ratio': r'$\beta$', 'splits': r"$r'$", 'target_set_size': r'$r$'}
DATASETS = ['zipf', 'emoji', 'fire']
PROTOCOLS = ['OUE', 'OLH_Server', 'HST_Server', 'HST_User']
PROTOCOL_LABEL = {'OUE': 'OUE', 'OLH_Server': 'OLH-Server', 'HST_Server': 'HST-Server', 'HST_User': 'HST-User'}


def apply_style():
    matplotlib.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'font.family': ['DejaVu Sans'], 'font.size': 9,
        'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})


def load(root):
    p = os.path.join(RESULTS, root, 'igr', 'igr_runs.csv')
    if not os.path.exists(p):
        return None, {}
    df = pd.read_csv(p, keep_default_na=False, na_values=[''])
    for c in ('epsilon', 'attacker_ratio', 'igr', 'mse', 'f1', 'flagged', 'n'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ('splits', 'target_set_size'):
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(int)
    df['variant'] = df['variant'].fillna('').astype(str)
    designs = {}
    for f in glob.glob(os.path.join(RESULTS, root, 'igr', 'design_*.json')):
        with open(f) as fh:
            designs[os.path.basename(f)[len('design_'):-len('.json')]] = json.load(fh)
    return df, designs


def swept(design):
    s = design.get('sweeps', {})
    return [p for p in PARAMS if len(s.get(p, [])) > 1]


def oat_slice(df, design, param):
    """Rows where every parameter except `param` sits at the design default."""
    d = design.get('defaults', {})
    m = np.ones(len(df), dtype=bool)
    for p in PARAMS:
        if p != param and p in d:
            m &= np.isclose(df[p].astype(float), float(d[p]))
    return df[m]


def plot_series(ax, g, x, st, label=None):
    g = g.groupby(x, as_index=False)['igr'].mean().sort_values(x)
    ax.plot(g[x], g['igr'], color=st['color'], linestyle=st['ls'], marker=st['marker'], linewidth=st['lw'],
            markersize=st['ms'], fillstyle=st['fill'], markeredgewidth=1.1, label=label or st['label'], zorder=3)


def _plain(v, _=None):
    return '0' if v == 0 else f'{v:g}'


def symlog_ticks(axis, lim, r):
    """Readable ticks for a symlog IGR axis: 0, +-1/r and the decades beyond."""
    lo, hi = lim
    cand = [0.0, 1.0 / r, -1.0 / r] + [s * 10.0 ** k for k in range(0, 4) for s in (1, -1)]
    axis.set_major_locator(FixedLocator(sorted(t for t in cand if lo <= t <= hi)))
    axis.set_major_formatter(FuncFormatter(_plain))
    axis.set_minor_locator(NullLocator())


def symlog_axis(ax, r):
    ax.set_yscale('symlog', linthresh=1.0 / r, linscale=1.0)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.axhline(1.0 / r, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)


def legend_handles(series, keys):
    out = []
    for k in keys:
        st = series[k]
        out.append(Line2D([], [], color=st['color'], linestyle=st['ls'], marker=st['marker'], linewidth=st['lw'],
                          markersize=st['ms'], fillstyle=st['fill'], markeredgewidth=1.1, label=st['label']))
    out.append(Line2D([], [], color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), label='Baseline attack (1/r)'))
    return out


def protocol_figure(root, protocol, df, design):
    params = swept(design)
    if not params:
        return None
    d = df[(df.protocol == protocol) & (df.variant.isin(['', 'original']))]
    if d.empty:
        return None
    one = len(params) == 1
    datasets = [x for x in DATASETS if x in set(d.dataset_type)]
    nr, nc = (1, len(datasets)) if one else (len(datasets), len(params))
    fig, axes = plt.subplots(nr, nc, figsize=(max(3.5 * nc, 7.0), 2.7 * nr + 0.9), sharey=True, squeeze=False)
    r = int(design['defaults'].get('target_set_size', 10))
    keys = [k for k in SERIES if k in set(d.detector)]
    for i, ds in enumerate(datasets):
        for j, p in enumerate(params):
            ax = axes[0, i] if one else axes[i, j]
            s = oat_slice(d[d.dataset_type == ds], design, p)
            for k in keys:
                if len(s[s.detector == k]):
                    plot_series(ax, s[s.detector == k], p, SERIES[k])
            symlog_axis(ax, r)
            if p == 'attacker_ratio':
                ax.set_xscale('log')
            ax.set_title(ds if one or j == 0 else '', loc='left', fontsize=9.5, color=INK)
            if one or i == len(datasets) - 1:
                ax.set_xlabel(PARAM_LABEL[p])
            if (one and i == 0) or (not one and j == 0):
                ax.set_ylabel('IGR' if one else f'{ds}\nIGR')
    if not one:
        for j, p in enumerate(params):
            axes[0, j].set_title(f'{datasets[0]}, vs {PARAM_LABEL[p]}', loc='left', fontsize=9.5)
    symlog_ticks(axes[0, 0].yaxis, axes[0, 0].get_ylim(), r)
    dflt = design.get('defaults', {})
    fig.suptitle(f'{PROTOCOL_LABEL[protocol]}: item gain ratio after removing the flagged users',
                 x=0.01, y=0.995, ha='left', va='top', fontsize=10.5, color=INK)
    fig.text(0.01, 0.995 - 0.3 / fig.get_figheight(),
             f'{root} · other parameters at ε={dflt.get("epsilon")}, β={dflt.get("attacker_ratio")}, r={r}, '
             f'r\'={dflt.get("splits")}', ha='left', va='top', fontsize=8.5, color=INK2)
    fig.legend(handles=legend_handles(SERIES, keys), loc='lower center', ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    H = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.55 / H, 1, 1 - 0.55 / H))
    path = os.path.join(OUT, f'igr_{root}_{protocol}.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def default_table(df, designs):
    rows = []
    for protocol, design in designs.items():
        d = df[(df.protocol == protocol) & (df.variant.isin(['', 'original']))]
        dflt = design.get('defaults', {})
        m = np.ones(len(d), dtype=bool)
        for p in PARAMS:
            if p in dflt:
                m &= np.isclose(d[p].astype(float), float(dflt[p]))
        rows.append(d[m])
    if not rows:
        return pd.DataFrame()
    t = pd.concat(rows).groupby(['protocol', 'dataset_type', 'detector'], as_index=False).agg(
        igr=('igr', 'mean'), mse=('mse', 'mean'), f1=('f1', 'mean'), flagged=('flagged', 'mean'), runs=('igr', 'size'))
    return t


def default_figure(root, t):
    if t.empty:
        return None
    cells = [(p, d) for p in PROTOCOLS for d in DATASETS if len(t[(t.protocol == p) & (t.dataset_type == d)])]
    keys = [k for k in SERIES if k in set(t.detector)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.42 * len(cells) + 1.6), sharey=True)
    y = np.arange(len(cells))[::-1]
    for k_i, k in enumerate(keys):
        st = SERIES[k]
        off = (k_i - (len(keys) - 1) / 2) * 0.11
        for ax, col in zip(axes, ('igr', 'mse')):
            vals = [t[(t.protocol == p) & (t.dataset_type == d) & (t.detector == k)][col] for p, d in cells]
            v = np.array([x.iloc[0] if len(x) else np.nan for x in vals])
            ax.plot(v, y + off, linestyle='none', marker=st['marker'] or 'd', color=st['color'],
                    markersize=max(st['ms'], 5), fillstyle=st['fill'], markeredgewidth=1.2, label=st['label'], zorder=3)
    r = 10 if t.empty or 'target_set_size' not in t else int(t['target_set_size'].iloc[0])
    axes[0].set_xscale('symlog', linthresh=1.0 / r)
    axes[0].axvline(0, color=AXIS, linewidth=0.8)
    axes[0].axvline(1.0 / r, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    symlog_ticks(axes[0].xaxis, axes[0].get_xlim(), r)
    axes[0].set_xlabel('IGR (symlog; dashed = 1/r)')
    axes[1].set_xscale('log')
    lo, hi = axes[1].get_xlim()
    axes[1].xaxis.set_major_locator(FixedLocator(np.geomspace(lo, hi, 5)[1:-1]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.2g}'))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel('MSE over all items (log)')
    axes[0].set_yticks(y, [f'{PROTOCOL_LABEL[p]} · {d}' for p, d in cells])
    for ax in axes:
        ax.grid(axis='y', visible=False)
    fig.suptitle(f'{root}: default point, removal of flagged users', x=0.01, ha='left', fontsize=10.5)
    hs = [Line2D([], [], linestyle='none', marker=SERIES[k]['marker'] or 'd', color=SERIES[k]['color'],
                 markersize=max(SERIES[k]['ms'], 5), fillstyle=SERIES[k]['fill'], markeredgewidth=1.2,
                 label=SERIES[k]['label']) for k in keys]
    fig.legend(handles=hs, loc='lower center', ncol=3, fontsize=8.5)
    H = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.55 / H, 1, 1 - 0.35 / H))
    path = os.path.join(OUT, f'igr_{root}_default.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def wsp_figure(df, root):
    v = df[df.variant.isin(list(WSP_SERIES))]
    if v.empty:
        return None, pd.DataFrame()
    panels = sorted(v.groupby(['protocol', 'dataset_type', 'detector']).groups)
    fig, axes = plt.subplots(1, len(panels), figsize=(max(3.4 * len(panels), 8.0), 3.6), sharey=True, squeeze=False)
    present = set()
    table = []
    for ax, (protocol, ds, det) in zip(axes[0], panels):
        cell = df[(df.protocol == protocol) & (df.dataset_type == ds)]
        ref = cell[cell.variant.isin(['', 'original'])]
        for k, st in WSP_SERIES.items():
            g = cell[(cell.detector == det) & (cell.variant == k)] if k in ('stage1', 'wsp', 'finetune', 'random') \
                else ref[ref.detector == k]
            if g.empty:
                continue
            present.add(k)
            plot_series(ax, g, 'epsilon', st)
            a = g.groupby('epsilon', as_index=False).agg(igr=('igr', 'mean'), igr_sd=('igr', 'std'), mse=('mse', 'mean'),
                                                         f1=('f1', 'mean'), runs=('igr', 'size'))
            a.insert(0, 'series', k)
            a.insert(0, 'model', det)
            a.insert(0, 'dataset_type', ds)
            a.insert(0, 'protocol', protocol)
            table.append(a)
        symlog_axis(ax, int(cell.target_set_size.iloc[0]))
        ax.set_title(f"{PROTOCOL_LABEL[protocol]} · {ds} · {'cross' if det == 'ours_cross' else 'in-dist'}",
                     loc='left', fontsize=9.5)
        ax.set_xlabel(PARAM_LABEL['epsilon'])
    axes[0, 0].set_ylabel('IGR')
    symlog_ticks(axes[0, 0].yaxis, axes[0, 0].get_ylim(), int(df.target_set_size.iloc[0]))
    fig.suptitle(f'{root}: IGR of the WSP runs (lines = mean over seeds)', x=0.01, ha='left', fontsize=10.5)
    fig.legend(handles=legend_handles(WSP_SERIES, [k for k in WSP_SERIES if k in present]), loc='lower center',
               ncol=4, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.2, 1, 0.92))
    path = os.path.join(OUT, 'igr_wsp.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, pd.concat(table, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+')
    args = ap.parse_args()
    roots = args.roots or sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                                 for p in glob.glob(os.path.join(RESULTS, '*', 'igr', 'igr_runs.csv')))
    if not roots:
        print(f'no results/*/igr/igr_runs.csv under {RESULTS}; run scripts/igr/run_igr.py collect first')
        return 1
    apply_style()
    os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
    made = []
    for root in roots:
        df, designs = load(root)
        if df is None:
            continue
        for protocol in PROTOCOLS:
            if protocol in designs:
                made.append(protocol_figure(root, protocol, df, designs[protocol]))
        t = default_table(df, designs)
        if not t.empty:
            t.to_csv(os.path.join(OUT, 'tables', f'igr_{root}_default.csv'), index=False)
        made.append(default_figure(root, t))
        sweeps = []
        for protocol, design in designs.items():
            for p in swept(design):
                s = oat_slice(df[(df.protocol == protocol) & (df.variant.isin(['', 'original']))], design, p)
                a = s.groupby(['protocol', 'dataset_type', 'detector', p], as_index=False).agg(
                    igr=('igr', 'mean'), mse=('mse', 'mean'), f1=('f1', 'mean'), flagged=('flagged', 'mean'),
                    runs=('igr', 'size')).rename(columns={p: 'value'})
                a.insert(3, 'param', p)
                sweeps.append(a)
        if sweeps:
            pd.concat(sweeps, ignore_index=True).to_csv(os.path.join(OUT, 'tables', f'igr_{root}_sweeps.csv'), index=False)
        path, wt = wsp_figure(df, root)
        if path:
            made.append(path)
            wt.to_csv(os.path.join(OUT, 'tables', 'igr_wsp.csv'), index=False)
    made = [m for m in made if m]
    print(f'{len(made)} figures -> {OUT}')
    for m in made:
        print('  ' + os.path.relpath(m, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
