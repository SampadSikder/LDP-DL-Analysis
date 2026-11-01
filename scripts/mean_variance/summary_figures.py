#!/usr/bin/env python
"""Summary figures and tables from a server results folder; per-round files are not needed
(except below_budget/per_round, which is small).

  python scripts/mean_variance/summary_figures.py
  python scripts/mean_variance/summary_figures.py --results-root results/mean_variance_server \
      --out results/figures/mean_variance_server

MSE comparisons are against the `standard` method's own output on the same rounds. aggregate.csv's
mse_*_before / ag_* / frac_worse use the unprojected label-based standard estimate instead, which is
far noisier at eps 0.1 and flatters every projected method, so they are not used here.
Writes PNGs to --out and CSV tables to --out/tables.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e6e5e0', '#b5b4ad'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
# key: (label, colour, marker, linestyle, linewidth). References in ink; defences take palette slots in order.
METHODS = {
    'standard': ('Standard (no defence)', INK2, 'o', '-', 2.4),
    'oracle_removal': ('Oracle removal (uses labels)', INK, 'x', '--', 1.3),
    'constant': ('Constant (training mean)', '#a9a8a1', 's', ':', 1.6),
    'kmeans': ('k-means (Li et al.)', SLOTS[0], 'D', '-', 1.5),
    'mlp_removal': ('MLP removal', SLOTS[1], '^', '-', 1.5),
    'ft_removal': ('FT-Transformer removal', SLOTS[2], 'v', '-', 1.5),
    'catboost_recovery': ('CatBoost recovery', SLOTS[3], 'P', '-', 1.5),
    'deepsets_recovery': ('Deep Sets recovery', SLOTS[4], 'h', '-', 1.5),
    'settransformer_recovery': ('Set Transformer recovery', SLOTS[5], '*', '-', 1.5),
}
ATTACK_STYLE = {'IPA': SLOTS[6], 'OPA': SLOTS[1]}
PROTO_LS = {'PM': '-', 'SR': '--'}
DIVERGING = LinearSegmentedColormap.from_list('better_worse', [
    (0.0, '#104281'), (0.25, '#3987e5'), (0.45, '#cde2fb'), (0.5, '#f0efec'),
    (0.55, '#f8d3cf'), (0.75, '#e34948'), (1.0, '#8f1f1f')])
SEQ_BLUE = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
ORDINAL_BLUE = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'figure.titlesize': 12,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': AXIS, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'legend.frameon': False, 'lines.markersize': 5, 'axes.titleweight': 'bold',
})


def style(m, **kw):
    lab, c, mk, ls, lw = METHODS[m]
    return {'label': lab, 'color': c, 'marker': mk, 'linestyle': ls, 'linewidth': lw, **kw}


def method_legend(fig, methods, y=1.0, ncol=5):
    h = [Line2D([], [], **style(m)) for m in methods]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=8.5,
               handlelength=2.6, columnspacing=1.6)


def header(fig, title, sub, y=1.07):
    fig.text(0.01, y + 0.3 / fig.get_figheight(), title, fontsize=13, fontweight='bold', ha='left', va='bottom')
    fig.text(0.01, y, sub, fontsize=8.5, color=INK2, ha='left', va='bottom')


class Out:
    def __init__(self, d):
        self.d = d
        os.makedirs(os.path.join(d, 'tables'), exist_ok=True)
        self.figs, self.tables = [], []

    def fig(self, fig, name):
        fig.savefig(os.path.join(self.d, name + '.png'), dpi=180, bbox_inches='tight', pad_inches=0.25)
        plt.close(fig)
        self.figs.append(name)
        print(f'  {name}.png')

    def table(self, df, name):
        df.to_csv(os.path.join(self.d, 'tables', name + '.csv'), index=False, float_format='%.6g')
        self.tables.append(name)


# ---------------------------------------------------------------- data

def load_breadth(root):
    a = pd.read_csv(os.path.join(root, 'breadth', 'aggregate.csv'))
    a['attack'] = a['attack'].fillna('none')
    a['view'] = np.where(a.partition == 'multi-source', 'multi',
                         np.where(a.train_sources == a.source, 'within', 'cross'))
    a['rounds_kind'] = np.where(a.attack == 'none', 'clean', 'attacked')
    k = ['partition', 'train_sources', 'source', 'protocol', 'epsilon', 'beta', 'attack']
    std = a[a.method == 'standard'][k + ['mse_mean', 'mse_var']].rename(
        columns={'mse_mean': 'std_mean', 'mse_var': 'std_var'})
    return a.merge(std, on=k, how='left')


def pooled(d, by):
    """Round-weighted MSE per group, and its ratio to the standard method on the same cells."""
    d = d.assign(wm=d.mse_mean * d.n_rounds, wv=d.mse_var * d.n_rounds,
                 sm=d.std_mean * d.n_rounds, sv=d.std_var * d.n_rounds)
    g = d.groupby(by).agg(n=('n_rounds', 'sum'), wm=('wm', 'sum'), wv=('wv', 'sum'),
                          sm=('sm', 'sum'), sv=('sv', 'sum')).reset_index()
    g['mse_mean'], g['mse_var'] = g.wm / g.n, g.wv / g.n
    g['ratio_mean'], g['ratio_var'] = g.wm / g.sm, g.wv / g.sv
    return g.drop(columns=['wm', 'wv', 'sm', 'sv'])


def wmean(d, by, cols):
    d = d.copy()
    for c in cols:
        d[c + '_w'] = d[c] * d.n_rounds
        d[c + '_n'] = d.n_rounds.where(d[c].notna(), 0)
    g = d.groupby(by)[[c + s for c in cols for s in ('_w', '_n')]].sum().reset_index()
    for c in cols:
        g[c] = g[c + '_w'] / g[c + '_n'].replace(0, np.nan)
    return g[by + cols]


def present(methods, d):
    have = set(d.method.unique())
    return [m for m in methods if m in have]


# ---------------------------------------------------------------- figures

def fig_overview(a, out):
    w = a[a.view == 'within']
    g = pooled(w, ['rounds_kind', 'protocol', 'epsilon', 'method'])
    out.table(g, 'ratio_vs_standard_within')
    meths = present([m for m in METHODS if m != 'standard'], g)
    eps = sorted(g.epsilon.unique())
    cols = [(p, e) for p in ('PM', 'SR') for e in eps]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), gridspec_kw={'hspace': 0.32, 'wspace': 0.04})
    for i, rk in enumerate(('attacked', 'clean')):
        for j, v in enumerate(('mean', 'var')):
            ax = axes[i, j]
            gi = g[g.rounds_kind == rk].set_index(['protocol', 'epsilon', 'method'])[f'ratio_{v}']
            M = np.array([[gi.get((p, e, m), np.nan) for p, e in cols] for m in meths])
            ax.imshow(np.clip(np.log2(M), -4, 4), cmap=DIVERGING, vmin=-4, vmax=4, aspect='auto')
            for r in range(M.shape[0]):
                for c in range(M.shape[1]):
                    x = M[r, c]
                    if np.isnan(x):
                        continue
                    t = f'{x:.2f}' if x < 0.1 else f'{x:.1f}' if x < 10 else f'{x:.0f}'
                    ax.text(c, r, t, ha='center', va='center', fontsize=7.5,
                            color='white' if abs(np.log2(x)) > 2.3 else INK)
            ax.set_xticks(range(len(cols)), [f'{e:g}' for _, e in cols])
            ax.set_yticks(range(len(meths)), [METHODS[m][0] for m in meths] if j == 0 else [])
            ax.axvline(len(eps) - 0.5, color=SURF, lw=4)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            ax.set_xlabel('ε   (left: PM, right: SR)')
            ax.set_title(f"{rk.capitalize()} rounds: error of the recovered {'mean' if v == 'mean' else 'variance'}",
                         loc='left', pad=20)
            ax.text((len(eps) - 1) / 2, -0.75, 'PM', ha='center', fontsize=9, fontweight='bold', color=INK2)
            ax.text(len(eps) + (len(eps) - 1) / 2, -0.75, 'SR', ha='center', fontsize=9, fontweight='bold', color=INK2)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-4, 4))
    cb = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.035, pad=0.08, aspect=50)
    cb.set_ticks(range(-4, 5), labels=['1/16', '1/8', '1/4', '1/2', '1', '2', '4', '8', '≥16'])
    cb.set_label('MSE ÷ Standard MSE on the same rounds   (blue: lower error than no defence; red: higher)')
    cb.outline.set_visible(False)
    header(fig, 'Which method beats no defence, and where',
           'Within-source (trained and tested on the same dataset). Each cell pools sources, β, targets, '
           'attacks (IPA + OPA) and seeds. Clean rounds have no fakes: anything above 1 there is damage.', y=0.955)
    out.fig(fig, '01_overview_ratio_vs_standard')


def fig_eps(a, out):
    w = a[(a.view == 'within') & (a.rounds_kind == 'attacked')]
    g = pooled(w, ['protocol', 'epsilon', 'method'])
    meths = present(list(METHODS), g)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, gridspec_kw={'hspace': 0.18, 'wspace': 0.22})
    for i, v in enumerate(('mean', 'var')):
        for j, p in enumerate(('PM', 'SR')):
            ax = axes[i, j]
            for m in meths:
                s = g[(g.protocol == p) & (g.method == m)].sort_values('epsilon')
                ax.plot(s.epsilon, s[f'mse_{v}'], **style(m))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xticks(sorted(g.epsilon.unique()), [f'{e:g}' for e in sorted(g.epsilon.unique())])
            ax.minorticks_off()
            ax.set_title(f"{p}: MSE of the {'mean' if v == 'mean' else 'variance'}", loc='left')
            if i == 1:
                ax.set_xlabel('ε (privacy budget)')
            if j == 0:
                ax.set_ylabel('MSE vs genuine value (log)')
    method_legend(fig, meths, y=0.905, ncol=5)
    header(fig, 'Error against privacy budget, attacked rounds',
           'Within-source; pooled over sources, β, targets, attacks and seeds.', y=0.985)
    out.fig(fig, '02_mse_vs_eps')


def fig_beta(a, out, eps_list=(1.0, 4.0)):
    w = a[a.view == 'within']
    g = pooled(w, ['protocol', 'epsilon', 'beta', 'method'])
    meths = present(list(METHODS), g)
    panels = [(p, e) for p in ('PM', 'SR') for e in eps_list]
    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 7.2), sharex=True,
                             gridspec_kw={'hspace': 0.18, 'wspace': 0.25})
    for i, v in enumerate(('mean', 'var')):
        for j, (p, e) in enumerate(panels):
            ax = axes[i, j]
            for m in meths:
                s = g[(g.protocol == p) & (g.epsilon == e) & (g.method == m)].sort_values('beta')
                ax.plot(s.beta, s[f'mse_{v}'], **style(m))
            ax.set_yscale('log')
            ax.axvspan(-0.03, 0.025, color=GRID, zorder=0)
            ax.set_title(f"{p}, ε = {e:g}: {'mean' if v == 'mean' else 'variance'}", loc='left')
            if i == 1:
                ax.set_xlabel('β (share of fake users); 0 = clean')
            if j == 0:
                ax.set_ylabel('MSE vs genuine value (log)')
            ax.set_xlim(-0.03, 0.83)
    method_legend(fig, meths, y=0.905, ncol=5)
    header(fig, 'Error against attack size',
           'Within-source; pooled over sources, targets, attacks and seeds. The shaded point at β = 0 is clean '
           'rounds.', y=0.985)
    out.fig(fig, '03_mse_vs_beta')


def fig_transfer(a, out, eps_list=(1.0, 4.0)):
    d = a[(a.rounds_kind == 'attacked') & a.epsilon.isin(eps_list)]
    g = pooled(d, ['protocol', 'epsilon', 'view', 'method'])
    out.table(g, 'transfer_within_cross_multi')
    pair = pooled(d, ['protocol', 'epsilon', 'train_sources', 'source', 'method'])
    out.table(pair, 'transfer_pairs')
    meths = present([m for m in METHODS if m != 'standard'], g)
    views = [('within', INK, 'o', 'within-source'), ('cross', SLOTS[1], 'D', 'cross-source (1 → other)'),
             ('multi', SLOTS[0], 's', 'multi-source (2 → held-out)')]
    cols = [(v, e) for v in ('mean', 'var') for e in eps_list]
    fig, axes = plt.subplots(2, len(cols), figsize=(15, 7.4), sharey=True, gridspec_kw={'hspace': 0.3, 'wspace': 0.08})
    for i, p in enumerate(('PM', 'SR')):
        for j, (v, e) in enumerate(cols):
            ax = axes[i, j]
            gi = g[(g.protocol == p) & (g.epsilon == e)].set_index(['method', 'view'])[f'ratio_{v}']
            for r, m in enumerate(meths):
                xs = [gi.get((m, vw), np.nan) for vw, *_ in views]
                ax.plot([np.nanmin(xs), np.nanmax(xs)], [r, r], color=AXIS, lw=1.2, zorder=1)
                for k, ((vw, c, mk, _), x) in enumerate(zip(views, xs)):
                    ax.scatter(x, r + (k - 1) * 0.2, color=c, marker=mk, s=34, zorder=3, edgecolor=SURF, linewidth=0.8)
            ax.axvline(1, color=INK2, lw=1, zorder=0)
            ax.set_xscale('log')
            ax.set_yticks(range(len(meths)), [METHODS[m][0] for m in meths])
            ax.invert_yaxis() if j == 0 and i == 0 else None
            ax.grid(axis='y', visible=False)
            ax.set_title(f"{p}, ε = {e:g}: {'mean' if v == 'mean' else 'variance'}", loc='left')
            if i == 1:
                ax.set_xlabel('MSE ÷ Standard MSE (log; < 1 is better)')
    h = [Line2D([], [], color=c, marker=mk, linestyle='', markersize=7, label=lab) for _, c, mk, lab in views]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.915), ncol=3, fontsize=9)
    header(fig, 'Does it transfer to a dataset it was not trained on?',
           'Attacked rounds; pooled over test sources, β, targets, attacks and seeds. Standard is the vertical '
           'line at 1.', y=0.985)
    out.fig(fig, '04_transfer_within_cross_multi')


def fig_detection(a, out):
    w = a[(a.view == 'within') & (a.rounds_kind == 'attacked')]
    cols = ['clf_pr_auc', 'clf_f1', 'clf_precision', 'clf_recall', 'clf_genuine_removal_rate', 'fallback_rate']
    g = wmean(w, ['protocol', 'epsilon', 'method'], cols)
    out.table(g[g.method.isin(['mlp_removal', 'ft_removal', 'deepsets_recovery', 'settransformer_recovery',
                               'oracle_removal'])], 'detection')
    chance = (w.beta * w.n_rounds).sum() / w.n_rounds.sum()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, gridspec_kw={'hspace': 0.22, 'wspace': 0.2})
    sc = present(['mlp_removal', 'ft_removal', 'deepsets_recovery', 'settransformer_recovery'], g)
    rm = present(['mlp_removal', 'ft_removal'], g)
    for j, p in enumerate(('PM', 'SR')):
        ax = axes[0, j]
        for m in sc:
            s = g[(g.protocol == p) & (g.method == m)].sort_values('epsilon')
            ax.plot(s.epsilon, s.clf_pr_auc, **style(m))
        ax.axhline(chance, color=INK2, lw=1, ls=':')
        ax.text(6, chance - 0.05, f'chance ≈ mean β = {chance:.2f}', color=INK2, fontsize=8, ha='right')
        ax.set_ylim(0, 1)
        ax.set_title(f'{p}: ranking fakes above genuine (PR-AUC)', loc='left')
        ax = axes[1, j]
        for m in rm:
            s = g[(g.protocol == p) & (g.method == m)].sort_values('epsilon')
            c = METHODS[m][1]
            ax.plot(s.epsilon, s.clf_recall, color=c, marker=METHODS[m][2], lw=1.6, ls='-')
            ax.plot(s.epsilon, s.clf_genuine_removal_rate, color=c, marker=METHODS[m][2], lw=1.4, ls='--',
                    markerfacecolor=SURF)
            ax.plot(s.epsilon, s.fallback_rate, color=c, lw=1.2, ls=':')
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f'{p}: what the removal classifiers remove', loc='left')
        ax.set_xlabel('ε (privacy budget)')
        for r in (0, 1):
            axes[r, j].set_xscale('log')
            axes[r, j].set_xticks(sorted(g.epsilon.unique()), [f'{e:g}' for e in sorted(g.epsilon.unique())])
            axes[r, j].minorticks_off()
    axes[1, 1].text(0.1, 0.3, 'SR: all three lines coincide. In most rounds every report is\n'
                    'flagged, a group would keep < 16 reports, and the estimate\nfalls back to Standard.',
                    fontsize=8, color=INK2, va='top')
    axes[0, 0].set_ylabel('PR-AUC')
    axes[1, 0].set_ylabel('share')
    method_legend(fig, sc, y=0.915, ncol=4)
    h = [Line2D([], [], color=INK2, ls='-', marker='o', label='fakes removed (recall)'),
         Line2D([], [], color=INK2, ls='--', marker='o', markerfacecolor=SURF, label='genuine removed'),
         Line2D([], [], color=INK2, ls=':', label='rounds falling back to all reports')]
    axes[1, 1].legend(handles=h, loc='center right', fontsize=8)
    header(fig, 'Per-report detection: SR reports carry no signal, PM classifiers never cross the threshold',
           'Within-source attacked rounds. When a group would keep < 16 reports the estimate falls back to all '
           'reports, i.e. to Standard.', y=0.985)
    out.fig(fig, '05_detection')


def fig_budget(root, out):
    b = pd.read_csv(os.path.join(root, 'budget_sweep', 'aggregate.csv'))
    b = b[b.train_sources == b.source]
    rows = [b.groupby(['protocol', 'budget_factor', 'method']).apply(
        lambda d: pd.Series({'mse_mean': np.average(d.mse_mean, weights=d.n_rounds),
                             'mse_var': np.average(d.mse_var, weights=d.n_rounds), 'n': d.n_rounds.sum()}),
        include_groups=False).reset_index()]
    files = glob.glob(os.path.join(root, 'below_budget', 'per_round', '*.csv'))
    if files:
        r = pd.concat([pd.read_csv(f, usecols=lambda c: c in {
            'method', 'protocol', 'source', 'train_sources', 'budget_factor', 'sq_err_mean', 'sq_err_var', 'attack'})
            for f in files], ignore_index=True)
        r = r[(r.train_sources == r.source) & r.attack.notna()]
        rows.append(r.groupby(['protocol', 'budget_factor', 'method']).agg(
            mse_mean=('sq_err_mean', 'mean'), mse_var=('sq_err_var', 'mean'), n=('sq_err_mean', 'size')).reset_index())
    g = pd.concat(rows, ignore_index=True).sort_values(['protocol', 'method', 'budget_factor'])
    out.table(g, 'budget')
    meths = present(list(METHODS), g)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, gridspec_kw={'hspace': 0.18, 'wspace': 0.22})
    for i, v in enumerate(('mean', 'var')):
        for j, p in enumerate(('PM', 'SR')):
            ax = axes[i, j]
            for m in meths:
                s = g[(g.protocol == p) & (g.method == m)]
                ax.plot(s.budget_factor, s[f'mse_{v}'], **style(m))
            ax.axvspan(0.45, 1.0, color=GRID, zorder=0)
            ax.set_yscale('log')
            ax.set_title(f"{p}, ε = 1: MSE of the {'mean' if v == 'mean' else 'variance'}", loc='left')
            if i == 1:
                ax.set_xlabel('m / m_min (fake users relative to the minimum the attack needs)')
            if j == 0:
                ax.set_ylabel('MSE vs genuine value (log)')
    axes[0, 0].text(0.5, 0.97, 'below budget:\nbest-effort attack', transform=axes[0, 0].get_xaxis_transform(),
                    va='top', fontsize=8, color=INK2)
    method_legend(fig, meths, y=0.905, ncol=3)
    header(fig, 'Error against attacker budget',
           'Within-source, fixed genuine population (n = 4,096). m/m_min < 1: frozen budget-sweep checkpoints on '
           'separately generated below-budget rounds.', y=0.985)
    out.fig(fig, '06_budget')


def fig_attack_validation(root, out):
    frames = []
    for d in sorted(glob.glob(os.path.join(root, 'paper_attack_*'))):
        f = os.path.join(d, 'attack_validation_summary.csv')
        if os.path.exists(f):
            frames.append(pd.read_csv(f).assign(experiment=os.path.basename(d)))
    if not frames:
        return
    s = pd.concat(frames, ignore_index=True)
    out.table(s, 'attack_validation')
    panels = [('paper_attack_eps', 'epsilon', 'ε'), ('paper_attack_beta_taxi', 'beta', 'β'),
              ('paper_attack_S1_taxi', 'dS1_n', 'attacker error in S(1)'),
              ('paper_attack_S2_taxi', 'dS2_n', 'attacker error in S(2)'),
              ('paper_attack_ne_taxi', 'dn', 'attacker error in n_e')]
    panels = [pn for pn in panels if (s.experiment == pn[0]).any()]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.4 * len(panels), 6.6), gridspec_kw={'hspace': 0.3,
                                                                                          'wspace': 0.3})
    for j, (exp, x, xl) in enumerate(panels):
        d = s[(s.experiment == exp) & (s.source == 'taxi')]
        for i, (col, yl) in enumerate((('mse_target_mean', 'MSE to target mean'),
                                       ('mse_target_var', 'MSE to target variance'))):
            ax = axes[i, j]
            for (p, at), dd in d.groupby(['protocol', 'attack']):
                dd = dd.sort_values(x)
                ax.plot(dd[x], dd[col], color=ATTACK_STYLE.get(at, MUTED), ls=PROTO_LS[p], marker='o', lw=1.5)
            ax.set_yscale('log')
            if x == 'epsilon':
                ax.set_xscale('log')
                ax.set_xticks([0.1, 1, 2, 4, 6], ['0.1', '1', '2', '4', '6'])
                ax.minorticks_off()
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(yl + ' (log)')
            if i == 0:
                ax.set_title({'epsilon': 'Fig. 10 analogue', 'beta': 'Fig. 11 analogue', 'dS1_n': 'Fig. 7 analogue',
                              'dS2_n': 'Fig. 8 analogue', 'dn': 'Fig. 9 analogue'}[x], loc='left')
    h = [Line2D([], [], color=ATTACK_STYLE[a], lw=1.6, marker='o', label=a) for a in ('IPA', 'OPA')] + \
        [Line2D([], [], color=INK2, ls=PROTO_LS[p], lw=1.6, label=p) for p in ('PM', 'SR')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=9)
    header(fig, 'Attack validation at paper scale (Taxi): how close the attack gets to its target',
           'Missing points: target infeasible for that attack (IPA cannot reach some targets). Income and the '
           'Table 4 / default runs are in tables/attack_validation.csv.', y=0.995)
    out.fig(fig, '07_attack_validation')


def fig_kmeans(root, out):
    frames = []
    for d in sorted(glob.glob(os.path.join(root, 'verify_kmeans_*'))):
        f = os.path.join(d, 'kmeans_verification_summary.csv')
        if os.path.exists(f):
            frames.append(pd.read_csv(f).assign(experiment=os.path.basename(d)))
    if not frames:
        return
    s = pd.concat(frames, ignore_index=True)
    out.table(s, 'kmeans_verification')
    panels = [('verify_kmeans_rate_taxi', 'rate', 'sampling rate r'), ('verify_kmeans_beta_taxi', 'beta', 'β (r = 0.6)'),
              ('verify_kmeans_mean_taxi', 'target_mean_value', 'target mean offset (r = 0.6)'),
              ('verify_kmeans_var_taxi', 'target_var_value', 'target variance (r = 0.6)')]
    panels = [pn for pn in panels if (s.experiment == pn[0]).any()]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.6 * len(panels), 6.4), gridspec_kw={'hspace': 0.32,
                                                                                          'wspace': 0.32})
    for j, (exp, x, xl) in enumerate(panels):
        d = s[s.experiment == exp]
        for i, (col, yl) in enumerate((('ag_mean', 'gain on mean'), ('ag_var', 'gain on variance'))):
            ax = axes[i, j]
            for (p, at), dd in d.groupby(['protocol', 'attack']):
                dd = dd.sort_values(x)
                ax.plot(dd[x], dd[col], color=ATTACK_STYLE.get(at, MUTED), ls=PROTO_LS[p], marker='o', lw=1.5)
            ax.axhline(0, color=INK2, lw=1)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
            ax.set_xlabel(xl)
            if x == 'target_mean_value':
                ax.set_xticks([-0.06, -0.03, 0, 0.03, 0.06])
            if j == 0:
                ax.set_ylabel(yl + '  (MSE before − after)')
    h = [Line2D([], [], color=ATTACK_STYLE[a], lw=1.6, marker='o', label=a) for a in ('IPA', 'OPA')] + \
        [Line2D([], [], color=INK2, ls=PROTO_LS[p], lw=1.6, label=p) for p in ('PM', 'SR')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=9)
    header(fig, 'k-means defence verification (paper Sec. 8.1, Fig. 12 analogue), Taxi, ε = 1',
           'Above 0: k-means lowers the error. PM: at most 3% of the error before. SR: from −75% (small r) to +46%.', y=0.995)
    out.fig(fig, '08_kmeans_fig12')


def fig_feasibility(root, out):
    files = sorted(glob.glob(os.path.join(root, 'feasibility', 'feasibility_eps*.csv')))
    if not files:
        return
    f = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    out.table(f, 'feasibility_beta_min')
    eps = sorted(f.epsilon.unique())
    attacks = [a for a in ('IPA', 'OPA-PM', 'OPA-SR') if a in set(f.attack)]
    srcs = [s for s in ('taxi', 'income', 'retirement') if s in set(f.source)]
    cmap = SEQ_BLUE.copy()
    cmap.set_bad('#d9d8d2')
    rows = [(s, a) for s in srcs for a in attacks]
    fig, axes = plt.subplots(len(rows), len(eps), figsize=(2.3 * len(eps), 1.95 * len(rows)), sharex=True,
                             sharey=True, gridspec_kw={'hspace': 0.12, 'wspace': 0.06})
    for i, (s, a) in enumerate(rows):
        for j, e in enumerate(eps):
            ax = axes[i, j]
            d = f[(f.source == s) & (f.attack == a) & (f.epsilon == e)]
            pv = d.pivot_table(index='target_variance', columns='target_mean', values='beta_min', dropna=False)
            im = ax.imshow(np.ma.masked_invalid(pv.values), cmap=cmap, vmin=0, vmax=1, aspect='auto', origin='lower',
                           extent=[pv.columns.min(), pv.columns.max(), pv.index.min(), pv.index.max()])
            ax.grid(False)
            if i == 0:
                ax.set_title(f'ε = {e:g}', loc='center')
            if j == 0:
                ax.set_ylabel(f'{s}\n{a}\ntarget var', fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel('target mean')
            ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.012, pad=0.035, aspect=60)
    cb.set_label('β_min: smallest share of fake users that reaches the target (gray: infeasible)')
    cb.outline.set_visible(False)
    header(fig, 'Theoretical feasibility maps (oracle knowledge)',
           'Rows: dataset × attack; columns: ε. Lighter = cheaper attack.', y=0.9)
    out.fig(fig, '09_feasibility_maps')


def fig_budget_curves(root, out):
    files = sorted(glob.glob(os.path.join(root, 'budget_curves', 'budget_curves_eps*.csv')))
    if not files:
        return
    c = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    out.table(c, 'budget_curves')
    c = c[c.knowledge == 'attacker estimate']
    eps = sorted(c.epsilon.unique())
    attacks = [a for a in ('IPA', 'OPA-PM', 'OPA-SR') if a in set(c.attack)]
    srcs = [s for s in ('taxi', 'income', 'retirement') if s in set(c.source)]
    cols = [(ax_, s) for ax_ in ('mean', 'variance') for s in srcs]
    fig, axes = plt.subplots(len(attacks), len(cols), figsize=(2.75 * len(cols), 2.5 * len(attacks)), sharey=True,
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.1})
    for i, a in enumerate(attacks):
        for j, (axis, s) in enumerate(cols):
            ax = axes[i, j]
            for k, e in enumerate(eps):
                d = c[(c.attack == a) & (c.source == s) & (c.axis == axis) & (c.epsilon == e)].sort_values('x')
                ax.plot(d.x, d.beta_min, color=ORDINAL_BLUE[k % len(ORDINAL_BLUE)], lw=1.5)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(f"{s}: vs target {'mean' if axis == 'mean' else 'variance'}", loc='left', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{a}\nβ_min')
            if i == len(attacks) - 1:
                ax.set_xlabel('target − true mean' if axis == 'mean' else 'target ÷ true variance')
            ax.tick_params(labelsize=7)
    h = [Line2D([], [], color=ORDINAL_BLUE[k % len(ORDINAL_BLUE)], lw=2, label=f'ε = {e:g}') for k, e in enumerate(eps)]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.94), ncol=len(eps), fontsize=9)
    header(fig, 'Attack cost: minimum fake share against target, attacker-estimated knowledge',
           'IPA does not depend on ε (its curves overlap). Oracle-knowledge curves are in tables/budget_curves.csv.', y=1.0)
    out.fig(fig, '10_budget_curves')


def fig_runtime(a, out):
    g = a.groupby('method').runtime_mean.median().reindex([m for m in METHODS if m in set(a.method)])
    out.table(g.reset_index().rename(columns={'runtime_mean': 'median_s_per_round'}), 'runtime')
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(g))
    ax.barh(y, g.values, color=[METHODS[m][1] for m in g.index], height=0.62)
    for yi, v in zip(y, g.values):
        ax.text(v * 1.15, yi, f'{v:.1e} s', va='center', fontsize=8, color=INK2)
    ax.set_yticks(y, [METHODS[m][0] for m in g.index])
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlim(g.min() / 3, g.max() * 20)
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel('inference time per round, seconds (log; median over cells)')
    header(fig, 'Inference cost per round', 'Training time per unit is in the server logs (eta.py).', y=1.0)
    out.fig(fig, '11_runtime')


# ---------------------------------------------------------------- attack success (attack_success.py output)

ASR_STATS = (('joint', 'mean and variance both hit'), ('mean', 'mean hit'), ('var', 'variance hit'))


def load_asr(root, exp):
    p = os.path.join(root, exp, 'attack_success.csv.gz')
    if not os.path.exists(p):
        print(f'  skip attack success for {exp}: run scripts/mean_variance/attack_success.py on the server '
              f'and copy {exp}/attack_success.csv.gz')
        return None
    g = pd.read_csv(p)
    g['view'] = np.where(g.partition.str.contains('multi'), 'multi',
                         np.where(g.train_sources == g.source, 'within', 'cross'))
    return g


def asr_rates(g, by, t='t50'):
    """Pool count columns per group; ASR before / after, kept fraction, MSE and utility gain vs no defence."""
    cols = ['n_rounds', 'sse_mean', 'sse_var', 'std_sse_mean', 'std_sse_var'] + [
        f'{k}_{s}_{t}' for s, _ in ASR_STATS for k in ('n', 'before', 'after', 'kept')]
    s = g.groupby(by, dropna=False)[cols].sum().reset_index()
    for st, _ in ASR_STATS:
        n = s[f'n_{st}_{t}'].replace(0, np.nan)
        s[f'asr_before_{st}'] = s[f'before_{st}_{t}'] / n
        s[f'asr_after_{st}'] = s[f'after_{st}_{t}'] / n
        s[f'kept_{st}'] = s[f'kept_{st}_{t}'] / s[f'before_{st}_{t}'].replace(0, np.nan)
    for v in ('mean', 'var'):
        s[f'mse_{v}'] = s[f'sse_{v}'] / s.n_rounds
        s[f'mse_before_{v}'] = s[f'std_sse_{v}'] / s.n_rounds
        s[f'ag_{v}'] = s[f'mse_before_{v}'] - s[f'mse_{v}']
        s[f'rel_ag_{v}'] = s[f'ag_{v}'] / s[f'mse_before_{v}']
    keep = [c for c in s if not c.startswith(('sse_', 'std_sse_')) and not c.endswith(f'_{t}')]
    return s[keep + [f'n_joint_{t}', f'before_joint_{t}']].rename(
        columns={f'n_joint_{t}': 'n_scored', f'before_joint_{t}': 'n_hit_before'})


def not_generated(root):
    """Attacked test rounds that were never built (budget too small), per protocol x attack x eps."""
    c = pd.read_csv(os.path.join(root, 'breadth', 'coverage.csv'))
    c = c[(c.split == 'test') & c.attack.notna() & (c.target != 'm2v2')]
    return c.assign(missing=c.status.isin(['infeasible_target', 'allocation_infeasible'])).groupby(
        ['protocol', 'attack', 'epsilon']).agg(n_attempted=('missing', 'size'), n_not_generated=('missing', 'sum'))


def fig_asr_eps(root, out):
    g = load_asr(root, 'breadth')
    if g is None:
        return
    att = g[g.attack != 'none']
    w = att[att.view == 'within']
    r = asr_rates(w, ['protocol', 'attack', 'epsilon', 'method'])
    miss = not_generated(root).reset_index()
    r = r.merge(miss, on=['protocol', 'attack', 'epsilon'], how='left')
    # attacks the budget could not build count as failures: ASR over every attempted round
    r['generated_share'] = 1 - r.n_not_generated / r.n_attempted
    r['asr_before_joint_attempted'] = r.asr_before_joint * r.generated_share
    out.table(r, 'attack_success_within')
    out.table(asr_rates(w, ['protocol', 'attack', 'epsilon', 'beta', 'method']), 'attack_success_by_beta')
    out.table(asr_rates(w, ['protocol', 'attack', 'epsilon', 'target', 'method']), 'attack_success_by_target')
    out.table(asr_rates(att, ['view', 'protocol', 'attack', 'epsilon', 'method']), 'attack_success_by_view')
    meths = present(list(METHODS), r)
    panels = [(p, a) for p in ('PM', 'SR') for a in ('IPA', 'OPA')]
    rows = [(f'asr_after_{s}', f'ASR, {lab}') for s, lab in ASR_STATS] + [
        ('kept_joint', 'Kept: share of hits (both) still hit')]
    fig, axes = plt.subplots(len(rows), len(panels), figsize=(15, 13), sharex=True,
                             gridspec_kw={'hspace': 0.22, 'wspace': 0.2})
    eps = sorted(r.epsilon.unique())
    for i, (col, lab) in enumerate(rows):
        for j, (p, a) in enumerate(panels):
            ax = axes[i, j]
            for m in meths:
                if col.startswith('kept') and m == 'standard':
                    continue
                s = r[(r.protocol == p) & (r.attack == a) & (r.method == m)].sort_values('epsilon')
                ax.plot(s.epsilon, s[col], **style(m))
            ax.set_xscale('log')
            ax.set_xticks(eps, [f'{e:g}' for e in eps])
            ax.minorticks_off()
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(f'{p} · {a}', loc='left')
            if j == 0:
                ax.set_ylabel(lab)
            if i == len(rows) - 1:
                ax.set_xlabel('ε (privacy budget)')
    method_legend(fig, meths, y=0.915, ncol=5)
    header(fig, 'Attack success before and after defence',
           'Within-source attacked rounds whose target moves the statistic. Hit: estimate within half the intended '
           'shift of the target. Standard = before defence. Oracle removal = genuine reports only, so its rate is '
           'what noise alone gives. Kept = hits before defence that are still hits after.', y=0.985)
    out.fig(fig, '12_attack_success_vs_eps')


def fig_asr_budget(root, out):
    parts = [x for x in (load_asr(root, 'budget_sweep'), load_asr(root, 'below_budget')) if x is not None]
    if not parts:
        return
    g = pd.concat(parts, ignore_index=True)
    g = g[(g.view == 'within') & (g.attack != 'none')]
    r = asr_rates(g, ['protocol', 'budget_factor', 'method']).sort_values(['protocol', 'method', 'budget_factor'])
    out.table(r, 'attack_success_budget')
    meths = present(list(METHODS), r)
    rows = [(f'asr_after_{s}', f'ASR, {lab}') for s, lab in ASR_STATS] + [
        ('kept_joint', 'Kept: share of hits (both) still hit')]
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 12), sharex=True, gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
    for i, (col, lab) in enumerate(rows):
        for j, p in enumerate(('PM', 'SR')):
            ax = axes[i, j]
            for m in meths:
                if col.startswith('kept') and m == 'standard':
                    continue
                s = r[(r.protocol == p) & (r.method == m)]
                ax.plot(s.budget_factor, s[col], **style(m))
            ax.axvspan(0.45, 1.0, color=GRID, zorder=0)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(f'{p}, ε = 1', loc='left')
            if j == 0:
                ax.set_ylabel(lab)
            if i == len(rows) - 1:
                ax.set_xlabel('m / m_min (fake users relative to the minimum the attack needs)')
    method_legend(fig, meths, y=0.915, ncol=3)
    header(fig, 'Attack success against attacker budget',
           'Within-source. Shaded: below budget (best-effort attack, frozen checkpoints). Hit: estimate within half '
           'the intended shift of the target. Standard = before defence.', y=0.985)
    out.fig(fig, '13_attack_success_budget')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--results-root', default='results/mean_variance_server')
    ap.add_argument('--out', default='results/figures/mean_variance_server')
    args = ap.parse_args()
    root = args.results_root if os.path.isabs(args.results_root) else os.path.join(ROOT, args.results_root)
    out = Out(args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out))
    a = load_breadth(root)
    fig_overview(a, out)
    fig_eps(a, out)
    fig_beta(a, out)
    fig_transfer(a, out)
    fig_detection(a, out)
    fig_budget(root, out)
    fig_attack_validation(root, out)
    fig_kmeans(root, out)
    fig_feasibility(root, out)
    fig_budget_curves(root, out)
    fig_runtime(a, out)
    fig_asr_eps(root, out)
    fig_asr_budget(root, out)
    print(f'{len(out.figs)} figures, {len(out.tables)} tables -> {out.d}')


if __name__ == '__main__':
    main()
