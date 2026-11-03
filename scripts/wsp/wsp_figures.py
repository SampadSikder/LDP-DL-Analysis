#!/usr/bin/env python
"""WSP comparison figures in the style of the existing mean/variance and frequency-oracle figures.

  python scripts/wsp/compare_wsp.py      # tables first (this script calls it if they are missing)
  python scripts/wsp/wsp_figures.py      # PNGs in results/figures/wsp/, tables in results/figures/wsp/tables/

Variants (colour + marker + line style, so identity never rests on colour alone):
  Stage 1 (gradient training, up to 50 epochs)   grey   circle   dashed
  + gradient fine-tune (gated)                   blue   square   dotted
  + random edit search (gated)                   amber  triangle dash-dot
  + LLM weight editing (WSP, gated)              magenta diamond solid
Colours: grey / blue / amber from the existing figures' palette plus one magenta; checked for colour-blind
separation (OKLab dE >= 9.9 for every pair under protan / deutan / tritan simulation).
No defence = the projected standard estimate on the same rounds (as in summary_figures.py).

Figure                      based on
wsp_01_overview             mean_variance_server/01_overview_ratio_vs_standard.png
wsp_04_transfer             mean_variance_server/04_transfer_within_cross_multi.png
wsp_mse_vs_eps_{PM,SR}      mean_variance_selected/mse_vs_eps.png
wsp_asr_vs_eps              mean_variance_server/12_attack_success_vs_eps.png
wsp_detection               mean_variance_server/05_detection.png (FT-Transformer removal only)
wsp_overall_table           mean_variance_selected/overall_table.png
wsp_freq_overall            frequency_oracle/.../fig3_gap + fig1 (overall test F1 per run, incl. control)
wsp_freq_cross              frequency_oracle/.../fig6_cross_f1_score.png (OUE, F1 vs epsilon)
wsp_cost, wsp_budget_curve, wsp_edit_anatomy   new
"""
import json
import os
import sys

from _common import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts', 'mean_variance'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'plots'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from summary_figures import DIVERGING, GRID, INK, INK2, MUTED, SURF, Out, asr_rates, header, load_asr  # noqa: E402

import compare_wsp  # noqa: E402
from attacker_detector.wsp.jobs import FIG_DIR, SWEEP, WSP_HOME, all_jobs, load_yaml, rel  # noqa: E402

OUT_DIR = FIG_DIR
VAR = {'stage1': ('Stage 1 (gradient, up to 50 epochs)', '#52514e', 'o', '--'),
       'ft': ('+ gradient fine-tune', '#2a78d6', 's', ':'),
       'rand': ('+ random edit search', '#eda100', '^', '-.'),
       'wsp': ('+ LLM weight editing (WSP)', '#c0328a', 'D', '-')}
BASE = {'ft_removal': 'FT-Transformer removal', 'deepsets_recovery': 'Deep Sets recovery',
        'settransformer_recovery': 'Set Transformer recovery'}
STAT = {'mean': 'mean', 'var': 'variance'}
SHORT = {'ft_removal': 'FT removal', 'deepsets_recovery': 'Deep Sets', 'settransformer_recovery': 'Set Transformer'}
VSHORT = {'stage1': 'Stage 1', 'ft': '+ fine-tune', 'rand': '+ random', 'wsp': '+ WSP'}


def row_label(b, v):
    return f'{SHORT[b]} · {VSHORT[v]}'


def vstyle(v, **kw):
    lab, c, mk, ls = VAR[v]
    return {'label': lab, 'color': c, 'marker': mk, 'linestyle': ls, 'linewidth': 1.8 if v == 'wsp' else 1.4,
            'markersize': 6, **kw}


def variant_legend(fig, variants, y=1.0, extra=(), bars=False):
    if bars:
        h = [Patch(facecolor=VAR[v][1], label=VAR[v][0]) for v in variants] + list(extra)
    else:
        h = [Line2D([], [], **vstyle(v)) for v in variants] + list(extra)
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, y), ncol=min(len(h), 5), fontsize=8.5,
               handlelength=2.6, columnspacing=1.6)


def no_defence_handle():
    return Line2D([], [], color=INK2, linewidth=7, alpha=0.3, solid_capstyle='round', label='No defence')


def pooled(c, keys):
    g = c.groupby(keys, dropna=False)[['n', 'sse_mean', 'sse_var', 'std_sse_mean', 'std_sse_var']].sum().reset_index()
    for s in ('mean', 'var'):
        g[f'mse_{s}'] = g[f'sse_{s}'] / g.n
        g[f'std_mse_{s}'] = g[f'std_sse_{s}'] / g.n
        g[f'ratio_{s}'] = g[f'sse_{s}'] / g[f'std_sse_{s}']
    return g


MAIN = {'budget': 25}


def pick_main_budget(*frames):
    """The probe budget the main figures show: 25 if it ran, else the most common one."""
    b = pd.concat([f['vbudget'] for f in frames if f is not None and 'vbudget' in f], ignore_index=True).dropna()
    MAIN['budget'] = 25 if (b == 25).any() or b.empty else int(b.mode().iloc[0])
    return MAIN['budget']


def main_variants(c):
    """Stage-1 rows plus each editor at the main budget, with `variant` set to the mode key of VAR."""
    if c is None or c.empty:
        return c
    if 'vmode' not in c:
        c = compare_wsp.add_parts(c.copy())
    keep = (c.vmode == 'stage1') | (c.vbudget == MAIN['budget'])
    out = c[keep & c.vmode.isin(list(VAR))].copy()
    out['variant'] = out['vmode']
    return out


# ============================================================== mean / variance

def fig_overview(cells, out):
    w = pooled(main_variants(cells[cells.view == 'within']), ['base', 'variant', 'protocol', 'epsilon', 'status'])
    out.table(w, 'wsp_overview')
    bases = [b for b in BASE if b in set(w.base)]
    rows = [(b, v) for b in bases for v in VAR if ((w.base == b) & (w.variant == v)).any()]
    cols = [(p, e) for p in ('PM', 'SR') for e in sorted(w.epsilon.unique()) if ((w.protocol == p) & (w.epsilon == e)).any()]
    if not rows or not cols:
        return
    h = 0.36 * len(rows) * 2 + 3.0
    fig, axes = plt.subplots(2, 2, figsize=(12.5, h), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.98, top=1 - 1.3 / h, bottom=1.25 / h, wspace=0.06, hspace=0.32)
    norm = LogNorm(1 / 16, 16)
    for ax, (status, s) in zip(axes.ravel(), [(a, b) for a in ('attacked', 'clean') for b in ('mean', 'var')]):
        M = np.full((len(rows), len(cols)), np.nan)
        for i, (b, v) in enumerate(rows):
            for j, (p, e) in enumerate(cols):
                r = w[(w.base == b) & (w.variant == v) & (w.protocol == p) & (w.epsilon == e) & (w.status == status)]
                if len(r):
                    M[i, j] = r[f'ratio_{s}'].iloc[0]
        ax.imshow(np.clip(M, 1 / 16, 16), cmap=DIVERGING, norm=norm, aspect='auto')
        ax.grid(False)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.2g}', ha='center', va='center', fontsize=7.5,
                            color='white' if abs(np.log2(M[i, j])) > 2.2 else INK)
        ax.set_xticks(range(len(cols)), [f'{p}\nε={e:g}' for p, e in cols])
        ax.set_yticks(range(len(rows)), [row_label(b, v) for b, v in rows], fontsize=8)
        for k in range(1, len(bases)):
            ax.axhline(sum(1 for r in rows if r[0] in bases[:k]) - 0.5, color=SURF, linewidth=3)
        ax.set_title(f'{status.capitalize()} rounds: error of the recovered {STAT[s]}', loc='left')
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=norm)
    cax = fig.add_axes([0.35, 0.35 / h, 0.45, 0.16 / h])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[1 / 16, 1 / 4, 1, 4, 16])
    cb.ax.set_xticklabels(['1/16', '1/4', '1', '4', '≥16'])
    cb.set_label('MSE ÷ no-defence MSE on the same rounds (blue: lower error; red: higher)')
    header(fig, 'Does weight editing change where each model beats no defence?',
           'Within-source test rounds (trained on taxi, tested on taxi), pooled over β, targets, attacks and 3 seeds. '
           'Each model block: Stage 1, then the three gated editors from the same start.', y=1 - 0.75 / h)
    out.fig(fig, 'wsp_01_overview')


def fig_transfer(cells, out, eps=4.0):
    if not (cells.epsilon == eps).any():
        eps = float(cells.epsilon.max())
    a = pooled(main_variants(cells[(cells.status == 'attacked') & (cells.epsilon == eps)]),
               ['base', 'variant', 'protocol', 'view'])
    out.table(a, 'wsp_transfer')
    if a.empty:
        return
    rows = [(b, v) for b in BASE for v in VAR if ((a.base == b) & (a.variant == v)).any()]
    fig, axes = plt.subplots(1, 4, figsize=(14, 0.32 * len(rows) + 2.2), sharey=True)
    for ax, (p, s) in zip(axes, [(p, s) for p in ('PM', 'SR') for s in ('mean', 'var')]):
        for i, (b, v) in enumerate(rows):
            for view, mk, col in (('within', 'o', INK), ('cross', 'D', '#eb6834')):
                r = a[(a.base == b) & (a.variant == v) & (a.protocol == p) & (a.view == view)]
                if len(r):
                    ax.plot(r[f'ratio_{s}'].iloc[0], i, mk, color=col, markersize=6, zorder=3)
            rr = a[(a.base == b) & (a.variant == v) & (a.protocol == p)]
            if len(rr) == 2:
                ax.plot(rr[f'ratio_{s}'], [i, i], color=MUTED, linewidth=1, zorder=2)
        ax.axvline(1, color=INK2, linewidth=1)
        ax.set_xscale('log')
        ax.set_title(f'{p}, ε = {eps:g}: {STAT[s]}', loc='left')
        ax.set_xlabel('MSE ÷ no-defence MSE (log; < 1 is better)')
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(rows)), [row_label(b, v) for b, v in rows], fontsize=8)
    fig.legend(handles=[Line2D([], [], marker='o', color=INK, linestyle='', label='within-source (taxi → taxi)'),
                        Line2D([], [], marker='D', color='#eb6834', linestyle='',
                               label='cross-source (taxi → income, retirement)')],
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    header(fig, 'Do edits made on taxi carry over to other datasets?',
           'Attacked test rounds. The agent and the gate never see income or retirement data.', y=1.07)
    out.fig(fig, 'wsp_04_transfer')


def fig_mse_eps(cells, out):
    w = pooled(main_variants(cells[cells.view == 'within']), ['base', 'variant', 'protocol', 'epsilon', 'status'])
    for p in ('PM', 'SR'):
        wp = w[w.protocol == p]
        bases = [b for b in BASE if b in set(wp.base)]
        if not bases:
            continue
        panels = [(st, s) for st in ('attacked', 'clean') for s in ('mean', 'var')]
        fig, axes = plt.subplots(len(bases), 4, figsize=(15, 3.0 * len(bases)), squeeze=False)
        for i, b in enumerate(bases):
            for j, (st, s) in enumerate(panels):
                ax = axes[i, j]
                d = wp[(wp.base == b) & (wp.status == st)]
                ref = d[d.variant == 'stage1'].sort_values('epsilon')
                if len(ref):
                    ax.plot(ref.epsilon, ref[f'std_mse_{s}'] * 1e3, color=INK2, linewidth=7, alpha=0.3,
                            solid_capstyle='round', zorder=1)
                for v in VAR:
                    dv = d[d.variant == v].sort_values('epsilon')
                    if len(dv):
                        ax.plot(dv.epsilon, dv[f'mse_{s}'] * 1e3, **vstyle(v), zorder=3)
                ax.set_yscale('log')
                ax.set_xticks(sorted(wp.epsilon.unique()))
                ax.set_title(f'{BASE[b]}: {st}, {STAT[s]}', loc='left', fontsize=9)
                if j == 0:
                    ax.set_ylabel('MSE vs genuine, ×10⁻³ (log)')
                if i == len(bases) - 1:
                    ax.set_xlabel('ε (privacy budget)')
        variant_legend(fig, list(VAR), y=1.0, extra=[no_defence_handle()])
        header(fig, f'{p}: error against privacy budget, before and after editing',
               'Within-source test rounds (taxi), pooled over β, targets, attacks and 3 seeds.', y=1.04)
        out.fig(fig, f'wsp_mse_vs_eps_{p}')


def fig_asr(asr, out):
    if asr is None or asr.empty:
        return
    g = asr[(asr.view == 'within') & (asr.attack != 'none')].copy()
    g[['base', 'variant']] = g.method.apply(lambda m: pd.Series(compare_wsp.split_method(m)))
    g = main_variants(g)
    r = asr_rates(g, ['base', 'variant', 'protocol', 'epsilon'])
    out.table(r, 'wsp_asr_vs_eps')
    bases = [b for b in BASE if b in set(r.base)]
    if not bases:
        return
    fig, axes = plt.subplots(len(bases), 2, figsize=(10, 2.8 * len(bases)), squeeze=False, sharey=True)
    for i, b in enumerate(bases):
        for j, p in enumerate(('PM', 'SR')):
            ax = axes[i, j]
            d = r[(r.base == b) & (r.protocol == p)]
            ref = d[d.variant == 'stage1'].sort_values('epsilon')
            if len(ref):
                ax.plot(ref.epsilon, ref.asr_before_joint, color=INK2, linewidth=7, alpha=0.3, solid_capstyle='round')
            for v in VAR:
                dv = d[d.variant == v].sort_values('epsilon')
                if len(dv):
                    ax.plot(dv.epsilon, dv.asr_after_joint, **vstyle(v))
            ax.set_title(f'{BASE[b]}: {p}', loc='left', fontsize=9)
            ax.set_xticks(sorted(r.epsilon.unique()))
            if j == 0:
                ax.set_ylabel('attack success (mean and variance hit)')
            if i == len(bases) - 1:
                ax.set_xlabel('ε')
    variant_legend(fig, list(VAR), y=1.0, extra=[no_defence_handle()])
    header(fig, 'Attack success before and after each defence',
           'Within-source attacked test rounds; a statistic is hit when the estimate lands within half the intended '
           'shift of the target (tolerance 0.5, as in summary_figures.py).', y=1.04)
    out.fig(fig, 'wsp_asr_vs_eps')


def fig_detection(cells, out):
    d = main_variants(cells[(cells.base == 'ft_removal') & (cells.view == 'within')])
    if d.empty or 'clf_pr_auc' not in d:
        return
    g = d.groupby(['variant', 'protocol', 'epsilon', 'status'])[['clf_pr_auc', 'removed_frac']].mean().reset_index()
    out.table(g, 'wsp_detection')
    panels = [('attacked', 'clf_pr_auc', 'PR-AUC on attacked rounds'),
              ('attacked', 'removed_frac', 'share of reports removed, attacked rounds'),
              ('clean', 'removed_frac', 'share of reports removed, clean rounds')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    cols = [(p, e) for p in ('PM', 'SR') for e in sorted(g.epsilon.unique())]
    x = np.arange(len(cols))
    width = 0.8 / len(VAR)
    for ax, (st, col, title) in zip(axes, panels):
        for k, v in enumerate(VAR):
            vals = [g[(g.variant == v) & (g.protocol == p) & (g.epsilon == e) & (g.status == st)][col].mean()
                    for p, e in cols]
            ax.bar(x + (k - 1.5) * width, vals, width * 0.9, color=VAR[v][1], label=VAR[v][0])
        ax.set_xticks(x, [f'{p}\nε={e:g}' for p, e in cols])
        ax.set_title(title, loc='left', fontsize=9)
        ax.grid(axis='x', visible=False)
    variant_legend(fig, list(VAR), y=1.0, bars=True)
    header(fig, 'FT-Transformer removal: detection before and after editing',
           'Within-source test rounds; the removal threshold is re-picked on validation for every weight set.', y=1.1)
    out.fig(fig, 'wsp_detection')


# ============================================================== cost, budget, anatomy

def fig_cost(cost, paired, out):
    if cost is None or cost.empty:
        return
    tl = {'mv_ft': 'FT removal (mean/var)', 'mv_ds': 'Deep Sets (mean/var)', 'mv_st': 'Set Transformer (mean/var)',
          'freq_ft': 'FT-Transformer (freq. oracle)'}
    targets = [t for t in tl if t in set(cost.target)]
    c25 = cost[cost.budget == MAIN['budget']]
    if c25.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.6 * len(targets) + 2.6))
    ax = axes[0]
    labels = [('stage1', 'Stage 1 training (gradient)', '#52514e'), ('finetune', '+ fine-tune', VAR['ft'][1]),
              ('random', '+ random search', VAR['rand'][1]), ('wsp', '+ WSP (incl. LLM wait)', VAR['wsp'][1])]
    h = 0.8 / len(labels)
    for k, (m, lab, col) in enumerate(labels):
        vals = []
        for t in targets:
            d = c25[c25.target == t]
            vals.append(d['stage1_wall_s'].mean() / 60 if m == 'stage1' else d[d['mode'] == m]['stage2_wall_s'].mean() / 60)
        ax.barh(np.arange(len(targets)) + (k - 1.5) * h, vals, h * 0.9, color=col, label=lab)
    ax.set_yticks(range(len(targets)), [tl[t] for t in targets])
    if np.nanmax([b.get_width() for b in ax.patches] + [0]) > 0:
        ax.set_xscale('log')
    ax.set_xlabel('wall-clock minutes per run (log)')
    ax.set_title('Time per run', loc='left')
    ax.invert_yaxis()
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axes[1]
    w = c25[c25['mode'] == 'wsp'].groupby('target')[['input_tokens', 'cached_tokens', 'output_tokens',
                                                     'reasoning_tokens', 'usd']].mean()
    parts = [('uncached input', lambda r: r.input_tokens - r.cached_tokens, '#86b6ef'),
             ('cached input', lambda r: r.cached_tokens, '#cde2fb'),
             ('output (visible)', lambda r: r.output_tokens - r.reasoning_tokens, '#c0328a'),
             ('reasoning', lambda r: r.reasoning_tokens, '#7a1d57')]
    left = np.zeros(len(targets))
    for lab, fn, col in parts:
        vals = np.array([fn(w.loc[t]) / 1e6 if t in w.index else 0 for t in targets])
        ax.barh(range(len(targets)), vals, 0.6, left=left, color=col, label=lab, edgecolor=SURF, linewidth=1.5)
        left += vals
    for i, t in enumerate(targets):
        if t in w.index:
            tok = left[i] * 1e6
            tl_ = f'{tok / 1e6:.2f}M' if tok >= 1e5 else f'{tok / 1e3:.1f}k'
            usd = w.loc[t].usd
            ax.text(left[i], i, f'  {tl_} · ${usd:.2f}' if usd < 10 else f'  {tl_} · ${usd:.0f}', va='center',
                    fontsize=8, color=INK)
    ax.set_xlim(0, max(left.max(), 1e-9) * 1.35)
    ax.set_yticks(range(len(targets)), [])
    ax.set_xlabel('tokens per WSP run, millions')
    ax.set_title(f"LLM tokens per run (budget {MAIN['budget']})", loc='left')
    ax.invert_yaxis()
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axes[2]
    if paired is not None and len(paired):
        pa = paired[(paired.status == 'attacked') & (paired.vmode != 'stage1')].copy()
        pa['combined'] = np.sqrt(pa.ratio_stage1_mean * pa.ratio_stage1_var)
        jm = cost[cost.kind == 'mv'].copy()
        jm['unit'] = jm.job.str.split('/').str[-1]
        pa['unit'] = pa.apply(lambda r: f"single-source__{r.protocol}_eps{r.epsilon:g}__{r.train_sources}__{r.base}"
                                        f"__s{int(r.seed)}", axis=1)
        pa = main_variants(pa)
        pa['mode'] = pa.variant.map({'wsp': 'wsp', 'ft': 'finetune', 'rand': 'random'})
        m = pa.merge(jm[jm.budget == MAIN['budget']][['unit', 'mode', 'usd']], on=['unit', 'mode'], how='inner')
        for v, mode in (('ft', 'finetune'), ('rand', 'random'), ('wsp', 'wsp')):
            d = m[m['mode'] == mode]
            if len(d):
                ax.scatter(d.usd if mode == 'wsp' else np.zeros(len(d)),
                           d.combined, color=VAR[v][1], marker=VAR[v][2], s=36, label=VAR[v][0],
                           edgecolor=SURF, linewidth=1.2, zorder=3)
        ax.axhline(1, color=INK2, linewidth=1)
        ax.set_yscale('log')
        ax.set_xlabel('LLM dollars per run (baselines: $0)')
        ax.set_ylabel('test MSE ÷ Stage 1 (geo-mean of mean, var)')
        ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title('Gain against cost, attacked rounds (mean/var)', loc='left')
    header(fig, 'What weight editing costs',
           'Stage 1 = gradient training up to 50 epochs. Tokens come from API usage fields; dollars at list prices '
           '(configs/wsp/pricing.yaml).', y=1.04)
    out.fig(fig, 'wsp_cost')


def fig_budget_curve(paired, cost, out, sw):
    bc = sw['mv']['budget_curve']
    if paired is None or paired.empty:
        return
    p, e = bc['protocols'][0], float(bc['epsilon'][0])
    d = paired[(paired.protocol == p) & (paired.epsilon == e) & (paired.vmode != 'stage1')].copy()
    d['mode'], d['budget'] = d.vmode, d.vbudget.astype(int)
    out.table(d, 'wsp_budget_curve')
    bases = [b for b in BASE if b in set(d.base)]
    if not bases:
        return
    fig, axes = plt.subplots(len(bases), 3, figsize=(13, 2.8 * len(bases)), squeeze=False)
    for i, b in enumerate(bases):
        for j, st in enumerate(('attacked', 'clean')):
            ax = axes[i, j]
            for v in ('ft', 'rand', 'wsp'):
                g = d[(d.base == b) & (d['mode'] == v) & (d.status == st)].groupby('budget').ratio_stage1_mean.mean()
                if len(g):
                    ax.plot(g.index, g.values, **vstyle(v))
            ax.axhline(1, color=INK2, linewidth=1)
            ax.set_xticks([10, 25, 50])
            ax.set_title(f'{BASE[b]}: {st}, mean', loc='left', fontsize=9)
            if j == 0:
                ax.set_ylabel('test MSE ÷ Stage 1')
            if i == len(bases) - 1:
                ax.set_xlabel('probe budget')
        ax = axes[i, 2]
        if cost is not None and len(cost):
            t = {'ft_removal': 'mv_ft', 'deepsets_recovery': 'mv_ds', 'settransformer_recovery': 'mv_st'}[b]
            cc = cost[(cost.target == t) & (cost['mode'] == 'wsp') & cost.job.str.contains(f'{p}_eps{e:g}__')]
            g = cc.groupby('budget').total_tokens.mean() / 1e6
            if len(g):
                ax.plot(g.index, g.values, **vstyle('wsp'))
        ax.set_xticks([10, 25, 50])
        ax.set_title(f'{BASE[b]}: tokens per run', loc='left', fontsize=9)
        ax.set_ylabel('millions')
    variant_legend(fig, ['ft', 'rand', 'wsp'], y=1.0)
    header(fig, f'Does a bigger probe budget buy more? ({p}, ε = {e:g}, taxi)',
           'Within-source test rounds, 3 seeds. Gate evaluations scale with the probe budget (configs/wsp/stage2.yaml).',
           y=1.04)
    out.fig(fig, 'wsp_budget_curve')


def _module(name):
    parts = name.split('.')
    return '.'.join(parts[:-1]) if parts[-1] in ('weight', 'bias') or parts[-1].endswith('_weight') else name


def fig_edit_anatomy(jobs, out):
    rows, mods = [], []
    for j in jobs.values():
        if j['mode'] != 'wsp' or j['budget'] != MAIN['budget']:
            continue
        p = os.path.join(j['run_dir'], 'edits.jsonl')
        if not os.path.exists(p):
            continue
        t = 'freq_ft' if j['kind'] == 'freq' else {'ft_removal': 'mv_ft', 'deepsets_recovery': 'mv_ds',
                                                    'settransformer_recovery': 'mv_st'}[j['method']]
        for e in (json.loads(line) for line in open(p) if line.strip()):
            rows.append({'target': t, 'accept': bool(e.get('accept')), 'charged': bool(e.get('gate_charged')),
                         'frac_changed': e.get('frac_changed'), 'rel_l2': e.get('rel_l2')})
            if e.get('accept'):
                for tt in e.get('tensors', []):
                    mods.append({'target': t, 'module': _module(tt['tensor']), 'changed': tt['changed']})
    if not rows:
        return
    r, m = pd.DataFrame(rows), pd.DataFrame(mods)
    out.table(r, 'wsp_edits')
    targets = list(r.target.unique())
    fig, axes = plt.subplots(1, len(targets) + 1, figsize=(4 * (len(targets) + 1), 3.6))
    s = r.groupby('target').agg(submitted=('accept', 'size'), accepted=('accept', 'sum'),
                                cap_rejects=('charged', lambda x: (~x).sum()))
    ax = axes[0]
    y = np.arange(len(s))
    ax.barh(y, s.submitted, 0.6, color='#cde2fb', label='submitted')
    ax.barh(y, s.accepted, 0.6, color=VAR['wsp'][1], label='accepted')
    for i, (t, row) in enumerate(s.iterrows()):
        ax.text(row.submitted, i, f'  {row.accepted}/{row.submitted}', va='center', fontsize=8)
    ax.set_yticks(y, s.index)
    ax.set_title('Edits per target (all runs)', loc='left', fontsize=9)
    ax.legend(fontsize=7.5)
    ax.invert_yaxis()
    for ax, t in zip(axes[1:], targets):
        d = m[m.target == t].groupby('module').changed.sum().sort_values(ascending=False).head(8) if len(m) else []
        if len(d):
            share = d / d.sum()
            ax.barh(range(len(share)), share.values, 0.6, color=VAR['wsp'][1])
            ax.set_yticks(range(len(share)), share.index, fontsize=7.5)
            ax.invert_yaxis()
        ax.set_title(f'{t}: where accepted edits land', loc='left', fontsize=9)
        ax.set_xlabel('share of changed weights')
    header(fig, 'What the LLM changed', 'Accepted edits only, budget 25, all seeds; module = tensor name without '
           '.weight / .bias.', y=1.04)
    out.fig(fig, 'wsp_edit_anatomy')


def fig_overall(cells, asr, val, cost, out):
    if cells is None or cells.empty:
        return
    w = pooled(main_variants(cells), ['base', 'variant', 'view', 'status'])
    rows = []
    for b in BASE:
        for v in VAR:
            d = w[(w.base == b) & (w.variant == v)]
            if d.empty:
                continue
            get = lambda view, st, s: d[(d.view == view) & (d.status == st)][f'ratio_{s}'].sum() or np.nan  # noqa: E731
            row = {'model': SHORT[b], 'variant': VSHORT[v], 'within attacked mean': get('within', 'attacked', 'mean'),
                   'within attacked var': get('within', 'attacked', 'var'),
                   'within clean mean': get('within', 'clean', 'mean'), 'within clean var': get('within', 'clean', 'var'),
                   'cross attacked mean': get('cross', 'attacked', 'mean')}
            if asr is not None and len(asr):
                suffix = '' if v == 'stage1' else f'_{v}' + ('' if MAIN['budget'] == 25 else f"_b{MAIN['budget']}")
                a = asr[(asr.view == 'within') & (asr.attack != 'none') & (asr.method == b + suffix)]
                if len(a):
                    rr = asr_rates(a, ['method'])
                    row['ASR after'] = float(rr.asr_after_joint.iloc[0])
            if cost is not None and len(cost) and v != 'stage1':
                t = {'ft_removal': 'mv_ft', 'deepsets_recovery': 'mv_ds', 'settransformer_recovery': 'mv_st'}[b]
                mode = {'wsp': 'wsp', 'ft': 'finetune', 'rand': 'random'}[v]
                cc = cost[(cost.target == t) & (cost['mode'] == mode) & (cost.budget == MAIN['budget'])]
                if len(cc):
                    row['accepted edits / run'] = cc.edits_accepted.mean()
                    row['tokens / run (M)'] = cc.total_tokens.mean() / 1e6
                    row['$ / run'] = cc.usd.mean()
            rows.append(row)
    t = pd.DataFrame(rows)
    out.table(t, 'wsp_overall')
    if t.empty:
        return
    show = t.copy()
    for c in show.columns[2:]:
        show[c] = show[c].map(lambda x: '' if pd.isna(x) else (f'{x:.2f}' if abs(x) < 100 else f'{x:,.0f}'))
    h = 0.3 * len(show) + 1.6
    fig = plt.figure(figsize=(min(18, 2 + 1.15 * len(show.columns)), h))
    ax = fig.add_axes([0.01, 0.02, 0.98, 1 - 1.15 / h])
    ax.axis('off')
    labels = [c.replace(' / ', ' /\n').replace('within ', 'within\n').replace('cross ', 'cross\n')
              for c in show.columns]
    tb = ax.table(cellText=show.values, colLabels=labels, loc='upper center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.auto_set_column_width(list(range(len(labels))))
    tb.scale(1, 1.35)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0:
            cell.set_text_props(fontweight='bold', color=INK)
            cell.set_facecolor('#f0efec')
            cell.set_height(cell.get_height() * 1.8)
    header(fig, 'Summary: MSE ÷ no defence (< 1 better), attack success, editing cost',
           'Test rounds pooled over PM/SR, ε, β, targets, attacks and seeds; taxi-trained models.', y=1 - 0.75 / h)
    out.fig(fig, 'wsp_overall_table')


# ============================================================== frequency oracle

RUN_LABEL = {'cross/OUE_Zipf_test': 'OUE: emoji+fire → zipf', 'cross/OUE_Emoji_test': 'OUE: zipf+fire → emoji',
             'cross/OUE_Fire_test': 'OUE: zipf+emoji → fire', 'in_dist/Fire_OUE': 'OUE: fire (in-distribution)',
             'cross/OLH_Server_Fire_test': 'OLH-Server → fire (control)'}


def fig_freq_overall(ft, out):
    if ft is None or ft.empty:
        return
    ft = main_variants(ft)
    runs = [r for r in RUN_LABEL if r in set(ft.run)]
    vs = [v for v in VAR if v in set(ft.variant)]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(runs))
    width = 0.8 / max(len(vs), 1)
    for k, v in enumerate(vs):
        g = ft[ft.variant == v].groupby('run').F1_Score
        mean = np.array([g.mean().get(r, np.nan) for r in runs])
        lo = mean - np.array([g.min().get(r, np.nan) for r in runs])
        hi = np.array([g.max().get(r, np.nan) for r in runs]) - mean
        ax.bar(x + (k - (len(vs) - 1) / 2) * width, mean, width * 0.9, color=VAR[v][1], label=VAR[v][0],
               yerr=[lo, hi], error_kw={'ecolor': INK2, 'elinewidth': 1, 'capsize': 2})
    ax.set_xticks(x, [RUN_LABEL[r] for r in runs], fontsize=8)
    ax.set_ylabel('test F1 (natural prevalence)')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='x', visible=False)
    variant_legend(fig, vs, y=1.0, bars=True)
    seeds = ' / '.join(str(s) for s in sorted(ft.seed.unique()))
    header(fig, 'Frequency oracle: test F1 before and after editing',
           f'Mean over seeds {seeds}, whiskers = min to max. Cross runs: the gate sees only the two training datasets.',
           y=1.1)
    out.fig(fig, 'wsp_freq_overall')


def fig_freq_cross(fc, out):
    from loader import load_oat
    frames = []
    base = os.path.join(ROOT, 'results', fc['out_root'], 'eval')
    if not os.path.isdir(base):
        return
    for name in sorted(os.listdir(base)):
        v, _, seed = name.rpartition('_s')
        mode, budget = compare_wsp.variant_parts(v)
        if mode not in VAR or (mode != 'stage1' and budget != MAIN['budget']):
            continue
        v = mode
        try:
            df = load_oat(f"{fc['out_root']}/eval/{name}")
        except FileNotFoundError:
            continue
        frames.append(df.assign(variant=v, seed=int(seed)))
    if not frames:
        return
    d = pd.concat(frames)
    d = d[(d.source == 'cross') & (d.protocol.str.upper() == 'OUE') & (d.param == 'epsilon')]
    out.table(d, 'wsp_freq_cross')
    if d.empty:
        return
    sets = sorted(d.dataset.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.2), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], sets):
        for v in VAR:
            g = d[(d.dataset == ds) & (d.variant == v)].groupby('x').F1_Score.mean()
            if len(g):
                ax.plot(g.index, g.values, **vstyle(v))
        ax.set_title(f'tested on {ds}', loc='left')
        ax.set_xlabel('ε')
        ax.set_ylim(0, 1.02)
    axes[0, 0].set_ylabel('F1')
    variant_legend(fig, list(VAR), y=1.0)
    header(fig, 'OUE cross-dataset detection against ε', 'Trained on the two other datasets; mean over seeds.', y=1.12)
    out.fig(fig, 'wsp_freq_cross')


def main():
    sw = load_yaml(SWEEP)
    jobs = all_jobs()
    out = Out(OUT_DIR)
    tdir = os.path.join(OUT_DIR, 'tables')
    if not os.path.exists(os.path.join(tdir, 'mv_paired.csv')) and not os.path.exists(os.path.join(tdir, 'freq_test.csv')):
        compare_wsp.main()
    read = lambda n: pd.read_csv(os.path.join(tdir, n)) if os.path.exists(os.path.join(tdir, n)) else None  # noqa: E731
    cells, paired, ft = read('mv_cells.csv'), read('mv_paired.csv'), read('freq_test.csv')
    cpath = os.path.join(WSP_HOME, 'cost_summary.csv')
    cost = pd.read_csv(cpath) if os.path.exists(cpath) else None
    asr = load_asr(rel(sw['mv']['out']), 'eval')
    print(f"main probe budget shown: {pick_main_budget(cells, ft)}")
    if cells is not None and len(cells):
        fig_overview(cells, out)
        fig_transfer(cells, out)
        fig_mse_eps(cells, out)
        fig_detection(cells, out)
        fig_overall(cells, asr, None, cost, out)
    fig_asr(asr, out)
    fig_budget_curve(paired, cost, out, sw)
    fig_cost(cost, paired, out)
    fig_edit_anatomy(jobs, out)
    fig_freq_overall(ft, out)
    fig_freq_cross(sw['freq'], out)
    print(f'{len(out.figs)} figures in {OUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
