"""Figures for the SR/PM utility-recovery benchmark (SVG + PNG + manifest).

Methods keep one colour and marker across every figure. MSE uses a log axis;
accuracy gain (which can be negative) uses a symmetric-log axis, never a plain log.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, GRID, MUTED = '#fcfcfb', '#0b0b0b', '#e1e0d9', '#898781'

METHOD_STYLE = {
    'standard':                {'label': 'Standard SR/PM', 'color': '#52514e', 'marker': 'o', 'ls': '-'},
    'constant':                {'label': 'Constant (train)', 'color': '#898781', 'marker': 'x', 'ls': ':'},
    'oracle_removal':          {'label': 'Oracle removal', 'color': '#0b0b0b', 'marker': '*', 'ls': '--'},
    'kmeans':                  {'label': 'K-means (Li et al.)', 'color': '#eb6834', 'marker': 's', 'ls': '-'},
    'mlp_removal':             {'label': 'MLP removal', 'color': '#1baf7a', 'marker': '^', 'ls': '-'},
    'ft_removal':              {'label': 'FT-Transformer removal', 'color': '#2a78d6', 'marker': 'v', 'ls': '-'},
    'catboost_recovery':       {'label': 'CatBoost recovery', 'color': '#c2410c', 'marker': 'D', 'ls': '-.'},
    'deepsets_recovery':       {'label': 'Deep Sets recovery', 'color': '#4a3aa7', 'marker': 'P', 'ls': '-'},
    'settransformer_recovery': {'label': 'Set Transformer recovery', 'color': '#b5179e', 'marker': 'X', 'ls': '-'},
}


def _style(name):
    base = name.split('[')[0]
    st = dict(METHOD_STYLE.get(base, {'label': name, 'color': MUTED, 'marker': '.', 'ls': '-'}))
    if '[' in name:
        st['label'] = f"{st['label']} {name[name.index('['):]}"
        st['ls'] = '--'
    return st


def _axes(ncols, title):
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 3.8), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.suptitle(title, color=INK, fontsize=11)
    return fig, axes


def save(fig, path_no_ext, manifest: dict):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_no_ext + '.svg')
    fig.savefig(path_no_ext + '.png', dpi=150)
    plt.close(fig)
    with open(path_no_ext + '.manifest.json', 'w') as f:
        json.dump(manifest, f, indent=1, default=str)


def sweep(agg, x: str, filters: dict, path_no_ext: str, title: str, inputs=None):
    """MSE_mean and MSE_variance vs x (log y), one line per method, CI band if present."""
    d = agg.copy()
    for k, v in filters.items():
        d = d[d[k] == v]
    fig, axes = _axes(2, title)
    for ax, col, lab in ((axes[0], 'mse_mean', 'MSE of mean'), (axes[1], 'mse_var', 'MSE of variance')):
        for method, g in d.groupby('method'):
            g = g.groupby(x)[col].mean().sort_index()
            if g.empty:
                continue
            st = _style(method)
            ax.plot(g.index, g.values, label=st['label'], color=st['color'], marker=st['marker'],
                    linestyle=st['ls'], linewidth=1.6, markersize=4.5)
        ax.set_yscale('log')
        ax.set_xlabel({'beta': r'$\beta$', 'epsilon': r'$\epsilon$'}.get(x, x))
        ax.set_ylabel(lab + ' (vs genuine)')
    axes[1].legend(fontsize=7, frameon=False, loc='best')
    save(fig, path_no_ext, {'figure': os.path.basename(path_no_ext), 'x': x, 'filters': filters,
                            'inputs': inputs, 'cells': int(len(d))})


def ag_bars(agg, filters: dict, path_no_ext: str, title: str, inputs=None):
    """Accuracy gain (mean, variance) per method on a symmetric-log axis; negative gains kept."""
    d = agg.copy()
    for k, v in filters.items():
        d = d[d[k] == v]
    fig, axes = _axes(2, title)
    g = d.groupby('method')[['ag_mean', 'ag_var']].mean()
    for ax, col, lab in ((axes[0], 'ag_mean', 'AG (mean)'), (axes[1], 'ag_var', 'AG (variance)')):
        vals = g[col]
        colors = [_style(m)['color'] for m in vals.index]
        ax.barh([_style(m)['label'] for m in vals.index], vals.values, color=colors)
        ax.axvline(0, color=INK, linewidth=0.8)
        lim = np.nanmax(np.abs(vals.values)) if len(vals) else 1.0
        ax.set_xscale('symlog', linthresh=max(lim * 1e-3, 1e-12))
        ax.set_xlabel(lab + ' = MSE_before - MSE_after')
    save(fig, path_no_ext, {'figure': os.path.basename(path_no_ext), 'filters': filters, 'inputs': inputs})


def heatmaps(agg, value: str, filters: dict, path_no_ext: str, title: str, inputs=None):
    """epsilon x beta grid per method. AG (can be negative) uses a diverging scale centred at 0."""
    from matplotlib.colors import TwoSlopeNorm
    d = agg.copy()
    for k, v in filters.items():
        d = d[d[k] == v]
    methods = sorted(d['method'].unique())
    if not methods:
        return
    fig, axes = _axes(len(methods), title)
    piv = {m: d[d.method == m].pivot_table(index='epsilon', columns='beta', values=value) for m in methods}
    lim = max(float(np.nanmax(np.abs(p.values))) for p in piv.values() if p.size) or 1.0
    diverging = value.startswith('ag_')
    for ax, m in zip(axes, methods):
        p = piv[m]
        kw = {'cmap': 'RdBu', 'norm': TwoSlopeNorm(0.0, -lim, lim)} if diverging else {'cmap': 'magma_r'}
        im = ax.imshow(p.values, aspect='auto', origin='lower', **kw)
        ax.set_xticks(range(len(p.columns)), [f'{c:g}' for c in p.columns], fontsize=7)
        ax.set_yticks(range(len(p.index)), [f'{e:g}' for e in p.index], fontsize=7)
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$\epsilon$')
        ax.set_title(_style(m)['label'], fontsize=8)
        ax.grid(False)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label=value)
    save(fig, path_no_ext, {'figure': os.path.basename(path_no_ext), 'value': value, 'filters': filters,
                            'inputs': inputs})


def transfer_matrix(agg, value: str, filters: dict, path_no_ext: str, title: str, inputs=None):
    """Grouped bars: destination source x training-source set, one group per method."""
    d = agg.copy()
    for k, v in filters.items():
        d = d[d[k] == v]
    g = d.groupby(['method', 'source', 'train_sources'])[value].mean().reset_index()
    dests = sorted(g['source'].unique())
    if not dests:
        return
    fig, axes = _axes(len(dests), title)
    for ax, dest in zip(axes, dests):
        sub = g[g.source == dest]
        trains = sorted(sub['train_sources'].unique(), key=lambda t: (t != dest, '+' in t, t))
        methods = sorted(sub['method'].unique())
        w = 0.8 / max(len(methods), 1)
        for i, m in enumerate(methods):
            vals = [sub[(sub.method == m) & (sub.train_sources == t)][value].mean() for t in trains]
            st = _style(m)
            ax.bar(np.arange(len(trains)) + i * w, vals, width=w, color=st['color'], label=st['label'])
        ax.set_xticks(np.arange(len(trains)) + 0.4 - w / 2, [('within: ' if t == dest else '') + t for t in trains],
                      rotation=30, ha='right', fontsize=7)
        ax.set_title(f'test on {dest}', fontsize=9)
        if not value.startswith('ag_'):
            ax.set_yscale('log')
        else:
            ax.axhline(0, color=INK, linewidth=0.8)
    axes[-1].legend(fontsize=6, frameon=False)
    save(fig, path_no_ext, {'figure': os.path.basename(path_no_ext), 'value': value, 'filters': filters,
                            'inputs': inputs})


def runtime_bars(results, path_no_ext: str, title: str, inputs=None):
    """Mean per-round inference time and mean fit time per method (log axis)."""
    g = results.groupby('method').agg(infer=('runtime', 'mean'), fit=('fit_time', 'mean'),
                                      rss=('peak_rss_mb', 'max'))
    fig, axes = _axes(3, title)
    for ax, col, lab in ((axes[0], 'infer', 's / round (inference)'), (axes[1], 'fit', 's (fit, per unit)'),
                         (axes[2], 'rss', 'peak RSS (MB)')):
        ax.barh([_style(m)['label'] for m in g.index], g[col].values, color=[_style(m)['color'] for m in g.index])
        if col != 'rss':
            ax.set_xscale('log')
        ax.set_xlabel(lab)
    save(fig, path_no_ext, {'figure': os.path.basename(path_no_ext), 'inputs': inputs})


COMBO_STYLE = {('SR', 'IPA'): ('#d1495b', '-', 'o'), ('PM', 'IPA'): ('#1baf7a', '-', 's'),
               ('SR', 'OPA'): ('#2a78d6', '--', '^'), ('PM', 'OPA'): ('#0b0b0b', '--', 'D')}


def kmeans_fig12(panels, path_no_ext: str, title: str, inputs=None):
    """Fig. 12 analogue. panels: list of (summary DataFrame, x column, x label). Rows: AG mean / AG variance."""
    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 6.0), facecolor=SURFACE, squeeze=False)
    for j, (df, x, xlab) in enumerate(panels):
        for i, col in enumerate(('ag_mean', 'ag_var')):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.7)
            for (prot, att), g in df.groupby(['protocol', 'attack']):
                g = g.groupby(x)[col].mean().sort_index()
                if g.empty:
                    continue
                c, ls, mk = COMBO_STYLE.get((prot, att), (MUTED, '-', '.'))
                ax.plot(g.index, g.values, color=c, linestyle=ls, marker=mk, markersize=4, label=f'{att} on {prot}')
            ax.axhline(0, color=INK, linewidth=0.8)
            vals = np.abs(df[col].to_numpy(float))
            vals = vals[np.isfinite(vals) & (vals > 0)]
            ax.set_yscale('symlog', linthresh=float(np.percentile(vals, 10)) if len(vals) else 1e-6)
            ax.set_xlabel(xlab)
            ax.set_ylabel('AG (mean)' if col == 'ag_mean' else 'AG (variance)')
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    save(fig, path_no_ext, {'figure': os.path.basename(path_no_ext), 'inputs': inputs,
                            'note': 'AG = MSE_before - MSE_after vs genuine moments; symlog y; negatives kept'})


ATTACK_COLOR = {'IPA': '#d1495b', 'OPA-SR': '#2a78d6', 'OPA-PM': '#0b0b0b'}


def budget_curves_plot(df, path_no_ext: str, title: str, inputs=None):
    """beta_min vs target-mean offset and vs target-variance ratio; solid = attacker estimate,
    dashed = oracle diagnostic (exact genuine information)."""
    sources = sorted(df['source'].unique())
    fig, axes = plt.subplots(2, len(sources), figsize=(4.4 * len(sources), 6.2), facecolor=SURFACE, squeeze=False)
    for j, src in enumerate(sources):
        for i, (axis, xlab) in enumerate((('mean', r'target mean $-$ true mean'),
                                          ('variance', r'target variance / true variance'))):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.7)
            d = df[(df.source == src) & (df.axis == axis)]
            for (att, who), g in d.groupby(['attack', 'knowledge']):
                g = g.sort_values('x')
                ax.plot(g['x'], g['beta_min'], color=ATTACK_COLOR.get(att, MUTED),
                        linestyle='-' if who == 'attacker estimate' else '--', linewidth=1.5,
                        label=f'{att} ({who})')
            ax.set_ylim(0, 1)
            ax.set_xlabel(xlab)
            ax.set_ylabel(r'$\beta_{\min}$')
            ax.set_title(src, fontsize=9)
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.suptitle(title, fontsize=11)
    save(fig, path_no_ext, {'figure': os.path.basename(path_no_ext), 'inputs': inputs,
                            'note': 'solid: attacker planning estimate; dashed: oracle diagnostic; gaps = infeasible'})


def utility_vs_budget(results, path_no_ext: str, title: str, x: str = 'budget_factor', inputs=None):
    """MSE (mean, variance) vs m / m_min for each method; m_min > 0 only. Evaluation metadata."""
    d = results[results['m_min'].fillna(0) > 0]
    fig, axes = _axes(2, title)
    for ax, col, lab in ((axes[0], 'sq_err_mean', 'MSE of mean'), (axes[1], 'sq_err_var', 'MSE of variance')):
        for method, g in d.groupby('method'):
            g = g.groupby(x)[col].mean().sort_index()
            st = _style(method)
            ax.plot(g.index, g.values, label=st['label'], color=st['color'], marker=st['marker'],
                    linestyle=st['ls'], linewidth=1.6, markersize=4.5)
        ax.set_yscale('log')
        ax.set_xlabel(r'$m / m_{\min}$' + (' (nominal factor)' if x == 'budget_factor' else ''))
        ax.set_ylabel(lab + ' (vs genuine)')
    axes[1].legend(fontsize=7, frameon=False)
    save(fig, path_no_ext, {'figure': os.path.basename(path_no_ext), 'x': x, 'inputs': inputs})


def construction_by_factor(manifest_df, path_no_ext: str, title: str, inputs=None):
    """Share of theory-feasible / constructed / allocation-infeasible rounds per budget factor."""
    d = manifest_df.copy()
    fig, axes = _axes(1, title)
    ax = axes[0]
    tab = d.groupby(['budget_factor', 'status']).size().unstack(fill_value=0)
    tab = tab.div(tab.sum(axis=1), axis=0)
    bottom = np.zeros(len(tab))
    palette = {'feasible': '#1baf7a', 'generated_with_target_error': '#2a78d6', 'allocation_infeasible': '#eb6834',
               'infeasible_target': '#898781', 'below_budget': '#4a3aa7', 'solver_failure': '#d1495b'}
    for col in tab.columns:
        ax.bar([f'{f:g}' for f in tab.index], tab[col].values, bottom=bottom, color=palette.get(col, MUTED), label=col)
        bottom += tab[col].values
    ax.set_xlabel(r'budget factor $m / m_{\min}$')
    ax.set_ylabel('share of rounds')
    ax.legend(fontsize=7, frameon=False)
    save(fig, path_no_ext, {'figure': os.path.basename(path_no_ext), 'inputs': inputs})
