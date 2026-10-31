"""Attack-implementation validation at paper scale, and attack-feasibility maps.

attack_validation: generates rounds in memory (nothing stored; rounds can hold
millions of reports) and records the server's standard estimate, the target and
the genuine moments. Summaries give MSE to the attacker's target (paper Sec. 7.1),
which validates the attack implementations against Figs. 5-11. These are not
defense results.

feasibility_maps: beta_min over a grid of benchmark-scale targets, from exact
knowledge of the full source (Figs. 2-4). Conditions follow feasibility.py
(corrected Eqs. 8-13).
"""

import json
import os

import numpy as np
import pandas as pd

from .experiment import enumerate_specs, make_banks
from .feasibility import beta_min
from .knowledge import exact
from .mechanisms import get_mechanism
from .metrics import shift_achieved
from .scenarios import generate_round
from .sources import SOURCES, load_source
from .transforms import Transform


def attack_validation(cfg: dict, out_dir: str, split: str = 'all', log_every: int = 50):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'attack_validation_rounds.csv')
    done = set(pd.read_csv(path, usecols=['round_id'])['round_id']) if os.path.exists(path) else set()
    banks = make_banks(cfg)
    specs = [s for s in enumerate_specs(cfg, split, banks) if s.round_id not in done]
    rows = []
    for i, s in enumerate(specs):
        r = generate_round(s, banks[s.source].values(split, s.rep), SOURCES[s.source].printed_targets)
        row = {k: v for k, v in s.to_dict().items() if k != 'domain'}
        row.update(status=r.status, realized_beta=r.meta['realized_beta'], n_fake=r.meta['n_fake'],
                   target_mean=r.meta.get('target_mean'), target_variance=r.meta.get('target_variance'),
                   **{k: r.labels.get(k) for k in ('genuine_mean', 'genuine_variance', 'clean_mean',
                                                   'clean_variance', 'standard_mean', 'standard_variance')})
        rows.append(row)
        if (i + 1) % log_every == 0 or i + 1 == len(specs):
            pd.DataFrame(rows).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
            rows = []
            print(f"  attack validation {i + 1:,}/{len(specs):,}", flush=True)
    return summarize_attack_validation(pd.read_csv(path), out_dir)


def summarize_attack_validation(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    by = ['source', 'protocol', 'epsilon', 'beta', 'attack', 'target', 'target_mean_value', 'target_var_value',
          'dS1_n', 'dS2_n', 'dn', 'spread_ratio', 'size_mode', 'square_mode', 'knowledge_regime']
    by = [b for b in by if b in df]
    ok = df[df['status'].isin(['feasible', 'generated_with_target_error'])].copy()
    ok['se_target_mean'] = (ok['standard_mean'] - ok['target_mean']) ** 2
    ok['se_target_var'] = (ok['standard_variance'] - ok['target_variance']) ** 2
    ok['shift_mean'] = shift_achieved(ok['standard_mean'], ok['genuine_mean'], ok['target_mean'])
    ok['shift_mean_paired'] = shift_achieved(ok['standard_mean'], ok['genuine_mean'], ok['target_mean'],
                                             baseline=ok['clean_mean'])
    agg = ok.groupby(by, dropna=False).agg(
        rounds=('round_id', 'count'), mse_target_mean=('se_target_mean', 'mean'),
        mse_target_var=('se_target_var', 'mean'), shift_mean_median=('shift_mean', 'median'),
        shift_mean_paired_median=('shift_mean_paired', 'median'),
        target_mean=('target_mean', 'mean'), target_variance=('target_variance', 'mean')).reset_index()
    status = df.groupby(by, dropna=False)['status'].agg(lambda s: dict(s.value_counts())).reset_index()
    agg = status.merge(agg, on=by, how='left')
    agg.to_csv(os.path.join(out_dir, 'attack_validation_summary.csv'), index=False)
    return agg


def feasibility_maps(sources, epsilon: float, out_dir: str, mean_grid=None, var_grid=None,
                     square_mode: str = 'raw', beta_cap: float = 0.99):
    """beta_min per (mu_t, sigma_t^2) for IPA, OPA-SR and OPA-PM; >= beta_cap means infeasible."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mean_grid = np.linspace(-1.0, 0.8, 19) if mean_grid is None else np.asarray(mean_grid)
    var_grid = np.linspace(0.005, 1.205, 17) if var_grid is None else np.asarray(var_grid)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for src in sources:
        d = load_source(src)
        tf = Transform(*d.spec.domain, square_mode=square_mode)
        k = exact(d.values)
        m_max = int(beta_cap / (1 - beta_cap) * len(d))
        for label, attack, protocol in (('IPA', 'IPA', 'SR'), ('OPA-SR', 'OPA', 'SR'), ('OPA-PM', 'OPA', 'PM')):
            mech = get_mechanism(protocol, epsilon)
            grid = np.full((len(var_grid), len(mean_grid)), np.nan)
            for i, v in enumerate(var_grid):
                for j, mu in enumerate(mean_grid):
                    b, _ = beta_min(attack, mech, tf, k, float(mu), float(v), m_max=m_max)
                    grid[i, j] = b
                    rows.append({'source': src, 'attack': label, 'epsilon': epsilon, 'target_mean': mu,
                                 'target_variance': v, 'beta_min': b})
            fig, ax = plt.subplots(figsize=(4.4, 3.4))
            im = ax.imshow(np.nan_to_num(grid, nan=1.0), origin='upper', aspect='auto', vmin=0, vmax=1,
                           cmap='magma', extent=[mean_grid[0], mean_grid[-1], var_grid[-1], var_grid[0]])
            t = tf.to_g1(d.values)
            ax.scatter([t.mean()], [t.var()], marker='s', s=30, facecolors='none', edgecolors='#2a78d6')
            ax.set_xlabel(r'target mean $\mu_t$')
            ax.set_ylabel(r'target variance $\sigma_t^2$')
            ax.set_title(f'{label}, {src}, $\\epsilon$={epsilon:g} (oracle diagnostic)', fontsize=9)
            fig.colorbar(im, ax=ax, label=r'min $\beta$ (1 = infeasible)')
            fig.tight_layout()
            base = os.path.join(out_dir, f'feasibility_{label}_{src}_eps{epsilon:g}')
            fig.savefig(base + '.svg')
            fig.savefig(base + '.png', dpi=150)
            plt.close(fig)
            with open(base + '.manifest.json', 'w') as f:
                json.dump({'source': src, 'attack': label, 'epsilon': epsilon, 'square_mode': square_mode,
                           'knowledge': 'exact (full source): ORACLE DIAGNOSTIC, not attacker planning',
                           'beta_cap': beta_cap,
                           'conditions': 'feasibility.py (corrected Eqs. 8-13)'}, f, indent=1)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, f'feasibility_eps{epsilon:g}.csv'), index=False)
    return out


# ---------------------------------------------------------------- k-means defense verification
_WORKER = {}


def _kmeans_init(cfg):
    _WORKER['banks'] = make_banks(cfg)


def _kmeans_task(args):
    """Generate one round in the worker; standard and k-means estimates at every rate (paired)."""
    from .methods.baselines import KMeansRecovery
    spec, split, rates, n_subsets = args
    r = generate_round(spec, _WORKER['banks'][spec.source].values(split, spec.rep),
                       SOURCES[spec.source].printed_targets)
    base = {k: v for k, v in spec.to_dict().items() if k != 'domain'}
    base.update(status=r.status, realized_beta=r.meta['realized_beta'], target_mean=r.meta.get('target_mean'),
                target_variance=r.meta.get('target_variance'),
                **{k: r.labels.get(k) for k in ('genuine_mean', 'genuine_variance', 'standard_mean',
                                                'standard_variance', 'clean_mean', 'clean_variance')})
    if not r.generated:
        return [dict(base, rate=np.nan, kmeans_mean=np.nan, kmeans_variance=np.nan)]
    obs = r.observation()
    rows = []
    for rate in rates:
        out = KMeansRecovery(rate=rate, n_subsets=n_subsets, n_jobs=1)._predict(obs, rate)
        rows.append(dict(base, rate=rate, kmeans_mean=out.mean_raw, kmeans_variance=out.var_raw))
    return rows


def kmeans_verification(cfg: dict, out_dir: str, rates, n_subsets: int = 5000, split: str = 'all',
                        n_jobs: int = 12, flush_every: int = 24, max_restarts: int = 5):
    """Paper Sec. 8.1 / Fig. 12: AG = MSE_before - MSE_after against the genuine moments, per rate.

    MSE_before uses the server's standard estimate on the same round, so gains are paired.
    Raw (unprojected) estimates, as in the paper. Resumable by round id. If a worker process
    is killed (seen on this machine without a reproducible cause; the same rounds run fine
    in-process), the saved rows are kept and the pool restarts for the remaining rounds.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'kmeans_verification_rounds.csv')
    all_specs = enumerate_specs(cfg, split, make_banks(cfg))
    for attempt in range(max_restarts + 1):
        done = set(pd.read_csv(path, usecols=['round_id'])['round_id']) if os.path.exists(path) else set()
        specs = [s for s in all_specs if s.round_id not in done]
        if not specs:
            break
        rows, n = [], 0
        try:
            with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context('spawn'), initializer=_kmeans_init,
                                     initargs=(cfg,)) as ex:
                for res in ex.map(_kmeans_task, [(s, split, tuple(rates), n_subsets) for s in specs], chunksize=1):
                    rows += res
                    n += 1
                    if n % flush_every == 0 or n == len(specs):
                        _append_rows(path, rows)
                        rows = []
                        print(f"  k-means verification {len(done) + n:,}/{len(all_specs):,} rounds", flush=True)
        except BrokenProcessPool:
            if rows:
                _append_rows(path, rows)
            print(f"  worker pool broke (attempt {attempt + 1}); resuming from saved rows", flush=True)
    else:
        raise RuntimeError(f"k-means verification incomplete after {max_restarts} restarts")
    return summarize_kmeans_verification(pd.read_csv(path), out_dir)


def _append_rows(path: str, rows):
    """Append with the existing header's column order, so every line has the same fields."""
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        cols = list(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in df.columns if c not in cols]
        if missing:
            raise ValueError(f"new columns {missing} not in {path}")
        df = df.reindex(columns=cols)
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)


def summarize_kmeans_verification(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    by = [c for c in ('protocol', 'attack', 'beta', 'target', 'target_mean_value', 'target_var_value',
                      'spread_ratio', 'rate') if c in df]
    ok = df[df['status'].isin(['feasible', 'generated_with_target_error'])].copy()
    ok['se_before_mean'] = (ok['standard_mean'] - ok['genuine_mean']) ** 2
    ok['se_after_mean'] = (ok['kmeans_mean'] - ok['genuine_mean']) ** 2
    ok['se_before_var'] = (ok['standard_variance'] - ok['genuine_variance']) ** 2
    ok['se_after_var'] = (ok['kmeans_variance'] - ok['genuine_variance']) ** 2
    agg = ok.groupby(by, dropna=False).agg(
        rounds=('round_id', 'nunique'), mse_before_mean=('se_before_mean', 'mean'),
        mse_after_mean=('se_after_mean', 'mean'), mse_before_var=('se_before_var', 'mean'),
        mse_after_var=('se_after_var', 'mean')).reset_index()
    agg['ag_mean'] = agg['mse_before_mean'] - agg['mse_after_mean']
    agg['ag_var'] = agg['mse_before_var'] - agg['mse_after_var']
    status = df.drop_duplicates('round_id').groupby([c for c in by if c != 'rate'], dropna=False)['status'] \
        .agg(lambda s: dict(s.value_counts())).reset_index()
    agg = agg.merge(status, on=[c for c in by if c != 'rate'], how='left')
    agg.to_csv(os.path.join(out_dir, 'kmeans_verification_summary.csv'), index=False)
    return agg


# ---------------------------------------------------------------- minimum-budget curves
def budget_curves(sources, epsilon: float, out_dir: str, mean_offsets=None, var_ratios=None,
                  square_mode: str = 'normalized', h: int = 1000, n_e_rule: str = 'sig2', seed: int = 0):
    """beta_min against target mean (variance preserved) and against target variance (mean preserved).

    Two curves per attack: the attacker's planning estimate (compromised sample of h users,
    n_e rounded) and the oracle diagnostic (exact genuine sums). Full sources.
    """
    from .feasibility import minimum_budget
    from .knowledge import from_compromised, round_count
    mean_offsets = np.linspace(-0.3, 0.3, 25) if mean_offsets is None else np.asarray(mean_offsets)
    var_ratios = np.linspace(0.25, 3.0, 23) if var_ratios is None else np.asarray(var_ratios)
    rows = []
    for src in sources:
        d = load_source(src)
        tf = Transform(*d.spec.domain, square_mode=square_mode)
        t = tf.to_g1(d.values)
        mu, var = float(t.mean()), float(t.var())
        planned, _ = from_compromised(d.values, h, round_count(len(d), n_e_rule), np.random.default_rng(seed))
        oracle = exact(d.values)
        grid = [('mean', mu + o, var, o) for o in mean_offsets] + [('variance', mu, var * r, r) for r in var_ratios]
        for label, attack, protocol in (('IPA', 'IPA', 'SR'), ('OPA-SR', 'OPA', 'SR'), ('OPA-PM', 'OPA', 'PM')):
            mech = get_mechanism(protocol, epsilon)
            for axis, mt, vt, x in grid:
                for who, k in (('attacker estimate', planned), ('oracle diagnostic', oracle)):
                    r = minimum_budget(attack, mech, tf, k, float(mt), float(vt), oracle=(who != 'attacker estimate'))
                    rows.append({'source': src, 'attack': label, 'epsilon': epsilon, 'axis': axis, 'x': float(x),
                                 'target_mean': float(mt), 'target_variance': float(vt), 'knowledge': who,
                                 'm_min': r['m_min'], 'beta_min': r['beta_min'], 'status': r['status'],
                                 'true_mean': mu, 'true_variance': var})
    out = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, f'budget_curves_eps{epsilon:g}.csv'), index=False)
    return out
