"""Feature views computed from a round's observation only (Round.observation()).

Nothing here sees labels, sources, targets, beta or generation order; the input
is the observation dict (reports, groups, counts, protocol, epsilon, domain,
square_mode). Per-report features depend on the report's value, its group and
the multiset of reports in the round, so shuffling a round permutes them
consistently, and identical SR (group, sign) cells get identical rows.
"""

import numpy as np
from scipy.optimize import nnls
from scipy.stats import rankdata

from .estimator import G1, G2, estimate
from .mechanisms import PM, SR, get_mechanism
from .transforms import Transform

NBR_WIDTHS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
KNN = (1, 4, 16)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
HIST_BINS = 8


def _setup(obs):
    tf = Transform(*obs['domain'], square_mode=obs['square_mode'])
    mech = get_mechanism(obs['protocol'], obs['epsilon'])
    return tf, mech


def standard_moments(obs):
    """Standard decoded (mean, variance) on the benchmark scale."""
    tf, mech = _setup(obs)
    e = estimate(obs['reports'], obs['groups'], mech, tf).normalized(tf)
    return e.mean, e.variance


def _knn(sorted_y: np.ndarray, k: int) -> np.ndarray:
    """Distance from each sorted value to its k-th nearest other value."""
    n = len(sorted_y)
    if n <= 1:
        return np.zeros(n)
    k = min(k, n - 1)
    pad = np.r_[np.full(k, np.inf), sorted_y, np.full(k, np.inf)]
    offs = np.r_[np.arange(-k, 0), np.arange(1, k + 1)]
    d = np.abs(pad[np.arange(n)[:, None] + k + offs[None, :]] - sorted_y[:, None])
    return np.partition(d, k - 1, axis=1)[:, k - 1]


def pm_group_features(y: np.ndarray, s: float):
    """One PM group's per-report features (PM_REPORT_NAMES) and context (PM_CONTEXT_NAMES), from the
    group's full set of reports; ties share their average rank."""
    n = len(y)
    W = s - 1.0
    order = np.argsort(y, kind='stable')
    ys = y[order]
    cols = [y / s, (s - np.abs(y)) / s, (rankdata(y) - 0.5) / max(n, 1)]
    dens = []
    for w in NBR_WIDTHS:
        h = w * W / 2.0
        cnt = np.searchsorted(ys, y + h, 'right') - np.searchsorted(ys, y - h, 'left') - 1
        lo, hi = np.maximum(y - h, -s), np.minimum(y + h, s)
        expected = max(n - 1, 1) * (hi - lo) / (2 * s)
        dens.append(cnt / np.maximum(expected, 1e-12))
    cols += dens
    spacing = 2 * s / max(n, 1)
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    for k in KNN:
        cols.append(_knn(ys, k)[inv] / spacing)
    ctx = np.r_[np.log(max(n, 1)), np.quantile(y / s, QUANTILES) if n else np.zeros(len(QUANTILES)),
                np.histogram(y / s, bins=HIST_BINS, range=(-1, 1))[0] / max(n, 1),
                dens[0].max() if n else 0.0, dens[0].mean() if n else 0.0, dens[0].std() if n else 0.0]
    return np.column_stack(cols), ctx


PM_REPORT_NAMES = (['y_over_s', 'boundary_dist', 'percentile']
                   + [f'nbr_density_{w:g}' for w in NBR_WIDTHS] + [f'knn_{k}' for k in KNN])
PM_CONTEXT_NAMES = (['log_group_size'] + [f'q{int(q * 100):02d}' for q in QUANTILES]
                    + [f'hist_{i}' for i in range(HIST_BINS)] + ['dens_max', 'dens_mean', 'dens_std'])


def pm_kernel_matrix(s: float, eps: float, n_bins: int = 128, n_atoms: int = 96):
    mech = PM(eps)
    edges = np.linspace(-s, s, n_bins + 1)
    atoms = np.linspace(-1, 1, n_atoms)
    l, r = mech.interval(atoms)
    a, b = edges[:-1, None], edges[1:, None]
    ov = np.clip(np.minimum(b, r[None]) - np.maximum(a, l[None]), 0, None)
    return edges, mech.pdf_high * ov + mech.pdf_low * ((b - a) - ov)


def pm_mixture_fit(y: np.ndarray, s: float, eps: float):
    """Weighted NNLS fit of the group's histogram by a mixture of PM output densities.
    Returns (edges, observed counts, fitted counts, weights); weights >= 0 and sum to 1."""
    edges, K = pm_kernel_matrix(s, eps)
    n = len(y)
    h = np.histogram(y, bins=edges)[0].astype(float)
    w = 1.0 / np.sqrt(np.maximum(h, 1.0))
    # sum-to-one row: without it the 1/h weighting fits low (sum ~0.86 on clean 1k groups)
    lam = 1e3 * np.sqrt(max(n, 1))
    A = np.vstack([w[:, None] * K * n, np.full(K.shape[1], lam)])
    pi, _ = nnls(A, np.r_[w * h, lam], maxiter=2000)
    pi /= pi.sum()
    return edges, h, n * K @ pi, pi


def _pm_mech_group(y: np.ndarray, s: float, eps: float):
    edges, h, fit, _ = pm_mixture_fit(y, s, eps)
    n = len(y)
    kb = np.clip(np.searchsorted(edges, y, side='right') - 1, 0, len(h) - 1)
    resid_ratio = np.log((h + 1.0) / (fit + 1.0))[kb]
    resid_share = ((h - fit) / max(n, 1))[kb]
    excess_total = float(np.clip(h - fit, 0, None).sum() / max(n, 1))
    return np.column_stack([resid_ratio, resid_share]), np.r_[excess_total]


PM_MECH_NAMES = ['fit_log_ratio', 'fit_excess_share']
PM_MECH_CONTEXT = ['fit_excess_total']


def report_features(obs, feature_set: str = 'base'):
    if obs['protocol'] == 'SR':
        return _sr_report_features(obs)
    return _pm_report_features(obs, feature_set)


def _pm_report_features(obs, feature_set):
    tf, mech = _setup(obs)
    y, g = np.asarray(obs['reports'], np.float64), np.asarray(obs['groups'])
    mean_n, var_n = standard_moments(obs)
    rep_names = list(PM_REPORT_NAMES) + (PM_MECH_NAMES if feature_set == 'mech' else [])
    ctx_names = (['group_decoded_mean'] + list(PM_CONTEXT_NAMES)
                 + (PM_MECH_CONTEXT if feature_set == 'mech' else []))
    d = 1 + len(rep_names) + len(ctx_names) + 2
    X = np.empty((len(y), d), dtype=np.float64)
    for grp in (G1, G2):
        idx = np.flatnonzero(g == grp)
        rf, ctx = pm_group_features(y[idx], mech.s)
        k, lo = (tf.k1, tf.a) if grp == G1 else (tf.k2, tf.q_lo)
        gmean = mech.decode(y[idx], k, lo).mean() if len(idx) else 0.0
        gmean_n = tf.norm_mean(gmean) if grp == G1 else (gmean - tf.q_lo) * tf.k2 - 1.0
        ctx = np.r_[gmean_n, ctx]
        if feature_set == 'mech':
            mf, mctx = _pm_mech_group(y[idx], mech.s, obs['epsilon'])
            rf, ctx = np.column_stack([rf, mf]), np.r_[ctx, mctx]
        X[idx] = np.column_stack([np.full(len(idx), float(grp == G1)), rf,
                                  np.tile(ctx, (len(idx), 1)), np.full((len(idx), 2), [mean_n, var_n])])
    names = ['is_g1'] + rep_names + ctx_names + ['standard_mean', 'standard_variance']
    return X.astype(np.float32), names


SR_NAMES = ['sign', 'is_g1', 'freq_g1_plus', 'freq_g1_minus', 'freq_g2_plus', 'freq_g2_minus',
            'log_n_g1', 'log_n_g2', 'standard_mean', 'standard_variance']


def sr_sign_frequencies(obs) -> np.ndarray:
    """Within-group sign shares (freq_g1_plus, freq_g1_minus, freq_g2_plus, freq_g2_minus)."""
    y, g = np.asarray(obs['reports'], np.float64), np.asarray(obs['groups'])
    n1, n2 = max(int(np.sum(g == G1)), 1), max(int(np.sum(g == G2)), 1)
    return np.array([np.sum((g == G1) & (y > 0)) / n1, np.sum((g == G1) & (y < 0)) / n1,
                     np.sum((g == G2) & (y > 0)) / n2, np.sum((g == G2) & (y < 0)) / n2])


def _sr_report_features(obs):
    y, g = np.asarray(obs['reports'], np.float64), np.asarray(obs['groups'])
    n1, n2 = max(int(np.sum(g == G1)), 1), max(int(np.sum(g == G2)), 1)
    f = sr_sign_frequencies(obs)
    mean_n, var_n = standard_moments(obs)
    row_ctx = np.r_[f, np.log(n1), np.log(n2), mean_n, var_n]
    X = np.column_stack([np.sign(y), (g == G1).astype(float), np.tile(row_ctx, (len(y), 1))])
    return X.astype(np.float32), list(SR_NAMES)


def round_summary(obs, n_bins: int = 32):
    tf, mech = _setup(obs)
    y, g = np.asarray(obs['reports'], np.float64), np.asarray(obs['groups'])
    mean_n, var_n = standard_moments(obs)
    vals, names = [mean_n, var_n], ['standard_mean', 'standard_variance']
    if obs['protocol'] == 'SR':
        X, nm = _sr_report_features(obs)
        vals += list(X[0, 2:8]) if len(X) else [0.0] * 6
        names += nm[2:8]
        return np.asarray(vals, np.float32), names
    for grp, tag in ((G1, 'g1'), (G2, 'g2')):
        yy = y[g == grp] / mech.s
        _, ctx = pm_group_features(y[g == grp], mech.s)
        hist = np.histogram(yy, bins=n_bins, range=(-1, 1))[0] / max(len(yy), 1)
        vals += list(ctx) + list(hist)
        names += [f'{tag}_{c}' for c in PM_CONTEXT_NAMES] + [f'{tag}_hist{n_bins}_{i}' for i in range(n_bins)]
    return np.asarray(vals, np.float32), names
