"""DiffStats-style feature extraction for user-level attacker detection."""

import math
import numpy as np
from scipy.stats import binom, wasserstein_distance
from scipy.spatial.distance import jensenshannon

# Every feature is dimensionless and domain-invariant by construction: each
# scale-dependent quantity is divided by its analytically known scale, derived
# from (d, p, q) alone. Nothing here is fitted from data or from a simulated
# "ideal" run, so the same feature means the same thing at any domain size,
# including one never seen in training, and is computable from only what a
# real LDP server observes (the attacked reports) plus public protocol
# parameters (epsilon, protocol, domain) -- never the true un-perturbed
# distribution, which a deployed server never has access to.
FEATURE_NAMES = [
    'num_ones_scaled',
    'one_deviation',
    'k_discrepancy_scaled',
    'k_observed_frequency_scaled',
    'k_theoretical_frequency_scaled',
    'freq_ratio',
    'is_anomalous_k',
    'overlap_anomalous_items_ratio',
    'max_item_freq_ratio',
    'mean_item_freq_ratio',
    'user_theoretical_deviation',
    'support_entropy_scaled',
    'max_support_value',
    'log_likelihood',
    'wasserstein_distance_scaled',
    'js_divergence_k',
]

# v2 drops the four k-features that are fixed functions of the user's
# standardized k-deviation -- given num_ones_scaled and one_deviation they add
# nothing -- and adds two target-block features (see _target_block_features).
# The k-features built from the run's *observed* k-histogram are kept: unlike
# the analytic ones they carry run-level context.
_V2_DROPPED = {
    'k_theoretical_frequency_scaled',
    'log_likelihood',
    'wasserstein_distance_scaled',
    'js_divergence_k',
}
FEATURE_NAMES_V2 = [f for f in FEATURE_NAMES if f not in _V2_DROPPED] + [
    'block_projection',
    'block_overlap',
]

FEATURE_SETS = {'v1': FEATURE_NAMES, 'v2': FEATURE_NAMES_V2}
# Chosen by scripts/sweep_feature_sets.py (results/sweep_v2): 16 gave the best
# single-feature AUC on OUE, HST_User and OLH across zipf/emoji; 32 and 64 were
# within noise on the downstream classifier, and fire held up at every M.
DEFAULT_BLOCK_CANDIDATES = 16


def feature_names_for(feature_set: str) -> list:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; "
                         f"expected one of {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set]


ANOMALY_THRESHOLD = 1.5
# Fraction of distinct k-values flagged as discrepant. A fixed *count* would flag
# a shrinking share as the domain grows, since the number of distinct k scales
# with sigma_k; a fraction is invariant by construction.
TOP_DISCREPANT_FRACTION = 0.10


def _protocol_params(protocol: str, epsilon: float, domain: int):
    """(p, q, expected_ones) used for num_ones_scaled/one_deviation and for
    compute_pi_hat's LDP debiasing inversion, which needs p != q."""
    if protocol == 'OUE':
        p = 0.5
        q = 1.0 / (math.exp(epsilon) + 1.0)
        expected_ones = p + (domain - 1) * q
    elif protocol in ('OLH', 'OLH_User', 'OLH_Server'):
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1.0 / g
        expected_ones = p + (domain - 1) * q
    elif protocol in ('HST_User', 'HST_Server'):
        p = math.exp(epsilon) / (math.exp(epsilon) + 1.0)
        q = 1.0 / (math.exp(epsilon) + 1.0)
        expected_ones = domain / 2.0
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    return p, q, expected_ones


def _protocol_pq(protocol: str, epsilon: float, domain: int):
    if protocol == 'OUE':
        p = 0.5
        q = 1.0 / (math.exp(epsilon) + 1.0)
    elif protocol in ('OLH', 'OLH_User', 'OLH_Server'):
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1.0 / g
    elif protocol in ('HST', 'HST_User', 'HST_Server'):
        p = q = 0.5
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    expected_ones = p + (domain - 1) * q
    p_binomial = expected_ones / domain
    return p, q, expected_ones, p_binomial


def _estimate_pi_hat(support_binary: np.ndarray, p: float, q: float):
    observed_freq_j = support_binary.mean(axis=0)
    denom = p - q
    if abs(denom) < 1e-12:
        return observed_freq_j
    pi_hat_j = np.clip((observed_freq_j - q) / denom, 0.0, 1.0)
    return pi_hat_j


def compute_pi_hat(support_list: np.ndarray, protocol: str, epsilon: float, domain: int):
    """Server-side reconstructed item distribution, kept for graph metadata."""
    p, q, _ = _protocol_params(protocol, epsilon, domain)
    support_binary = (support_list > 0).astype(np.float64)
    return _estimate_pi_hat(support_binary, p, q), p, q


def _k_lookup_table(one_list: np.ndarray, domain: int, p_binomial: float,
                    with_analytic: bool = True):
    """
    Per-unique-k features, as {name: array over distinct k} plus the index that
    maps each user to their k. Every k-dependent feature is a function of the
    user's k alone, so each is computed once per distinct k.

    The four analytic columns (k_theoretical_frequency_scaled, log_likelihood,
    wasserstein_distance_scaled, js_divergence_k) depend on k only through the
    analytic null, so given num_ones_scaled / one_deviation they add nothing.
    Only the v1 feature set uses them; with_analytic=False skips them, including
    the per-k Wasserstein / JS loop that dominates this function's cost.
    """
    k_values, inverse, k_counts = np.unique(
        one_list.astype(int), return_inverse=True, return_counts=True
    )
    sigma_k = math.sqrt(max(domain * p_binomial * (1.0 - p_binomial), 1e-12))

    observed_freq = k_counts / len(one_list)
    theoretical_freq = binom.pmf(k_values, domain, p_binomial)
    k_discrepancies = np.abs(observed_freq - theoretical_freq)

    threshold = np.percentile(k_discrepancies, 100.0 * (1.0 - TOP_DISCREPANT_FRACTION))
    is_anomalous_k = (k_discrepancies >= threshold).astype(np.float64)

    # Ratio of two same-unit masses: already dimensionless.
    freq_ratio = observed_freq / (theoretical_freq + 1e-10)

    table = {
        'k_discrepancy_scaled':        k_discrepancies * sigma_k,
        'k_observed_frequency_scaled': observed_freq * sigma_k,
        'freq_ratio':                  freq_ratio,
        'is_anomalous_k':              is_anomalous_k,
    }
    if not with_analytic:
        return table, inverse, sigma_k

    theoretical_freq_norm = theoretical_freq / (np.sum(theoretical_freq) + 1e-10)
    n_k = len(k_values)
    wasserstein = np.empty(n_k, dtype=np.float64)
    js_divergence = np.empty(n_k, dtype=np.float64)
    for idx in range(n_k):
        one_hot = np.zeros(n_k)
        one_hot[idx] = 1.0
        wasserstein[idx] = wasserstein_distance(
            k_values, k_values,
            u_weights=one_hot,
            v_weights=theoretical_freq_norm,
        )
        js_divergence[idx] = jensenshannon(one_hot, theoretical_freq_norm)

    table.update({
        'k_theoretical_frequency_scaled': theoretical_freq * sigma_k,
        'log_likelihood':                 np.log(theoretical_freq * sigma_k + 1e-10),
        'wasserstein_distance_scaled':    wasserstein / sigma_k,  # distance to a mass of width sigma_k, not d
        'js_divergence_k':                np.nan_to_num(js_divergence, nan=0.0),
    })
    return table, inverse, sigma_k


def _target_block_features(reported_f: np.ndarray, p_binomial: float, n_candidates: int):
    """Per-user score against the run's most coordinated block of items.

    Every attacker in a run promotes a subset of the same target set, so the
    targets are reported together far more often than chance. Benign users do
    not create such a block: each holds a single true item, so even genuinely
    popular items co-occur only at their marginal-product rate.

      1. z-score each item's report count against the analytic binomial null;
      2. keep the top n_candidates items with z > 0 -- the targets plus any
         genuinely popular items;
      3. take the leading eigenvector v of the candidates' off-diagonal
         covariance, which concentrates on the jointly-reported block;
      4. score each user against that block.

    Uses only the reports and public protocol parameters, never labels or the
    true item distribution.

    Returns (block_projection, block_overlap), both dimensionless:
      block_projection  the user's centered candidate reports projected onto v,
                        divided by that projection's std within the run
      block_overlap     share of the block's positive loading mass the user reports
    """
    n = reported_f.shape[0]
    zeros = np.zeros(n, dtype=np.float64)

    counts = reported_f.sum(axis=0)
    null_sd = math.sqrt(max(n * p_binomial * (1.0 - p_binomial), 1e-12))
    z = (counts - n * p_binomial) / null_sd

    order = np.argsort(-z)[:n_candidates]
    cand = order[z[order] > 0]
    if len(cand) < 2:
        return zeros, zeros

    Xs = reported_f[:, cand]
    Xc = Xs - Xs.mean(axis=0)
    cov = Xc.T @ Xc / n
    np.fill_diagonal(cov, 0.0)  # variance says nothing about co-occurrence
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    if v.sum() < 0:
        v = -v

    proj = Xc @ v
    proj_sd = proj.std()
    block_projection = proj / proj_sd if proj_sd > 1e-12 else zeros

    v_pos = np.clip(v, 0.0, None)
    mass = v_pos.sum()
    block_overlap = Xs @ v_pos / mass if mass > 1e-12 else zeros
    return block_projection, block_overlap


def extract_user_level_features_diffstats_style(
    support_list: np.ndarray,
    one_list: np.ndarray,
    epsilon: float,
    protocol: str,
    domain: int,
    n: int,
    feature_set: str = 'v1',
    block_candidates: int = DEFAULT_BLOCK_CANDIDATES,
) -> np.ndarray:
    """
    Extract features following DiffStats methodology.

    Every baseline here is purely analytic -- derived from (epsilon, protocol,
    domain) alone via the same noise model construct_omega uses -- so nothing
    requires a simulated "ideal" reference or the true un-perturbed
    distribution, both of which a deployed server never has access to. The
    per-user framing is what makes a naive uniform item prior discriminative
    despite carrying no true item-popularity signal: all attackers in a graph
    concentrate on the same small target_set, so their individual deviations
    from the shared baseline are correlated, while a benign user's deviation
    reflects only their own independently-drawn item.

    Args:
        support_list: Attacked support matrix (n, domain)
        one_list: Number of 1s per user (attacked)
        epsilon: Privacy parameter
        protocol: Protocol label, e.g. 'OUE', 'OLH_Server', 'HST_User'
        domain: Domain size
        n: Number of users
        feature_set: 'v1' (16 columns, FEATURE_NAMES) or 'v2' (14 columns,
            FEATURE_NAMES_V2)
        block_candidates: candidate items for the v2 target-block features

    Returns:
        Feature matrix (n, len(feature_names_for(feature_set))), columns in
        that order.
    """
    names = feature_names_for(feature_set)
    _, _, expected_ones = _protocol_params(protocol, epsilon, domain)
    _, _, _, p_binomial = _protocol_pq(protocol, epsilon, domain)

    support = np.asarray(support_list, dtype=np.float64)
    one_list = np.asarray(one_list, dtype=np.float64)

    k_table, k_index, sigma_k = _k_lookup_table(
        one_list, domain, p_binomial, with_analytic=(feature_set == 'v1'))
    per_user_k = {name: values[k_index] for name, values in k_table.items()}

    one_deviation = np.abs(one_list - expected_ones) / sigma_k

    # Every item is equally likely to be reported under complete ignorance of
    # true item popularity -- the only baseline computable without the true
    # (never-observed-at-inference) distribution. See module docstring.
    item_counts = support.sum(axis=0)
    expected_item_counts = n * p_binomial
    item_frequency_ratio = item_counts / (expected_item_counts + 1e-10)
    anomalous_items = item_frequency_ratio > ANOMALY_THRESHOLD

    reported = support > 0
    reported_f = reported.astype(np.float64)
    num_reported = reported.sum(axis=1).astype(np.float64)
    has_any = num_reported > 0

    overlap_count = (reported & anomalous_items).sum(axis=1).astype(np.float64)
    overlap_ratio = overlap_count / (num_reported + 1e-10)

    max_item_freq_ratio = np.where(
        has_any,
        np.where(reported, item_frequency_ratio, -np.inf).max(axis=1),
        0.0,
    )

    denom = np.where(has_any, num_reported, 1.0)
    ratio_sum = (reported * item_frequency_ratio).sum(axis=1)
    mean_item_freq_ratio = np.where(has_any, ratio_sum / denom, 0.0)

    # Binarized: support_list is 0/1 for OUE/OLH but real-valued (+-c-scaled)
    # for HST, so compare the reported-or-not indicator, not the raw values.
    user_theoretical_deviation = np.mean(np.abs(reported_f - p_binomial), axis=1)

    support_probs = support / (one_list[:, np.newaxis] + 1e-10)
    positive = support_probs > 0
    safe_probs = np.where(positive, support_probs, 1.0)
    support_entropy = -(
        np.where(positive, safe_probs * np.log(safe_probs), 0.0)
    ).sum(axis=1)
    support_entropy_scaled = support_entropy / math.log(domain)

    max_support_value = support.max(axis=1)

    columns = {
        'num_ones_scaled':                one_list / expected_ones,
        'one_deviation':                  one_deviation,
        **per_user_k,
        'overlap_anomalous_items_ratio':  overlap_ratio,
        'max_item_freq_ratio':            max_item_freq_ratio,
        'mean_item_freq_ratio':           mean_item_freq_ratio,
        'user_theoretical_deviation':     user_theoretical_deviation,
        'support_entropy_scaled':         support_entropy_scaled,
        'max_support_value':              max_support_value,
    }
    if feature_set == 'v2':
        columns['block_projection'], columns['block_overlap'] = \
            _target_block_features(reported_f, p_binomial, block_candidates)

    return np.column_stack([columns[name] for name in names])
