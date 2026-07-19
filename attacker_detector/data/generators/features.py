import math
import numpy as np
from scipy.stats import chi2, rankdata

FEATURE_NAMES = [
    'num_ones',
    'chisq',
    'neg_log_p',
    'deviation',
    'deviation_percentile',
    'deviation_zscore',
    'mean_z_initial',
    'max_abs_z_initial',
    'positional_chisq_initial',
    'overlap_count_initial',
    'overlap_ratio_initial',
    'mean_z_robust',
    'max_abs_z_robust',
    'positional_chisq_robust',
    'overlap_count_robust',
    'overlap_ratio_robust',
    'entropy',
    'max_support',
]


def _protocol_params(protocol: str, epsilon: float, domain: int):
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


def _compute_positional_features(
    support_binary: np.ndarray,
    pi_hat_j: np.ndarray,
    p: float,
    q: float,
    n: int,
    domain: int,
):
    # Analytical expected per-item report probability
    E_bit_j = pi_hat_j * p + (1.0 - pi_hat_j) * q          
    Var_bit_j = E_bit_j * (1.0 - E_bit_j)                  
    std_bit_j = np.sqrt(np.maximum(Var_bit_j, 1e-12))       

    # z-score for each user × item: z_ij = (observed_bit - E_bit_j) / std
    z_matrix = (support_binary.astype(np.float64) - E_bit_j[np.newaxis, :]) / std_bit_j[np.newaxis, :]

    num_reported = support_binary.sum(axis=1).astype(np.float64) 
    num_reported_safe = np.maximum(num_reported, 1.0)

    # Aggregate over reported items only
    z_reported = z_matrix * support_binary  

    mean_z = z_reported.sum(axis=1) / num_reported_safe
    abs_z = np.abs(z_matrix)
    max_abs_z = abs_z.max(axis=1)
    positional_chisq = (z_reported ** 2).sum(axis=1)

    anomalous = (abs_z > 2.0).astype(np.float64) * support_binary
    overlap_count = anomalous.sum(axis=1)
    overlap_ratio = overlap_count / num_reported_safe

    positional_feats = np.column_stack([
        mean_z,
        max_abs_z,
        positional_chisq,
        overlap_count,
        overlap_ratio,
    ])

    return positional_feats, positional_chisq


def _estimate_pi_hat(support_binary: np.ndarray, p: float, q: float):
    observed_freq_j = support_binary.mean(axis=0)  
    denom = p - q
    if abs(denom) < 1e-12:
        return observed_freq_j  
    pi_hat_j = np.clip((observed_freq_j - q) / denom, 0.0, 1.0)
    return pi_hat_j


def extract_user_level_features(
    support_list: np.ndarray,
    one_list: np.ndarray,
    epsilon: float,
    protocol: str,
    domain: int,
    n: int,
    robust_iterations: int = 2,
) -> tuple:
    """
    Returns
    -------
    features : ndarray (n, 18)
        Feature matrix.
    metadata : dict
        Per-graph metadata: pi_hat, p, q, epsilon, protocol.
    """
    p, q, expected_ones = _protocol_params(protocol, epsilon, domain)

    # Binarize support for positional features
    if protocol in ('HST_User', 'HST_Server'):
        # HST: support_list = y_i * s_vectors, where s_vectors ∈ {-1, 1}.
        # one_list = sum(s_vectors == 1, axis=1). When y_i < 0 the sign
        # flips, so (support_list > 0) would give s_vectors == -1 instead.
        # Detect y sign per user and recover s_vectors == 1.
        num_positive = np.sum(support_list > 0, axis=1)
        y_positive = (num_positive == one_list)
        support_binary = np.where(
            y_positive[:, np.newaxis],
            (support_list > 0).astype(np.float64),
            (support_list < 0).astype(np.float64),
        )
    else:
        support_binary = support_list.astype(np.float64)

    E1 = max(expected_ones, 1e-10)
    E0 = max(domain - expected_ones, 1e-10)

    num_ones = one_list.astype(np.float64)

    # Chi-square statistic
    chisq = ((num_ones - E1) ** 2) * (1.0 / E1 + 1.0 / E0)

    # neg_log_p: use logsf directly to avoid underflow
    log_sf = chi2.logsf(chisq, df=1)
    neg_log_p = -log_sf
    # Cap at 99.9th percentile of finite values (instead of arbitrary 1000)
    finite_mask = np.isfinite(neg_log_p)
    if finite_mask.any():
        cap = np.percentile(neg_log_p[finite_mask], 99.9)
        neg_log_p = np.where(finite_mask, np.minimum(neg_log_p, cap), cap)
    else:
        neg_log_p = np.zeros_like(neg_log_p)

    # Deviation
    deviation = np.abs(num_ones - E1)

    ranks = rankdata(deviation, method='average')
    deviation_percentile = (ranks - 1.0) / max(n - 1.0, 1.0)

    dev_mean = np.mean(deviation)
    dev_std = np.std(deviation)
    deviation_zscore = (deviation - dev_mean) / max(dev_std, 1e-10)

    pi_hat_initial = _estimate_pi_hat(support_binary, p, q)
    initial_pos_feats, initial_user_chisq = _compute_positional_features(
        support_binary, pi_hat_initial, p, q, n, domain
    )

    if robust_iterations > 0:
        pi_hat_robust = pi_hat_initial.copy()
        user_weights = np.ones(n, dtype=np.float64)

        for _iter in range(robust_iterations):
            threshold = np.percentile(initial_user_chisq, 90)
            user_weights = (initial_user_chisq <= threshold).astype(np.float64)

            if user_weights.sum() > 0:
                weighted_freq = (support_binary * user_weights[:, np.newaxis]).sum(axis=0) / user_weights.sum()
                denom = p - q
                if abs(denom) >= 1e-12:
                    pi_hat_robust = np.clip((weighted_freq - q) / denom, 0.0, 1.0)

            robust_pos_feats, initial_user_chisq = _compute_positional_features(
                support_binary, pi_hat_robust, p, q, n, domain
            )

        pi_hat_final = pi_hat_robust
    else:
        robust_pos_feats = initial_pos_feats.copy()
        pi_hat_final = pi_hat_initial


    support_probs = support_binary / np.maximum(support_binary.sum(axis=1, keepdims=True), 1e-10)
    log_probs = np.where(support_probs > 0, np.log(support_probs + 1e-10), 0.0)
    entropy = -np.sum(support_probs * log_probs, axis=1)

    max_support = np.max(support_list, axis=1)

    features = np.column_stack([
        num_ones,                     # 1
        chisq,                        # 2
        neg_log_p,                    # 3
        deviation,                    # 4
        deviation_percentile,         # 5
        deviation_zscore,             # 6
        initial_pos_feats,            # 7-11  (5 cols)
        robust_pos_feats,             # 12-16 (5 cols)
        entropy,                      # 17
        max_support,                  # 18
    ])

    metadata = {
        'pi_hat': pi_hat_final,
        'p': p,
        'q': q,
        'epsilon': epsilon,
        'protocol': protocol,
    }

    return features, metadata
