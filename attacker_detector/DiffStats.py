"""Diffstats fake-user detection, ported from MDPA_LDP.

Source: Diffstats.py of https://github.com/Marvin-huoshan/MDPA_LDP (Li et al.,
"Mitigating Data Poisoning Attacks to Local Differential Privacy", CCS 2025).

Diffstats compares the histogram of how many items each user supports with the
binomial count genuine users would produce, then removes the users behind the
largest discrepancies who also support the most-supported items, keeping the
removal whose cleaned histogram fits the binomial best (chi-square + p-value).

The selection follows upstream step for step, quirks included (marked
"upstream:"). Rewritten only where the flagged users cannot change: the
DataFrame copies / drops are boolean masks and the per-user removal loop is
vectorized. The plots, the KS / kurtosis / skewness bookkeeping, the cleaned
frequency estimate and the printed metrics are left out -- diffstats_detect
returns the flagged users and the caller scores them against the labels.
"""

import itertools
import math

import numpy as np
from scipy.stats import binom, chisquare

OLH_PROTOCOLS = ('OLH', 'OLH_User', 'OLH_Server')
HST_PROTOCOLS = ('HST_User', 'HST_Server')

TOP_DISCREPANCIES = 10   # thresholds tried: the 10 largest k-discrepancies
TOP_ITEMS = 6            # candidate items: the 6 most supported by anomalous users


def amend(d, e, p, q, n, k_size):
    """Upstream's correction added to the anomalous k-bins before the
    chi-square test (verbatim)."""
    p_prime = (p + (d - 1) * q) / d
    lower_bound = int(d * p_prime - math.floor(k_size / 2))
    upper_bound = int(d * p_prime + math.floor(k_size / 2))
    prob_interval = binom.cdf(upper_bound, d, p_prime) - binom.cdf(lower_bound - 1, d, p_prime)
    expected_occurrences = n * prob_interval * q
    return expected_occurrences / k_size


def _column_sums(support, rows, block=65536):
    """support[rows].sum(axis=0), without materializing all of support[rows]."""
    total = np.zeros(support.shape[1], dtype=np.int64)
    for start in range(0, len(rows), block):
        total += support[rows[start:start + block]].sum(axis=0)
    return total


def diffstats_detect(support, one_list, epsilon, domain, protocol):
    """Users Diffstats flags as fake in one LDP run.

    :param support: (n, d) support matrix; entries > 0 count as support
    :param one_list: (n,) number of items each user's report supports, as the
        generator returns it
    :param protocol: OUE, OLH_User / OLH_Server, HST_User / HST_Server
    :return: sorted indices of the flagged users
    """
    e, d = epsilon, domain
    if protocol == 'OUE':
        p = 1 / 2
        q = 1 / (math.exp(e) + 1)
        p_binomial = (1 / d) * (p + (d - 1) * q)
    elif protocol in OLH_PROTOCOLS:
        g = int(round(math.exp(e))) + 1
        p = math.exp(e) / (math.exp(e) + g - 1)
        q = 1 / g
        p_binomial = (1 / d) * (p + (d - 1) * q)
    elif protocol in HST_PROTOCOLS:
        p_binomial = 1 / 2
        p = q = 1 / 2
    else:
        raise ValueError(f"Diffstats: unknown protocol {protocol!r}")

    # upstream: `if perturb_method == 'HST_Server' or 'HST_User'` is always
    # true, so every protocol's support is binarized (OUE / OLH already are).
    support = support if support.dtype == bool else support > 0
    ones = np.asarray(one_list)
    n = len(ones)

    values, kpos, counts = np.unique(ones, return_inverse=True, return_counts=True)
    values = values.astype(int)
    kpos = kpos.ravel()
    observed_freq = counts / n
    k_values = values

    theoretical_pdf = binom.pmf(k_values, d, p_binomial)
    theoretical_freq = theoretical_pdf * d
    # upstream: the binomial frequency is scaled by d / n, not compared as is
    difference = np.abs(observed_freq - theoretical_freq / n)

    top_n_diff = sorted(difference, reverse=True)[:TOP_DISCREPANCIES]
    top_n_diff.reverse()
    pmf_k = [binom.pmf(k, d, p_binomial) for k in k_values]

    best_metric = float('inf')
    best_removed = np.array([], dtype=np.int64)
    for threshold in top_n_diff:
        anomalous = difference > threshold
        anomalies_index = np.flatnonzero(anomalous[kpos])

        support_sum = _column_sums(support, anomalies_index)
        if len(support_sum) <= TOP_ITEMS:
            valid_support_indices = list(np.argsort(support_sum)[::-1])
        else:
            valid_support_indices = list(np.argsort(support_sum)[-TOP_ITEMS:][::-1])
        # upstream's branch for an empty combination list is unreachable: d >= 1
        # always leaves at least one candidate item.
        all_combinations = []
        for r in range(1, len(valid_support_indices) + 1):
            all_combinations.extend(itertools.combinations(valid_support_indices, r))

        column = {item: j for j, item in enumerate(valid_support_indices)}
        candidate_support = support[np.ix_(anomalies_index, valid_support_indices)]
        index_1 = np.where(anomalous)

        for combination in all_combinations:
            hit = candidate_support[:, [column[item] for item in combination]].any(axis=1)
            to_remove = anomalies_index[hit]
            n_clean = n - len(to_remove)

            # upstream: np.unique over the cleaned users. Aligned to k_values
            # (0 where every user of a k was removed) this is the same array,
            # except when a whole bin empties -- there upstream's chisquare
            # raises on the length mismatch.
            observed_count = counts - np.bincount(kpos[to_remove], minlength=len(values))
            # upstream: len() of the np.where tuple, so k_size is always 1
            amends = amend(d, e, p, q, n_clean, len(index_1))
            for ix in index_1:
                observed_count[ix] += int(amends)
            expected_counts = [n_clean * pmf for pmf in pmf_k]
            observed_total = sum(observed_count)
            expected_total = sum(expected_counts)
            adjusted_expected_counts = [count * observed_total / expected_total
                                        for count in expected_counts]
            with np.errstate(divide='ignore', invalid='ignore'):
                chi2_stat, p_value = chisquare(observed_count, f_exp=adjusted_expected_counts)
            combine_metric = chi2_stat + p_value

            if combine_metric < best_metric:
                best_metric = combine_metric
                best_removed = to_remove

    # upstream: if no combination ever scores below inf (every chi-square is
    # inf / nan) the bookkeeping raises NameError; here nobody is flagged.
    return np.sort(best_removed)
