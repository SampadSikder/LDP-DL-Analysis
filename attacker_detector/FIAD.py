"""FIAD fake-user detection (frequent itemset mining), ported from MDPA_LDP.

Source: FIAD_detect.py of https://github.com/Marvin-huoshan/MDPA_LDP (Li et al.,
"Mitigating Data Poisoning Attacks to Local Differential Privacy", CCS 2025).

FIAD mines itemsets of 4-12 items that co-occur in users' reports more often
than independent perturbation allows (a per-size threshold tau_z, set so the
false-positive rate is at most eta), and flags every user whose report supports
one of those itemsets.

The detection follows upstream step for step, quirks included (marked
"upstream:"). Rewritten only where the flagged users cannot change: tau_z is
found by one vectorized search per itemset size instead of a Python loop per
itemset, itemsets repeated across chunks are looked up once, and upstream's
joblib.Parallel(n_jobs=1) is a loop (or, with n_jobs > 1, a process pool over
chunks). The cleaned frequency estimate and the printed metrics are left out --
fiad_detect returns the flagged users and the caller scores them against the
labels.
"""

import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import betainc

OLH_PROTOCOLS = ('OLH', 'OLH_User', 'OLH_Server')
HST_PROTOCOLS = ('HST_User', 'HST_Server')

CHUNK_SIZE = 20000   # upstream: users per fpgrowth call
MIN_ITEMSET, MAX_ITEMSET = 4, 12


def _mine_chunk(chunk, min_support, max_len):
    from mlxtend.frequent_patterns import fpgrowth
    return fpgrowth(pd.DataFrame(chunk), min_support=min_support,
                    use_colnames=True, max_len=max_len)


def _frequent_itemsets(support, min_support, max_len, chunk_size, n_jobs=1):
    """fpgrowth on each chunk of users, n_jobs chunks at a time.

    upstream: an itemset's support is its frequency within the chunk it was
    mined from, yet it is compared with tau_z / n for the whole population, and
    the same itemset can come back from several chunks. Chunks are mined
    independently and concatenated in order, so n_jobs does not change the result.
    """
    chunks = [(support[start:start + chunk_size], min_support, max_len)
              for start in range(0, support.shape[0], chunk_size)]
    if n_jobs > 1 and len(chunks) > 1:
        with Pool(processes=min(n_jobs, len(chunks))) as pool:
            frames = pool.starmap(_mine_chunk, chunks)
    else:
        frames = [_mine_chunk(*chunk) for chunk in chunks]
    itemsets = pd.concat(frames, ignore_index=True)
    sizes = itemsets['itemsets'].apply(len)
    return itemsets[(sizes >= MIN_ITEMSET) & (sizes <= MAX_ITEMSET)]


def _tau_oue(z, n, p, q, eta):
    """OUE: the first tau above n p q^(z-1) whose Chebyshev false-positive
    bound is <= eta (upstream's linear search over tau, vectorized)."""
    tau_z_min = n * p * (q ** (z - 1))
    taus = np.arange(int(tau_z_min) + 1, n + 1)
    fpr_bound = (n * p * q ** (z - 1) * (1 - p * q ** (z - 1))) / ((taus - n * p * q ** (z - 1)) ** 2)
    hits = np.flatnonzero(fpr_bound <= eta)
    return int(taus[hits[0]]) if len(hits) else None


def _tau_betainc(z, n, q, eta):
    """OLH / HST: the smallest tau with I_{q^(z-1)}(tau, n - tau + 1) <= eta,
    i.e. P(Binomial(n, q^(z-1)) >= tau) <= eta (upstream's calculate_tau_z,
    vectorized)."""
    taus = np.arange(1, n + 1)
    hits = np.flatnonzero(betainc(taus, n - taus + 1, q ** (z - 1)) <= eta)
    return int(taus[hits[0]]) if len(hits) else None


def fiad_detect(support, epsilon, protocol, eta=0.1, chunk_size=CHUNK_SIZE, n_jobs=1):
    """Users FIAD flags as fake in one LDP run.

    :param support: (n, d) support matrix; entries > 0 count as support (HST
        reports are binarized this way, as upstream does)
    :param protocol: OUE, OLH_User / OLH_Server, HST_User / HST_Server
    :param eta: false-positive bound per itemset (upstream default 0.1)
    :param n_jobs: processes mining chunks in parallel (upstream: 1)
    :return: sorted indices of the flagged users
    """
    support = support if support.dtype == bool else support > 0
    n = support.shape[0]

    if protocol == 'OUE':
        p = 1 / 2
        q = 1 / (math.exp(epsilon) + 1)
        max_len = 12
        tau_z = lambda z: _tau_oue(z, n, p, q, eta)
    elif protocol in OLH_PROTOCOLS:
        g = int(round(math.exp(epsilon))) + 1
        q = 1 / g
        max_len = 12
        tau_z = lambda z: _tau_betainc(z, n, q, eta)
    elif protocol in HST_PROTOCOLS:
        q = 1 / 2
        max_len = 10   # upstream: process_chunk_HST
        tau_z = lambda z: _tau_betainc(z, n, q, eta)
    else:
        raise ValueError(f"FIAD: unknown protocol {protocol!r}")

    # upstream: min_support is q, the probability a genuine report supports
    # an item it does not hold
    itemsets = _frequent_itemsets(support, q, max_len, chunk_size, n_jobs)

    taus = {}
    abnormal = set()
    for itemset, itemset_support in zip(itemsets['itemsets'], itemsets['support']):
        z = len(itemset)
        if z not in taus:
            taus[z] = tau_z(z)
        if taus[z] is not None and itemset_support >= taus[z] / n:
            abnormal.add(itemset)

    flagged = np.zeros(n, dtype=bool)
    for itemset in abnormal:
        flagged |= support[:, sorted(itemset)].all(axis=1)
    return np.flatnonzero(flagged)
