"""Go/no-go probe for server-side APA on OLH-Server.

One LDP run (defaults: zipf, eps=1, beta=0.05, r=10, r'=4, n=50k). Every fake
user's buckets and r'-subset are fixed once; three bucket rules are then
compared on exactly those choices:

  mga-a   bucket covering the most of the user's targets (current generator)
  user    S-APA, per-user strict count matching (choose_server_apa_buckets 'user')
  hist    S-APA, population-level quota (choose_server_apa_buckets 'hist')

Per rule:
  count leak    count-only AUC (max(AUC, 1-AUC)) and KS distance to genuine counts
  quota L1      share of fake users outside the APA quota (0 = histogram exact)
  attack gain   targets supported per fake user (genuine level = r/g) and the
                summed OLH frequency gain on the r targets
  detectability v2 features + gradient boosting, 5-fold CV: AUC and best F1

Go if 'hist' (or 'user') brings the count leak to ~0.5 and keeps most of
MGA-A's gain.

    python scripts/probe_server_apa.py --processes 64
"""

import argparse
import math
import os
import sys
import time

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Pool
from functools import partial

from attacker_detector.data.generators.attacks import (
    _olh_server_buckets, apa_quota, choose_server_apa_buckets)
from attacker_detector.data.generators.features import (
    extract_user_level_features_diffstats_style)
from attacker_detector.data.generators.pca_graph import get_distribution_generator
from attacker_detector.data.generators.protocols import construct_omega


def hash_all(n, g, domain, processes):
    with Pool(processes=processes) as pool:
        rows = pool.map(partial(_olh_server_buckets, g=g, domain=domain),
                        [(i, i) for i in range(n)], chunksize=256)
    buckets = np.empty((n, domain), dtype=np.uint8)
    for i, row in rows:
        buckets[i] = row
    return buckets


def best_f1(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    return float((2 * pr * rc / np.maximum(pr + rc, 1e-12)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='zipf')
    ap.add_argument('--domain', type=int, default=1024)
    ap.add_argument('--n', type=int, default=50_000)
    ap.add_argument('--epsilon', type=float, default=1.0)
    ap.add_argument('--ratio', type=float, default=0.05)
    ap.add_argument('--targets', type=int, default=10)
    ap.add_argument('--splits', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--processes', type=int, default=os.cpu_count())
    args = ap.parse_args()

    n, d, eps = args.n, args.domain, args.epsilon
    np.random.seed(args.seed)
    get_distribution_generator(args.dataset)(n, d, seed=args.seed)  # same RNG path as the generator
    targets = np.sort(np.random.choice(d, size=args.targets, replace=False))
    g = int(round(math.exp(eps))) + 1
    p = math.exp(eps) / (math.exp(eps) + g - 1)
    m = int(round(n * args.ratio))
    start = n - m

    t0 = time.time()
    buckets = hash_all(n, g, d, args.processes)
    print(f"hashed {n} users x {d} items in {time.time() - t0:.0f}s (g={g}, m={m})")

    # Genuine reports: the generator's benign users report bucket 0 (Y = 0).
    genuine = (buckets == 0)
    fake_b = buckets[start:]
    subsets = np.array([np.random.choice(targets, args.splits, replace=False)
                        for _ in range(m)])
    sizes = np.stack([(fake_b == b).sum(1) for b in range(g)], axis=1)
    tb = np.take_along_axis(fake_b, subsets, axis=1)
    coverage = np.stack([(tb == b).sum(1) for b in range(g)], axis=1)
    omega = construct_omega(eps, d, 'OLH_Server')
    quota = apa_quota(m, omega)

    rules = {
        'mga-a': np.argmax(coverage + 1e-3 * np.random.rand(m, g), axis=1),
        'user': choose_server_apa_buckets(sizes, coverage, omega, 'user'),
        'hist': choose_server_apa_buckets(sizes, coverage, omega, 'hist'),
    }

    a = g / (p * g - 1)
    clean_est = a * genuine[:, targets].sum(0) - n / (p * g - 1)
    y = np.r_[np.zeros(start, int), np.ones(m, int)]
    rows = []
    for name, chosen in rules.items():
        support = genuine.copy()
        support[start:] = fake_b == chosen[:, None].astype(np.uint8)
        counts = support.sum(1)
        fake_k, gen_k = counts[start:], counts[:start]

        auc_k = roc_auc_score(y, counts)
        hist = np.bincount(fake_k, minlength=len(quota))[:len(quota)]
        quota_l1 = 0.5 * np.abs(hist - quota).sum() / m + (fake_k >= len(quota)).sum() / m

        est = a * support[:, targets].sum(0) - n / (p * g - 1)
        gain = float((est - clean_est).sum() / n)

        X = extract_user_level_features_diffstats_style(
            support.astype(np.float64), counts, eps, 'OLH_Server', d, n, feature_set='v2')
        prob = cross_val_predict(
            HistGradientBoostingClassifier(max_iter=300, random_state=0), X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]

        rows.append({
            'rule': name,
            'count_AUC': max(auc_k, 1 - auc_k),
            'count_KS': ks_2samp(fake_k, gen_k).statistic,
            'quota_L1': quota_l1,
            'own_targets_covered': float(np.take_along_axis(coverage, chosen[:, None], 1).mean()),
            'targets_supported': float(support[start:, targets].sum(1).mean()),
            'freq_gain_sum': gain,
            'v2_AUC': roc_auc_score(y, prob),
            'v2_best_F1': best_f1(y, prob),
        })
        print(f"  {name}: done")

    print(f"\ngenuine users support {genuine[:start][:, targets].sum(1).mean():.2f} "
          f"of the {args.targets} targets on average (r/g = {args.targets / g:.2f})")
    keys = list(rows[0])
    print("  ".join(f"{k:>19}" for k in keys))
    for r in rows:
        print("  ".join(f"{r[k]:>19}" if isinstance(r[k], str) else f"{r[k]:>19.4f}" for k in keys))


if __name__ == '__main__':
    main()
