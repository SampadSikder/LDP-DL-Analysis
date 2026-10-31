"""Reference methods: standard aggregation, constant training moments, oracle removal, k-means.

k-means recovery (Li et al., Sec. 8.1), per group:
  draw 5,000 subsets of size max(1, round(r * n_g)) without replacement,
  decode each subset's mean, cluster the 5,000 means into two clusters
  (1-D k-means, init at the 25th / 75th percentiles, n_init = 1, 100 iterations),
  keep the cluster holding more subsets (tie: the tighter cluster), and use its
  centre as the group estimate. The sampling rate r is picked on validation
  rounds by utility; the generator is seeded identically for every round.
SR shortcut: a subset's +1 count is hypergeometric, which is exact and fast.
"""

import numpy as np

from ..estimator import G1, G2, Estimate, estimate, estimate_retained
from ..features import standard_moments
from ..mechanisms import SR, get_mechanism
from ..metrics import UtilityScales
from ..transforms import Transform
from .base import Method, MethodOutput

KMEANS_RATES = tuple(round(0.1 * i, 1) for i in range(1, 10))


class StandardMethod(Method):
    name, kind = 'standard', 'recovery'

    def predict(self, obs, labels=None):
        m, v = standard_moments(obs)
        return MethodOutput(m, v)


class ConstantMethod(Method):
    """Average genuine moments of the training rounds (training data only)."""
    name, kind = 'constant', 'recovery'

    def fit(self, train, val):
        self.mean_ = float(np.mean([r.labels['genuine_mean'] for r in train]))
        self.var_ = float(np.mean([r.labels['genuine_variance'] for r in train]))
        return self

    def predict(self, obs, labels=None):
        return MethodOutput(self.mean_, self.var_)

    def state(self):
        return {'name': self.name, 'mean': self.mean_, 'variance': self.var_}


class OracleRemoval(Method):
    """Standard aggregation of the actual genuine reports. Evaluation reference only."""
    name, kind, needs_labels = 'oracle_removal', 'reference', True

    def predict(self, obs, labels=None):
        tf = Transform(*obs['domain'], square_mode=obs['square_mode'])
        mech = get_mechanism(obs['protocol'], obs['epsilon'])
        keep = ~labels['is_fake']
        est, fb = estimate_retained(obs['reports'], obs['groups'], keep, mech, tf)
        e = est.normalized(tf)
        return MethodOutput(e.mean, e.variance, removed=~keep, fallback=fb)


def _kmeans_1d(v: np.ndarray, iters: int = 100):
    c = np.quantile(v, [0.25, 0.75]).astype(float)
    if c[0] == c[1]:
        return np.zeros(len(v), dtype=int), c
    for _ in range(iters):
        lab = (np.abs(v - c[1]) < np.abs(v - c[0])).astype(int)
        new = np.array([v[lab == j].mean() if np.any(lab == j) else c[j] for j in (0, 1)])
        if np.allclose(new, c):
            break
        c = new
    return lab, c


def kmeans_group_center(decoded: np.ndarray, rate: float, n_subsets: int, rng,
                        sr_values=None) -> float:
    n = len(decoded)
    k = max(1, int(round(rate * n)))
    if sr_values is not None:                      # decoded takes two values: exact hypergeometric
        hi, lo = sr_values
        n_hi = int(np.sum(decoded == hi))
        cnt = rng.hypergeometric(n_hi, n - n_hi, k, size=n_subsets)
        means = (cnt * hi + (k - cnt) * lo) / k
    else:
        means = np.empty(n_subsets)
        chunk = max(1, int(4e6 // max(n, 1)))
        for start in range(0, n_subsets, chunk):
            b = min(chunk, n_subsets - start)
            keys = rng.random((b, n), dtype=np.float32)
            idx = np.argpartition(keys, k - 1, axis=1)[:, :k]
            means[start:start + b] = decoded[idx].mean(axis=1)
    lab, c = _kmeans_1d(means)
    counts = np.bincount(lab, minlength=2)
    if counts[0] == counts[1]:
        spread = [means[lab == j].var() for j in (0, 1)]
        j = int(np.argmin(spread))
    else:
        j = int(np.argmax(counts))
    return float(c[j])


def _kmeans_task(args):
    """Worker: (method config, observation) -> (mean_raw, var_raw, runtime)."""
    import time
    rate, n_subsets, seed, obs = args
    t0 = time.perf_counter()
    out = KMeansRecovery(rate=rate, n_subsets=n_subsets, seed=seed, n_jobs=1)._predict(obs, rate)
    return out.mean_raw, out.var_raw, time.perf_counter() - t0


class KMeansRecovery(Method):
    """Rate selection uses at most `val_max` validation rounds (a fixed random subsample,
    seeded), and every prediction can run in a process pool (`n_jobs`)."""
    name, kind = 'kmeans', 'recovery'

    def __init__(self, rate: float = None, n_subsets: int = 5000, rates=KMEANS_RATES, seed: int = 0,
                 n_jobs: int = None, val_max: int = 300):
        import os
        self.rate, self.n_subsets, self.rates, self.seed = rate, n_subsets, rates, seed
        self.n_jobs = n_jobs if n_jobs is not None else max(1, min(12, (os.cpu_count() or 2) - 2))
        self.val_max = val_max

    def _run_many(self, observations, rate):
        if self.n_jobs <= 1 or len(observations) < 8:
            return [_kmeans_task((rate, self.n_subsets, self.seed, o)) for o in observations]
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.n_jobs, mp_context=mp.get_context('spawn')) as ex:
            return list(ex.map(_kmeans_task, [(rate, self.n_subsets, self.seed, o) for o in observations],
                               chunksize=8))

    def predict_many(self, observations, labels=None):
        outs = []
        for m, v, t in self._run_many(observations, self.rate):
            o = MethodOutput(m, v, info={'rate': self.rate})
            o.runtime = t
            outs.append(o)
        return outs

    def _predict(self, obs, rate):
        tf = Transform(*obs['domain'], square_mode=obs['square_mode'])
        mech = get_mechanism(obs['protocol'], obs['epsilon'])
        y, g = np.asarray(obs['reports'], float), np.asarray(obs['groups'])
        rng = np.random.default_rng(self.seed)
        centers = []
        for grp, (k, lo) in ((G1, (tf.k1, tf.a)), (G2, (tf.k2, tf.q_lo))):
            dec = mech.decode(y[g == grp], k, lo)
            sr = None
            if isinstance(mech, SR):
                sr = (float(mech.decode(1.0, k, lo)), float(mech.decode(-1.0, k, lo)))
            centers.append(kmeans_group_center(dec, rate, self.n_subsets, rng, sr))
        e = Estimate.from_moments(centers[0], tf.second_from_query(centers[1], centers[0])).normalized(tf)
        return MethodOutput(e.mean, e.variance, info={'rate': rate})

    def fit(self, train, val):
        if self.rate is not None or not val:
            self.rate = self.rate or 0.5
            return self
        scales = UtilityScales.from_rounds(train or val)
        if len(val) > self.val_max:
            pick = np.random.default_rng(self.seed).choice(len(val), self.val_max, replace=False)
            val = [val[i] for i in sorted(pick)]
        scores = {}
        for r in self.rates:
            res = self._run_many([v.observation for v in val], r)
            err = []
            for (m, var, _), v in zip(res, val):
                o = MethodOutput(m, var)
                err.append(scales.loss(o.mean - v.labels['genuine_mean'], o.var - v.labels['genuine_variance']))
            scores[r] = float(np.mean(err))
        self.rate = min(scores, key=scores.get)
        self.val_scores_ = scores
        self.val_rounds_used_ = len(val)
        return self

    def predict(self, obs, labels=None):
        return self._predict(obs, self.rate)

    def state(self):
        return {'name': self.name, 'rate': self.rate, 'n_subsets': self.n_subsets,
                'val_scores': getattr(self, 'val_scores_', None),
                'val_rounds_used': getattr(self, 'val_rounds_used_', None), 'val_max': self.val_max}
