"""Group split, user reports and the server's mean / variance estimate.
"""

from dataclasses import dataclass

import numpy as np

from .mechanisms import Mechanism
from .transforms import Transform

G1, G2 = 1, 2


def split_groups(n_users: int, rng: np.random.Generator) -> np.ndarray:
    groups = np.full(n_users, G2, dtype=np.int8)
    groups[rng.permutation(n_users)[: (n_users + 1) // 2]] = G1
    return groups


def perturb_users(x, groups, mech: Mechanism, tf: Transform,
                  rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    reports = np.empty(len(x), dtype=np.float64)
    g1 = groups == G1
    reports[g1] = mech.perturb(tf.to_g1(x[g1]), rng)
    reports[~g1] = mech.perturb(tf.to_g2(x[~g1]), rng)
    return reports


@dataclass(frozen=True)
class Estimate:
    mean: float
    second_moment: float
    variance: float

    @classmethod
    def from_moments(cls, mean: float, second: float) -> "Estimate":
        return cls(float(mean), float(second), float(second - mean ** 2))

    def normalized(self, tf: Transform) -> "Estimate":
        mean = float(tf.norm_mean(self.mean))
        var = float(tf.norm_var(self.variance))
        return Estimate(mean, var + mean ** 2, var)


def group_decoded_means(reports, groups, mech: Mechanism, tf: Transform):
    reports = np.asarray(reports, dtype=np.float64)
    g1 = groups == G1
    n1, n2 = int(g1.sum()), int((~g1).sum())
    m1 = float(mech.decode(reports[g1], tf.k1, tf.a).mean()) if n1 else float('nan')
    m2 = float(mech.decode(reports[~g1], tf.k2, tf.q_lo).mean()) if n2 else float('nan')
    return m1, m2, n1, n2


def estimate(reports, groups, mech: Mechanism, tf: Transform) -> Estimate:
    m1, m2, n1, n2 = group_decoded_means(reports, groups, mech, tf)
    if not n1 or not n2:
        raise ValueError("both groups need at least one report")
    return Estimate.from_moments(m1, tf.second_from_query(m2, m1))


def estimate_retained(reports, groups, keep, mech: Mechanism, tf: Transform,
                      min_per_group: int = 1):
    """Standard estimate on the retained reports, with an observation-only fallback.
    """
    reports = np.asarray(reports, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    fallback = []
    use = keep.copy()
    for g in (G1, G2):
        in_g = groups == g
        if int((keep & in_g).sum()) < min_per_group:
            use[in_g] = True
            fallback.append(g)
    return estimate(reports[use], groups[use], mech, tf), fallback


def exact_moments(x) -> Estimate:
    x = np.asarray(x, dtype=np.float64)
    return Estimate.from_moments(x.mean(), np.mean(x * x))
