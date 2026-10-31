"""Rescale raw values onto the [-1, 1] input domain of SR and PM before sending to randomizer.

[-1, 1] bounds randomizer inputs only (genuine users, IPA fakes via to_g1 / to_g2).
"""

from dataclasses import dataclass

import numpy as np

SQUARE_MODES = ('raw', 'normalized')


@dataclass(frozen=True)
class Transform:
    """Declared input range [a, b] and the g1 / g2 rescaling it implies.
    """

    a: float
    b: float
    square_mode: str = 'raw'

    def __post_init__(self):
        if not np.isfinite(self.a) or not np.isfinite(self.b) or not self.b > self.a:
            raise ValueError(f"need finite a < b, got a={self.a}, b={self.b}")
        if self.square_mode not in SQUARE_MODES:
            raise ValueError(f"square_mode must be one of {SQUARE_MODES}, got {self.square_mode!r}")

    @classmethod
    def from_data(cls, x, square_mode: str = 'raw') -> "Transform":
        x = np.asarray(x, dtype=np.float64)
        return cls(float(x.min()), float(x.max()), square_mode)

    @property
    def lo2(self) -> float:
        return 0.0 if self.a < 0.0 < self.b else min(self.a ** 2, self.b ** 2)

    @property
    def hi2(self) -> float:
        return max(self.a ** 2, self.b ** 2)

    @property
    def k1(self) -> float:
        return 2.0 / (self.b - self.a)

    @property
    def _c(self) -> float:
        return -1.0 - self.k1 * self.a

    def to_g1(self, x):
        return -1.0 + self.k1 * (np.asarray(x, dtype=np.float64) - self.a)

    def from_g1(self, xt):
        return self.a + (np.asarray(xt, dtype=np.float64) + 1.0) / self.k1

    @property
    def q_lo(self) -> float:
        return self.lo2 if self.square_mode == 'raw' else 0.0

    @property
    def q_hi(self) -> float:
        return self.hi2 if self.square_mode == 'raw' else 1.0

    @property
    def k2(self) -> float:
        return 2.0 / (self.q_hi - self.q_lo)

    def query(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x * x if self.square_mode == 'raw' else self.to_g1(x) ** 2

    def to_g2(self, x):
        """x (not q(x)) -> x~2."""
        return -1.0 + self.k2 * (self.query(x) - self.q_lo)

    def from_g2(self, xt):
        """x~2 -> q(x)."""
        return self.q_lo + (np.asarray(xt, dtype=np.float64) + 1.0) / self.k2

    def query_mean(self, mean, second):
        """E[q(x)] for raw moments E[x] = mean, E[x^2] = second."""
        if self.square_mode == 'raw':
            return second
        c, k = self._c, self.k1
        return c * c + 2.0 * c * k * mean + k * k * second

    def second_from_query(self, query_mean, mean):
        """Inverse of query_mean: raw E[x^2] from E[q(x)] and raw E[x]."""
        if self.square_mode == 'raw':
            return query_mean
        c, k = self._c, self.k1
        return (query_mean - c * c - 2.0 * c * k * mean) / (k * k)

    # ---- statistics on the benchmark's [-1, 1] scale --------------------
    def norm_mean(self, mu):
        return -1.0 + self.k1 * (mu - self.a)

    def denorm_mean(self, mu_n):
        return self.a + (mu_n + 1.0) / self.k1

    def norm_var(self, var):
        return self.k1 ** 2 * var

    def denorm_var(self, var_n):
        return var_n / self.k1 ** 2

    def norm_sums(self, S1, S2, n):
        """(sum x, sum x^2) over n users -> (sum x~1, sum x~1^2)."""
        c, k = self._c, self.k1
        return n * c + k * S1, n * c * c + 2.0 * c * k * S1 + k * k * S2

    def denorm_sums(self, S1n, S2n, n):
        """Inverse of norm_sums: (sum x~1, sum x~1^2) over n users -> (sum x, sum x^2)."""
        c, k = self._c, self.k1
        S1 = (S1n - n * c) / k
        return S1, (S2n - n * c * c - 2.0 * c * k * S1) / (k * k)

    def raw_targets(self, mean_n, var_n):
        """Benchmark-scale target (mean, variance) -> raw (E[x], E[x^2])."""
        mean = float(self.denorm_mean(mean_n))
        return mean, float(self.denorm_var(var_n)) + mean ** 2
