"""What the attacker knows about the genuine users: n_e, S1_e = sum x, S2_e = sum x^2.
Any regime can be shifted with .with_error(...) for the paper's Figs. 7-9 sweeps.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

N_E_RULES = ('exact', 'sig1', 'sig2')


def round_count(n: int, rule: str) -> float:
    """Published-style user count. 'sig1' / 'sig2' round to 1 / 2 significant figures."""
    if rule not in N_E_RULES:
        raise ValueError(f"n_e rule must be one of {N_E_RULES}, got {rule!r}")
    if rule == 'exact' or n <= 0:
        return float(n)
    digits = int(rule[-1])
    scale = 10 ** (int(math.floor(math.log10(n))) - digits + 1)
    return float(round(n / scale) * scale)


@dataclass(frozen=True)
class AttackerKnowledge:
    n_e: float
    S1_e: float
    S2_e: float
    regime: str
    h: int = 0               

    @property
    def mean_e(self) -> float:
        return self.S1_e / self.n_e

    @property
    def second_e(self) -> float:
        return self.S2_e / self.n_e

    def with_error(self, dS1: float = 0.0, dS2: float = 0.0, dn: float = 0.0) -> "AttackerKnowledge":
        """Additive estimation errors, as in Figs. 7-9 (x-axis S_e - S)."""
        return replace(self, n_e=self.n_e + dn, S1_e=self.S1_e + dS1, S2_e=self.S2_e + dS2,
                       regime=f"{self.regime}+error")


def exact(x) -> AttackerKnowledge:
    x = np.asarray(x, dtype=np.float64)
    return AttackerKnowledge(float(len(x)), float(x.sum()), float(np.sum(x * x)), 'exact')


def from_compromised(x, h: int, n_e: float, rng: np.random.Generator):
    """Sample min(h, n) genuine users without replacement; returns (knowledge, sample indices)."""
    x = np.asarray(x, dtype=np.float64)
    h_used = int(min(h, len(x)))
    if h_used < 1:
        raise ValueError("need at least one genuine user to compromise")
    idx = rng.choice(len(x), size=h_used, replace=False)
    c = x[idx]
    k = AttackerKnowledge(float(n_e), float(n_e / h_used * c.sum()),
                          float(n_e / h_used * np.sum(c * c)), 'compromised', h_used)
    return k, idx


def from_history(mean_e: float, var_e: float, n_e: float) -> AttackerKnowledge:
    return AttackerKnowledge(float(n_e), float(n_e * mean_e), float(n_e * (var_e + mean_e ** 2)), 'history')
