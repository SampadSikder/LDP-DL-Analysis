"""SR and PM: user-side perturbation and server-side decoding.

Li et al., USENIX Security 2023, Sec. 2.2.1 (SR, Duchi et al.) and 2.2.2 (PM,
Wang et al. ICDE'19). Inputs are already on [-1, 1] (see transforms.py).
"""

import math

import numpy as np

# Inputs this far outside [-1, 1] are a caller bug; anything closer is float noise
# from the rescaling (e.g. x = b giving 1.0000000000000002) and is clipped.
_DOMAIN_TOL = 1e-9


def _check_domain(xt) -> np.ndarray:
    xt = np.asarray(xt, dtype=np.float64)
    if xt.size and (xt.min() < -1.0 - _DOMAIN_TOL or xt.max() > 1.0 + _DOMAIN_TOL):
        raise ValueError(f"inputs must lie in [-1, 1], got [{xt.min()}, {xt.max()}]")
    return np.clip(xt, -1.0, 1.0)


class Mechanism:
    name = ''

    def __init__(self, epsilon: float):
        if not epsilon > 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.epsilon = float(epsilon)

    @property
    def cap(self) -> float:
        raise NotImplementedError

    def perturb(self, xt, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def unbias(self, y) -> np.ndarray:
        raise NotImplementedError

    def decode(self, y, k: float, lo: float) -> np.ndarray:
        """Phi: report -> unbiased estimate of the raw value (x for g1, x^2 for g2)."""
        return (self.unbias(y) + 1.0) / k + lo

    def __repr__(self):
        return f"{self.name}(epsilon={self.epsilon})"


class SR(Mechanism):
    name = 'SR'

    def __init__(self, epsilon: float):
        super().__init__(epsilon)
        e = math.exp(self.epsilon)
        self.p = e / (1.0 + e)
        self.q = 1.0 - self.p

    @property
    def cap(self) -> float:
        return 1.0 / (self.p - self.q)

    def prob_plus(self, xt) -> np.ndarray:
        return self.q + (self.p - self.q) * (1.0 + _check_domain(xt)) / 2.0

    def perturb(self, xt, rng: np.random.Generator) -> np.ndarray:
        prob = self.prob_plus(xt)
        return np.where(rng.random(prob.shape) < prob, 1.0, -1.0)

    def unbias(self, y) -> np.ndarray:
        return np.asarray(y, dtype=np.float64) / (self.p - self.q)


class PM(Mechanism):
    """Piecewise mechanism: report in [-s, s], dense on [l(x~), r(x~)].

    With probability e^{eps/2} / (e^{eps/2} + 1) the report is uniform on
    [l(x~), r(x~)], otherwise uniform on [-s, l(x~)) U (r(x~), s].
    """

    name = 'PM'

    def __init__(self, epsilon: float):
        super().__init__(epsilon)
        self.e2 = math.exp(self.epsilon / 2.0)
        self.s = (self.e2 + 1.0) / (self.e2 - 1.0)
        self.p_inside = self.e2 / (self.e2 + 1.0)
        self.pdf_high = self.e2 * (self.e2 - 1.0) / (2.0 * (self.e2 + 1.0))
        self.pdf_low = (self.e2 - 1.0) / (2.0 * (self.e2 + math.exp(self.epsilon)))

    @property
    def cap(self) -> float:
        return self.s

    def interval(self, xt):
        """(l(x~), r(x~)); the width r - l = s - 1 is the same for every input."""
        xt = _check_domain(xt)
        l = (self.e2 * xt - 1.0) / (self.e2 - 1.0)
        return l, l + (self.s - 1.0)

    def pdf(self, y, xt) -> np.ndarray:
        """Report density at y given input x~ (0 outside [-s, s])."""
        y = np.asarray(y, dtype=np.float64)
        l, r = self.interval(xt)
        dens = np.where((y >= l) & (y <= r), self.pdf_high, self.pdf_low)
        return np.where(np.abs(y) <= self.s, dens, 0.0)

    def perturb(self, xt, rng: np.random.Generator) -> np.ndarray:
        l, r = self.interval(xt)
        inside = rng.random(l.shape) < self.p_inside
        u = rng.random(l.shape)
        v = u * (self.s + 1.0)
        left_len = l + self.s
        outside = np.where(v < left_len, -self.s + v, r + (v - left_len))
        return np.where(inside, l + u * (r - l), outside)

    def unbias(self, y) -> np.ndarray:
        return np.asarray(y, dtype=np.float64)


MECHANISMS = {'SR': SR, 'PM': PM}


def get_mechanism(name: str, epsilon: float) -> Mechanism:
    if name not in MECHANISMS:
        raise ValueError(f"unknown mechanism {name!r}; expected one of {sorted(MECHANISMS)}")
    return MECHANISMS[name](epsilon)
