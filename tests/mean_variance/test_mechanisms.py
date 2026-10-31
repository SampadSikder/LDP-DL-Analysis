import math
import unittest

import numpy as np
from scipy.stats import chisquare

from attacker_detector.mean_variance.mechanisms import SR, PM, get_mechanism
from attacker_detector.mean_variance.transforms import Transform

N = 200_000
Z = 5.0          # tolerance in standard errors for Monte Carlo checks
EPSILONS = (0.5, 1.0, 4.0)
INPUTS = (-1.0, -0.3, 0.0, 0.5, 1.0)


class TestFactory(unittest.TestCase):
    def test_get_mechanism(self):
        self.assertIsInstance(get_mechanism('SR', 1.0), SR)
        self.assertIsInstance(get_mechanism('PM', 1.0), PM)
        with self.assertRaises(ValueError):
            get_mechanism('OUE', 1.0)

    def test_rejects_nonpositive_epsilon(self):
        for cls in (SR, PM):
            for eps in (0.0, -1.0):
                with self.assertRaises(ValueError):
                    cls(eps)

    def test_rejects_inputs_outside_domain(self):
        rng = np.random.default_rng(0)
        for mech in (SR(1.0), PM(1.0)):
            with self.assertRaises(ValueError):
                mech.perturb(np.array([0.0, 1.5]), rng)
            mech.perturb(np.array([1.0 + 1e-12, -1.0 - 1e-12]), rng)   # float noise is clipped


class TestSR(unittest.TestCase):
    def test_constants_and_cap(self):
        for eps in EPSILONS:
            m = SR(eps)
            self.assertAlmostEqual(m.p, math.exp(eps) / (1 + math.exp(eps)), places=15)
            self.assertAlmostEqual(m.p + m.q, 1.0, places=15)
            self.assertAlmostEqual(m.cap, 1 / math.tanh(eps / 2), places=12)

    def test_outputs_are_plus_minus_one(self):
        m, rng = SR(1.0), np.random.default_rng(1)
        y = m.perturb(np.linspace(-1, 1, 10_000), rng)
        self.assertEqual(y.dtype, np.float64)
        self.assertTrue(set(np.unique(y)) <= {-1.0, 1.0})

    def test_probability_formula(self):
        for eps in EPSILONS:
            m = SR(eps)
            np.testing.assert_allclose(m.prob_plus(np.array([-1.0, 0.0, 1.0])), [m.q, 0.5, m.p], rtol=1e-14)

    def test_ldp_ratio_is_e_eps(self):
        for eps in EPSILONS:
            m = SR(eps)
            hi, lo = m.prob_plus(1.0), m.prob_plus(-1.0)
            self.assertAlmostEqual(hi / lo, math.exp(eps), places=10)
            self.assertAlmostEqual((1 - lo) / (1 - hi), math.exp(eps), places=10)

    def test_empirical_probability(self):
        rng = np.random.default_rng(2)
        for eps in EPSILONS:
            m = SR(eps)
            for xt in INPUTS:
                pr = float(m.prob_plus(xt))
                freq = np.mean(m.perturb(np.full(N, xt), rng) == 1.0)
                self.assertLess(abs(freq - pr), Z * math.sqrt(pr * (1 - pr) / N), (eps, xt))

    def test_unbiased(self):
        rng = np.random.default_rng(3)
        for eps in EPSILONS:
            m = SR(eps)
            for xt in INPUTS:
                u = m.unbias(m.perturb(np.full(N, xt), rng))
                se = math.sqrt((m.cap ** 2 - xt ** 2) / N)
                self.assertLess(abs(u.mean() - xt), Z * se, (eps, xt))


class TestPM(unittest.TestCase):
    def test_constants_and_cap(self):
        for eps in EPSILONS:
            m = PM(eps)
            e2 = math.exp(eps / 2)
            self.assertAlmostEqual(m.s, (e2 + 1) / (e2 - 1), places=12)
            self.assertAlmostEqual(m.cap, 1 / math.tanh(eps / 4), places=12)
            self.assertAlmostEqual(m.pdf_high / m.pdf_low, math.exp(eps), places=10)
            # densities integrate to 1: plateau of width s - 1, rest of width s + 1
            self.assertAlmostEqual(m.pdf_high * (m.s - 1) + m.pdf_low * (m.s + 1), 1.0, places=12)
            self.assertAlmostEqual(m.pdf_high * (m.s - 1), m.p_inside, places=12)

    def test_interval(self):
        for eps in EPSILONS:
            m = PM(eps)
            l, r = m.interval(np.array(INPUTS))
            np.testing.assert_allclose(r - l, m.s - 1, rtol=1e-12)
            self.assertAlmostEqual(float(m.interval(-1.0)[0]), -m.s, places=12)
            self.assertAlmostEqual(float(m.interval(1.0)[1]), m.s, places=12)

    def test_outputs_in_range(self):
        rng = np.random.default_rng(4)
        for eps in EPSILONS:
            m = PM(eps)
            y = m.perturb(rng.uniform(-1, 1, N), rng)
            self.assertGreaterEqual(y.min(), -m.s)
            self.assertLessEqual(y.max(), m.s)

    def test_mass_inside_interval(self):
        rng = np.random.default_rng(5)
        for eps in EPSILONS:
            m = PM(eps)
            for xt in INPUTS:
                y = m.perturb(np.full(N, xt), rng)
                l, r = m.interval(xt)
                frac = np.mean((y >= l) & (y <= r))
                se = math.sqrt(m.p_inside * (1 - m.p_inside) / N)
                self.assertLess(abs(frac - m.p_inside), Z * se, (eps, xt))

    def test_histogram_matches_pdf(self):
        rng = np.random.default_rng(6)
        for eps in EPSILONS:
            m = PM(eps)
            for xt in (-0.7, 0.2, 1.0):
                y = m.perturb(np.full(N, xt), rng)
                edges = np.linspace(-m.s, m.s, 51)
                l, r = m.interval(xt)
                overlap = np.clip(np.minimum(edges[1:], r) - np.maximum(edges[:-1], l), 0, None)
                prob = m.pdf_low * np.diff(edges) + (m.pdf_high - m.pdf_low) * overlap
                obs = np.histogram(y, bins=edges)[0]
                p = chisquare(obs, prob / prob.sum() * N).pvalue
                self.assertGreater(p, 1e-4, (eps, xt))

    def test_unbiased_and_variance(self):
        rng = np.random.default_rng(7)
        for eps in EPSILONS:
            m = PM(eps)
            e2 = m.e2
            for xt in INPUTS:
                # Var[y | x~] from the piecewise density, and the closed form behind Lemma 1
                l, r = m.interval(xt)
                ey2 = m.pdf_low * 2 * m.s ** 3 / 3 + (m.pdf_high - m.pdf_low) * (r ** 3 - l ** 3) / 3
                var = ey2 - xt ** 2
                self.assertAlmostEqual(var, xt ** 2 / (e2 - 1) + (e2 + 3) / (3 * (e2 - 1) ** 2), places=10)
                y = m.perturb(np.full(N, xt), rng)
                self.assertLess(abs(y.mean() - xt), Z * math.sqrt(var / N), (eps, xt))
                self.assertLess(abs(y.var() / var - 1), 0.03, (eps, xt))

    def test_pdf_function(self):
        m = PM(1.0)
        l, r = m.interval(0.3)
        self.assertEqual(float(m.pdf((l + r) / 2, 0.3)), m.pdf_high)
        self.assertEqual(float(m.pdf(-m.s, 0.3)), m.pdf_low)
        self.assertEqual(float(m.pdf(m.s + 0.1, 0.3)), 0.0)


class TestDecode(unittest.TestCase):
    """Phi maps an expected report back to the raw value: x for g1, x^2 for g2."""

    def test_decode_expected_report(self):
        tf = Transform(1.0, 265.0)
        x = np.linspace(1.0, 265.0, 101)
        for mech in (SR(1.0), PM(1.0)):
            scale = (mech.p - mech.q) if isinstance(mech, SR) else 1.0   # E[y] = scale * x~
            np.testing.assert_allclose(mech.decode(scale * tf.to_g1(x), tf.k1, tf.a), x, rtol=1e-12)
            np.testing.assert_allclose(mech.decode(scale * tf.to_g2(x), tf.k2, tf.lo2), x ** 2, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
