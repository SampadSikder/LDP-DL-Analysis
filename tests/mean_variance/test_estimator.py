import math
import unittest

import numpy as np

from attacker_detector.mean_variance.estimator import (
    G1, G2, Estimate, split_groups, perturb_users, estimate, estimate_retained, exact_moments,
)
from attacker_detector.mean_variance.mechanisms import SR, PM
from attacker_detector.mean_variance.transforms import Transform

Z = 5.0


def skewed_population(n=20_000, a=0.0, b=100.0):
    """Fixed, deterministic, right-skewed values filling [a, b] (no randomness)."""
    return a + (b - a) * (np.arange(n) / (n - 1)) ** 3


class TestSplitGroups(unittest.TestCase):
    def test_exact_halves(self):
        for n in (10, 11, 100_001):
            g = split_groups(n, np.random.default_rng(0))
            self.assertEqual(len(g), n)
            self.assertTrue(set(np.unique(g)) <= {G1, G2})
            self.assertEqual(int(np.sum(g == G1)), (n + 1) // 2)

    def test_reproducible_and_random(self):
        a = split_groups(1000, np.random.default_rng(42))
        b = split_groups(1000, np.random.default_rng(42))
        c = split_groups(1000, np.random.default_rng(43))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))


class TestPerturbUsers(unittest.TestCase):
    def test_groups_report_their_own_value(self):
        # PM at very large eps: plateau width s - 1 ~ 4e-9, so a report ~ its input
        tf, mech = Transform(0.0, 100.0), PM(40.0)
        x = skewed_population(1000)
        rng = np.random.default_rng(1)
        g = split_groups(len(x), rng)
        y = perturb_users(x, g, mech, tf, rng)
        np.testing.assert_allclose(y[g == G1], tf.to_g1(x[g == G1]), atol=1e-7)
        np.testing.assert_allclose(y[g == G2], tf.to_g2(x[g == G2]), atol=1e-7)


class TestEstimate(unittest.TestCase):
    def test_variance_identity_and_group_mean(self):
        tf, mech = Transform(0.0, 100.0), SR(1.0)
        x = skewed_population(10_000)
        rng = np.random.default_rng(2)
        g = split_groups(len(x), rng)
        y = perturb_users(x, g, mech, tf, rng)
        est = estimate(y, g, mech, tf)
        self.assertAlmostEqual(est.variance, est.second_moment - est.mean ** 2, places=12)
        # paper's form: 2 / (n + m) * sum over g1 equals the g1 mean for exact halves
        paper = 2.0 / len(x) * mech.decode(y[g == G1], tf.k1, tf.a).sum()
        self.assertAlmostEqual(est.mean, paper, places=9)

    def test_needs_both_groups(self):
        tf, mech = Transform(0.0, 1.0), PM(1.0)
        with self.assertRaises(ValueError):
            estimate(np.zeros(4), np.full(4, G1, dtype=np.int8), mech, tf)

    def test_retained_subset_and_fallback(self):
        tf, mech = Transform(0.0, 100.0), PM(1.0)
        x = skewed_population(2000)
        rng = np.random.default_rng(5)
        g = split_groups(len(x), rng)
        y = perturb_users(x, g, mech, tf, rng)
        keep = np.ones(len(x), dtype=bool)
        full = estimate(y, g, mech, tf)
        est, fb = estimate_retained(y, g, keep, mech, tf)
        self.assertEqual(fb, [])
        self.assertAlmostEqual(est.mean, full.mean, places=12)
        # drop every g2 report: g2 falls back to all its reports, g1 keeps its subset
        keep = (g == G1) & (np.arange(len(x)) % 2 == 0)
        est, fb = estimate_retained(y, g, keep, mech, tf, min_per_group=5)
        self.assertEqual(fb, [G2])
        sub = (g == G1) & keep
        self.assertAlmostEqual(est.mean, float(mech.decode(y[sub], tf.k1, tf.a).mean()), places=12)
        self.assertAlmostEqual(est.second_moment, float(mech.decode(y[g == G2], tf.k2, tf.q_lo).mean()), places=9)

    def test_unbiased_normalized_square_mode(self):
        tf = Transform(0.0, 100.0, 'normalized')
        x = skewed_population()
        truth = exact_moments(x)
        runs, mech = 300, PM(1.0)
        rng = np.random.default_rng(6)
        seconds = np.empty(runs)
        for i in range(runs):
            g = split_groups(len(x), rng)
            seconds[i] = estimate(perturb_users(x, g, mech, tf, rng), g, mech, tf).second_moment
        self.assertLess(abs(seconds.mean() - truth.second_moment), Z * seconds.std() / math.sqrt(runs))

    def test_unbiased_without_attack(self):
        tf = Transform(0.0, 100.0)
        x = skewed_population()
        truth = exact_moments(x)
        runs = 300
        for mech in (SR(1.0), PM(1.0), PM(4.0)):
            rng = np.random.default_rng(3)
            means, seconds = np.empty(runs), np.empty(runs)
            for i in range(runs):
                g = split_groups(len(x), rng)
                est = estimate(perturb_users(x, g, mech, tf, rng), g, mech, tf)
                means[i], seconds[i] = est.mean, est.second_moment
            self.assertLess(abs(means.mean() - truth.mean), Z * means.std() / math.sqrt(runs), mech)
            self.assertLess(abs(seconds.mean() - truth.second_moment),
                            Z * seconds.std() / math.sqrt(runs), mech)


class TestMoments(unittest.TestCase):
    def test_exact_moments(self):
        x = skewed_population(5000)
        t = exact_moments(x)
        self.assertAlmostEqual(t.mean, x.mean(), places=12)
        self.assertAlmostEqual(t.variance, x.var(), places=8)
        self.assertAlmostEqual(t.second_moment, np.mean(x * x), places=8)

    def test_normalized(self):
        tf = Transform(0.0, 100.0)
        x = skewed_population(5000)
        n = exact_moments(x).normalized(tf)
        xt = tf.to_g1(x)
        self.assertAlmostEqual(n.mean, xt.mean(), places=12)
        self.assertAlmostEqual(n.variance, xt.var(), places=10)
        self.assertAlmostEqual(n.second_moment, np.mean(xt * xt), places=10)
        self.assertIsInstance(n, Estimate)


if __name__ == '__main__':
    unittest.main()
