import math
import unittest

import numpy as np

from attacker_detector.mean_variance.knowledge import (
    AttackerKnowledge, exact, from_compromised, from_history, round_count,
)


class TestKnowledge(unittest.TestCase):
    def test_round_count(self):
        # paper Table 4 n_e values come from these counts
        self.assertEqual(round_count(83_130, 'sig1'), 80_000)
        self.assertEqual(round_count(2_390_203, 'sig2'), 2_400_000)
        self.assertEqual(round_count(97_220, 'sig1'), 100_000)
        self.assertEqual(round_count(4096, 'exact'), 4096)
        with self.assertRaises(ValueError):
            round_count(10, 'sig3')

    def test_exact(self):
        x = np.arange(1.0, 101.0)
        k = exact(x)
        self.assertEqual((k.n_e, k.S1_e, k.S2_e, k.regime), (100.0, x.sum(), np.sum(x * x), 'exact'))
        self.assertAlmostEqual(k.mean_e, x.mean())

    def test_compromised_unbiased_and_sample_stays_genuine(self):
        x = (np.arange(20_000) / 19_999.0) ** 3 * 100
        rng = np.random.default_rng(0)
        s1 = []
        for _ in range(400):
            k, idx = from_compromised(x, 1000, len(x), rng)
            self.assertEqual(len(np.unique(idx)), 1000)
            self.assertTrue(np.all(idx < len(x)))
            s1.append(k.S1_e)
        s1 = np.asarray(s1)
        self.assertLess(abs(s1.mean() - x.sum()), 5 * s1.std() / math.sqrt(len(s1)))
        self.assertEqual(k.h, 1000)
        self.assertEqual(k.regime, 'compromised')

    def test_compromised_caps_sample_at_population(self):
        k, idx = from_compromised(np.arange(10.0), 1000, 10, np.random.default_rng(1))
        self.assertEqual(k.h, 10)
        self.assertAlmostEqual(k.S1_e, 45.0)

    def test_with_error_and_history(self):
        k = AttackerKnowledge(100.0, 50.0, 40.0, 'exact').with_error(dS1=5, dS2=-4, dn=10)
        self.assertEqual((k.n_e, k.S1_e, k.S2_e), (110.0, 55.0, 36.0))
        self.assertEqual(k.regime, 'exact+error')
        h = from_history(0.5, 0.1, 100)
        self.assertAlmostEqual(h.S2_e, 100 * 0.35)


class TestErrorInjection(unittest.TestCase):
    def test_errors_land_on_benchmark_scale_sums(self):
        from attacker_detector.mean_variance.scenarios import inject_knowledge_error
        from attacker_detector.mean_variance.transforms import Transform
        tf = Transform(1.0, 265.0)
        x = 1 + 264 * (np.arange(5000) / 4999.0) ** 2
        k = exact(x)
        S1n, S2n = tf.norm_sums(k.S1_e, k.S2_e, k.n_e)
        for d1, d2, dn in ((-131.5, 1217.0, -3130), (0.0, 0.0, 500), (40.0, 0.0, 0)):
            e = inject_knowledge_error(k, tf, d1, d2, dn)
            a1, a2 = tf.norm_sums(e.S1_e, e.S2_e, e.n_e)
            self.assertAlmostEqual(e.n_e, k.n_e + dn)
            self.assertAlmostEqual(a1, S1n + d1, delta=1e-6 * abs(S1n) + 1e-6)
            self.assertAlmostEqual(a2, S2n + d2, delta=1e-6 * abs(S2n) + 1e-6)

    def test_denorm_sums_inverse(self):
        from attacker_detector.mean_variance.transforms import Transform
        tf = Transform(-3.0, 7.0)
        x = np.linspace(-3, 7, 101)
        S1, S2 = x.sum(), np.sum(x * x)
        b1, b2 = tf.denorm_sums(*tf.norm_sums(S1, S2, len(x)), len(x))
        self.assertAlmostEqual(b1, S1, places=8)
        self.assertAlmostEqual(b2, S2, places=6)


if __name__ == '__main__':
    unittest.main()
