import unittest

import numpy as np

from attacker_detector.mean_variance.transforms import Transform


class TestTransform(unittest.TestCase):
    def setUp(self):
        self.tf = Transform(1.0, 265.0)          # the paper's Taxi range
        self.x = np.linspace(1.0, 265.0, 1001)

    def test_rejects_bad_range(self):
        for a, b in ((1.0, 1.0), (2.0, 1.0), (0.0, np.inf), (np.nan, 1.0)):
            with self.assertRaises(ValueError):
                Transform(a, b)

    def test_from_data_uses_min_max(self):
        tf = Transform.from_data([5.0, -2.0, 7.5])
        self.assertEqual((tf.a, tf.b), (-2.0, 7.5))

    def test_g1_endpoints_and_round_trip(self):
        self.assertAlmostEqual(float(self.tf.to_g1(1.0)), -1.0, places=12)
        self.assertAlmostEqual(float(self.tf.to_g1(265.0)), 1.0, places=12)
        np.testing.assert_allclose(self.tf.from_g1(self.tf.to_g1(self.x)), self.x, rtol=1e-12)

    def test_g2_matches_paper_form_when_a_nonnegative(self):
        self.assertEqual(self.tf.lo2, 1.0)
        self.assertEqual(self.tf.hi2, 265.0 ** 2)
        self.assertAlmostEqual(self.tf.k2, 2.0 / abs(265.0 ** 2 - 1.0), places=15)
        self.assertAlmostEqual(float(self.tf.to_g2(1.0)), -1.0, places=12)
        self.assertAlmostEqual(float(self.tf.to_g2(265.0)), 1.0, places=12)
        np.testing.assert_allclose(self.tf.from_g2(self.tf.to_g2(self.x)), self.x ** 2, rtol=1e-12)

    def test_g2_range_when_a_negative(self):
        tf = Transform(-3.0, 2.0)
        self.assertEqual((tf.lo2, tf.hi2), (0.0, 9.0))
        xs = np.linspace(-3.0, 2.0, 501)
        xt2 = tf.to_g2(xs)
        self.assertGreaterEqual(xt2.min(), -1.0 - 1e-12)
        self.assertLessEqual(xt2.max(), 1.0 + 1e-12)
        self.assertAlmostEqual(float(tf.to_g2(0.0)), -1.0, places=12)
        self.assertAlmostEqual(float(tf.to_g2(-3.0)), 1.0, places=12)

    def test_normalized_moments_match_direct(self):
        x = 1.0 + 264.0 * (np.arange(5000) / 5000.0) ** 3     # skewed, inside [1, 265]
        xt = self.tf.to_g1(x)
        self.assertAlmostEqual(self.tf.norm_mean(x.mean()), xt.mean(), places=12)
        self.assertAlmostEqual(self.tf.norm_var(x.var()), xt.var(), places=12)
        self.assertAlmostEqual(self.tf.denorm_mean(xt.mean()), x.mean(), places=9)
        self.assertAlmostEqual(self.tf.denorm_var(xt.var()), x.var(), places=6)

    def test_rejects_unknown_square_mode(self):
        with self.assertRaises(ValueError):
            Transform(0.0, 1.0, 'cubed')

    def test_normalized_square_mode(self):
        tf = Transform(-3.0, 2.0, 'normalized')
        xs = np.linspace(-3.0, 2.0, 501)
        xt2 = tf.to_g2(xs)
        self.assertGreaterEqual(xt2.min(), -1.0 - 1e-12)
        self.assertLessEqual(xt2.max(), 1.0 + 1e-12)
        np.testing.assert_allclose(tf.from_g2(xt2), tf.to_g1(xs) ** 2, rtol=1e-12, atol=1e-15)
        self.assertEqual((tf.q_lo, tf.q_hi, tf.k2), (0.0, 1.0, 2.0))

    def test_query_mean_round_trip_both_modes(self):
        x = 1.0 + 264.0 * (np.arange(5000) / 5000.0) ** 3
        mean, second = x.mean(), np.mean(x * x)
        for mode in ('raw', 'normalized'):
            tf = Transform(1.0, 265.0, mode)
            self.assertAlmostEqual(tf.query_mean(mean, second), tf.query(x).mean(), delta=1e-9 * max(1, second))
            back = tf.second_from_query(tf.query_mean(mean, second), mean)
            self.assertAlmostEqual(back / second, 1.0, places=10)

    def test_raw_targets_inverse(self):
        x = 1.0 + 264.0 * (np.arange(5000) / 5000.0) ** 3
        mean_n, var_n = self.tf.norm_mean(x.mean()), self.tf.norm_var(x.var())
        mean, second = self.tf.raw_targets(mean_n, var_n)
        self.assertAlmostEqual(mean, x.mean(), places=8)
        self.assertAlmostEqual(second / np.mean(x * x), 1.0, places=10)

    def test_norm_sums_match_direct(self):
        x = 1.0 + 264.0 * (np.arange(5000) / 5000.0) ** 3
        xt = self.tf.to_g1(x)
        S1n, S2n = self.tf.norm_sums(x.sum(), np.sum(x * x), len(x))
        self.assertAlmostEqual(S1n, xt.sum(), delta=1e-6 * len(x))
        self.assertAlmostEqual(S2n, np.sum(xt * xt), delta=1e-6 * len(x))


if __name__ == '__main__':
    unittest.main()
