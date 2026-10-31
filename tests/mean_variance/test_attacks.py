import math
import unittest

import numpy as np

from attacker_detector.mean_variance.attacks import (
    AttackStatus, group_targets, ipa_bounds, ipa_inputs, ipa_requirements, opa, pm_spread,
    required_fake_sums, thm1_vector, unbiased_fake_sum,
)
from attacker_detector.mean_variance.estimator import (
    G1, G2, estimate, exact_moments, perturb_users, split_groups,
)
from attacker_detector.mean_variance.knowledge import exact
from attacker_detector.mean_variance.mechanisms import PM, SR
from attacker_detector.mean_variance.transforms import Transform

Z = 5.0


def population(n=20_000):
    return 100.0 * (np.arange(n) / (n - 1)) ** 3          # skewed, fills [0, 100]


def attacked_round(mech, tf, x, m, mean_t, var_t, rng, attack='OPA', **kw):
    """One round: exact knowledge, fakes get the remaining slots of an exact-halves split."""
    n = len(x)
    N = n + m
    groups = split_groups(N, rng)
    perm = rng.permutation(N)
    gen, fake = perm[:n], perm[n:]
    reports = np.empty(N)
    reports[gen] = perturb_users(x, groups[gen], mech, tf, rng)
    k = exact(x)
    if attack == 'OPA':
        res = opa(mech, tf, k, groups[fake], N, mean_t, var_t, rng, **kw)
        if not res.status.generated:
            return res, None
        reports[fake] = res.reports
    else:
        res = ipa_inputs(tf, k, m, mean_t, var_t, rng, **kw)
        if not res.status.generated:
            return res, None
        reports[fake] = perturb_users(res.inputs, groups[fake], mech, tf, rng)
    return res, estimate(reports, groups, mech, tf).normalized(tf)


class TestTargets(unittest.TestCase):
    def test_group_targets_both_modes(self):
        for mode in ('raw', 'normalized'):
            tf = Transform(0.0, 100.0, mode)
            mean_raw, q = group_targets(tf, 0.1, 0.2)
            self.assertAlmostEqual(tf.norm_mean(mean_raw), 0.1, places=12)
            second = tf.second_from_query(q, mean_raw)
            self.assertAlmostEqual(tf.norm_var(second - mean_raw ** 2), 0.2, places=10)

    def test_paper_typos_miss_the_target(self):
        """Sec. 4.2.2's (sigma^2 + mu) and Eq. 3's -S_e^(2) give a different second moment."""
        tf = Transform(-1.0, 1.0, 'normalized')     # raw == benchmark scale
        mean_t, var_t = -0.5, 0.1
        _, q_correct = group_targets(tf, mean_t, var_t)
        self.assertAlmostEqual(q_correct, var_t + mean_t ** 2, places=12)
        self.assertNotAlmostEqual(var_t + mean_t, var_t + mean_t ** 2, places=3)
        x = np.linspace(-1, 1, 101)
        k = exact(x)
        A, B = ipa_requirements(tf, k, 50, mean_t, var_t)
        _, S2n = tf.norm_sums(k.S1_e, k.S2_e, k.n_e)
        B_printed_sign = (k.n_e + 50) * (var_t + mean_t ** 2) + S2n     # Eq. 3 as printed, expanded
        self.assertAlmostEqual(B, (k.n_e + 50) * (var_t + mean_t ** 2) - S2n)
        self.assertNotAlmostEqual(B, B_printed_sign, places=3)


class TestOPA(unittest.TestCase):
    def setUp(self):
        self.x = population()
        t = exact_moments(self.x)
        self.tf = Transform(0.0, 100.0)
        n = t.normalized(self.tf)
        self.mean_t, self.var_t = n.mean + 0.05, n.variance - 0.01
        self.m = 2222                                    # beta ~ 0.1, N = 22,222 (even)

    def test_expected_estimate_hits_target(self):
        runs = 200
        for mode in ('raw', 'normalized'):
            tf = Transform(0.0, 100.0, mode)
            for mech in (SR(1.0), PM(1.0)):
                rng = np.random.default_rng(10)
                means, seconds = [], []
                for _ in range(runs):
                    res, est = attacked_round(mech, tf, self.x, self.m, self.mean_t, self.var_t, rng)
                    self.assertTrue(res.status.generated, (mode, mech, res.info))
                    means.append(est.mean)
                    seconds.append(est.second_moment)
                means, seconds = np.asarray(means), np.asarray(seconds)
                q_t = self.var_t + self.mean_t ** 2      # Eq. 5's slack: E[mu^2] -> mu_t^2
                self.assertLess(abs(means.mean() - self.mean_t), Z * means.std() / math.sqrt(runs), (mode, mech))
                self.assertLess(abs(seconds.mean() - q_t), Z * seconds.std() / math.sqrt(runs), (mode, mech))

    def test_sr_counts(self):
        mech, rng = SR(1.0), np.random.default_rng(11)
        groups = split_groups(self.m, rng)
        res = opa(mech, self.tf, exact(self.x), groups, len(self.x) + self.m, self.mean_t, self.var_t, rng)
        self.assertTrue(res.status.generated)
        self.assertTrue(set(np.unique(res.reports)) <= {-1.0, 1.0})
        self.assertEqual(res.reports.dtype, np.float64)
        for g in (G1, G2):
            gi = res.info['groups'][g]
            vals = res.reports[groups == g]
            self.assertEqual(int(np.sum(vals == 1.0)), gi['n_plus'])
            self.assertLessEqual(abs(vals.sum() - gi['output_sum_target']), 1.0 + 1e-9)

    def test_sr_infeasible_is_not_generated(self):
        mech, rng = SR(1.0), np.random.default_rng(12)
        groups = split_groups(20, rng)                    # 20 fakes cannot move 20,000 users to +0.9
        res = opa(mech, self.tf, exact(self.x), groups, len(self.x) + 20, 0.9, 0.01, rng)
        self.assertEqual(res.status, AttackStatus.INFEASIBLE)
        self.assertIsNone(res.reports)

    def test_pm_sum_support_and_spread(self):
        mech, rng = PM(1.0), np.random.default_rng(13)
        groups = split_groups(self.m, rng)
        N = len(self.x) + self.m
        k = exact(self.x)
        R = required_fake_sums(self.tf, k, self.m, {g: int(np.sum(groups == g)) for g in (G1, G2)},
                               N, self.mean_t, self.var_t)
        for ratio in (0.0, 0.05, 0.25, 1.0):
            res = opa(mech, self.tf, k, groups, N, self.mean_t, self.var_t, rng, spread_ratio=ratio)
            self.assertEqual(res.status, AttackStatus.FEASIBLE)
            for g in (G1, G2):
                vals = res.reports[groups == g]
                U = unbiased_fake_sum(self.tf, g, R[g], len(vals))
                self.assertLess(abs(vals.sum() - U), 1e-8 * len(vals))
                self.assertLess(np.abs(vals).max(), mech.s)          # strictly inside, never at +-s
                sp = res.info['groups'][g]['spread']
                if ratio == 0.0:
                    self.assertEqual(np.ptp(vals), 0.0)
                else:
                    self.assertLessEqual(sp['realized_half_width'], sp['achievable_half_width'] * 1.01 + 1e-12)
                    if not sp['truncated']:
                        self.assertGreater(sp['realized_half_width'], 0.95 * sp['nominal_half_width'])

    def test_pm_spread_truncated_near_boundary(self):
        rng, s = np.random.default_rng(14), PM(1.0).s
        c = s - 0.1
        v, info = pm_spread(c, 5000, s, 2.0, rng)
        self.assertTrue(info['truncated'])
        self.assertAlmostEqual(info['achievable_half_width'], 0.1)
        self.assertLess(v.max(), s)
        self.assertAlmostEqual(v.sum(), c * 5000, delta=1e-8 * 5000)

    def test_pm_infeasible(self):
        mech, rng = PM(4.0), np.random.default_rng(15)       # s ~ 1.31
        groups = split_groups(20, rng)
        res = opa(mech, self.tf, exact(self.x), groups, len(self.x) + 20, 0.9, 0.01, rng)
        self.assertEqual(res.status, AttackStatus.INFEASIBLE)

    def test_group_aware_rule(self):
        runs, mech = 200, PM(1.0)
        rng = np.random.default_rng(16)
        means = np.asarray([attacked_round(mech, self.tf, self.x, self.m, self.mean_t, self.var_t, rng,
                                           sum_rule='group_aware')[1].mean for _ in range(runs)])
        self.assertLess(abs(means.mean() - self.mean_t), Z * means.std() / math.sqrt(runs))


class TestIPA(unittest.TestCase):
    def test_thm1_is_the_maximum(self):
        rng = np.random.default_rng(20)
        for A, m in ((3.7, 10), (-2.2, 9), (0.0, 7), (9.99, 10)):
            y = thm1_vector(A, m)
            self.assertAlmostEqual(y.sum(), A, places=12)
            self.assertTrue(np.all(np.abs(y) <= 1 + 1e-12))
            Bmin, Bmax = ipa_bounds(A, m)
            self.assertAlmostEqual(Bmax, float(np.sum(y ** 2)), places=9)   # analytic == constructed
            for _ in range(2000):                       # random feasible points never exceed it
                z = rng.uniform(-1, 1, m)
                z = np.clip(z + (A - z.sum()) / m, -1, 1)
                if abs(z.sum() - A) < 1e-9:
                    self.assertLessEqual(np.dot(z, z), Bmax + 1e-9)
                    self.assertGreaterEqual(np.dot(z, z), Bmin - 1e-9)

    def test_closed_form_solves_eqs_1_2(self):
        tf = Transform(0.0, 100.0)
        x = population()
        t = exact_moments(x).normalized(tf)
        k = exact(x)
        m = 5000
        res = ipa_inputs(tf, k, m, t.mean + 0.02, t.variance + 0.01, np.random.default_rng(21))
        self.assertEqual(res.status, AttackStatus.FEASIBLE)
        y = tf.to_g1(res.inputs)
        self.assertTrue(np.all((res.inputs >= 0) & (res.inputs <= 100)))
        self.assertAlmostEqual(y.sum(), res.info['A'], delta=1e-6 * m)
        self.assertAlmostEqual(np.dot(y, y), res.info['B'], delta=1e-6 * m)

    def test_infeasible_targets(self):
        tf, x = Transform(0.0, 100.0), population()
        k, rng = exact(x), np.random.default_rng(22)
        t = exact_moments(x).normalized(tf)
        cases = [(0.9, 0.01, 100),                  # |A| > m
                 (t.mean, 0.9, 1000),               # B > B_max
                 (t.mean + 0.05, 0.0, 3000)]        # B < A^2 / m
        for mean_t, var_t, m in cases:
            res = ipa_inputs(tf, k, m, mean_t, var_t, rng)
            self.assertEqual(res.status, AttackStatus.INFEASIBLE, (mean_t, var_t, m, res.info))
            self.assertIsNone(res.inputs)

    def test_adam_converges_and_failure_is_flagged(self):
        tf = Transform(-1.0, 1.0)
        x = np.linspace(-1, 1, 2001)
        k, m = exact(x), 300
        t = exact_moments(x).normalized(tf)
        res = ipa_inputs(tf, k, m, t.mean + 0.05, t.variance - 0.02, np.random.default_rng(23),
                         solver='adam', adam_iters=3000, adam_lr=1e-3)
        self.assertTrue(res.status.generated, res.info)
        self.assertLess(abs(res.info['residual_sum']), 1e-6)
        bad = ipa_inputs(tf, k, m, t.mean + 0.05, t.variance - 0.02, np.random.default_rng(23),
                         solver='adam', adam_iters=2, adam_lr=1e-6)
        self.assertEqual(bad.status, AttackStatus.SOLVER_FAILURE)
        self.assertIsNone(bad.inputs)

    def test_expected_estimate_hits_target_through_randomizer(self):
        tf, x = Transform(0.0, 100.0), population()
        t = exact_moments(x).normalized(tf)
        mean_t, var_t = t.mean + 0.02, t.variance + 0.01
        runs = 200
        for mech in (SR(1.0), PM(1.0)):
            rng = np.random.default_rng(24)
            ests = [attacked_round(mech, tf, x, 5000, mean_t, var_t, rng, attack='IPA')[1] for _ in range(runs)]
            means = np.asarray([e.mean for e in ests])
            seconds = np.asarray([e.second_moment for e in ests])
            self.assertLess(abs(means.mean() - mean_t), Z * means.std() / math.sqrt(runs), mech)
            self.assertLess(abs(seconds.mean() - (var_t + mean_t ** 2)), Z * seconds.std() / math.sqrt(runs), mech)


if __name__ == '__main__':
    unittest.main()
