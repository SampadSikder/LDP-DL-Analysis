import unittest

import numpy as np

from attacker_detector.mean_variance.attacks import AttackStatus, ipa_inputs, opa
from attacker_detector.mean_variance.estimator import split_groups
from attacker_detector.mean_variance.feasibility import (
    beta_min, ipa_sufficient, opa_sufficient, sufficient,
)
from attacker_detector.mean_variance.knowledge import AttackerKnowledge
from attacker_detector.mean_variance.mechanisms import PM, SR
from attacker_detector.mean_variance.transforms import Transform


def table3_knowledge(n, mu, var):
    """Exact knowledge from benchmark-scale moments, with Transform(-1, 1) so raw == benchmark."""
    return AttackerKnowledge(float(n), n * mu, n * (var + mu ** 2), 'exact')


class TestFeasibility(unittest.TestCase):
    def setUp(self):
        self.tf = Transform(-1.0, 1.0)
        self.k = table3_knowledge(83_130, -0.022, 0.34)          # Taxi, Table 3

    def test_opa_mean_condition_closed_form(self):
        """g1 alone: m/(n+m) >= |mu_t - mu| / (cap -+ mu) (derived 2026-10-28)."""
        for mech in (SR(1.0), PM(1.0)):
            b, m_star = beta_min('OPA', mech, self.tf, self.k, 0.06, 0.34 + 0.022 ** 2 - 0.06 ** 2)
            closed = (0.06 + 0.022) / (mech.cap + 0.022)
            # g2 may bind instead; the mean condition is a lower bound and binds here
            self.assertGreaterEqual(b, closed - 1e-4)
            self.assertLess(b, closed + 2e-3, mech)

    def test_beta_min_is_minimal(self):
        for attack in ('IPA', 'OPA'):
            for mech in (SR(1.0), PM(1.0)):
                b, m_star = beta_min(attack, mech, self.tf, self.k, 0.06, 0.33)
                self.assertTrue(sufficient(attack, mech, self.tf, self.k, m_star, 0.06, 0.33))
                self.assertFalse(sufficient(attack, mech, self.tf, self.k, m_star - 1, 0.06, 0.33))

    def test_opa_needs_fewer_fakes_than_ipa(self):
        b_ipa, _ = beta_min('IPA', SR(1.0), self.tf, self.k, 0.06, 0.33)
        for mech in (SR(1.0), PM(1.0)):
            b_opa, _ = beta_min('OPA', mech, self.tf, self.k, 0.06, 0.33)
            self.assertLess(b_opa, b_ipa)
        self.assertLess(beta_min('OPA', PM(1.0), self.tf, self.k, 0.06, 0.33)[0],
                        beta_min('OPA', SR(1.0), self.tf, self.k, 0.06, 0.33)[0])

    def test_opa_pm_uses_full_support(self):
        """OPA-PM reports span [-s, s], not [-1, 1]; a [-1, 1] cap would overstate beta_min."""
        mu, mu_t = -0.022, 0.06
        beta_cap1 = (mu_t - mu) / (1.0 + mu)          # g1 mean bound if reports were capped at 1
        for eps in (0.5, 1.0, 2.0, 4.0):
            mech = PM(eps)
            b, m_star = beta_min('OPA', mech, self.tf, self.k, mu_t, 0.33)
            self.assertLess(b, beta_cap1, eps)
            rng = np.random.default_rng(1)
            res = opa(mech, self.tf, self.k, split_groups(m_star, rng), 83_130 + m_star, mu_t, 0.33, rng)
            self.assertTrue(res.status.generated, eps)
            self.assertGreater(np.abs(res.reports).max(), 1.0, eps)
            self.assertLess(np.abs(res.reports).max(), mech.s, eps)

    def test_sufficient_matches_generator(self):
        """At m* the generators succeed; well below it they report infeasible."""
        rng = np.random.default_rng(0)
        k = table3_knowledge(5000, -0.022, 0.34)
        for mech in (SR(1.0), PM(1.0)):
            _, m_star = beta_min('OPA', mech, self.tf, k, 0.06, 0.33)
            m = int(m_star * 1.05) + 2
            groups = np.r_[np.ones(m // 2, dtype=np.int8), np.full(m - m // 2, 2, dtype=np.int8)]
            self.assertTrue(opa(mech, self.tf, k, groups, 5000 + m, 0.06, 0.33, rng).status.generated)
            m = max(2, m_star // 3)
            groups = split_groups(m, rng)
            self.assertEqual(opa(mech, self.tf, k, groups, 5000 + m, 0.06, 0.33, rng).status,
                             AttackStatus.INFEASIBLE)
        _, m_star = beta_min('IPA', SR(1.0), self.tf, k, 0.06, 0.33)
        self.assertTrue(ipa_inputs(self.tf, k, m_star, 0.06, 0.33, rng).status.generated)
        self.assertEqual(ipa_inputs(self.tf, k, max(1, m_star // 3), 0.06, 0.33, rng).status,
                         AttackStatus.INFEASIBLE)


if __name__ == '__main__':
    unittest.main()
