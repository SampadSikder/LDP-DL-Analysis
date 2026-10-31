"""Mandatory budget stage: m_min / beta_min before generation, per-attack checks, budget ladder,
theory vs construction vs realized outcomes, below-budget evaluation."""
import math
import os
import unittest

import numpy as np

from attacker_detector.mean_variance.attacks import AttackStatus, ipa_bounds
from attacker_detector.mean_variance.feasibility import (
    budget_ladder, check, check_ipa, check_opa, minimum_budget, planned_allocation,
)
from attacker_detector.mean_variance.knowledge import AttackerKnowledge, exact
from attacker_detector.mean_variance.mechanisms import PM, SR
from attacker_detector.mean_variance.scenarios import RoundSpec, generate_round
from attacker_detector.mean_variance.sources import DATA_ROOT, SOURCES
from attacker_detector.mean_variance.transforms import Transform

TF = Transform(-1.0, 1.0, 'normalized')     # raw == benchmark scale


def knowledge(n=5000, mu=-0.022, var=0.34):
    return AttackerKnowledge(float(n), n * mu, n * (var + mu ** 2), 'exact')


class TestChecks(unittest.TestCase):
    def test_planned_allocation(self):
        self.assertEqual(planned_allocation(7), {1: 4, 2: 3})
        self.assertEqual(planned_allocation(8), {1: 4, 2: 4})

    def test_opa_sr_integer_counts_and_error(self):
        k = knowledge()
        c = check_opa(SR(1.0), TF, k, 600, 0.06, 0.33)
        self.assertTrue(c.feasible)
        for g, gi in c.groups.items():
            self.assertEqual(gi['n_plus'] + gi['n_minus'], gi['m_g'])
            self.assertLessEqual(abs(gi['output_sum_achieved'] - gi['output_sum']), 1.0 + 1e-9)
            self.assertLessEqual(abs(gi['output_sum']), gi['m_g'])
        self.assertIn('approx_error_mean', c.info)
        self.assertLess(abs(c.info['approx_error_mean']), 1e-3)
        self.assertFalse(check_opa(SR(1.0), TF, k, 50, 0.06, 0.33).feasible)

    def test_opa_pm_support_and_spread(self):
        k = knowledge()
        c = check_opa(PM(1.0), TF, k, 400, 0.06, 0.33, spread_ratio=1.0)
        self.assertTrue(c.feasible)
        for gi in c.groups.values():
            self.assertLessEqual(abs(gi['per_fake_value']), PM(1.0).s)
            self.assertEqual(gi['spread_truncated'], gi['spread_requested'] > gi['spread_room'])

    def test_ipa_joint_not_independent(self):
        """A and B each reachable on their own, but not together (B < A^2/m)."""
        k = knowledge()
        m = 3000
        mean_t, var_t = 0.2, 0.2                      # A = 1710 <= m, B = 217.6 in [0, m], but B < A^2/m = 975
        c = check_ipa(TF, k, m, mean_t, var_t)
        A, B = c.info['A'], c.info['B']
        self.assertLessEqual(abs(A), m)                # sum alone is reachable
        self.assertTrue(0 <= B <= m)                   # sum of squares alone is in [0, m]
        self.assertLess(B, A * A / m)                  # jointly impossible
        self.assertFalse(c.feasible)
        self.assertIn('A alone is reachable', c.info['reason'])

    def test_oracle_label(self):
        self.assertTrue(check('OPA', PM(1.0), TF, knowledge(), 400, 0.06, 0.33, oracle=True).oracle)
        self.assertTrue(minimum_budget('IPA', SR(1.0), TF, knowledge(), 0.06, 0.33, oracle=True)['oracle'])


class TestMinimumBudget(unittest.TestCase):
    def test_minimal_and_ladder_rechecked(self):
        k = knowledge()
        for attack, mech in (('OPA', SR(1.0)), ('OPA', PM(1.0)), ('IPA', SR(1.0))):
            r = minimum_budget(attack, mech, TF, k, 0.06, 0.40)
            m_min = r['m_min']
            self.assertEqual(r['status'], 'feasible')
            self.assertAlmostEqual(r['q_t'], 0.06 ** 2 + 0.40)
            self.assertAlmostEqual(r['beta_min'], m_min / (5000 + m_min))
            self.assertTrue(check(attack, mech, TF, k, m_min, 0.06, 0.40).feasible)
            self.assertFalse(check(attack, mech, TF, k, m_min - 1, 0.06, 0.40).feasible)
            for m in budget_ladder(m_min):              # every rounded rung rechecked
                self.assertTrue(check(attack, mech, TF, k, m, 0.06, 0.40).feasible, (attack, mech, m))

    def test_ladder_rounding(self):
        self.assertEqual(budget_ladder(10), [10, 11, 13, 15, 20])
        self.assertEqual(budget_ladder(1), [1, 2])     # duplicates removed

    def test_no_fakes_needed(self):
        k = knowledge(mu=0.1, var=0.2)
        r = minimum_budget('OPA', PM(1.0), TF, k, 0.1, 0.2)
        self.assertEqual((r['m_min'], r['status']), (0, 'no_fakes_needed'))

    def test_attacker_estimate_vs_oracle_differ(self):
        x = np.linspace(-1, 1, 4001) ** 3
        truth = exact(x)
        planned = truth.with_error(dS1=40.0)           # attacker's estimate is off
        a = minimum_budget('OPA', PM(1.0), TF, planned, 0.2, 0.15)
        o = minimum_budget('OPA', PM(1.0), TF, truth, 0.2, 0.15, oracle=True)
        self.assertNotEqual(a['m_min'], o['m_min'])


HAVE_TAXI = any(os.path.exists(os.path.join(DATA_ROOT, 'taxi', p)) for p in SOURCES['taxi'].patterns)


def _spec(**kw):
    base = dict(experiment='b', split='train', source='taxi', population_id='p', domain=(1.0, 265.0),
                protocol='PM', epsilon=1.0, total_size=4000, beta=0.1, rep=0, attack='OPA', target='m1v1',
                target_mode='offset', square_mode='normalized', spread_ratio=0.25, size_mode='fixed_genuine')
    base.update(kw)
    return RoundSpec(**base)


@unittest.skipUnless(HAVE_TAXI, "taxi data not present")
class TestBudgetStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from attacker_detector.mean_variance.experiment import PopulationBank
        cls.pop = PopulationBank('taxi', 8192).values('train', 0)
        cls.printed = SOURCES['taxi'].printed_targets

    def test_budget_recorded_before_generation(self):
        r = generate_round(_spec(), self.pop, self.printed)
        b = r.meta['budget']
        for key in ('m_min', 'beta_min', 'oracle_m_min', 'oracle_beta_min', 'm', 'budget_ratio'):
            self.assertIn(key, b)
        self.assertEqual(b['m'], int(round(0.1 * 4000 / 0.9)))
        self.assertAlmostEqual(r.meta['target_second_moment'], r.meta['target_mean'] ** 2 + r.meta['target_variance'])
        self.assertIn(r.meta['theory']['status'], ('feasible', 'infeasible'))
        if r.generated:
            self.assertEqual(r.meta['construction']['status'], 'constructed')
            self.assertTrue(r.meta['construction']['support_ok'])
            self.assertIn('target_error_mean', r.meta['realized'])

    def test_fixed_beta_below_budget_not_generated_nor_raised(self):
        r = generate_round(_spec(beta=0.005, protocol='SR', spread_ratio=None), self.pop, self.printed)
        self.assertEqual(r.status, 'infeasible_target')
        self.assertFalse(r.generated)
        self.assertEqual(r.meta['theory']['status'], 'infeasible')
        self.assertEqual(r.meta['construction']['status'], 'not_attempted')
        self.assertEqual(r.meta['budget']['m'], int(round(0.005 * 4000 / 0.995)))   # budget unchanged
        self.assertLess(r.meta['budget']['m'], r.meta['budget']['m_min'])

    def test_budget_ladder_rounds(self):
        for f in (1.0, 1.1, 1.25, 1.5, 2.0):
            r = generate_round(_spec(budget_mode='mmin_multiple', budget_factor=f, beta=0.0), self.pop, self.printed)
            b = r.meta['budget']
            self.assertEqual(b['m'], math.ceil(f * b['m_min'] - 1e-9))
            self.assertEqual(r.meta['theory']['status'], 'feasible')
            if r.generated:
                self.assertEqual(len(r.reports), 4000 + b['m'])
            else:                                      # realized group split can miss a tight plan
                self.assertEqual(r.meta['construction']['status'], 'allocation_infeasible')

    def test_below_budget_saturated_is_separate(self):
        r = generate_round(_spec(budget_mode='mmin_multiple', budget_factor=0.5, beta=0.0, below_budget='saturate'),
                           self.pop, self.printed)
        self.assertEqual(r.meta['theory']['status'], 'infeasible')
        self.assertEqual(r.status, 'below_budget')
        self.assertEqual(r.meta['construction']['status'], 'below_budget_saturated')
        self.assertTrue(r.generated)
        self.assertTrue(r.meta['construction']['support_ok'])
        # it still damages the estimate toward the target, without reaching it
        shift = r.labels['standard_mean'] - r.labels['clean_mean']
        self.assertGreater(shift * (r.meta['target_mean'] - r.labels['genuine_mean']), 0)

    def test_solver_failure_is_not_infeasibility(self):
        r = generate_round(_spec(attack='IPA', target='m3v3', beta=0.3, spread_ratio=None, ipa_solver='adam'),
                           self.pop, self.printed)
        if r.meta['theory']['status'] == 'feasible' and r.meta['construction']['status'] == 'solver_failure':
            self.assertEqual(r.status, 'solver_failure')
        # force a failure: zero-iteration optimizer on a theoretically feasible target
        import attacker_detector.mean_variance.scenarios as S
        orig = S.ipa_inputs
        S.ipa_inputs = lambda *a, **kw: orig(*a, **{**kw, 'adam_iters': 1, 'adam_lr': 1e-9})
        try:
            r = generate_round(_spec(attack='IPA', target='m3v3', beta=0.3, spread_ratio=None, ipa_solver='adam'),
                               self.pop, self.printed)
        finally:
            S.ipa_inputs = orig
        self.assertEqual(r.meta['theory']['status'], 'feasible')
        self.assertEqual(r.meta['construction']['status'], 'solver_failure')
        self.assertFalse(r.generated)

    def test_default_budget_fields_keep_round_ids(self):
        a = _spec()
        b = _spec(budget_mode='beta', budget_factor=None, below_budget='skip')
        self.assertEqual(a.round_id, b.round_id)
        self.assertNotEqual(a.round_id, _spec(budget_mode='mmin_multiple', budget_factor=1.0).round_id)


if __name__ == '__main__':
    unittest.main()
