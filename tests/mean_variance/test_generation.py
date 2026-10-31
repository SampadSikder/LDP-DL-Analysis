"""Populations, round generation, views, pairing and store resume (needs the Taxi file)."""
import os
import shutil
import tempfile
import unittest

import numpy as np

from attacker_detector.mean_variance.estimator import G1, G2, estimate
from attacker_detector.mean_variance.experiment import PopulationBank, enumerate_specs
from attacker_detector.mean_variance.mechanisms import get_mechanism
from attacker_detector.mean_variance.populations import (
    SPLITS, build_population, record_partition, sample_recipe,
)
from attacker_detector.mean_variance.scenarios import RoundSpec, generate_round, resolve_targets
from attacker_detector.mean_variance.sources import DATA_ROOT, SOURCES, load_source
from attacker_detector.mean_variance.store import RoundStore
from attacker_detector.mean_variance.transforms import Transform

HAVE_TAXI = any(os.path.exists(os.path.join(DATA_ROOT, 'taxi', p)) for p in SOURCES['taxi'].patterns)
OBSERVATION_KEYS = {'reports', 'groups', 'n_g1', 'n_g2', 'protocol', 'epsilon', 'domain', 'square_mode'}


@unittest.skipUnless(HAVE_TAXI, "taxi data not present")
class TestPopulations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_source('taxi')
        cls.part = record_partition(cls.data)

    def test_partition_fixed_and_proportional(self):
        again = record_partition(self.data)
        np.testing.assert_array_equal(self.part, again)
        frac = np.bincount(self.part, minlength=3) / len(self.part)
        np.testing.assert_allclose(frac, [0.6, 0.2, 0.2], atol=0.01)

    def test_population_stays_in_its_split(self):
        for split in SPLITS:
            rec = sample_recipe(self.data, self.part, split, 3, 4096)
            vals, idx = build_population(self.data, self.part, rec)
            self.assertEqual(len(vals), 4096)
            self.assertTrue(np.all(self.part[idx] == SPLITS.index(split)))
            self.assertEqual(len(np.unique(idx)), 4096)            # without replacement
            np.testing.assert_array_equal(vals, build_population(self.data, self.part, rec)[0])

    def test_recipes_vary_moments(self):
        tf = Transform(*SOURCES['taxi'].domain)
        means = [tf.to_g1(build_population(self.data, self.part,
                                           sample_recipe(self.data, self.part, 'train', i, 4096))[0]).mean()
                 for i in range(20)]
        self.assertGreater(np.std(means), 0.05)      # far more than resampling noise (~0.009)


def _spec(**kw):
    base = dict(experiment='t', split='train', source='taxi', population_id='p', domain=(1.0, 265.0),
                protocol='PM', epsilon=1.0, total_size=4096, beta=0.1, rep=0, attack='OPA', target='m1v3',
                spread_ratio=0.25)
    base.update(kw)
    return RoundSpec(**base)


@unittest.skipUnless(HAVE_TAXI, "taxi data not present")
class TestRounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = PopulationBank('taxi', 8192)
        cls.pop = cls.bank.values('train', 0)
        cls.printed = SOURCES['taxi'].printed_targets

    def test_views_and_counts(self):
        r = generate_round(_spec(), self.pop, self.printed)
        self.assertTrue(r.generated, r.meta)
        obs = r.observation()
        self.assertEqual(set(obs), OBSERVATION_KEYS)
        self.assertEqual(len(obs['reports']), 4096)
        self.assertEqual(obs['n_g1'], 2048)
        m = int(round(0.1 * 4096))
        self.assertEqual(int(r.is_fake.sum()), m)
        self.assertAlmostEqual(r.meta['realized_beta'], m / 4096)
        self.assertEqual(r.meta['m_g1'] + r.meta['m_g2'], m)
        # labels: genuine empirical moments of the first n population values (ddof = 0)
        tf = Transform(1.0, 265.0)
        xt = tf.to_g1(self.pop[:4096 - m])
        self.assertAlmostEqual(r.labels['genuine_mean'], xt.mean(), places=12)
        self.assertAlmostEqual(r.labels['genuine_variance'], xt.var(), places=12)

    def test_deterministic_and_checksum(self):
        a = generate_round(_spec(), self.pop, self.printed)
        b = generate_round(_spec(), self.pop, self.printed)
        self.assertEqual(a.meta['checksum'], b.meta['checksum'])
        c = generate_round(_spec(rep=1), self.pop, self.printed)
        self.assertNotEqual(a.meta['checksum'], c.meta['checksum'])

    def test_attack_variants_share_genuine_reports(self):
        a = generate_round(_spec(attack='OPA', target='m1v3'), self.pop, self.printed)
        b = generate_round(_spec(attack='IPA', target='m3v1', spread_ratio=None), self.pop, self.printed)
        self.assertEqual(a.spec.base_round_id, b.spec.base_round_id)
        np.testing.assert_array_equal(np.sort(a.reports[~a.is_fake]), np.sort(b.reports[~b.is_fake]))

    def test_clean_counterpart_is_genuine_subset(self):
        r = generate_round(_spec(), self.pop, self.printed)
        tf, mech = Transform(1.0, 265.0), get_mechanism('PM', 1.0)
        g = ~r.is_fake
        est = estimate(r.reports[g], r.groups[g], mech, tf).normalized(tf)
        self.assertAlmostEqual(est.mean, r.labels['clean_mean'], places=10)

    def test_infeasible_round_has_no_observation(self):
        r = generate_round(_spec(protocol='SR', beta=0.05, target='m1v1', spread_ratio=None,
                                 epsilon=6.0, target_mode='absolute'), self.pop, self.printed)
        # taxi: mean +0.08 with 5% fakes at eps 6 (SR cap ~1.005) is out of reach
        self.assertEqual(r.status, 'infeasible_target')
        self.assertFalse(r.generated)

    def test_clean_round(self):
        r = generate_round(_spec(attack='none', target='clean', beta=0.0, spread_ratio=None), self.pop, self.printed)
        self.assertTrue(r.generated)
        self.assertEqual(int(r.is_fake.sum()), 0)
        self.assertEqual(len(r.reports), 4096)

    def test_offset_targets(self):
        from attacker_detector.mean_variance.estimator import Estimate
        g = Estimate(0.1, 0.35, 0.34)
        mt, vt = resolve_targets(self.printed, 'offset', g, 0, 2)
        self.assertAlmostEqual(mt, 0.1 + 0.082)
        self.assertAlmostEqual(vt, 0.34 * 0.40 / 0.34)
        mt, vt = resolve_targets(self.printed, 'absolute', g, 1, 1)
        self.assertEqual((mt, vt), (0.1, 0.34))


@unittest.skipUnless(HAVE_TAXI, "taxi data not present")
class TestStore(unittest.TestCase):
    def test_generate_resume_load(self):
        cfg = {'experiment': 'unit', 'protocols': ['SR', 'PM'], 'sources': ['taxi'], 'epsilon': [1.0],
               'beta': [0.0, 0.2], 'attacks': ['OPA', 'IPA'], 'targets': ['m1v1', 'm2v2'],
               'rounds': {'train': 2, 'val': 1, 'test': 1}, 'sizes': [2048]}
        banks = {'taxi': PopulationBank('taxi', 2048)}
        specs = enumerate_specs(cfg, 'train', banks)
        self.assertEqual(len(specs), 2 * 2 * (1 + 2 * 2))       # protocols x reps x (clean + attacks x targets)
        tmp = tempfile.mkdtemp()
        try:
            store = RoundStore(tmp)
            self.assertEqual(store.generate(specs[:7], banks, log_every=0), 7)
            self.assertEqual(store.generate(specs, banks, log_every=0), len(specs) - 7)   # resume
            self.assertEqual(store.generate(specs, banks, log_every=0), 0)
            man = store.manifest()
            self.assertEqual(len(man), len(specs))
            loaded = store.load('train', 'PM', 1.0, 'taxi')
            ids = [rid for rid, *_ in loaded]
            self.assertEqual(len(ids), len(set(ids)))                # no duplicates after resume
            n_gen = sum(1 for s in specs if s.protocol == 'PM' and man[s.round_id]['status'] != 'infeasible_target')
            self.assertEqual(len(loaded), n_gen)
            rid, reports, groups, fake = loaded[0]
            spec = next(s for s in specs if s.round_id == rid)
            r = generate_round(spec, banks['taxi'].values('train', spec.rep), SOURCES['taxi'].printed_targets)
            np.testing.assert_allclose(reports, r.reports.astype(np.float32), rtol=0, atol=0)
            np.testing.assert_array_equal(groups, r.groups)
        finally:
            shutil.rmtree(tmp)

    def test_crash_after_shard_before_manifest(self):
        """Shard written, manifest lines lost: re-running must not duplicate rounds."""
        cfg = {'experiment': 'unit2', 'protocols': ['PM'], 'sources': ['taxi'], 'epsilon': [1.0],
               'beta': [0.0, 0.2], 'attacks': ['OPA'], 'targets': ['m1v1'],
               'rounds': {'train': 3, 'val': 1, 'test': 1}, 'sizes': [2048]}
        banks = {'taxi': PopulationBank('taxi', 2048)}
        specs = enumerate_specs(cfg, 'train', banks)
        tmp = tempfile.mkdtemp()
        try:
            store = RoundStore(tmp)
            store.generate(specs, banks, log_every=0)
            with open(store.manifest_path) as f:
                lines = f.readlines()
            with open(store.manifest_path, 'w') as f:        # simulate the crash
                f.writelines(lines[:2])
            store.generate(specs, banks, log_every=0)
            ids = [rid for rid, *_ in store.load('train', 'PM', 1.0, 'taxi')]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(store.manifest()), len(specs))
        finally:
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
