"""Acceptance checks for feature views and methods: permutation invariance, identical SR
cells, no hidden inputs, deterministic edge cases, clean-round metric conventions."""
import inspect
import unittest

import numpy as np
import torch

from attacker_detector.mean_variance import features as F
from attacker_detector.mean_variance.estimator import G1, estimate, estimate_retained, perturb_users, split_groups
from attacker_detector.mean_variance.mechanisms import PM, SR
from attacker_detector.mean_variance.methods import (
    ClassifierRemoval, ConstantMethod, KMeansRecovery, SetRecovery, StandardMethod, TrainingRound,
)
from attacker_detector.mean_variance.metrics import classification, project_moments, shift_achieved
from attacker_detector.mean_variance.transforms import Transform

FORBIDDEN = ('fake', 'label', 'beta', 'target', 'source', 'attack', 'genuine', 'round_id', 'population', 'seed')


def make_obs(protocol='PM', n=1500, eps=1.0, seed=0, fake=0):
    rng = np.random.default_rng(seed)
    tf = Transform(0.0, 100.0)
    mech = PM(eps) if protocol == 'PM' else SR(eps)
    x = 100 * rng.beta(2, 5, n)
    g = split_groups(n + fake, rng)
    y = np.r_[perturb_users(x, g[:n], mech, tf, rng), np.full(fake, 0.8 if protocol == 'PM' else 1.0)]
    is_fake = np.r_[np.zeros(n, bool), np.ones(fake, bool)]
    obs = {'reports': y, 'groups': g, 'n_g1': int(np.sum(g == 1)), 'n_g2': int(np.sum(g == 2)),
           'protocol': protocol, 'epsilon': eps, 'domain': (0.0, 100.0), 'square_mode': 'raw'}
    tm = tf.to_g1(x)
    labels = {'genuine_mean': float(tm.mean()), 'genuine_variance': float(tm.var())}
    return obs, is_fake, labels


def shuffled(obs, perm):
    o = dict(obs)
    o['reports'], o['groups'] = obs['reports'][perm], obs['groups'][perm]
    return o


class TestFeatureViews(unittest.TestCase):
    def test_pm_features_permute_consistently(self):
        for fs in ('base', 'mech'):
            obs, *_ = make_obs('PM', fake=100)
            perm = np.random.default_rng(1).permutation(len(obs['reports']))
            X, names = F.report_features(obs, fs)
            Xs, _ = F.report_features(shuffled(obs, perm), fs)
            np.testing.assert_allclose(Xs, X[perm], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(F.round_summary(obs)[0], F.round_summary(shuffled(obs, perm))[0], atol=1e-6)
            self.assertEqual(X.shape[1], len(names))

    def test_sr_identical_cells_identical_rows(self):
        obs, *_ = make_obs('SR')
        X, _ = F.report_features(obs)
        key = obs['groups'] * 10 + (obs['reports'] > 0)
        for k in np.unique(key):
            rows = X[key == k]
            self.assertTrue(np.all(rows == rows[0]))
        self.assertEqual(len(np.unique(X, axis=0)), len(np.unique(key)))

    def test_no_hidden_inputs(self):
        for fn in (F.report_features, F.round_summary, F.standard_moments):
            params = list(inspect.signature(fn).parameters)
            self.assertEqual(params[0], 'obs')
            self.assertFalse(any(any(f in p for f in FORBIDDEN) for p in params), params)
        for protocol in ('SR', 'PM'):
            obs, *_ = make_obs(protocol)
            names = F.report_features(obs, 'mech' if protocol == 'PM' else 'base')[1] + F.round_summary(obs)[1]
            self.assertFalse(any(any(f in n for f in FORBIDDEN) for n in names), names)


def training_rounds(protocol, k=24, fake_frac=(0.0, 0.1, 0.3)):
    out = []
    for i in range(k):
        n = 1200
        fake = int(fake_frac[i % len(fake_frac)] * n)
        obs, is_fake, labels = make_obs(protocol, n=n, seed=100 + i, fake=fake)
        out.append(TrainingRound(f'r{i}', obs, is_fake, labels, {'n_fake': fake, 'realized_beta': fake / (n + fake)}))
    return out


class TestMethods(unittest.TestCase):
    def test_standard_matches_estimator_and_projection(self):
        obs, *_ = make_obs('PM')
        out = StandardMethod().predict(obs)
        tf, mech = Transform(0.0, 100.0), PM(1.0)
        e = estimate(obs['reports'], obs['groups'], mech, tf).normalized(tf)
        self.assertAlmostEqual(out.mean_raw, e.mean)
        self.assertEqual((out.mean, out.var), project_moments(e.mean, e.variance))
        self.assertEqual(project_moments(1.4, -0.2), (1.0, 0.0))
        self.assertEqual(project_moments(0.5, 0.9), (0.5, 0.75))

    def test_constant_uses_training_only(self):
        tr = training_rounds('PM', k=6)
        m = ConstantMethod().fit(tr, [])
        self.assertAlmostEqual(m.mean_, np.mean([r.labels['genuine_mean'] for r in tr]))

    def test_kmeans_deterministic(self):
        obs, *_ = make_obs('PM', fake=200)
        km = KMeansRecovery(rate=0.3, n_subsets=500)
        a, b = km.predict(obs), km.predict(obs)
        self.assertEqual((a.mean_raw, a.var_raw), (b.mean_raw, b.var_raw))
        obs_sr, *_ = make_obs('SR')
        out = KMeansRecovery(rate=0.5, n_subsets=500).predict(obs_sr)
        self.assertTrue(np.isfinite(out.mean_raw))

    def test_removal_edge_case_fallback(self):
        obs, *_ = make_obs('PM')
        tf, mech = Transform(0.0, 100.0), PM(1.0)
        est, fb = estimate_retained(obs['reports'], obs['groups'], np.zeros(len(obs['reports']), bool), mech, tf, 16)
        self.assertEqual(fb, [1, 2])
        full = estimate(obs['reports'], obs['groups'], mech, tf)
        self.assertAlmostEqual(est.mean, full.mean)

    def test_classifier_scores_permute_and_cells_identical(self):
        for protocol in ('SR', 'PM'):
            tr, va = training_rounds(protocol, 18), training_rounds(protocol, 6)
            m = ClassifierRemoval('mlp', epochs=2, reports_per_round=128, device='cpu').fit(tr, va)
            obs, *_ = make_obs(protocol, fake=150, seed=7)
            perm = np.random.default_rng(2).permutation(len(obs['reports']))
            a, b = m.predict(obs), m.predict(shuffled(obs, perm))
            np.testing.assert_allclose(b.scores, a.scores[perm], rtol=1e-5, atol=1e-6)
            self.assertAlmostEqual(a.mean_raw, b.mean_raw, places=9)
            self.assertIn(m.threshold, [float(t) for t in __import__(
                'attacker_detector.mean_variance.methods.removal', fromlist=['THRESHOLDS']).THRESHOLDS])
            if protocol == 'SR':
                key = obs['groups'] * 10 + (obs['reports'] > 0)
                for k in np.unique(key):
                    self.assertEqual(len(np.unique(a.scores[key == k])), 1)

    def test_set_models_permutation_invariant(self):
        for arch in ('deepsets', 'settransformer'):
            tr, va = training_rounds('PM', 12), training_rounds('PM', 4)
            m = SetRecovery(arch, epochs=1, token_cap=256, device='cpu').fit(tr, va)
            obs, *_ = make_obs('PM', fake=120, seed=9)
            perm = np.random.default_rng(3).permutation(len(obs['reports']))
            a, b = m.predict(obs), m.predict(shuffled(obs, perm))
            self.assertAlmostEqual(a.mean_raw, b.mean_raw, places=5)
            self.assertAlmostEqual(a.var_raw, b.var_raw, places=5)
            np.testing.assert_allclose(b.scores, a.scores[perm], rtol=1e-4, atol=1e-5)
            self.assertTrue(-1 <= a.mean_raw <= 1 and 0 <= a.var_raw <= 1 - a.mean_raw ** 2 + 1e-7)


class TestCheckpoints(unittest.TestCase):
    def test_frozen_round_trip(self):
        import tempfile, shutil
        from attacker_detector.mean_variance.methods.checkpoint import load_checkpoint, save_checkpoint
        tr, va = training_rounds('PM', 12), training_rounds('PM', 4)
        obs, *_ = make_obs('PM', fake=100, seed=11)
        tmp = tempfile.mkdtemp()
        try:
            for m in (ClassifierRemoval('ft', epochs=1, reports_per_round=64, device='cpu'),
                      SetRecovery('deepsets', epochs=1, token_cap=128, device='cpu'),
                      KMeansRecovery(rate=0.4, n_subsets=200), ConstantMethod()):
                m.fit(tr, va)
                save_checkpoint(m, f"{tmp}/{m.name}")
                back = load_checkpoint(f"{tmp}/{m.name}")
                a, b = m.predict(obs), back.predict(obs)
                self.assertAlmostEqual(a.mean_raw, b.mean_raw, places=6, msg=m.name)
                self.assertAlmostEqual(a.var_raw, b.var_raw, places=6, msg=m.name)
        finally:
            shutil.rmtree(tmp)


class TestResultFiles(unittest.TestCase):
    def test_append_rows_keeps_one_column_layout(self):
        import os, tempfile
        import pandas as pd
        from attacker_detector.mean_variance.validation import _append_rows
        path = os.path.join(tempfile.mkdtemp(), 'rows.csv')
        _append_rows(path, [{'round_id': 'a', 'rate': np.nan, 'kmeans_mean': np.nan}])
        _append_rows(path, [{'kmeans_mean': 0.1, 'round_id': 'b', 'rate': 0.6}])   # other key order
        d = pd.read_csv(path)
        self.assertEqual(list(d.columns), ['round_id', 'rate', 'kmeans_mean'])
        self.assertEqual(d.loc[1, 'kmeans_mean'], 0.1)
        with self.assertRaises(ValueError):
            _append_rows(path, [{'round_id': 'c', 'new_col': 1}])


class TestMetrics(unittest.TestCase):
    def test_clean_round_conventions(self):
        fake = np.zeros(10, bool)
        c = classification(fake, np.zeros(10, bool))
        self.assertTrue(np.isnan(c['precision']) and np.isnan(c['recall']) and np.isnan(c['f1']))
        self.assertEqual(c['genuine_removal_rate'], 0.0)
        c = classification(fake, np.r_[np.ones(2, bool), np.zeros(8, bool)])
        self.assertEqual(c['precision'], 0.0)
        self.assertAlmostEqual(c['genuine_removal_rate'], 0.2)

    def test_attacked_round_metrics(self):
        fake = np.r_[np.ones(4, bool), np.zeros(6, bool)]
        removed = np.r_[np.ones(3, bool), np.zeros(5, bool), np.ones(2, bool)]
        c = classification(fake, removed, scores=removed.astype(float))
        self.assertAlmostEqual(c['precision'], 3 / 5)
        self.assertAlmostEqual(c['recall'], 3 / 4)
        self.assertAlmostEqual(c['f1'], 2 * 0.6 * 0.75 / 1.35)

    def test_shift_achieved(self):
        self.assertEqual(float(shift_achieved(0.3, 0.1, 0.3)), 1.0)
        self.assertEqual(float(shift_achieved(0.1, 0.1, 0.3)), 0.0)
        self.assertTrue(np.isnan(shift_achieved(0.2, 0.1, 0.1)))
        self.assertAlmostEqual(float(shift_achieved(0.25, 0.1, 0.3, baseline=0.15)), 0.5)


if __name__ == '__main__':
    unittest.main()
