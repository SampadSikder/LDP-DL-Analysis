"""IGR evaluation (attacker_detector/igr.py, scripts/igr/run_igr.py): small simulated runs, CPU.

    python -m unittest discover -s tests/igr -t .
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from attacker_detector import igr  # noqa: E402

SCRIPT = os.path.join(ROOT, 'scripts', 'igr', 'run_igr.py')


def _run(protocol, n=1500, eps=1.0, beta=0.1, seed=11, dataset='fire', h_ao=0):
    from config import DATASET_CONFIGS
    return dict(root='fake_v3f', dataset_type=dataset, protocol=protocol, epsilon=eps, attacker_ratio=beta,
                target_set_size=10, splits=4, replicate=1, n=n, domain=DATASET_CONFIGS[dataset]['domain'],
                h_ao=h_ao, seed=seed)


def _simulate_with_features(run):
    from attacker_detector.data.generators import extract_user_level_features_diffstats_style
    X, targets, support, labels, est, one = igr.simulate(run, 1)
    v2 = extract_user_level_features_diffstats_style(support, one, run['epsilon'], run['protocol'], run['domain'],
                                                     run['n'], feature_set='v2', block_candidates=16)
    return X, targets, support, labels, est, one, v2


def write_data_dir(path, runs_v2):
    """A v2 dataset directory laid out like generate_dataset.py writes it."""
    from attacker_detector.data.generators import FEATURE_NAMES_V2
    os.makedirs(path, exist_ok=True)
    feats, labels, conf = [], [], []
    for exp_id, (run, v2, lab) in enumerate(runs_v2):
        n = len(lab)
        c = np.empty((n, 10), dtype=object)
        c[:, :7] = [str(run['target_set_size']), repr(float(run['attacker_ratio'])), run['protocol'],
                    str(run['splits']), repr(float(run['epsilon'])), run['dataset_type'], str(n)]
        c[:, 7] = str(exp_id)
        c[:, 8] = np.arange(n).astype(str)
        c[:, 9] = '0'
        feats.append(v2)
        labels.append(lab)
        conf.append(c)
    F = np.vstack(feats).astype(np.float32).astype(np.float64)
    mean, std = F.mean(0), F.std(0)
    std[std < 1e-8] = 1.0
    np.save(os.path.join(path, 'features.npy'), ((F - mean) / std).astype(np.float32))
    np.save(os.path.join(path, 'labels.npy'), np.concatenate(labels).astype(np.float32))
    np.save(os.path.join(path, 'config.npy'), np.vstack(conf).astype(str))
    with open(os.path.join(path, 'norm_stats.json'), 'w') as f:
        json.dump({'feature_names': list(FEATURE_NAMES_V2), 'mean': mean.tolist(), 'std': std.tolist()}, f)


class TestEstimator(unittest.TestCase):
    def test_reproduces_generator_estimate(self):
        for protocol in ('OUE', 'OLH_Server', 'HST_Server', 'HST_User'):
            run = _run(protocol, n=1200)
            X, targets, support, labels, est, one = igr.simulate(run, 1)
            sup = igr.Support.from_dense(support)
            self.assertTrue(np.array_equal(sup.bits, support > 0), protocol)
            rd = igr.RunData(run, labels, targets, igr.genuine_frequency(X, labels, run['domain']), sup)
            np.testing.assert_allclose(rd.estimate() * run['n'], est, rtol=1e-9, atol=1e-6, err_msg=protocol)
            self.assertEqual(len(targets), 10)
            # the target set is the generator's: MGA-A fakes support every one of them far more than genuine users
            fake = labels == 1
            self.assertGreater(sup.bits[fake][:, targets].mean(), sup.bits[~fake][:, targets].mean(), protocol)


class TestMetrics(unittest.TestCase):
    def _toy(self):
        # OUE, eps = ln 3 -> q = 1/4, p = 1/2, A = 4, B = 1; 4 genuine + 2 fake users, d = 3, targets {0}
        run = dict(_run('OUE'), epsilon=float(np.log(3.0)), attacker_ratio=2 / 6, target_set_size=1, n=6)
        bits = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]], dtype=bool)
        f_true = np.array([0.5, 0.25, 0.25])
        return igr.RunData(run, [0, 0, 0, 0, 1, 1], [0], f_true, igr.Support(bits, 0.0, 1.0))

    def test_igr_by_hand(self):
        rd = self._toy()
        A, B = igr.estimator('OUE', float(np.log(3.0)))
        self.assertAlmostEqual(A, 4.0)
        self.assertAlmostEqual(B, 1.0)
        m = rd.metrics(None)                         # all reports: C_0 = 4 of 6
        self.assertAlmostEqual(m['target_est'], 4 * 4 / 6 - 1)
        self.assertAlmostEqual(m['igr'], (4 * 4 / 6 - 1 - 0.5) / (2 / 6 * 1))
        o = rd.metrics(rd.labels == 1)               # genuine only: C_0 = 2 of 4
        self.assertAlmostEqual(o['target_est'], 4 * 2 / 4 - 1)
        self.assertAlmostEqual(o['f1'], 1.0)
        self.assertEqual((o['flagged'], o['kept'], o['TP'], o['FP']), (2, 4, 2, 0))
        bad = rd.metrics(np.array([1, 0, 0, 1, 0, 0], bool))   # removes the two genuine target supporters
        self.assertAlmostEqual(bad['target_est'], 4 * 2 / 4 - 1)
        self.assertEqual((bad['TP'], bad['FP'], bad['FN']), (0, 2, 2))
        self.assertTrue(np.isnan(rd.metrics(np.ones(6, bool))['igr']))

    def test_cache_roundtrip(self):
        rd = self._toy()
        rd.feats = np.arange(24, dtype=np.float64).reshape(6, 4)
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, 'c', 'r.npz')
            rd.save(p)
            r2 = igr.RunData.load(p)
        for mask in (None, rd.labels == 1, np.array([1, 0, 1, 0, 0, 1], bool)):
            self.assertEqual(rd.metrics(mask), r2.metrics(mask))
        np.testing.assert_array_equal(r2.feats, rd.feats)
        self.assertEqual(r2.run, json.loads(json.dumps(rd.run)))


class TestFeaturePath(unittest.TestCase):
    """Our model must see the same input main.py builds (load_npy_dataset + apply_feature_set_v3)."""

    def test_matches_main_py(self):
        from attacker_detector.data import load_npy_dataset
        from attacker_detector.data.dataset import apply_feature_set_v3
        a, b = _run('OUE', n=1500, seed=5), _run('OUE', n=1500, eps=0.5, seed=6)
        sa, sb = _simulate_with_features(a), _simulate_with_features(b)
        with tempfile.TemporaryDirectory() as t:
            d = os.path.join(t, 'OUE')
            write_data_dir(d, [(a, sa[6], sa[3]), (b, sb[6], sb[3])])
            ds = load_npy_dataset(d)
            apply_feature_set_v3(ds)
            main_rows = ds.features[:len(sa[3])]
            norm = igr.train_v3_norm(d, os.path.join(t, 'norm.json'))
            self.assertTrue(os.path.exists(os.path.join(t, 'norm.json')))
            ours = igr.model_input(igr.v3_features(sa[6], a, sa[3]), norm)
        np.testing.assert_allclose(ours, main_rows, atol=2e-4)


class TestCLI(unittest.TestCase):
    """norms -> run (resumable) -> collect -> wsp on a fake root, all in a temp results folder."""

    @classmethod
    def setUpClass(cls):
        import torch
        from attacker_detector.wsp.adapters import build_freq_model
        cls.tmp = tempfile.mkdtemp()
        cls.results = os.path.join(cls.tmp, 'results')
        cls.data = os.path.join(cls.tmp, 'output', 'fake')
        a, b = _run('OUE', n=1200, seed=7), _run('OUE', n=1200, eps=0.5, seed=8)
        sa, sb = _simulate_with_features(a), _simulate_with_features(b)
        write_data_dir(os.path.join(cls.data, 'OUE'), [(a, sa[6], sa[3]), (b, sb[6], sb[3])])
        runs = [{'dataset_type': 'fire', 'protocol': 'OUE', 'epsilon': e, 'attacker_ratio': 0.1, 'target_set_size': 10,
                 'splits': 4, 'train': 1, 'holdout': 1} for e in (1.0, 0.5)]
        design = {'design': 'oat', 'attack': 'mga-a', 'feature_set': 'v2', 'block_candidates': 16,
                  'protocols': ['OUE'], 'datasets': ['fire'], 'n': 1200, 'runs': runs,
                  'defaults': {'epsilon': 1.0, 'attacker_ratio': 0.1, 'target_set_size': 10, 'splits': 4}}
        with open(os.path.join(cls.data, 'OUE', 'design.json'), 'w') as f:
            json.dump(design, f)
        spec = igr_script().ft_spec()
        torch.manual_seed(0)
        for sub in ('in_dist/Fire_OUE', 'cross/OUE_Fire_test'):
            d = os.path.join(cls.results, 'fake_v3f', sub)
            os.makedirs(d)
            torch.save({'model_state_dict': build_freq_model(spec, 4).state_dict()}, os.path.join(d, 'model.pt'))
            np.savez(os.path.join(d, 'test_predictions.npz'), prob=np.r_[np.full(900, 0.1), np.full(300, 0.9)],
                     label=np.r_[np.zeros(1080), np.ones(120)].astype(np.int8), epsilon=np.full(1200, 1.0, np.float32),
                     attacker_ratio=np.full(1200, 0.1, np.float32), splits=np.full(1200, 4, np.int8),
                     dataset=np.array(['fire'] * 1200))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cli(self, *args):
        env = dict(os.environ, IGR_RESULTS=self.results, MPLCONFIGDIR=self.tmp, PYTHONDONTWRITEBYTECODE='1')
        p = subprocess.run([sys.executable, SCRIPT, *args, '--roots', 'fake_v3f', '--data-map', f'fake_v3f={self.data}',
                            '--protocols', 'OUE', '--processors', '1', '--device', 'cpu'],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        return p.stdout

    def test_pipeline(self):
        import pandas as pd
        igr_dir = os.path.join(self.results, 'fake_v3f', 'igr')
        out = self.cli('plan')
        self.assertIn('2 runs', out)
        self.cli('norms')
        self.assertTrue(os.path.exists(os.path.join(igr_dir, 'v3_norm_OUE.json')))
        self.assertTrue(os.path.exists(os.path.join(igr_dir, 'design_OUE.json')))
        out = self.cli('run', '--cache', 'all')
        self.assertIn('ours_cross', out)
        files = sorted(os.listdir(os.path.join(igr_dir, 'runs', 'Fire_OUE')))
        self.assertEqual(files, ['eps0.5_b0.1_r10_s4_rep1.csv', 'eps1_b0.1_r10_s4_rep1.csv'])
        self.assertEqual(len(os.listdir(os.path.join(igr_dir, 'cache', 'Fire_OUE'))), 2)
        self.assertEqual(self.cli('run', '--cache', 'all').count('(done)'), 2)     # resumable

        self.cli('collect')
        df = pd.read_csv(os.path.join(igr_dir, 'igr_runs.csv'))
        self.assertEqual(sorted(df.detector.unique()),
                         sorted(['no_defense', 'oracle', 'ours_in_dist', 'ours_cross', 'fiad', 'diffstats']))
        self.assertEqual(len(df), 12)
        self.assertTrue(np.isfinite(df.igr).all())
        o = df[df.detector == 'oracle']
        self.assertTrue((o.f1 == 1.0).all() and (o.FP == 0).all())
        self.assertTrue((df[df.detector == 'no_defense'].flagged == 0).all())
        # the attack shows: without a defence the targets gain far more than the noise floor
        nd = df[df.detector == 'no_defense'].igr.to_numpy()
        self.assertTrue((nd > o.igr.to_numpy()).all())
        chk = pd.read_csv(os.path.join(igr_dir, 'igr_f1_check.csv'))
        self.assertIn('f1_stored', chk.columns)
        self.assertEqual(len(chk.dropna(subset=['f1_stored'])), 2)                  # eps 1.0 for both models

        # wsp step: models scored on the cached users, nothing simulated again
        mod = igr_script()
        mod.RESULTS = self.results
        ck = os.path.join(self.results, 'fake_v3f', 'in_dist', 'Fire_OUE', 'model.pt')
        mod.wsp_models = lambda: ('fake_v3f', {'Fire_OUE': [('ours_in_dist', 'stage1', 43, '', ck, 'in_dist/Fire_OUE'),
                                                            ('ours_in_dist', 'wsp', 43, 25, ck, 'in_dist/Fire_OUE')]})
        self.assertEqual(mod.cmd_wsp(Namespace(processors=1, device='cpu')), 0)
        w = pd.read_csv(os.path.join(igr_dir, 'runs', 'Fire_OUE', 'eps1_b0.1_r10_s4_rep1.wsp.csv'))
        self.assertEqual(list(w.variant), ['stage1', 'wsp'])
        base = df[(df.detector == 'ours_in_dist') & (df.epsilon == 1.0)].iloc[0]
        for col in ('igr', 'flagged', 'f1', 'mse'):                                   # same model, same users
            self.assertAlmostEqual(float(w[col].iloc[0]), float(base[col]), places=9)


_MOD = None


def igr_script():
    global _MOD
    if _MOD is None:
        cwd = os.getcwd()
        spec = importlib.util.spec_from_file_location('run_igr', SCRIPT)
        _MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MOD)
        os.chdir(cwd)
    return _MOD


if __name__ == '__main__':
    unittest.main()
