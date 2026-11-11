"""Reference-beta variant (attacker_detector/reference_beta.py, main.py --source-beta): split by run
at one source beta, exact budgets, optimization-only scaler, validation-selected frozen threshold.

    python -m unittest discover -s tests/reference_beta -t .
"""
import json
import math
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

from attacker_detector import reference_beta as rb  # noqa: E402
from attacker_detector.data.dataset import NpyDataset, load_npy_dataset, apply_feature_set_v3  # noqa: E402
from attacker_detector.data.generators import FEATURE_NAMES_V2  # noqa: E402


# ------------------------------------------------------------------------------------------
# synthetic one-at-a-time datasets laid out like generate_dataset.py writes them
# ------------------------------------------------------------------------------------------

def _run_rows(rng, n, beta, shift=2.0):
    """v2-like features of one complete run: genuine first, then round(beta * n) attackers."""
    n_att = int(round(beta * n))
    y = np.r_[np.zeros(n - n_att), np.ones(n_att)].astype(np.float32)
    F = rng.normal(size=(n, len(FEATURE_NAMES_V2)))
    F[:, FEATURE_NAMES_V2.index('num_ones_scaled')] = 1.0 + 0.05 * rng.normal(size=n)
    for name in ('block_projection', 'block_overlap'):
        F[n - n_att:, FEATURE_NAMES_V2.index(name)] += shift
    return F, y


def write_dir(path, runs, *, n, role='dataset', offset=0, seed=42, defaults_beta=0.05, rng=None):
    """runs: list of (dataset, eps, beta, splits, n_train, n_holdout). Each replicate = one run."""
    rng = rng or np.random.default_rng(0)
    os.makedirs(path, exist_ok=True)
    feats, labels, conf, design_runs = [], [], [], []
    exp_id = 0
    for ds, eps, beta, splits, n_train, n_hold in runs:
        for rep in range(n_train + n_hold):
            F, y = _run_rows(rng, n, beta)
            c = np.empty((n, 10), dtype=object)
            c[:, :7] = ['10', repr(float(beta)), 'OUE', str(splits), repr(float(eps)), ds, str(n)]
            c[:, 7] = str(exp_id)
            c[:, 8] = np.arange(n).astype(str)
            c[:, 9] = str(int(rep >= n_train))
            feats.append(F), labels.append(y), conf.append(c)
            exp_id += 1
        design_runs.append({'dataset_type': ds, 'protocol': 'OUE', 'epsilon': eps, 'attacker_ratio': beta,
                            'target_set_size': 10, 'splits': splits, 'train': n_train, 'holdout': n_hold})
    np.save(os.path.join(path, 'features.npy'), np.vstack(feats).astype(np.float32))
    np.save(os.path.join(path, 'labels.npy'), np.concatenate(labels))
    np.save(os.path.join(path, 'config.npy'), np.vstack(conf).astype(str))
    k = len(FEATURE_NAMES_V2)
    with open(os.path.join(path, 'norm_stats.json'), 'w') as f:
        json.dump({'feature_names': list(FEATURE_NAMES_V2), 'mean': [0.0] * k, 'std': [1.0] * k}, f)
    design = {'design': 'oat', 'attack': 'mga-a', 'feature_set': 'v2', 'block_candidates': 16, 'n': n,
              'protocols': ['OUE'], 'datasets': sorted({r[0] for r in runs}), 'runs': design_runs,
              'defaults': {'epsilon': 1.0, 'attacker_ratio': defaults_beta, 'target_set_size': 10, 'splits': 4},
              'sweeps': {'epsilon': sorted({r[1] for r in runs}), 'attacker_ratio': sorted({r[2] for r in runs}),
                         'target_set_size': [10], 'splits': sorted({r[3] for r in runs})},
              'role': role, 'replicate_offset': offset, 'seed': seed}
    with open(os.path.join(path, 'design.json'), 'w') as f:
        json.dump(design, f)


MAIN_RUNS = [  # (dataset, eps, beta, splits, train runs, held-out runs)
    (ds, eps, beta, sp, tr, 1)
    for ds in ('zipf', 'emoji', 'fire')
    for eps, beta, sp, tr in ((1.0, 0.05, 4, 2), (0.5, 0.05, 4, 2), (1.0, 0.05, 2, 2),
                              (1.0, 0.1, 4, 1), (1.0, 0.2, 4, 1))
]
VAL_RUNS = [(ds, eps, 0.05, sp, 1, 0) for ds in ('zipf', 'emoji', 'fire')
            for eps, sp in ((1.0, 4), (0.5, 4), (1.0, 2))]


def _load_raw(path, protocol='OUE', dataset=None):
    d = load_npy_dataset(path)
    apply_feature_set_v3(d, normalize=False)
    idx = np.where(d.protocols == protocol)[0]
    if dataset:
        idx = idx[d.dataset_types[idx] == dataset]
    d.features, d.labels, d.config = d.features[idx], d.labels[idx], d.config[idx]
    return d


class _Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.main = os.path.join(cls.tmp, 'main')
        cls.val = os.path.join(cls.tmp, 'val')
        write_dir(cls.main, MAIN_RUNS, n=2000)
        write_dir(cls.val, VAL_RUNS, n=2000, role='validation', offset=1000, rng=np.random.default_rng(1))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------
# split
# ------------------------------------------------------------------------------------------

class TestSplit(_Data):
    def _split(self, train, test, beta=0.05, total=1000, main=None, val=None):
        ds = _load_raw(main or self.main, dataset=None if len(train) > 1 else train[0])
        dv = _load_raw(val or self.val, dataset=None if len(train) > 1 else train[0])
        raw = np.asarray(ds.features).copy()
        sp = rb.build_split(ds, dv, source_beta=beta, train_datasets=train, test_dataset=test,
                            train_samples=total, val_size=0.1, test_samples=500, seed=7)
        return ds, dv, raw, sp

    def test_only_source_beta_and_exact_budget(self):
        ds, dv, raw, sp = self._split(['fire'], 'fire')
        self.assertTrue(np.allclose(ds.attacker_ratios[sp['train_indices']], 0.05))
        self.assertTrue(np.allclose(dv.attacker_ratios[sp['val_indices']], 0.05))
        self.assertEqual((len(sp['train_indices']), int(sp['y_train'].sum())), (900, 450))
        self.assertEqual((len(sp['val_indices']), int(sp['y_val'].sum())), (100, 50))
        m = sp['manifest']
        self.assertEqual(len(m['eligible_configs']), 3)                      # 0.1 / 0.2 points excluded
        self.assertEqual(m['excluded_training_betas'], [0.1, 0.2])
        self.assertEqual([c['optimization_rows_per_class'] for c in m['eligible_configs']], [150, 150, 150])
        self.assertEqual(sum(c['validation_rows_per_class'] for c in m['eligible_configs']), 50)
        self.assertEqual(m['test_betas'], [0.05, 0.1, 0.2])                   # the test sweep is untouched
        self.assertEqual(len(np.unique(sp['train_indices'])), 900)           # no row twice

    def test_remainder_allocation(self):
        self.assertEqual(rb.allocate_evenly(25_000, 9), [2778] * 7 + [2777] * 2)
        for total, k in ((225_000, 9), (225_000, 18), (25_000, 27), (7, 3)):
            parts = rb.allocate_evenly(total, k)
            self.assertEqual(sum(parts), total)
            self.assertLessEqual(max(parts) - min(parts), 1)
        ds, _, _, sp = self._split(['fire'], 'fire', total=1020)       # 51 validation rows/class over 3 configs
        q = [c['validation_rows_per_class'] for c in sp['manifest']['eligible_configs']]
        self.assertEqual(sum(q) * 2, len(sp['val_indices']))

    def test_runs_do_not_overlap(self):
        ds, dv, _, sp = self._split(['fire'], 'fire')
        opt_runs = set(ds.experiment_ids[sp['train_indices']])
        test_runs = set(ds.experiment_ids[sp['sens_indices']])
        self.assertFalse(opt_runs & test_runs)
        self.assertTrue((ds.holdout[sp['train_indices']] == 0).all())
        self.assertTrue((ds.holdout[sp['sens_indices']] == 1).all())
        # validation rows come from the other directory; its design proves its runs are new replicates
        rb._check_val_design(ds.design, dv.design, 0.05)
        bad = dict(dv.design, replicate_offset=2)
        with self.assertRaises(ValueError):
            rb._check_val_design(ds.design, bad, 0.05)

    def test_fixed_share_groups(self):
        for beta in (0.10, 0.15):
            d = os.path.join(self.tmp, f'b{beta}')
            write_dir(d + '_main', [(ds, eps, beta, 4, 1, 1) for ds in ('zipf', 'emoji', 'fire')
                                    for eps in (0.5, 1.0, 2.0)], n=2000, defaults_beta=beta)
            write_dir(d + '_val', [(ds, eps, beta, 4, 1, 0) for ds in ('zipf', 'emoji', 'fire')
                                   for eps in (0.5, 1.0, 2.0)], n=2000, role='validation', offset=1000,
                      defaults_beta=beta)
            ds, dv, _, sp = self._split(['zipf'], 'zipf', beta=beta, main=d + '_main', val=d + '_val')
            self.assertTrue(np.allclose(ds.attacker_ratios[sp['train_indices']], beta))
            self.assertTrue(np.allclose(dv.attacker_ratios[sp['val_indices']], beta))
            self.assertEqual(sp['manifest']['source_beta'], beta)

    def test_cross_dataset_never_touches_target(self):
        ds, dv, _, sp = self._split(['zipf', 'emoji'], 'fire')
        self.assertTrue(set(ds.dataset_types[sp['train_indices']]) <= {'zipf', 'emoji'})
        self.assertTrue(set(dv.dataset_types[sp['val_indices']]) <= {'zipf', 'emoji'})
        self.assertEqual(set(ds.dataset_types[sp['sens_indices']]), {'fire'})
        self.assertEqual(len(sp['manifest']['eligible_configs']), 6)
        self.assertEqual(len(sp['train_indices']), 900)

    def test_scaler_from_optimization_rows_only(self):
        ds, dv, raw, sp = self._split(['fire'], 'fire')
        mu = raw[sp['train_indices']].mean(axis=0)
        sd = raw[sp['train_indices']].std(axis=0)
        np.testing.assert_allclose(sp['scaler']['mean'], mu, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(sp['scaler']['std'], sd, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(sp['X_train'].mean(axis=0), 0.0, atol=1e-5)
        self.assertGreater(np.abs(raw.mean(axis=0) - mu).max(), 1e-6)       # all rows would give another fit

    def test_missing_quota_is_an_error(self):
        with self.assertRaises(ValueError):
            self._split(['fire'], 'fire', total=100_000)


# ------------------------------------------------------------------------------------------
# threshold
# ------------------------------------------------------------------------------------------

class TestThreshold(unittest.TestCase):
    def test_weights(self):
        y = np.r_[np.zeros(50), np.ones(50)]
        for beta, (wp, wn) in ((0.05, (0.10, 1.90)), (0.10, (0.20, 1.80)), (0.15, (0.30, 1.70))):
            a, b, s = rb.validation_weights(y, beta)
            self.assertAlmostEqual(a, wp)
            self.assertAlmostEqual(b, wn)
            self.assertEqual(s, 0.5)

    def test_weighted_f1_by_hand(self):
        # 2 attackers, 2 genuine, beta 0.05 -> attacker weight 0.1, genuine 1.9
        s = np.array([3.0, 1.0, 2.0, -1.0])
        y = np.array([1, 1, 0, 0])
        m = rb.weighted_metrics(s, y, 1.5, 0.1, 1.9)            # flags rows 0 and 2: TP 0.1, FP 1.9, FN 0.1
        self.assertAlmostEqual(m['f1'], 0.2 / (0.2 + 1.9 + 0.1))
        self.assertAlmostEqual(m['precision'], 0.1 / 2.0)
        self.assertAlmostEqual(m['recall'], 0.5)
        self.assertAlmostEqual(m['fpr'], 0.5)
        sel = rb.select_threshold(s, y, 0.05)                    # best: flag only row 0 -> F1 = 0.2/0.3
        self.assertAlmostEqual(sel['validation_f1'], 0.2 / 0.3)
        self.assertEqual(sel['threshold'], 2.5)                  # midpoint of 3 and 2
        self.assertEqual(sel['candidates'], 5)                   # none, 4 distinct boundaries (last = all)

    def test_ties_and_extremes(self):
        s = np.array([1.0, 1.0, 1.0, 0.0])
        y = np.array([1, 0, 1, 0])
        sel = rb.select_threshold(s, y, 0.5)
        flagged = rb.decide(s, sel['threshold'])
        self.assertTrue(flagged[:3].all() or not flagged[:3].any())    # tied rows move together
        allpos = rb.select_threshold(np.ones(4), np.array([1, 1, 1, 0]), 0.75)
        self.assertEqual(allpos['threshold'], -np.inf)                  # flag all is the only useful point
        self.assertEqual(allpos['operating_point'], 'flag_all')
        noinfo = rb.select_threshold(np.zeros(4), np.array([1, 0, 0, 0]), 0.0001)
        self.assertIn(noinfo['threshold'], (np.inf, -np.inf))
        none = rb.weighted_metrics(np.zeros(3), np.array([1, 0, 0]), np.inf)
        self.assertEqual((none['f1'], none['precision'], none['recall']), (0.0, 0.0, 0.0))

    def test_ties_on_f1_choose_higher_threshold(self):
        s = np.array([4.0, 3.0, 2.0, 1.0])
        y = np.array([1, 0, 1, 0])
        # flag {4}: F1 = 2/(2+0+1)=0.667; flag {4,3,2}: TP 2 FP 1 -> 4/5 = 0.8 -> wins; check exact
        sel = rb.select_threshold(s, y, 0.5)
        self.assertAlmostEqual(sel['validation_f1'], 0.8)
        s2 = np.array([2.0, 1.0])
        y2 = np.array([1, 1])                                    # flag-all and flag {2,1} identical
        self.assertEqual(rb.select_threshold(np.r_[s2, 0.0], np.r_[y2, 0], 0.5)['threshold'], 0.5)

    def test_save_load_preserves_predictions(self):
        rng = np.random.default_rng(3)
        s = rng.normal(size=500)
        y = (rng.random(500) < 0.5).astype(int)
        sel = rb.select_threshold(s, y, 0.05)
        with tempfile.TemporaryDirectory() as t:
            rb.save_threshold(t, dict(sel, score_scale=rb.SCORE_SCALE, decision_operator=rb.DECISION_OPERATOR))
            rec = rb.load_threshold(t)
        test = rng.normal(size=10_000)
        self.assertTrue(np.array_equal(rb.decide(test, rec['threshold']), rb.decide(test, sel['threshold'])))
        for v in (np.inf, -np.inf):
            with tempfile.TemporaryDirectory() as t:
                rb.save_threshold(t, {'threshold': v, 'score_scale': 'logit', 'decision_operator': '>'})
                self.assertEqual(rb.load_threshold(t)['threshold'], v)

    def test_missing_threshold_fails(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                rb.load_threshold(t)


# ------------------------------------------------------------------------------------------
# utility-selected threshold
# ------------------------------------------------------------------------------------------

class TestUtility(unittest.TestCase):
    def _run(self, rng, n=400, beta=0.1):
        from attacker_detector.utility_threshold import UtilityRun
        y = (np.arange(n) >= n - int(beta * n)).astype(np.int8)
        ts = rng.integers(0, 4, size=n).astype(float) + 6.0 * y      # attackers support many targets
        run = {'attacker_ratio': beta, 'target_set_size': 10, 'dataset_type': 'fire', 'epsilon': 1.0,
               'splits': 4, 'replicate': 2000, 'seed': 1}
        return UtilityRun(run, np.zeros((n, 4)), y, ts, A=2.0, B=0.1, true_t=0.6)

    def test_curve_is_exact(self):
        from attacker_detector.utility_threshold import igr_curve
        rng = np.random.default_rng(0)
        u = self._run(rng)
        s = np.round(rng.normal(size=u.n), 1)                        # many ties
        thr = np.r_[np.inf, np.unique(s)[::-1] - 1e-9, -np.inf, 0.0, 0.05]
        got = igr_curve(s, u, thr)
        for t, g in zip(thr, got):
            want = u.igr(rb.decide(s, t))
            self.assertTrue((np.isnan(g) and np.isnan(want)) or abs(g - want) < 1e-12, (t, g, want))

    def test_curve_matches_rundata_on_a_simulated_run(self):
        from attacker_detector import igr
        from attacker_detector.utility_threshold import UtilityRun, igr_curve
        from config import DATASET_CONFIGS
        run = dict(root='x', dataset_type='fire', protocol='OUE', epsilon=1.0, attacker_ratio=0.1,
                   target_set_size=10, splits=4, replicate=1, n=1500,
                   domain=DATASET_CONFIGS['fire']['domain'], h_ao=0, seed=3)
        X, targets, support, labels, _, _ = igr.simulate(run, 1)
        rd = igr.RunData(run, labels, targets, igr.genuine_frequency(X, labels, run['domain']),
                         igr.Support.from_dense(support), np.zeros((len(labels), 4)))
        u = UtilityRun.from_rundata(rd)
        s = np.random.default_rng(1).normal(size=u.n) + 2 * rd.labels
        for t in (np.inf, 2.0, 1.0, 0.0):
            self.assertAlmostEqual(igr_curve(s, u, [t])[0], rd.metrics(rb.decide(s, t))['igr'], places=10)

    def test_select(self):
        from attacker_detector.utility_threshold import select, summarize, candidate_thresholds
        rng = np.random.default_rng(2)
        runs = [self._run(rng) for _ in range(3)]
        scores = [rng.normal(size=u.n) + 3 * u.labels for u in runs]
        thr, obj, cand = select(scores, runs)
        self.assertNotEqual(thr, -np.inf)                             # flag-all is never a candidate
        self.assertTrue(np.isfinite(cand[1:]).all())
        self.assertAlmostEqual(obj.min(), summarize(thr, scores, runs)['mean_igr2'], places=12)
        self.assertLessEqual(obj.min(), summarize(np.inf, scores, runs)['mean_igr2'])
        self.assertEqual(len(cand), len(np.unique(np.concatenate(scores))))
        # nobody's removal moves the estimate -> every candidate ties -> the highest (flag none)
        from attacker_detector.utility_threshold import UtilityRun
        flat = UtilityRun(runs[0].run, np.zeros((50, 4)), np.r_[np.zeros(45), np.ones(5)], np.full(50, 2.0),
                          A=1.0, B=0.0, true_t=1.5)
        self.assertEqual(select([np.linspace(0, 1, 50)], [flat])[0], np.inf)

    def test_cache_roundtrip(self):
        from attacker_detector.utility_threshold import UtilityRun
        u = self._run(np.random.default_rng(4))
        with tempfile.TemporaryDirectory() as t:
            u.save(os.path.join(t, 'a', 'r.npz'))
            v = UtilityRun.load(os.path.join(t, 'a', 'r.npz'))
        f = np.r_[np.zeros(u.n - 30, bool), np.ones(30, bool)]
        self.assertAlmostEqual(u.igr(f), v.igr(f), places=12)
        self.assertEqual(v.run['replicate'], 2000)


# ------------------------------------------------------------------------------------------
# main.py end to end
# ------------------------------------------------------------------------------------------

class TestCLI(_Data):
    FT = ['-m', 'ft_transformer', '--train-samples', '1000', '--test-samples', '1500', '--val-size', '0.10',
          '--balanced-batches', '--k-folds', '0', '--epochs', '2', '--patience', '2', '--batch-size', '256',
          '--lr', '0.0005', '--d-token', '16', '--n-heads', '2', '--n-layers', '1', '--seed', '42',
          '--no-plot', '--save-predictions', '--feature-set', 'v3', '--protocol', 'OUE']

    def cli(self, *extra, ok=True):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=self.tmp, CUDA_VISIBLE_DEVICES='')
        p = subprocess.run([sys.executable, os.path.join(ROOT, 'main.py'), '-d', self.main, '--val-data', self.val,
                            '--source-beta', '0.05', *self.FT, *extra], capture_output=True, text=True, env=env,
                           cwd=ROOT)
        if ok:
            self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        return p

    def test_train_freeze_evaluate(self):
        import pandas as pd
        out = os.path.join(self.tmp, 'res', 'in_dist', 'Fire_OUE')
        self.cli('--dataset', 'fire', '-o', out)
        rec = rb.load_threshold(out)
        man = json.load(open(os.path.join(out, 'split_manifest.json')))
        self.assertEqual((rec['source_beta'], rec['threshold_reference_beta'], rec['training_class_share']),
                         (0.05, 0.05, 0.5))
        self.assertEqual(man['budget']['optimization_rows'], 900)
        self.assertEqual(man['budget']['validation_rows'], 100)
        self.assertEqual(rec['model']['checkpoint_sha256'], rb.file_sha256(os.path.join(out, 'model.pt')))
        # the sampled test and the full held-out evaluation carry the same stored threshold
        tr = pd.read_csv(os.path.join(out, 'test_results.csv'))
        sens = pd.read_csv(os.path.join(out, 'sensitivity_test_results.csv'))
        self.assertTrue(np.allclose(tr.Threshold, rec['threshold']) and np.allclose(sens.Threshold, rec['threshold']))
        z = np.load(os.path.join(out, 'test_predictions.npz'))
        self.assertEqual(float(z['threshold']), rec['threshold'])
        pred, y = rb.decide(z['logit'], rec['threshold']), z['label'].astype(bool)
        f1 = 2 * (pred & y).sum() / (2 * (pred & y).sum() + (pred & ~y).sum() + (~pred & y).sum())
        self.assertAlmostEqual(float(sens[sens.Parameter_Type == 'overall'].F1_Score.iloc[0]), f1, places=6)
        # the test beta of a row never enters the decision: same scores, other metadata -> same flags
        sub = np.isclose(z['attacker_ratio'], 0.2)
        self.assertTrue(sub.any())
        self.assertTrue(np.array_equal(rb.decide(z['logit'][sub], rec['threshold']), pred[sub]))
        # resume: a complete run is not redone; another setting in the same folder is refused
        self.assertIn('nothing to do', self.cli('--dataset', 'fire', '-o', out).stdout)
        p = self.cli('--dataset', 'fire', '-o', out, '--threshold-reference-beta', '0.1', ok=False)
        self.assertNotEqual(p.returncode, 0)
        # re-evaluation from the stored scores reproduces the run's own outputs at threshold.json exactly
        cwd = os.getcwd()
        try:
            os.chdir(ROOT)
            rb.evaluate_threshold(out, 'threshold.json', 'check')
        finally:
            os.chdir(cwd)
        for name in ('test_results', 'sensitivity_test_results', 'sensitivity_test_balanced'):
            a = pd.read_csv(os.path.join(out, f'{name}.csv'))
            b = pd.read_csv(os.path.join(out, f'{name}_check.csv'))
            cols = [c for c in a.columns if c in ('Parameter_Type', 'Value', 'Accuracy', 'Precision', 'Recall',
                                                  'F1_Score', 'Count', 'Attackers', 'n_samples', 'Threshold')]
            pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-12, obj=name)
        # ... and at another frozen threshold file it stamps and applies that one
        u = json.load(open(os.path.join(out, 'threshold.json')))
        u['threshold'] = rec['threshold'] + 1.0
        json.dump(u, open(os.path.join(out, 'threshold_utility.json'), 'w'))
        cwd = os.getcwd()
        try:
            os.chdir(ROOT)
            rb.evaluate_threshold(out, 'threshold_utility.json', 'utility')
        finally:
            os.chdir(cwd)
        su = pd.read_csv(os.path.join(out, 'sensitivity_test_results_utility.csv'))
        self.assertTrue(np.allclose(su.Threshold, rec['threshold'] + 1.0))
        pu = rb.decide(z['logit'], rec['threshold'] + 1.0)
        f1u = 2 * (pu & y).sum() / max(1, 2 * (pu & y).sum() + (pu & ~y).sum() + (~pu & y).sum())
        self.assertAlmostEqual(float(su[su.Parameter_Type == 'overall'].F1_Score.iloc[0]), f1u, places=6)
        # a folder of the standard pipeline is never overwritten
        old = os.path.join(self.tmp, 'old'); os.makedirs(old); open(os.path.join(old, 'model.pt'), 'w').close()
        self.assertNotEqual(self.cli('--dataset', 'fire', '-o', old, ok=False).returncode, 0)

    def test_cross(self):
        out = os.path.join(self.tmp, 'res', 'cross', 'OUE_Fire_test')
        self.cli('--training-method', 'cross', '--train-dataset', 'zipf', 'emoji', '--test-dataset', 'fire', '-o', out)
        man = json.load(open(os.path.join(out, 'split_manifest.json')))
        self.assertEqual({c['dataset_type'] for c in man['eligible_configs']}, {'zipf', 'emoji'})
        self.assertEqual({r['dataset_type'] for r in man['test_runs']}, {'fire'})
        self.assertTrue(math.isfinite(rb.load_threshold(out)['threshold'])
                        or rb.load_threshold(out)['operating_point'] != 'interior')

    def test_argument_checks(self):
        p = self.cli('--dataset', 'fire', '-o', os.path.join(self.tmp, 'x'), '--k-folds', '3', ok=False)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('--k-folds 0', p.stderr)


if __name__ == '__main__':
    unittest.main()
