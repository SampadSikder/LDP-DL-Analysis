"""MSE threshold study (attacker_detector/mse_threshold_study.py): a miniature of the whole pipeline. The
source is a miniature threshold and scaling study (tests/threshold_scaling), built here first.

    python -m unittest discover -s tests/mse_threshold -t .
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from attacker_detector import mse_threshold_study as ms
from attacker_detector import mse_threshold_table as mt
from attacker_detector import reference_beta as refbeta
from attacker_detector import threshold_scaling_study as st
from tests.threshold_scaling import test_study as base

N, ROOT_NAME, P = base.N, base.ROOT_NAME, base.P


class MseStudyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        res, ref = os.path.join(cls.tmp, 'results'), os.path.join(cls.tmp, 'reference')
        os.makedirs(os.path.join(ref, 'designs', ROOT_NAME))
        with open(os.path.join(ref, 'designs', ROOT_NAME, f'design_{P}.json'), 'w') as f:
            json.dump(base._design(), f)
        pooled, bfix = os.path.join(res, 'archive', f'{ROOT_NAME}_v3f'), os.path.join(res, f'{ROOT_NAME}_bfix')
        for D in st.DATASETS:
            for kind in ('in_dist', 'cross'):
                base._fake_model(os.path.join(pooled, kind, st.cell(D, P, kind)), False)
                base._fake_model(os.path.join(bfix, kind, st.cell(D, P, kind)), True)
        os.makedirs(os.path.join(pooled, 'igr'))
        refbeta._write_json(os.path.join(pooled, 'igr', f'v3_norm_{P}.json'),
                            {'feature_names': st.V3, 'mean': [0.0] * 4, 'std': [1.0] * 4})
        override = os.path.join(cls.tmp, 'override.json')
        with open(override, 'w') as f:
            json.dump({'roots': {ROOT_NAME: {'attack': 'mga-a', 'source_beta': 0.05, 'protocols': [P]}},
                       'reference_dir': ref,
                       'train': {'train_samples': 600, 'epochs': 2, 'patience': 2, 'batch_size': 64}}, f)
        # the source: a finished threshold and scaling study with every trained variant
        cls.res, cls.source = res, os.path.join(res, 'source_study')
        sbase = ['--run-dir', cls.source, '--processors', '1']
        cls.rc = st.main(['init', '--results', res, '--config', override] + sbase)
        for cmd in ('simulate', 'train', 'select'):
            cls.rc |= st.main([cmd] + sbase)
        cls.before = cls._listing(cls.source) + cls._listing(pooled) + cls._listing(bfix)
        cls.run_dir = os.path.join(res, 'mse_run')
        cls.base = ['--run-dir', cls.run_dir, '--processors', '1']
        cls.rc |= ms.main(['init', '--results', res, '--source', cls.source] + cls.base)
        for cmd in ('simulate', 'select', 'score', 'table'):
            cls.rc |= ms.main([cmd] + cls.base)
        cls.after = cls._listing(cls.source) + cls._listing(pooled) + cls._listing(bfix)
        cls.cfg = ms.load_cfg(cls.run_dir)
        cls.src = ms.source_cfg(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def _listing(d):
        return sorted((os.path.join(r, f), os.path.getsize(os.path.join(r, f)), os.path.getmtime(os.path.join(r, f)))
                      for r, _, fs in os.walk(d) for f in fs)

    def test_all_stages_succeed_and_the_source_is_untouched(self):
        self.assertEqual(self.rc, 0)
        self.assertEqual(self.before, self.after)

    def test_init_refuses_an_existing_directory_and_an_unfinished_source(self):
        with self.assertRaises(SystemExit):
            ms.main(['init', '--results', self.res, '--source', self.source] + self.base)
        v = os.path.join(st.variant_dir(self.src, 'pooled_beta__four_features_unscaled', ROOT_NAME, 'cross', 'fire', P),
                         'variant.json')
        os.rename(v, v + '.hidden')
        try:
            with self.assertRaises(SystemExit):
                ms.main(['init', '--results', self.res, '--source', self.source,
                         '--run-dir', os.path.join(self.res, 'mse_run2')])
            self.assertFalse(os.path.exists(os.path.join(self.res, 'mse_run2')))
        finally:
            os.rename(v + '.hidden', v)

    def test_plan_has_calibration_and_held_out_runs_only(self):
        df = pd.DataFrame(ms.plan(self.cfg))
        self.assertEqual(df.groupby('role').size().to_dict(), {'heldout_test': 9, 'utility_validation': 9})
        # the same runs (seeds) as the source study planned for these roles
        src = pd.DataFrame(st.plan(self.src))
        src = src[src.role.isin(ms.ROLES)]
        self.assertEqual(sorted(df.seed), sorted(src.seed))
        self.assertTrue((df[df.role == 'utility_validation'].replicate == 2000).all())

    def test_target_item_error(self):
        u = st.load_runs(self.cfg, ROOT_NAME, 'heldout_test', P, ['zipf'])[0]
        rng = np.random.default_rng(0)
        for flagged in (np.zeros(u.n, bool), u.labels == 1, rng.random(u.n) < 0.3):
            est = ms.target_estimate(u, flagged)
            # the summed estimate is what IGR is made of
            self.assertAlmostEqual((est.sum() - u.targets['true'].sum()) / (u.beta * u.r), u.igr(flagged), places=9)
            vals = u.targets['lo'] + (u.targets['hi'] - u.targets['lo']) * u.targets['bits'][~flagged]
            want = np.mean((u.A * vals.sum(axis=0) / (~flagged).sum() - u.B - u.targets['true']) ** 2)
            self.assertAlmostEqual(ms.mse_target(u, flagged), want, places=15)
        self.assertTrue(np.isnan(ms.mse_target(u, np.ones(u.n, bool))))
        # removing exactly the fake users leaves less error on the targets than removing nobody
        self.assertLess(ms.mse_target(u, u.labels == 1), ms.mse_target(u, np.zeros(u.n, bool)))

    def test_selection_is_the_exact_minimum_over_all_candidates(self):
        det = next(d for d in ms.detectors(self.cfg, self.src, ROOT_NAME, P, 'in_dist', 'fire')
                   if d.get('variant') == 'fixed_beta__four_features_unscaled')
        runs = st.load_runs(self.cfg, ROOT_NAME, 'utility_validation', P, ['fire'], betas=[0.05])
        scorer = st.Scorer(torch.device('cpu'))
        scores = [scorer.logits(det, u) for u in runs]
        thr, obj, cand = ms.select(scores, runs)
        for i in list(range(0, len(cand), 37)) + [len(cand) - 1]:
            flags = [refbeta.decide(s, cand[i]) for s in scores]
            if any(f.all() for f in flags):      # a cut-off that empties a run can never be selected
                self.assertEqual((obj['mse'][i], obj['igr'][i]), (np.inf, np.inf))
                continue
            self.assertAlmostEqual(obj['mse'][i], np.mean([ms.mse_target(u, f) for u, f in zip(runs, flags)]), places=15)
            self.assertAlmostEqual(obj['igr'][i], np.mean([u.igr(f) ** 2 for u, f in zip(runs, flags)]), places=9)
        self.assertEqual(thr['mse'], cand[int(np.argmin(obj['mse']))])
        self.assertEqual(thr['igr'], cand[int(np.argmin(obj['igr']))])
        self.assertEqual(det['thresholds']['mse'], thr['mse'])           # what the select stage froze
        self.assertEqual(det['thresholds']['igr'], thr['igr'])
        with self.assertRaises(RuntimeError):                            # a test run cannot select a threshold
            held = st.load_runs(self.cfg, ROOT_NAME, 'heldout_test', P, ['fire'])[:1]
            ms._select(det, held, scorer, os.path.join(self.tmp, 'nope'), {'variant': 'x'})

    def test_thresholds_use_calibration_runs_only(self):
        held = {int(r['seed']) for r in ms.plan(self.cfg) if r['role'] == 'heldout_test'}
        for variant, cals in ms.MSE_MODELS.items():
            for kind in ('in_dist', 'cross'):
                for cal in ('source_beta', 'all_beta'):
                    d = ms.threshold_dir(self.cfg, variant, ROOT_NAME, kind, 'emoji', P, cal)
                    for objective in ms.OBJECTIVES:
                        path = os.path.join(d, f'threshold_{objective}.json')
                        self.assertEqual(os.path.exists(path), cal in cals, path)
                        if cal not in cals:
                            continue
                        rec = st._read(path)
                        self.assertEqual(rec['calibration_betas'], [0.05] if cal == 'source_beta' else [0.05, 0.1])
                        self.assertFalse(rec['uses_test_information'])
                        self.assertFalse(held & {int(r['seed']) for r in rec['utility_validation_runs']})
                        self.assertTrue(all(r['replicate'] == 2000 for r in rec['utility_validation_runs']))
                        if kind == 'cross':
                            self.assertNotIn('emoji', {r['dataset_type'] for r in rec['utility_validation_runs']})
                        self.assertIn('source_study', rec['model']['checkpoint'])
                        # the MSE threshold is at least as good on its own objective as the IGR threshold
                        other = st._read(os.path.join(d, 'threshold_igr.json'))
                        mine = st._read(os.path.join(d, 'threshold_mse.json'))
                        self.assertLessEqual(mine['validation_mean_mse_target'], other['validation_mean_mse_target'])
                        self.assertLessEqual(other['validation_mean_igr2'], mine['validation_mean_igr2'])

    def test_scores_and_tables(self):
        df = pd.read_csv(os.path.join(self.run_dir, 'scores', ROOT_NAME, 'scored_runs.csv'), keep_default_na=False)
        self.assertEqual(len(df.groupby(['dataset_type', 'epsilon', 'attacker_ratio'])), 9)
        for model, rules in (('fixed_beta_4f_unscaled', {'f1', 'utility', 'mse', 'igr'}),
                             ('fixed_beta_3f_unscaled', {'f1', 'utility', 'mse', 'igr'}),
                             ('pooled_beta_4f_unscaled', {'fixed_p0.5', 'f1', 'utility', 'utility_all_beta', 'mse', 'igr',
                                                          'mse_all_beta', 'igr_all_beta'}),
                             ('pooled_beta', {'fixed_p0.5', 'fixed_prior', 'utility', 'utility_all_beta'})):
            for kind in ('in_dist', 'cross'):
                self.assertEqual(set(df[(df.model == model) & (df.kind == kind)].threshold_rule), rules, model)
        u = st.load_runs(self.cfg, ROOT_NAME, 'heldout_test', P, ['zipf'])[0]
        row = df[(df.dataset_type == 'zipf') & (df.model == 'oracle') & (df.seed == u.run['seed'])].iloc[0]
        self.assertAlmostEqual(float(row.mse_target), ms.mse_target(u, u.labels == 1), places=15)
        self.assertEqual(int(row.FP), 0)
        side = os.path.join(self.run_dir, 'comparison', 'side_by_side')
        tab = pd.read_csv(os.path.join(side, 'side_by_side_by_test_beta.csv'))
        self.assertEqual(sorted(tab.test_beta.unique()), [0.05, 0.1])
        best = pd.read_csv(os.path.join(side, 'best_igr_reference.csv'))
        self.assertEqual(len(best), 2)                                    # one per kind (one attack, one protocol)
        self.assertTrue(best.variant.str.endswith(mt.SOURCE).all())
        for kind in ('in_dist', 'cross'):
            got = tab[(tab.kind == kind) & (tab.test_beta == 0.1)]
            self.assertTrue(set(mt.MAIN + [mt.BEST, 'Diffstats', 'No removal']) <= set(got.variant))
            # the reference row repeats the chosen version of the source study
            name = best[best.kind == kind].variant.iloc[0]
            self.assertEqual(got[got.variant == mt.BEST].reference_version.iloc[0], name)
            self.assertAlmostEqual(got[got.variant == mt.BEST].mse_target.iloc[0],
                                   got[got.variant == name].mse_target.iloc[0], places=15)
            self.assertTrue(os.path.exists(os.path.join(self.run_dir, 'comparison', kind, 'sweep_test_beta.csv')))
        pc = pd.read_csv(os.path.join(side, 'paired_comparisons.csv'))
        self.assertEqual(pc.comparison.nunique(), len(mt.PAIRS))
        self.assertEqual(len(pc), len(mt.PAIRS) * 2 * 2)                 # comparisons x kinds x test betas
        for f in ('limitations.md', 'paired_comparisons.md', 'side_by_side_by_test_beta.md'):
            self.assertTrue(os.path.exists(os.path.join(side, f)), f)


if __name__ == '__main__':
    unittest.main()
