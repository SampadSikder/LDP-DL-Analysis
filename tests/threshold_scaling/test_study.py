"""Threshold and scaling study (attacker_detector/threshold_scaling_study.py): a miniature of the whole
pipeline on tiny simulated runs, with stand-in "reused" checkpoints.

    python -m unittest discover -s tests/threshold_scaling -t .
"""
import json
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from attacker_detector import igr
from attacker_detector import reference_beta as refbeta
from attacker_detector import threshold_scaling_study as st
from attacker_detector import threshold_scaling_table as tb

N, ROOT_NAME, P = 3000, 'mini', 'OUE'
FIRST = list(st.TRAINED)[:3]                    # what the run directory starts with
ADDED = 'fixed_beta__four_features_unscaled'    # added to the finished run directory by `extend`
POOLED = ['pooled_beta__four_features_scaled__control', 'pooled_beta__four_features_unscaled']   # added too
HP = [v for v, spec in st.TRAINED.items() if 'scope' in spec]      # the small hyperparameter test, added too


def _design():
    runs = [{'dataset_type': d, 'protocol': P, 'epsilon': e, 'attacker_ratio': b, 'target_set_size': 10,
             'splits': 4, 'train': 1, 'holdout': 1}
            for d in st.DATASETS for e, b in ((0.5, 0.05), (1.0, 0.05), (1.0, 0.1))]
    return {'design': 'oat', 'attack': 'mga-a', 'feature_set': 'v2', 'block_candidates': 16, 'protocols': [P],
            'datasets': list(st.DATASETS), 'n': N, 'runs': runs,
            'defaults': {'epsilon': 1.0, 'attacker_ratio': 0.05, 'target_set_size': 10, 'splits': 4}}


def _fake_model(mdir, fixed_beta):
    os.makedirs(mdir, exist_ok=True)
    torch.manual_seed(len(mdir))
    torch.save({'model_state_dict': igr.build_ft(4).state_dict()}, os.path.join(mdir, 'model.pt'))
    if not fixed_beta:
        return
    sha = refbeta.file_sha256(os.path.join(mdir, 'model.pt'))
    refbeta._write_json(os.path.join(mdir, 'scaler.json'), {'feature_names': st.V3, 'mean': [0.0] * 4, 'std': [1.0] * 4})
    for name, thr in (('threshold.json', 0.1), ('threshold_utility.json', 0.3)):
        refbeta._write_json(os.path.join(mdir, name), {
            'model': {'checkpoint_sha256': sha}, 'threshold': thr, 'protocol': P,
            'score_scale': refbeta.SCORE_SCALE, 'decision_operator': refbeta.DECISION_OPERATOR})


class StudyPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        res = os.path.join(cls.tmp, 'results')
        ref = os.path.join(cls.tmp, 'reference')
        os.makedirs(os.path.join(ref, 'designs', ROOT_NAME))
        with open(os.path.join(ref, 'designs', ROOT_NAME, f'design_{P}.json'), 'w') as f:
            json.dump(_design(), f)
        pooled, bfix = os.path.join(res, 'archive', f'{ROOT_NAME}_v3f'), os.path.join(res, f'{ROOT_NAME}_bfix')
        for D in st.DATASETS:
            for kind in ('in_dist', 'cross'):
                _fake_model(os.path.join(pooled, kind, st.cell(D, P, kind)), False)
                _fake_model(os.path.join(bfix, kind, st.cell(D, P, kind)), True)
        os.makedirs(os.path.join(pooled, 'igr'))
        refbeta._write_json(os.path.join(pooled, 'igr', f'v3_norm_{P}.json'),
                            {'feature_names': st.V3, 'mean': [0.0] * 4, 'std': [1.0] * 4})
        cls.cfg_path = os.path.join(cls.tmp, 'override.json')
        with open(cls.cfg_path, 'w') as f:
            json.dump({'roots': {ROOT_NAME: {'attack': 'mga-a', 'source_beta': 0.05, 'protocols': [P]}},
                       'reference_dir': ref, 'trained_variants': FIRST,
                       'train': {'train_samples': 600, 'epochs': 2, 'patience': 2, 'batch_size': 64}}, f)
        cls.res, cls.run_dir = res, os.path.join(res, 'study_run')
        cls.base = ['--run-dir', cls.run_dir, '--processors', '1']
        cls.rc = st.main(['init', '--results', res, '--config', cls.cfg_path] + cls.base)
        cls.before = cls._listing(pooled) + cls._listing(bfix)
        for cmd in ('simulate', 'train', 'select', 'score', 'table'):
            cls.rc |= st.main([cmd] + cls.base)
        # a variant added afterwards: only its models are trained, everything is scored again
        scored = os.path.join(cls.run_dir, 'scores', ROOT_NAME, 'scored_runs.csv')
        cls.first_scores = pd.read_csv(scored, keep_default_na=False)
        cls.first_models = cls._listing(os.path.join(cls.run_dir, 'variants'))
        cls.first_runs = cls._listing(os.path.join(cls.run_dir, 'simulated_runs'))
        for cmd in ('extend', 'simulate', 'train', 'select', 'score', 'table'):
            cls.rc |= st.main([cmd] + cls.base)
        # a baseline added afterwards: its own simulation of the held-out runs, then the tables again
        cls.second_scores = pd.read_csv(scored, keep_default_na=False)
        cls.second_runs = cls._listing(os.path.join(cls.run_dir, 'simulated_runs'))
        cls.rc |= st.main(['baselines', '--methods', 'fiad', 'diffstats'] + cls.base)
        cls.rc |= st.main(['table'] + cls.base)
        cls.after = cls._listing(pooled) + cls._listing(bfix)
        cls.cfg = st.load_cfg(cls.run_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def _listing(d):
        return sorted((os.path.join(r, f), os.path.getsize(os.path.join(r, f)), os.path.getmtime(os.path.join(r, f)))
                      for r, _, fs in os.walk(d) for f in fs)

    def test_all_stages_succeed_and_reused_results_untouched(self):
        self.assertEqual(self.rc, 0)
        self.assertEqual(self.before, self.after)

    def test_init_refuses_an_existing_directory(self):
        with self.assertRaises(SystemExit):
            st.main(['init', '--results', self.res, '--config', self.cfg_path] + self.base)

    def test_pooled_checkpoints_of_a_protocol_can_come_from_another_root(self):
        other, ref = 'mini2', os.path.join(self.tmp, 'reference')
        os.makedirs(os.path.join(ref, 'designs', other))
        shutil.copyfile(os.path.join(ref, 'designs', ROOT_NAME, f'design_{P}.json'),
                        os.path.join(ref, 'designs', other, f'design_{P}.json'))
        for D in st.DATASETS:
            for kind in ('in_dist', 'cross'):
                _fake_model(os.path.join(self.res, f'{other}_bfix', kind, st.cell(D, P, kind)), True)
        cfg_path, run_dir = os.path.join(self.tmp, 'override2.json'), os.path.join(self.res, 'study_run2')
        with open(cfg_path, 'w') as f:
            json.dump({'roots': {other: {'attack': 'mga-a', 'source_beta': 0.05, 'protocols': [P],
                                         'pooled_from': {P: ROOT_NAME}}}, 'reference_dir': ref}, f)
        self.assertEqual(st.main(['init', '--results', self.res, '--config', cfg_path, '--run-dir', run_dir]), 0)
        cfg = st.load_cfg(run_dir)
        pooled = os.path.join(self.res, 'archive', f'{ROOT_NAME}_v3f')
        self.assertEqual(cfg['roots'][other]['pooled_dirs'], {P: pooled})
        self.assertIn(f'{ROOT_NAME}_v3f', cfg['roots'][other]['checkpoints'][f'pooled/in_dist/{st.cell("fire", P, "in_dist")}']['path'])
        self.assertTrue(cfg['roots'][other]['v3_norm'][P]['from'].startswith(pooled))
        det = st.detectors(cfg, other, P, 'cross', 'fire')[0]
        self.assertEqual((det['model'], det['dir']), ('pooled_beta', os.path.join(pooled, 'cross', st.cell('fire', P, 'cross'))))

    def test_plan_roles(self):
        df = pd.DataFrame(st.plan(self.cfg))
        n = df.groupby('role').size()
        # per dataset: 3 configs, 2 of them at the source beta; the third trains the pooled-beta models
        self.assertEqual(n.to_dict(), {'heldout_test': 9, 'train_source': 6, 'validation': 6, 'utility_validation': 9,
                                       'train_other': 3, 'validation_other': 3})
        self.assertEqual(set(df[df.role.isin(['train_other', 'validation_other'])].attacker_ratio), {0.1})
        first = dict(self.cfg, trained_variants={v: st.TRAINED[v] for v in FIRST})      # before the extension
        self.assertEqual(len(st.plan(first)), 30)
        self.assertEqual(df.seed.nunique(), len(df))
        self.assertTrue((df[df.role == 'validation'].replicate == 1000).all())
        self.assertTrue((df[df.role == 'utility_validation'].replicate == 2000).all())
        self.assertTrue((df[df.role == 'heldout_test'].replicate == 1).all())
        self.assertEqual(set(df[df.role == 'train_source'].attacker_ratio), {0.05})

    def test_unscaled_features_are_the_scaled_ones_before_the_per_run_step(self):
        u = st.load_runs(self.cfg, ROOT_NAME, 'heldout_test', P, ['fire'])[0]
        for c in range(3):
            x = u.raw3[:, c]
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            want = (x - np.median(x)) / (iqr if iqr > 1e-9 else 1.0)
            np.testing.assert_allclose(u.feats[:, c], want, rtol=1e-3, atol=1e-3)
        self.assertEqual(u.inputs('raw', st.RAW3).shape, (N, 3))
        self.assertEqual(u.inputs('scaled', st.V3).shape, (N, 4))
        # four features unscaled: the three columns before the per-run step, excess_share unchanged
        mixed = u.inputs('raw', st.V3)
        np.testing.assert_array_equal(mixed[:, :3], u.raw3)
        np.testing.assert_array_equal(mixed[:, 3], u.feats[:, 3])
        self.assertIn('diffstats', u.flags)
        # per-target support: its row sums are the summed report IGR uses
        t = u.targets
        self.assertEqual(t['bits'].shape, (N, 10))
        np.testing.assert_allclose((t['lo'] + (t['hi'] - t['lo']) * t['bits']).sum(axis=1), u.target_sum)
        self.assertAlmostEqual(float(t['true'].sum()), u.true_t, places=12)

    def test_hyperparameter_test_variants(self):
        self.assertEqual(len(HP), 4)
        for variant in HP:
            spec = st.TRAINED[variant]
            d = st.variant_dir(self.cfg, variant, ROOT_NAME, 'in_dist', 'zipf', P)
            hp = st._read(os.path.join(d, 'variant.json'))['identity']['hyperparameters']
            want = {**self.cfg['train'], **spec['train']}
            self.assertEqual((hp['dropout'], hp['batch_size'], hp['d_token']),
                             (want['dropout'], want['batch_size'], want['d_token']))
            # one dataset, in-distribution only; the usual-settings partner sees the same rows
            self.assertFalse(os.path.exists(st.variant_dir(self.cfg, variant, ROOT_NAME, 'cross', 'zipf', P)))
            self.assertFalse(os.path.exists(st.variant_dir(self.cfg, variant, ROOT_NAME, 'in_dist', 'fire', P)))
            a, b = (st._read(os.path.join(st.variant_dir(self.cfg, v, ROOT_NAME, 'in_dist', 'zipf', P),
                                          'split_manifest.json'))['eligible_configs'] for v in (variant, spec['base_variant']))
            self.assertEqual(a, b)
            det = next(x for x in st.detectors(self.cfg, ROOT_NAME, P, 'in_dist', 'zipf') if x.get('variant') == variant)
            self.assertEqual(det['ft_spec']['d_token'], want['d_token'])
            u = st.load_runs(self.cfg, ROOT_NAME, 'heldout_test', P, ['zipf'])[0]
            self.assertEqual(st.Scorer(torch.device('cpu')).logits(det, u).shape, (N,))     # the checkpoint loads
        self.assertEqual({st.TRAINED[v]['train'].get('d_token') for v in HP}, {None, 128})
        self.assertEqual({st.TRAINED[v]['train'].get('batch_size') for v in HP}, {None, 64})
        hp_dir = os.path.join(self.run_dir, 'comparison', 'hyperparameters')
        pc = pd.read_csv(os.path.join(hp_dir, 'paired_overall.csv'))
        self.assertEqual(len(pc), len(tb.HP_PAIRS))            # one cell here: OUE, zipf, in-distribution
        self.assertEqual(set(pc.dataset_type), {'zipf'})
        self.assertEqual(set(pc.kind), {'in_dist'})
        self.assertTrue((pc.n_runs == 3).all())
        for f in ('overall.csv', 'by_test_beta.csv', 'paired_by_test_beta.csv', 'hyperparameters.md'):
            self.assertTrue(os.path.exists(os.path.join(hp_dir, f)), f)

    def test_trained_variants(self):
        for variant, spec in st.TRAINED.items():
            if 'scope' in spec:
                continue
            for kind, n_cfg in (('in_dist', 2), ('cross', 4)):
                d = st.variant_dir(self.cfg, variant, ROOT_NAME, kind, 'fire', P)
                man = st._read(os.path.join(d, 'split_manifest.json'))
                self.assertEqual(man['budget']['optimization_rows'], 540)
                self.assertEqual(man['budget']['optimization_attackers'], 270)
                self.assertEqual(man['budget']['validation_rows'], 60)
                pooled = spec.get('training_betas') == 'all'
                self.assertEqual(len(man['eligible_configs']), n_cfg * 3 // 2 if pooled else n_cfg)
                self.assertEqual(man['training_betas'], [0.05, 0.1] if pooled else [0.05])
                q = [c['optimization_rows_per_class'] for c in man['eligible_configs']]
                self.assertLessEqual(max(q) - min(q), 1)        # every config weighs the same
                self.assertEqual(st._read(os.path.join(d, 'scaler.json'))['feature_names'], spec['features'])
                rec = refbeta.load_threshold(d)
                self.assertEqual(rec['model']['checkpoint_sha256'], refbeta.file_sha256(os.path.join(d, 'model.pt')))
        # the variants draw the same rows: same seed, same quotas, same runs
        a, b = (st._read(os.path.join(st.variant_dir(self.cfg, v, ROOT_NAME, 'in_dist', 'zipf', P),
                                      'split_manifest.json'))['eligible_configs'] for v in list(st.TRAINED)[:2])
        self.assertEqual(a, b)

    def test_extend_adds_a_variant_and_keeps_what_was_finished(self):
        self.assertEqual(list(self.cfg['trained_variants']), FIRST + [ADDED] + POOLED + HP)
        self.assertEqual([e['variants'] for e in self.cfg['extensions']], [[ADDED] + POOLED + HP])
        now = dict((p, (s, m)) for p, s, m in self._listing(os.path.join(self.run_dir, 'variants')))
        for p, size, mtime in self.first_models:           # no finished model or threshold was written again
            self.assertEqual(now[p], (size, mtime), p)
        runs = dict((p, (s, m)) for p, s, m in self._listing(os.path.join(self.run_dir, 'simulated_runs')))
        for p, size, mtime in self.first_runs:             # no stored run was simulated again
            self.assertEqual(runs[p], (size, mtime), p)
        self.assertEqual(len(runs) - len(self.first_runs), 6)
        d = st.variant_dir(self.cfg, ADDED, ROOT_NAME, 'cross', 'fire', P)
        self.assertEqual(st._read(os.path.join(d, 'scaler.json'))['feature_names'], st.V3)
        self.assertEqual(st._read(os.path.join(d, 'variant.json'))['identity']['feature_source'], 'raw')
        self.assertTrue(os.path.exists(os.path.join(d, 'threshold_utility.json')))
        # the rows scored before the extension come out the same; the new model's rows are added
        df = pd.read_csv(os.path.join(self.run_dir, 'scores', ROOT_NAME, 'scored_runs.csv'), keep_default_na=False)
        key = ['dataset_type', 'epsilon', 'attacker_ratio', 'kind', 'model', 'threshold_rule']
        new = df.model.isin(['fixed_beta_4f_unscaled', 'pooled_beta_4f_scaled_control', 'pooled_beta_4f_unscaled']
                            + [st.TRAINED[v]['model'] for v in HP])
        old = df[~new].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(old, self.first_scores.sort_values(key).reset_index(drop=True))
        # runs x kinds x threshold rules: fixed beta 2; each pooled model p > 0.5, F1, utility, utility all-beta
        # plus the hyperparameter test on the 3 zipf runs, in-distribution: 2 fixed-beta models x 2 rules and
        # 2 pooled-beta models x 4 rules
        self.assertEqual(len(df) - len(old), 9 * 2 * (2 + 2 * 4) + 3 * (2 * 2 + 2 * 4))
        self.assertEqual(set(df[df.model == 'pooled_beta_4f_unscaled'].threshold_rule),
                         {'fixed_p0.5', 'f1', 'utility', 'utility_all_beta'})
        rec = st._read(os.path.join(st.variant_dir(self.cfg, POOLED[1], ROOT_NAME, 'cross', 'fire', P),
                                    st.ALL_BETA_DIR, 'threshold_utility.json'))
        self.assertEqual(rec['calibration_betas'], [0.05, 0.1])
        self.assertNotIn('fire', {r['dataset_type'] for r in rec['utility_validation_runs']})
        self.assertEqual(st.main(['extend'] + self.base), 0)          # nothing left to add
        self.assertEqual(len(st.load_cfg(self.run_dir)['extensions']), 1)
        with self.assertRaises(SystemExit):
            st.main(['extend', '--variants', 'no_such_variant'] + self.base)

    def test_utility_thresholds_use_calibration_runs_only(self):
        held = {int(r['seed']) for r in st.plan(self.cfg) if r['role'] == 'heldout_test'}
        for variant, betas in ((st.POOLED_UTILITY, [0.05]), (st.POOLED_UTILITY_ALL, [0.05, 0.1]),
                               ('fixed_beta__three_features_unscaled', [0.05])):
            for kind in ('in_dist', 'cross'):
                rec = st._read(os.path.join(st.variant_dir(self.cfg, variant, ROOT_NAME, kind, 'emoji', P),
                                            'threshold_utility.json'))
                self.assertEqual(rec['calibration_betas'], betas)
                self.assertFalse(rec['uses_test_information'])
                self.assertFalse(held & {int(r['seed']) for r in rec['utility_validation_runs']})
                self.assertTrue(all(r['replicate'] == 2000 for r in rec['utility_validation_runs']))
                if kind == 'cross':
                    self.assertNotIn('emoji', {r['dataset_type'] for r in rec['utility_validation_runs']})
        pooled = st._read(os.path.join(st.variant_dir(self.cfg, st.POOLED_UTILITY, ROOT_NAME, 'in_dist', 'emoji', P),
                                       'threshold_utility.json'))
        self.assertIn('archive', pooled['model']['checkpoint'])
        u = st.load_runs(self.cfg, ROOT_NAME, 'heldout_test', P, ['emoji'])[0]
        det = st.detectors(self.cfg, ROOT_NAME, P, 'in_dist', 'emoji')[0]
        with self.assertRaises(RuntimeError):
            st._select(det, [u], st.Scorer(torch.device('cpu')), os.path.join(self.tmp, 'nope'), {'variant': 'x'})

    def test_baseline_added_after_the_simulation(self):
        held = [r for r in st.plan(self.cfg) if r['role'] == 'heldout_test']
        files = [st.baseline_path(self.cfg, r) for r in held]
        self.assertTrue(all(os.path.exists(f) for f in files))
        # nothing of the finished run directory changed: stored runs and scored rows
        self.assertEqual(self.second_runs, self._listing(os.path.join(self.run_dir, 'simulated_runs')))
        scored = pd.read_csv(os.path.join(self.run_dir, 'scores', ROOT_NAME, 'scored_runs.csv'), keep_default_na=False)
        pd.testing.assert_frame_equal(scored, self.second_scores)
        self.assertNotIn('fiad', set(scored.model))
        rows = pd.read_csv(os.path.join(self.run_dir, 'baseline_runs', 'baseline_runs.csv'), keep_default_na=False)
        self.assertEqual(rows.groupby('model').size().to_dict(),
                         {m: len(held) for m in ('no_defense', 'oracle', 'fiad', 'diffstats')})
        o = rows[rows.model == 'oracle']
        self.assertTrue((o.FP == 0).all() and (o.FN == 0).all())
        self.assertTrue((rows[rows.model == 'no_defense'].flagged == 0).all())
        # the same design and seed as the study's simulation of the run: the same users are fake and the
        # attack is present in both. Whether the LDP noise comes out identical depends on how the
        # generator's worker processes inherit the seeded RNG state, so the two IGR values may be equal
        # (they are on Linux) or differ; the table stage measures the difference and says which it was.
        u = st.load_runs(self.cfg, ROOT_NAME, 'heldout_test', P, ['zipf'])[0]
        mine = rows[(rows.seed == u.run['seed']) & (rows.model == 'oracle')].iloc[0]
        self.assertEqual(int(mine.TP), int(u.labels.sum()))
        self.assertTrue(np.isfinite(float(mine.igr)))
        none_row = rows[(rows.seed == u.run['seed']) & (rows.model == 'no_defense')].iloc[0]
        self.assertGreater(float(none_row.igr), 0)
        self.assertGreater(u.igr(np.zeros(u.n, dtype=bool)), 0)
        # a finished run is skipped
        before = self._listing(os.path.join(self.run_dir, 'baseline_runs', ROOT_NAME))
        self.assertEqual(st.main(['baselines', '--methods', 'fiad'] + self.base), 0)
        self.assertEqual(before, self._listing(os.path.join(self.run_dir, 'baseline_runs', ROOT_NAME)))
        # FIAD is in the tables (it was not flagged at simulation time); Diffstats keeps its own rows
        side = os.path.join(self.run_dir, 'comparison', 'side_by_side')
        tab = pd.read_csv(os.path.join(side, 'side_by_side_by_test_beta.csv'))
        for kind in ('in_dist', 'cross'):
            f = tab[(tab.kind == kind) & (tab.variant == 'FIAD')]
            self.assertEqual(sorted(f.test_beta), [0.05, 0.1])
            self.assertEqual(int(f.n_runs.sum()), len(held))
            self.assertEqual(len(tab[(tab.kind == kind) & (tab.variant == 'Diffstats')]), 2)
            self.assertIn('FIAD', set(pd.read_csv(os.path.join(self.run_dir, 'comparison', kind,
                                                               'sweep_test_beta.csv')).variant))
        chk = pd.read_csv(os.path.join(side, 'baseline_resimulation_check.csv'))
        self.assertEqual(set(chk.model), {'no_defense', 'oracle', 'diffstats'})
        self.assertEqual(len(chk), 3 * len(held))
        self.assertIn('fiad was added after the simulation', open(os.path.join(side, 'limitations.md')).read())

    def test_scores_and_tables(self):
        df = pd.read_csv(os.path.join(self.run_dir, 'scores', ROOT_NAME, 'scored_runs.csv'), keep_default_na=False)
        per_run = df.groupby(['dataset_type', 'epsilon', 'attacker_ratio']).size()
        self.assertEqual(len(per_run), 9)
        # 3 reference rows + 2 kinds x (pooled 4 rules + existing 2 + 4 fixed-beta trained x 2 + 2 pooled trained x 4)
        # zipf runs also carry the 12 rows of the hyperparameter test
        zipf = per_run.index.get_level_values('dataset_type') == 'zipf'
        self.assertTrue((per_run[~zipf] == 3 + 2 * 22).all())
        self.assertTrue((per_run[zipf] == 3 + 2 * 22 + 12).all())
        u = st.load_runs(self.cfg, ROOT_NAME, 'heldout_test', P, ['zipf'])[0]
        row = df[(df.dataset_type == 'zipf') & (df.model == 'oracle') & (df.seed == u.run['seed'])].iloc[0]
        self.assertEqual(int(row.TP), int(u.labels.sum()))
        self.assertEqual(int(row.FP), 0)
        self.assertTrue(math.isclose(float(row.igr), u.igr(u.labels == 1), rel_tol=1e-9, abs_tol=1e-12))
        side = os.path.join(self.run_dir, 'comparison', 'side_by_side')
        tab = pd.read_csv(os.path.join(side, 'side_by_side_by_test_beta.csv'))
        self.assertEqual(sorted(tab.test_beta.unique()), [0.05, 0.1])
        for kind in ('in_dist', 'cross'):
            got = set(tab[(tab.kind == kind) & (tab.test_beta == 0.1)].variant)
            self.assertTrue(set(tb.MAIN) <= got, set(tb.MAIN) - got)
            self.assertTrue(os.path.exists(os.path.join(self.run_dir, 'comparison', kind, 'sweep_test_beta.csv')))
        pc = pd.read_csv(os.path.join(side, 'paired_comparisons.csv'))
        self.assertEqual(pc.comparison.nunique(), len(tb.PAIRS))
        self.assertEqual(len(pc), len(tb.PAIRS) * 2 * 2)          # comparisons x kinds x test betas
        for f in ('limitations.md', 'paired_comparisons.md', 'side_by_side_by_test_beta.md', 'replication_check.csv',
                  'replication_check_pooled.csv'):
            self.assertTrue(os.path.exists(os.path.join(side, f)), f)


if __name__ == '__main__':
    unittest.main()
