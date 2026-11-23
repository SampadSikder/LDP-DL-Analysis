"""Threshold and scaling study on the frequency oracles (main sweeps, source beta 0.05).

Two experiments, scored together on the same simulated held-out users:

1. Pooled beta with a utility-selected threshold. The earlier pooled-beta checkpoints (results/archive/
   <root>_v3f) are reused unchanged: same four v3 features, same global z-score, same weights. Only the
   cut-off changes: the mean-IGR^2 threshold of attacker_detector/utility_threshold.py, selected on
   utility-validation runs. Nothing of a test run (labels, beta, targets, IGR) enters the selection.
2. Fixed beta without the per-run median/IQR scaling. Four models are trained the fixed-beta way
   (attacker_detector/reference_beta.py: source-beta runs only, 50/50 rows, same budget, seed, quotas):
     fixed_beta__four_features_scaled__control   block_projection, block_overlap, one_deviation
                                                 (per-run median/IQR) + excess_share
     fixed_beta__three_features_scaled           the same without excess_share
     fixed_beta__three_features_unscaled         the same three, per-run median/IQR NOT applied
     fixed_beta__four_features_unscaled          the same three without per-run median/IQR + excess_share
                                                 (added later; `extend` adds it to an existing run directory)
   Each then gets one global z-score fitted on its optimization rows (fixed constants, the same for every
   run), its F1-selected threshold and its own utility threshold.
3. Pooled beta without the per-run median/IQR scaling (added later). Two models are trained by the same
   procedure on the training runs of EVERY config of the sweep (every beta; equal rows per config, 50/50):
     pooled_beta__four_features_scaled__control  the four v3 features (per-run median/IQR on three)
     pooled_beta__four_features_unscaled         the same three before the per-run median/IQR + excess_share
   They are scored at p > 0.5 (the reused pooled model's fixed rule), at their F1-selected threshold and
   at utility thresholds calibrated at the source beta and on every config.

The server no longer holds the main-sweep datasets, and datasets never stored reports. So every run the
study needs is simulated once from the saved design (attacker_detector/igr.simulate, seeded like the
datasets: same true items and targets, fresh LDP noise) and kept compactly (StudyRun):
  train_source        replicate 0.. of every config at the source beta   (what the datasets trained on)
  validation          replicate 1000                                      (early stopping, F1 threshold)
  utility_validation  replicate 2000                                      (utility threshold)
  heldout_test        the dataset's held-out replicate of every config    (all test betas)
  train_other, validation_other   the same two roles for the configs at the other betas; only planned
                      when a pooled-beta model is trained here
Every model and every threshold is scored on the same heldout_test users, next to no removal, perfect
removal and Diffstats.

Everything is written below one new run directory; existing result folders are only read.
Driver: scripts/run_threshold_scaling_study.sh.
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from . import igr
from . import reference_beta as refbeta
from . import utility_threshold as ut

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MARKER = 'study_config.json'
REFERENCE = os.path.join(ROOT, 'scripts', 'threshold_scaling_study', 'reference')
V3 = ['block_projection', 'block_overlap', 'one_deviation', 'excess_share']
RAW3 = ['block_projection', 'block_overlap', 'one_deviation']
ROLES = ('train_source', 'validation', 'utility_validation', 'heldout_test', 'train_other', 'validation_other')
DATASETS = ('zipf', 'emoji', 'fire')

TRAINED = {
    'fixed_beta__four_features_scaled__control': {
        'model': 'fixed_beta_4f_scaled_control', 'source': 'scaled', 'features': V3,
        'per_run_scaling': 'median/IQR of the run on block_projection, block_overlap, one_deviation'},
    'fixed_beta__three_features_scaled': {
        'model': 'fixed_beta_3f_scaled', 'source': 'scaled', 'features': RAW3,
        'per_run_scaling': 'median/IQR of the run on all three features'},
    'fixed_beta__three_features_unscaled': {
        'model': 'fixed_beta_3f_unscaled', 'source': 'raw', 'features': RAW3,
        'per_run_scaling': 'none (extractor values as they are)'},
    'fixed_beta__four_features_unscaled': {
        'model': 'fixed_beta_4f_unscaled', 'source': 'raw', 'features': V3,
        'per_run_scaling': 'none on block_projection, block_overlap, one_deviation (extractor values as they '
                           'are); excess_share as in the four-feature models (it never had the median/IQR step)'},
    # trained on every config of the sweep (all betas), like the reused pooled-beta checkpoints
    'pooled_beta__four_features_scaled__control': {
        'model': 'pooled_beta_4f_scaled_control', 'source': 'scaled', 'features': V3, 'training_betas': 'all',
        'per_run_scaling': 'median/IQR of the run on block_projection, block_overlap, one_deviation'},
    'pooled_beta__four_features_unscaled': {
        'model': 'pooled_beta_4f_unscaled', 'source': 'raw', 'features': V3, 'training_betas': 'all',
        'per_run_scaling': 'none on block_projection, block_overlap, one_deviation (extractor values as they '
                           'are); excess_share as in the four-feature models (it never had the median/IQR step)'},
}
ALL_BETA_DIR = 'all_beta_calibration'      # second utility threshold of a pooled model trained here


def pooled_training(cfg):
    """True when a trained variant uses the training runs of every beta (they are then planned)."""
    return any(v.get('training_betas') == 'all' for v in cfg['trained_variants'].values())
POOLED_UTILITY = 'pooled_beta__utility_threshold'
POOLED_UTILITY_ALL = 'pooled_beta__utility_threshold__all_beta_calibration'

DEFAULT_ROOTS = {       # data root -> the main sweeps (HST_Server has no APA: its APA runs are MGA-A runs)
    # pooled_from: the MGA-A pooled HST_Server models of the results are the ones trained in the APA root
    # (same grid, MGA-A, every fake user draws its own target subset). The oat_v3_n500k training data of
    # HST_Server came from the earlier generator, where all fake users shared one subset; models trained
    # on it may still lie in <results>/oat_v3_n500k_v3f and are not used.
    'oat_v3_n500k': {'attack': 'mga-a', 'source_beta': 0.05,
                     'protocols': ['OUE', 'HST_User', 'OLH_Server', 'HST_Server'],
                     'pooled_from': {'HST_Server': 'oat_apa_n500k'}},
    'oat_apa_n500k': {'attack': 'apa', 'source_beta': 0.05, 'protocols': ['OUE', 'HST_User', 'OLH_Server']},
}
# the fixed-beta runs' settings (scripts/run_beta_fixed.sh FT array)
TRAIN = {'train_samples': 500_000, 'val_size': 0.10, 'seed': 42, 'epochs': 50, 'patience': 10,
         'batch_size': 256, 'lr': 0.0005, 'dropout': 0.2, 'balanced_batches': True, **igr.FT_SPEC}
VAL_OFFSET = 1000       # reference_beta.validation_generation_args


def say(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def _read(path):
    with open(path) as f:
        return json.load(f)


def cell(dataset, protocol, kind='in_dist'):
    D = dataset.capitalize()
    return f'{D}_{protocol}' if kind == 'in_dist' else f'{protocol}_{D}_test'


def train_datasets(kind, dataset):
    return [dataset] if kind == 'in_dist' else [d for d in DATASETS if d != dataset]


# ------------------------------------------------------------------------------------------
# compact run store
# ------------------------------------------------------------------------------------------

class StudyRun(ut.UtilityRun):
    """UtilityRun (v3 features, labels, each user's summed report on the targets) plus the three block /
    count features before the per-run median/IQR and, on held-out runs, the baselines' flags."""

    def __init__(self, run, feats, labels, target_sum, A, B, true_t, raw3, flags=None):
        super().__init__(run, feats, labels, target_sum, A, B, true_t)
        self.raw3 = np.asarray(raw3, dtype=np.float64)
        self.flags = dict(flags or {})

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp.npz'
        np.savez(tmp, feats=self.feats.astype(np.float32), raw3=self.raw3.astype(np.float32),
                 labels=self.labels, target_sum=self.target_sum, A=self.A, B=self.B, true_t=self.true_t,
                 run=json.dumps(self.run, default=refbeta._jsonable),
                 **{f'flag_{k}': np.packbits(np.asarray(v, dtype=bool)) for k, v in self.flags.items()})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            n = len(z['labels'])
            flags = {k[5:]: np.unpackbits(z[k], count=n).astype(bool) for k in z.files if k.startswith('flag_')}
            return cls(json.loads(str(z['run'])), z['feats'].astype(np.float64), z['labels'], z['target_sum'],
                       float(z['A']), float(z['B']), float(z['true_t']), z['raw3'].astype(np.float64), flags)

    def inputs(self, source, features):
        """Model input columns. 'raw': the three block / count features before the per-run median/IQR;
        excess_share has no such step and is the same column in both sources."""
        if source == 'raw':
            return np.column_stack([self.raw3[:, RAW3.index(c)] if c in RAW3 else self.feats[:, V3.index(c)]
                                    for c in features])
        return self.feats[:, [V3.index(c) for c in features]]

    def metrics(self, flagged):
        f = np.asarray(flagged, dtype=bool)
        y = self.labels.astype(bool)
        tp, fp, fn = int((f & y).sum()), int((f & ~y).sum()), int((~f & y).sum())
        tn = self.n - tp - fp - fn
        g = self.igr(f)
        return {'flagged': tp + fp, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'precision': tp / (tp + fp) if tp + fp else 0.0, 'recall': tp / (tp + fn) if tp + fn else 0.0,
                'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                'fpr': fp / (fp + tn) if fp + tn else 0.0, 'igr': g, 'igr2': g * g}


def simulate_run(run, design, processors=4, baselines=()):
    """One complete LDP run -> StudyRun. The scaled features go through igr.v3_features (main.py's path);
    the unscaled ones are the same extractor columns before that transform."""
    from .data.generators import extract_user_level_features_diffstats_style
    from .data.generators.features import feature_names_for
    X, targets, support, labels, estimate, one_list = igr.simulate(run, processors)
    n, d = len(labels), int(run['domain'])
    sup = igr.Support.from_dense(support)
    v2 = extract_user_level_features_diffstats_style(
        support_list=support, one_list=one_list, epsilon=float(run['epsilon']), protocol=run['protocol'],
        domain=d, n=n, feature_set=design['feature_set'], block_candidates=design['block_candidates'])
    del support
    names = feature_names_for(design['feature_set'])
    raw3 = np.asarray(v2, dtype=np.float64)[:, [names.index(c) for c in RAW3]]
    feats = igr.v3_features(v2, run, labels)
    rd = igr.RunData(run, labels, targets, igr.genuine_frequency(X, labels, d), sup, feats)
    dev = np.abs(rd.estimate() * n - estimate).max() / max(1.0, np.abs(estimate).max())
    if dev > 1e-6:
        raise RuntimeError(f'estimator does not reproduce the generator estimate (rel. dev {dev:.2e})')
    u = ut.UtilityRun.from_rundata(rd)
    flags = {}
    if baselines:
        from .baselines import BASELINES
        for m in baselines:
            try:
                mask = np.zeros(n, dtype=bool)
                mask[np.asarray(BASELINES[m](sup.bits, one_list, run, processors), dtype=np.int64)] = True
                flags[m] = mask
            except Exception as e:          # e.g. the upstream Diffstats bin crash: keep the run, no row
                say(f'    [WARN] {m} failed on {igr.run_tag(run)}: {e}')
    return StudyRun(run, feats, labels, u.target_sum, u.A, u.B, u.true_t, raw3, flags)


# ------------------------------------------------------------------------------------------
# configuration: one new run directory, existing results only read
# ------------------------------------------------------------------------------------------

def load_cfg(run_dir):
    path = os.path.join(run_dir, MARKER)
    if not os.path.exists(path):
        raise SystemExit(f'{run_dir} is not a run directory of this study (no {MARKER}); run init first')
    cfg = _read(path)
    cfg['run_dir'] = run_dir
    return cfg


def _first_dir(*paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


def _git():
    try:
        rev = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True).stdout.strip()
        return {'commit': rev, 'uncommitted_changes': bool(dirty)}
    except OSError:
        return {'commit': None}


def pooled_dir(root_cfg, protocol):
    """Results directory of the pooled-beta models of one protocol (another root's for `pooled_from`)."""
    return (root_cfg.get('pooled_dirs') or {}).get(protocol) or root_cfg.get('pooled_dir')


def reused_models(root_cfg, protocols):
    """(kind, dataset, protocol, 'pooled' | 'fixed_beta', model dir) of every reused checkpoint."""
    out = []
    for P in protocols:
        for D in DATASETS:
            for kind in ('in_dist', 'cross'):
                for which, base in (('pooled', pooled_dir(root_cfg, P)), ('fixed_beta', root_cfg.get('fixed_beta_dir'))):
                    if base:
                        out.append((kind, D, P, which, os.path.join(base, kind, cell(D, P, kind))))
    return out


def cmd_init(args):
    """Create the run directory (refused if it exists), resolve and verify everything that is reused."""
    run_dir = args.run_dir
    if os.path.exists(run_dir):
        raise SystemExit(f'{run_dir} already exists; every invocation writes to a new directory '
                         f'(continue an interrupted one with its own stages, not init)')
    results = os.path.abspath(args.results)
    if os.path.commonpath([os.path.abspath(run_dir), results]) != results:
        raise SystemExit(f'the run directory must be below {results}')
    override = _read(args.config) if args.config else {}
    roots_spec = override.get('roots') or {r: DEFAULT_ROOTS[r] for r in (args.roots or DEFAULT_ROOTS)}
    reference = override.get('reference_dir', REFERENCE)
    ref = _read(os.path.join(reference, 'reused.json')) if os.path.exists(os.path.join(reference, 'reused.json')) else {}
    problems, roots = [], {}
    for name, spec in roots_spec.items():
        rc = dict(spec)
        rc.setdefault('pooled_dir', _first_dir(os.path.join(results, 'archive', f'{name}_v3f'),
                                               os.path.join(results, f'{name}_v3f')))
        rc.setdefault('fixed_beta_dir', _first_dir(os.path.join(results, f'{name}_bfix')))
        rc['pooled_dirs'] = {}
        for P, src in (rc.get('pooled_from') or {}).items():
            rc['pooled_dirs'][P] = _first_dir(os.path.join(results, 'archive', f'{src}_v3f'),
                                              os.path.join(results, f'{src}_v3f'))
            if not rc['pooled_dirs'][P]:
                problems.append(f'{name} {P}: no pooled results of {src} (looked in {results}/archive/{src}_v3f, '
                                f'{results}/{src}_v3f)')
        for which in ('pooled_dir', 'fixed_beta_dir'):
            if not rc[which] and (which == 'fixed_beta_dir' or set(rc['protocols']) - set(rc['pooled_dirs'])):
                problems.append(f'{name}: no {which} (looked in {results}/archive/{name}_v3f, {results}/{name}_v3f, '
                                f'{results}/{name}_bfix)')
        rc['design'], rc['v3_norm'], rc['checkpoints'] = {}, {}, {}
        rref = ref.get(name, {})
        for P in rc['protocols']:
            dpath = next((p for p in (os.path.join(reference, 'designs', name, f'design_{P}.json'),
                                      os.path.join(rc['fixed_beta_dir'] or '', 'igr', f'design_{P}.json'),
                                      os.path.join(rc['pooled_dir'] or '', 'igr', f'design_{P}.json'))
                          if os.path.exists(p)), None)
            if dpath is None:
                problems.append(f'{name} {P}: no design_{P}.json')
                continue
            design = _read(dpath)
            if design.get('attack') != rc['attack'] or design.get('design') != 'oat' or design.get('feature_set') != 'v2':
                problems.append(f'{dpath}: attack {design.get("attack")}, design {design.get("design")}, '
                                f'feature set {design.get("feature_set")} -- expected {rc["attack"]}, oat, v2')
            if not math.isclose(float(design['defaults']['attacker_ratio']), rc['source_beta'], abs_tol=1e-9):
                problems.append(f'{dpath}: default beta is not the source beta {rc["source_beta"]}')
            rc['design'][P] = dpath
            # pooled model's global z-score: the stored file of its training data directory
            npath = os.path.join(pooled_dir(rc, P) or '', 'igr', f'v3_norm_{P}.json')
            norm = _read(npath) if os.path.exists(npath) else None
            rnorm = (rref.get('v3_norm') or {}).get(P)
            if norm and rnorm and not (np.allclose(norm['mean'], rnorm['mean'], rtol=1e-9)
                                       and np.allclose(norm['std'], rnorm['std'], rtol=1e-9)):
                problems.append(f'{npath} differs from the reference copy in {reference}/reused.json')
            norm = norm or rnorm
            if norm is None and pooled_dir(rc, P):
                problems.append(f'{name} {P}: no v3_norm_{P}.json for the pooled model')
            if norm:
                rc['v3_norm'][P] = {'mean': list(norm['mean']), 'std': list(norm['std']),
                                    'feature_names': list(norm['feature_names']),
                                    'from': npath if os.path.exists(npath) else f'{reference}/reused.json'}
        for kind, D, P, which, mdir in reused_models(rc, rc['protocols']):
            mp = os.path.join(mdir, 'model.pt')
            if not os.path.exists(mp):
                problems.append(f'missing checkpoint {mp}')
                continue
            sha = refbeta.file_sha256(mp)
            key = f'{which}/{kind}/{cell(D, P, kind)}'
            want = (rref.get('checkpoints') or {}).get(key)
            if want and want != sha:
                problems.append(f'{mp}: sha256 {sha[:12]} is not the verified checkpoint {want[:12]}')
            if which == 'fixed_beta':
                for name_ in ('threshold.json', 'threshold_utility.json', 'scaler.json'):
                    if not os.path.exists(os.path.join(mdir, name_)):
                        problems.append(f'missing {mdir}/{name_}')
                if os.path.exists(os.path.join(mdir, 'threshold.json')) and \
                        refbeta.load_threshold(mdir)['model']['checkpoint_sha256'] != sha:
                    problems.append(f'{mp} is not the checkpoint of its threshold.json')
            rc['checkpoints'][key] = {'path': os.path.relpath(mp, ROOT), 'sha256': sha,
                                      'matches_reference': bool(want) and want == sha}
        roots[name] = rc
    if problems:
        raise SystemExit('init: nothing was created.\n  ' + '\n  '.join(problems))
    cfg = {'study': 'threshold_and_scaling_study', 'created': time.strftime('%F %T'), 'git': _git(),
           'command': ' '.join(sys.argv), 'results': results, 'roots': roots,
           'train': {**TRAIN, **override.get('train', {})},
           'trained_variants': {k: v for k, v in TRAINED.items()
                                if k in override.get('trained_variants', list(TRAINED))},
           'pooled_all_beta_calibration': bool(override.get('pooled_all_beta_calibration', True)),
           'heldout_baselines': list(override.get('heldout_baselines', args.baselines)),
           'base_seed': 42, 'utility_objective': ut.OBJECTIVE, 'utility_search': ut.SEARCH,
           'write_paths': {k: os.path.join(run_dir, k) for k in
                           ('simulated_runs', 'variants', 'scores', 'comparison', 'logs')}}
    for p in cfg['write_paths'].values():           # resolved destinations: all new, all inside the run dir
        if os.path.commonpath([os.path.abspath(p), os.path.abspath(run_dir)]) != os.path.abspath(run_dir) \
                or os.path.exists(p):
            raise SystemExit(f'write path {p} is outside the run directory or exists')
    os.makedirs(run_dir)
    for p in cfg['write_paths'].values():
        os.makedirs(p)
    refbeta._write_json(os.path.join(run_dir, MARKER), cfg)
    n_ck = sum(len(r['checkpoints']) for r in roots.values())
    say(f'init: {run_dir} created; {n_ck} reused checkpoints verified; writes go to')
    for k, p in cfg['write_paths'].items():
        say(f'    {k}: {p}')
    for name, rc in roots.items():
        say(f"    {name}: pooled {rc['pooled_dir']}, fixed beta {rc['fixed_beta_dir']}")
        for P, d in rc['pooled_dirs'].items():
            say(f"    {name} {P}: pooled models and their z-score from {d}")
    return 0


def cmd_extend(args):
    """Add trained variants to an existing run directory (default: every variant of TRAINED it does not
    have). Only study_config.json changes; train and select then skip what is finished, score and table are
    redone with the new models next to the old ones on the same held-out users."""
    cfg = load_cfg(args.run_dir)
    unknown = [v for v in args.variants or [] if v not in TRAINED]
    if unknown:
        raise SystemExit(f'extend: unknown variant(s) {unknown}; known: {list(TRAINED)}')
    new = [v for v in (args.variants or TRAINED) if v not in cfg['trained_variants']]
    if not new:
        say(f"extend: nothing to add; the run directory has {list(cfg['trained_variants'])}")
        return 0
    missing = [p for p in map(lambda r: run_path(cfg, r), plan(cfg)) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'extend: {len(missing)} simulated runs are missing (first: {missing[0]}); nothing was '
                         f'changed. Run the simulate stage on this run directory first (finished runs are skipped).')
    for v in new:
        cfg['trained_variants'][v] = TRAINED[v]
    cfg.setdefault('extensions', []).append({'added': time.strftime('%F %T'), 'variants': new, 'git': _git(),
                                             'command': ' '.join(sys.argv)})
    refbeta._write_json(os.path.join(args.run_dir, MARKER), {k: v for k, v in cfg.items() if k != 'run_dir'})
    n = sum(len(rc['protocols']) for rc in cfg['roots'].values()) * len(DATASETS) * 2
    say(f'extend: added {new} ({n} models each) to {args.run_dir}')
    todo = sum(not os.path.exists(run_path(cfg, r)) for r in plan(cfg))
    if todo:
        say(f'extend: {todo} more runs to simulate (training and validation runs at the other betas); run the '
            f'simulate stage before train')
    return 0


# ------------------------------------------------------------------------------------------
# plan and simulation
# ------------------------------------------------------------------------------------------

def plan(cfg):
    from config import DATASET_CONFIGS
    from generate_dataset import ATTACK_H_AO, _oat_seed
    runs = []
    for name, rc in cfg['roots'].items():
        for P in rc['protocols']:
            design = _read(rc['design'][P])
            base_seed = int(design.get('seed') or cfg['base_seed'])
            used = max(int(g.get('train', 0)) + int(g.get('holdout', 0)) for g in design['runs'])
            val_rep = max(VAL_OFFSET, used)
            for g in design['runs']:
                src = math.isclose(float(g['attacker_ratio']), rc['source_beta'], abs_tol=1e-9)
                reps = [('heldout_test', int(g.get('train', 0)) + j) for j in range(max(1, int(g.get('holdout', 0))))]
                if src:
                    reps += [('train_source', j) for j in range(int(g.get('train', 1)))] + [('validation', val_rep)]
                if src or cfg['pooled_all_beta_calibration']:
                    reps.append(('utility_validation', ut.REPLICATE_OFFSET))
                if not src and pooled_training(cfg):
                    reps += [('train_other', j) for j in range(int(g.get('train', 1)))] + [('validation_other', val_rep)]
                for role, rep in reps:
                    runs.append({
                        'root': name, 'attack': rc['attack'], 'role': role, 'dataset_type': g['dataset_type'],
                        'protocol': P, 'epsilon': g['epsilon'], 'attacker_ratio': g['attacker_ratio'],
                        'target_set_size': g['target_set_size'], 'splits': g['splits'], 'replicate': rep,
                        'n': int(design['n']), 'domain': DATASET_CONFIGS[g['dataset_type']]['domain'],
                        'h_ao': ATTACK_H_AO[rc['attack']],
                        'seed': _oat_seed(base_seed, g['dataset_type'], P, g['epsilon'], g['attacker_ratio'],
                                          g['target_set_size'], g['splits'], rep)})
    seen = {}
    for r in runs:      # a seed used twice would put the same users in two roles
        key = (r['root'], r['seed'], r['protocol'], r['dataset_type'])
        if key in seen:
            raise RuntimeError(f'seed {r["seed"]} planned for both {seen[key]} and {r["role"]}')
        seen[key] = r['role']
    order = {r: i for i, r in enumerate(ROLES)}
    runs.sort(key=lambda r: (order[r['role']], r['root'], r['protocol'], r['dataset_type'], float(r['epsilon']),
                             float(r['attacker_ratio']), int(r['splits']), int(r['replicate'])))
    return runs


def run_path(cfg, run):
    return os.path.join(cfg['run_dir'], 'simulated_runs', run['root'], run['role'],
                        cell(run['dataset_type'], run['protocol']), igr.run_tag(run) + '.npz')


def load_runs(cfg, root, role, protocol, datasets, betas=None):
    """Stored runs of a role, in plan order; fails when a planned run is missing."""
    out = []
    for r in plan(cfg):
        if r['root'] != root or r['role'] != role or r['protocol'] != protocol or r['dataset_type'] not in datasets:
            continue
        if betas is not None and not any(math.isclose(float(r['attacker_ratio']), b, abs_tol=1e-9) for b in betas):
            continue
        p = run_path(cfg, r)
        if not os.path.exists(p):
            raise FileNotFoundError(f'{p} is missing; run the simulate stage first')
        u = StudyRun.load(p)
        if u.run['role'] != role or int(u.run['seed']) != int(r['seed']):
            raise RuntimeError(f'{p} does not hold the planned {role} run')
        out.append(u)
    return out


def cmd_plan(args):
    cfg = load_cfg(args.run_dir)
    df = pd.DataFrame(plan(cfg))
    print(df.groupby(['root', 'role', 'protocol']).size().unstack('role').fillna(0).astype(int).to_string())
    print(f'{len(df)} runs; calibration betas: '
          f"{sorted(df[df.role == 'utility_validation'].attacker_ratio.unique())}; test betas: "
          f"{sorted(df[df.role == 'heldout_test'].attacker_ratio.unique())}")
    return 0


def cmd_simulate(args):
    cfg = load_cfg(args.run_dir)
    runs = plan(cfg)
    mine = runs[args.shard::args.num_shards]
    say(f'simulate shard {args.shard}/{args.num_shards}: {len(mine)} of {len(runs)} runs')
    designs, failed = {}, 0
    for i, run in enumerate(mine, 1):
        path = run_path(cfg, run)
        desc = f"[{i}/{len(mine)}] {run['root']} {run['role']} {cell(run['dataset_type'], run['protocol'])} {igr.run_tag(run)}"
        if os.path.exists(path):
            say(f'{desc}  (done)')
            continue
        try:
            t0 = time.time()
            key = (run['root'], run['protocol'])
            if key not in designs:
                designs[key] = _read(cfg['roots'][run['root']]['design'][run['protocol']])
            u = simulate_run(run, designs[key], args.processors,
                             cfg['heldout_baselines'] if run['role'] == 'heldout_test' else ())
            u.save(path)
            say(f'{desc}  {time.time() - t0:.0f}s  no-defence IGR {u.igr(np.zeros(u.n, bool)):+.3f}')
        except Exception as e:
            import traceback
            failed += 1
            say(f'{desc}  FAILED: {e}\n{traceback.format_exc()}')
    say(f'simulate shard {args.shard} finished; {failed} failed')
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------
# training (the fixed-beta procedure on stored runs)
# ------------------------------------------------------------------------------------------

def variant_dir(cfg, variant, root, kind, dataset, protocol):
    return os.path.join(cfg['run_dir'], 'variants', variant, root, kind, cell(dataset, protocol, kind))


def _stack(runs, spec):
    X = np.concatenate([u.inputs(spec['source'], spec['features']) for u in runs])
    y = np.concatenate([u.labels for u in runs]).astype(int)
    frame = pd.DataFrame({
        'dataset_type': np.concatenate([np.full(u.n, u.run['dataset_type']) for u in runs]),
        'protocol': np.concatenate([np.full(u.n, u.run['protocol']) for u in runs]),
        'epsilon': np.concatenate([np.full(u.n, round(float(u.run['epsilon']), 9)) for u in runs]),
        'attacker_ratio': np.concatenate([np.full(u.n, round(float(u.run['attacker_ratio']), 9)) for u in runs]),
        'target_set_size': np.concatenate([np.full(u.n, int(u.run['target_set_size'])) for u in runs]),
        'splits': np.concatenate([np.full(u.n, int(u.run['splits'])) for u in runs]),
        'run': np.concatenate([np.full(u.n, int(u.run['seed'])) for u in runs]),
        'y': y, 'idx': np.arange(len(y))})
    return X, y, frame


def train_jobs(cfg):
    return [(v, root, P, kind, D) for v in cfg['trained_variants'] for root, rc in cfg['roots'].items()
            for P in rc['protocols'] for D in DATASETS for kind in ('in_dist', 'cross')]


def train_model(cfg, variant, root, protocol, kind, dataset, device):
    import torch
    from .models import FT_TRANSFORMER_HP_KEYS, get_model
    from .training import Trainer
    spec, hp, rc = cfg['trained_variants'][variant], cfg['train'], cfg['roots'][root]
    out = variant_dir(cfg, variant, root, kind, dataset, protocol)
    vpath = os.path.join(out, 'variant.json')
    if os.path.exists(vpath) and _read(vpath).get('status') == 'complete':
        return 'skip'
    os.makedirs(out, exist_ok=True)
    src_beta, tds = float(rc['source_beta']), train_datasets(kind, dataset)
    pooled = spec.get('training_betas') == 'all'
    identity = {'study': cfg['study'], 'variant': variant, 'root': root, 'attack': rc['attack'], 'protocol': protocol,
                'training_method': 'none' if kind == 'in_dist' else 'cross', 'train_datasets': tds,
                'test_dataset': dataset, 'source_beta': src_beta, 'features': spec['features'],
                'training_betas': 'every beta of the sweep' if pooled else 'source beta only',
                'per_run_scaling': spec['per_run_scaling'], 'feature_source': spec['source'],
                'global_scaling': 'one z-score fitted on the optimization rows, applied unchanged everywhere',
                'hyperparameters': hp}
    started = time.strftime('%F %T')
    refbeta._write_json(vpath, {'variant': variant, 'status': 'running', 'identity': identity, 'started': started})

    opt_runs = load_runs(cfg, root, 'train_source', protocol, tds)
    val_runs = load_runs(cfg, root, 'validation', protocol, tds)
    if pooled:          # plus the configs at the other betas: every config of the sweep, equal rows each
        opt_runs += load_runs(cfg, root, 'train_other', protocol, tds)
        val_runs += load_runs(cfg, root, 'validation_other', protocol, tds)
    X_opt, y_opt, f_opt = _stack(opt_runs, spec)
    X_va, y_va, f_va = _stack(val_runs, spec)
    del opt_runs, val_runs
    at_source = np.isclose(f_opt.attacker_ratio, src_beta, atol=1e-9).all() and \
        np.isclose(f_va.attacker_ratio, src_beta, atol=1e-9).all()
    if not pooled and not at_source:
        raise RuntimeError('training or validation runs at another share than the source beta')
    if pooled and at_source:
        raise RuntimeError('a pooled-beta model needs training runs at the other betas')
    tr_idx, va_idx, cfg_rows = refbeta.sample_development_rows(f_opt, f_va, hp['train_samples'], hp['val_size'],
                                                               hp['seed'])
    mean, std = X_opt[tr_idx].mean(axis=0), X_opt[tr_idx].std(axis=0)
    std[std < 1e-12] = 1.0
    scaler = {'feature_names': list(spec['features']), 'mean': mean.tolist(), 'std': std.tolist(),
              'fitted_on': 'optimization rows', 'rows': int(len(tr_idx)), 'per_run_scaling': spec['per_run_scaling']}
    refbeta._write_json(os.path.join(out, 'scaler.json'), scaler)
    scaler_sha = refbeta.file_sha256(os.path.join(out, 'scaler.json'))

    def scale(a):
        return ((a - mean) / std).astype(np.float32)

    X_tr, y_tr = scale(X_opt[tr_idx]), y_opt[tr_idx].astype(np.float32)
    X_val, y_val = scale(X_va[va_idx]), y_va[va_idx].astype(np.float32)
    manifest = {'variant': variant, 'source_beta': src_beta,
                'training_betas': sorted({float(c['source_beta']) for c in cfg_rows}),
                'training_class_share': 0.5, 'train_datasets': tds,
                'test_dataset': dataset, 'protocols': [protocol],
                'budget': {'development_rows': int(hp['train_samples']), 'optimization_rows': int(len(tr_idx)),
                           'optimization_attackers': int(y_tr.sum()), 'validation_rows': int(len(va_idx)),
                           'validation_attackers': int(y_val.sum())},
                'eligible_configs': cfg_rows,
                'optimization_run_count': int(sum(len(c['optimization_runs']) for c in cfg_rows)),
                'validation_run_count': int(sum(len(c['validation_runs']) for c in cfg_rows)),
                'split_unit': 'complete simulated LDP run; a run id is the run\'s seed',
                'seed': hp['seed'], 'scaler_sha256': scaler_sha}
    refbeta._write_json(os.path.join(out, 'split_manifest.json'), manifest)

    torch.manual_seed(hp['seed'])
    model = get_model('ft_transformer', input_dim=X_tr.shape[1], dropout_rate=hp['dropout'],
                      **{k: hp[k] for k in FT_TRANSFORMER_HP_KEYS if hp.get(k) is not None})
    trainer = Trainer(model, device, learning_rate=hp['lr'], model_type='ft_transformer', epochs=hp['epochs'],
                      pos_weight=None)
    res = trainer.fit_with_validation(X_tr, y_tr, X_val, y_val, epochs=hp['epochs'], batch_size=hp['batch_size'],
                                      patience=hp['patience'], balanced_batches=hp['balanced_batches'])
    pd.DataFrame(res['history']).to_csv(os.path.join(out, 'training_history.csv'), index=False)
    trainer.save(os.path.join(out, 'model.pt'))
    model_sha = refbeta.file_sha256(os.path.join(out, 'model.pt'))
    sel = refbeta.select_threshold(refbeta.model_logits(model, X_val, device), y_val, src_beta)
    refbeta.save_threshold(out, {
        'variant': variant,
        'model': {'type': 'ft_transformer', 'output_dir': os.path.relpath(out, ROOT), 'checkpoint': 'model.pt',
                  'checkpoint_sha256': model_sha, 'scaler': 'scaler.json', 'scaler_sha256': scaler_sha},
        'protocol': protocol, 'training_method': identity['training_method'], 'train_datasets': tds,
        'test_dataset': dataset, 'source_beta': src_beta, 'threshold_reference_beta': src_beta,
        'training_class_share': 0.5, 'seed': hp['seed'], 'features': spec['features'],
        'per_run_scaling': spec['per_run_scaling'], 'eligible_configs': len(cfg_rows),
        'checkpoint_selection': refbeta.CHECKPOINT_CRITERION, 'best_epoch': res['best_epoch'],
        'best_val_f1_at_p0.5': res['best_val_f1'], 'objective': refbeta.OBJECTIVE, 'search': refbeta.SEARCH,
        'score_scale': refbeta.SCORE_SCALE, 'decision_operator': refbeta.DECISION_OPERATOR, **sel})
    refbeta._write_json(vpath, {'variant': variant, 'status': 'complete', 'identity': identity, 'started': started,
                                'finished': time.strftime('%F %T'), 'model_sha256': model_sha,
                                'scaler_sha256': scaler_sha})
    return f"best epoch {res['best_epoch']}, val F1 {sel['validation_f1']:.4f} at logit > {sel['threshold']:.4g}"


def cmd_train(args):
    import torch
    cfg = load_cfg(args.run_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    jobs = train_jobs(cfg)[args.shard::args.num_shards]
    say(f'train shard {args.shard}/{args.num_shards}: {len(jobs)} models on {device}')
    failed = 0
    for i, job in enumerate(jobs, 1):
        desc = f'[{i}/{len(jobs)}] ' + os.path.relpath(variant_dir(cfg, job[0], job[1], job[3], job[4], job[2]), cfg['run_dir'])
        try:
            t0 = time.time()
            say(f'{desc}  {train_model(cfg, *job, device)}  ({time.time() - t0:.0f}s)')
        except Exception as e:
            import traceback
            failed += 1
            say(f'{desc}  FAILED: {e}\n{traceback.format_exc()}')
    say(f'train shard {args.shard} finished; {failed} failed')
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------
# detectors: reused and new models, one interface
# ------------------------------------------------------------------------------------------

def _thr(mdir, name):
    return refbeta.load_threshold(mdir, name)['threshold'] if os.path.exists(os.path.join(mdir, name)) else None


def detectors(cfg, root, protocol, kind, dataset):
    """Models scored on a (protocol, test dataset, kind) cell: name, checkpoint, input columns, the fixed
    z-score and the frozen thresholds each has so far."""
    rc = cfg['roots'][root]
    c = cell(dataset, protocol, kind)
    out = []
    if pooled_dir(rc, protocol):
        b = float(_read(rc['design'][protocol])['defaults']['attacker_ratio'])      # known at training time
        thr = {'fixed_p0.5': 0.0, 'fixed_prior': igr.prior_cutoff(b)}
        for rule, variant in (('utility', POOLED_UTILITY), ('utility_all_beta', POOLED_UTILITY_ALL)):
            t = _thr(variant_dir(cfg, variant, root, kind, dataset, protocol), 'threshold_utility.json')
            if t is not None:
                thr[rule] = t
        out.append({'model': 'pooled_beta', 'dir': os.path.join(pooled_dir(rc, protocol), kind, c), 'source': 'scaled',
                    'features': V3, 'norm': rc['v3_norm'][protocol], 'thresholds': thr})
    if rc.get('fixed_beta_dir'):
        d = os.path.join(rc['fixed_beta_dir'], kind, c)
        out.append({'model': 'fixed_beta_4f_scaled_existing', 'dir': d, 'source': 'scaled', 'features': V3,
                    'norm': refbeta.load_scaler(d),
                    'thresholds': {k: v for k, v in (('f1', _thr(d, 'threshold.json')),
                                                     ('utility', _thr(d, 'threshold_utility.json'))) if v is not None}})
    for variant, spec in cfg['trained_variants'].items():
        d = variant_dir(cfg, variant, root, kind, dataset, protocol)
        if not os.path.exists(os.path.join(d, 'threshold.json')):
            continue
        thr = [('f1', _thr(d, 'threshold.json')), ('utility', _thr(d, 'threshold_utility.json'))]
        pooled = spec.get('training_betas') == 'all'
        if pooled:      # the reused pooled model's fixed rule and its second calibration
            thr = [('fixed_p0.5', 0.0)] + thr + [('utility_all_beta',
                                                  _thr(os.path.join(d, ALL_BETA_DIR), 'threshold_utility.json'))]
        out.append({'model': spec['model'], 'variant': variant, 'dir': d, 'source': spec['source'],
                    'features': spec['features'], 'norm': refbeta.load_scaler(d), 'pooled_training': pooled,
                    'thresholds': {k: v for k, v in thr if v is not None}})
    return out


class Scorer:
    """Checkpoints loaded once; logits of a stored run under a detector."""

    def __init__(self, device):
        self.device, self._m = device, {}

    def logits(self, det, u):
        path = os.path.join(det['dir'], 'model.pt')
        if path not in self._m:
            import torch
            ck = torch.load(path, map_location='cpu')
            self._m[path] = (igr.build_ft(len(det['features']), igr.FT_SPEC,
                                          ck['model_state_dict'] if 'model_state_dict' in ck else ck, self.device),
                             refbeta.file_sha256(path))
        model, _ = self._m[path]
        if list(det['norm'].get('feature_names', det['features'])) != list(det['features']):
            raise RuntimeError(f'{path}: scaler columns do not match the detector features')
        return igr.ft_logits(model, igr.model_input(u.inputs(det['source'], det['features']), det['norm']),
                             self.device).astype(np.float64)

    def sha(self, det):
        return self._m[os.path.join(det['dir'], 'model.pt')][1]


def cells(cfg):
    return [(root, P, D) for root, rc in cfg['roots'].items() for P in rc['protocols'] for D in DATASETS]


def _device(args):
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------------------------------------
# utility thresholds (calibration runs only)
# ------------------------------------------------------------------------------------------

def _select(det, runs, scorer, out_dir, extra):
    """Mean-IGR^2 threshold of one model on its utility-validation runs -> threshold_utility.json."""
    for u in runs:      # the selection never sees a test run
        if u.run['role'] != 'utility_validation' or int(u.run['replicate']) < ut.REPLICATE_OFFSET:
            raise RuntimeError('only utility-validation runs may select a threshold')
    scores = [scorer.logits(det, u) for u in runs]
    thr, _, cand = ut.select(scores, runs)
    at = ut.summarize(thr, scores, runs)
    ref = {name: {k: v for k, v in ut.summarize(t, scores, runs).items() if k != 'per_run_igr'}
           for name, t in [('no_defense', np.inf)] + [(f'at_{k}', v) for k, v in det['thresholds'].items()
                                                      if not k.startswith('utility')]}
    betas = sorted({round(u.beta, 9) for u in runs})
    rec = {'variant': extra['variant'], 'threshold_kind': 'utility',
           'model': {'type': 'ft_transformer', 'checkpoint': os.path.relpath(os.path.join(det['dir'], 'model.pt'), ROOT),
                     'checkpoint_sha256': scorer.sha(det), 'features': det['features'],
                     'feature_source': det['source'], 'global_scaling': det['norm']},
           **extra, 'objective': ut.OBJECTIVE, 'search': ut.SEARCH, 'score_scale': refbeta.SCORE_SCALE,
           'decision_operator': refbeta.DECISION_OPERATOR,
           'threshold': thr if math.isfinite(thr) else ('inf' if thr > 0 else '-inf'),
           'threshold_probability': refbeta._sigmoid(thr),
           'operating_point': 'flag_none' if thr == np.inf else 'interior', 'candidates': int(len(cand)),
           'calibration_betas': betas,
           'calibration_runs_per_beta': {f'{b:g}': int(sum(math.isclose(u.beta, b, abs_tol=1e-9) for u in runs))
                                         for b in betas},
           'calibration_weighting': 'every utility-validation run counts once in the mean of IGR^2 (no per-beta or '
                                    'per-user weights); IGR is divided by beta * r, so a run at a small beta has a '
                                    'larger IGR for the same absolute error',
           'uses_test_information': False,
           'utility_validation_runs': [{'dataset_type': u.run['dataset_type'], 'epsilon': u.run['epsilon'],
                                        'attacker_ratio': u.beta, 'splits': u.run['splits'],
                                        'replicate': u.run['replicate'], 'seed': u.run['seed'], 'n': u.n}
                                       for u in runs],
           'validation_runs': len(runs), **{f'validation_{k}': v for k, v in at.items()}, 'reference': ref}
    os.makedirs(out_dir, exist_ok=True)
    refbeta._write_json(os.path.join(out_dir, 'threshold_utility.json'), rec)
    return rec


def cmd_select(args):
    cfg = load_cfg(args.run_dir)
    scorer, failed = Scorer(_device(args)), 0
    mine = cells(cfg)[args.shard::args.num_shards]
    say(f'select shard {args.shard}/{args.num_shards}: {len(mine)} cells')
    for root, P, D in mine:
        rc = cfg['roots'][root]
        for kind in ('in_dist', 'cross'):
            tds = train_datasets(kind, D)
            try:
                src_runs = load_runs(cfg, root, 'utility_validation', P, tds, betas=[rc['source_beta']])
                all_runs = load_runs(cfg, root, 'utility_validation', P, tds) if cfg['pooled_all_beta_calibration'] else None
                base = {'root': root, 'attack': rc['attack'], 'protocol': P,
                        'training_method': 'none' if kind == 'in_dist' else 'cross', 'train_datasets': tds,
                        'test_dataset': D, 'source_beta': rc['source_beta']}
                for det in detectors(cfg, root, P, kind, D):
                    todo = []
                    if det['model'] == 'pooled_beta':
                        todo.append((POOLED_UTILITY, src_runs, 'the source-beta utility-validation runs the '
                                                              'fixed-beta models use'))
                        if all_runs is not None:
                            todo.append((POOLED_UTILITY_ALL, all_runs, 'one utility-validation run of every config '
                                                                       'of the sweep (every training beta)'))
                    elif 'variant' in det:
                        todo.append((det['variant'], src_runs, 'utility-validation runs at the source beta'))
                        if det.get('pooled_training') and all_runs is not None:
                            todo.append((os.path.join(det['variant'], ALL_BETA_DIR), all_runs,
                                         'one utility-validation run of every config of the sweep (every '
                                         'training beta)'))
                    for variant, runs, pop in todo:
                        out = variant_dir(cfg, os.path.dirname(variant) or variant, root, kind, D, P)
                        if os.path.basename(variant) == ALL_BETA_DIR:
                            out = os.path.join(out, ALL_BETA_DIR)
                        if os.path.exists(os.path.join(out, 'threshold_utility.json')):
                            continue
                        rec = _select(det, runs, scorer, out, {**base, 'variant': variant,
                                                                'calibration_population': pop})
                        say(f"{os.path.relpath(out, cfg['run_dir'])}: logit > {rec['threshold']}  mean IGR^2 "
                            f"{rec['validation_mean_igr2']:.4f} (none {rec['reference']['no_defense']['mean_igr2']:.4f})  "
                            f"flagged {100 * rec['validation_flagged_share']:.1f}%  betas {rec['calibration_betas']}  "
                            f"{len(runs)} runs")
            except Exception as e:
                import traceback
                failed += 1
                say(f'{root} {P} {D} {kind}  FAILED: {e}\n{traceback.format_exc()}')
    say(f'select shard {args.shard} finished; {failed} failed')
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------
# scoring on the held-out runs (every row of a run is on the same users)
# ------------------------------------------------------------------------------------------

KEYS = ['root', 'attack', 'dataset_type', 'protocol', 'epsilon', 'attacker_ratio', 'target_set_size', 'splits',
        'replicate', 'seed', 'n']


def cmd_score(args):
    cfg = load_cfg(args.run_dir)
    scorer, failed = Scorer(_device(args)), 0
    mine = cells(cfg)[args.shard::args.num_shards]
    say(f'score shard {args.shard}/{args.num_shards}: {len(mine)} cells')
    for root, P, D in mine:
        out = os.path.join(cfg['run_dir'], 'scores', root, 'cells', cell(D, P) + '.csv')
        try:
            rows = []
            dets = {kind: detectors(cfg, root, P, kind, D) for kind in ('in_dist', 'cross')}
            for u in load_runs(cfg, root, 'heldout_test', P, [D]):
                key = {k: u.run.get(k) for k in KEYS}
                none = np.zeros(u.n, dtype=bool)
                for name, mask in [('no_defense', none), ('oracle', u.labels == 1)] + sorted(u.flags.items()):
                    rows.append({**key, 'kind': 'both', 'model': name, 'threshold_rule': '', 'threshold': '',
                                 'model_sha': '', **u.metrics(mask)})
                for kind, lst in dets.items():
                    for det in lst:
                        logit = scorer.logits(det, u)
                        for rule, thr in det['thresholds'].items():
                            rows.append({**key, 'kind': kind, 'model': det['model'], 'threshold_rule': rule,
                                         'threshold': thr, 'model_sha': scorer.sha(det)[:16],
                                         **u.metrics(refbeta.decide(logit, thr))})
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pd.DataFrame(rows).to_csv(f'{out}.{os.getpid()}.tmp', index=False)
            os.replace(f'{out}.{os.getpid()}.tmp', out)
            say(f'{root} {cell(D, P)}: {len(rows)} rows')
        except Exception as e:
            import traceback
            failed += 1
            say(f'{root} {cell(D, P)}  FAILED: {e}\n{traceback.format_exc()}')
    say(f'score shard {args.shard} finished; {failed} failed')
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('cmd', choices=['init', 'extend', 'plan', 'simulate', 'train', 'select', 'score', 'table'])
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'), help='init: where existing results live')
    ap.add_argument('--roots', nargs='+', choices=sorted(DEFAULT_ROOTS), help='init: data roots (default: both)')
    ap.add_argument('--config', help='init: JSON overriding roots / train settings (tests, smoke runs)')
    ap.add_argument('--baselines', nargs='*', default=['diffstats'], choices=['diffstats', 'fiad'],
                    help='init: baselines flagged on the held-out runs (FIAD is slow)')
    ap.add_argument('--variants', nargs='+', help='extend: trained variants to add (default: all it lacks)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--processors', type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == 'table':
        from .threshold_scaling_table import build
        return build(load_cfg(args.run_dir))
    return {'init': cmd_init, 'extend': cmd_extend, 'plan': cmd_plan, 'simulate': cmd_simulate, 'train': cmd_train,
            'select': cmd_select, 'score': cmd_score}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
