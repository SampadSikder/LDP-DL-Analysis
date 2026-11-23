"""MSE threshold study on the frequency oracles: the cut-off of a detector is chosen on the squared error of
the target items instead of on IGR.

The threshold and scaling study (attacker_detector/threshold_scaling_study.py) picks its dynamic threshold
by the lowest mean IGR^2 on utility-validation runs. IGR sums the error over the r target items and divides
it by beta * r. Here the objective is the per-item error, with nothing divided out:

    MSE of a run = mean over its r target items t of (f_hat_t - f_t)^2        (igr.RunData 'mse_target')
    threshold    = lowest mean of that MSE over the utility-validation runs

  f_t    true frequency of t among the genuine users
  f_hat  the protocol's frequency estimate from the reports left after removing the flagged users

Nothing is trained. The models are the ones of a finished threshold and scaling study (the source run
directory), read only and identified by their sha256. Every model here has NO per-run median/IQR scaling:
  fixed_beta__four_features_unscaled     block_projection, block_overlap, one_deviation, excess_share
  fixed_beta__three_features_unscaled    the same without excess_share
  pooled_beta__four_features_unscaled    four features, trained on every beta of the sweep
Each gets two new frozen thresholds, selected on the same utility-validation runs (source beta; the pooled
model also on every config of the sweep):
  threshold_mse.json   lowest mean target-item MSE
  threshold_igr.json   lowest mean IGR^2 (the earlier objective, as the like-for-like partner)

A per-item error needs each user's support on every target item, which the earlier simulations did not
keep (they kept the sum over the targets, all IGR needs). So the utility-validation and held-out runs are
simulated again here from the same designs and seeds (same true items and targets, fresh LDP noise) and
stored with those bits. Every model of the source study is scored on these held-out users too, at the
thresholds that study froze, so that its best IGR-cut-off version can be shown as a reference on the same
users.

Nothing of a test run (labels, beta, targets, MSE, IGR) enters a threshold. Everything is written below
one new run directory; the source run directory and older result folders are only read.
Driver: scripts/run_mse_threshold_study.sh.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from . import reference_beta as refbeta
from . import threshold_scaling_study as st
from . import utility_threshold as ut

ROOT = st.ROOT
MARKER = 'study_config.json'
ROLES = ('utility_validation', 'heldout_test')
ALL = st.ALL_BETA_DIR
# source-study variant -> calibration populations of its new thresholds
MSE_MODELS = {
    'fixed_beta__four_features_unscaled': ['source_beta'],
    'fixed_beta__three_features_unscaled': ['source_beta'],
    'pooled_beta__four_features_unscaled': ['source_beta', 'all_beta'],
}
OBJECTIVES = {
    'mse': 'mean over utility-validation runs of the mean squared error of the r target items, '
           '(f_hat_t - f_t)^2, after removing the flagged users',
    'igr': ut.OBJECTIVE,
}
say = st.say


# ------------------------------------------------------------------------------------------
# target-item error of a stored run
# ------------------------------------------------------------------------------------------

def _targets(u):
    if u.targets is None:
        raise RuntimeError(f"run {u.run.get('seed')} was stored without its per-target support; simulate it with "
                           f"this study")
    return u.targets


def target_estimate(u, flagged):
    """(r,) frequency estimates of the target items after removing the flagged users; nan if nobody is left."""
    t = _targets(u)
    f = np.asarray(flagged, dtype=bool)
    kept = u.n - int(f.sum())
    if kept == 0:
        return np.full(len(t['true']), np.nan)
    cnt = t['bits'].sum(axis=0) - t['bits'][f].sum(axis=0)
    return u.A * (t['lo'] * kept + (t['hi'] - t['lo']) * cnt) / kept - u.B


def mse_target(u, flagged):
    """Mean squared error of the target items after removing the flagged users."""
    return float(np.mean((target_estimate(u, flagged) - _targets(u)['true']) ** 2))


def prefix_objectives(u, order):
    """Both objectives of one run for every prefix of `order` being flagged: entry k = the first k users of
    `order` removed (k = 0 .. n; the last, everyone removed, is nan). Returns {'mse': .., 'igr': IGR^2}."""
    t = _targets(u)
    r = len(t['true'])
    cnt = np.zeros((u.n + 1, r), dtype=np.int64)
    np.cumsum(t['bits'][order], axis=0, out=cnt[1:])
    kept = (u.n - np.arange(u.n + 1)).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        est = u.A * (t['lo'] * kept[:, None] + (t['hi'] - t['lo']) * (cnt[-1] - cnt)) / kept[:, None] - u.B
        mse = np.mean((est - t['true']) ** 2, axis=1)
        igr = (est.sum(axis=1) - t['true'].sum()) / (u.beta * u.r)
    mse[-1] = igr[-1] = np.nan
    return {'mse': mse, 'igr': igr * igr}


def select(score_list, runs):
    """Thresholds with the lowest mean target-item MSE and with the lowest mean IGR^2 over the runs, from the
    same candidates as attacker_detector.utility_threshold (every distinct pooled score boundary plus
    flag-none; flag-all excluded; ties -> the higher threshold). Returns {'mse': thr, 'igr': thr}, the
    objective arrays and the candidates."""
    if not runs or len(score_list) != len(runs):
        raise ValueError('need one score array per utility-validation run')
    for s, u in zip(score_list, runs):
        if len(s) != u.n or not np.isfinite(s).all():
            raise ValueError('scores must be finite and one per user of the run')
    cand = ut.candidate_thresholds(score_list)
    tot = {k: np.zeros(len(cand)) for k in OBJECTIVES}
    for s, u in zip(score_list, runs):
        s = np.asarray(s, dtype=np.float64)
        order = np.argsort(-s, kind='stable')
        k = np.searchsorted(-s[order], -cand, side='left')         # users with score > threshold
        for name, v in prefix_objectives(u, order).items():
            v = v[k]
            tot[name] += np.where(np.isnan(v), np.inf, v)
    obj = {name: v / len(runs) for name, v in tot.items()}
    return {name: float(cand[int(np.argmin(v))]) for name, v in obj.items()}, obj, cand


def summarize(threshold, score_list, runs):
    """Detection, IGR and target-item MSE of one threshold on utility-validation runs."""
    out = {k: v for k, v in ut.summarize(threshold, score_list, runs).items() if k != 'per_run_igr'}
    mse = np.asarray([mse_target(u, refbeta.decide(s, threshold)) for s, u in zip(score_list, runs)])
    out['mean_mse_target'] = float('inf') if np.isnan(mse).any() else float(mse.mean())
    return out


# ------------------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------------------

def load_cfg(run_dir):
    path = os.path.join(run_dir, MARKER)
    if not os.path.exists(path) or st._read(path).get('study') != 'mse_threshold_study':
        raise SystemExit(f'{run_dir} is not a run directory of the MSE threshold study; run init first')
    cfg = st._read(path)
    cfg['run_dir'] = run_dir
    return cfg


def source_cfg(cfg):
    """Config of the source study; its models and frozen thresholds are read through it."""
    return st.load_cfg(cfg['source_run_dir'])


def _model_key(variant, root, kind, dataset, protocol):
    return f'{variant}/{root}/{kind}/{st.cell(dataset, protocol, kind)}'


def cmd_init(args):
    """Create the run directory (refused if it exists) after checking that the source study holds every model
    this study needs."""
    run_dir = args.run_dir
    if os.path.exists(run_dir):
        raise SystemExit(f'{run_dir} already exists; every invocation writes to a new directory')
    results = os.path.abspath(args.results)
    if os.path.commonpath([os.path.abspath(run_dir), results]) != results:
        raise SystemExit(f'the run directory must be below {results}')
    source = os.path.abspath(args.source)
    if os.path.commonpath([os.path.abspath(run_dir), source]) == source:
        raise SystemExit('the run directory must not lie inside the source run directory')
    src = st.load_cfg(source)
    problems, models = [], {}
    for variant in MSE_MODELS:
        if variant not in src['trained_variants']:
            problems.append(f'{source} has no variant {variant}; add it there first (scripts/'
                            f'run_threshold_scaling_study.sh extend ...)')
            continue
        for root, rc in src['roots'].items():
            for P in rc['protocols']:
                for D in st.DATASETS:
                    for kind in ('in_dist', 'cross'):
                        d = st.variant_dir(src, variant, root, kind, D, P)
                        vpath = os.path.join(d, 'variant.json')
                        if not os.path.exists(vpath) or st._read(vpath).get('status') != 'complete':
                            problems.append(f'{d}: model not finished')
                            continue
                        sha = refbeta.file_sha256(os.path.join(d, 'model.pt'))
                        if sha != st._read(vpath).get('model_sha256'):
                            problems.append(f'{d}/model.pt is not the checkpoint of its variant.json')
                        for name in ('scaler.json', 'threshold.json', 'threshold_utility.json'):
                            if not os.path.exists(os.path.join(d, name)):
                                problems.append(f'missing {d}/{name}')
                        models[_model_key(variant, root, kind, D, P)] = sha
    for root, rc in src['roots'].items():
        for P, dpath in rc['design'].items():
            if not os.path.exists(dpath):
                problems.append(f'{root} {P}: design {dpath} is missing')
    if problems:
        raise SystemExit('init: nothing was created.\n  ' + '\n  '.join(problems[:40])
                         + (f'\n  ... and {len(problems) - 40} more' if len(problems) > 40 else ''))
    cfg = {'study': 'mse_threshold_study', 'created': time.strftime('%F %T'), 'git': st._git(),
           'command': ' '.join(sys.argv), 'results': results, 'source_run_dir': source,
           'source_study': {'created': src.get('created'), 'git': src.get('git'),
                            'extensions': src.get('extensions', []),
                            'trained_variants': list(src['trained_variants'])},
           'roots': src['roots'], 'mse_models': MSE_MODELS, 'models_sha256': models,
           # st.plan reads these three; no model is trained here, so no training run is planned
           'trained_variants': {}, 'pooled_all_beta_calibration': True, 'base_seed': src.get('base_seed', 42),
           'heldout_baselines': list(args.baselines), 'objectives': OBJECTIVES, 'search': ut.SEARCH,
           'write_paths': {k: os.path.join(run_dir, k) for k in
                           ('simulated_runs', 'thresholds', 'scores', 'comparison', 'logs')}}
    os.makedirs(run_dir)
    for p in cfg['write_paths'].values():
        os.makedirs(p)
    refbeta._write_json(os.path.join(run_dir, MARKER), cfg)
    say(f'init: {run_dir} created; {len(models)} models of {source} verified (read only); writes go to')
    for k, p in cfg['write_paths'].items():
        say(f'    {k}: {p}')
    return 0


# ------------------------------------------------------------------------------------------
# plan and simulation: utility-validation and held-out runs only
# ------------------------------------------------------------------------------------------

def plan(cfg):
    return [r for r in st.plan(cfg) if r['role'] in ROLES]


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
    return st.simulate_runs(cfg, plan(cfg), args)


# ------------------------------------------------------------------------------------------
# detectors: the source study's models, with its frozen thresholds and the ones selected here
# ------------------------------------------------------------------------------------------

def threshold_dir(cfg, variant, root, kind, dataset, protocol, calibration='source_beta'):
    d = os.path.join(cfg['run_dir'], 'thresholds', variant, root, kind, st.cell(dataset, protocol, kind))
    return d if calibration == 'source_beta' else os.path.join(d, ALL)


def detectors(cfg, src, root, protocol, kind, dataset):
    """Every model of the source study on a cell. Thresholds: the ones that study froze (its rule names) and,
    for the models of this study, 'mse' / 'igr' (source-beta calibration) and 'mse_all_beta' / 'igr_all_beta'."""
    out = []
    for det in st.detectors(src, root, protocol, kind, dataset):
        thr = dict(det['thresholds'])
        for calibration in cfg['mse_models'].get(det.get('variant'), []):
            d = threshold_dir(cfg, det['variant'], root, kind, dataset, protocol, calibration)
            for objective in OBJECTIVES:
                t = st._thr(d, f'threshold_{objective}.json')
                if t is not None:
                    thr[objective if calibration == 'source_beta' else f'{objective}_all_beta'] = t
        out.append(dict(det, thresholds=thr))
    return out


def _check_sha(cfg, det, root, kind, dataset, protocol, scorer):
    want = cfg['models_sha256'].get(_model_key(det.get('variant'), root, kind, dataset, protocol))
    if want and scorer.sha(det) != want:
        raise RuntimeError(f"{det['dir']}/model.pt changed since init (sha256 {scorer.sha(det)[:12]}, "
                           f"expected {want[:12]})")


# ------------------------------------------------------------------------------------------
# threshold selection (utility-validation runs only)
# ------------------------------------------------------------------------------------------

def _select(det, runs, scorer, out_dir, extra):
    """Both thresholds of one model on one calibration population -> threshold_mse.json, threshold_igr.json."""
    for u in runs:      # the selection never sees a test run
        if u.run['role'] != 'utility_validation' or int(u.run['replicate']) < ut.REPLICATE_OFFSET:
            raise RuntimeError('only utility-validation runs may select a threshold')
    scores = [scorer.logits(det, u) for u in runs]
    thr, _, cand = select(scores, runs)
    ref = {name: summarize(t, scores, runs)
           for name, t in [('no_defense', np.inf)] + [(f'at_{k}', v) for k, v in det['thresholds'].items()
                                                      if k not in ('mse', 'igr', 'mse_all_beta', 'igr_all_beta')]}
    betas = sorted({round(u.beta, 9) for u in runs})
    recs = {}
    for objective, t in thr.items():
        rec = {'variant': extra['variant'], 'threshold_kind': objective,
               'model': {'type': 'ft_transformer', 'checkpoint': os.path.relpath(os.path.join(det['dir'], 'model.pt'), ROOT),
                         'checkpoint_sha256': scorer.sha(det), 'features': det['features'],
                         'feature_source': det['source'], 'global_scaling': det['norm']},
               **extra, 'objective': OBJECTIVES[objective], 'search': ut.SEARCH,
               'score_scale': refbeta.SCORE_SCALE, 'decision_operator': refbeta.DECISION_OPERATOR,
               'threshold': t if math.isfinite(t) else ('inf' if t > 0 else '-inf'),
               'threshold_probability': refbeta._sigmoid(t),
               'operating_point': 'flag_none' if t == np.inf else 'interior', 'candidates': int(len(cand)),
               'calibration_betas': betas,
               'calibration_runs_per_beta': {f'{b:g}': int(sum(math.isclose(u.beta, b, abs_tol=1e-9) for u in runs))
                                             for b in betas},
               'calibration_weighting': 'every utility-validation run counts once. The target-item MSE is in '
                                        'frequency units and is not divided by beta: a run with more leftover '
                                        'gain or more LDP noise weighs more. IGR is divided by beta * r, so a '
                                        'run at a small beta weighs more there.',
               'uses_test_information': False,
               'utility_validation_runs': [{'dataset_type': u.run['dataset_type'], 'epsilon': u.run['epsilon'],
                                            'attacker_ratio': u.beta, 'splits': u.run['splits'],
                                            'replicate': u.run['replicate'], 'seed': u.run['seed'], 'n': u.n}
                                           for u in runs],
               'validation_runs': len(runs),
               **{f'validation_{k}': v for k, v in summarize(t, scores, runs).items()}, 'reference': ref}
        os.makedirs(out_dir, exist_ok=True)
        refbeta._write_json(os.path.join(out_dir, f'threshold_{objective}.json'), rec)
        recs[objective] = rec
    return recs


def cmd_select(args):
    cfg = load_cfg(args.run_dir)
    src = source_cfg(cfg)
    scorer, failed = st.Scorer(st._device(args)), 0
    mine = st.cells(cfg)[args.shard::args.num_shards]
    say(f'select shard {args.shard}/{args.num_shards}: {len(mine)} cells')
    for root, P, D in mine:
        rc = cfg['roots'][root]
        for kind in ('in_dist', 'cross'):
            tds = st.train_datasets(kind, D)
            try:
                pops = {'source_beta': (st.load_runs(cfg, root, 'utility_validation', P, tds, betas=[rc['source_beta']]),
                                        'utility-validation runs at the source beta'),
                        'all_beta': (st.load_runs(cfg, root, 'utility_validation', P, tds),
                                     'one utility-validation run of every config of the sweep (every beta)')}
                base = {'root': root, 'attack': rc['attack'], 'protocol': P,
                        'training_method': 'none' if kind == 'in_dist' else 'cross', 'train_datasets': tds,
                        'test_dataset': D, 'source_beta': rc['source_beta']}
                for det in detectors(cfg, src, root, P, kind, D):
                    for calibration in cfg['mse_models'].get(det.get('variant'), []):
                        out = threshold_dir(cfg, det['variant'], root, kind, D, P, calibration)
                        if all(os.path.exists(os.path.join(out, f'threshold_{o}.json')) for o in OBJECTIVES):
                            continue
                        runs, pop = pops[calibration]
                        recs = _select(det, runs, scorer, out, {**base, 'variant': det['variant'],
                                                                'calibration': calibration,
                                                                'calibration_population': pop})
                        _check_sha(cfg, det, root, kind, D, P, scorer)
                        m, g = recs['mse'], recs['igr']
                        say(f"{os.path.relpath(out, cfg['run_dir'])}: MSE cut-off logit > {m['threshold']} "
                            f"(MSE {m['validation_mean_mse_target']:.3e}, none "
                            f"{m['reference']['no_defense']['mean_mse_target']:.3e}, flagged "
                            f"{100 * m['validation_flagged_share']:.1f}%)  IGR cut-off logit > {g['threshold']} "
                            f"(MSE {g['validation_mean_mse_target']:.3e})  {len(runs)} runs")
            except Exception as e:
                import traceback
                failed += 1
                say(f'{root} {P} {D} {kind}  FAILED: {e}\n{traceback.format_exc()}')
    say(f'select shard {args.shard} finished; {failed} failed')
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------
# scoring on the held-out runs (every row of a run is on the same users)
# ------------------------------------------------------------------------------------------

def cmd_score(args):
    cfg = load_cfg(args.run_dir)
    src = source_cfg(cfg)
    scorer, failed = st.Scorer(st._device(args)), 0
    mine = st.cells(cfg)[args.shard::args.num_shards]
    say(f'score shard {args.shard}/{args.num_shards}: {len(mine)} cells')
    for root, P, D in mine:
        out = os.path.join(cfg['run_dir'], 'scores', root, 'cells', st.cell(D, P) + '.csv')
        try:
            rows = []
            dets = {kind: detectors(cfg, src, root, P, kind, D) for kind in ('in_dist', 'cross')}
            for kind, lst in dets.items():
                have = {d.get('variant'): d['thresholds'] for d in lst}
                for variant, cals in cfg['mse_models'].items():
                    need = {'mse', 'igr'} | ({'mse_all_beta', 'igr_all_beta'} if 'all_beta' in cals else set())
                    if not need <= set(have.get(variant, {})):
                        raise RuntimeError(f'{variant} {kind}: thresholds {sorted(need - set(have.get(variant, {})))} '
                                           f'are missing; run the select stage first')
            for u in st.load_runs(cfg, root, 'heldout_test', P, [D]):
                key = {k: u.run.get(k) for k in st.KEYS}

                def row(flagged):
                    return {**u.metrics(flagged), 'mse_target': mse_target(u, flagged)}

                none = np.zeros(u.n, dtype=bool)
                for name, mask in [('no_defense', none), ('oracle', u.labels == 1)] + sorted(u.flags.items()):
                    rows.append({**key, 'kind': 'both', 'model': name, 'threshold_rule': '', 'threshold': '',
                                 'model_sha': '', **row(mask)})
                for kind, lst in dets.items():
                    for det in lst:
                        logit = scorer.logits(det, u)
                        _check_sha(cfg, det, root, kind, D, P, scorer)
                        for rule, thr in det['thresholds'].items():
                            rows.append({**key, 'kind': kind, 'model': det['model'], 'threshold_rule': rule,
                                         'threshold': thr, 'model_sha': scorer.sha(det)[:16],
                                         **row(refbeta.decide(logit, thr))})
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pd.DataFrame(rows).to_csv(f'{out}.{os.getpid()}.tmp', index=False)
            os.replace(f'{out}.{os.getpid()}.tmp', out)
            say(f'{root} {st.cell(D, P)}: {len(rows)} rows')
        except Exception as e:
            import traceback
            failed += 1
            say(f'{root} {st.cell(D, P)}  FAILED: {e}\n{traceback.format_exc()}')
    say(f'score shard {args.shard} finished; {failed} failed')
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('cmd', choices=['init', 'plan', 'simulate', 'select', 'score', 'table'])
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--source', help='init: finished run directory of the threshold and scaling study (read only)')
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'), help='init: the run directory must be below it')
    ap.add_argument('--baselines', nargs='*', default=['diffstats'], choices=['diffstats', 'fiad'],
                    help='init: baselines flagged on the held-out runs (FIAD is slow)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--processors', type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == 'init':
        if not args.source:
            ap.error('init needs --source <run directory of the threshold and scaling study>')
        return cmd_init(args)
    if args.cmd == 'table':
        from .mse_threshold_table import build
        return build(load_cfg(args.run_dir))
    return {'plan': cmd_plan, 'simulate': cmd_simulate, 'select': cmd_select, 'score': cmd_score}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
