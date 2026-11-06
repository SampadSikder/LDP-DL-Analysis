#!/usr/bin/env python
"""IGR (item gain ratio) of removal defences on the frequency oracles: our v3 FT-Transformer vs
FIAD vs Diffstats on the same simulated users. Definition and method: attacker_detector/igr.py.

  python scripts/igr/run_igr.py norms                    # v3 z-score of each training data dir (serial)
  python scripts/igr/run_igr.py run --shard 0 --num-shards 8 --processors 4
  python scripts/igr/run_igr.py run --baselines diffstats  # skip FIAD (most of the run time)
  python scripts/igr/run_igr.py wsp                      # after Codex: WSP-edited models on cached runs
  python scripts/igr/run_igr.py collect                  # merge + F1 check vs the stored predictions

Outputs per results root (results/<root>/igr/):
  runs/<Dataset>_<PROTOCOL>/<run>.csv      one row per detector (a rerun skips finished runs)
  runs/<Dataset>_<PROTOCOL>/<run>.wsp.csv  WSP Stage-1 / edited models (wsp step)
  cache/<Dataset>_<PROTOCOL>/<run>.npz     reports (as bits) + our features of the cached cells
  v3_norm_<PROTOCOL>.json, design_<PROTOCOL>.json
  igr_runs.csv, igr_f1_check.csv           (collect)
scripts/igr/run_igr_server.sh drives all of it; scripts/igr/igr_figures.py draws the figures.
"""
import argparse
import fcntl
import glob
import json
import os
import shutil
import sys
import time
from argparse import Namespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from attacker_detector import igr  # noqa: E402

V3F_ROOTS = ['oat_b015_mga-a_n500k_v3f', 'oat_v3_n500k_v3f', 'oat_b010_mga-a_n500k_v3f',
             'oat_higheps_mga-a_n500k_v3f', 'oat_apa_n500k_v3f']
PROTOCOLS = ['OUE', 'OLH_Server', 'HST_Server', 'HST_User']
RESULTS = os.environ.get('IGR_RESULTS') or os.path.join(ROOT, 'results')   # override for tests


def say(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def res_dir(root):
    return os.path.join(RESULTS, root)


def igr_dir(root):
    return os.path.join(res_dir(root), 'igr')


def data_dir(root, protocol, data_map):
    base = data_map.get(root) or os.path.join('output', root[:-len('_v3f')] if root.endswith('_v3f') else root)
    return os.path.join(base if os.path.isabs(base) else os.path.join(ROOT, base), protocol)


def wsp_freq():
    from attacker_detector.wsp.jobs import SWEEP, load_yaml     # WSP_SWEEP env overrides it
    return load_yaml(SWEEP)['freq']


def wsp_available():
    """The WSP add-on is optional: without it IGR runs as usual, only the wsp step is unavailable."""
    try:
        from attacker_detector.wsp.jobs import SWEEP
    except ImportError:
        return False
    return os.path.exists(SWEEP)


def ft_spec():
    """FT-Transformer arguments of the original runs (scripts/run_extra_eps.sh)."""
    return dict(igr.FT_SPEC)


def wsp_cells():
    """(root, cell) pairs the WSP runs are tested on -- cached so edited models can be scored later.
    Empty without the WSP add-on."""
    if not wsp_available():
        return set()
    from attacker_detector.wsp.jobs import parse_freq_run
    fc = wsp_freq()
    out = set()
    for run in fc['runs']:
        _, protocol, dataset = parse_freq_run(run)
        out.add((fc['root'], igr.cell_name(dataset, protocol)))
    return out


def plan(args):
    """Every held-out LDP run of every selected root / protocol, as main.py --baseline plans them."""
    from attacker_detector.baselines import plan_runs
    data_map = dict(kv.split('=', 1) for kv in (args.data_map or []))
    runs = []
    for root in args.roots:
        if not os.path.isdir(res_dir(root)):
            say(f'[WARN] no results/{root}; skipped')
            continue
        for protocol in args.protocols:
            ddir = data_dir(root, protocol, data_map)
            if not os.path.exists(os.path.join(ddir, 'design.json')):
                say(f'[WARN] {ddir}/design.json missing; {root} {protocol} skipped')
                continue
            ns = Namespace(data_path=ddir, protocol=protocol, dataset=None, test_dataset=None,
                           eval_dataset=None, baseline_runs=args.replicates, seed=args.seed)
            rs, design, attack = plan_runs(ns)
            if design is None or design.get('feature_set') != 'v2':
                say(f'[WARN] {ddir}: our model needs a v2 dataset (design.json feature_set); skipped')
                continue
            for r in rs:
                if args.datasets and r['dataset_type'] not in args.datasets:
                    continue
                r.update(root=root, data_dir=ddir, attack=attack)
                runs.append(r)
    first = wsp_cells()
    runs.sort(key=lambda r: ((r['root'], igr.cell_name(r['dataset_type'], r['protocol'])) not in first,
                             args.roots.index(r['root']), PROTOCOLS.index(r['protocol']), r['dataset_type'],
                             float(r['epsilon']), float(r['attacker_ratio']), int(r['splits']),
                             int(r['target_set_size']), int(r['replicate'])))
    return runs


def _locked(path, fn):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.lock', 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def norm_for(root, protocol, ddir):
    path = os.path.join(igr_dir(root), f'v3_norm_{protocol}.json')
    return _locked(path, lambda: igr.train_v3_norm(ddir, path))


def design_for(root, protocol, ddir):
    dst = os.path.join(igr_dir(root), f'design_{protocol}.json')
    if not os.path.exists(dst):
        os.makedirs(igr_dir(root), exist_ok=True)
        shutil.copyfile(os.path.join(ddir, 'design.json'), dst + '.tmp')
        os.replace(dst + '.tmp', dst)
    with open(dst) as f:
        return json.load(f)


def cmd_norms(args):
    seen = set()
    for r in plan(args):
        key = (r['root'], r['protocol'])
        if key in seen:
            continue
        seen.add(key)
        design_for(r['root'], r['protocol'], r['data_dir'])
        t0 = time.time()
        nm = norm_for(r['root'], r['protocol'], r['data_dir'])
        say(f"v3 norm {r['root']} {r['protocol']}: {nm['rows']:,} rows, {nm['runs']} runs "
            f"({time.time() - t0:.0f}s)  mean={np.round(nm['mean'], 4).tolist()} std={np.round(nm['std'], 4).tolist()}")
    return 0


class Models:
    """FT checkpoints, loaded once per process."""

    def __init__(self, device):
        self.device, self.spec, self._m = device, ft_spec(), {}

    def get(self, path):
        if path not in self._m:
            self._m[path] = igr.load_ft(path, self.spec, self.device)
        return self._m[path]


def original_models(root, dataset, protocol):
    D = dataset.capitalize()
    return [('ours_in_dist', os.path.join(res_dir(root), 'in_dist', f'{D}_{protocol}', 'model.pt')),
            ('ours_cross', os.path.join(res_dir(root), 'cross', f'{protocol}_{D}_test', 'model.pt'))]


def _row(run, detector, m, seconds, variant='', model_seed='', budget=''):
    row = {k: run.get(k) for k in igr.ROW_KEYS}
    row.update(m, detector=detector, variant=variant, model_seed=model_seed, budget=budget,
               seconds=round(seconds, 2))
    return row


def _write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=igr.ROW_COLS).to_csv(path + '.tmp', index=False)
    os.replace(path + '.tmp', path)


BASELINE_METHODS = ('fiad', 'diffstats')


def _finished(out_csv, methods):
    """The run's CSV exists and has a row for every baseline asked for. A run finished without
    some baseline is simulated again when that baseline is wanted later, so all of its rows
    stay on the same users."""
    if not os.path.exists(out_csv):
        return False
    return set(methods) <= set(pd.read_csv(out_csv, usecols=['detector']).detector)


def process_run(run, args, models, cache_cells):
    from attacker_detector.baselines import BASELINES
    from attacker_detector.data.generators import extract_user_level_features_diffstats_style

    root, protocol, dataset = run['root'], run['protocol'], run['dataset_type']
    cell = igr.cell_name(dataset, protocol)
    out_csv = os.path.join(igr_dir(root), 'runs', cell, igr.run_tag(run) + '.csv')
    cache = (root, cell) in cache_cells
    cache_path = os.path.join(igr_dir(root), 'cache', cell, igr.run_tag(run) + '.npz')
    methods = [m for m in BASELINE_METHODS if m in args.baselines]
    if _finished(out_csv, methods) and (not cache or os.path.exists(cache_path)):
        return None

    design = design_for(root, protocol, run['data_dir'])
    norm = norm_for(root, protocol, run['data_dir'])
    t0 = time.time()
    X, targets, support, labels, estimate, one_list = igr.simulate(run, args.processors)
    n, d, eps = len(labels), int(run['domain']), float(run['epsilon'])
    t_sim = time.time() - t0

    t0 = time.time()
    sup = igr.Support.from_dense(support)
    v2 = extract_user_level_features_diffstats_style(
        support_list=support, one_list=one_list, epsilon=eps, protocol=protocol, domain=d, n=n,
        feature_set=design['feature_set'], block_candidates=design['block_candidates'])
    del support
    feats = igr.v3_features(v2, run, labels)
    del v2
    rd = igr.RunData(run, labels, targets, igr.genuine_frequency(X, labels, d), sup, feats)
    dev = np.abs(rd.estimate() * n - estimate).max() / max(1.0, np.abs(estimate).max())
    if dev > 1e-6:
        raise RuntimeError(f'estimator does not reproduce the generator estimate (rel. dev {dev:.2e})')
    t_feat = time.time() - t0

    rows = [_row(run, 'no_defense', rd.metrics(None), 0.0),
            _row(run, 'oracle', rd.metrics(rd.labels == 1), 0.0)]
    Xm = igr.model_input(feats, norm)
    for det, path in original_models(root, dataset, protocol):
        if not os.path.exists(path):
            say(f'    [WARN] {path} missing; {det} skipped')
            continue
        t1 = time.time()
        rows.append(_row(run, det, rd.metrics(igr.ft_flags(models.get(path), Xm, models.device)),
                         time.time() - t1, variant='original'))
    for method in methods:
        t1 = time.time()
        flagged = np.zeros(n, dtype=bool)
        flagged[np.asarray(BASELINES[method](sup.bits, one_list, run, args.processors), dtype=np.int64)] = True
        rows.append(_row(run, method, rd.metrics(flagged), time.time() - t1))
    if cache:
        rd.save(cache_path)
    _write_rows(out_csv, rows)
    return rows, t_sim, t_feat


def cmd_run(args):
    import torch
    torch.set_num_threads(max(1, args.processors))
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else ('cpu' if args.device == 'auto' else args.device))
    models = Models(device)
    cache_cells = set() if args.cache == 'none' else wsp_cells()
    runs = plan(args)
    if args.cache == 'all':
        cache_cells = {(r['root'], igr.cell_name(r['dataset_type'], r['protocol'])) for r in runs}
    mine = runs[args.shard::args.num_shards]
    say(f'shard {args.shard}/{args.num_shards}: {len(mine)} of {len(runs)} runs, device {device}, '
        f'{args.processors} processors')
    failed = 0
    for i, run in enumerate(mine, 1):
        desc = (f"[{i}/{len(mine)}] {run['root']} {igr.cell_name(run['dataset_type'], run['protocol'])} "
                f"{igr.run_tag(run)}")
        t0 = time.time()
        try:
            res = process_run(run, args, models, cache_cells)
        except Exception as e:  # keep the shard going; the run is retried on the next launch
            import traceback
            failed += 1
            say(f'{desc}  FAILED: {e}\n{traceback.format_exc()}')
            continue
        if res is None:
            say(f'{desc}  (done)')
            continue
        rows, t_sim, t_feat = res
        parts = '  '.join(f"{r['detector']}: F1={r['f1']:.3f} IGR={r['igr']:+.3f}" for r in rows)
        say(f'{desc}  sim {t_sim:.0f}s feats {t_feat:.0f}s total {time.time() - t0:.0f}s\n    {parts}')
    say(f'shard {args.shard} finished; {failed} failed')
    return 1 if failed else 0


def wsp_models():
    """Stage-1 and Stage-2 FT checkpoints of the WSP freq runs, grouped by the cell they test on."""
    from attacker_detector.wsp.jobs import all_jobs, parse_freq_run
    fc = wsp_freq()
    by_cell, seen = {}, set()
    for j in all_jobs().values():
        if j['kind'] != 'freq':
            continue
        method, protocol, dataset = parse_freq_run(j['run'])
        cell = igr.cell_name(dataset, protocol)
        det = 'ours_cross' if method == 'cross' else 'ours_in_dist'
        lst = by_cell.setdefault(cell, [])
        s1 = j['spec']['model_path']
        if (s1, j['seed']) not in seen and os.path.exists(s1):
            seen.add((s1, j['seed']))
            lst.append((det, 'stage1', j['seed'], '', s1, j['run']))
        ck = os.path.join(j['run_dir'], 'checkpoint', 'model.pt')
        if os.path.exists(ck):
            lst.append((det, j['mode'], j['seed'], j['budget'], ck, j['run']))
    return fc['root'], by_cell


def cmd_wsp(args):
    import torch
    torch.set_num_threads(max(1, args.processors))
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else ('cpu' if args.device == 'auto' else args.device))
    models = Models(device)
    if not wsp_available():
        say('wsp step needs the WSP add-on (attacker_detector/wsp, configs/wsp/sweep.yaml), not found here')
        return 1
    root, by_cell = wsp_models()
    n_edit = sum(1 for lst in by_cell.values() for m in lst if m[1] != 'stage1')
    say(f'WSP models: {sum(len(v) for v in by_cell.values())} ({n_edit} edited) over {len(by_cell)} cells of {root}')
    if not n_edit:
        say('[WARN] no Stage-2 checkpoints yet (results/<out_root>/stage2/*/*/checkpoint/model.pt); '
            'run after import_codex / the finetune and random baselines. Scoring Stage 1 only.')
    for cell, lst in sorted(by_cell.items()):
        caches = sorted(glob.glob(os.path.join(igr_dir(root), 'cache', cell, '*.npz')))
        if not caches:
            say(f'[WARN] no cached runs for {root} {cell}; run the "run" step first')
            continue
        protocol = cell.split('_', 1)[1]
        with open(os.path.join(igr_dir(root), f'v3_norm_{protocol}.json')) as f:
            norm = json.load(f)
        for path in caches:
            rd = igr.RunData.load(path)
            Xm = igr.model_input(rd.feats, norm)
            rows = []
            for det, variant, seed, budget, ck, run in lst:
                t1 = time.time()
                rows.append(_row(rd.run, det, rd.metrics(igr.ft_flags(models.get(ck), Xm, device)),
                                 time.time() - t1, variant=variant, model_seed=seed, budget=budget))
            _write_rows(os.path.join(igr_dir(root), 'runs', cell, os.path.basename(path)[:-4] + '.wsp.csv'), rows)
            say(f'{cell} {os.path.basename(path)[:-4]}: ' + '  '.join(
                f"{r['variant']}{r['model_seed']}:{r['igr']:+.3f}" for r in rows))
    return 0


def _stored_f1(model_dir):
    """F1 at 0.5 per held-out config, from main.py's test_predictions.npz."""
    p = os.path.join(model_dir, 'test_predictions.npz')
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        y, pred = z['label'].astype(bool), z['prob'] > 0.5
        df = pd.DataFrame({'epsilon': z['epsilon'].astype(float).round(6),
                           'attacker_ratio': z['attacker_ratio'].astype(float).round(6),
                           'splits': z['splits'].astype(int),
                           'tp': pred & y, 'fp': pred & ~y, 'fn': ~pred & y})
    s = df.groupby(['epsilon', 'attacker_ratio', 'splits'], as_index=False)[['tp', 'fp', 'fn']].sum()
    s['f1_stored'] = 2 * s.tp / (2 * s.tp + s.fp + s.fn).clip(lower=1)
    return s[['epsilon', 'attacker_ratio', 'splits', 'f1_stored']]


def cmd_collect(args):
    for root in args.roots:
        files = sorted(glob.glob(os.path.join(igr_dir(root), 'runs', '*', '*.csv')))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv(os.path.join(igr_dir(root), 'igr_runs.csv'), index=False)
        orig = df[df.detector.str.startswith('ours') & (df.variant == 'original')].copy()
        checks = []
        for (det, ds, proto), g in orig.groupby(['detector', 'dataset_type', 'protocol']):
            D = ds.capitalize()
            mdir = os.path.join(res_dir(root), *(('in_dist', f'{D}_{proto}') if det == 'ours_in_dist'
                                                else ('cross', f'{proto}_{D}_test')))
            st = _stored_f1(mdir)
            if st is None:
                continue
            g = g.groupby(['epsilon', 'attacker_ratio', 'splits'], as_index=False)['f1'].mean()
            g['epsilon'] = g.epsilon.astype(float).round(6)
            g['attacker_ratio'] = g.attacker_ratio.astype(float).round(6)
            m = g.merge(st, on=['epsilon', 'attacker_ratio', 'splits'], how='left')
            m.insert(0, 'protocol', proto)
            m.insert(0, 'dataset_type', ds)
            m.insert(0, 'detector', det)
            checks.append(m)
        n_runs = df.groupby(['dataset_type', 'protocol', 'epsilon', 'attacker_ratio', 'target_set_size',
                             'splits', 'replicate']).ngroups
        say(f'{root}: {n_runs} runs, {len(df)} rows -> {igr_dir(root)}/igr_runs.csv')
        if checks:
            c = pd.concat(checks, ignore_index=True)
            c['diff'] = c.f1 - c.f1_stored
            c.to_csv(os.path.join(igr_dir(root), 'igr_f1_check.csv'), index=False)
            say(f'    our F1 on the re-simulated runs vs the stored held-out runs: mean |diff| '
                f'{c["diff"].abs().mean():.4f}, max {c["diff"].abs().max():.4f} over {len(c)} configs '
                f'(fresh LDP noise, so small differences are expected; a large one means the features differ)')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['norms', 'run', 'wsp', 'collect', 'plan'])
    ap.add_argument('--roots', nargs='+', default=V3F_ROOTS)
    ap.add_argument('--protocols', nargs='+', default=PROTOCOLS, choices=PROTOCOLS)
    ap.add_argument('--datasets', nargs='+', choices=['zipf', 'emoji', 'fire'])
    ap.add_argument('--data-map', nargs='+', metavar='ROOT=DATA_ROOT',
                    help='data root of a results root (default output/<root without _v3f>)')
    ap.add_argument('--replicates', type=int, help='simulated runs per config (default: the held-out count)')
    ap.add_argument('--seed', type=int, default=42, help="base seed of the datasets (main.py's default)")
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--processors', type=int, default=4, help='simulation / FIAD / torch threads per shard')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--cache', choices=['wsp', 'all', 'none'], default='wsp',
                    help='which runs keep reports + features for scoring models later (wsp = cells the WSP runs test on)')
    ap.add_argument('--baselines', nargs='*', choices=BASELINE_METHODS, default=list(BASELINE_METHODS),
                    help='baseline detectors scored next to ours (default: both). --baselines diffstats skips '
                         'FIAD, which takes most of the run time; --baselines with no names scores ours only')
    args = ap.parse_args()
    if args.cmd == 'plan':
        runs = plan(args)
        cells = pd.Series([f"{r['root']} {igr.cell_name(r['dataset_type'], r['protocol'])}" for r in runs])
        print(cells.value_counts(sort=False).to_string())
        print(f'{len(runs)} runs')
        return 0
    return {'norms': cmd_norms, 'run': cmd_run, 'wsp': cmd_wsp, 'collect': cmd_collect}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
