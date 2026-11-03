#!/usr/bin/env python
"""WSP Stage 1, frequency oracle.

  python scripts/wsp/stage1_freq.py train     # main.py (unchanged) for every seed without a Stage-1 run
  python scripts/wsp/stage1_freq.py verify    # rebuild each split, check test F1 matches the run's
                                              # test_results.csv, write the train/val cache Stage 2 reads
The reuse seed (42) is the existing run folder results/<root>/<run> (original runs already used
--epochs 50 --patience 10). Other seeds go to results/<out_root>/stage1_s<seed>/<run>.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from _common import ROOT

from attacker_detector.wsp.jobs import SWEEP, freq_spec, load_yaml, parse_freq_run


def main_args(fc, run, seed, out):
    method, protocol, dataset = parse_freq_run(run)
    a = ['main.py', '-d', os.path.join(fc['data'], protocol), *fc['main_args'], '--protocol', protocol]
    if method == 'cross':
        a += ['--training-method', 'cross', '--train-dataset', *[d for d in fc['datasets'] if d != dataset],
              '--test-dataset', dataset]
    else:
        a += ['--dataset', dataset]
    return a + ['--seed', str(seed), '-o', out]


def train(fc, jobs_n, gpus):
    todo = []
    for run in fc['runs']:
        for seed in fc['seeds']:
            spec = freq_spec(fc, run, seed)
            out = spec['stage1_dir']
            if os.path.exists(os.path.join(out, 'sensitivity_test_results.csv')):
                print(f'skip {run} s{seed}: {os.path.relpath(out, ROOT)} exists')
                continue
            if int(seed) == int(fc.get('reuse_seed', -1)):
                raise SystemExit(f'reuse seed run missing: {out}. Copy the original runs to the server first.')
            todo.append((run, seed, out))
    slots = queue.Queue()
    for i in range(max(1, jobs_n)):
        slots.put(gpus[i % len(gpus)] if gpus else '')
    fails = []

    def one(item):
        run, seed, out = item
        gpu = slots.get()
        try:
            os.makedirs(out, exist_ok=True)
            cmd = [sys.executable] + main_args(fc, run, seed, out)
            env = dict(os.environ)
            if gpu != '':
                env['CUDA_VISIBLE_DEVICES'] = gpu
            t0 = time.time()
            print(f"[{time.strftime('%T')}] train {run} s{seed} (gpu {gpu or '-'})", flush=True)
            with open(os.path.join(out, 'stage1_main.log'), 'w') as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
            wall = time.time() - t0
            with open(os.path.join(out, 'wsp_stage1.json'), 'w') as f:
                json.dump({'cmd': cmd[1:], 'wall_s': wall, 'returncode': rc, 'gpu': gpu}, f, indent=1)
            print(f"[{time.strftime('%T')}] done  {run} s{seed}: rc {rc}, {wall / 60:.1f} min", flush=True)
            if rc:
                fails.append((run, seed))
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max(1, jobs_n)) as ex:
        list(ex.map(one, todo))
    if fails:
        raise SystemExit(f'failed: {fails}')


def verify(fc, device):
    import pandas as pd
    import torch
    from attacker_detector.wsp.adapters import build_freq_model, build_freq_split, freq_arrays, load_freq_state
    from attacker_detector.wsp.frozen_eval import _test_metrics
    rows = []
    for run in fc['runs']:
        for seed in fc['seeds']:
            spec = freq_spec(fc, run, seed)
            cache = os.path.join(ROOT, 'results', fc['out_root'], 'cache', f"{run.replace('/', '__')}__s{seed}.npz")
            t0 = time.time()
            ds, split = build_freq_split(spec)
            freq_arrays(ds, split, cache)
            model = build_freq_model(spec, split['X_test'].shape[1], load_freq_state(spec['model_path']), device)
            m = _test_metrics(model, split['X_test'], split['y_test'], torch.device(device))
            saved = pd.read_csv(os.path.join(spec['stage1_dir'], 'test_results.csv')).iloc[0]
            diff = abs(m['F1_Score'] - saved['F1_Score'])
            ok = diff < 1e-4 and int(saved['n_samples']) == len(split['y_test'])
            rows.append({'run': run, 'seed': seed, 'f1_rebuilt': m['F1_Score'], 'f1_saved': saved['F1_Score'],
                         'n_test': len(split['y_test']), 'n_saved': int(saved['n_samples']), 'ok': ok})
            print(f"{'OK ' if ok else 'MISMATCH'} {run} s{seed}: F1 rebuilt {m['F1_Score']:.6f} vs saved "
                  f"{saved['F1_Score']:.6f} ({time.time() - t0:.0f}s)", flush=True)
            del ds, split
    out = os.path.join(ROOT, 'results', fc['out_root'], 'stage1_verify.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    bad = [r for r in rows if not r['ok']]
    if bad:
        raise SystemExit(f'{len(bad)} run(s) do not reproduce; check configs/wsp/sweep.yaml freq.spec / main_args '
                         f'against the original commands ({out})')
    print(f'all {len(rows)} Stage-1 runs reproduce; caches written. {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'verify'])
    ap.add_argument('--jobs', type=int, default=3, help='parallel main.py processes (train)')
    ap.add_argument('--gpus', default=os.environ.get('WSP_GPUS', ''))
    ap.add_argument('--device', default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '0') != '' else 'cpu')
    args = ap.parse_args()
    fc = load_yaml(SWEEP)['freq']
    if args.cmd == 'train':
        gpus = [g for g in args.gpus.split(',') if g] or None
        train(fc, args.jobs, gpus)
    else:
        import torch
        verify(fc, args.device if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    main()
