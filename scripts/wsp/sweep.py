#!/usr/bin/env python
"""Run many WSP jobs in parallel, one GPU slot each, resumable (finished jobs are skipped).

  python scripts/wsp/sweep.py baselines            # finetune + random on tier A (no tokens)
  python scripts/wsp/sweep.py pilot                # the pilot LLM runs (configs/wsp/sweep.yaml: pilot)
  python scripts/wsp/sweep.py wsp                  # every tier-A LLM run (after you approve the pilot)
  python scripts/wsp/sweep.py wsp --set tier_b     # later week
Options: --concurrency N (default WSP_CONCURRENCY or 4), --gpus 0,1,2, --dry-run.
Before each LLM job starts, the shared token ledger must be below stop_launch_at x global cap.
Logs: results/wsp/logs/<job id>.log; summary: results/wsp/logs/sweep_<what>.log.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from _common import ROOT, load_env

from attacker_detector.wsp.cost import Ledger
from attacker_detector.wsp.jobs import STAGE2, WSP_HOME, all_jobs, is_done, load_yaml, select

LOGS = os.path.join(WSP_HOME, 'logs')


def gpu_list(arg):
    if arg:
        return [g for g in arg.split(',') if g != '']
    try:
        out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=30).stdout
        n = sum(1 for line in out.splitlines() if line.startswith('GPU'))
    except (OSError, subprocess.SubprocessError):
        n = 0
    return [str(i) for i in range(n)] or ['']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['baselines', 'pilot', 'wsp'])
    ap.add_argument('--set', nargs='+', default=['tier_a'])
    ap.add_argument('--kind', nargs='+', help='mv and/or freq (default both)')
    ap.add_argument('--concurrency', type=int, default=int(os.environ.get('WSP_CONCURRENCY', 4)))
    ap.add_argument('--gpus', default=os.environ.get('WSP_GPUS'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    load_env()
    s2 = load_yaml(STAGE2)
    jobs = all_jobs()
    if args.what == 'baselines':
        sel = select(jobs, args.set, ['finetune', 'random'], kinds=args.kind)
    elif args.what == 'pilot':
        sel = select(jobs, pilot_only=True, modes=['wsp'], kinds=args.kind)
    else:
        sel = select(jobs, args.set, ['wsp'], kinds=args.kind)
    todo = [j for j in sel if not is_done(j)]
    print(f'{args.what}: {len(sel)} jobs, {len(sel) - len(todo)} done, {len(todo)} to run', flush=True)
    if args.dry_run:
        for j in todo:
            print('  ' + j['id'])
        return 0
    os.makedirs(LOGS, exist_ok=True)
    cap = int(os.environ.get('WSP_TOKEN_CAP') or s2['global_token_cap'])
    stop_at = s2.get('stop_launch_at', 0.9) * cap
    ledger = Ledger()
    gpus = gpu_list(args.gpus)
    slots = queue.Queue()
    for i in range(max(1, args.concurrency)):
        slots.put(gpus[i % len(gpus)])
    threads = max(1, (os.cpu_count() or 2) // max(1, args.concurrency))
    halt = threading.Event()
    results = {}

    def run(j):
        if halt.is_set():
            results[j['id']] = 'not started (token cap)'
            return
        if j['mode'] == 'wsp':
            tok, usd = ledger.total()
            if tok >= stop_at:
                halt.set()
                results[j['id']] = f'not started: ledger {tok:,} >= {stop_at:,.0f} tokens'
                print(f"[{time.strftime('%T')}] token cap reached ({tok:,} tokens, ${usd:,.0f}); no new LLM runs",
                      flush=True)
                return
        gpu = slots.get()
        try:
            env = dict(os.environ)
            if gpu != '':
                env['CUDA_VISIBLE_DEVICES'] = gpu
            for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                env.setdefault(v, str(threads))
            log = os.path.join(LOGS, j['id'].replace('/', '__') + '.log')
            t0 = time.time()
            print(f"[{time.strftime('%T')}] start {j['id']} (gpu {gpu or '-'})", flush=True)
            with open(log, 'a') as f:
                rc = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'wsp', 'run_wsp.py'), 'job',
                                     '--id', j['id']], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
            status = 'ok' if rc == 0 else f'exit {rc} (see {os.path.relpath(log, ROOT)})'
            results[j['id']] = status
            tok, usd = ledger.total()
            print(f"[{time.strftime('%T')}] end   {j['id']}: {status}, {(time.time() - t0) / 60:.1f} min; "
                  f"ledger {tok:,} tokens ${usd:,.0f}", flush=True)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max(1, args.concurrency)) as ex:
        list(ex.map(run, todo))
    bad = {k: v for k, v in results.items() if v != 'ok'}
    tok, usd = ledger.total()
    print(f'finished: {len(results) - len(bad)} ok, {len(bad)} not ok; ledger {tok:,} tokens (cap {cap:,}), ${usd:,.0f}')
    for k, v in sorted(bad.items()):
        print(f'  {k}: {v}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
