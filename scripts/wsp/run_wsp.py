#!/usr/bin/env python
"""Run or list WSP Stage-2 jobs (one job = one unit x mode x probe budget).

  python scripts/wsp/run_wsp.py list --set tier_a [--mode wsp] [--kind mv]
  python scripts/wsp/run_wsp.py list --pilot
  python scripts/wsp/run_wsp.py job --id mv/wsp_b10/single-source__PM_eps4__taxi__deepsets_recovery__s0

Exit codes for `job`: 0 finished or skipped, 2 stopped with an error, 3 paused at the global token cap
(re-run the same command to resume in both cases).
"""
import argparse
import sys

from _common import load_env

from attacker_detector.wsp.jobs import all_jobs, is_done, load_yaml, select, SWEEP
from attacker_detector.wsp.runner import Job


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    ls = sub.add_parser('list')
    ls.add_argument('--set', nargs='+', default=['tier_a'])
    ls.add_argument('--mode', nargs='+')
    ls.add_argument('--kind', nargs='+')
    ls.add_argument('--pilot', action='store_true')
    jb = sub.add_parser('job')
    jb.add_argument('--id', required=True)
    jb.add_argument('--device', default='auto')
    args = ap.parse_args()
    jobs = all_jobs()
    if args.cmd == 'list':
        sel = select(jobs, args.set, args.mode, args.pilot, args.kind)
        for j in sel:
            print(('done ' if is_done(j) else 'todo ') + j['id'])
        print(f'# {len(sel)} jobs, {sum(is_done(j) for j in sel)} done', file=sys.stderr)
        return 0
    if args.id not in jobs:
        print(f'unknown job id {args.id!r}; see `run_wsp.py list`', file=sys.stderr)
        return 1
    job = jobs[args.id]
    if job['mode'] == 'wsp':
        load_env()
    stop = Job(job, load_yaml(SWEEP), device=args.device).run()
    print(f'{args.id}: {stop}', flush=True)
    return {'error': 2, 'llm_error': 2, 'paused': 3}.get(stop, 0)


if __name__ == '__main__':
    sys.exit(main())
