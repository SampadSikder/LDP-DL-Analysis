#!/usr/bin/env python
"""Codex's interface to WSP runs (task description: scripts/wsp/codex/WSP_TASK.md). Call it as scripts/wsp/wspctl.

  queue                      runs in the bundle and their status
  next                       start (or resume) the next run, print its model card
  start JOB_ID [--force]     start / resume one run
  card                       model card, committed validation metrics, budget, gate rules
  stats [NAME ...] [--staged]          tensor statistics
  read NAME [INDEX] [--staged]         raw values of a slice (INDEX like '0:4, :')
  py FILE | py -c CODE       numpy code in the sandbox: W = staged weights (edit in place), W0 committed,
                             WS Stage 1. Only W persists between calls.
  probe [metrics|errors|activations] [--committed]   forward pass on the training probe set (1 probe)
  submit "RATIONALE"         gate the staged weights on validation data (1 gate evaluation)
  history | status
  finish "SUMMARY"           end the run: final checkpoint, cost (tokens from the Codex log), audit
Human side:
  pack                       re-count tokens of finished runs, tar them for the server
  tokens                     token use per run (from the Codex session logs)
"""
import argparse
import fcntl
import json
import os
import sys
import tarfile
import time
from contextlib import contextmanager

CWD0 = os.getcwd()
import _common  # noqa: E402,F401  (repo on sys.path, cwd = repo root)

from attacker_detector.wsp.codex import (CODEX_DIR, CodexJob, audit, codex_jobs, current_rollout,  # noqa: E402
                                         run_records, weekly_guard)
from attacker_detector.wsp.jobs import ROOT, STAGE2, load_yaml  # noqa: E402
from attacker_detector.wsp.log import dumps  # noqa: E402
from attacker_detector.wsp.runner import _write  # noqa: E402
from attacker_detector.wsp.tools import tool_functions  # noqa: E402

ACTIVE = os.path.join(CODEX_DIR, 'active.json')
CTL = 'scripts/wsp/wspctl'


@contextmanager
def locked():
    os.makedirs(CODEX_DIR, exist_ok=True)
    with open(os.path.join(CODEX_DIR, '.lock'), 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def active_id():
    if not os.path.exists(ACTIVE):
        return None
    with open(ACTIVE) as f:
        return json.load(f)['job']


def order(jobs):
    return sorted(jobs.values(), key=lambda j: (not j.get('pilot', False), j['kind'], j['id']))


def status_of(j, act):
    if os.path.exists(os.path.join(j['run_dir'], 'done.json')):
        return 'done'
    if j['id'] == act:
        return 'active'
    return 'started' if os.path.exists(os.path.join(j['run_dir'], 'codex.json')) else 'todo'


def budget_line(job):
    b = job.s2['budgets'][job.job['budget']]
    return f"budget: {b['probes']} probes, {b['gate_evals']} gate evaluations, {b['token_cap']:,} tokens"


def guard(job, st, others):
    u = job.usage(others)
    job.apply_usage(st, u)
    cap = st.budget.token_cap
    if cap and u['total'] >= cap:
        return f'error: token cap reached ({u["total"]:,} of {cap:,} tokens). Run: {CTL} finish "<summary>"'
    r = st.budget.stop_reason()
    if r:
        return f'error: this run is over ({r}). Run: {CTL} finish "<summary>"'
    return None


def start(jobs, jid, force=False):
    act = active_id()
    if act and act != jid and not force and status_of(jobs[act], act) == 'active':
        return f'error: run {act} is still active. Finish it first ({CTL} finish "<summary>") or pass --force.'
    j = jobs[jid]
    if os.path.exists(os.path.join(j['run_dir'], 'done.json')):
        return f'{jid} is already finished. Run `{CTL} next` for the next one.'
    job = CodexJob(j)
    roll = current_rollout()
    lines = []
    if not job.started():
        cap = (job.s2.get('codex') or {}).get('weekly_pct_cap', 50)
        stop, msg = weekly_guard(cap)
        if stop:
            return f'error: {msg}. No new run this week: stop here and tell the user.'
        lines.append(msg)
        check = job.begin(roll)
        if check.get('checked'):
            lines.append('bundle check: Stage-1 validation ' + ('matches the server' if check['ok'] else
                         f"DIFFERS from the server ({check['field']}: rel diff {check['max_rel_diff']:.2e}); "
                         'continue, the user will see this in codex.json'))
        lines.insert(0, f'started {jid}')
    else:
        lines.insert(0, f'resumed {jid}')
    if roll is None:
        lines.append('note: no Codex session log found under $CODEX_HOME/sessions, so tokens cannot be counted')
    _write(ACTIVE, {'job': jid, 't': time.time()})
    st = job.open(roll)
    u = job.usage(run_records(jobs))
    job.apply_usage(st, u)
    lines += [budget_line(job), tool_functions(st)['model_card']()]
    job.close()
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(prog=CTL, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--job', help='run id (default: the active run)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('queue')
    sub.add_parser('next')
    s = sub.add_parser('start')
    s.add_argument('id')
    s.add_argument('--force', action='store_true')
    sub.add_parser('card')
    s = sub.add_parser('stats')
    s.add_argument('names', nargs='*')
    s.add_argument('--staged', action='store_true')
    s = sub.add_parser('read')
    s.add_argument('name')
    s.add_argument('index', nargs='?', default='')
    s.add_argument('--staged', action='store_true')
    s = sub.add_parser('py')
    s.add_argument('file', nargs='?')
    s.add_argument('-c', dest='code')
    s = sub.add_parser('probe')
    s.add_argument('kind', nargs='?', default='metrics')
    s.add_argument('--committed', action='store_true')
    s = sub.add_parser('submit')
    s.add_argument('rationale')
    sub.add_parser('history')
    sub.add_parser('status')
    s = sub.add_parser('finish')
    s.add_argument('summary')
    s = sub.add_parser('pack')
    s.add_argument('--out')
    sub.add_parser('tokens')
    args = ap.parse_args()

    with locked():
        jobs, _man = codex_jobs()
        act = active_id()
        if args.cmd == 'queue':
            rows = [(status_of(j, act), j['id']) for j in order(jobs)]
            for stt, jid in rows:
                print(f'{stt:8s} {jid}')
            n = {k: sum(r[0] == k for r in rows) for k in ('done', 'active', 'started', 'todo')}
            print(f"# {len(rows)} runs: {n['done']} done, {n['active']} active, {n['started']} started, {n['todo']} todo")
            return 0
        if args.cmd == 'next':
            if act and status_of(jobs[act], act) == 'active':
                print(start(jobs, act))
                return 0
            todo = [j for j in order(jobs) if status_of(j, act) == 'started'] + \
                   [j for j in order(jobs) if status_of(j, act) == 'todo']
            if not todo:
                print('queue empty: every bundled run is finished. Stop and tell the user to run `wspctl pack`.')
                return 0
            print(start(jobs, todo[0]['id']))
            return 0
        if args.cmd == 'start':
            if args.id not in jobs:
                print(f'error: unknown run {args.id!r}; see `{CTL} queue`')
                return 2
            print(start(jobs, args.id, args.force))
            return 0
        if args.cmd in ('pack', 'tokens'):
            return human(args, jobs)

        jid = args.job or act
        if not jid or jid not in jobs:
            print(f'error: no active run. Run `{CTL} next`.')
            return 2
        job = CodexJob(jobs[jid])
        if job.done():
            print(f'{jid} is finished. Run `{CTL} next` for the next run.')
            return 0
        if not job.started():
            print(f'error: {jid} is not started. Run `{CTL} start {jid}`.')
            return 2
        roll = current_rollout()
        st = job.open(roll)
        fns = tool_functions(st)
        others = run_records(jobs)
        out = ''
        try:
            if args.cmd == 'card':
                job.apply_usage(st, job.usage(others))
                out = fns['model_card']()
            elif args.cmd == 'stats':
                out = fns['tensor_stats'](args.names or None, args.staged)
            elif args.cmd == 'read':
                out = fns['read_tensor'](args.name, args.index, args.staged)
            elif args.cmd == 'py':
                if args.code is not None:
                    code = args.code
                elif args.file in (None, '-'):
                    code = sys.stdin.read()
                else:
                    path = args.file if os.path.isabs(args.file) else os.path.join(CWD0, args.file)
                    code = open(path).read()
                out = fns['python'](code)
            elif args.cmd == 'probe':
                out = guard(job, st, others) or fns['probe'](args.kind, 'committed' if args.committed else 'staged')
            elif args.cmd == 'submit':
                out = guard(job, st, others) or fns['submit_edit'](args.rationale)
            elif args.cmd == 'history':
                job.apply_usage(st, job.usage(others))
                out = fns['history']()
            elif args.cmd == 'status':
                job.apply_usage(st, job.usage(others))
                b = st.budget
                out = dumps({'run': jid, 'budget': b.status(), 'stop_reason': b.stop_reason(),
                             'committed_validation': st.cur_val.report})
            elif args.cmd == 'finish':
                u, au = job.finish(args.summary, roll, others)
                if os.path.exists(ACTIVE) and active_id() == jid:
                    os.remove(ACTIVE)
                with open(os.path.join(job.dir, 'wsp.json')) as f:
                    w = json.load(f)
                left = sum(status_of(j, None) != 'done' for j in jobs.values())
                out = '\n'.join([
                    f'finished {jid}: {w["edits_accepted"]} of {w["edits_submitted"]} edits accepted',
                    f'validation: Stage 1 {dumps(w["val_stage1"])}',
                    f'            final   {dumps(w["val_final"])}',
                    f'tokens: {u["total"]:,} over {u["calls"]} model calls; tool calls {au["tool_calls"]}, '
                    f'audit flags {len(au["flags"])}',
                    f'{left} runs left. Next: `{CTL} next` (or stop if the user asked for one run).'])
            note = job.sandbox.note if job.sandbox else None
            if note:
                out = note + '\n' + out
        finally:
            err = job.close()
            if err and args.cmd == 'py':
                out += f'\nwarning: staged W is not valid ({err}); the last valid staged weights are kept'
            st.log.event('cli', argv=sys.argv[1:], out=out[:4000])
        print(out)
        return 0


def human(args, jobs):
    recs = run_records(jobs)
    done = [j for j in order(jobs) if os.path.exists(os.path.join(j['run_dir'], 'done.json'))]
    rows = []
    for j in done:
        job = CodexJob(j)
        u = job.usage(recs)
        m = job.meta()
        au = audit(m.get('rollouts'), m['start'], m.get('end'))
        _write(job.p['audit'], au)
        job.annotate(u, au)
        rows.append((j['id'], u, len(au['flags'])))
    tot = sum(r[1]['total'] for r in rows)
    for jid, u, nf in rows:
        wk = (u.get('weekly_pct') or {}).get('used')
        print(f"{u['total']:>12,} tokens {u['calls']:>4} calls  weekly {wk if wk is not None else '?':>5}  "
              f"flags {nf}  {u.get('model')}/{u.get('effort')}  {jid}")
    print(f'# {len(rows)} finished runs, {tot:,} tokens')
    if args.cmd == 'tokens':
        return 0
    if not done:
        print('nothing to pack')
        return 1
    out = args.out or os.path.join(CODEX_DIR, f'codex_results_{time.strftime("%Y%m%d_%H%M")}.tar.gz')
    skip = {'sandbox', 'cur_val.pkl', 'probe_cache.pkl', 'staged.npz'}
    with tarfile.open(out, 'w:gz') as tar:
        for j in done:
            for name in sorted(os.listdir(j['run_dir'])):
                if name not in skip:
                    p = os.path.join(j['run_dir'], name)
                    tar.add(p, arcname=os.path.relpath(p, ROOT))
    print(f'wrote {os.path.relpath(out, ROOT)} ({os.path.getsize(out) / 2 ** 20:.1f} MB). Copy it to the server '
          f'repo root and run: bash scripts/wsp/run_wsp_server.sh import_codex eval report')
    return 0


if __name__ == '__main__':
    sys.exit(main())
