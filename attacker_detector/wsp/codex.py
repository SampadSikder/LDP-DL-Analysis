"""Codex as the WSP agent (ChatGPT plan, no API key).

Split across two machines:
  server  export_bundle(): per Stage-1 unit, the trained weights plus exactly the probe (training subset) and
          validation data the harness would use -- no other training data, no test data.
  local   CodexJob: the same RunState / gate / budget as the API agent, driven one command at a time by
          scripts/wsp/wspctl (Codex runs it as a shell command). Run state lives in the job's run_dir, so the
          finished run has the same files as a server run (checkpoint/, wsp.json, cost.json, done.json ...)
          and goes back to the server for test evaluation.

Tokens: Codex writes every model response's usage to its session log (~/.codex/sessions/**/rollout-*.jsonl).
Each response is attributed to the WSP run active at that moment (plus the lead-in after the user's message
and the tail before the next one). The weekly-limit percentage comes from the same log.
"""

import datetime
import glob
import json
import os
import pickle
import platform
import re
import shutil
import time
from types import SimpleNamespace

import numpy as np

from .budget import Budget
from .cost import CostTracker, Pricing
from .gate import DEFAULT_GATE
from .jobs import ROOT, STAGE2, all_jobs, load_yaml
from .log import RunLog, dumps
from .runner import Job, _write
from .sandbox import Sandbox, load_npz, save_npz

CODEX_DIR = os.environ.get('WSP_CODEX_DIR') or os.path.join(ROOT, 'results', 'wsp_codex')


def bundle_dir():
    return os.environ.get('WSP_BUNDLE') or os.path.join(CODEX_DIR, 'bundle')


def _json(path):
    with open(path) as f:
        return json.load(f)


def unit_id(job):
    if job['kind'] == 'mv':
        return 'mv/' + job['unit']
    return f"freq/{job['run'].replace('/', '__')}__s{job['seed']}"


def _cell_key(p, eps, srcs):
    return f"{p}_eps{float(eps):g}_{'+'.join(srcs)}"


# ====================================================================== export (server)

def export_bundle(jobs, sweep_cfg, s2, device, out=None, log=print, force=False):
    """Write the Codex bundle for these (mode 'wsp') jobs. Resumable: finished units are skipped."""
    import torch

    from .adapters import _probe_subset
    from .runner import build_adapter, mv_context
    out = out or bundle_dir()
    os.makedirs(out, exist_ok=True)
    man_path = os.path.join(out, 'manifest.json')
    man = _json(man_path) if os.path.exists(man_path) else {}
    man.setdefault('cells', {})
    man.setdefault('units', {})
    man.setdefault('jobs', {})
    man['probe'] = s2['probe']
    units = {}
    for j in jobs:
        units.setdefault(unit_id(j), j)

    mv_units = [j for j in units.values() if j['kind'] == 'mv']
    if mv_units:
        from ..mean_variance.store import RoundStore, shard_name
        ctx = mv_context(sweep_cfg)
        rroot = os.path.join(out, 'mv', 'rounds')
        mpath = os.path.join(rroot, 'manifest.jsonl')
        have = set()
        if os.path.exists(mpath):
            have = {json.loads(line)['round_id'] for line in open(mpath) if line.strip()}
        for p, eps, srcs in sorted({(j['protocol'], float(j['eps']), tuple(j['train_sources'])) for j in mv_units}):
            key = _cell_key(p, eps, srcs)
            if key in man['cells'] and not force:
                continue
            train = ctx.rounds('train', p, eps, srcs)
            val = ctx.rounds('val', p, eps, srcs)
            probe = _probe_subset(train, s2['probe']['rounds'], s2['probe']['seed'])
            for split, rounds in (('train', probe), ('val', val)):
                for src in srcs:
                    rs = [r for r in rounds if ctx.manifest[r.round_id]['source'] == src]
                    RoundStore._save_shard(os.path.join(rroot, split, shard_name(p, eps, src)), {
                        'reports': [r.observation['reports'] for r in rs],
                        'groups': [r.observation['groups'] for r in rs],
                        'is_fake': [r.is_fake for r in rs], 'round_ids': [r.round_id for r in rs]})
            with open(mpath, 'a') as f:
                for r in probe + val:
                    if r.round_id not in have:
                        f.write(json.dumps(ctx.manifest[r.round_id]) + '\n')
                        have.add(r.round_id)
            man['cells'][key] = {'probe_rounds': len(probe), 'val_rounds': len(val), 'train_rounds': len(train)}
            _write(man_path, man)
            log(f'cell {key}: {len(probe)} probe rounds (of {len(train)} training), {len(val)} validation rounds')

    for uid, j in sorted(units.items()):
        if uid in man['units'] and not force:
            continue
        t0 = time.perf_counter()
        a = build_adapter(j, sweep_cfg, s2, device)
        info = {'val_report': a.evaluate('val').report, 'stage1_cost': a.stage1_cost(),
                'fit_time': getattr(a, 'stage1_fit_time', None), 'n_params': a.n_params(),
                'eval_flops': {'probe': a.eval_flops('probe'), 'val': a.eval_flops('val')},
                'server_device': str(device), 'exported': time.strftime('%F %T')}
        if j['kind'] == 'mv':
            dst = os.path.join(out, 'mv', 'checkpoints', j['unit'])
            shutil.copytree(mv_context(sweep_cfg).checkpoint(j['unit']), dst, dirs_exist_ok=True)
            info.update(probe_n=len(a.data['probe']['meta']), val_n=len(a.data['val']['meta']))
            path = os.path.join(out, 'mv', 'stage1', j['unit'] + '.json')
        else:
            d = os.path.join(out, uid)
            os.makedirs(d, exist_ok=True)
            Xp, yp, cp = a.data['probe']
            Xv, yv, cv = a.data['val']
            np.savez(os.path.join(d, 'arrays.npz'), X_train=Xp, y_train=yp, config_train=cp, X_val=Xv, y_val=yv,
                     config_val=cv, feature_names=np.array(a.feature_names, dtype=str))
            torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in a.model.state_dict().items()}},
                       os.path.join(d, 'model.pt'))
            with open(os.path.join(d, 'spec.json'), 'w') as f:
                json.dump(a.spec, f, indent=1, default=str)
            info.update(probe_n=len(Xp), val_n=len(Xv))
            path = os.path.join(d, 'stage1.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write(path, info)
        man['units'][uid] = {'kind': j['kind'], 'stage1_json': os.path.relpath(path, out)}
        _write(man_path, man)
        log(f'unit {uid}: {info["n_params"]:,} weights, val {info["val_report"]}, {time.perf_counter() - t0:.0f}s')
        del a
    for j in jobs:
        man['jobs'][j['id']] = {'unit': unit_id(j), 'kind': j['kind'], 'budget': j['budget'],
                                'set': j.get('set'), 'pilot': bool(j.get('pilot'))}
    man['updated'] = time.strftime('%F %T')
    _write(man_path, man)
    return man


def bundle_manifest():
    p = os.path.join(bundle_dir(), 'manifest.json')
    if not os.path.exists(p):
        raise SystemExit(f'no Codex bundle at {bundle_dir()}: copy results/wsp_codex_bundle.tar.gz from the server '
                         'and unpack it at the repo root (tar xzf results/wsp_codex_bundle.tar.gz)')
    return _json(p)


def bundle_adapter(job, s2, device):
    """The run's adapter rebuilt from the bundle: same probe and validation data as on the server."""
    from .adapters import FreqFTAdapter, FTRemovalAdapter, SetRecoveryAdapter, build_freq_model, load_freq_state
    b = bundle_dir()
    seed = s2['probe']['seed']
    if job['kind'] == 'mv':
        from ..mean_variance.evaluate import rounds_from_store
        from ..mean_variance.methods.checkpoint import load_checkpoint
        from ..mean_variance.store import RoundStore
        store = RoundStore(os.path.join(b, 'mv', 'rounds'))
        manifest = store.manifest()
        p, eps, srcs = job['protocol'], float(job['eps']), list(job['train_sources'])
        probe = rounds_from_store(store, manifest, 'train', p, eps, srcs)
        val = rounds_from_store(store, manifest, 'val', p, eps, srcs)
        method = load_checkpoint(os.path.join(b, 'mv', 'checkpoints', job['unit']), str(device))
        cls = FTRemovalAdapter if job['method'].startswith('ft_removal') else SetRecoveryAdapter
        # the bundle holds exactly the server's probe subset; asking for all of it keeps its order
        a = cls(method, probe, val, manifest, probe_rounds=len(probe), seed=seed)
        info = _json(os.path.join(b, 'mv', 'stage1', job['unit'] + '.json'))
    else:
        d = os.path.join(b, unit_id(job))
        with np.load(os.path.join(d, 'arrays.npz'), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['feature_names'] = [str(x) for x in arrays['feature_names']]
        spec = _json(os.path.join(d, 'spec.json'))
        model = build_freq_model(spec, arrays['X_train'].shape[1], load_freq_state(os.path.join(d, 'model.pt')), device)
        a = FreqFTAdapter(model, spec, arrays, device, probe_rows=len(arrays['X_train']), seed=seed)
        info = _json(os.path.join(d, 'stage1.json'))
    a.stage1_fit_time = info.get('fit_time')
    s1 = dict(info.get('stage1_cost') or {})
    a.stage1_cost = lambda: dict(s1)
    a.flops_per_eval.update(info.get('eval_flops') or {})
    a.bundle_info = info
    return a


def compare_reports(local, server, tol=1e-4):
    """Max relative difference over the numeric fields both reports share."""
    if not server:
        return {'checked': False}
    worst, field = 0.0, None
    for k, v in server.items():
        lv = local.get(k)
        if isinstance(v, (int, float)) and isinstance(lv, (int, float)):
            d = abs(lv - v) / max(abs(v), 1e-8)
            if d > worst:
                worst, field = d, k
    return {'checked': True, 'max_rel_diff': worst, 'field': field, 'ok': worst <= tol}


# ====================================================================== Codex session logs

def codex_home():
    return os.environ.get('CODEX_HOME') or os.path.expanduser('~/.codex')


def _ts(s):
    if not isinstance(s, str):
        return None
    try:
        return datetime.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def rollout_files(since=0.0):
    out = []
    for sub in ('sessions', 'archived_sessions'):
        for f in glob.glob(os.path.join(codex_home(), sub, '**', '*.jsonl'), recursive=True):
            try:
                if os.path.getmtime(f) >= since - 120:
                    out.append(f)
            except OSError:
                pass
    return sorted(set(out), key=os.path.getmtime)


def current_rollout(max_age_s=900):
    """The session log Codex is writing right now (most recently modified), or None."""
    files = rollout_files(time.time() - max_age_s)
    return files[-1] if files else None


_ZERO = {'input': 0, 'cached': 0, 'output': 0, 'reasoning': 0, 'total': 0}


def _usage(u):
    inp, out = int(u.get('input_tokens') or 0), int(u.get('output_tokens') or 0)
    return {'input': inp, 'cached': int(u.get('cached_input_tokens') or 0), 'output': out,
            'reasoning': int(u.get('reasoning_output_tokens') or 0),
            'total': int(u.get('total_tokens') or inp + out)}


_PARSED = {}


def parse_rollout(path):
    """Model calls (time, usage), tool calls (time, text), user messages, weekly-limit readings and
    model / effort settings from one Codex session log. Tolerates unknown or partial lines."""
    try:
        sig = (os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return None
    if _PARSED.get(path, (None,))[0] == sig:
        return _PARSED[path][1]
    R = {'path': path, 'calls': [], 'tools': [], 'users': [], 'limits': [], 'contexts': []}
    seen, counts = set(), []
    with open(path, errors='replace') as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict):
                continue
            t = _ts(d.get('timestamp'))
            if t is None:
                continue
            typ, p = d.get('type'), d.get('payload') or {}
            if not isinstance(p, dict):
                continue
            pt = p.get('type')
            if typ == 'token_usage_record' and isinstance(p.get('usage'), dict):
                rid = p.get('response_id') or f"{t}:{json.dumps(p['usage'], sort_keys=True)}"
                if rid not in seen:
                    seen.add(rid)
                    R['calls'].append((t, _usage(p['usage'])))
            elif pt == 'token_count':
                tot = (p.get('info') or {}).get('total_token_usage')
                if isinstance(tot, dict):
                    counts.append((t, _usage(tot)))
                for slot in ((p.get('rate_limits') or {}).get('primary'), (p.get('rate_limits') or {}).get('secondary')):
                    if isinstance(slot, dict) and slot.get('window_minutes') == 10080 and slot.get('used_percent') is not None:
                        R['limits'].append((t, float(slot['used_percent']), slot.get('resets_at')))
            elif typ == 'turn_context':
                eff = p.get('effort') or ((p.get('collaboration_mode') or {}).get('settings') or {}).get('reasoning_effort')
                R['contexts'].append((t, p.get('model'), eff))
            elif typ == 'response_item' and pt in ('function_call', 'custom_tool_call', 'local_shell_call'):
                txt = p.get('arguments') or p.get('input') or json.dumps(p.get('action') or '')
                R['tools'].append((t, str(txt)))
            elif typ == 'response_item' and pt == 'message' and p.get('role') == 'user':
                R['users'].append(t)
    if not R['calls'] and counts:            # older log format: per-call usage = steps of the running total
        prev = dict(_ZERO)
        for t, tot in counts:
            if tot['total'] < prev['total']:
                prev = dict(_ZERO)
            step = {k: tot[k] - prev[k] for k in _ZERO}
            if step['total'] > 0:
                R['calls'].append((t, step))
            prev = tot
    _PARSED[path] = (sig, R)
    return R


def attribute(runs):
    """{run id: [(t, usage)]}. runs: dicts with id, start, end (None = active), rollouts (paths).
    A model call belongs to the run active at its time; otherwise to the next run in the same session if no
    user message comes in between (the lead-in), otherwise to the previous run on the same terms (the tail)."""
    res = {r['id']: [] for r in runs}
    by_path = {}
    for r in runs:
        for p in r.get('rollouts') or []:
            by_path.setdefault(p, []).append(r)
    for p, rs in by_path.items():
        R = parse_rollout(p)
        if R is None:
            continue
        rs = sorted(rs, key=lambda r: r['start'])
        users = sorted(R['users'])
        for t, u in R['calls']:
            run = next((r for r in rs if r['start'] <= t <= (r['end'] or float('inf'))), None)
            if run is None:
                nxt = [r for r in rs if r['start'] > t]
                if nxt and not any(t < x < nxt[0]['start'] for x in users):
                    run = nxt[0]
            if run is None:
                prv = [r for r in rs if r['end'] and r['end'] < t]
                if prv and not any(prv[-1]['end'] < x < t for x in users):
                    run = prv[-1]
            if run is not None:
                res[run['id']].append((t, u))
    return res


def summarize_usage(calls, rollouts, start, end, pricing=None):
    pricing = pricing or Pricing.load()
    tot = dict(_ZERO)
    usd = 0.0
    for _, u in calls:
        for k in _ZERO:
            tot[k] += u[k]
        usd += pricing.usd(u['input'], u['cached'], u['output'])
    t_lo = min([t for t, _ in calls] + [start])
    t_hi = max([t for t, _ in calls] + [end or time.time()])
    models, limits = [], []
    for p in rollouts or []:
        R = parse_rollout(p)
        if R:
            models += [(t, m, e) for t, m, e in R['contexts'] if t <= t_hi]
            limits += [x for x in R['limits'] if t_lo - 60 <= x[0] <= t_hi + 60]
    models.sort()
    limits.sort()
    model, effort = (models[-1][1], models[-1][2]) if models else (None, None)
    wk = None
    if limits:
        a, b = limits[0][1], limits[-1][1]
        wk = {'start': a, 'end': b, 'used': round(b - a if b >= a else b, 3)}
    return {**tot, 'calls': len(calls), 'usd': round(usd, 4), 'wait_s': 0.0,
            'max_call_tokens': max([u['total'] for _, u in calls], default=0), 'agent': 'codex', 'model': model,
            'effort': effort, 'weekly_pct': wk,
            'usd_basis': f'API list price of {pricing.model} (configs/wsp/pricing.yaml); the ChatGPT plan pays',
            'priced_model_matches': bool(model and pricing.model and model == pricing.model)}


def weekly_reading(since_s=8 * 86400):
    best = None
    for f in rollout_files(time.time() - since_s):
        R = parse_rollout(f)
        if R and R['limits'] and (best is None or R['limits'][-1][0] > best[0]):
            best = R['limits'][-1]
    return best                                   # (t, used_percent, resets_at) or None


def weekly_guard(cap_points):
    """Refuse new runs once WSP has used cap_points of this week's limit (baseline = first WSP start)."""
    r = weekly_reading()
    if r is None:
        return None, 'weekly limit unknown (no Codex session log with rate limits found)'
    _, pct, resets = r
    path = os.path.join(CODEX_DIR, 'weekly.json')
    rec = _json(path) if os.path.exists(path) else {}
    key = str(int(round(float(resets) / 3600) * 3600)) if resets else 'unknown'
    base = rec.setdefault(key, {'baseline_pct': pct, 'since': time.strftime('%F %T')})['baseline_pct']
    os.makedirs(CODEX_DIR, exist_ok=True)
    _write(path, rec)
    used = pct - base
    msg = f'weekly limit {pct:.1f}% used; WSP used {used:.1f} of {cap_points} points this week'
    return (used >= cap_points), msg


AUDIT_RULES = [
    ('read bundle data directly', re.compile(r'wsp_codex/bundle|arrays\.npz|/rounds/(train|val)|checkpoints/')),
    ('touched run state files', re.compile(r'(state\.json|latest\.npz|staged\.npz|committed\.npz|cur_val\.pkl|'
                                           r'probe_cache\.pkl|stage1\.npz)')),
    ('used torch outside wspctl', re.compile(r'\bimport torch\b|\btorch\.(nn|optim|autograd|load|save)\b')),
    ('gradient / optimizer code', re.compile(r'\.backward\(|\boptim\.|requires_grad')),
]


def audit(rollouts, start, end):
    """Tool calls Codex made during the run, with flags for anything the rules forbid (for human review)."""
    n, n_ctl, flags = 0, 0, []
    for p in rollouts or []:
        R = parse_rollout(p)
        if not R:
            continue
        for t, txt in R['tools']:
            if t < start - 1 or (end and t > end + 1):
                continue
            n += 1
            n_ctl += 'wspctl' in txt
            for name, rx in AUDIT_RULES:
                m = rx.search(txt)
                if m:
                    lo = max(0, m.start() - 80)
                    flags.append({'t': t, 'rule': name, 'excerpt': txt[lo:m.end() + 80]})
    return {'tool_calls': n, 'wspctl_calls': n_ctl, 'flags': flags}


# ====================================================================== one run, one command at a time

class LazyAdapter:
    """Builds the adapter (loads data, model) only when a command needs a forward pass."""

    def __init__(self, build, on_build=None):
        self._build, self._on_build, self._a = build, on_build, None

    @property
    def built(self):
        return self._a is not None

    @property
    def counters(self):
        return self._a.counters if self._a is not None else {'compute_s': 0.0, 'train_steps': 0, 'train_flops': 0}

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if self._a is None:
            self._a = self._build()
            if self._on_build:
                self._on_build(self._a)
        return getattr(self._a, name)


class LazySandbox:
    """numpy kernel started on first use; the staged W is kept on disk between commands."""

    def __init__(self, job):
        self.job, self.sb = job, None
        self.note = None

    def _get(self):
        if self.sb is None:
            j, st = self.job, self.job.st
            c = j.s2['sandbox']
            self.sb = Sandbox(os.path.join(j.dir, 'sandbox'), c['timeout_s'], c['mem_gb'], threads=c['threads'])
            self.sb.start(st.W_cur, st.W_start, staged_path=j.p['staged'])
            if self.sb.init_note and self.sb.init_note != 'loaded':
                self.note = 'staged weights reset to committed: ' + self.sb.init_note
        return self.sb

    def exec(self, code):
        return self._get().exec(code)

    def staged(self):
        return self._get().staged()

    def commit(self, W):
        self._get().commit(W)

    def close(self):
        """Persist the staged W (if valid) and stop the kernel."""
        if self.sb is None:
            return None
        W, err = self.sb.staged()
        if err is None:
            save_npz(self.job.p['staged'] + '.tmp.npz', W)
            os.replace(self.job.p['staged'] + '.tmp.npz', self.job.p['staged'])
        self.sb.close()
        self.sb = None
        return err


class CodexJob(Job):
    FILES = {'meta': 'codex.json', 'stage1': 'stage1.npz', 'cur_val': 'cur_val.pkl', 'card': 'card.json',
             'staged': 'staged.npz', 'probes': 'probes.json', 'probe_cache': 'probe_cache.pkl', 'audit': 'audit.json'}

    def __init__(self, job, device=None):
        s2 = load_yaml(STAGE2)
        dev = device or os.environ.get('WSP_DEVICE') or (s2.get('codex') or {}).get('device', 'cpu')
        super().__init__(job, None, device=dev)
        self.p = {k: os.path.join(self.dir, v) for k, v in self.FILES.items()}
        self.st = None
        self.sandbox = None
        self._threshold = None

    # ------------------------------------------------------------ files
    def meta(self):
        return _json(self.p['meta']) if os.path.exists(self.p['meta']) else None

    def write_meta(self, m):
        _write(self.p['meta'], m)

    def started(self):
        return os.path.exists(self.p['meta'])

    def done(self):
        return os.path.exists(os.path.join(self.dir, 'done.json'))

    def _budget(self):
        b, s2 = self.s2['budgets'][self.job['budget']], self.s2
        return Budget(b['probes'], b['gate_evals'], token_cap=b['token_cap'],
                      stop_after_rejects=s2['stop_after_rejects'])

    def _build(self):
        return bundle_adapter(self.job, self.s2, self.device)

    def _save(self, st):
        """Same files as the server runner (latest.npz, state.json, edits.jsonl) plus the CLI's caches."""
        a = st.adapter
        self._counters(st)
        if getattr(a, 'built', True):
            self._threshold = getattr(getattr(a, 'method', None), 'threshold', self._threshold)
        save_npz(os.path.join(self.dir, 'latest.npz'), st.W_cur)
        _write(os.path.join(self.dir, 'state.json'), {
            'budget': st.budget.state(), 'cost': st.cost.state(), 'n_edits': len(st.edits),
            'threshold': self._threshold})
        with open(os.path.join(self.dir, 'edits.jsonl'), 'w') as f:
            for e in st.edits:
                f.write(dumps(e) + '\n')
        with open(self.p['cur_val'], 'wb') as f:
            pickle.dump(st.cur_val, f)
        _write(self.p['probes'], st.probes)
        with open(self.p['probe_cache'], 'wb') as f:
            pickle.dump(st._cur_probe, f)

    # ------------------------------------------------------------ lifecycle
    def begin(self, rollout):
        """First start: Stage-1 validation score, files, and a check against the server's score."""
        from .tools import RunState
        t0 = time.perf_counter()
        a = self._build()
        W_start = a.weights()
        val = a.evaluate('val')
        setup_s = time.perf_counter() - t0
        check = compare_reports(val.report, a.bundle_info.get('val_report'))
        save_npz(self.p['stage1'], W_start)
        with open(self.p['cur_val'], 'wb') as f:
            pickle.dump(val, f)
        card = a.describe()
        _write(self.p['card'], card)
        log = RunLog(os.path.join(self.dir, 'transcript.jsonl'))
        st = RunState(a, self._budget(), CostTracker(self.job['id']), log, {'agent': self.s2['agent']},
                      {**DEFAULT_GATE, **self.s2['gate']}, W_start, {k: v.copy() for k, v in W_start.items()}, val,
                      card=card, val_stage1=val.report)
        self._base = {}
        self._save(st)
        now = time.time()
        self.write_meta({'job': self.job['id'], 'agent': 'codex', 'start': now,
                         'start_iso': time.strftime('%F %T', time.localtime(now)), 'end': None,
                         'rollouts': [rollout] if rollout else [], 'device': str(self.device),
                         'host': platform.node(), 'setup_s': round(setup_s, 2), 'val_stage1': val.report,
                         'bundle_check': check})
        log.event('start', job=self.job, agent='codex', val_stage1=val.report, n_params=a.n_params(),
                  bundle_check=check, setup_s=round(setup_s, 1))
        return check

    def open(self, rollout=None):
        """RunState from the run's files. The adapter and the sandbox start only if a command needs them."""
        from .tools import RunState
        m = self.meta()
        if rollout and rollout not in m['rollouts']:
            m['rollouts'].append(rollout)
            self.write_meta(m)
        s = _json(os.path.join(self.dir, 'state.json'))
        with open(self.p['cur_val'], 'rb') as f:
            cur_val = pickle.load(f)
        budget = self._budget()
        st_holder = {}
        lazy = LazyAdapter(self._build, on_build=lambda a: (a.load_weights(st_holder['st'].W_cur),
                                                            a.on_accept(st_holder['st'].cur_val)))
        st = RunState(lazy, budget, CostTracker(self.job['id']), RunLog(os.path.join(self.dir, 'transcript.jsonl')),
                      {'agent': self.s2['agent']}, {**DEFAULT_GATE, **self.s2['gate']}, load_npz(self.p['stage1']),
                      load_npz(os.path.join(self.dir, 'latest.npz')), cur_val, card=_json(self.p['card']),
                      val_stage1=m['val_stage1'], on_commit=self._save, on_save=self._save)
        st_holder['st'] = st
        budget.restore(s['budget'])
        budget.elapsed_before = time.time() - m['start']         # wall time since the run started
        st.cost.restore(s['cost'])
        h = s['cost'].get('harness', {})
        self._base = {k: h.get(k, 0) or 0 for k in ('compute_s', 'train_steps', 'train_flops')}
        self._threshold = s.get('threshold')
        ep = os.path.join(self.dir, 'edits.jsonl')
        st.edits = [json.loads(line) for line in open(ep) if line.strip()] if os.path.exists(ep) else []
        st.probes = _json(self.p['probes']) if os.path.exists(self.p['probes']) else []
        if os.path.exists(self.p['probe_cache']):
            with open(self.p['probe_cache'], 'rb') as f:
                st._cur_probe = pickle.load(f)
        self.sandbox = LazySandbox(self)
        st.sandbox = self.sandbox
        self.st = st
        return st

    def close(self):
        err = self.sandbox.close() if self.sandbox else None
        if self.st is not None:
            self._save(self.st)
        return err

    # ------------------------------------------------------------ tokens
    def run_record(self):
        m = self.meta()
        return {'id': self.job['id'], 'start': m['start'], 'end': m.get('end'), 'rollouts': m.get('rollouts', [])}

    def usage(self, others=()):
        recs = [self.run_record()] + [o for o in others if o['id'] != self.job['id']]
        calls = attribute(recs)[self.job['id']]
        m = self.meta()
        return summarize_usage(calls, m.get('rollouts'), m['start'], m.get('end'))

    def apply_usage(self, st, u):
        st.cost.llm.update({k: u[k] for k in u if k not in ('weekly_pct',)})
        st.cost.llm['weekly_pct'] = u.get('weekly_pct')
        st.budget.tokens_used = u['total']
        st.budget.calls = u['calls']

    def finish(self, summary, rollout=None, others=()):
        st = self.st or self.open(rollout)
        st.finished, st.finish_summary = True, summary
        m = self.meta()
        m['end'] = time.time()
        self.write_meta(m)
        self.sandbox.close()
        u = self.usage(others)
        self.apply_usage(st, u)
        stop = 'finished'
        self._finalize(st, stop, SimpleNamespace(report=m['val_stage1']), m.get('setup_s'))
        au = audit(m.get('rollouts'), m['start'], m['end'])
        _write(self.p['audit'], au)
        self.annotate(u, au)
        return u, au

    def annotate(self, u, au=None):
        """Write Codex usage into cost.json / wsp.json (also after the fact, when `pack` re-attributes)."""
        cp, wp = os.path.join(self.dir, 'cost.json'), os.path.join(self.dir, 'wsp.json')
        if os.path.exists(cp):
            c = _json(cp)
            c['stage2']['llm'].update(u)
            c['agent'] = 'codex'
            _write(cp, c)
        if os.path.exists(wp):
            w = _json(wp)
            w['agent'] = 'codex'
            w['codex'] = {'model': u.get('model'), 'effort': u.get('effort'), 'tokens': u['total'],
                          'calls': u['calls'], 'weekly_pct': u.get('weekly_pct')}
            if au is not None:
                w['codex']['audit'] = {'tool_calls': au['tool_calls'], 'wspctl_calls': au['wspctl_calls'],
                                       'flags': len(au['flags'])}
            _write(wp, w)


def codex_jobs():
    """{job id: job} for the bundled jobs that exist in this checkout's sweep config."""
    man = bundle_manifest()
    jobs = all_jobs()
    out = {}
    for jid, rec in man['jobs'].items():
        if jid in jobs:
            out[jid] = jobs[jid]
    return out, man


def run_records(jobs):
    out = []
    for j in jobs.values():
        p = os.path.join(j['run_dir'], 'codex.json')
        if os.path.exists(p):
            m = _json(p)
            out.append({'id': j['id'], 'start': m['start'], 'end': m.get('end'), 'rollouts': m.get('rollouts', [])})
    return out
