"""Run one Stage-2 job: build the adapter from its Stage-1 checkpoint, then edit with the LLM agent
(mode 'wsp') or with a baseline ('finetune': gated gradient epochs; 'random': gated random perturbations),
under the same probe / gate budget. Writes, under the job's run_dir:

  checkpoint/      edited model in the existing checkpoint format (+ run_spec.json for freq)
  wsp.json         summary: stop reason, Stage-1 vs final validation metrics, edits, budget
  cost.json        tokens / $ / forward passes / FLOPs / GPU time for Stage 2, and Stage-1 training cost
  edits.jsonl, transcript.jsonl, llm_calls.jsonl, state.json, latest.npz, done.json
A crashed or interrupted run resumes from latest.npz and state.json (tokens already spent stay counted).
"""

import json
import os
import time
import traceback

import numpy as np
import torch

from .adapters import FreqFTAdapter, FTRemovalAdapter, MVContext, SetRecoveryAdapter
from .budget import Budget
from .cost import CostTracker, Ledger, Pricing
from .gate import DEFAULT_GATE, edit_stats
from .jobs import STAGE2, load_yaml, rel
from .log import RunLog, dumps
from .sandbox import Sandbox, load_npz, save_npz
from .tools import RunState, make_tools

_MV_CTX = {}


def device_for(pref='auto'):
    if pref != 'auto':
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def mv_context(sweep_cfg):
    mc = sweep_cfg['mv']
    k = (mc['config'], mc['stage1_root'])
    if k not in _MV_CTX:
        _MV_CTX[k] = MVContext(rel(mc['config']), rel(mc['stage1_root']))
    return _MV_CTX[k]


def build_adapter(job, sweep_cfg, s2, device):
    if job['kind'] == 'mv':
        from ..mean_variance.methods.checkpoint import load_checkpoint
        ctx = mv_context(sweep_cfg)
        ck = ctx.checkpoint(job['unit'])
        if not os.path.exists(os.path.join(ck, 'state.json')):
            raise FileNotFoundError(f'Stage-1 checkpoint missing: {ck} (run stage1_mv first)')
        method = load_checkpoint(ck, str(device))
        train = ctx.rounds('train', job['protocol'], job['eps'], job['train_sources'])
        val = ctx.rounds('val', job['protocol'], job['eps'], job['train_sources'])
        pr = s2['probe']
        cls = FTRemovalAdapter if job['method'].startswith('ft_removal') else SetRecoveryAdapter
        a = cls(method, train, val, ctx.manifest, probe_rounds=pr['rounds'], seed=pr['seed'])
        a.stage1_fit_time = ctx.fit_time(job['unit'])
        return a
    spec = job['spec']
    if not os.path.exists(spec['model_path']):
        raise FileNotFoundError(f"Stage-1 model missing: {spec['model_path']} (run stage1_freq first)")
    fc = sweep_cfg['freq']
    cache = os.path.join(rel('results'), fc['out_root'], 'cache', f"{job['run'].replace('/', '__')}__s{job['seed']}.npz")
    a = FreqFTAdapter.from_spec(spec, spec['model_path'], device, cache, s2['probe']['rows'], s2['probe']['seed'])
    meta = os.path.join(spec['stage1_dir'], 'wsp_stage1.json')
    a.stage1_fit_time = json.load(open(meta)).get('wall_s') if os.path.exists(meta) else None
    return a


def _write(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(dumps(obj, indent=1))
    os.replace(tmp, path)


class Job:
    def __init__(self, job, sweep_cfg, env=None, device='auto', s2_path=None):
        self.job, self.sweep = job, sweep_cfg
        self.s2 = load_yaml(s2_path or STAGE2)
        self.env = env or os.environ
        self.device = device_for(device)
        self.dir = job['run_dir']
        os.makedirs(self.dir, exist_ok=True)

    # ------------------------------------------------------------ persistence
    def _counters(self, st):
        base = getattr(self, '_base', {})
        for k in ('compute_s', 'train_steps', 'train_flops'):
            st.cost.harness[k] = base.get(k, 0) + st.adapter.counters[k]

    def _save(self, st):
        self._counters(st)
        save_npz(os.path.join(self.dir, 'latest.npz'), st.W_cur)
        _write(os.path.join(self.dir, 'state.json'), {
            'budget': st.budget.state(), 'cost': st.cost.state(), 'n_edits': len(st.edits),
            'threshold': getattr(getattr(st.adapter, 'method', None), 'threshold', None)})
        with open(os.path.join(self.dir, 'edits.jsonl'), 'w') as f:
            for e in st.edits:
                f.write(dumps(e) + '\n')

    def _restore(self, st):
        p = os.path.join(self.dir, 'state.json')
        if not os.path.exists(p):
            return False
        s = json.load(open(p))
        st.budget.restore(s['budget'])
        st.cost.restore(s['cost'])
        h = s['cost'].get('harness', {})
        self._base = {k: h.get(k, 0) or 0 for k in ('compute_s', 'train_steps', 'train_flops')}
        ep = os.path.join(self.dir, 'edits.jsonl')
        if os.path.exists(ep):
            st.edits = [json.loads(line) for line in open(ep) if line.strip()]
        W = load_npz(os.path.join(self.dir, 'latest.npz'))
        st.W_cur = W
        st.adapter.load_weights(W)
        st.cur_val = st.adapter.evaluate('val')
        st.adapter.on_accept(st.cur_val)
        st.log.event('resume', state=s['budget'])
        return True

    # ------------------------------------------------------------ main
    def run(self):
        job, s2 = self.job, self.s2
        if os.path.exists(os.path.join(self.dir, 'done.json')):
            return 'skipped'
        b = s2['budgets'][job['budget']]
        is_llm = job['mode'] == 'wsp'
        ledger = Ledger()
        global_cap = int(self.env.get('WSP_TOKEN_CAP') or s2['global_token_cap'])
        budget = Budget(b['probes'], b['gate_evals'], token_cap=b['token_cap'] if is_llm else None,
                        max_calls=s2['max_llm_calls'] if is_llm else None, wall_s=60 * s2['wall_clock_min'],
                        stop_after_rejects=s2['stop_after_rejects'] if is_llm else None,
                        ledger=ledger if is_llm else None, global_cap=global_cap if is_llm else None)
        if is_llm:
            tok, _ = ledger.total()
            if tok >= global_cap:
                raise SystemExit(f'global token cap reached ({tok:,} >= {global_cap:,}); not starting {job["id"]}')
        log = RunLog(os.path.join(self.dir, 'transcript.jsonl'))
        cost = CostTracker(job['id'], Pricing.load() if is_llm else None, ledger if is_llm else None,
                           os.path.join(self.dir, 'llm_calls.jsonl') if is_llm else None)
        t0 = time.perf_counter()
        adapter = build_adapter(job, self.sweep, s2, self.device)
        setup_s = time.perf_counter() - t0
        W_start = adapter.weights()
        val_stage1 = adapter.evaluate('val')
        gate_cfg = {**DEFAULT_GATE, **s2['gate']}
        st = RunState(adapter, budget, cost, log, {'agent': s2['agent']}, gate_cfg, W_start,
                      {k: v.copy() for k, v in W_start.items()}, val_stage1,
                      on_commit=self._save, on_save=self._save, val_stage1=val_stage1.report)
        resumed = self._restore(st)
        log.event('start', job=job, resumed=resumed, val_stage1=val_stage1.report, n_params=adapter.n_params(),
                  setup_s=round(setup_s, 1))
        error = None
        try:
            if job['mode'] == 'wsp':
                stop = self._run_wsp(st)
            elif job['mode'] == 'finetune':
                stop = self._run_finetune(st)
            elif job['mode'] == 'random':
                stop = self._run_random(st)
            else:
                raise ValueError(f"unknown mode {job['mode']!r}")
        except Exception:                       # noqa: BLE001 - keep what was committed, report, allow resume
            error = traceback.format_exc()
            stop = 'error'
        self._save(st)
        if stop == 'global_token_cap':          # weekly cap: pause, resume next week (not done)
            _write(os.path.join(self.dir, 'paused.json'), {'stop': stop, 'time': time.strftime('%F %T')})
            log.event('paused', stop=stop)
            return 'paused'
        if stop in ('llm_error', 'error'):
            _write(os.path.join(self.dir, 'error.json'), {'stop': stop, 'error': error or getattr(self, 'agent_error', '')})
            log.event('stopped_with_error', stop=stop, error=(error or '')[-3000:])
            return stop
        self._finalize(st, stop, val_stage1, setup_s)
        return stop

    def _finalize(self, st, stop, val_stage1, setup_s):
        a = st.adapter
        a.load_weights(st.W_cur)
        a.on_accept(st.cur_val)
        a.save(os.path.join(self.dir, 'checkpoint'))
        h = st.cost.harness
        self._counters(st)
        h['forward_flops'] = (h['probe_evals'] * a.eval_flops('probe') + h['gate_evals'] * a.eval_flops('val'))
        stage1 = {'fit_wall_s': getattr(a, 'stage1_fit_time', None), **a.stage1_cost()}
        drift = edit_stats(st.W_cur, st.W_start)
        accepted = [e for e in st.edits if e.get('accept')]
        summary = {'job': self.job['id'], 'mode': self.job['mode'], 'budget': self.job['budget'], 'stop': stop,
                   'finish_summary': st.finish_summary, 'val_stage1': val_stage1.report,
                   'val_final': st.cur_val.report, 'edits_submitted': len(st.edits), 'edits_accepted': len(accepted),
                   'weights_changed_vs_stage1': drift['changed'], 'frac_changed_vs_stage1': drift['frac_changed'],
                   'rel_l2_vs_stage1': drift['rel_l2'], 'tensors_changed': drift['tensors'][:20],
                   'budget_status': st.budget.status(), 'n_params': a.n_params(), 'setup_s': setup_s}
        _write(os.path.join(self.dir, 'wsp.json'), summary)
        _write(os.path.join(self.dir, 'cost.json'), {'job': self.job['id'], 'mode': self.job['mode'],
                                                      'budget': self.job['budget'], 'stage2': st.cost.state(),
                                                      'stage1': stage1, 'elapsed_s': st.budget.elapsed()})
        _write(os.path.join(self.dir, 'done.json'), {'stop': stop, 'time': time.strftime('%F %T')})
        st.log.event('done', stop=stop, val_final=st.cur_val.report)

    # ------------------------------------------------------------ modes
    def _run_wsp(self, st):
        from .agent import Agent, make_llm
        sb = self.s2['sandbox']
        st.sandbox = Sandbox(os.path.join(self.dir, 'sandbox'), sb['timeout_s'], sb['mem_gb'], threads=sb['threads'])
        st.sandbox.start(st.W_cur, st.W_start)
        try:
            tools, fns = make_tools(st)
            cfg = {'agent': {**self.s2['agent'], 'max_llm_calls': self.s2['max_llm_calls']}}
            agent = Agent(st, make_llm(self.env), tools, fns, cfg)
            stop = agent.run()
            self.agent_error = agent.error
        finally:
            st.sandbox.close()
        return stop

    def _run_finetune(self, st):
        fc = self.s2['finetune']
        a = st.adapter
        steps = max(1, int(round(fc['epochs_per_gate'] * a.epoch_steps())))
        while st.budget.gates_left() > 0 and not st.budget.stop_reason():
            a.load_weights(st.W_cur)
            a.finetune(steps, fc['lr_scale'])
            W = a.weights()
            st.gate(W, f'{fc["epochs_per_gate"]} gradient epoch(s), lr x{fc["lr_scale"]}', caps=False,
                    source='finetune')
            a.load_weights(st.W_cur)
            st.save()
        return st.budget.stop_reason() or 'budget_spent'

    def _run_random(self, st):
        rc, caps = self.s2['random'], st.gate_cfg
        a = st.adapter
        rng = np.random.default_rng(1000 + self.job.get('seed', 0))
        names = list(st.W_cur)
        total = sum(v.size for v in st.W_cur.values())
        cur_probe = a.evaluate('probe')
        st.cost.harness['probe_evals'] += 1
        while st.budget.probes_left() > 0 and st.budget.gates_left() > 0 and not st.budget.stop_reason():
            W = {k: v.copy() for k, v in st.W_cur.items()}
            name = names[rng.integers(len(names))]
            t = W[name]
            k = int(min(t.size, max(1, rng.uniform(0.05, 1.0) * caps['max_changed_frac'] * total)))
            idx = rng.choice(t.size, k, replace=False)
            sigma = float(rng.choice(rc['sigmas'])) * (float(t.std()) + 1e-3)
            delta = np.zeros(t.size, np.float32)
            delta[idx] = rng.normal(0, sigma, k)
            norm_all = np.sqrt(sum(float(np.sum(v.astype(np.float64) ** 2)) for v in st.W_cur.values()))
            lim = 0.9 * caps['max_rel_l2'] * norm_all
            dn = float(np.linalg.norm(delta))
            if dn > lim:
                delta *= lim / dn
            W[name] = (t.reshape(-1) + delta).reshape(t.shape)
            st.budget.charge_probe()
            with a.using(W):
                p = a.evaluate('probe')
            st.cost.harness['probe_evals'] += 1
            if p.primary < cur_probe.primary:
                rec = st.gate(W, f'random perturbation of {name} ({k} entries, sigma {sigma:.3g})', source='random')
                if rec.get('accept'):
                    cur_probe = p
            st.save()
        return st.budget.stop_reason() or 'budget_spent'
