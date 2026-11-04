"""Codex mode: bundle export -> identical data and scores locally, session-log token accounting, and the wspctl
command line end to end (start, py, probe, submit, finish, pack) with a fake Codex session log."""
import datetime
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from tests.wsp.helpers import ROOT, freq_arrays, tiny_rounds

from attacker_detector.wsp import codex as cx
from attacker_detector.wsp.jobs import STAGE2, load_yaml, mv_key


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat().replace('+00:00', 'Z')


class FakeCodex:
    """Writes a Codex-style session log (only the fields the parser reads)."""

    def __init__(self, home, name='rollout-2026-11-03T10-00-00-test.jsonl', pct=10.0):
        self.path = os.path.join(home, 'sessions', '2026', '11', '03', name)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.n, self.cum, self.pct = 0, {'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0,
                                         'reasoning_output_tokens': 0, 'total_tokens': 0}, pct

    def _w(self, typ, payload, t=None):
        with open(self.path, 'a') as f:
            f.write(json.dumps({'timestamp': iso(t or time.time()), 'type': typ, 'payload': payload}) + '\n')

    def user(self, t=None):
        self._w('response_item', {'type': 'message', 'role': 'user', 'content': []}, t)

    def context(self, model='gpt-6-astra', effort='xhigh', t=None):
        self._w('turn_context', {'model': model, 'effort': effort}, t)

    def call(self, inp=1000, cached=600, out=200, reasoning=120, cmd=None, t=None, dup=False):
        self.n += 1
        u = {'input_tokens': inp, 'cached_input_tokens': cached, 'output_tokens': out,
             'reasoning_output_tokens': reasoning, 'total_tokens': inp + out}
        for k in self.cum:
            self.cum[k] += u[k]
        self.pct += 0.01
        for _ in range(2 if dup else 1):
            self._w('token_usage_record', {'response_id': f'resp{self.n}', 'usage': u}, t)
        self._w('event_msg', {'type': 'token_count', 'info': {'total_token_usage': dict(self.cum)},
                              'rate_limits': {'primary': {'used_percent': self.pct, 'window_minutes': 10080,
                                                          'resets_at': 1793149455}}}, t)
        if cmd:
            self._w('response_item', {'type': 'custom_tool_call', 'name': 'exec',
                                      'input': f'text(await tools.exec_command({{cmd:"{cmd}"}}))'}, t)
        return u['total_tokens']


class TestRolloutParsing(unittest.TestCase):
    def test_attribution_lead_in_tail_and_user_boundaries(self):
        home = tempfile.mkdtemp()
        fc = FakeCodex(home)
        t0 = time.time() - 1000
        fc.user(t0)
        fc.context(t=t0 + 1)
        lead = fc.call(t=t0 + 2)                     # reads the task file before `wspctl next`
        a1 = fc.call(t=t0 + 10, dup=True)            # duplicate record of one response: counted once
        a2 = fc.call(t=t0 + 20)
        gap = fc.call(t=t0 + 35)                     # between runs -> lead-in of run 2
        b1 = fc.call(t=t0 + 50)
        tail = fc.call(t=t0 + 70)                    # after run 2's finish, before the next user message
        fc.user(t0 + 80)
        fc.call(t=t0 + 90)                           # new user turn -> not WSP
        runs = [{'id': 'r1', 'start': t0 + 5, 'end': t0 + 30, 'rollouts': [fc.path]},
                {'id': 'r2', 'start': t0 + 40, 'end': t0 + 60, 'rollouts': [fc.path]}]
        res = cx.attribute(runs)
        tot = {k: sum(u['total'] for _, u in v) for k, v in res.items()}
        self.assertEqual(tot['r1'], lead + a1 + a2)
        self.assertEqual(tot['r2'], gap + b1 + tail)
        s = cx.summarize_usage(res['r1'], [fc.path], t0 + 5, t0 + 30)
        self.assertEqual((s['model'], s['effort'], s['calls']), ('gpt-6-astra', 'xhigh', 3))
        self.assertEqual(s['cached'], 3 * 600)
        self.assertGreater(s['usd'], 0)
        self.assertTrue(s['priced_model_matches'])

    def test_old_format_uses_running_total(self):
        home = tempfile.mkdtemp()
        p = os.path.join(home, 'r.jsonl')
        t = time.time()
        with open(p, 'w') as f:
            for i, tot in enumerate((100, 100, 250)):            # repeated reading = no new call
                f.write(json.dumps({'timestamp': iso(t + i), 'type': 'event_msg', 'payload': {
                    'type': 'token_count', 'info': {'total_token_usage': {'input_tokens': tot, 'output_tokens': 0,
                                                                          'total_tokens': tot}}}}) + '\n')
            f.write('not json\n')
        R = cx.parse_rollout(p)
        self.assertEqual([u['total'] for _, u in R['calls']], [100, 150])


class TestKernelStaged(unittest.TestCase):
    def test_staged_resume_and_mismatch(self):
        from attacker_detector.wsp.sandbox import Sandbox, save_npz
        d = tempfile.mkdtemp()
        W = {'a': np.zeros((2, 3), np.float32)}
        good, bad = os.path.join(d, 'good.npz'), os.path.join(d, 'bad.npz')
        save_npz(good, {'a': np.ones((2, 3), np.float32)})
        save_npz(bad, {'a': np.ones((3, 3), np.float32)})
        with Sandbox(os.path.join(d, 'sb1')).start(W, W, staged_path=good) as sb:
            self.assertEqual(sb.init_note, 'loaded')
            self.assertEqual(float(sb.staged()[0]['a'].sum()), 6.0)
        with Sandbox(os.path.join(d, 'sb2')).start(W, W, staged_path=bad) as sb:
            self.assertIn('ignored', sb.init_note)
            self.assertEqual(float(sb.staged()[0]['a'].sum()), 0.0)


# ---------------------------------------------------------------- export -> local rebuild

def _store_rounds(root, train, val):
    """Rounds written to and read back from a RoundStore, as on the server."""
    from attacker_detector.mean_variance.evaluate import rounds_from_store
    from attacker_detector.mean_variance.store import RoundStore, shard_name
    store = RoundStore(root)
    man = {}
    for split, rounds in (('train', train), ('val', val)):
        RoundStore._save_shard(os.path.join(root, split, shard_name('PM', 1.0, 'taxi')), {
            'reports': [r.observation['reports'] for r in rounds], 'groups': [r.observation['groups'] for r in rounds],
            'is_fake': [r.is_fake for r in rounds], 'round_ids': [r.round_id for r in rounds]})
        for r in rounds:
            nf = r.meta['n_fake']
            man[r.round_id] = {'round_id': r.round_id, 'source': 'taxi', 'labels': r.labels, 'meta': r.meta,
                               'population_id': 'p0', 'domain': [0.0, 100.0], 'square_mode': 'raw',
                               'beta': r.meta['realized_beta'], 'attack': 'IPA' if nf else 'none', 'target': 'm1v1'}
    with open(store.manifest_path, 'w') as f:
        for rec in man.values():
            f.write(json.dumps(rec) + '\n')
    m = store.manifest()
    return (rounds_from_store(store, m, 'train', 'PM', 1.0, ['taxi']),
            rounds_from_store(store, m, 'val', 'PM', 1.0, ['taxi']), m)


class FakeCtx:
    def __init__(self, train, val, manifest, ck):
        self.train, self.val, self.manifest, self.ck = train, val, manifest, ck

    def rounds(self, split, protocol, eps, sources):
        return list(self.train if split == 'train' else self.val)

    def checkpoint(self, key):
        return self.ck[key]

    def fit_time(self, key):
        return 12.5


def build_world(tmp, methods=('ft_removal', 'deepsets_recovery')):
    """Server-like Stage 1 (tiny): rounds in a store, fitted checkpoints, and the matching jobs."""
    from attacker_detector.mean_variance.methods import ClassifierRemoval, SetRecovery
    from attacker_detector.mean_variance.methods.checkpoint import save_checkpoint
    train, val, man = _store_rounds(os.path.join(tmp, 'server_rounds'), tiny_rounds('PM', 14, 100),
                                    tiny_rounds('PM', 6, 300))
    ck, jobs = {}, []
    for m_name in methods:
        torch.manual_seed(0)
        m = (ClassifierRemoval('ft', epochs=1, reports_per_round=64, device='cpu') if m_name == 'ft_removal' else
             SetRecovery('deepsets', epochs=1, device='cpu', token_cap=256))
        m.fit(train, val)
        key = mv_key('PM', 1.0, 'taxi', m_name, 0)
        ck[key] = os.path.join(tmp, 'ck', key)
        save_checkpoint(m, ck[key])
        jobs.append({'id': f'mv/wsp_b10/{key}', 'kind': 'mv', 'mode': 'wsp', 'budget': 10, 'unit': key,
                     'method': m_name, 'protocol': 'PM', 'eps': 1.0, 'train_sources': ['taxi'], 'seed': 0,
                     'set': 'tier_a', 'run_dir': os.path.join(tmp, 'mv_out', 'stage2', 'wsp_b10', key)})
    return FakeCtx(train, val, man, ck), jobs


def s2_small():
    s2 = load_yaml(STAGE2)
    s2['probe'] = {'rounds': 6, 'rows': 500, 'seed': 0}
    return s2


class TestBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.ctx, cls.jobs = build_world(cls.tmp)
        cls.s2 = s2_small()
        cls.bundle = os.path.join(cls.tmp, 'bundle')
        from attacker_detector.wsp import runner
        orig = runner.build_adapter

        def fake_build(job, sweep_cfg, s2, device):
            if job['kind'] == 'freq':
                from attacker_detector.wsp.adapters import FreqFTAdapter, build_freq_model
                arr = freq_arrays()
                spec = {'stage1_dir': cls.tmp, 'run': 'cross/OUE_Fire_test', 'seed': 42}
                torch.manual_seed(0)
                return FreqFTAdapter(build_freq_model(spec, 4), spec, arr, 'cpu', probe_rows=500, seed=0)
            return orig(job, sweep_cfg, s2, device)
        cls.freq_job = {'id': 'freq/wsp_b25/cross__OUE_Fire_test__s42', 'kind': 'freq', 'mode': 'wsp', 'budget': 25,
                        'run': 'cross/OUE_Fire_test', 'seed': 42, 'set': 'tier_a', 'run_dir': os.path.join(cls.tmp, 'f')}
        with mock.patch.object(runner, 'mv_context', lambda cfg: cls.ctx), \
                mock.patch.object(runner, 'build_adapter', fake_build):
            cls.man = cx.export_bundle(cls.jobs + [cls.freq_job], None, cls.s2, torch.device('cpu'), out=cls.bundle,
                                       log=lambda s: None)
            cls.server = {j['id']: fake_build(j, None, cls.s2, torch.device('cpu')) for j in cls.jobs + [cls.freq_job]}

    def test_manifest(self):
        self.assertEqual(len(self.man['units']), 3)
        self.assertEqual(self.man['cells']['PM_eps1_taxi'], {'probe_rounds': 6, 'val_rounds': 6, 'train_rounds': 14})
        self.assertEqual(set(self.man['jobs']), {j['id'] for j in self.jobs + [self.freq_job]})

    def test_local_rebuild_is_identical(self):
        with mock.patch.dict(os.environ, {'WSP_BUNDLE': self.bundle}):
            for j in self.jobs + [self.freq_job]:
                loc = cx.bundle_adapter(j, self.s2, torch.device('cpu'))
                srv = self.server[j['id']]
                for split in ('probe', 'val'):
                    a, b = loc.evaluate(split), srv.evaluate(split)
                    for f in ('units', 'pred', 'y'):
                        if getattr(b, f, None) is not None:
                            np.testing.assert_allclose(getattr(a, f), getattr(b, f), rtol=1e-6, atol=1e-7,
                                                       err_msg=f'{j["id"]} {split} {f}')
                    self.assertEqual(a.report, b.report)
                chk = cx.compare_reports(loc.evaluate('val').report, loc.bundle_info['val_report'])
                self.assertTrue(chk['ok'], chk)
                self.assertEqual(loc.stage1_cost(), srv.stage1_cost())
                self.assertEqual(loc.eval_flops('val'), srv.eval_flops('val'))
                if j['kind'] == 'mv':
                    self.assertEqual([m['round_id'] for m in loc.data['probe']['meta']],
                                     [m['round_id'] for m in srv.data['probe']['meta']])
                else:
                    np.testing.assert_array_equal(loc.data['probe'][0], srv.data['probe'][0])

    def test_stage1_ready(self):
        from attacker_detector.wsp import runner
        self.ctx.out_dir = os.path.join(self.tmp, 'stage1')
        done, todo = self.jobs
        os.makedirs(os.path.join(self.ctx.out_dir, 'units'), exist_ok=True)
        with open(os.path.join(self.ctx.out_dir, 'units', done['unit'] + '.done.json'), 'w') as f:
            json.dump({'status': 'completed'}, f)
        with open(os.path.join(self.ctx.out_dir, 'units', todo['unit'] + '.done.json'), 'w') as f:
            f.write('{"stat')                                            # being written
        fdir = os.path.join(self.tmp, 'freq_s1')
        os.makedirs(fdir, exist_ok=True)
        fj = {'kind': 'freq', 'spec': {'stage1_dir': fdir, 'model_path': os.path.join(fdir, 'model.pt')}}
        with mock.patch.object(runner, 'mv_context', lambda cfg: self.ctx):
            self.assertTrue(cx.stage1_ready(done, None))
            self.assertFalse(cx.stage1_ready(todo, None))
            self.assertFalse(cx.stage1_ready(fj, None))
            for name in ('model.pt', 'sensitivity_test_results.csv'):
                open(os.path.join(fdir, name), 'w').close()
            self.assertTrue(cx.stage1_ready(fj, None))

    def test_no_test_or_extra_training_data(self):
        from attacker_detector.mean_variance.store import RoundStore
        st = RoundStore(os.path.join(self.bundle, 'mv', 'rounds'))
        self.assertEqual(len(st.load('train', 'PM', 1.0, 'taxi')), 6)          # probe subset only
        self.assertEqual(st.load('test', 'PM', 1.0, 'taxi'), [])
        with np.load(os.path.join(self.bundle, 'freq', 'cross__OUE_Fire_test__s42', 'arrays.npz')) as z:
            self.assertEqual(len(z['X_train']), 500)
            self.assertNotIn('X_test', z.files)


class TestWspctl(unittest.TestCase):
    """The real command line in a subprocess, against a bundle and a fake Codex session log."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tmp = tempfile.mkdtemp()
        ctx, jobs = build_world(tmp, methods=('deepsets_recovery',))
        s2 = s2_small()
        from attacker_detector.wsp import runner
        bundle = os.path.join(tmp, 'bundle')
        with mock.patch.object(runner, 'mv_context', lambda cfg: ctx):
            cx.export_bundle(jobs, None, s2, torch.device('cpu'), out=bundle, log=lambda s: None)
        grid = {'protocols': ['PM'], 'epsilon': [1.0], 'train_sources': ['taxi'], 'seeds': [0], 'budgets': [10]}
        none = dict(grid, protocols=[])
        sweep = load_yaml(os.path.join(ROOT, 'configs', 'wsp', 'sweep.yaml'))
        sweep['mv'].update(methods=['deepsets_recovery'], out=os.path.join(tmp, 'mv_out'), tier_a=grid,
                           budget_curve=none, tier_b=none)
        sweep['freq']['runs'] = []
        sweep['pilot'] = []
        s2['budgets'][10] = {'probes': 3, 'gate_evals': 2, 'token_cap': 50000}
        with open(os.path.join(tmp, 'sweep.yaml'), 'w') as f:
            yaml.safe_dump(sweep, f)
        with open(os.path.join(tmp, 'stage2.yaml'), 'w') as f:
            yaml.safe_dump(s2, f)
        cls.home = os.path.join(tmp, 'codex_home')
        cls.env = {**os.environ, 'WSP_SWEEP': os.path.join(tmp, 'sweep.yaml'), 'WSP_STAGE2': os.path.join(tmp, 'stage2.yaml'),
                   'WSP_BUNDLE': bundle, 'WSP_CODEX_DIR': os.path.join(tmp, 'codex'), 'CODEX_HOME': cls.home,
                   'WSP_DEVICE': 'cpu'}
        cls.job = jobs[0]
        cls.fc = FakeCodex(cls.home)
        other = FakeCodex(cls.home, name='rollout-other.jsonl')             # an unrelated Codex session
        other.call(inp=99999, cmd='ls')
        cls.fc.user()
        cls.fc.context()

    def ctl(self, *args, tokens=0):
        if tokens:
            self.fc.call(inp=tokens, cached=tokens // 2, out=100,
                         cmd='scripts/wsp/wspctl ' + ' '.join(a if ' ' not in a else repr(a) for a in args))
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'wsp', 'wspctl.py'), *args],
                           env=self.env, capture_output=True, text=True, cwd=self.tmp, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_full_run(self):
        out = self.ctl('queue')
        self.assertIn('todo', out)
        out = self.ctl('next', tokens=1000)
        self.assertIn('started ' + self.job['id'], out)
        self.assertIn('matches the server', out)
        self.assertIn('budget: 3 probes', out)
        self.assertIn('committed_validation', out)
        out = self.ctl('next')                                            # active run -> resumed, not a new one
        self.assertIn('resumed', out)
        with open(os.path.join(self.tmp, 'edit.py'), 'w') as f:
            f.write("k = names()[0][0]\nW[k] = W[k] * np.float32(1.001)\nprint('edited', k)\n")
        out = self.ctl('py', 'edit.py', tokens=2000)
        self.assertIn('edited', out)
        out = self.ctl('py', '-c', 'print(diff()["changed"])', tokens=500)  # staged edit survived the restart
        self.assertGreater(int(out.strip().splitlines()[-1]), 0)
        out = self.ctl('stats', '--staged')
        self.assertIn('mean', out)
        out = self.ctl('probe', 'metrics', tokens=1500)
        self.assertEqual(json.loads(out)['probes_left'], 2)
        self.fc.call(cmd='cat ' + os.path.join(self.env['WSP_BUNDLE'], 'mv', 'rounds', 'val', 'PM_eps1_taxi.npz'))
        out = self.ctl('submit', 'scale first tensor by 1.001 to test the gate', tokens=800)
        v = json.loads(out)
        self.assertIn('accept', v)
        self.assertEqual(v['budget']['gate_evals_left'], 1)
        out = self.ctl('status')
        self.assertGreater(json.loads(out)['budget']['tokens_used'], 0)
        out = self.ctl('finish', 'tested the loop', tokens=300)
        self.assertIn('finished ' + self.job['id'], out)
        self.fc.call(inp=50, cached=0, out=10)                             # Codex's closing message (tail)
        rd = self.job['run_dir']
        for f in ('done.json', 'wsp.json', 'cost.json', 'audit.json', 'checkpoint/state.json', 'edits.jsonl'):
            self.assertTrue(os.path.exists(os.path.join(rd, f)), f)
        cost = json.load(open(os.path.join(rd, 'cost.json')))
        llm = cost['stage2']['llm']
        self.assertEqual(llm['agent'], 'codex')
        self.assertEqual(llm['model'], 'gpt-6-astra')
        self.assertNotIn(99999, [llm['input']])                             # other session not counted
        audit = json.load(open(os.path.join(rd, 'audit.json')))
        self.assertTrue(any(f['rule'] == 'read bundle data directly' for f in audit['flags']))
        self.assertGreaterEqual(audit['wspctl_calls'], 5)
        self.assertFalse(os.path.exists(os.path.join(self.env['WSP_CODEX_DIR'], 'active.json')))
        out = self.ctl('next')
        self.assertIn('queue empty', out)
        out = self.ctl('pack')
        self.assertIn('wrote', out)
        llm2 = json.load(open(os.path.join(rd, 'cost.json')))['stage2']['llm']
        self.assertEqual(llm2['total'], llm['total'] + 60)                  # tail added at pack time
        expect = sum(t + 100 for t in (1000, 2000, 500, 1500, 800, 300)) + (self.fc_extra_call_total()) + 60
        self.assertEqual(llm2['total'], expect)
        tars = [f for f in os.listdir(self.env['WSP_CODEX_DIR']) if f.endswith('.tar.gz')]
        with tarfile.open(os.path.join(self.env['WSP_CODEX_DIR'], tars[0])) as t:
            names = t.getnames()
        self.assertTrue(any(n.endswith('wsp.json') for n in names))
        self.assertFalse(any('sandbox' in n or n.endswith('.pkl') for n in names))

    def fc_extra_call_total(self):
        return 1000 + 200                                                   # the `cat bundle` call (defaults)


if __name__ == '__main__':
    unittest.main()
