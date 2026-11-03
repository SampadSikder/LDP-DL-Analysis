"""Full agent loop with a scripted fake chat model (no API calls): tools, sandbox, gate, cost, compaction,
plus the two baseline loops."""
import json
import os
import tempfile
import unittest

from tests.wsp.helpers import fitted, tiny_rounds

from attacker_detector.wsp.adapters import SetRecoveryAdapter
from attacker_detector.wsp.budget import Budget
from attacker_detector.wsp.cost import CostTracker, Ledger, Pricing
from attacker_detector.wsp.gate import DEFAULT_GATE
from attacker_detector.wsp.jobs import STAGE2, all_jobs, load_yaml, parse_freq_run
from attacker_detector.wsp.log import RunLog
from attacker_detector.wsp.sandbox import Sandbox
from attacker_detector.wsp.tools import RunState, make_tools

USAGE = {'input_tokens': 1000, 'output_tokens': 120, 'total_tokens': 1120,
         'input_token_details': {'cache_read': 400}, 'output_token_details': {'reasoning': 80}}


def scripted_model(steps):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Scripted(BaseChatModel):
        @property
        def _llm_type(self):
            return 'scripted'

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            i = min(state['i'], len(steps) - 1)
            state['i'] += 1
            state['seen'].append(len(messages))
            name, args = steps[i]
            calls = [] if name is None else [{'name': name, 'args': args, 'id': f'call_{state["i"]}'}]
            return ChatResult(generations=[ChatGeneration(message=AIMessage('', tool_calls=calls,
                                                                            usage_metadata=dict(USAGE)))])

    state = {'i': 0, 'seen': []}
    return Scripted(), state


class TestAgentLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train, val = tiny_rounds('PM', 12, 100), tiny_rounds('PM', 6, 300)
        cls.m = fitted('mv_ds', train, val)
        cls.train, cls.val = train, val

    def state(self, tmp, with_sandbox=True, **bkw):
        a = SetRecoveryAdapter(self.m, self.train, self.val, None, probe_rounds=6)
        W = a.weights()
        s2 = load_yaml(STAGE2)
        led = Ledger(os.path.join(tmp, 'ledger.jsonl'))
        b = Budget(bkw.get('probes', 5), bkw.get('gates', 3), token_cap=bkw.get('cap', 10 ** 6), ledger=led,
                   global_cap=10 ** 7, stop_after_rejects=6, max_calls=50)
        cost = CostTracker('t', Pricing.load(), led, os.path.join(tmp, 'calls.jsonl'))
        cur = a.evaluate('val')
        st = RunState(a, b, cost, RunLog(os.path.join(tmp, 'transcript.jsonl')),
                      {'agent': {**s2['agent'], 'keep_turns': 2}}, {**DEFAULT_GATE, **s2['gate']}, W,
                      {k: v.copy() for k, v in W.items()}, cur, val_stage1=cur.report)
        if with_sandbox:
            st.sandbox = Sandbox(os.path.join(tmp, 'sb'), 20, 2).start(st.W_cur, st.W_start)
        return st

    def test_scripted_run(self):
        from attacker_detector.wsp.agent import Agent
        tmp = tempfile.mkdtemp()
        st = self.state(tmp)
        steps = [('model_card', {}), ('tensor_stats', {'names': ['rho.4.bias']}),
                 ('read_tensor', {'name': 'rho.4.weight', 'index': '0, 0:8'}),
                 ('python', {'code': "W['rho.4.bias'][1] += 0.05\nprint(diff(show=False)['changed'])"}),
                 ('probe', {'kind': 'metrics', 'weights': 'staged'}),
                 ('submit_edit', {'rationale': 'shift variance logit'}), ('history', {}),
                 (None, {}), ('python', {'code': 'revert()'}), ('finish', {'summary': 'test done'})]
        llm, s = scripted_model(steps)
        tools, fns = make_tools(st)
        agent = Agent(st, llm, tools, fns, {'agent': {**st.cfg['agent'], 'max_llm_calls': 50}})
        try:
            stop = agent.run()
        finally:
            st.sandbox.close()
        self.assertEqual(stop, 'finished')
        self.assertEqual(st.cost.llm['calls'], len(steps))
        self.assertEqual(st.cost.llm['total'], len(steps) * 1120)
        self.assertEqual(st.cost.llm['reasoning'], len(steps) * 80)
        self.assertEqual(st.budget.probes_used, 1)
        self.assertEqual(len(st.edits), 1)
        self.assertTrue(st.edits[0]['gate_charged'])
        self.assertEqual(Ledger(os.path.join(tmp, 'ledger.jsonl')).total()[0], len(steps) * 1120)
        with open(os.path.join(tmp, 'transcript.jsonl')) as f:
            events = [json.loads(line) for line in f]
        types = [e['type'] for e in events]
        tool_errors = [(e['name'], e['result'][:200]) for e in events
                       if e['type'] == 'tool' and str(e['result']).startswith('error')]
        self.assertEqual(tool_errors, [])
        probe = [e for e in events if e['type'] == 'tool' and e['name'] == 'probe'][0]
        self.assertIn('committed_overall', json.loads(probe['result']))
        self.assertIn('gate', types)
        self.assertIn('compaction', types)                  # keep_turns=2 forces it
        self.assertLess(max(s['seen']), 20)                 # history stays bounded

    def test_token_cap_stops(self):
        from attacker_detector.wsp.agent import Agent
        tmp = tempfile.mkdtemp()
        st = self.state(tmp, cap=3000)
        llm, s = scripted_model([('history', {})] * 20)
        tools, fns = make_tools(st)
        agent = Agent(st, llm, tools, fns, {'agent': {**st.cfg['agent'], 'max_llm_calls': 50}})
        try:
            stop = agent.run()
        finally:
            st.sandbox.close()
        self.assertEqual(stop, 'token_cap')
        self.assertLessEqual(st.budget.tokens_used, 3000)

    def test_baseline_loops(self):
        from attacker_detector.wsp.runner import Job
        for mode in ('finetune', 'random'):
            tmp = tempfile.mkdtemp()
            st = self.state(tmp, with_sandbox=False, probes=4, gates=2)
            job = object.__new__(Job)
            job.s2, job.job = load_yaml(STAGE2), {'seed': 0}
            stop = job._run_finetune(st) if mode == 'finetune' else job._run_random(st)
            self.assertIn(stop, ('budget_spent', 'gate_budget_spent'))
            self.assertLessEqual(st.budget.gates_used, 2)
            if mode == 'finetune':
                self.assertEqual(len(st.edits), 2)
                self.assertGreater(st.adapter.counters['train_flops'], 0)


class TestJobs(unittest.TestCase):
    def test_grid(self):
        jobs = all_jobs()
        tier_a_wsp = [j for j in jobs.values() if j['set'] == 'tier_a' and j['mode'] == 'wsp']
        self.assertEqual(len(tier_a_wsp), 54 + 18 + 15)
        self.assertEqual(sum(1 for j in jobs.values() if j.get('pilot')), 2)
        self.assertEqual(parse_freq_run('cross/OLH_Server_Fire_test'), ('cross', 'OLH_Server', 'fire'))
        self.assertEqual(parse_freq_run('in_dist/Fire_OUE'), ('none', 'OUE', 'fire'))
        j = jobs['freq/wsp_b25/cross__OUE_Fire_test__s42']
        self.assertEqual(j['spec']['train_dataset'], ['zipf', 'emoji'])
        self.assertTrue(j['spec']['stage1_dir'].endswith('results/oat_b015_mga-a_n500k_v3f/cross/OUE_Fire_test'))
        self.assertIn('stage1_s43', jobs['freq/wsp_b25/cross__OUE_Fire_test__s43']['spec']['stage1_dir'])


if __name__ == '__main__':
    unittest.main()
