"""Sandbox, gate, cost and budget."""
import os
import tempfile
import unittest

import numpy as np

from tests.wsp.helpers import ROOT  # noqa: F401  (sys.path)

from attacker_detector.wsp.budget import Budget
from attacker_detector.wsp.cost import CostTracker, Ledger, Pricing, usage_of
from attacker_detector.wsp.gate import Score, cap_problems, decide, edit_stats, paired_delta
from attacker_detector.wsp.kernel import check_code
from attacker_detector.wsp.sandbox import Sandbox


class TestPolicy(unittest.TestCase):
    def test_allowed(self):
        self.assertEqual(check_code('import numpy as np\nW["a"][0] += 1\nprint(np.linalg.svd(W["a"]))'), [])

    def test_blocked(self):
        for code in ('import os', 'open("x")', 'np.load("x")', 'from numpy import load', 'x.__class__',
                     'import numpy.lib.npyio', '__import__("os")', 'getattr(np, "load")', 'import torch',
                     'from numpy.lib import format', 'import operator', 'eval("1")', 'np.save("f", 1)'):
            self.assertTrue(check_code(code), code)


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.W = {'a': np.arange(6, dtype=np.float32).reshape(2, 3), 'b': np.ones(3, np.float32)}
        self.sb = Sandbox(tempfile.mkdtemp(), timeout_s=10, mem_gb=2).start(self.W, self.W)

    def tearDown(self):
        self.sb.close()

    def test_edit_dump_commit(self):
        r = self.sb.exec('W["a"][0, 0] = 100.0\nprint(diff(show=False)["changed"])')
        self.assertTrue(r['ok'], r)
        self.assertEqual(r['out'].strip(), '1')
        W, err = self.sb.staged()
        self.assertIsNone(err)
        self.assertEqual(W['a'][0, 0], 100.0)
        self.assertEqual(W['b'].tolist(), [1, 1, 1])
        self.sb.commit(W)
        r = self.sb.exec('print(float(W0["a"][0, 0]))')
        self.assertEqual(r['out'].strip(), '100.0')

    def test_blocked_and_errors(self):
        self.assertIn('blocked', self.sb.exec('import os')['error'])
        r = self.sb.exec('1/0')
        self.assertFalse(r['ok'])
        self.assertIn('ZeroDivisionError', r['error'])
        r = self.sb.exec('W0["a"][0, 0] = 5')            # committed weights are read-only
        self.assertFalse(r['ok'])
        self.sb.exec('W["a"] = np.zeros((3, 3))')
        W, err = self.sb.staged()
        self.assertIsNone(W)
        self.assertIn('shape', err)

    def test_no_secrets_in_env(self):
        os.environ['OPENAI_API_KEY'] = 'sk-test-should-not-leak-0123456789'
        sb = Sandbox(tempfile.mkdtemp(), timeout_s=10).start(self.W, self.W)
        try:
            r = sb.exec('import collections\nprint(1)')
            self.assertTrue(r['ok'])
            self.assertTrue(check_code('import os; os.environ'))
        finally:
            sb.close()
            os.environ.pop('OPENAI_API_KEY')

    def test_timeout_restarts(self):
        sb = Sandbox(tempfile.mkdtemp(), timeout_s=2).start(self.W, self.W)
        try:
            r = sb.exec('while True:\n    pass')
            self.assertFalse(r['ok'])
            self.assertIn('timeout', r['error'])
            self.assertTrue(sb.exec('print(1)')['ok'])
        finally:
            sb.close()


class TestGate(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.W = {'a': rng.normal(size=(50, 50)).astype(np.float32), 'b': rng.normal(size=50).astype(np.float32)}

    def test_caps(self):
        W2 = {k: v.copy() for k, v in self.W.items()}
        W2['b'][:3] += 0.01
        st = edit_stats(W2, self.W)
        self.assertEqual(st['changed'], 3)
        self.assertEqual(cap_problems(st, {}), [])
        W3 = {k: v * 2 for k, v in self.W.items()}
        self.assertTrue(cap_problems(edit_stats(W3, self.W), {}))
        W4 = {k: v.copy() for k, v in self.W.items()}
        W4['b'][0] = np.nan
        self.assertIn('non-finite values', cap_problems(edit_stats(W4, self.W), {}))

    def test_decide(self):
        rng = np.random.default_rng(1)
        old = Score(1.0, units=rng.normal(1.0, 0.1, 500), guards={'g': 1.0})
        better = Score(0.9, units=old.units - 0.1, guards={'g': 1.0})
        worse = Score(1.1, units=old.units + 0.1, guards={'g': 1.0})
        guard_bad = Score(0.9, units=old.units - 0.1, guards={'g': 1.5})
        st = {'keys_ok': True, 'shape_ok': True, 'finite': True, 'changed': 1, 'frac_changed': 0.0, 'rel_l2': 0.0}
        self.assertTrue(decide(old, better, st)['accept'])
        self.assertFalse(decide(old, worse, st)['accept'])
        self.assertFalse(decide(old, guard_bad, st)['accept'])

    def test_f1_bootstrap(self):
        rng = np.random.default_rng(2)
        y = (rng.random(4000) < 0.5).astype(np.int8)
        po = np.where(rng.random(4000) < 0.8, y, 1 - y).astype(np.int8)
        pn = np.where(rng.random(4000) < 0.9, y, 1 - y).astype(np.int8)
        d = paired_delta(Score(0.2, y=y, pred=po), Score(0.1, y=y, pred=pn), n_boot=300)
        self.assertLess(d['upper'], 0)


class TestCost(unittest.TestCase):
    def test_pricing(self):
        p = Pricing.load()
        self.assertAlmostEqual(p.usd(100_000, 0, 0), 1.0)
        self.assertAlmostEqual(p.usd(100_000, 100_000, 0), 0.1)
        self.assertAlmostEqual(p.usd(0, 0, 100_000), 5.0)
        self.assertAlmostEqual(p.usd(300_000, 0, 0), 300_000 * 20 / 1e6)     # long-context reprice

    def test_tracker_and_ledger(self):
        from langchain_core.messages import AIMessage
        path = os.path.join(tempfile.mkdtemp(), 'ledger.jsonl')
        led = Ledger(path)
        ct = CostTracker('r1', Pricing.load(), led)
        msg = AIMessage('x', usage_metadata={'input_tokens': 1000, 'output_tokens': 200, 'total_tokens': 1200,
                                             'input_token_details': {'cache_read': 600},
                                             'output_token_details': {'reasoning': 150}})
        self.assertEqual(usage_of(msg)['reasoning'], 150)
        ct.add_call(msg, 1.0)
        ct.add_call(msg, 1.0)
        self.assertEqual(ct.llm['total'], 2400)
        self.assertAlmostEqual(ct.llm['usd'], 2 * (400 * 10 + 600 * 1 + 200 * 50) / 1e6)
        self.assertEqual(led.total()[0], 2400)

    def test_budget(self):
        path = os.path.join(tempfile.mkdtemp(), 'ledger.jsonl')
        b = Budget(2, 1, token_cap=1000, ledger=Ledger(path), global_cap=10_000, stop_after_rejects=2)
        self.assertTrue(b.charge_probe() and b.charge_probe())
        self.assertFalse(b.charge_probe())
        b.charge_call(600)
        self.assertEqual(b.llm_stop_reason(), 'token_cap')          # 600 + next ~600 > 1000
        b2 = Budget(5, 5, stop_after_rejects=2)
        b2.verdict(False)
        b2.verdict(False)
        self.assertEqual(b2.stop_reason(), 'rejects_in_a_row')


if __name__ == '__main__':
    unittest.main()
