"""Adapters on tiny synthetic data: round trips, leak guard, gate behaviour, baselines, checkpoint format."""
import tempfile
import unittest

import numpy as np
import torch

from tests.wsp.helpers import fitted, freq_arrays, tiny_rounds

from attacker_detector.wsp.adapters import FreqFTAdapter, FTRemovalAdapter, SetRecoveryAdapter, build_freq_model
from attacker_detector.wsp.gate import decide, edit_stats


def make(kind):
    train, val = tiny_rounds('PM', 12, 100), tiny_rounds('PM', 6, 300)
    m = fitted(kind, train, val)
    cls = FTRemovalAdapter if kind == 'mv_ft' else SetRecoveryAdapter
    return cls(m, train, val, None, probe_rounds=6, seed=0)


class AdapterChecks:
    kind = None

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.a = make(cls.kind)

    def test_roundtrip_and_noop(self):
        a = self.a
        W = a.weights()
        s1 = a.evaluate('val')
        a.load_weights(W)
        s2 = a.evaluate('val')
        np.testing.assert_allclose(s1.units, s2.units, rtol=0, atol=1e-7)
        self.assertEqual(s1.guards.keys(), s2.guards.keys())

    def test_leak_guard(self):
        with self.assertRaises(ValueError):
            self.a.evaluate('test')

    def test_using_restores(self):
        a = self.a
        W = a.weights()
        bad = {k: v + np.float32(5.0) * (np.abs(v) > 0) for k, v in W.items()}
        with a.using(bad):
            pass
        for k, v in a.weights().items():
            np.testing.assert_array_equal(v, W[k])

    def test_worse_edit_rejected(self):
        a = self.a
        W = a.weights()
        old = a.evaluate('val')
        rng = np.random.default_rng(0)
        bad = {k: v + rng.normal(0, 3 * (v.std() + 1e-3), v.shape).astype(np.float32) for k, v in W.items()}
        with a.using(bad):
            new = a.evaluate('val')
        v = decide(old, new, {'keys_ok': True, 'shape_ok': True, 'finite': True, 'changed': 1, 'frac_changed': 0,
                              'rel_l2': 0})
        self.assertFalse(v['accept'])

    def test_probes_and_card(self):
        a = self.a
        for kind in ('metrics', 'errors', 'activations'):
            r = a.probe(kind)
            self.assertTrue(r)
        card = a.describe()
        self.assertEqual(card['n_params'], sum(p.numel() for p in a.model.parameters()))
        self.assertGreater(a.eval_flops('probe'), 0)

    def test_finetune_and_save(self):
        from attacker_detector.mean_variance.methods.checkpoint import load_checkpoint
        a = self.a
        W = a.weights()
        a.finetune(2, 0.1)
        self.assertGreater(edit_stats(a.weights(), W)['changed'], 0)
        self.assertGreater(a.counters['train_flops'], 0)
        s = a.evaluate('val')
        a.on_accept(s)
        d = tempfile.mkdtemp()
        a.save(d)
        m2 = load_checkpoint(d, 'cpu')
        r = a.data['val']['rounds'][0] if 'rounds' in a.data['val'] else None
        obs = r.observation if r is not None else None
        if obs is not None:
            o1, o2 = a.method.predict(obs), m2.predict(obs)
            self.assertAlmostEqual(o1.mean, o2.mean, places=6)
        cost = a.stage1_cost()
        self.assertGreater(cost['train_flops'], 0)
        a.load_weights(W)


class TestSetDeepSets(AdapterChecks, unittest.TestCase):
    kind = 'mv_ds'


class TestSetTransformer(AdapterChecks, unittest.TestCase):
    kind = 'mv_st'


class TestFTRemoval(AdapterChecks, unittest.TestCase):
    kind = 'mv_ft'


class TestFreq(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        arr = freq_arrays()
        spec = {'d_token': 16, 'n_heads': 4, 'n_layers': 1, 'ffn_d_multiplier': 1.33, 'attention_dropout': 0.0,
                'residual_dropout': 0.0, 'lr': 1e-3, 'batch_size': 64, 'seed': 0, 'protocol': 'OUE',
                'training_method': 'cross', 'train_dataset': ['zipf', 'emoji'], 'test_dataset': 'fire',
                'stage1_dir': tempfile.mkdtemp()}
        model = build_freq_model(spec, 4)
        self.a = FreqFTAdapter(model, spec, arr, 'cpu', probe_rows=500)

    def test_gate_f1(self):
        a = self.a
        old = a.evaluate('val')
        a.finetune(60, 1.0)
        new = a.evaluate('val')
        v = decide(old, new, edit_stats(a.weights(), a.weights()) | {'changed': 1}, {'max_changed_frac': 1.0,
                                                                                    'max_rel_l2': 1e9})
        self.assertLess(new.primary, old.primary)          # learnable synthetic signal
        self.assertTrue(v['accept'], v['reasons'])
        for kind in ('metrics', 'errors', 'activations'):
            self.assertTrue(a.probe(kind))
        with self.assertRaises(ValueError):
            a.evaluate('test')


if __name__ == '__main__':
    unittest.main()
