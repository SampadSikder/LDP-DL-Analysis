"""Small synthetic fixtures for the WSP tests (no data files, CPU, seconds)."""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def tiny_rounds(protocol, k, seed0, n=400):
    """Synthetic rounds (Beta inputs, constant-value fakes); same recipe as scripts/mean_variance/check_cuda.py."""
    from attacker_detector.mean_variance.estimator import perturb_users, split_groups
    from attacker_detector.mean_variance.mechanisms import PM, SR
    from attacker_detector.mean_variance.methods import TrainingRound
    from attacker_detector.mean_variance.transforms import Transform
    tf, mech, out = Transform(0.0, 100.0), (PM(1.0) if protocol == 'PM' else SR(1.0)), []
    for i in range(k):
        rng = np.random.default_rng(seed0 + i)
        fake = (0, n // 10, 3 * n // 10)[i % 3]
        x = 100 * rng.beta(2, 5, n)
        g = split_groups(n + fake, rng)
        y = np.r_[perturb_users(x, g[:n], mech, tf, rng), np.full(fake, 0.8 if protocol == 'PM' else 1.0)]
        obs = {'reports': y, 'groups': g, 'n_g1': int(np.sum(g == 1)), 'n_g2': int(np.sum(g == 2)),
               'protocol': protocol, 'epsilon': 1.0, 'domain': (0.0, 100.0), 'square_mode': 'raw'}
        t = tf.to_g1(x)
        out.append(TrainingRound(f'r{seed0 + i}', obs, np.r_[np.zeros(n, bool), np.ones(fake, bool)],
                                 {'genuine_mean': float(t.mean()), 'genuine_variance': float(t.var())},
                                 {'n_fake': fake, 'realized_beta': fake / (n + fake)}))
    return out


def fitted(kind, train, val):
    """A Stage-1-like method trained for one epoch, saved and reloaded through the existing checkpoint code."""
    import torch
    from attacker_detector.mean_variance.methods import ClassifierRemoval, SetRecovery
    from attacker_detector.mean_variance.methods.checkpoint import load_checkpoint, save_checkpoint
    torch.manual_seed(0)
    if kind == 'mv_ft':
        m = ClassifierRemoval('ft', epochs=1, reports_per_round=64, device='cpu')
    else:
        m = SetRecovery('deepsets' if kind == 'mv_ds' else 'settransformer', epochs=1, device='cpu', token_cap=256)
    m.fit(train, val)
    d = tempfile.mkdtemp()
    save_checkpoint(m, d)
    return load_checkpoint(d, 'cpu')


def freq_arrays(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.float32)
    X = rng.normal(0, 1, (n, 4)).astype(np.float32)
    X[:, 0] += 1.5 * y
    conf = np.empty((n, 10), dtype=object)
    conf[:] = '0'
    conf[:, 4] = rng.choice(['0.5', '1.0', '2.0'], n)          # epsilon
    conf[:, 5] = rng.choice(['zipf', 'emoji'], n)              # dataset
    conf[:, 1] = '0.15'
    conf[:, 0] = '10'
    conf[:, 3] = '4'
    conf = conf.astype(str)
    h = n * 2 // 3
    return {'X_train': X[:h], 'y_train': y[:h], 'X_val': X[h:], 'y_val': y[h:], 'config_train': conf[:h],
            'config_val': conf[h:], 'feature_names': ['block_projection', 'block_overlap', 'one_deviation',
                                                      'excess_share']}
