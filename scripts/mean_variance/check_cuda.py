#!/usr/bin/env python
"""Preflight: does the benchmark code actually run on CUDA?

  python scripts/mean_variance/check_cuda.py            # exit 0 = all checks pass
  python scripts/mean_variance/check_cuda.py --config configs/mean_variance/breadth.yaml

Checks, in order (each prints PASS / FAIL):
  1. driver      nvidia-smi lists GPUs; torch is a CUDA build and sees the same number
  2. kernels     a matmul + backward runs on every visible GPU (catches driver / arch mismatch)
  3. picker      methods.removal.device_for('auto') returns cuda
  4. methods     every neural method in the config, built as `evaluate` builds it, sits on cuda
  5. training    MLP, FT-Transformer, Deep Sets and Set Transformer each fit one epoch on tiny
                 synthetic rounds; weights end on cuda and predict() works
  6. shards      one subprocess per GPU with CUDA_VISIBLE_DEVICES=i (as run_server.sh starts
                 evaluate shards) sees exactly that one GPU and picks cuda
No dataset files are needed.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

FAILED = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        msg = fn()
        print(f"PASS  {name:9} {msg or ''} ({time.perf_counter() - t0:.1f}s)", flush=True)
    except Exception as e:                       # report every check, then exit non-zero
        FAILED.append(name)
        hint = ''
        if isinstance(e, ModuleNotFoundError):   # env problem, not a CUDA problem
            pkg = {'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}.get(e.name, e.name)
            hint = f"  -> missing package, not a CUDA problem: {sys.executable} -m pip install {pkg}"
        print(f"FAIL  {name:9} {type(e).__name__}: {e}{hint}", flush=True)


def nvidia_smi_count():
    if not shutil.which('nvidia-smi'):
        return 0
    out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout
    return sum(line.startswith('GPU') for line in out.splitlines())


def check_driver():
    n_smi, n_torch = nvidia_smi_count(), torch.cuda.device_count()
    if torch.version.cuda is None:
        raise RuntimeError(f"torch {torch.__version__} is a CPU-only build (nvidia-smi sees {n_smi} GPUs); "
                           "install a CUDA build of torch")
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} (CUDA {torch.version.cuda}) cannot reach a GPU; "
                           f"nvidia-smi sees {n_smi}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    if 'CUDA_VISIBLE_DEVICES' not in os.environ and n_smi and n_smi != n_torch:
        raise RuntimeError(f"nvidia-smi sees {n_smi} GPUs but torch sees {n_torch}")
    names = ', '.join(f"{i}:{torch.cuda.get_device_name(i)} "
                      f"({torch.cuda.get_device_properties(i).total_memory / 2 ** 30:.0f} GB)"
                      for i in range(n_torch))
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, {n_torch} GPU(s): {names}"


def check_kernels():
    for i in range(torch.cuda.device_count()):
        dev = torch.device(f'cuda:{i}')
        a = torch.randn(512, 512, device=dev, requires_grad=True)
        (a @ a).sum().backward()
        torch.cuda.synchronize(dev)
        if not torch.isfinite(a.grad).all():
            raise RuntimeError(f"non-finite gradient on {dev}")
    return f"matmul + backward ok on {torch.cuda.device_count()} GPU(s)"


def check_picker():
    from attacker_detector.mean_variance.methods.removal import device_for
    d = device_for('auto')
    if d.type != 'cuda':
        raise RuntimeError(f"device_for('auto') -> {d}, expected cuda")
    return f"device_for('auto') -> {d}"


def check_methods(cfg_path):
    from attacker_detector.mean_variance.experiment import load_config
    from attacker_detector.mean_variance.methods import make_method
    cfg = load_config(cfg_path)
    out = []
    for name in cfg['methods']:
        try:
            m = make_method(name, 0, cfg)
        except Exception as e:          # e.g. catboost missing: not a CUDA question
            out.append(f"{name}=skip({type(e).__name__})")
            continue
        dev = getattr(m, 'device', None)
        if dev is None:
            continue                     # non-neural method (standard, kmeans, catboost, ...)
        if torch.device(dev).type != 'cuda':
            raise RuntimeError(f"{name} built on {dev}, expected cuda")
        out.append(f"{name}={dev}")
    if not any('=cuda' in o for o in out):
        raise RuntimeError(f"no neural method in {cfg_path}: {cfg['methods']}")
    return f"{os.path.basename(cfg_path)}: " + ', '.join(out)


def tiny_rounds(protocol, k, seed0):
    """Synthetic rounds (Beta inputs, a few constant-value fakes): enough to run fit() once."""
    from attacker_detector.mean_variance.estimator import perturb_users, split_groups
    from attacker_detector.mean_variance.mechanisms import PM, SR
    from attacker_detector.mean_variance.methods import TrainingRound
    from attacker_detector.mean_variance.transforms import Transform
    tf, mech, out = Transform(0.0, 100.0), (PM(1.0) if protocol == 'PM' else SR(1.0)), []
    for i in range(k):
        rng = np.random.default_rng(seed0 + i)
        n, fake = 600, (0, 60, 180)[i % 3]
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


def check_training():
    from attacker_detector.mean_variance.methods import ClassifierRemoval, SetRecovery
    tr, va = tiny_rounds('PM', 9, 100), tiny_rounds('PM', 3, 200)
    out = []
    for m in (ClassifierRemoval('mlp', epochs=1, reports_per_round=64),
              ClassifierRemoval('ft', epochs=1, reports_per_round=64),
              SetRecovery('deepsets', epochs=1, token_cap=128),
              SetRecovery('settransformer', epochs=1, token_cap=128)):
        t0 = time.perf_counter()
        m.fit(tr, va)
        dev = next(m.model.parameters()).device
        if dev.type != 'cuda':
            raise RuntimeError(f"{m.arch} weights on {dev}, expected cuda")
        p = m.predict(va[0].observation)
        if not (np.isfinite(p.mean_raw) and np.isfinite(p.var_raw)):
            raise RuntimeError(f"{m.arch} predicted non-finite moments")
        out.append(f"{m.arch}@{dev} {time.perf_counter() - t0:.1f}s")
    return ', '.join(out)


CHILD = """
import os, sys, torch
sys.path.insert(0, os.getcwd())
from attacker_detector.mean_variance.methods.removal import device_for
n, d = torch.cuda.device_count(), device_for('auto')
x = torch.randn(256, 256, device=d); (x @ x).sum().item()
print(n, d.type, torch.cuda.get_device_name(0) if n else '-')
"""


def check_shards():
    n = torch.cuda.device_count()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out = []
    for i in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        r = subprocess.run([sys.executable, '-c', CHILD], env=env, cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES={i}: {(r.stderr.strip().splitlines() or ['no output'])[-1]}")
        count, dev, name = r.stdout.split()[0], r.stdout.split()[1], ' '.join(r.stdout.split()[2:])
        if count != '1' or dev != 'cuda':
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES={i}: sees {count} GPU(s), picks {dev}")
        out.append(f"{i}:{name}")
    return f"each shard sees its own GPU -> {', '.join(out)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/mean_variance/breadth.yaml',
                    help="config whose neural methods are checked (default: breadth)")
    args = ap.parse_args()
    check('driver', check_driver)
    if FAILED:                                  # nothing below can pass without a GPU
        print("CUDA NOT DETECTED: neural methods would run on CPU")
        return 1
    check('kernels', check_kernels)
    check('picker', check_picker)
    check('methods', lambda: check_methods(args.config))
    check('training', check_training)
    check('shards', check_shards)
    print("CUDA OK: benchmark code runs on GPU" if not FAILED else f"CUDA CHECK FAILED: {', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
