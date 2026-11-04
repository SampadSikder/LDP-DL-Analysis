"""Sandbox kernel: runs LLM-written numpy code on copies of the model weights.

sandbox.py starts it as `python -I kernel.py <workdir>` with a scrubbed environment (no API key) and
talks to it with one JSON line per command on stdin / one JSON line per answer on stdout.

What user code gets:
  W    dict name -> float32 array, the staged weights (edit these)
  W0   committed weights (read-only); WS  Stage-1 weights (read-only)
  np, math, names(), diff(), revert(name=None)
Only numpy / math and a few pure-Python stdlib modules import. There is no torch, so no autograd and no
optimizer; no file, process or network access. This guards against accidents and shortcuts; it is not a
security boundary against hostile code. For that, set WSP_SANDBOX_PREFIX (e.g. bwrap or firejail).
"""

import ast
import builtins
import contextlib
import io
import json
import math
import sys
import traceback

import numpy as np

ALLOWED_MODULES = {'numpy', 'math', 'itertools', 'functools', 'statistics', 'collections', 'heapq'}
BLOCKED_NAMES = {'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'locals', 'vars', 'getattr',
                 'setattr', 'delattr', 'input', 'breakpoint', 'help', 'exit', 'quit', 'memoryview'}
BLOCKED_ATTRS = {'load', 'loads', 'save', 'savez', 'savez_compressed', 'savetxt', 'loadtxt', 'genfromtxt',
                 'fromfile', 'tofile', 'memmap', 'fromregex', 'DataSource', 'ctypeslib', 'ctypes', 'f2py',
                 'distutils', 'testing', 'lib', 'os', 'sys', 'system', 'popen', 'dump', 'dumps'}
OUT_CHARS = 6000


def check_code(code: str) -> list:
    """Reasons the code may not run (empty list = allowed)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f'syntax error: {e}']
    def bad_part(p):
        return p.startswith('_') or p in BLOCKED_ATTRS

    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split('.')
                if parts[0] not in ALLOWED_MODULES or any(bad_part(p) for p in parts):
                    bad.append(f'import {a.name} not allowed')
        elif isinstance(node, ast.ImportFrom):
            parts = (node.module or '').split('.')
            if node.level or parts[0] not in ALLOWED_MODULES or any(bad_part(p) for p in parts):
                bad.append(f'from {node.module} import not allowed')
            bad += [f'import of {a.name!r} not allowed' for a in node.names if a.name == '*' or bad_part(a.name)]
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith('__'):
                bad.append(f'name {node.id!r} not allowed')
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith('_') or node.attr in BLOCKED_ATTRS:
                bad.append(f'attribute .{node.attr} not allowed')
    return sorted(set(bad))


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split('.')[0] not in ALLOWED_MODULES:
        raise ImportError(f'import of {name!r} is not allowed in the sandbox')
    return builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {k: getattr(builtins, k) for k in dir(builtins)
                 if not k.startswith('_') and k not in BLOCKED_NAMES}
SAFE_BUILTINS.update(__import__=_safe_import, __build_class__=builtins.__build_class__, __name__='sandbox')

STATE = {'W': {}, 'W0': {}, 'WS': {}}


def _readonly(d):
    out = {}
    for k, v in d.items():
        a = np.array(v, dtype=np.float32, copy=True)
        a.setflags(write=False)
        out[k] = a
    return out


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], dtype=np.float32) for k in z.files}


def names():
    """Tensor names with shapes."""
    return [(k, tuple(v.shape)) for k, v in STATE['W0'].items()]


def diff(show=True):
    """Staged W vs committed W0: changed entries and relative L2 change per tensor."""
    W, W0 = NS['W'], STATE['W0']
    rows, tot_d, tot_n, changed, total = [], 0.0, 0.0, 0, 0
    for k, a0 in W0.items():
        a = np.asarray(W.get(k, a0), dtype=np.float32)
        total += a0.size
        if a.shape != a0.shape:
            rows.append((k, 'SHAPE CHANGED', a.shape, a0.shape))
            continue
        d = a - a0
        n = int(np.count_nonzero(d))
        tot_d += float(np.sum(d.astype(np.float64) ** 2))
        tot_n += float(np.sum(a0.astype(np.float64) ** 2))
        if n:
            changed += n
            rows.append((k, n, round(float(np.linalg.norm(d) / (np.linalg.norm(a0) + 1e-12)), 5)))
    out = {'changed': changed, 'total': total, 'frac_changed': changed / max(total, 1),
           'rel_l2': math.sqrt(tot_d / max(tot_n, 1e-24)), 'tensors': rows}
    if show:
        print(json.dumps(out, default=str))
    return out


def revert(name=None):
    """Reset staged weights (one tensor or all) to the committed ones."""
    keys = [name] if name else list(STATE['W0'])
    for k in keys:
        NS['W'][k] = np.array(STATE['W0'][k], dtype=np.float32, copy=True)


NS = {}


def _fresh_namespace():
    NS.clear()
    NS.update(__builtins__=SAFE_BUILTINS, np=np, math=math, names=names, diff=diff, revert=revert,
              W={k: np.array(v, dtype=np.float32, copy=True) for k, v in STATE['W0'].items()},
              W0=STATE['W0'], WS=STATE['WS'])


def _exec(code):
    bad = check_code(code)
    if bad:
        return {'ok': False, 'out': '', 'error': 'blocked: ' + '; '.join(bad)}
    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(code, '<sandbox>', 'exec'), NS)
    except BaseException:                       # noqa: BLE001 - report every failure to the agent
        tb = traceback.format_exc().strip().splitlines()
        err = '\n'.join(tb[-6:])
    out = buf.getvalue()
    if len(out) > OUT_CHARS:
        out = out[:OUT_CHARS] + f'\n... [{len(out) - OUT_CHARS} more chars cut]'
    return {'ok': err is None, 'out': out, 'error': err}


def _dump(path):
    W = NS.get('W')
    if not isinstance(W, dict):
        return {'ok': False, 'error': 'W is no longer a dict of arrays'}
    missing = sorted(set(STATE['W0']) - set(W))
    extra = sorted(set(W) - set(STATE['W0']))
    if missing or extra:
        return {'ok': False, 'error': f'W keys changed (missing {missing[:5]}, extra {extra[:5]})'}
    arrays = {}
    for k, a0 in STATE['W0'].items():
        try:
            a = np.asarray(W[k], dtype=np.float32)
        except Exception as e:                  # noqa: BLE001
            return {'ok': False, 'error': f'W[{k!r}] is not numeric: {e}'}
        if a.shape != a0.shape:
            return {'ok': False, 'error': f'W[{k!r}] shape {a.shape} != {a0.shape}'}
        arrays[k] = a
    np.savez(path, **arrays)
    return {'ok': True}


def main(workdir):
    proto = sys.stdout
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        op = msg.get('op')
        try:
            if op == 'init':
                STATE['W0'] = _readonly(_load(msg['weights']))
                STATE['WS'] = _readonly(_load(msg['start']))
                _fresh_namespace()
                res = {'ok': True}
                if msg.get('staged'):           # staged edit kept on disk between separate CLI calls
                    S = _load(msg['staged'])
                    same = set(S) == set(STATE['W0']) and all(S[k].shape == a.shape for k, a in STATE['W0'].items())
                    if same:
                        NS['W'] = {k: np.array(S[k], dtype=np.float32, copy=True) for k in STATE['W0']}
                    res['staged'] = 'loaded' if same else 'ignored (keys or shapes differ from committed)'
            elif op == 'commit':                # committed weights changed: reset staged to them
                STATE['W0'] = _readonly(_load(msg['weights']))
                NS['W0'] = STATE['W0']
                NS['W'] = {k: np.array(v, dtype=np.float32, copy=True) for k, v in STATE['W0'].items()}
                res = {'ok': True}
            elif op == 'exec':
                res = _exec(msg['code'])
            elif op == 'dump':
                res = _dump(msg['path'])
            elif op == 'ping':
                res = {'ok': True}
            else:
                res = {'ok': False, 'error': f'unknown op {op!r}'}
        except Exception:                       # noqa: BLE001
            res = {'ok': False, 'error': traceback.format_exc()[-1500:]}
        proto.write(json.dumps(res) + '\n')
        proto.flush()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
