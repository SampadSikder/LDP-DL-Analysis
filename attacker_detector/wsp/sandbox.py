"""Harness side of the sandbox: a long-lived kernel process per run (see kernel.py).

The kernel runs with `python -I`, an environment that holds no API key or project paths, a temporary
working directory, memory / file-size limits and a per-call wall-clock timeout. A timed-out kernel is
killed and restarted from the committed weights (staged edits are lost, and the agent is told).
"""

import json
import os
import queue
import shlex
import subprocess
import sys
import threading

import numpy as np

from .kernel import check_code  # noqa: F401  (re-exported for the harness and tests)

KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernel.py')


def save_npz(path, W):
    np.savez(path, **{k: np.asarray(v, dtype=np.float32) for k, v in W.items()})


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], dtype=np.float32) for k in z.files}


def _limits(mem_gb):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 2 ** 20, 512 * 2 ** 20))
            if sys.platform.startswith('linux'):
                lim = int(mem_gb * 2 ** 30)
                resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except (ImportError, ValueError, OSError):
            pass
        os.setsid()
    return apply


class Sandbox:
    def __init__(self, workdir, timeout_s=60, mem_gb=8, prefix=None, threads=2):
        self.dir = os.path.abspath(workdir)
        os.makedirs(self.dir, exist_ok=True)
        self.timeout = timeout_s
        self.mem_gb = mem_gb
        self.prefix = shlex.split(prefix if prefix is not None else os.environ.get('WSP_SANDBOX_PREFIX', ''))
        self.threads = str(threads)
        self.proc = None
        self.q = None
        self.committed_path = os.path.join(self.dir, 'committed.npz')
        self.start_path = os.path.join(self.dir, 'start.npz')
        self.staged_path = os.path.join(self.dir, 'staged.npz')
        self.restarts = 0

    # ---------------------------------------------------------------- process
    def _env(self):
        env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': self.dir, 'LANG': 'C.UTF-8',
               'PYTHONDONTWRITEBYTECODE': '1'}
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[v] = self.threads
        return env

    def _spawn(self):
        cmd = self.prefix + [sys.executable, '-I', KERNEL, self.dir]
        self._err = open(os.path.join(self.dir, 'kernel.stderr'), 'a')
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                     cwd=self.dir, env=self._env(), text=True, bufsize=1,
                                     preexec_fn=_limits(self.mem_gb) if os.name == 'posix' else None)
        self.q = queue.Queue()
        out, q = self.proc.stdout, self.q

        def reader():
            for line in out:
                q.put(line)
            q.put(None)
        threading.Thread(target=reader, daemon=True).start()

    def _call(self, msg, timeout=None):
        if self.proc is None or self.proc.poll() is not None:
            return {'ok': False, 'error': 'sandbox not running'}
        try:
            self.proc.stdin.write(json.dumps(msg) + '\n')
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            return {'ok': False, 'error': f'sandbox pipe error: {e}'}
        try:
            line = self.q.get(timeout=timeout or self.timeout)
        except queue.Empty:
            self._restart()
            return {'ok': False, 'error': f'timeout after {timeout or self.timeout}s; sandbox restarted '
                                          'from committed weights, staged edits lost'}
        if line is None:
            self._restart()
            return {'ok': False, 'error': 'sandbox crashed (memory limit?); restarted from committed weights, '
                                          'staged edits lost'}
        return json.loads(line)

    def _kill(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 9)
            except (ProcessLookupError, PermissionError, OSError):
                self.proc.kill()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except (OSError, AttributeError):
                    pass
        if getattr(self, '_err', None):
            self._err.close()
            self._err = None
        self.proc = None

    def _restart(self):
        self._kill()
        self.restarts += 1
        self._spawn()
        self._call({'op': 'init', 'weights': self.committed_path, 'start': self.start_path}, timeout=120)

    # ---------------------------------------------------------------- API
    def start(self, committed: dict, start: dict, staged_path=None):
        """staged_path: optional .npz of staged weights to resume from (used by the Codex CLI)."""
        save_npz(self.committed_path, committed)
        save_npz(self.start_path, start)
        self._spawn()
        msg = {'op': 'init', 'weights': self.committed_path, 'start': self.start_path}
        if staged_path and os.path.exists(staged_path):
            msg['staged'] = os.path.abspath(staged_path)
        r = self._call(msg, timeout=120)
        if not r.get('ok'):
            raise RuntimeError(f"sandbox init failed: {r}")
        self.init_note = r.get('staged')
        return self

    def exec(self, code: str) -> dict:
        return self._call({'op': 'exec', 'code': code})

    def staged(self):
        """(weights, error): the staged W from the kernel, validated for keys and shapes."""
        if os.path.exists(self.staged_path):
            os.remove(self.staged_path)
        r = self._call({'op': 'dump', 'path': self.staged_path})
        if not r.get('ok'):
            return None, r.get('error', 'dump failed')
        return load_npz(self.staged_path), None

    def commit(self, weights: dict):
        """New committed weights; staged weights reset to them."""
        save_npz(self.committed_path, weights)
        r = self._call({'op': 'commit', 'weights': self.committed_path}, timeout=120)
        if not r.get('ok'):
            self._restart()

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        self._kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
