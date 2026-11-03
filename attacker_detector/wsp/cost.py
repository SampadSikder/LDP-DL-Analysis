"""Token, dollar and compute accounting.

Token counts come from each API response's usage metadata (never estimated). Dollars use
configs/wsp/pricing.yaml. A shared ledger (JSON lines, file-locked) sums tokens across every run and
process, so the weekly cap holds for the whole sweep.
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import yaml

try:
    import fcntl
except ImportError:                     # pragma: no cover - Windows
    fcntl = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_PRICING = os.path.join(ROOT, 'configs', 'wsp', 'pricing.yaml')
DEFAULT_LEDGER = os.path.join(os.environ.get('WSP_HOME') or os.path.join(ROOT, 'results', 'wsp'), 'ledger.jsonl')


@dataclass
class Pricing:
    input: float
    cached_input: float
    output: float
    long_threshold: int
    long_input: float
    long_cached_input: float
    long_output: float
    model: str = ''

    @classmethod
    def load(cls, path=None):
        with open(path or DEFAULT_PRICING) as f:
            p = yaml.safe_load(f)
        u, lc = p['usd_per_million'], p.get('long_context', {})
        return cls(u['input'], u['cached_input'], u['output'], int(lc.get('threshold_input_tokens', 10 ** 12)),
                   lc.get('input', u['input']), lc.get('cached_input', u['cached_input']),
                   lc.get('output', u['output']), p.get('model', ''))

    def usd(self, inp, cached, out):
        """Dollars for one request. Output tokens include reasoning tokens (billed as output)."""
        if inp > self.long_threshold:
            ri, rc, ro = self.long_input, self.long_cached_input, self.long_output
        else:
            ri, rc, ro = self.input, self.cached_input, self.output
        return ((inp - cached) * ri + cached * rc + out * ro) / 1e6


def usage_of(msg) -> dict:
    """{input, cached, output, reasoning, total} from a LangChain AIMessage."""
    um = getattr(msg, 'usage_metadata', None) or {}
    if not um:
        tu = (getattr(msg, 'response_metadata', None) or {}).get('token_usage') or {}
        um = {'input_tokens': tu.get('prompt_tokens', 0), 'output_tokens': tu.get('completion_tokens', 0),
              'input_token_details': {'cache_read': (tu.get('prompt_tokens_details') or {}).get('cached_tokens', 0)},
              'output_token_details': {'reasoning': (tu.get('completion_tokens_details') or {})
                                       .get('reasoning_tokens', 0)}}
    inp = int(um.get('input_tokens') or 0)
    out = int(um.get('output_tokens') or 0)
    cached = int((um.get('input_token_details') or {}).get('cache_read') or 0)
    reasoning = int((um.get('output_token_details') or {}).get('reasoning') or 0)
    return {'input': inp, 'cached': cached, 'output': out, 'reasoning': reasoning,
            'total': int(um.get('total_tokens') or inp + out)}


@contextmanager
def _locked(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.lock', 'a') as lf:
        if fcntl:
            fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_UN)


class Ledger:
    """Every LLM call of every run, one JSON line each. total() is what the global cap checks."""

    def __init__(self, path=None, since=None):
        self.path = path or os.environ.get('WSP_LEDGER') or DEFAULT_LEDGER
        self.since = since if since is not None else os.environ.get('WSP_LEDGER_SINCE')   # ISO date/time

    def add(self, run_id, usage, usd):
        rec = {'t': time.time(), 'iso': time.strftime('%Y-%m-%dT%H:%M:%S'), 'run': run_id, 'usd': usd, **usage}
        with _locked(self.path):
            with open(self.path, 'a') as f:
                f.write(json.dumps(rec) + '\n')

    def total(self):
        """(tokens, usd) since WSP_LEDGER_SINCE (all entries if unset)."""
        if not os.path.exists(self.path):
            return 0, 0.0
        tok, usd = 0, 0.0
        with _locked(self.path):
            with open(self.path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if self.since and r.get('iso', '') < self.since:
                        continue
                    tok += int(r.get('total', 0))
                    usd += float(r.get('usd', 0.0))
        return tok, usd


class CostTracker:
    """Per-run totals: LLM usage, harness time, forward passes and their FLOPs."""

    KEYS = ('input', 'cached', 'output', 'reasoning', 'total')

    def __init__(self, run_id, pricing: Pricing = None, ledger: Ledger = None, calls_path=None):
        self.run_id = run_id
        self.pricing = pricing
        self.ledger = ledger
        self.calls_path = calls_path
        self.llm = {k: 0 for k in self.KEYS}
        self.llm.update(calls=0, usd=0.0, wait_s=0.0, max_call_tokens=0)
        self.harness = {'probe_evals': 0, 'gate_evals': 0,
                        'forward_flops': 0, 'train_flops': 0, 'train_steps': 0, 'compute_s': 0.0,
                        'sandbox_s': 0.0, 'sandbox_calls': 0}

    def add_call(self, msg, wait_s):
        u = usage_of(msg)
        usd = self.pricing.usd(u['input'], u['cached'], u['output']) if self.pricing else 0.0
        for k in self.KEYS:
            self.llm[k] += u[k]
        self.llm['calls'] += 1
        self.llm['usd'] += usd
        self.llm['wait_s'] += wait_s
        self.llm['max_call_tokens'] = max(self.llm['max_call_tokens'], u['total'])
        if self.ledger:
            self.ledger.add(self.run_id, u, usd)
        if self.calls_path:
            with open(self.calls_path, 'a') as f:
                f.write(json.dumps({'t': time.time(), 'wait_s': round(wait_s, 3), 'usd': usd, **u}) + '\n')
        return u, usd

    def state(self):
        return {'llm': dict(self.llm), 'harness': dict(self.harness)}

    def restore(self, st):
        self.llm.update(st.get('llm', {}))
        self.harness.update(st.get('harness', {}))


def forward_flops(fn):
    """FLOPs of one call of fn() (torch FlopCounterMode; matmul-type ops only)."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        fn()
    return int(fc.get_total_flops())
