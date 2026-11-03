"""Per-run limits: probes, gate evaluations, tokens, LLM calls, wall clock, rejects in a row,
plus the global token cap read from the shared ledger."""

import time


class Budget:
    def __init__(self, probes, gate_evals, token_cap=None, max_calls=None, wall_s=None,
                 stop_after_rejects=None, ledger=None, global_cap=None):
        self.probes, self.gate_evals = int(probes), int(gate_evals)
        self.token_cap, self.max_calls, self.wall_s = token_cap, max_calls, wall_s
        self.stop_after_rejects = stop_after_rejects
        self.ledger, self.global_cap = ledger, global_cap
        self.probes_used = self.gates_used = self.tokens_used = self.calls = 0
        self.rejects_in_row = self.accepted = self.rejected = 0
        self.last_call_tokens = 0
        self.elapsed_before = 0.0
        self.t0 = time.monotonic()

    # ------------------------------------------------------------ charges
    def probes_left(self):
        return self.probes - self.probes_used

    def gates_left(self):
        return self.gate_evals - self.gates_used

    def charge_probe(self):
        if self.probes_left() <= 0:
            return False
        self.probes_used += 1
        return True

    def charge_gate(self):
        if self.gates_left() <= 0:
            return False
        self.gates_used += 1
        return True

    def verdict(self, accepted):
        if accepted:
            self.accepted += 1
            self.rejects_in_row = 0
        else:
            self.rejected += 1
            self.rejects_in_row += 1

    def charge_call(self, tokens):
        self.calls += 1
        self.tokens_used += int(tokens)
        self.last_call_tokens = int(tokens)

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.t0

    # ------------------------------------------------------------ stopping
    def llm_stop_reason(self):
        """Why no further LLM call may be made (None = go). Checked before every call."""
        if self.token_cap and self.tokens_used + self.last_call_tokens > self.token_cap:
            return 'token_cap'
        if self.max_calls and self.calls >= self.max_calls:
            return 'max_calls'
        if self.global_cap and self.ledger is not None:
            tok, _ = self.ledger.total()
            if tok + self.last_call_tokens > self.global_cap:
                return 'global_token_cap'
        return self.stop_reason()

    def stop_reason(self):
        if self.wall_s and self.elapsed() > self.wall_s:
            return 'wall_clock'
        if self.stop_after_rejects and self.rejects_in_row >= self.stop_after_rejects:
            return 'rejects_in_a_row'
        if self.gates_left() <= 0:
            return 'gate_budget_spent'
        return None

    def status(self):
        return {'probes_left': self.probes_left(), 'gate_evals_left': self.gates_left(),
                'tokens_used': self.tokens_used, 'token_cap': self.token_cap, 'llm_calls': self.calls,
                'accepted': self.accepted, 'rejected': self.rejected, 'rejects_in_row': self.rejects_in_row,
                'elapsed_min': round(self.elapsed() / 60, 1)}

    def state(self):
        return {k: getattr(self, k) for k in ('probes_used', 'gates_used', 'tokens_used', 'calls',
                                               'rejects_in_row', 'accepted', 'rejected', 'last_call_tokens')} | {
            'elapsed_s': self.elapsed()}

    def restore(self, st):
        for k in ('probes_used', 'gates_used', 'tokens_used', 'calls', 'rejects_in_row', 'accepted', 'rejected',
                  'last_call_tokens'):
            setattr(self, k, int(st.get(k, 0)))
        self.elapsed_before = float(st.get('elapsed_s', 0.0))
        self.t0 = time.monotonic()
