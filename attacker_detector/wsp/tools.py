"""Agent tools. Each returns a string (JSON where useful), capped in length.

Free: model_card, tensor_stats, read_tensor, python, history.
Costs 1 probe: probe.  Costs 1 gate evaluation: submit_edit.  finish ends the run.
"""

import re
import time

import numpy as np

from .gate import cap_problems, decide, edit_stats
from .log import dumps

PROBE_KINDS = ('metrics', 'errors', 'activations')


def _cap(text, n):
    return text if len(text) <= n else text[:n] + f'\n... [{len(text) - n} chars cut; ask for less]'


def _tensor_summary(a):
    a64 = a.astype(np.float64)
    r = {'shape': list(a.shape), 'mean': round(float(a64.mean()), 6), 'std': round(float(a64.std()), 6),
         'min': round(float(a64.min()), 6), 'max': round(float(a64.max()), 6),
         'l2': round(float(np.linalg.norm(a64)), 5), 'near_zero_frac': round(float(np.mean(np.abs(a) < 1e-8)), 4)}
    if a.ndim == 2 and min(a.shape) > 1:
        sv = np.linalg.svd(a64, compute_uv=False)
        rn = np.linalg.norm(a64, axis=1)
        r.update(top_singular_values=[round(float(x), 4) for x in sv[:5]],
                 stable_rank=round(float((sv ** 2).sum() / max(sv[0] ** 2, 1e-24)), 3),
                 row_norm_min_med_max=[round(float(x), 4) for x in (rn.min(), np.median(rn), rn.max())],
                 zero_rows=int((rn < 1e-8).sum()))
    return r


_IDX = re.compile(r'^[\d\s:,\-]*$')


def _parse_index(index):
    """'0:4, :' -> (slice(0, 4), slice(None)); only integers, ':' and ','."""
    if not index or not index.strip():
        return ()
    if not _IDX.match(index):
        raise ValueError("index may contain only integers, ':' and ',' (e.g. '0:4, :')")
    out = []
    for part in index.split(','):
        part = part.strip()
        if ':' in part:
            bits = [int(b) if b.strip() else None for b in part.split(':')]
            out.append(slice(*bits))
        elif part:
            out.append(int(part))
        else:
            out.append(slice(None))
    return tuple(out)


def _schemas():
    """Argument schemas of the LangChain tools (pydantic is only needed for the API agent)."""
    from pydantic import BaseModel, Field

    class ModelCardArgs(BaseModel):
        pass


    class TensorStatsArgs(BaseModel):
        names: list[str] = Field(default_factory=list, description='tensor names; empty = all tensors')
        staged: bool = Field(False, description='true = your staged weights in the sandbox, false = committed')


    class ReadTensorArgs(BaseModel):
        name: str = Field(description='tensor name')
        index: str = Field('', description="numpy-style index, e.g. '0:4, :' or '3' (integers, ':' and ',' only)")
        staged: bool = Field(False, description='true = staged weights, false = committed')


    class PythonArgs(BaseModel):
        code: str = Field(description='Python code (numpy only). Edit W[name] in place to stage an edit.')


    class ProbeArgs(BaseModel):
        kind: str = Field('metrics', description="'metrics' (per-slice table), 'errors' (worst cases), "
                                                 "'activations' (per-layer unit statistics)")
        weights: str = Field('staged', description="'staged' (your edit, compared with committed) or 'committed'")


    class SubmitArgs(BaseModel):
        rationale: str = Field(description='what the staged edit changes and why it should help (1-3 sentences)')


    class FinishArgs(BaseModel):
        summary: str = Field(description='what you found and changed, and why you stop')

    return {'model_card': ModelCardArgs, 'tensor_stats': TensorStatsArgs, 'read_tensor': ReadTensorArgs,
            'python': PythonArgs, 'probe': ProbeArgs, 'submit_edit': SubmitArgs, 'history': ModelCardArgs,
            'finish': FinishArgs}


TOOL_DOCS = {
    'model_card': 'Task, architecture, inputs, metric, tensor names/shapes, committed validation metrics, budget '
                  'left and gate rules. Free.',
    'tensor_stats': 'Summary statistics per tensor (2-D: top singular values, stable rank, row norms). Free.',
    'read_tensor': 'Raw values of a slice of one tensor (max 2048 values). Free.',
    'python': 'Run numpy code in a persistent sandbox. W = staged weights (edit in place), W0 = committed, WS = '
              'Stage-1 start (read-only). Helpers: names(), diff(), revert(name=None). No data, no torch, no files, '
              'no network. Free.',
    'probe': 'Forward pass on a fixed subset of TRAINING data. With staged weights the result includes the change '
             'against committed weights. Costs 1 probe.',
    'submit_edit': 'Gate the staged weights on VALIDATION data. Kept only if within the edit caps, reliably better '
                   '(paired bootstrap) and no guard metric worse. Costs 1 gate evaluation. Rejected edits stay '
                   'staged; call revert() in python to reset.',
    'history': 'Your submitted edits with verdicts, and probes so far. Free.',
    'finish': 'End the run when no further gain is likely or the budget is spent.',
}


def make_tools(ctx):
    """(LangChain tools, {name: callable}) bound to one run context."""
    from langchain_core.tools import StructuredTool
    fns = tool_functions(ctx)
    schemas = _schemas()
    tools = [StructuredTool.from_function(func=fns[n], name=n, description=d, args_schema=schemas[n])
             for n, d in TOOL_DOCS.items()]
    return tools, fns


def tool_functions(ctx):
    """{name: callable} bound to one run context (no LangChain needed; the Codex CLI uses these directly)."""
    cap = ctx.cfg['agent']['tool_output_chars']

    def model_card():
        card = dict(ctx.card)
        card['committed_validation'] = ctx.cur_val.report
        card['budget'] = ctx.budget.status()
        card['gate'] = ctx.gate_rules
        return _cap(dumps(card), cap * 2)

    def tensor_stats(names=None, staged=False):
        W, err = ctx.weights(staged)
        if err:
            return f'error: {err}'
        names = names or list(W)
        bad = [n for n in names if n not in W]
        if bad:
            return f'error: unknown tensors {bad[:5]}; call model_card for names'
        return _cap(dumps({n: _tensor_summary(W[n]) for n in names}), cap)

    def read_tensor(name, index='', staged=False):
        W, err = ctx.weights(staged)
        if err:
            return f'error: {err}'
        if name not in W:
            return f'error: unknown tensor {name!r}'
        try:
            a = W[name][_parse_index(index)]
        except (ValueError, IndexError, TypeError) as e:
            return f'error: {e}'
        a = np.asarray(a)
        if a.size > 2048:
            return f'error: selection has {a.size} values (max 2048); narrow the index'
        return _cap(f'{name}[{index or ":"}] shape {list(a.shape)}\n'
                    + np.array2string(a, precision=4, threshold=4096, max_line_width=160, suppress_small=False), cap)

    def python(code):
        t0 = time.perf_counter()
        r = ctx.sandbox.exec(code)
        ctx.cost.harness['sandbox_s'] += time.perf_counter() - t0
        ctx.cost.harness['sandbox_calls'] += 1
        txt = (r.get('out') or '')
        if r.get('error'):
            txt += ('\n' if txt else '') + 'ERROR: ' + r['error']
        return _cap(txt or '(no output)', cap)

    def probe(kind='metrics', weights='staged'):
        if kind not in PROBE_KINDS:
            return f'error: kind must be one of {PROBE_KINDS}'
        if weights not in ('staged', 'committed'):
            return "error: weights must be 'staged' or 'committed'"
        if ctx.budget.probes_left() <= 0:
            return 'error: no probes left. Submit an edit (submit_edit) or finish.'
        W = None
        if weights == 'staged':
            W, err = ctx.sandbox.staged()
            if err:
                return f'error: {err}'
            st = edit_stats(W, ctx.W_cur)
            if not (st['shape_ok'] and st['finite']):
                return 'error: staged weights have a changed shape or non-finite values'
        res = ctx.run_probe(kind, W)
        ctx.budget.charge_probe()
        res['probes_left'] = ctx.budget.probes_left()
        return _cap(dumps(res), cap)

    def submit_edit(rationale):
        return _cap(dumps(ctx.submit(rationale)), cap)

    def history():
        return _cap(dumps(ctx.history()), cap)

    def finish(summary):
        ctx.finished = True
        ctx.finish_summary = summary
        return 'Run finished. The committed weights are final.'

    return {'model_card': model_card, 'tensor_stats': tensor_stats, 'read_tensor': read_tensor, 'python': python,
            'probe': probe, 'submit_edit': submit_edit, 'history': history, 'finish': finish}


class RunState:
    """Mutable state of one run, shared by tools, agent and baseline loops."""

    def __init__(self, adapter, budget, cost, log, cfg, gate_cfg, W_start, W_cur, cur_val, sandbox=None,
                 on_commit=None, on_save=None, val_stage1=None, card=None):
        self.adapter, self.budget, self.cost, self.log = adapter, budget, cost, log
        self.cfg, self.gate_cfg = cfg, gate_cfg
        self.W_start, self.W_cur, self.cur_val = W_start, W_cur, cur_val
        self.sandbox = sandbox
        self.on_commit, self.on_save = on_commit, on_save
        self.val_stage1 = val_stage1
        self.card = card if card is not None else adapter.describe()
        self.gate_rules = {k: gate_cfg[k] for k in ('max_changed_frac', 'max_rel_l2', 'confidence', 'guard_tol')}
        self.edits, self.probes = [], []
        self.finished, self.finish_summary = False, ''
        self._cur_probe = {}

    def save(self):
        if self.on_save:
            self.on_save(self)

    # ------------------------------------------------------------ weights
    def weights(self, staged):
        if staged:
            return self.sandbox.staged()
        return self.W_cur, None

    # ------------------------------------------------------------ probes
    def _count(self, split):
        self.cost.harness[f'{split}_evals'] += 1

    def run_probe(self, kind, W=None):
        a = self.adapter
        if W is None:
            res = a.probe(kind)
            self._count('probe')
        else:
            with a.using(W):
                res = a.probe(kind)
            self._count('probe')
            if kind in ('metrics', 'errors'):
                cur = self._committed_probe(kind)
                res['committed_overall'] = cur.get('overall')
                if kind == 'metrics':
                    cs = {r['slice']: r for r in cur.get('slices', [])}
                    for r in res.get('slices', []):
                        c = cs.get(r['slice'])
                        if c is not None:
                            for k in ('loss', 'f1'):
                                if k in r and k in c and r[k] is not None and c[k] is not None:
                                    r[f'd_{k}'] = round(r[k] - c[k], 6)
        self.probes.append({'kind': kind, 'weights': 'committed' if W is None else 'staged',
                            'overall': res.get('overall')})
        self.log.event('probe', kind=kind, staged=W is not None, overall=res.get('overall'))
        return res

    def _committed_probe(self, kind):
        """Committed-weight probe results, computed once per committed weight set (not charged)."""
        if kind not in self._cur_probe:
            self._cur_probe[kind] = self.adapter.probe(kind)
            self._count('probe')
        return self._cur_probe[kind]

    # ------------------------------------------------------------ gate
    def gate(self, W, rationale, caps=True, source='agent'):
        """Score W on validation and keep it if the gate accepts. Returns the verdict record."""
        stats = edit_stats(W, self.W_cur, self.W_start)
        gcfg = self.gate_cfg if caps else {**self.gate_cfg, 'max_changed_frac': 1.0, 'max_rel_l2': float('inf')}
        problems = cap_problems(stats, gcfg)
        rec = {'id': len(self.edits) + 1, 'source': source, 'rationale': rationale,
               'changed': stats['changed'], 'frac_changed': round(stats['frac_changed'], 6),
               'rel_l2': round(stats['rel_l2'], 6), 'drift_from_stage1': stats['drift_from_stage1'],
               'tensors': stats['tensors'][:8]}
        if problems:
            self.budget.verdict(False)
            rec.update(accept=False, gate_charged=False, reasons=problems)
        elif not self.budget.charge_gate():
            rec.update(accept=False, gate_charged=False, reasons=['no gate evaluations left'])
        else:
            with self.adapter.using(W):
                new = self.adapter.evaluate('val')
            self._count('gate')
            v = decide(self.cur_val, new, stats, gcfg)
            self.budget.verdict(v['accept'])
            rec.update(accept=v['accept'], gate_charged=True, reasons=v['reasons'], delta=v['delta'],
                       upper=v['upper'], guards=v['guards'], val_before=self.cur_val.report, val_after=new.report)
            if v['accept']:
                self.W_cur = {k: np.array(x, np.float32, copy=True) for k, x in W.items()}
                self.adapter.load_weights(self.W_cur)
                self.adapter.on_accept(new)
                self.cur_val = new
                self._cur_probe = {}
                if self.sandbox is not None:
                    self.sandbox.commit(self.W_cur)
                if self.on_commit:
                    self.on_commit(self)
        self.edits.append(rec)
        self.log.event('gate', **{k: v for k, v in rec.items() if k not in ('val_before',)})
        return rec

    def submit(self, rationale):
        W, err = self.sandbox.staged()
        if err:
            return {'accept': False, 'reasons': [err]}
        rec = self.gate(W, rationale)
        out = {k: rec[k] for k in ('id', 'accept', 'reasons', 'changed', 'frac_changed', 'rel_l2') if k in rec}
        if rec.get('gate_charged'):
            out.update(change_in_loss=rec['delta'], upper_bound=rec['upper'], val_after=rec['val_after'],
                       guards=rec['guards'])
        out['budget'] = self.budget.status()
        out['note'] = ('accepted: committed weights updated, staged = committed' if rec['accept']
                       else 'rejected: committed weights unchanged; your staged edit is still in W')
        return out

    def history(self):
        return {'edits': [{k: e.get(k) for k in ('id', 'rationale', 'accept', 'reasons', 'delta', 'frac_changed',
                                                 'rel_l2')} for e in self.edits],
                'probes': self.probes[-12:], 'committed_validation': self.cur_val.report,
                'budget': self.budget.status()}
