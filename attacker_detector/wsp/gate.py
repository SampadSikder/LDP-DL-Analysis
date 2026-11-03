"""Acceptance gate: plain code, never the LLM.

An edit is kept only if
  * keys and shapes are unchanged and every value is finite,
  * it changes at most `max_changed_frac` of all weights and has relative L2 <= `max_rel_l2`
    (both against the committed weights),
  * the validation score improves: one-sided paired bootstrap, the `confidence` upper bound of
    (new - old) is below 0 (losses: lower is better),
  * no guard metric gets worse by more than `guard_tol` (relative).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DEFAULT_GATE = {'max_changed_frac': 0.05, 'max_rel_l2': 0.10, 'bootstrap': 1000, 'confidence': 0.90,
                'guard_tol': 0.02, 'seed': 0}


@dataclass
class Score:
    primary: float                              # lower is better
    units: Optional[np.ndarray] = None          # per-unit losses (rounds), for the paired bootstrap
    guards: dict = field(default_factory=dict)  # lower is better
    report: dict = field(default_factory=dict)  # readable metrics for the agent and logs
    y: Optional[np.ndarray] = None              # F1-type scores: labels and 0/1 predictions
    pred: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)   # adapter state tied to this score (e.g. a threshold)


def edit_stats(W_new, W_cur, W_start=None):
    keys_ok = set(W_new) == set(W_cur)
    shape_ok, finite = True, True
    changed = total = 0
    d2 = n2 = s2 = s0 = 0.0
    per = []
    for k, a0 in W_cur.items():
        a = W_new.get(k)
        total += a0.size
        if a is None or a.shape != a0.shape:
            shape_ok = False
            continue
        if not np.all(np.isfinite(a)):
            finite = False
        d = a.astype(np.float64) - a0
        n = int(np.count_nonzero(d))
        dd, nn = float(np.sum(d ** 2)), float(np.sum(a0.astype(np.float64) ** 2))
        d2, n2 = d2 + dd, n2 + nn
        if n:
            changed += n
            per.append({'tensor': k, 'changed': n, 'size': int(a0.size),
                        'rel_l2': float(np.sqrt(dd / max(nn, 1e-24)))})
        if W_start is not None and k in W_start:
            ds = a.astype(np.float64) - W_start[k]
            s2 += float(np.sum(ds ** 2))
            s0 += float(np.sum(W_start[k].astype(np.float64) ** 2))
    return {'keys_ok': keys_ok, 'shape_ok': shape_ok, 'finite': finite, 'changed': changed, 'total': total,
            'frac_changed': changed / max(total, 1), 'rel_l2': float(np.sqrt(d2 / max(n2, 1e-24))),
            'drift_from_stage1': float(np.sqrt(s2 / max(s0, 1e-24))) if W_start is not None else None,
            'tensors': sorted(per, key=lambda r: -r['rel_l2'])}


def cap_problems(stats, cfg):
    cfg = {**DEFAULT_GATE, **(cfg or {})}
    p = []
    if not stats['keys_ok']:
        p.append('tensor names changed')
    if not stats['shape_ok']:
        p.append('a tensor shape changed')
    if not stats['finite']:
        p.append('non-finite values')
    if stats['changed'] == 0:
        p.append('no weights changed')
    if stats['frac_changed'] > cfg['max_changed_frac']:
        p.append(f"changes {stats['frac_changed']:.2%} of weights > cap {cfg['max_changed_frac']:.2%}")
    if stats['rel_l2'] > cfg['max_rel_l2']:
        p.append(f"relative L2 change {stats['rel_l2']:.4f} > cap {cfg['max_rel_l2']}")
    return p


def _f1(y, pred, w):
    tp = (w * ((pred == 1) & (y == 1))).sum(-1)
    fp = (w * ((pred == 1) & (y == 0))).sum(-1)
    fn = (w * ((pred == 0) & (y == 1))).sum(-1)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


def paired_delta(old: Score, new: Score, n_boot=1000, confidence=0.90, seed=0):
    """Mean change new - old of the loss and its bootstrap (lower, upper) bounds at `confidence`
    one-sided each. Negative = better."""
    rng = np.random.default_rng(seed)
    if old.units is not None and new.units is not None:
        d = np.asarray(new.units, float) - np.asarray(old.units, float)
        n = len(d)
        boots = np.empty(n_boot)
        for i in range(0, n_boot, 200):
            b = min(200, n_boot - i)
            boots[i:i + b] = d[rng.integers(0, n, size=(b, n))].mean(1)
        delta = float(d.mean())
    elif old.y is not None and new.y is not None:
        y, po, pn = old.y.astype(np.int8), old.pred.astype(np.int8), new.pred.astype(np.int8)
        boots = np.empty(n_boot)
        for i in range(0, n_boot, 100):
            b = min(100, n_boot - i)
            w = rng.poisson(1.0, size=(b, len(y))).astype(np.float32)
            boots[i:i + b] = _f1(y, po, w) - _f1(y, pn, w)       # loss = 1 - F1: new - old
        ones = np.ones((1, len(y)), np.float32)
        delta = float((_f1(y, po, ones) - _f1(y, pn, ones))[0])
    else:
        return {'delta': float(new.primary - old.primary), 'lower': np.nan, 'upper': np.nan}
    return {'delta': delta, 'lower': float(np.quantile(boots, 1 - confidence)),
            'upper': float(np.quantile(boots, confidence))}


def decide(old: Score, new: Score, stats, cfg=None):
    cfg = {**DEFAULT_GATE, **(cfg or {})}
    reasons = cap_problems(stats, cfg)
    pd_ = paired_delta(old, new, cfg['bootstrap'], cfg['confidence'], cfg['seed'])
    if not reasons:
        if not np.isfinite(pd_['upper']) or pd_['upper'] >= 0:
            reasons.append(f"no reliable improvement: change {pd_['delta']:+.5g}, "
                           f"{cfg['confidence']:.0%} upper bound {pd_['upper']:+.5g} (needs < 0)")
    guards = {}
    for g, v0 in old.guards.items():
        v1 = new.guards.get(g, np.inf)
        lim = v0 + cfg['guard_tol'] * abs(v0) + 1e-12
        guards[g] = {'old': v0, 'new': v1, 'limit': lim}
        if not reasons and not (v1 <= lim):
            reasons.append(f"guard {g} worse: {v1:.5g} > limit {lim:.5g} (was {v0:.5g})")
    return {'accept': not reasons, 'reasons': reasons, 'delta': pd_['delta'], 'lower': pd_['lower'],
            'upper': pd_['upper'], 'old_primary': old.primary, 'new_primary': new.primary, 'guards': guards}
