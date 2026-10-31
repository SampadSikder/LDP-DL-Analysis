"""Theoretical feasibility checks and the minimum attack budget.
"""

from dataclasses import dataclass, field

import numpy as np

from .attacks import group_targets, ipa_bounds, ipa_requirements, required_fake_sums, unbiased_fake_sum
from .estimator import G1, G2, Estimate
from .knowledge import AttackerKnowledge
from .mechanisms import PM, SR, Mechanism
from .transforms import Transform

BUDGET_FACTORS = (1.0, 1.1, 1.25, 1.5, 2.0)
_TOL = 1e-9


def planned_allocation(m: int) -> dict:
    return {G1: (m + 1) // 2, G2: m // 2}


@dataclass
class Check:
    attack: str
    m: int
    feasible: bool
    oracle: bool = False
    groups: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'attack': self.attack, 'm': self.m, 'feasible': self.feasible, 'oracle': self.oracle,
                'groups': {str(g): v for g, v in self.groups.items()}, **self.info}


def _benchmark(tf: Transform, mean_raw: float, q_mean: float):
    e = Estimate.from_moments(mean_raw, tf.second_from_query(q_mean, mean_raw)).normalized(tf)
    return e.mean, e.variance


def check_opa(mech: Mechanism, tf: Transform, k: AttackerKnowledge, m: int, mean_t: float, var_t: float,
              spread_ratio: float = 0.0, sum_rule: str = 'paper', oracle: bool = False) -> Check:
    name = f"OPA-{mech.name}"
    alloc = planned_allocation(m)
    n_total = int(round(k.n_e)) + m
    R = required_fake_sums(tf, k, m, alloc, n_total, mean_t, var_t, sum_rule)
    T_targets = dict(zip((G1, G2), group_targets(tf, mean_t, var_t)))
    groups, ok = {}, True
    dec_resid = {}
    for g in (G1, G2):
        m_g = alloc[g]
        U = unbiased_fake_sum(tf, g, R[g], m_g)
        k_g = tf.k1 if g == G1 else tf.k2
        gi = {'m_g': m_g, 'R': float(R[g]), 'U': float(U)}
        if isinstance(mech, SR):
            T = (mech.p - mech.q) * U
            cap = float(m_g)
            feas = abs(T) <= cap + _TOL * max(1, m_g)
            n_plus = int(np.clip(np.rint((m_g + T) / 2.0), 0, m_g))
            achieved = 2 * n_plus - m_g
            gi.update(output_sum=float(T), output_cap=cap, n_plus=n_plus, n_minus=m_g - n_plus,
                      output_sum_achieved=float(achieved), output_sum_residual=float(achieved - T))
            dec_resid[g] = (achieved - T) / (mech.p - mech.q) / k_g
        elif isinstance(mech, PM):
            cap = mech.s * m_g
            feas = abs(U) <= cap + _TOL * max(1, m_g)
            c = U / m_g if m_g else (0.0 if abs(U) <= _TOL else np.inf)
            room = mech.s - abs(c)
            req_hw = spread_ratio * (mech.s - 1.0)
            gi.update(output_sum=float(U), output_cap=float(cap), per_fake_value=float(c),
                      spread_room=float(room), spread_requested=float(req_hw),
                      spread_truncated=bool(feas and req_hw > room + 1e-12))
            dec_resid[g] = 0.0
        else:
            raise TypeError(f"unsupported mechanism {mech!r}")
        gi['margin'] = float(cap - abs(gi['output_sum']))
        gi['feasible'] = bool(feas)
        ok &= feas
        groups[int(g)] = gi
    info = {}
    if ok:
        # target approximation error (benchmark scale) from integer counts, at the planned group sizes
        size = {G1: n_total / 2.0, G2: n_total / 2.0}
        base = _benchmark(tf, T_targets[G1], T_targets[G2])
        got = _benchmark(tf, T_targets[G1] + dec_resid[G1] / size[G1], T_targets[G2] + dec_resid[G2] / size[G2])
        info.update(approx_error_mean=float(got[0] - base[0]), approx_error_variance=float(got[1] - base[1]))
    return Check(name, m, bool(ok), oracle, groups, info)


def check_ipa(tf: Transform, k: AttackerKnowledge, m: int, mean_t: float, var_t: float,
              oracle: bool = False) -> Check:
    if m < 1:
        return Check('IPA', m, False, oracle, info={'reason': 'no fakes'})
    A, B = ipa_requirements(tf, k, m, mean_t, var_t)
    info = {'A': float(A), 'B': float(B), 'A_cap': float(m)}
    if abs(A) > m + _TOL * m:
        return Check('IPA', m, False, oracle, info=dict(info, reason='|A| > m'))
    Bmin, Bmax = ipa_bounds(A, m)
    tol = _TOL * max(1.0, m)
    feas = Bmin - tol <= B <= Bmax + tol
    info.update(B_min=float(Bmin), B_max=float(Bmax), margin=float(min(B - Bmin, Bmax - B)))
    if not feas:
        info['reason'] = 'B outside [A^2/m, B_max]' + (' (A alone is reachable)' if abs(A) <= m else '')
    return Check('IPA', m, bool(feas), oracle, info=info)


def check(attack: str, mech: Mechanism, tf: Transform, k: AttackerKnowledge, m: int, mean_t: float,
          var_t: float, spread_ratio: float = 0.0, sum_rule: str = 'paper', oracle: bool = False) -> Check:
    if attack == 'IPA':
        return check_ipa(tf, k, m, mean_t, var_t, oracle)
    if attack == 'OPA':
        return check_opa(mech, tf, k, m, mean_t, var_t, spread_ratio, sum_rule, oracle)
    raise ValueError(f"unknown attack {attack!r}")


def needs_no_fakes(tf: Transform, k: AttackerKnowledge, mean_t: float, var_t: float, tol: float = 1e-12) -> bool:
    """True when the attacker's own estimate already equals the target (m_min = 0)."""
    mean_e = tf.norm_mean(k.mean_e)
    var_e = tf.norm_var(k.second_e - k.mean_e ** 2)
    return abs(mean_e - mean_t) <= tol and abs(var_e - var_t) <= tol


def minimum_budget(attack: str, mech: Mechanism, tf: Transform, k: AttackerKnowledge, mean_t: float,
                   var_t: float, m_max: int = None, sum_rule: str = 'paper', oracle: bool = False,
                   scan: int = 64) -> dict:
    """m_min = min{m : both target-moment constraints are feasible}, beta_min = m_min / (n + m_min).
    """
    n = int(round(k.n_e))
    out = {'attack': attack if attack == 'IPA' else f"OPA-{mech.name}", 'oracle': oracle, 'n_planning': n,
           'q_t': float(mean_t ** 2 + var_t)}
    if needs_no_fakes(tf, k, mean_t, var_t):
        return dict(out, m_min=0, beta_min=0.0, status='no_fakes_needed')
    m_max = m_max or 100 * max(n, 1)
    ok = lambda m: check(attack, mech, tf, k, m, mean_t, var_t, sum_rule=sum_rule, oracle=oracle).feasible
    hi = 1
    while hi <= m_max and not ok(hi):
        hi *= 2
    if hi > m_max:
        if not ok(m_max):
            return dict(out, m_min=None, beta_min=float('nan'), status='infeasible_up_to_m_max', m_max=m_max)
        hi = m_max
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    m_min = hi
    for m in range(max(1, hi - scan), hi):
        if ok(m):
            m_min = m
            break
    return dict(out, m_min=int(m_min), beta_min=m_min / (n + m_min), status='feasible')


def budget_ladder(m_min: int, factors=BUDGET_FACTORS) -> list:
    out = []
    for f in factors:
        m = int(np.ceil(f * m_min - 1e-9))
        if m not in out:
            out.append(m)
    return out


def opa_sufficient(mech, tf, k, m, mean_t, var_t):
    c = check_opa(mech, tf, k, m, mean_t, var_t)
    return c.feasible, {g: abs(v['output_sum']) / max(v['m_g'], 1) for g, v in c.groups.items()}


def ipa_sufficient(tf, k, m, mean_t, var_t):
    c = check_ipa(tf, k, m, mean_t, var_t)
    return c.feasible, c.info


def sufficient(attack, mech, tf, k, m, mean_t, var_t) -> bool:
    return check(attack, mech, tf, k, m, mean_t, var_t).feasible


def beta_min(attack, mech, tf, k, mean_t, var_t, m_max=None):
    r = minimum_budget(attack, mech, tf, k, mean_t, var_t, m_max=m_max)
    return r['beta_min'], r['m_min']
