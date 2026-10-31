"""Input (IPA) and output (OPA) poisoning attacks on SR / PM mean-variance estimation.

Li et al., USENIX Security 2023, Sec. 4. Every equation is re-derived from the
estimator in estimator.py (group means of Phi over exact halves of all N users):

  server's group-g estimate   (G_g + F_g) / N_g, with N_g = N / 2 for exact halves
  attacker's goal             E[estimate_g] = T_g, where
                                T_1 = target raw mean, and
                                T_2 = E[q(x)] at the target (raw mean, E[x^2] = var_t + mean_t^2)

  required fake decoded sum   R_g = (n_e + m)/2 * T_g - S_g,e / 2        
      with S_1,e = sum x and S_2,e = sum q(x), both from the attacker's knowledge
  sum of unbias(y) over fakes U_g = k_g (R_g - m_g lo_g) - m_g          
  SR output sum               (p - q) U_g    
  PM output sum               U_g            


IPA solves Eqs. 1-2 on the benchmark scale (y in [-1, 1]) and its inputs then go
through the actual randomizer (scenarios.py). OPA writes reports directly.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .estimator import G1, G2
from .knowledge import AttackerKnowledge
from .mechanisms import Mechanism, SR, PM
from .transforms import Transform

SUM_RULES = ('paper', 'group_aware')
PM_SPREAD_RATIOS = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)   # nominal half-width / plateau width
_TOL = 1e-9


class AttackStatus(str, Enum):
    FEASIBLE = 'feasible'                       # constructed, constraints met
    TARGET_ERROR = 'generated_with_target_error'  # constructed; integer rounding / solver residual
    INFEASIBLE = 'infeasible_target'            # cannot be constructed with the realized group allocation
    SOLVER_FAILURE = 'solver_failure'           # optimizer did not converge (not proof of infeasibility)
    SATURATED = 'below_budget_saturated'        # below-budget best effort (separate evaluation only)

    @property
    def generated(self) -> bool:
        return self in (AttackStatus.FEASIBLE, AttackStatus.TARGET_ERROR, AttackStatus.SATURATED)


@dataclass
class AttackResult:
    status: AttackStatus
    reports: Optional[np.ndarray] = None    # OPA: one report per fake, aligned with fake_groups
    inputs: Optional[np.ndarray] = None     # IPA: raw-unit fake inputs (randomized later)
    info: dict = field(default_factory=dict)


def group_targets(tf: Transform, mean_t: float, var_t: float):
    mean_raw, second_raw = tf.raw_targets(mean_t, var_t)
    return mean_raw, float(tf.query_mean(mean_raw, second_raw))


def knowledge_sums(tf: Transform, k: AttackerKnowledge): ## This should be an estimate
    """Estimated genuine-population totals: (sum x, sum q(x)).

    Derived only from the attacker's estimated count and moments.
    Does not access the current genuine users' raw values or reports."""
    return k.S1_e, k.n_e * float(tf.query_mean(k.mean_e, k.second_e))


def required_fake_sums(tf: Transform, k: AttackerKnowledge, m: int, m_per_group: dict,
                       n_total: int, mean_t: float, var_t: float, sum_rule: str = 'paper') -> dict:
    """{group: R_g}, the decoded sum the group's fakes must contribute.
    """
    if sum_rule not in SUM_RULES:
        raise ValueError(f"sum_rule must be one of {SUM_RULES}, got {sum_rule!r}")
    T = dict(zip((G1, G2), group_targets(tf, mean_t, var_t)))
    S = dict(zip((G1, G2), knowledge_sums(tf, k)))
    sizes = {G1: (n_total + 1) // 2, G2: n_total // 2}
    out = {}
    for g in (G1, G2):
        if sum_rule == 'paper':
            out[g] = (k.n_e + m) / 2.0 * T[g] - S[g] / 2.0
        else:
            size_e = (k.n_e + m) * sizes[g] / n_total
            out[g] = size_e * T[g] - (size_e - m_per_group[g]) * S[g] / k.n_e
    return out


def unbiased_fake_sum(tf: Transform, g: int, R: float, m_g: int) -> float:
    k, lo = (tf.k1, tf.a) if g == G1 else (tf.k2, tf.q_lo)
    return k * (R - m_g * lo) - m_g


# --------------------------------------------------------------------------- OPA
def pm_spread(c: float, m_g: int, s: float, half_width: float, rng: np.random.Generator):
    """m_g PM reports with mean exactly c, spread up to `half_width`, inside (-s, s).

    Sampler:
      1. h = min(half_width, s - |c|), the widest window around c that fits;
         `truncated` records h < half_width.
      2. u_i ~ Uniform(-h, h) i.i.d.; d_i = u_i - mean(u)  (so sum d = 0).
      3. Centering can push a value past the window by |mean(u)|. All d_i are
         scaled by one factor lam <= 1 so that max |d_i| <= h and every value
         stays strictly inside (-s, s); scaling keeps sum d = 0.
      4. Reports c + lam d_i. Nothing is clipped, so the sum is c * m_g and the
         realized half-width never exceeds the achievable one.
    """
    info = {'nominal_half_width': float(half_width)}
    if m_g == 0:
        info.update(achievable_half_width=0.0, truncated=False, realized_half_width=0.0,
                    realized_std=0.0, scale=1.0)
        return np.empty(0), info
    room = s - abs(c)
    h = max(0.0, min(half_width, room))
    info['achievable_half_width'] = float(h)
    info['truncated'] = bool(h < half_width - 1e-12)
    if h <= 0.0 or m_g == 1:
        v = np.full(m_g, float(c))
        info.update(realized_half_width=0.0, realized_std=0.0, scale=1.0)
        return v, info
    d = rng.uniform(-h, h, m_g)
    d -= d.mean()
    up, down = d.max(), -d.min()
    lam = min(1.0, h / max(up, down))
    limit_up, limit_down = (s - c) * (1 - 1e-9), (s + c) * (1 - 1e-9)
    if up > limit_up:
        lam = min(lam, limit_up / up)
    if down > limit_down:
        lam = min(lam, limit_down / down)
    v = c + lam * d
    info.update(realized_half_width=float(np.max(np.abs(v - c))), realized_std=float(v.std()),
                scale=float(lam))
    return v, info


def opa(mech: Mechanism, tf: Transform, k: AttackerKnowledge, fake_groups: np.ndarray,
        n_total: int, mean_t: float, var_t: float, rng: np.random.Generator,
        spread_ratio: float = 0.25, sum_rule: str = 'paper', saturate: bool = False) -> AttackResult:
    """Output poisoning: SR fakes send +-1 counts, PM fakes send spread values.
    """
    fake_groups = np.asarray(fake_groups)
    m = len(fake_groups)
    m_per_group = {g: int(np.sum(fake_groups == g)) for g in (G1, G2)}
    R = required_fake_sums(tf, k, m, m_per_group, n_total, mean_t, var_t, sum_rule)
    reports = np.empty(m, dtype=np.float64)
    info = {'sum_rule': sum_rule, 'groups': {}}
    status = AttackStatus.FEASIBLE
    for g in (G1, G2):
        idx = np.flatnonzero(fake_groups == g)
        m_g = len(idx)
        U = unbiased_fake_sum(tf, g, R[g], m_g)
        gi = {'m_g': m_g, 'R': float(R[g]), 'U': float(U)}
        if isinstance(mech, SR):
            T = (mech.p - mech.q) * U
            gi['output_sum_target'] = float(T)
            if abs(T) > m_g + _TOL * max(1, m_g):
                gi['reason'] = f"|output sum| {abs(T):.4g} > m_g {m_g}"
                if not saturate:
                    info['groups'][int(g)] = gi
                    return AttackResult(AttackStatus.INFEASIBLE, info=info)
                gi['saturated'] = True
                status = AttackStatus.SATURATED
            n_plus = int(np.clip(np.rint((m_g + T) / 2.0), 0, m_g))
            vals = np.r_[np.ones(n_plus), -np.ones(m_g - n_plus)]
            rng.shuffle(vals)
            resid = float((2 * n_plus - m_g) - T)
            gi.update(n_plus=n_plus, output_sum_residual=resid,
                      decoded_sum_residual=resid / (mech.p - mech.q) / (tf.k1 if g == G1 else tf.k2))
            if abs(resid) > _TOL and status != AttackStatus.SATURATED:
                status = AttackStatus.TARGET_ERROR
        elif isinstance(mech, PM):
            if m_g == 0:
                if abs(U) > _TOL:
                    gi['reason'] = "no fakes in group but nonzero sum required"
                    info['groups'][int(g)] = gi
                    if not saturate:
                        return AttackResult(AttackStatus.INFEASIBLE, info=info)
                    status = AttackStatus.SATURATED
                vals, sp = np.empty(0), {}
            else:
                c = U / m_g
                gi['per_fake_value'] = float(c)
                if abs(c) > mech.s + _TOL:
                    gi['reason'] = f"|per-fake value| {abs(c):.4g} > s {mech.s:.4g}"
                    if not saturate:
                        info['groups'][int(g)] = gi
                        return AttackResult(AttackStatus.INFEASIBLE, info=info)
                    gi['saturated'] = True
                    status = AttackStatus.SATURATED
                    c = float(np.sign(c) * mech.s * (1 - 1e-9))
                c = float(np.clip(c, -mech.s, mech.s))
                vals, sp = pm_spread(c, m_g, mech.s, spread_ratio * (mech.s - 1.0), rng)
                gi['spread'] = sp
                gi['output_sum_residual'] = float(vals.sum() - U)
        else:
            raise TypeError(f"unsupported mechanism {mech!r}")
        reports[idx] = vals
        info['groups'][int(g)] = gi
    info['spread_ratio'] = float(spread_ratio) if isinstance(mech, PM) else None
    return AttackResult(status, reports=reports, info=info)


def ipa_requirements(tf: Transform, k: AttackerKnowledge, m: int, mean_t: float, var_t: float):
    S1n, S2n = tf.norm_sums(k.S1_e, k.S2_e, k.n_e)
    N_e = k.n_e + m
    return N_e * mean_t - S1n, N_e * (var_t + mean_t ** 2) - S2n


def thm1_vector(A: float, m: int) -> np.ndarray:
    """maximiser of Sum y^2 under Sum y = A, y in [-1, 1]."""
    if abs(A) >= m:
        return np.full(m, np.sign(A) if A else 0.0)
    j = int(np.floor((A + m) / 2.0))
    left = A - j + (m - 1 - j)
    return np.r_[np.ones(j), -np.ones(m - 1 - j), [left]]


def ipa_bounds(A: float, m: int):
    if abs(A) >= m:
        return A * A / m, float(m)
    j = np.floor((A + m) / 2.0)
    left = A - j + (m - 1 - j)
    return A * A / m, float(m - 1 + left * left)


def _project(z: np.ndarray, A: float, iters: int = 60) -> np.ndarray:
    lo, hi = z.min() - 1.0, z.max() + 1.0
    for _ in range(iters):
        tau = 0.5 * (lo + hi)
        if np.clip(z - tau, -1.0, 1.0).sum() > A:
            lo = tau
        else:
            hi = tau
    return np.clip(z - 0.5 * (lo + hi), -1.0, 1.0)


def _adam(A: float, B: float, m: int, rng, lr: float, iters: int):
    """
    The gradient is projected onto the tangent space of Sum y = A before the Adam
    step. Without that, Adam's per-coordinate normalisation moves every value by
    about lr in the direction of its sign, and once all values share a sign the
    sum projection cancels the step exactly (a fixed point above B). Adam keeps
    oscillating at about lr near the optimum, so the iterate with the smallest
    residual is returned.
    """
    y = _project(rng.uniform(-1.0, 1.0, m), A)
    best, best_res = y, abs(np.dot(y, y) - B)
    mom, vel = np.zeros(m), np.zeros(m)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        grad = 4.0 * (np.dot(y, y) - B) * (y - y.mean())
        mom = b1 * mom + (1 - b1) * grad
        vel = b2 * vel + (1 - b2) * grad * grad
        y = _project(y - lr * (mom / (1 - b1 ** t)) / (np.sqrt(vel / (1 - b2 ** t)) + eps), A)
        res = abs(np.dot(y, y) - B)
        if res < best_res:
            best, best_res = y, res
    return best


def ipa_inputs(tf: Transform, k: AttackerKnowledge, m: int, mean_t: float, var_t: float,
               rng: np.random.Generator, solver: str = 'closed_form', adam_lr: float = 1e-3,
               adam_iters: int = 10_000, accept_rel: float = 1e-3, saturate: bool = False) -> AttackResult:
    """Bounded fake inputs for IPA, returned in raw units (randomized by the caller).
    """
    A, B = ipa_requirements(tf, k, m, mean_t, var_t)
    info = {'A': float(A), 'B': float(B), 'solver': solver}
    if m < 1:
        info['reason'] = 'no fakes'
        return AttackResult(AttackStatus.INFEASIBLE, info=info)
    Bmin, Bmax = ipa_bounds(A, m) if abs(A) <= m else (np.inf, -np.inf)
    info.update(B_min=float(Bmin), B_max=float(Bmax))
    tol = _TOL * max(1.0, m)
    saturated = False
    if abs(A) > m + tol or B < Bmin - tol or B > Bmax + tol:
        info['reason'] = 'outside Eqs. 8-9 (Thm. 1 bounds)'
        if not saturate:
            return AttackResult(AttackStatus.INFEASIBLE, info=info)
        # below-budget best effort: nearest jointly reachable (A, B)
        A = float(np.clip(A, -m, m))
        Bmin, Bmax = ipa_bounds(A, m)
        B = float(np.clip(B, Bmin, Bmax))
        info.update(A_used=A, B_used=B)
        saturated, solver = True, 'closed_form'
    if solver == 'closed_form':
        span = Bmax - Bmin
        lam = float(np.sqrt(np.clip((B - Bmin) / span, 0.0, 1.0))) if span > 0 else 0.0
        y = (1.0 - lam) * (A / m) + lam * thm1_vector(A, m)
        rng.shuffle(y)
        info['lambda'] = lam
        converged = True
    elif solver == 'adam':
        y = _adam(A, B, m, rng, adam_lr, adam_iters)
        converged = None
    else:
        raise ValueError(f"unknown IPA solver {solver!r}")
    y = np.clip(y, -1.0, 1.0)
    res_sum, res_sq = float(y.sum() - A), float(np.dot(y, y) - B)
    rel = abs(res_sq) / max(1.0, abs(B))
    converged = bool(rel <= accept_rel) if converged is None else converged
    info.update(residual_sum=res_sum, residual_sq=res_sq, relative_residual_sq=rel, converged=converged)
    if not converged:
        return AttackResult(AttackStatus.SOLVER_FAILURE, info=info)
    status = AttackStatus.FEASIBLE if max(abs(res_sum), abs(res_sq)) <= 1e-6 * max(1.0, m) \
        else AttackStatus.TARGET_ERROR
    if saturated:
        status = AttackStatus.SATURATED
    x = np.clip(tf.from_g1(y), tf.a, tf.b)
    return AttackResult(status, inputs=x, info=info)
