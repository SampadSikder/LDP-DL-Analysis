"""Generation of one reporting round, split into what the server sees and what it does not.

A round is fully determined by its RoundSpec (seeded), so it can be regenerated
bit for bit. Its checksum, stored in the manifest, verifies that.
"""

import hashlib
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import numpy as np

from .attacks import AttackStatus, ipa_inputs, opa
from .feasibility import check, minimum_budget
from .estimator import G1, G2, estimate, exact_moments, perturb_users, split_groups
from .knowledge import exact, from_compromised, round_count
from .mechanisms import SR, get_mechanism
from .populations import stable_seed
from .transforms import Transform

ATTACKS = ('none', 'IPA', 'OPA')
TARGET_MODES = ('offset', 'absolute')


def target_label(mi: int, vi: int) -> str:
    return f"m{mi + 1}v{vi + 1}"


def resolve_targets(printed: dict, mode: str, genuine_n, mi: int, vi: int):
    """Benchmark-scale (mean_t, var_t). Index 1 preserves the round's genuine moment."""
    pm, pv = printed['mean'], printed['variance']
    if mode == 'absolute':
        mean_t = genuine_n.mean if mi == 1 else pm[mi]
        var_t = genuine_n.variance if vi == 1 else pv[vi]
    elif mode == 'offset':
        mean_t = genuine_n.mean + (pm[mi] - pm[1])
        var_t = genuine_n.variance * pv[vi] / pv[1]
    else:
        raise ValueError(f"target mode must be one of {TARGET_MODES}")
    return float(mean_t), float(var_t)


@dataclass(frozen=True)
class RoundSpec:
    experiment: str
    split: str
    source: str
    population_id: str
    domain: tuple
    protocol: str
    epsilon: float
    total_size: int
    beta: float
    rep: int
    attack: str = 'none'
    target: str = 'clean'                
    target_mode: str = 'offset'
    square_mode: str = 'raw'
    knowledge_regime: str = 'compromised'
    knowledge_h: int = 1000
    n_e_rule: str = 'sig2'
    dS1_n: float = 0.0                    # attacker estimation errors, benchmark-scale sums
    dS2_n: float = 0.0
    dn: float = 0.0
    spread_ratio: Optional[float] = None
    ipa_solver: str = 'closed_form'
    sum_rule: str = 'paper'
    size_mode: str = 'fixed_total'        # 'fixed_genuine': total_size = n, m = beta n / (1 - beta) (paper scale)
    target_mean_value: Optional[float] = None   # target == 'explicit'; None preserves the genuine moment
    target_var_value: Optional[float] = None
    budget_mode: str = 'beta'             # 'beta': m from beta | 'mmin_multiple': m = ceil(budget_factor * m_min)
    budget_factor: Optional[float] = None
    below_budget: str = 'skip'            # 'skip': infeasible -> not generated | 'saturate': separate evaluation

    def _id_dict(self) -> dict:
        """Fields hashed into ids; budget fields at their defaults are omitted so ids of
        rounds generated before they existed do not change."""
        d = asdict(self)
        for f, default in (('budget_mode', 'beta'), ('budget_factor', None), ('below_budget', 'skip')):
            if d[f] == default:
                d.pop(f)
        return d

    @property
    def base_round_id(self) -> str:
        key = (self.experiment, self.population_id, self.protocol, self.epsilon, self.square_mode,
               self.total_size, self.beta, self.rep, self.knowledge_regime, self.knowledge_h, self.n_e_rule,
               self.size_mode)
        if self.budget_mode != 'beta':
            key = key + (self.budget_mode, self.budget_factor)
        return f"b{stable_seed('base', *key) % 16 ** 12:012x}"

    @property
    def round_id(self) -> str:
        return f"r{stable_seed('round', self._id_dict()) % 16 ** 12:012x}"

    @property
    def clean_counterpart_id(self) -> str:
        """The genuine-only view of this round (oracle removal), same genuine reports."""
        return f"{self.round_id}:genuine"

    def to_dict(self) -> dict:
        d = asdict(self)
        d['domain'] = list(self.domain)
        d.update(round_id=self.round_id, base_round_id=self.base_round_id,
                 clean_counterpart_id=self.clean_counterpart_id)
        return d


@dataclass
class Round:
    spec: RoundSpec
    status: str
    reports: Optional[np.ndarray] = None       
    groups: Optional[np.ndarray] = None        
    is_fake: Optional[np.ndarray] = None       
    labels: dict = field(default_factory=dict)  
    meta: dict = field(default_factory=dict)    

    @property
    def generated(self) -> bool:
        return self.reports is not None

    def observation(self) -> dict:
        """Everything a model may see. Nothing else from the round reaches a model."""
        s = self.spec
        return {'reports': self.reports, 'groups': self.groups,
                'n_g1': int(np.sum(self.groups == G1)), 'n_g2': int(np.sum(self.groups == G2)),
                'protocol': s.protocol, 'epsilon': s.epsilon, 'domain': tuple(s.domain),
                'square_mode': s.square_mode}

    def checksum(self) -> str:
        h = hashlib.sha256(self.reports.tobytes())
        h.update(self.groups.tobytes())
        h.update(self.is_fake.tobytes())
        return h.hexdigest()[:16]


def _knowledge(spec: RoundSpec, x_gen, tf: Transform, rng):
    n_e = round_count(len(x_gen), spec.n_e_rule)
    if spec.knowledge_regime == 'compromised':
        k, _ = from_compromised(x_gen, spec.knowledge_h, n_e, rng)
    elif spec.knowledge_regime == 'exact':
        k = exact(x_gen)
    else:
        raise ValueError(f"unknown knowledge regime {spec.knowledge_regime!r}")
    if spec.dS1_n or spec.dS2_n or spec.dn:
        k = inject_knowledge_error(k, tf, spec.dS1_n, spec.dS2_n, spec.dn)
    return k


def inject_knowledge_error(k, tf: Transform, dS1_n: float, dS2_n: float, dn: float):
    """Errors defined on the benchmark-scale sums (as in Table 4 / Figs. 7-9).

    The attacker's benchmark-scale knowledge becomes (S~1 + dS1_n, S~2 + dS2_n) at count
    n_e + dn, converted back to raw sums at that count. Shifting raw sums and n_e
    separately would move S~1 by dn * c as well (x~1 = c + k1 x), so it is not done.
    """
    S1n, S2n = tf.norm_sums(k.S1_e, k.S2_e, k.n_e)
    n_e = k.n_e + dn
    S1, S2 = tf.denorm_sums(S1n + dS1_n, S2n + dS2_n, n_e)
    return replace(k, n_e=n_e, S1_e=S1, S2_e=S2, regime=f"{k.regime}+error")


GENERATOR_LABEL = {AttackStatus.FEASIBLE: 'constructed', AttackStatus.TARGET_ERROR: 'constructed_with_rounding_error',
                   AttackStatus.SOLVER_FAILURE: 'solver_failure', AttackStatus.INFEASIBLE: 'allocation_infeasible',
                   AttackStatus.SATURATED: 'below_budget_saturated'}
ROUND_STATUS = {AttackStatus.FEASIBLE: 'feasible', AttackStatus.TARGET_ERROR: 'generated_with_target_error',
                AttackStatus.SOLVER_FAILURE: 'solver_failure', AttackStatus.INFEASIBLE: 'allocation_infeasible',
                AttackStatus.SATURATED: 'below_budget'}


def _targets(spec, printed_targets, truth_n):
    if spec.target == 'explicit':
        mean_t = truth_n.mean if spec.target_mean_value is None else float(spec.target_mean_value)
        var_t = truth_n.variance if spec.target_var_value is None else float(spec.target_var_value)
        return mean_t, var_t
    mi, vi = int(spec.target[1]) - 1, int(spec.target[3]) - 1
    return resolve_targets(printed_targets, spec.target_mode, truth_n, mi, vi)


def _verify_construction(mech, tf, res, fake_slots, fake_reports, attack):
    """Constraint residuals and support of what the generator produced."""
    out = {'support_ok': True}
    if attack == 'OPA':
        for g, gi in res.info.get('groups', {}).items():
            vals = fake_reports[fake_slots == int(g)]
            out[f'g{g}_unbiased_sum_residual'] = float(mech.unbias(vals).sum() - gi['U']) if len(vals) else 0.0
        if isinstance(mech, SR):
            out['support_ok'] = bool(np.all(np.isin(fake_reports, (-1.0, 1.0))))
        else:
            out['support_ok'] = bool(np.all(np.abs(fake_reports) < mech.s))
    else:
        out.update(residual_sum=res.info.get('residual_sum'), residual_sq=res.info.get('residual_sq'),
                   converged=res.info.get('converged'))
        out['support_ok'] = bool(res.inputs is None or
                                 np.all((res.inputs >= tf.a) & (res.inputs <= tf.b)))
    return out


def generate_round(spec: RoundSpec, population: np.ndarray, printed_targets: dict) -> Round:
    """Build one round from the parent population's values (pool order is random).
    """
    if spec.attack not in ATTACKS:
        raise ValueError(f"attack must be one of {ATTACKS}")
    tf = Transform(*spec.domain, square_mode=spec.square_mode)
    mech = get_mechanism(spec.protocol, spec.epsilon)
    attacked = spec.attack != 'none'
    if spec.budget_mode not in ('beta', 'mmin_multiple'):
        raise ValueError(f"unknown budget_mode {spec.budget_mode!r}")
    if spec.size_mode == 'fixed_genuine':
        n = spec.total_size
    elif spec.size_mode == 'fixed_total':
        if spec.budget_mode != 'beta':
            raise ValueError("budget_mode 'mmin_multiple' needs size_mode 'fixed_genuine' (n fixed)")
        n = spec.total_size - (int(round(spec.beta * spec.total_size)) if attacked else 0)
    else:
        raise ValueError(f"unknown size_mode {spec.size_mode!r}")
    if n < 2 or n > len(population):
        raise ValueError(f"genuine count {n} outside [2, {len(population)}]")
    x_gen = population[:n]
    truth = exact_moments(x_gen)
    truth_n = truth.normalized(tf)
    labels = {'genuine_mean': truth_n.mean, 'genuine_second_moment': truth_n.second_moment,
              'genuine_variance': truth_n.variance, 'genuine_mean_raw': truth.mean,
              'genuine_second_moment_raw': truth.second_moment, 'genuine_variance_raw': truth.variance}
    base = np.random.SeedSequence(stable_seed('base-stream', spec.base_round_id))
    s_groups, s_perturb, s_know = [np.random.default_rng(s) for s in base.spawn(3)]
    rnd = np.random.SeedSequence(stable_seed('round-stream', spec.round_id))
    s_attack, s_attack_perturb, s_shuffle = [np.random.default_rng(s) for s in rnd.spawn(3)]

    meta = {'n_genuine': n}
    m, saturate = 0, False
    if attacked:
        mean_t, var_t = _targets(spec, printed_targets, truth_n)
        k = _knowledge(spec, x_gen, tf, s_know)
        plan = minimum_budget(spec.attack, mech, tf, k, mean_t, var_t, sum_rule=spec.sum_rule)
        orc = minimum_budget(spec.attack, mech, tf, exact(x_gen), mean_t, var_t, sum_rule=spec.sum_rule, oracle=True)
        if spec.budget_mode == 'beta':
            m = (int(round(spec.beta * n / (1.0 - spec.beta))) if spec.size_mode == 'fixed_genuine'
                 else spec.total_size - n)
        elif plan['m_min'] is not None:
            m = int(np.ceil(spec.budget_factor * plan['m_min'] - 1e-9))
        meta.update(target_mean=mean_t, target_variance=var_t, target_second_moment=mean_t ** 2 + var_t,
                    target_valid_population=bool(var_t >= 0 and var_t <= (1 - mean_t) * (1 + mean_t)),
                    knowledge={'regime': k.regime, 'n_e': k.n_e, 'h': k.h, 'S1_e': k.S1_e, 'S2_e': k.S2_e})
        meta.update(n_fake=m, realized_beta=m / (n + m))
        meta['budget'] = {'mode': spec.budget_mode, 'factor': spec.budget_factor, 'm': m,
                          'm_min': plan['m_min'], 'beta_min': plan['beta_min'], 'plan_status': plan['status'],
                          'n_planning': plan['n_planning'],
                          'oracle_m_min': orc['m_min'], 'oracle_beta_min': orc['beta_min'],
                          'budget_ratio': (m / plan['m_min']) if plan['m_min'] else float('nan'),
                          'oracle_budget_ratio': (m / orc['m_min']) if orc['m_min'] else float('nan')}
        if m < 1:
            reason = 'no_fakes_needed' if plan['status'] == 'no_fakes_needed' else 'no_budget_exists'
            meta['theory'] = {'feasible': plan['status'] == 'no_fakes_needed', 'status': reason}
            meta['construction'] = {'status': 'not_attempted'}
            return Round(spec, reason, labels=labels, meta=meta)
        theory = check(spec.attack, mech, tf, k, m, mean_t, var_t, spread_ratio=spec.spread_ratio or 0.0,
                       sum_rule=spec.sum_rule)
        meta['theory'] = _jsonable(dict(theory.to_dict(), status='feasible' if theory.feasible else 'infeasible'))
        if not theory.feasible:
            if spec.below_budget != 'saturate':
                meta['construction'] = {'status': 'not_attempted'}
                return Round(spec, 'infeasible_target', labels=labels, meta=meta)
            saturate = True

    N = n + m
    groups = split_groups(N, s_groups)
    gen_slots, fake_slots = groups[:n], groups[n:]       # slot order is irrelevant: shuffled below
    gen_reports = perturb_users(x_gen, gen_slots, mech, tf, s_perturb)
    clean = estimate(gen_reports, gen_slots, mech, tf).normalized(tf)
    labels.update(clean_mean=clean.mean, clean_variance=clean.variance)
    meta.update(n_fake=m, realized_beta=m / N, m_g1=int(np.sum(fake_slots == G1)), m_g2=int(np.sum(fake_slots == G2)),
                n_g1_genuine=int(np.sum(gen_slots == G1)), n_g2_genuine=int(np.sum(gen_slots == G2)))

    fake_reports = np.empty(0)
    status = AttackStatus.FEASIBLE.value
    if attacked:
        if spec.attack == 'OPA':
            res = opa(mech, tf, k, fake_slots, N, mean_t, var_t, s_attack, spread_ratio=spec.spread_ratio or 0.0,
                      sum_rule=spec.sum_rule, saturate=saturate)
            fake_reports = res.reports
        else:
            res = ipa_inputs(tf, k, m, mean_t, var_t, s_attack, solver=spec.ipa_solver, saturate=saturate)
            if res.status.generated:
                fake_reports = perturb_users(res.inputs, fake_slots, mech, tf, s_attack_perturb)
        status = ROUND_STATUS[res.status]
        meta['attack_info'] = _jsonable(res.info)
        meta['construction'] = {'status': GENERATOR_LABEL[res.status]}
        if not res.status.generated:
            return Round(spec, status, labels=labels, meta=meta)
        meta['construction'].update(_jsonable(_verify_construction(mech, tf, res, fake_slots, fake_reports,
                                                                   spec.attack)))

    reports = np.r_[gen_reports, fake_reports]
    is_fake = np.r_[np.zeros(n, dtype=bool), np.ones(m, dtype=bool)]
    order = s_shuffle.permutation(N)
    rnd_out = Round(spec, status, reports[order], groups[order], is_fake[order], labels, meta)
    est = estimate(rnd_out.reports, rnd_out.groups, mech, tf).normalized(tf)
    labels.update(standard_mean=est.mean, standard_variance=est.variance)
    if attacked:
        meta['realized'] = {'target_error_mean': est.mean - mean_t, 'target_error_variance': est.variance - var_t}
    meta['checksum'] = rnd_out.checksum()
    return rnd_out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
