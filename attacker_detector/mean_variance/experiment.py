"""Experiment configuration -> round specs; parent-population banks per split.
"""

import itertools
import os
from dataclasses import dataclass

import numpy as np
import yaml

from .populations import SPLITS, build_population, record_partition, sample_recipe, stable_seed
from .scenarios import RoundSpec, target_label
from .sources import SOURCES, load_source

PM_MAIN_SPREADS = (0.05, 0.10, 0.25, 0.50, 1.0)


def load_config(path: str) -> dict:
    """YAML config; `extends: other.yaml` (relative to this file) loads that first, then overrides
    its top-level keys."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop('extends', None)
    if base:
        cfg = {**load_config(os.path.join(os.path.dirname(os.path.abspath(path)), base)), **cfg}
    cfg.setdefault('experiment', 'unnamed')
    return cfg


class PopulationBank:
    """Lazily builds and caches parent populations of one source."""

    def __init__(self, source: str, pool_size, base_seed: int = 0, plain: bool = False):
        self.data = load_source(source)
        self.partition = record_partition(self.data)
        self.pool_size = len(self.data) if pool_size == 'full' else int(pool_size)
        self.base_seed = base_seed
        self.plain = plain
        self._recipes, self._values = {}, {}

    def recipe(self, split: str, index: int):
        key = (split, index)
        if key not in self._recipes:
            self._recipes[key] = sample_recipe(self.data, self.partition, split, index, self.pool_size,
                                               self.base_seed, plain=self.plain)
        return self._recipes[key]

    def values(self, split: str, index: int) -> np.ndarray:
        key = (split, index)
        if key not in self._values:
            self._values[key] = build_population(self.data, self.partition, self.recipe(split, index))[0]
        return self._values[key]


def spread_for(protocol: str, attack: str, rep: int, cfg: dict):
    """OPA-PM spread ratio for replicate `rep`: fixed if configured, else cycled over the main set."""
    if protocol != 'PM' or attack != 'OPA':
        return None
    fixed = cfg.get('pm_spread_ratio')
    if fixed is not None:
        return float(fixed)
    ratios = cfg.get('pm_spread_ratios', PM_MAIN_SPREADS)
    return float(ratios[stable_seed('spread', rep) % len(ratios)])


def enumerate_specs(cfg: dict, split: str, banks: dict):
    """All RoundSpecs of one split. Clean rounds are one cell per (protocol, source, epsilon)."""
    rounds = cfg['rounds'][split]
    sizes = cfg['sizes']
    targets = [(mi, vi) for mi, vi in itertools.product(range(3), range(3))]
    if cfg.get('targets'):
        targets = [(int(t[1]) - 1, int(t[3]) - 1) for t in cfg['targets']]
    common = dict(target_mode=cfg.get('target_mode', 'offset'), square_mode=cfg.get('square_mode', 'normalized'),
                  knowledge_regime=cfg.get('knowledge_regime', 'compromised'),
                  knowledge_h=cfg.get('knowledge_h', 1000), n_e_rule=cfg.get('n_e_rule', 'sig2'),
                  ipa_solver=cfg.get('ipa_solver', 'closed_form'), sum_rule=cfg.get('sum_rule', 'paper'),
                  size_mode=cfg.get('size_mode', 'fixed_total'),
                  dS1_n=float(cfg.get('dS1_n', 0.0)), dS2_n=float(cfg.get('dS2_n', 0.0)), dn=float(cfg.get('dn', 0.0)))
    if cfg.get('sweep'):
        return _sweep_specs(cfg, split, banks, common)
    if cfg.get('budget_sweep'):
        return _budget_specs(cfg, split, banks, common)
    specs = []
    for protocol, source, eps in itertools.product(cfg['protocols'], cfg['sources'], cfg['epsilon']):
        bank = banks[source]
        for rep in range(rounds):
            pop_id = bank.recipe(split, rep).population_id
            size = resolve_size(sizes[rep % len(sizes)], bank)
            base = dict(experiment=cfg['experiment'], split=split, source=source, population_id=pop_id,
                        domain=tuple(SOURCES[source].domain), protocol=protocol, epsilon=float(eps),
                        total_size=size, rep=rep, **common)
            for beta in cfg['beta']:
                if beta == 0:
                    specs.append(RoundSpec(beta=0.0, attack='none', target='clean', **base))
                    continue
                for attack, (mi, vi) in itertools.product(cfg['attacks'], targets):
                    specs.append(RoundSpec(beta=float(beta), attack=attack, target=target_label(mi, vi),
                                           spread_ratio=spread_for(protocol, attack, rep, cfg), **base))
    return specs


SWEEP_PARAMS = ('target_mean', 'target_variance', 'dS1_n', 'dS2_n', 'dn', 'spread_ratio')


def _sweep_specs(cfg: dict, split: str, banks: dict, common: dict):
    """One-factor sweep: a single parameter varies; everything else at the reference default.

    cfg['sweep'] = {'param': one of SWEEP_PARAMS, 'values': [...]}. Target sweeps use explicit
    benchmark-scale targets; the other moment is taken from cfg['sweep']['other'] (a value or
    'preserve'). Knowledge-error sweeps shift the attacker's benchmark-scale sums / count.
    """
    sw = cfg['sweep']
    param = sw['param']
    if param not in SWEEP_PARAMS:
        raise ValueError(f"sweep param must be one of {SWEEP_PARAMS}")
    others = sw.get('other', ['preserve'])
    specs = []
    sizes = cfg['sizes']
    for protocol, source, eps, beta, attack in itertools.product(cfg['protocols'], cfg['sources'], cfg['epsilon'],
                                                                 cfg['beta'], cfg['attacks']):
        bank = banks[source]
        for rep in range(cfg['rounds'][split]):
            base = dict(experiment=cfg['experiment'], split=split, source=source,
                        population_id=bank.recipe(split, rep).population_id, domain=tuple(SOURCES[source].domain),
                        protocol=protocol, epsilon=float(eps), total_size=resolve_size(sizes[rep % len(sizes)], bank),
                        rep=rep,
                        beta=float(beta), attack=attack, **common)
            for value, other in itertools.product(sw['values'], others):
                kw = dict(base)
                other_v = None if other == 'preserve' else float(other)
                if param == 'target_mean':
                    kw.update(target='explicit', target_mean_value=float(value), target_var_value=other_v)
                elif param == 'target_variance':
                    kw.update(target='explicit', target_var_value=float(value), target_mean_value=other_v)
                else:
                    kw.update(target=sw.get('target', 'm1v1'))
                    kw[param] = float(value)
                if param != 'spread_ratio':
                    kw['spread_ratio'] = spread_for(protocol, attack, rep, cfg)
                elif protocol != 'PM' or attack != 'OPA':
                    continue
                specs.append(RoundSpec(**kw))
    return specs


def resolve_size(size, bank) -> int:
    return len(bank.data) if size == 'full' else int(size)


def _budget_specs(cfg: dict, split: str, banks: dict, common: dict):
    """Budgets around the minimum (proposed extension; kept apart from the paper's settings).

    cfg['budget_sweep'] = {'factors': [1.0, 1.1, 1.25, 1.5, 2.0], 'below_budget': 'skip'}.
    Each round gets m = ceil(factor * m_min), with m_min computed in generate_round from
    the attacker's estimates for that round's target, and the theory check is redone at
    the rounded budget. Needs size_mode 'fixed_genuine' (the genuine population is fixed).
    Factors < 1 with below_budget 'saturate' form the separately labelled below-budget
    evaluation. The spec's beta is unused (0) and the realized beta is recorded instead.
    """
    if common['size_mode'] != 'fixed_genuine':
        raise ValueError("budget_sweep needs size_mode: fixed_genuine")
    bs = cfg['budget_sweep']
    targets = [(int(t[1]) - 1, int(t[3]) - 1) for t in cfg['targets']] if cfg.get('targets') else \
        [(mi, vi) for mi, vi in itertools.product(range(3), range(3)) if (mi, vi) != (1, 1)]
    specs = []
    for protocol, source, eps, attack in itertools.product(cfg['protocols'], cfg['sources'], cfg['epsilon'],
                                                           cfg['attacks']):
        bank = banks[source]
        for rep in range(cfg['rounds'][split]):
            base = dict(experiment=cfg['experiment'], split=split, source=source,
                        population_id=bank.recipe(split, rep).population_id, domain=tuple(SOURCES[source].domain),
                        protocol=protocol, epsilon=float(eps),
                        total_size=resolve_size(cfg['sizes'][rep % len(cfg['sizes'])], bank), rep=rep, beta=0.0,
                        attack=attack, spread_ratio=spread_for(protocol, attack, rep, cfg), **common)
            for (mi, vi), f in itertools.product(targets, bs['factors']):
                specs.append(RoundSpec(target=target_label(mi, vi), budget_mode='mmin_multiple', budget_factor=float(f),
                                       below_budget=bs.get('below_budget', 'skip'), **base))
    return specs


def make_banks(cfg: dict) -> dict:
    pool = cfg.get('pool_size') or ('full' if 'full' in cfg['sizes'] else max(cfg['sizes']))
    return {s: PopulationBank(s, pool, cfg.get('population_seed', 0), cfg.get('plain_populations', False))
            for s in cfg['sources']}
