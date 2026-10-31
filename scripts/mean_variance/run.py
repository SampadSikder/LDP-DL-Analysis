#!/usr/bin/env python
"""CLI for the SR/PM utility-recovery benchmark.

  python scripts/mean_variance/run.py audit-data
  python scripts/mean_variance/run.py generate --config configs/mean_variance/smoke.yaml
  python scripts/mean_variance/run.py evaluate --config configs/mean_variance/smoke.yaml
  python scripts/mean_variance/run.py report   --config configs/mean_variance/smoke.yaml
  python scripts/mean_variance/run.py smoke     # all of the above on the smoke config

Generation, training and evaluation are resumable: completed rounds and
completed (training set, method, seed) units are skipped.
"""

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

from attacker_detector.mean_variance import plots
from attacker_detector.mean_variance.evaluate import (aggregate, coverage_table, evaluate_frozen, load_results,
                                                     run_unit)
from attacker_detector.mean_variance.experiment import enumerate_specs, load_config, make_banks
from attacker_detector.mean_variance.sources import SOURCES, audit
from attacker_detector.mean_variance.store import RoundStore
from attacker_detector.mean_variance.validation import (attack_validation, budget_curves, feasibility_maps,
                                                       kmeans_verification)

RESULTS_ROOT = 'results/mean_variance'
_ROOT_OVERRIDE = None          # --results-root: every output of this process goes under it


def results_root(cfg=None):
    return _ROOT_OVERRIDE or (cfg or {}).get('results_root', RESULTS_ROOT)


def out_dir(cfg):
    return os.path.join(results_root(cfg), cfg['experiment'])


def cmd_audit(args):
    rows = []
    for s in args.sources or list(SOURCES):
        try:
            r = audit(s)
        except FileNotFoundError as e:
            print(f"{s}: MISSING ({e})")
            continue
        prov = r.pop('provenance')
        rows.append(r)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))
        os.makedirs(results_root(), exist_ok=True)
        with open(os.path.join(results_root(), f'provenance_{s}.json'), 'w') as f:
            json.dump({**prov, 'stats': r}, f, indent=1)


def cmd_generate(cfg, splits):
    banks = make_banks(cfg)
    store = RoundStore(os.path.join(out_dir(cfg), 'rounds'))
    for split in splits:
        specs = enumerate_specs(cfg, split, banks)
        t0 = time.perf_counter()
        n = store.generate(specs, banks)
        print(f"[{split}] specs {len(specs):,}; generated {n:,} new in {time.perf_counter() - t0:.1f}s")
    cov = coverage_table(store.manifest())
    cov.to_csv(os.path.join(out_dir(cfg), 'coverage.csv'), index=False)
    print(cov.groupby(['split', 'status']).size().to_string())


def training_sets(cfg):
    sources = cfg['sources']
    sets = [('single-source', (s,), tuple(sources)) for s in sources]
    if cfg.get('multi_source', True) and len(sources) >= 3:
        for test in sources:
            train = tuple(s for s in sources if s != test)
            if len(train) == 2:
                sets.append(('multi-source', train, (test,)))
    return sets


def cmd_evaluate(cfg, methods=None, shard=(0, 1)):
    store = RoundStore(os.path.join(out_dir(cfg), 'rounds'))
    manifest = store.manifest()
    banks = make_banks(cfg)
    allowed = {s.round_id for split in ('train', 'val', 'test') for s in enumerate_specs(cfg, split, banks)}
    missing = allowed - set(manifest)
    if missing:
        raise SystemExit(f"{len(missing):,} configured rounds not generated yet: run `generate` first")
    methods = methods or cfg['methods']
    seeds = cfg.get('seeds', [0])
    units = [(protocol, eps, tag, train_src, test_src, m, seed)
             for protocol, eps in itertools.product(cfg['protocols'], cfg['epsilon'])
             for tag, train_src, test_src in training_sets(cfg)
             for m in methods
             for seed in (seeds if m not in ('standard', 'oracle_removal', 'constant', 'kmeans') else [0])]
    i, n = shard
    print(f"{len(units):,} units; this process runs shard {i}/{n}: {len(units[i::n]):,}")
    for protocol, eps, tag, train_src, test_src, m, seed in units[i::n]:
        t0 = time.perf_counter()
        status = run_unit(store, manifest, protocol, float(eps), list(train_src), list(test_src),
                          m, seed, cfg, out_dir(cfg), tag, allowed)
        print(f"{protocol} eps={eps} {tag} train={'+'.join(train_src)} {m} s{seed}: {status} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)


def cmd_report(cfg):
    od = out_dir(cfg)
    res = load_results(od)
    if res.empty:
        print("no results")
        return
    budget = 'budget_mode' in res and (res['budget_mode'] == 'mmin_multiple').any()
    # budget runs have beta = 0 everywhere (m = factor * m_min): split cells by factor instead
    agg = aggregate(res, by=('partition', 'train_sources', 'source', 'protocol', 'epsilon', 'beta', 'attack', 'method')
                    + (('budget_factor',) if budget else ()))
    agg.to_csv(os.path.join(od, 'aggregate.csv'), index=False)
    res.groupby(['method', 'kind', 'partition']).agg(rounds=('round_id', 'nunique'),
                                                     mse_mean=('sq_err_mean', 'mean'),
                                                     mse_var=('sq_err_var', 'mean')).to_csv(
        os.path.join(od, 'summary_by_method.csv'))
    fig_dir = os.path.join(od, 'figures')
    inputs = sorted(os.listdir(os.path.join(od, 'per_round')))
    for protocol in cfg['protocols']:
        for src in cfg['sources']:
            within = agg[(agg.partition == 'single-source') & (agg.train_sources == src) & (agg.source == src)]
            plots.sweep(within, 'beta', {'protocol': protocol}, os.path.join(fig_dir, f'beta_{protocol}_{src}'),
                        f"{protocol}, within-source {src}: utility vs beta", inputs)
    for protocol in cfg['protocols']:
        within = agg[(agg.partition == 'single-source') & (agg.train_sources == agg.source)]
        for val in ('ag_mean', 'ag_var', 'mse_mean', 'mse_var'):
            if within['epsilon'].nunique() > 1 or within['beta'].nunique() > 1:
                plots.heatmaps(within, val, {'protocol': protocol}, os.path.join(fig_dir, f'heat_{val}_{protocol}'),
                               f"{protocol}, within-source: {val} by epsilon x beta", inputs)
        for val in ('mse_mean', 'mse_var', 'ag_mean', 'ag_var'):
            plots.transfer_matrix(agg[agg.attack != 'none'], val, {'protocol': protocol},
                                  os.path.join(fig_dir, f'transfer_{val}_{protocol}'),
                                  f"{protocol}: {val}, within vs cross vs multi-source", inputs)
    plots.runtime_bars(res, os.path.join(fig_dir, 'runtime'), "runtime and memory per method", inputs)
    if budget:
        b = res[res['budget_mode'] == 'mmin_multiple']
        for protocol in cfg['protocols']:
            within = b[(b.protocol == protocol) & (b.train_sources == b.source)]
            plots.utility_vs_budget(within, os.path.join(fig_dir, f'utility_vs_budget_{protocol}'),
                                    f"{protocol}, within-source: utility vs m/m_min", inputs=inputs)
        man = pd.DataFrame(RoundStore(os.path.join(od, 'rounds')).manifest().values())
        man = man[man['budget_mode'] == 'mmin_multiple']
        plots.construction_by_factor(man, os.path.join(fig_dir, 'construction_by_factor'),
                                     "outcome per budget rung (theory, construction)", ['manifest.jsonl'])
    print(f"aggregate: {os.path.join(od, 'aggregate.csv')}  figures: {fig_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['audit-data', 'generate', 'evaluate', 'report', 'smoke',
                                        'attack-validate', 'feasibility-maps', 'evaluate-frozen',
                                        'verify-kmeans', 'budget-curves'])
    ap.add_argument('--rates', nargs='+', type=float, help="verify-kmeans sampling rates (default config / 0.1..0.9)")
    ap.add_argument('--checkpoints', help="evaluate-frozen: <results>/<train experiment>/checkpoints")
    ap.add_argument('--epsilon', type=float, default=1.0)
    ap.add_argument('--shard', default='0/1', help="i/n: run every n-th evaluation unit starting at i")
    ap.add_argument('--config', default='configs/mean_variance/smoke.yaml')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--methods', nargs='+')
    ap.add_argument('--sources', nargs='+')
    ap.add_argument('--results-root', help=f"folder for every output (default {RESULTS_ROOT}; "
                                           "overrides a config's results_root)")
    args = ap.parse_args()
    global _ROOT_OVERRIDE
    _ROOT_OVERRIDE = args.results_root
    if args.command == 'audit-data':
        return cmd_audit(args)
    if args.command == 'feasibility-maps':
        out = feasibility_maps(args.sources or list(SOURCES), args.epsilon,
                               os.path.join(results_root(), 'feasibility'))
        print(out.groupby(['source', 'attack'])['beta_min'].describe().to_string())
        return
    if args.command == 'evaluate-frozen':
        cfg = load_config(args.config)
        store = RoundStore(os.path.join(out_dir(cfg), 'rounds'))
        banks = make_banks(cfg)
        allowed = {s.round_id for s in enumerate_specs(cfg, 'test', banks)}
        n = evaluate_frozen(args.checkpoints, store, store.manifest(), cfg, out_dir(cfg), allowed, args.methods)
        print(f"frozen evaluations written: {n}")
        return
    if args.command == 'budget-curves':
        out_d = os.path.join(results_root(), 'budget_curves')
        df = budget_curves(args.sources or list(SOURCES), args.epsilon, out_d)
        plots.budget_curves_plot(df, os.path.join(out_d, f'beta_min_curves_eps{args.epsilon:g}'),
                                 f'minimum attacker fraction, eps {args.epsilon:g}', [f'budget_curves_eps{args.epsilon:g}.csv'])
        print(df.groupby(['source', 'attack', 'knowledge'])['beta_min'].describe().round(3).to_string())
        return
    if args.command == 'verify-kmeans':
        cfg = load_config(args.config)
        rates = args.rates or cfg.get('kmeans_rates') or [round(0.1 * i, 1) for i in range(1, 10)]
        agg = kmeans_verification(cfg, out_dir(cfg), rates, cfg.get('kmeans_subsets', 5000),
                                  n_jobs=cfg.get('n_jobs', 12))
        print(agg.to_string(max_rows=200))
        return
    if args.command == 'attack-validate':
        cfg = load_config(args.config)
        agg = attack_validation(cfg, out_dir(cfg))
        print(agg.to_string(max_rows=60))
        return
    cfg = load_config(args.config)
    if args.command in ('generate', 'smoke'):
        cmd_generate(cfg, args.splits)
    if args.command in ('evaluate', 'smoke'):
        i, n = (int(v) for v in args.shard.split('/'))
        cmd_evaluate(cfg, args.methods, (i, n))
    if args.command in ('report', 'smoke'):
        cmd_report(cfg)


if __name__ == '__main__':
    main()
