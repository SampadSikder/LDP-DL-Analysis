#!/usr/bin/env python
"""Training cost of every WSP run: LLM tokens and dollars (Stage 2) next to gradient cost (Stage 1, fine-tune).

  python scripts/wsp/cost_report.py            # results/wsp/cost_summary.csv + printed totals
  python scripts/wsp/cost_report.py --pilot    # tokens per pilot run -> projected sweep total vs the cap

Columns: job, kind, target, mode, budget, stop, accepted edits, LLM calls, input / cached / output / reasoning /
total tokens, usd, LLM wait s, probe and gate evaluations, Stage-2 forward FLOPs, fine-tune steps / FLOPs,
Stage-2 GPU s and wall s, Stage-1 epochs / optimizer steps / training FLOPs / wall s, val loss before / after.
"""
import argparse
import json
import os
import sys

from _common import load_env

import pandas as pd

from attacker_detector.wsp.cost import Ledger
from attacker_detector.wsp.jobs import STAGE2, WSP_HOME, all_jobs, is_done, load_yaml, select


def target_of(job):
    if job['kind'] == 'freq':
        return 'freq_ft'
    return {'ft_removal': 'mv_ft', 'deepsets_recovery': 'mv_ds', 'settransformer_recovery': 'mv_st'}[job['method']]


def collect(jobs):
    rows = []
    for j in jobs:
        p = os.path.join(j['run_dir'], 'cost.json')
        if not os.path.exists(p):
            continue
        c, w = json.load(open(p)), json.load(open(os.path.join(j['run_dir'], 'wsp.json')))
        llm, h, s1 = c['stage2']['llm'], c['stage2']['harness'], c['stage1']
        vb, vf = w['val_stage1'], w['val_final']
        key = 'f1' if 'f1' in vb else 'loss'
        rows.append({'job': j['id'], 'kind': j['kind'], 'target': target_of(j), 'mode': j['mode'],
                     'budget': j['budget'], 'seed': j['seed'], 'stop': w['stop'],
                     'edits_submitted': w['edits_submitted'], 'edits_accepted': w['edits_accepted'],
                     'llm_calls': llm['calls'], 'input_tokens': llm['input'], 'cached_tokens': llm['cached'],
                     'output_tokens': llm['output'], 'reasoning_tokens': llm['reasoning'], 'total_tokens': llm['total'],
                     'usd': llm['usd'], 'llm_wait_s': llm['wait_s'],
                     'agent': llm.get('agent', 'api' if j['mode'] == 'wsp' else None), 'model': llm.get('model'),
                     'effort': llm.get('effort'), 'weekly_pct_used': (llm.get('weekly_pct') or {}).get('used'),
                     'probe_evals': h['probe_evals'],
                     'gate_evals': h['gate_evals'], 'stage2_forward_flops': h['forward_flops'],
                     'finetune_steps': h['train_steps'], 'finetune_flops': h['train_flops'],
                     'stage2_compute_s': h.get('compute_s', h.get('gpu_s')), 'sandbox_s': h['sandbox_s'], 'stage2_wall_s': c['elapsed_s'],
                     'stage1_epochs': s1.get('epochs_run'), 'stage1_steps': s1.get('optimizer_steps'),
                     'stage1_train_flops': s1.get('train_flops'), 'stage1_wall_s': s1.get('fit_wall_s'),
                     'val_metric': key, 'val_stage1': vb.get(key), 'val_final': vf.get(key),
                     'weights_changed': w['weights_changed_vs_stage1'], 'rel_l2_vs_stage1': w['rel_l2_vs_stage1']})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--set', nargs='+', default=['tier_a'])
    args = ap.parse_args()
    load_env()
    s2 = load_yaml(STAGE2)
    jobs = all_jobs()
    df = collect(list(jobs.values()))
    out = os.path.join(WSP_HOME, 'cost_summary.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    cap = int(os.environ.get('WSP_TOKEN_CAP') or s2['global_token_cap'])
    tok, usd = Ledger().total()
    print(f'ledger: {tok:,} tokens, ${usd:,.2f} (cap {cap:,} = {tok / cap:.1%} used)')
    if df.empty:
        print('no finished runs yet')
        return 0
    pd.set_option('display.width', 200)
    g = df.groupby(['target', 'mode', 'budget']).agg(
        runs=('job', 'count'), accepted=('edits_accepted', 'mean'), tokens=('total_tokens', 'mean'),
        reasoning=('reasoning_tokens', 'mean'), cached=('cached_tokens', 'mean'), usd=('usd', 'mean'),
        wall_min=('stage2_wall_s', lambda s: s.mean() / 60), compute_s=('stage2_compute_s', 'mean'),
        stage1_wall_min=('stage1_wall_s', lambda s: s.mean() / 60))
    print(g.round(1).to_string())
    cx = df[df['agent'] == 'codex']
    if len(cx):
        pairs = sorted({f'{m}/{e}' for m, e in zip(cx['model'].fillna('?'), cx['effort'].fillna('?'))})
        print(f"codex: {len(cx)} runs, {int(cx['total_tokens'].sum()):,} tokens "
              f"({cx['total_tokens'].mean():,.0f}/run), weekly limit used ~{cx['weekly_pct_used'].fillna(0).sum():.1f} "
              f"points, model/effort {pairs}; usd = API list-price equivalent (the ChatGPT plan pays)")
    print(f'written: {out}')
    if args.pilot:
        p = df[(df['mode'] == 'wsp') & df['job'].isin([j['id'] for j in jobs.values() if j.get('pilot')])]
        if p.empty:
            print('no finished pilot runs')
            return 1
        per_probe = (p['total_tokens'] / p['budget']).mean()
        todo = [j for j in select(jobs, args.set, ['wsp']) if not is_done(j)]
        proj = sum(min(per_probe * j['budget'], s2['budgets'][j['budget']]['token_cap']) for j in todo)
        usd_per_tok = (p['usd'].sum() / max(p['total_tokens'].sum(), 1))
        print(f'pilot: {len(p)} run(s), {p["total_tokens"].mean():,.0f} tokens/run '
              f'({per_probe:,.0f} per probe of budget), ${p["usd"].mean():.2f}/run, '
              f'hit cap: {(p["stop"] == "token_cap").sum()}')
        print(f'projected remaining {len(todo)} LLM runs: ~{proj:,.0f} tokens (~${proj * usd_per_tok:,.0f}); '
              f'ledger now {tok:,}; total ~{tok + proj:,.0f} vs cap {cap:,}')
        verdict = 'GO' if tok + proj <= 0.9 * cap else 'REDUCE the grid (configs/wsp/sweep.yaml) or wait for next week'
        print(f'verdict: {verdict}. Also compare your dashboard % before/after the pilot with these token counts.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
