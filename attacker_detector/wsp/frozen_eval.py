"""Test evaluation of Stage-1 and edited models, written in the formats the existing scripts read.

Mean/variance: <mv.out>/eval/per_round/<key>.csv, rows from evaluate._row (same columns as breadth), with the
method renamed per variant so rows never mix:
    stage1 -> <method>        wsp -> <method>_wsp        finetune -> <method>_ft        random -> <method>_rand
    (budget other than 25 adds _b<budget>). Then scripts/mean_variance/attack_success.py runs unchanged on
    <mv.out>/eval and the WSP figures read its attack_success.csv.gz.
Frequency oracle: results/<out_root>/eval/<variant>_s<seed>/{in_dist,cross}/<name>/ with test_results.csv,
sensitivity_test_results.csv and sensitivity_test_balanced.csv (main.py's formats), so
scripts/plots/loader.load_oat('<out_root>/eval/<variant>_s<seed>') reads them.
"""

import os
import time

import numpy as np
import pandas as pd
import torch

from .jobs import is_done, rel

SUFFIX = {'stage1': '', 'wsp': '_wsp', 'finetune': '_ft', 'random': '_rand'}


def variant_name(method, mode, budget):
    if mode == 'stage1':
        return method
    return method + SUFFIX[mode] + ('' if int(budget) == 25 else f'_b{budget}')


def freq_variant(mode, budget):
    return 'stage1' if mode == 'stage1' else SUFFIX[mode][1:] + ('' if int(budget) == 25 else f'_b{budget}')


def _units(jobs, kind):
    units = {}
    for j in jobs:
        if j['kind'] != kind:
            continue
        k = j['unit'] if kind == 'mv' else (j['run'], j['seed'])
        u = units.setdefault(k, {'job': j, 'variants': [('stage1', 25, None)]})
        if is_done(j):
            u['variants'].append((j['mode'], j['budget'], os.path.join(j['run_dir'], 'checkpoint')))
    return units


def eval_mv(sweep_cfg, jobs, shard=(0, 1), device='cpu'):
    from ..mean_variance.evaluate import _row
    from ..mean_variance.methods.checkpoint import load_checkpoint
    from .runner import mv_context
    ctx = mv_context(sweep_cfg)
    out = os.path.join(rel(sweep_cfg['mv']['out']), 'eval', 'per_round')
    os.makedirs(out, exist_ok=True)
    units = sorted(_units(jobs, 'mv').items())
    i, n = shard
    done = 0
    for unit, u in units[i::n]:
        j = u['job']
        for mode, budget, ck in u['variants']:
            name = variant_name(j['method'], mode, budget)
            key = f"single-source__{j['protocol']}_eps{j['eps']:g}__{'+'.join(j['train_sources'])}__{name}__s{j['seed']}"
            path = os.path.join(out, key + '.csv')
            if os.path.exists(path):
                continue
            t0 = time.perf_counter()
            m = load_checkpoint(ck or ctx.checkpoint(unit), device)
            m.name = name
            rows = []
            for dest in ctx.cfg['sources']:
                test = ctx.rounds('test', j['protocol'], j['eps'], [dest])
                outs = m.predict_many([r.observation for r in test])
                rows += [_row(m, j['seed'], 'single-source', r, ctx.manifest[r.round_id], o) for r, o in zip(test, outs)]
            df = pd.DataFrame(rows)
            df['train_sources'] = '+'.join(j['train_sources'])
            df['wsp_mode'], df['wsp_budget'] = mode, budget
            df.to_csv(path + '.tmp', index=False)
            os.replace(path + '.tmp', path)
            done += 1
            print(f'  {key}: {len(df)} rows ({time.perf_counter() - t0:.0f}s)', flush=True)
    return done


def _test_metrics(model, X, y, device, batch=8192):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            probs.append(torch.sigmoid(model(torch.from_numpy(np.asarray(X[i:i + batch], np.float32)).to(device)))
                         .cpu().numpy().ravel())
    p = np.concatenate(probs)
    pred = (p > 0.5).astype(int)
    return {'Accuracy': accuracy_score(y, pred), 'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred, zero_division=0), 'F1_Score': f1_score(y, pred, zero_division=0)}


def eval_freq(sweep_cfg, jobs, device='cpu', only_stage1=False, sensitivity=True):
    """Test metrics (+ sensitivity CSVs) of every variant, one dataset load per (run, seed)."""
    from ..analysis import run_sensitivity_analysis
    from .adapters import build_freq_model, build_freq_split, load_freq_state
    fc = sweep_cfg['freq']
    base = os.path.join(rel('results'), fc['out_root'], 'eval')
    rows = []
    for (run, seed), u in sorted(_units(jobs, 'freq').items()):
        spec = u['job']['spec']
        todo = []
        for mode, budget, ck in u['variants']:
            if only_stage1 and mode != 'stage1':
                continue
            d = os.path.join(base, f'{freq_variant(mode, budget)}_s{seed}', run)
            if not os.path.exists(os.path.join(d, 'sensitivity_test_results.csv' if sensitivity else 'test_results.csv')):
                todo.append((mode, budget, os.path.join(ck, 'model.pt') if ck else spec['model_path'], d))
        if not todo:
            continue
        t0 = time.perf_counter()
        ds, split = build_freq_split(spec)
        X, y = split['X_test'], split['y_test']
        label = f"Test ({spec['test_dataset']})" if spec['training_method'] == 'cross' else 'Test'
        for mode, budget, path, d in todo:
            os.makedirs(d, exist_ok=True)
            model = build_freq_model(spec, X.shape[1], load_freq_state(path), device)
            m = _test_metrics(model, X, y, device)
            pd.DataFrame([{'label': label, 'n_samples': len(y), **m}]).to_csv(os.path.join(d, 'test_results.csv'),
                                                                             index=False)
            rows.append({'run': run, 'seed': seed, 'variant': freq_variant(mode, budget), **m})
            if sensitivity:
                defaults = ds.design['defaults'] if ds.is_oat else None
                for idx, fname in ((split['sens_indices'], 'sensitivity_test_results.csv'),
                                   (split['sens_balanced_indices'], 'sensitivity_test_balanced.csv')):
                    df = run_sensitivity_analysis(model, ds.features[idx], ds.labels[idx], device,
                                                  config_array=ds.config[idx], batch_size=4096, oat_defaults=defaults)
                    df.to_csv(os.path.join(d, fname), index=False)
            print(f'  {run} s{seed} {freq_variant(mode, budget)}: F1 {m["F1_Score"]:.4f}', flush=True)
        del ds, split
        print(f'  {run} s{seed}: {len(todo)} variant(s) in {time.perf_counter() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows)
