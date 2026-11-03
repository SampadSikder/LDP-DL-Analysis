"""Job list of the WSP sweep (configs/wsp/sweep.yaml). A job = one (unit, mode, probe budget).

ids:  mv/<mode>_b<budget>/<unit key>              unit key as in the Stage-1 checkpoints folder
      freq/<mode>_b<budget>/<run with / -> __>__s<seed>
"""

import itertools
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SWEEP = os.environ.get('WSP_SWEEP') or os.path.join(ROOT, 'configs', 'wsp', 'sweep.yaml')
STAGE2 = os.environ.get('WSP_STAGE2') or os.path.join(ROOT, 'configs', 'wsp', 'stage2.yaml')
WSP_HOME = os.environ.get('WSP_HOME') or os.path.join(ROOT, 'results', 'wsp')        # ledger, logs, cost summary
FIG_DIR = os.environ.get('WSP_FIG_DIR') or os.path.join(ROOT, 'results', 'figures', 'wsp')


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def rel(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def mv_key(protocol, eps, source, method, seed):
    return f"single-source__{protocol}_eps{float(eps):g}__{source}__{method}__s{seed}"


def parse_freq_run(run):
    """'cross/OLH_Server_Fire_test' -> ('cross', 'OLH_Server', 'fire'); 'in_dist/Fire_OUE' -> ('none', 'OUE', 'fire')."""
    kind, name = run.split('/')
    if kind == 'cross':
        body = name[:-len('_test')] if name.endswith('_test') else name
        protocol, dataset = body.rsplit('_', 1)
        return 'cross', protocol, dataset.lower()
    dataset, protocol = name.split('_', 1)
    return 'none', protocol, dataset.lower()


def freq_spec(fc, run, seed):
    method, protocol, dataset = parse_freq_run(run)
    spec = dict(fc['spec'])
    spec.update(run=run, seed=int(seed), protocol=protocol, data=os.path.join(rel(fc['data']), protocol),
                training_method=method)
    if method == 'cross':
        spec.update(train_dataset=[d for d in fc['datasets'] if d != dataset], test_dataset=dataset)
    else:
        spec.update(dataset=dataset)
    if int(seed) == int(fc.get('reuse_seed', -1)):
        spec['stage1_dir'] = os.path.join(ROOT, 'results', fc['root'], run)
    else:
        spec['stage1_dir'] = os.path.join(ROOT, 'results', fc['out_root'], f'stage1_s{seed}', run)
    spec['model_path'] = os.path.join(spec['stage1_dir'], 'model.pt')
    return spec


def _mv_jobs(mc, grid, modes):
    out = []
    for p, eps, src, m, seed, b, mode in itertools.product(grid['protocols'], grid['epsilon'], grid['train_sources'],
                                                          mc['methods'], grid['seeds'], grid['budgets'], modes):
        key = mv_key(p, eps, src, m, seed)
        out.append({'id': f'mv/{mode}_b{b}/{key}', 'kind': 'mv', 'mode': mode, 'budget': int(b), 'unit': key,
                    'method': m, 'protocol': p, 'eps': float(eps), 'train_sources': [src], 'seed': int(seed),
                    'run_dir': os.path.join(rel(mc['out']), 'stage2', f'{mode}_b{b}', key)})
    return out


def _freq_jobs(fc, modes):
    out = []
    for run, seed, b, mode in itertools.product(fc['runs'], fc['seeds'], fc['budgets'], modes):
        flat = run.replace('/', '__')
        out.append({'id': f'freq/{mode}_b{b}/{flat}__s{seed}', 'kind': 'freq', 'mode': mode, 'budget': int(b),
                    'run': run, 'seed': int(seed), 'spec': freq_spec(fc, run, seed),
                    'run_dir': os.path.join(ROOT, 'results', fc['out_root'], 'stage2', f'{mode}_b{b}',
                                            f'{flat}__s{seed}')})
    return out


def all_jobs(sweep=None):
    sw = load_yaml(sweep or SWEEP)
    modes = sw['modes']
    jobs = {}
    for name in ('tier_a', 'budget_curve', 'tier_b'):
        for j in _mv_jobs(sw['mv'], sw['mv'][name], modes):
            j.setdefault('set', 'tier_b' if name == 'tier_b' else 'tier_a')
            jobs.setdefault(j['id'], j)
    for j in _freq_jobs(sw['freq'], modes):
        j['set'] = 'tier_a'
        jobs[j['id']] = j
    pilot = set(sw.get('pilot', []))
    for jid in pilot:
        if jid not in jobs:
            raise ValueError(f'pilot job {jid!r} is not in the sweep grid')
        jobs[jid]['pilot'] = True
    return jobs


def select(jobs, sets=('tier_a',), modes=None, pilot_only=False, kinds=None):
    out = []
    for j in jobs.values():
        if pilot_only and not j.get('pilot'):
            continue
        if not pilot_only and j['set'] not in sets:
            continue
        if modes and j['mode'] not in modes:
            continue
        if kinds and j['kind'] not in kinds:
            continue
        out.append(j)
    return sorted(out, key=lambda j: (not j.get('pilot', False), j['kind'], j['id']))


def is_done(job):
    return os.path.exists(os.path.join(job['run_dir'], 'done.json'))
