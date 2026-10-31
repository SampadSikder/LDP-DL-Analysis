"""Benchmark data sources: loaders, declared domains, provenance.

Files live in datasets/mean_variance/<source>/

"""

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'mean_variance')
CACHE_ROOT = os.path.join(DATA_ROOT, '.cache')


@dataclass(frozen=True)
class SourceSpec:
    name: str
    folder: str
    patterns: tuple
    column: str
    filter: str
    domain: tuple                 # declared (a, b)
    domain_class: str             # paper-settings classification
    covariates: tuple
    table3: dict = field(default_factory=dict)
    printed_targets: dict = field(default_factory=dict)
    n_e_paper: float = 0.0


SOURCES = {
    'taxi': SourceSpec(
        'taxi', 'taxi', ('green_tripdata_2020-12.parquet', 'green_tripdata_2020-12.csv'),
        'DOLocationID', 'none', (1.0, 265.0), 'inferred-from-figure',
        ('weekday', 'hour_bin', 'payment_type', 'vendor'),
        table3={'n': 83130, 'mean_n': -0.022, 'var_n': 0.34, 'S1_n': -2194, 'S2_n': 28362},
        printed_targets={'mean': (0.06, -0.022, -0.06), 'variance': (0.33, 0.34, 0.40)},
        n_e_paper=80_000),
    'income': SourceSpec(
        'income', 'income', ('psam_pusa.csv', 'psam_pusb.csv'),
        'PINCP', 'PINCP > 0 (no ADJINC)', (1.0, 1_629_000.0), 'inferred-from-figure',
        ('state', 'age_bin', 'sex'),
        table3={'n': 2390203, 'mean_n': -0.93, 'var_n': 0.007, 'S1_n': -2239154, 'S2_n': 2115289},
        printed_targets={'mean': (-0.92, -0.93, -0.94), 'variance': (0.005, 0.007, 0.009)},
        n_e_paper=2_400_000),
    'retirement': SourceSpec(
        'retirement', 'retirement', ('employee-compensation.csv',),
        'Retirement', 'Year Type == Fiscal and Year in {2019, 2020}', (-2000.0, 125_000.0),
        'proposed-extension', ('org_group', 'year'),
        table3={'n': 97220, 'mean_n': -0.87, 'var_n': 0.025, 'S1_n': -85000, 'S2_n': 76752},
        printed_targets={'mean': (-0.86, -0.87, -0.88), 'variance': (0.020, 0.025, 0.030)},
        n_e_paper=100_000),
}


@dataclass
class SourceData:
    spec: SourceSpec
    values: np.ndarray                 
    covariates: dict                       
    provenance: dict

    @property
    def name(self) -> str:
        return self.spec.name

    def __len__(self) -> int:
        return len(self.values)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _files(spec: SourceSpec):
    folder = os.path.join(DATA_ROOT, spec.folder)
    if spec.name == 'taxi':          # either format, parquet preferred
        for p in spec.patterns:
            hits = glob.glob(os.path.join(folder, p))
            if hits:
                return hits[:1]
        hits = []
    else:
        hits = [h for p in spec.patterns for h in sorted(glob.glob(os.path.join(folder, p)))]
    if not hits or (spec.name == 'income' and len(hits) < 2):
        raise FileNotFoundError(
            f"source {spec.name!r}: expected {spec.patterns} in {folder}. "
            f"See docs/mean_variance/data_dictionary.md for download instructions.")
    return hits


def _codes(series) -> np.ndarray:
    return pd.factorize(series, sort=True)[0].astype(np.int32)


def _read_taxi(path):
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, parse_dates=['lpep_pickup_datetime'])
    t = pd.to_datetime(df['lpep_pickup_datetime'])
    cov = {'weekday': _codes(t.dt.weekday), 'hour_bin': _codes(t.dt.hour // 6),
           'payment_type': _codes(df['payment_type'].fillna(0)), 'vendor': _codes(df['VendorID'])}
    return df['DOLocationID'].to_numpy(np.float64), cov


def _read_income(paths):
    parts = [pd.read_csv(p, usecols=['PINCP', 'ST', 'AGEP', 'SEX']) for p in paths]
    df = pd.concat(parts, ignore_index=True)
    df = df[df['PINCP'] > 0]
    cov = {'state': _codes(df['ST']), 'age_bin': _codes(pd.cut(df['AGEP'], [0, 24, 34, 44, 54, 64, 200])),
           'sex': _codes(df['SEX'])}
    return df['PINCP'].to_numpy(np.float64), cov


def _read_retirement(path):
    df = pd.read_csv(path, usecols=['Year Type', 'Year', 'Organization Group', 'Retirement'], low_memory=False)
    df = df[(df['Year Type'] == 'Fiscal') & df['Year'].isin([2019, 2020])]
    cov = {'org_group': _codes(df['Organization Group'].fillna('NA')), 'year': _codes(df['Year'])}
    return df['Retirement'].to_numpy(np.float64), cov


def file_provenance(paths) -> list:
    """Checksums are cached against (size, mtime) so big files are hashed once."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    index_path = os.path.join(CACHE_ROOT, 'checksums.json')
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
    out, changed = [], False
    for p in paths:
        st = os.stat(p)
        key = f"{os.path.relpath(p, PROJECT_ROOT)}|{st.st_size}|{int(st.st_mtime)}"
        if key not in index:
            index[key] = _sha256(p)
            changed = True
        out.append({'file': os.path.relpath(p, PROJECT_ROOT), 'bytes': st.st_size, 'sha256': index[key]})
    if changed:
        # atomic: parallel processes (evaluate shards, k-means pools) read this file concurrently
        tmp = f"{index_path}.{os.getpid()}.tmp"
        with open(tmp, 'w') as f:
            json.dump(index, f, indent=1)
        os.replace(tmp, index_path)
    return out


def load_source(name: str) -> SourceData:
    if name not in SOURCES:
        raise ValueError(f"unknown source {name!r}; expected one of {sorted(SOURCES)}")
    spec = SOURCES[name]
    paths = _files(spec)
    files = file_provenance(paths)
    digest = hashlib.sha256('|'.join(f['sha256'] for f in files).encode()).hexdigest()[:16]
    cache = os.path.join(CACHE_ROOT, f"{name}_{digest}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        values = z['values']
        cov = {k[4:]: z[k] for k in z.files if k.startswith('cov_')}
    else:
        if name == 'taxi':
            values, cov = _read_taxi(paths[0])
        elif name == 'income':
            values, cov = _read_income(paths)
        else:
            values, cov = _read_retirement(paths[0])
        tmp = f"{cache[:-4]}.{os.getpid()}.tmp.npz"          # atomic, as in file_provenance
        np.savez(tmp, values=values, **{f'cov_{k}': v for k, v in cov.items()})
        os.replace(tmp, cache)
    a, b = spec.domain
    if values.min() < a or values.max() > b:
        raise ValueError(f"{name}: data range [{values.min()}, {values.max()}] exceeds declared domain {spec.domain}")
    prov = {'source': name, 'files': files, 'column': spec.column, 'filter': spec.filter,
            'domain': list(spec.domain), 'domain_class': spec.domain_class, 'n_records': int(len(values)),
            'data_min': float(values.min()), 'data_max': float(values.max())}
    return SourceData(spec, values, cov, prov)


def audit(name: str) -> dict:
    """Source statistics on the declared domain next to the paper's Table 3."""
    from .transforms import Transform
    d = load_source(name)
    tf = Transform(*d.spec.domain)
    xt = tf.to_g1(d.values)
    row = {'source': name, 'n': len(d), 'raw_mean': float(d.values.mean()), 'raw_var': float(d.values.var()),
           'mean_n': float(xt.mean()), 'var_n': float(xt.var()), 'S1_n': float(xt.sum()),
           'S2_n': float(np.sum(xt * xt))}
    row.update({f'paper_{k}': v for k, v in d.spec.table3.items()})
    row['provenance'] = d.provenance
    return row
