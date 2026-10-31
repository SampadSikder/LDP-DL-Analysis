"""Record partitions and parent-population recipes.

"""

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import rankdata

from .sources import SourceData

SPLITS = ('train', 'val', 'test')
PARTITION_SALT = 'mean-variance-partition-v1'


def stable_seed(*parts) -> int:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], 'little') >> 1


def record_partition(data: SourceData, fractions=(0.6, 0.2, 0.2)) -> np.ndarray:
    """Split code per record (0 train, 1 val, 2 test); fixed by the salt and the source name."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    u = np.random.default_rng(zlib.crc32(f"{PARTITION_SALT}|{data.name}".encode())).random(len(data))
    return np.searchsorted(np.cumsum(fractions)[:-1], u, side='right').astype(np.int8)


@dataclass(frozen=True)
class PopulationRecipe:
    source: str
    split: str
    index: int
    pool_size: int
    subgroup_cov: str = ''
    subgroup_codes: tuple = ()
    tilt: float = 0.0
    conc: float = 0.0
    center: float = 0.5
    seed: int = 0

    @property
    def population_id(self) -> str:
        return f"{self.source}-{self.split}-{self.index:04d}-{stable_seed(asdict(self)) % 16 ** 8:08x}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d['subgroup_codes'] = list(self.subgroup_codes)
        d['population_id'] = self.population_id
        return d


def _eligible(data: SourceData, partition: np.ndarray, split: str, cov: str = '', codes=()):
    """Records of one split; split 'all' (paper-scale reproduction only) uses every record."""
    mask = np.ones(len(data), bool) if split == 'all' else partition == SPLITS.index(split)
    if cov:
        mask &= np.isin(data.covariates[cov], np.asarray(codes))
    return np.flatnonzero(mask)


def sample_recipe(data: SourceData, partition: np.ndarray, split: str, index: int, pool_size: int,
                  base_seed: int = 0, p_subgroup: float = 0.3, tilt_max: float = 3.0,
                  conc_max: float = 20.0, p_conc: float = 0.5, plain: bool = False) -> PopulationRecipe:
    """Draw one recipe; plain=True gives an untilted random sample of the pool."""
    seed = stable_seed('recipe', data.name, split, index, pool_size, base_seed)
    if plain:
        return PopulationRecipe(data.name, split, index, pool_size, seed=seed)
    rng = np.random.default_rng(seed)
    cov, codes = '', ()
    if rng.random() < p_subgroup and data.covariates:
        name = str(rng.choice(sorted(data.covariates)))
        cats = rng.permutation(np.unique(data.covariates[name]))
        chosen = []
        for c in cats:                       # add categories until the pool is comfortably big
            chosen.append(int(c))
            if len(_eligible(data, partition, split, name, chosen)) >= 1.5 * pool_size:
                break
        if len(chosen) < len(cats):          # a strict subset: a real subpopulation
            cov, codes = name, tuple(sorted(chosen))
    tilt = float(rng.uniform(-tilt_max, tilt_max))
    conc = float(rng.uniform(0.0, conc_max)) if rng.random() < p_conc else 0.0
    center = float(rng.uniform(0.0, 1.0))
    return PopulationRecipe(data.name, split, index, pool_size, cov, codes, tilt, conc, center, seed)


def build_population(data: SourceData, partition: np.ndarray, recipe: PopulationRecipe):
    """(values, record indices), both in random order, len = pool_size."""
    idx = _eligible(data, partition, recipe.split, recipe.subgroup_cov, recipe.subgroup_codes)
    if len(idx) < recipe.pool_size:
        raise ValueError(f"{recipe.population_id}: {len(idx)} eligible records < pool_size {recipe.pool_size}")
    rng = np.random.default_rng(recipe.seed + 1)
    v = data.values[idx]
    r = (rankdata(v) - 0.5) / len(v)
    logw = recipe.tilt * (2.0 * r - 1.0) - recipe.conc * (r - recipe.center) ** 2
    keys = logw + rng.gumbel(size=len(v))
    pick = np.argpartition(-keys, recipe.pool_size - 1)[: recipe.pool_size]
    pick = rng.permutation(pick)
    return v[pick], idx[pick]
