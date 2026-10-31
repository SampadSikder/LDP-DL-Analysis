"""Method registry. make_method raises MethodUnavailable instead of substituting another method."""

from .base import Method, MethodOutput, TrainingRound
from .baselines import ConstantMethod, KMeansRecovery, OracleRemoval, StandardMethod
from .catboost_recovery import CATBOOST_AVAILABLE, CatBoostRecovery
from .removal import ClassifierRemoval
from .set_models import LOSS_VARIANTS, SetRecovery


class MethodUnavailable(RuntimeError):
    pass


def make_method(name: str, seed: int = 0, cfg: dict = None) -> Method:
    cfg = cfg or {}
    nn_cfg = cfg.get('neural', {})
    if name == 'standard':
        return StandardMethod()
    if name == 'constant':
        return ConstantMethod()
    if name == 'oracle_removal':
        return OracleRemoval()
    if name == 'kmeans':
        return KMeansRecovery(n_subsets=cfg.get('kmeans_subsets', 5000), seed=0)
    if name in ('mlp_removal', 'ft_removal', 'mlp_removal_mech', 'ft_removal_mech'):
        arch = name.split('_')[0]
        fs = 'mech' if name.endswith('_mech') else 'base'
        return ClassifierRemoval(arch, feature_set=fs, seed=seed, **cfg.get('classifier', {}))
    if name == 'catboost_recovery':
        if not CATBOOST_AVAILABLE:
            raise MethodUnavailable("catboost not installed (pip install catboost)")
        return CatBoostRecovery(seed=seed, **cfg.get('catboost', {}))
    for arch in ('deepsets', 'settransformer'):
        if name == f'{arch}_recovery':
            return SetRecovery(arch, seed=seed, **nn_cfg)
        for variant in LOSS_VARIANTS:
            if name == f'{arch}_recovery[{variant}]':
                return SetRecovery(arch, loss=variant, seed=seed, **nn_cfg)
    raise ValueError(f"unknown method {name!r}")


ALL_METHODS = ('standard', 'constant', 'oracle_removal', 'kmeans', 'mlp_removal', 'ft_removal',
               'catboost_recovery', 'deepsets_recovery', 'settransformer_recovery')
