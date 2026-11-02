from .base import Method, MethodOutput, TrainingRound
from .baselines import ConstantMethod, KMeansRecovery, OracleRemoval, StandardMethod
from .catboost_recovery import CATBOOST_AVAILABLE, CatBoostRecovery
from .removal import ClassifierRemoval
import re

from .set_models import FEATURE_VIEWS, LOSS_VARIANTS, SetRecovery

# <arch>_recovery[_<view>][[<loss>]]; the view is part of the name so outputs never collide
SET_NAME = re.compile(r'^(deepsets|settransformer)_recovery(?:_(context|context_report))?(?:\[(.+)\])?$')


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
    m = SET_NAME.match(name)
    if m:
        arch, view, loss = m.group(1), m.group(2) or 'minimal', m.group(3) or 'utility+clean+aux'
        if loss not in LOSS_VARIANTS:
            raise ValueError(f"unknown loss variant in {name!r}; one of {LOSS_VARIANTS}")
        kw = dict(nn_cfg)
        cfg_view = kw.pop('feature_view', None)
        if cfg_view is not None and cfg_view != view:
            # a config-level view would write under the baseline's name and overwrite its results
            raise ValueError(f"neural.feature_view={cfg_view!r} conflicts with method {name!r}; select the view "
                             f"by method name, e.g. {arch}_recovery_{cfg_view}")
        return SetRecovery(arch, loss=loss, feature_view=view, seed=seed, **kw)
    raise ValueError(f"unknown method {name!r}")


ALL_METHODS = ('standard', 'constant', 'oracle_removal', 'kmeans', 'mlp_removal', 'ft_removal',
               'catboost_recovery', 'deepsets_recovery', 'settransformer_recovery')
