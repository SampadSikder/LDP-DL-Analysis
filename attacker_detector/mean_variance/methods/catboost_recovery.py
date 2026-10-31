"""CatBoost recovery: regress the genuine mean and second moment from a round summary.

Two regressors (mean, second moment) on features.round_summary; variance is
derived as second - mean^2, so the projection in MethodOutput applies. This is a
standard-loss regression baseline, not the neural hybrid loss. If catboost is not
installed the method reports itself unavailable and is never silently replaced.
"""

import numpy as np

from ..features import round_summary
from .base import Method, MethodOutput

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:          # pragma: no cover - depends on the environment
    CATBOOST_AVAILABLE = False


class CatBoostRecovery(Method):
    name, kind = 'catboost_recovery', 'recovery'

    def __init__(self, iterations: int = 1000, depth: int = 6, learning_rate: float = 0.05, seed: int = 0):
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost is not installed: pip install catboost (env ldp-attacker-detector)")
        self.params = dict(iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=seed,
                           loss_function='RMSE', verbose=False, allow_writing_files=False)

    def fit(self, train, val):
        X = np.stack([round_summary(r.observation)[0] for r in train])
        Xv = np.stack([round_summary(r.observation)[0] for r in val])
        self.models = {}
        for key in ('genuine_mean', 'genuine_second_moment'):
            y = np.array([r.labels[key] for r in train])
            yv = np.array([r.labels[key] for r in val])
            m = CatBoostRegressor(**self.params, use_best_model=True, od_type='Iter', od_wait=50)
            m.fit(X, y, eval_set=(Xv, yv))
            self.models[key] = m
        return self

    def predict(self, obs, labels=None):
        x = round_summary(obs)[0][None]
        mean = float(self.models['genuine_mean'].predict(x)[0])
        second = float(self.models['genuine_second_moment'].predict(x)[0])
        return MethodOutput(mean, second - mean ** 2)

    def state(self):
        return {'name': self.name, **self.params,
                'best_iterations': {k: int(m.get_best_iteration() or 0) for k, m in self.models.items()}}
