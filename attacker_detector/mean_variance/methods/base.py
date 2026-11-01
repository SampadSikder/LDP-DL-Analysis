"""Common interface for every utility-recovery method.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..metrics import project_moments


@dataclass
class TrainingRound:
    round_id: str
    observation: dict
    is_fake: np.ndarray
    labels: dict
    meta: dict
    population_id: str = ''


@dataclass
class MethodOutput:
    mean_raw: float
    var_raw: float
    var: float = 0.0
    removed: Optional[np.ndarray] = None      
    scores: Optional[np.ndarray] = None
    fallback: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    runtime: float = 0.0

    def __post_init__(self):
        self.mean, self.var = project_moments(self.mean_raw, self.var_raw)


class Method:
    name = 'method'
    kind = 'recovery'         
    needs_labels = False

    def fit(self, train, val):
        return self

    def predict(self, observation: dict, labels: dict = None) -> MethodOutput:
        raise NotImplementedError

    def timed_predict(self, observation: dict, labels: dict = None) -> MethodOutput:
        t0 = time.perf_counter()
        out = self.predict(observation, labels)
        out.runtime = time.perf_counter() - t0
        return out

    def predict_many(self, observations, labels=None):
        """Outputs for many rounds; per-round runtime is measured where each prediction runs."""
        labels = labels or [None] * len(observations)
        return [self.timed_predict(o, l) for o, l in zip(observations, labels)]

    def state(self) -> dict:
        """JSON-serialisable configuration / frozen choices (thresholds, rates)."""
        return {'name': self.name}
