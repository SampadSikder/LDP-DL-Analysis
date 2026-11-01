"""Utility and diagnostic metrics (benchmark scale [-1, 1]).

Primary (defense comparisons): error against the genuine users' empirical moments
    MSE_mean = mean (mean_hat - mean_genuine)^2,  MSE_variance likewise
    AG = MSE_before - MSE_after (paper Sec. 8.1), reported separately for mean and variance;
    negative gains are kept
Attack validation: MSE to the attacker's target (paper Sec. 7.1), shift-achieved ratio.
Attack success (per round): each attacked statistic lands within tol of its intended shift;
ASR before (no defence) / after a defence, and the fraction of hits that survive the defence.
Classifier diagnostics: precision, recall, F1, PR-AUC, genuine-removal rate.
Clean rounds (no fakes): recall and F1 are undefined (NaN); precision is NaN when
nothing is removed and 0 otherwise; the genuine-removal rate is always defined.
"""

import numpy as np
from sklearn.metrics import average_precision_score


class UtilityScales:
    """Loss scales from training rounds only: std of the genuine moments across rounds.

    L = ((mean_hat - mean) / scale_mean)^2 + lambda_var ((var_hat - var) / scale_var)^2
    """

    def __init__(self, scale_mean: float, scale_var: float, lambda_var: float = 1.0):
        self.mean, self.var, self.lambda_var = float(scale_mean), float(scale_var), float(lambda_var)

    @classmethod
    def from_rounds(cls, rounds, lambda_var: float = 1.0, floor=(1e-3, 1e-4)):
        m = np.std([r.labels['genuine_mean'] for r in rounds])
        v = np.std([r.labels['genuine_variance'] for r in rounds])
        return cls(max(m, floor[0]), max(v, floor[1]), lambda_var)

    def loss(self, d_mean, d_var):
        return (np.asarray(d_mean) / self.mean) ** 2 + self.lambda_var * (np.asarray(d_var) / self.var) ** 2

    def state(self) -> dict:
        return {'scale_mean': self.mean, 'scale_var': self.var, 'lambda_var': self.lambda_var}


def project_moments(mean: float, var: float):
    """Feasible normalized moments: -1 <= mean <= 1, 0 <= var <= 1 - mean^2."""
    m = float(np.clip(mean, -1.0, 1.0))
    return m, float(np.clip(var, 0.0, 1.0 - m * m))


def squared_errors(pred_mean, pred_var, true_mean, true_var):
    return (pred_mean - true_mean) ** 2, (pred_var - true_var) ** 2


def accuracy_gain(mse_before: float, mse_after: float) -> float:
    return mse_before - mse_after


def shift_achieved(est, truth, target, baseline=None):
    """(est - base) / (target - truth), base = truth or a paired genuine-only estimate; NaN if target == truth."""
    est, truth, target = np.asarray(est, float), np.asarray(truth, float), np.asarray(target, float)
    base = truth if baseline is None else np.asarray(baseline, float)
    den = target - truth
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(den) > 1e-15, (est - base) / den, np.nan)


def attack_success(est, truth, target, tol=0.5):
    """1 if the estimate lands within tol of the intended shift, |est - target| <= tol |target - truth|, else 0.

    Same as shift_achieved(est, truth, target) in [1 - tol, 1 + tol]. NaN where the statistic is not
    attacked (target == truth, e.g. the preserved index of an offset target) or there is no target.
    """
    est, truth, target = (np.asarray(a, float) for a in (est, truth, target))
    shift = np.abs(target - truth)
    with np.errstate(invalid='ignore'):
        hit = (np.abs(est - target) <= tol * shift).astype(float)
        return np.where(shift > 1e-12, hit, np.nan)


def joint_success(*flags):
    """1 if every attacked statistic is hit, 0 if any is missed, NaN if none is attacked."""
    f = np.vstack([np.asarray(x, float) for x in flags])
    return np.where((~np.isnan(f)).any(0), np.where((f == 0).any(0), 0.0, 1.0), np.nan)


def classification(is_fake, removed, scores=None) -> dict:
    is_fake, removed = np.asarray(is_fake, bool), np.asarray(removed, bool)
    tp = int(np.sum(is_fake & removed))
    fp = int(np.sum(~is_fake & removed))
    fn = int(np.sum(is_fake & ~removed))
    n_gen = int(np.sum(~is_fake))
    out = {'tp': tp, 'fp': fp, 'fn': fn,
           'genuine_removal_rate': fp / n_gen if n_gen else np.nan,
           'precision': tp / (tp + fp) if tp + fp else np.nan,
           'recall': tp / (tp + fn) if tp + fn else np.nan}
    p, r = out['precision'], out['recall']
    out['f1'] = 2 * p * r / (p + r) if (tp + fn) and (tp + fp) and (p + r) > 0 else (0.0 if (tp + fn) else np.nan)
    if (tp + fn) == 0:
        out['precision'] = np.nan if (tp + fp) == 0 else 0.0
        out['f1'] = np.nan
    out['pr_auc'] = (float(average_precision_score(is_fake, scores))
                     if scores is not None and 0 < is_fake.sum() < len(is_fake) else np.nan)
    return out
