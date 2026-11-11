"""Utility-selected threshold for reference-beta models (attacker_detector/reference_beta.py).

The F1-selected threshold (threshold.json) is kept. This adds a second frozen threshold per model,
threshold_utility.json, chosen on utility recovery instead of detection F1:

    minimise  mean over utility-validation runs of IGR(threshold)^2

IGR (attacker_detector/igr.py) = (sum_t f_hat_t - sum_t f_t) / (beta * r) after removing the flagged users of
a complete run: 0 = the attack's gain on the targets is gone, < 0 = genuine supporters of the targets were
removed too. Squaring penalises leftover gain, over-removal and run-to-run noise alike.

Utility-validation runs are new complete runs of the model's eligible configs at its source beta, seeded with
replicate indices from 2000 (training runs use 0.., held-out runs follow them, validation runs 1000..), on the
model's source datasets only (a cross-dataset model never sees its test dataset). They are simulated once per
(data root, protocol, dataset, config) by scripts/igr/run_igr.py uval and stored compactly (UtilityRun): the
per-run v3 features, labels and each user's summed report on the run's target items, which is all IGR needs.

Candidates are every distinct score boundary of the pooled utility-validation scores plus flag-none; flag-all
is excluded (nothing left to estimate from). Ties on the objective go to the higher threshold (fewer users
removed). Rule at test time: logit > threshold, as for the F1 threshold.
"""

import math
import os

import numpy as np

from .reference_beta import (DECISION_OPERATOR, SCORE_SCALE, VARIANT, _sigmoid, _write_json, decide,
                             file_sha256, load_scaler, load_threshold)

THRESHOLD_FILE = 'threshold_utility.json'
OBJECTIVE = 'mean over utility-validation runs of IGR^2 after removing the flagged users'
SEARCH = ('every distinct utility-validation score boundary (midpoint between neighbouring distinct scores) '
          'plus flag-none; flag-all excluded; ties -> the higher threshold')
REPLICATE_OFFSET = 2000


class UtilityRun:
    """What IGR needs from one complete run, without the (n, d) report matrix."""

    def __init__(self, run, feats, labels, target_sum, A, B, true_t):
        self.run = dict(run)
        self.feats = np.asarray(feats, dtype=np.float64)          # (n, 4) v3 features before any scaler
        self.labels = np.asarray(labels).astype(np.int8)
        self.target_sum = np.asarray(target_sum, dtype=np.float64)  # sum of the user's support over targets
        self.A, self.B, self.true_t = float(A), float(B), float(true_t)
        self.beta = float(run['attacker_ratio'])
        self.r = int(run['target_set_size'])

    @property
    def n(self):
        return len(self.labels)

    @classmethod
    def from_rundata(cls, rd):
        """From attacker_detector.igr.RunData (full support bits)."""
        bits = rd.support.bits[:, rd.targets]
        target_sum = rd.support.lo * len(rd.targets) + (rd.support.hi - rd.support.lo) * bits.sum(axis=1)
        return cls(rd.run, rd.feats, rd.labels, target_sum, rd.A, rd.B, rd.f_true[rd.targets].sum())

    def igr(self, flagged):
        """IGR after removing the users in `flagged` (bool mask); nan when nobody is left."""
        flagged = np.asarray(flagged, dtype=bool)
        kept = self.n - int(flagged.sum())
        if kept == 0:
            return float('nan')
        est = self.A * (self.target_sum.sum() - self.target_sum[flagged].sum()) / kept - self.r * self.B
        return (est - self.true_t) / (self.beta * self.r)

    def save(self, path):
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp.npz'
        np.savez(tmp, feats=self.feats.astype(np.float32), labels=self.labels,
                 target_sum=self.target_sum, A=self.A, B=self.B, true_t=self.true_t,
                 run=json.dumps(self.run, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        import json
        with np.load(path, allow_pickle=False) as z:
            return cls(json.loads(str(z['run'])), z['feats'].astype(np.float64), z['labels'], z['target_sum'],
                       float(z['A']), float(z['B']), float(z['true_t']))


def igr_curve(scores, urun, thresholds):
    """IGR of one run at every threshold (any order) under logit > threshold; nan where every user is flagged.
    Exact: users are sorted once and the flagged set of a threshold is a prefix of that order."""
    s = np.asarray(scores, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64)
    order = np.argsort(-s, kind='stable')
    s_desc = s[order]
    csum = np.r_[0.0, np.cumsum(urun.target_sum[order])]
    # flagged count at thr = #{s > thr}; s_desc is descending, so search the ascending negation
    k = np.searchsorted(-s_desc, -thr, side='left')
    kept = urun.n - k
    with np.errstate(divide='ignore', invalid='ignore'):
        est = urun.A * (csum[-1] - csum[k]) / kept - urun.r * urun.B
        out = (est - urun.true_t) / (urun.beta * urun.r)
    out[kept == 0] = np.nan
    return out


def candidate_thresholds(score_list):
    """Flag-none (+inf), then the midpoint below every distinct pooled score, highest first; the last one
    (below the lowest score, = flag-all) is dropped."""
    pooled = np.unique(np.concatenate([np.asarray(s, dtype=np.float64) for s in score_list]))[::-1]
    mids = (pooled[:-1] + pooled[1:]) / 2.0
    return np.r_[np.inf, mids]


def select(score_list, runs):
    """Threshold with the lowest mean IGR^2 over the runs. score_list[i] are run i's logits."""
    if not runs or len(score_list) != len(runs):
        raise ValueError('need one score array per utility-validation run')
    for s, u in zip(score_list, runs):
        if len(s) != u.n or not np.isfinite(s).all():
            raise ValueError('scores must be finite and one per user of the run')
    cand = candidate_thresholds(score_list)
    sq = np.zeros(len(cand))
    for s, u in zip(score_list, runs):
        g = igr_curve(s, u, cand)
        sq += np.where(np.isnan(g), np.inf, g * g)
    obj = sq / len(runs)
    best = int(np.argmin(obj))                       # first minimum = highest threshold
    return float(cand[best]), obj, cand


def summarize(threshold, score_list, runs):
    """IGR and detection metrics of one threshold on the utility-validation runs."""
    igrs, tp, fp, fn, tn, flagged = [], 0, 0, 0, 0, 0
    for s, u in zip(score_list, runs):
        f = decide(s, threshold)
        y = u.labels.astype(bool)
        igrs.append(u.igr(f))
        tp += int((f & y).sum()); fp += int((f & ~y).sum()); fn += int((~f & y).sum()); tn += int((~f & ~y).sum())
        flagged += int(f.sum())
    igrs = np.asarray(igrs)
    n = tp + fp + fn + tn
    emptied = np.isnan(igrs)          # every user of a run flagged: nothing left to estimate from
    return {
        'mean_igr2': float('inf') if emptied.any() else float(np.mean(igrs ** 2)),
        'mean_igr': float(np.mean(igrs)) if not emptied.any() else float('nan'),
        'mean_abs_igr': float(np.mean(np.abs(igrs))) if not emptied.any() else float('nan'),
        'runs_with_everyone_flagged': int(emptied.sum()),
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        'precision': tp / (tp + fp) if tp + fp else 0.0, 'recall': tp / (tp + fn) if tp + fn else 0.0,
        'fpr': fp / (fp + tn) if fp + tn else 0.0, 'flagged_share': flagged / n if n else 0.0,
        'per_run_igr': igrs.tolist(),
    }


def model_runs(model_dir, cache_dirs):
    """The utility-validation runs a model may use: its protocol, its source datasets, its source beta."""
    rec = load_threshold(model_dir)
    import glob
    runs, files = [], []
    for ds in rec['train_datasets']:
        cell = f"{ds.capitalize()}_{rec['protocol']}"
        for d in cache_dirs:
            for p in sorted(glob.glob(os.path.join(d, cell, '*.npz'))):
                u = UtilityRun.load(p)
                if not math.isclose(u.beta, rec['source_beta'], abs_tol=1e-9):
                    raise ValueError(f'{p}: beta {u.beta} is not the source beta {rec["source_beta"]}')
                if u.run['dataset_type'] == rec['test_dataset'] and rec['training_method'] == 'cross':
                    raise ValueError(f'{p}: test dataset run offered to a cross-dataset model')
                runs.append(u), files.append(p)
    return rec, runs, files


def select_for_model(model_dir, cache_dirs, model, device, ft_logits, model_input):
    """Pick, freeze and save threshold_utility.json for one reference-beta model."""
    rec, runs, files = model_runs(model_dir, cache_dirs)
    if not runs:
        raise FileNotFoundError(f'no utility-validation runs for {model_dir} in {cache_dirs}')
    sha = file_sha256(os.path.join(model_dir, 'model.pt'))
    if sha != rec['model']['checkpoint_sha256']:
        raise RuntimeError(f'{model_dir}/model.pt is not the checkpoint of its threshold.json')
    scaler = load_scaler(model_dir)
    scores = [ft_logits(model, model_input(u.feats, scaler), device).astype(np.float64) for u in runs]
    thr, obj, cand = select(scores, runs)
    at = summarize(thr, scores, runs)
    ref = {name: summarize(t, scores, runs) for name, t in
           (('at_f1_threshold', rec['threshold']), ('at_p0.5', 0.0), ('no_defense', np.inf))}
    out = {
        'variant': VARIANT, 'threshold_kind': 'utility',
        'model': rec['model'], 'protocol': rec['protocol'], 'training_method': rec['training_method'],
        'train_datasets': rec['train_datasets'], 'test_dataset': rec['test_dataset'],
        'source_beta': rec['source_beta'], 'seed': rec['seed'],
        'objective': OBJECTIVE, 'search': SEARCH, 'score_scale': SCORE_SCALE,
        'decision_operator': DECISION_OPERATOR,
        'threshold': thr if math.isfinite(thr) else ('inf' if thr > 0 else '-inf'),
        'threshold_probability': _sigmoid(thr),
        'operating_point': 'flag_none' if thr == np.inf else 'interior',
        'candidates': int(len(cand)),
        'utility_validation_runs': [{'file': os.path.relpath(p), 'dataset_type': u.run['dataset_type'],
                                     'epsilon': u.run['epsilon'], 'attacker_ratio': u.beta,
                                     'splits': u.run['splits'], 'replicate': u.run['replicate'],
                                     'seed': u.run['seed'], 'n': u.n} for p, u in zip(files, runs)],
        'validation_runs': len(runs),
        **{f'validation_{k}': v for k, v in at.items()},
        'reference': ref,
    }
    _write_json(os.path.join(model_dir, THRESHOLD_FILE), out)
    return out
