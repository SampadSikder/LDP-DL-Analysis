"""Item gain ratio (IGR) of removal defences on the frequency oracles.

IGR is the utility metric of Li et al., "Mitigating Data Poisoning Attacks to Local Differential
Privacy" (CCS 2025, arXiv 2506.02156, Eq. 2), computed the way their code does it
(MDPA_LDP/Recovery.py):

    IGR = (sum_t f_hat_t - sum_t f_t) / (beta * r)

  t       the r target items of the run
  f_t     true frequency of t among the genuine users
  f_hat   the protocol's unbiased frequency estimate from the reports the server keeps,
          divided by the number of reports kept (all n reports, fakes included, without a defence)
  beta    fake-user share, r = target-set size

The baseline attack (each fake user reports a random target honestly) gives about 1/r.
IGR = 0 means the attack's gain on the targets is gone. IGR < 0 means the targets were pushed
below their true frequency, i.e. too many genuine supporters of the targets were removed.

The paper post-processes the attacked estimate. Here the defence is removal: drop the users
a detector flags, then re-estimate from the rest. These are scored on the same simulated users:
  ours_in_dist / ours_cross  the v3 FT-Transformer of results/<root>/{in_dist,cross}/, at two cut-offs:
                             variant 'original' = p > 0.5 (main.py's rule, a 50/50 prior, as trained);
                             variant 'prior' = p > 1 - beta_train, beta_train = the design's default
                             attacker share (known when training, not read from the scored run).
                             A reference-beta model (attacker_detector/reference_beta.py, a model dir
                             with threshold.json) is scored once, variant 'frozen': its own scaler and
                             its validation-selected threshold, logit > threshold
  fiad, diffstats            the MDPA_LDP detectors (attacker_detector/FIAD.py, DiffStats.py)
  no_defense                 every report kept
  oracle                     exactly the fake users removed (the floor: LDP noise only)
The rows also carry MSE over all items (and over the targets) against the genuine users' true
frequencies, plus each detector's TP/FP/FN/TN on the run.

The datasets keep features only, not reports. So each held-out LDP run of a dataset's design is
simulated again, seeded exactly like attacker_detector/baselines.py (main.py --baseline). The true
items and the target set match the dataset's held-out run; the perturbation randomness is fresh.
Our model is then fed features built from that simulated run, through the same path main.py uses:
the v2 extractor, the v3 per-run transform, and the global z-score of the training data directory.
"""

import json
import math
import os
import random
import time

import numpy as np

DETECTOR_ORDER = ['no_defense', 'ours_in_dist', 'ours_cross', 'fiad', 'diffstats', 'oracle']
ROW_KEYS = ['root', 'dataset_type', 'protocol', 'epsilon', 'attacker_ratio', 'target_set_size',
            'splits', 'replicate', 'n', 'seed']
ROW_COLS = ROW_KEYS + ['detector', 'variant', 'model_seed', 'budget', 'prior_beta', 'threshold', 'model_sha',
                       'flagged', 'kept',
                       'TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'f1',
                       'target_true', 'target_est', 'gain', 'igr', 'mse', 'mse_target', 'seconds']


def run_tag(run):
    """File name of one simulated run (unique within a <Dataset>_<PROTOCOL> cell)."""
    return (f"eps{float(run['epsilon']):g}_b{float(run['attacker_ratio']):g}_r{int(run['target_set_size'])}"
            f"_s{int(run['splits'])}_rep{int(run['replicate'])}")


def cell_name(dataset_type, protocol):
    return f'{dataset_type.capitalize()}_{protocol}'


# ------------------------------------------------------------------------------------------
# frequency estimation
# ------------------------------------------------------------------------------------------

def estimator(protocol, epsilon):
    """(A, B) with f_v(S) = A * C_v(S) / |S| - B, where C_v(S) = sum over users in S of the
    support matrix column v. Same estimators as the generators (attacks.py) use."""
    e = math.exp(epsilon)
    if protocol == 'OUE':
        p, q = 0.5, 1.0 / (e + 1.0)
        return 1.0 / (p - q), q / (p - q)
    if protocol.startswith('OLH'):
        g = int(round(e)) + 1
        p = e / (e + g - 1)
        return g / (p * g - 1), 1.0 / (p * g - 1)
    if protocol.startswith('HST'):
        return 1.0, 0.0          # support = y * s, already unbiased per user
    raise ValueError(f'no estimator for protocol {protocol!r}')


class Support:
    """A run's support matrix as one bit per (user, item), support > 0, plus the two values the
    bits stand for: 0 / 1 for OUE and OLH, -c / +c for HST."""

    def __init__(self, bits, lo, hi):
        self.bits = bits                  # (n, d) bool
        self.lo, self.hi = float(lo), float(hi)

    @classmethod
    def from_dense(cls, support, chunk=50_000):
        lo, hi = float(support.min()), float(support.max())
        if lo == hi:
            raise ValueError('support matrix is constant')
        bits = np.empty(support.shape, dtype=bool)
        for a in range(0, len(support), chunk):
            s = support[a:a + chunk]
            if not ((s == lo) | (s == hi)).all():
                raise ValueError(f'support matrix takes more than two values ({lo}, {hi}, ...)')
            np.greater(s, (lo + hi) / 2, out=bits[a:a + chunk])
        return cls(bits, lo, hi)

    @property
    def n(self):
        return self.bits.shape[0]

    @property
    def d(self):
        return self.bits.shape[1]

    def colsum(self, mask=None, chunk=50_000):
        """Column sums of the support values over the users in mask (all users if None)."""
        idx = np.arange(self.n) if mask is None else np.flatnonzero(mask)
        cnt = np.zeros(self.d, dtype=np.int64)
        for a in range(0, len(idx), chunk):
            cnt += self.bits[idx[a:a + chunk]].sum(axis=0)
        return self.lo * len(idx) + (self.hi - self.lo) * cnt

    def save_arrays(self):
        return {'support_bits': np.packbits(self.bits, axis=1), 'support_lo': self.lo,
                'support_hi': self.hi, 'domain': self.d}

    @classmethod
    def load_arrays(cls, z):
        d = int(z['domain'])
        bits = np.unpackbits(z['support_bits'], axis=1, count=d).astype(bool)
        return cls(bits, float(z['support_lo']), float(z['support_hi']))


class RunData:
    """Everything IGR needs from one simulated run."""

    def __init__(self, run, labels, targets, f_true, support, feats=None):
        self.run = run
        self.labels = np.asarray(labels).astype(np.int8)
        self.targets = np.asarray(sorted(int(t) for t in targets), dtype=np.int64)
        self.f_true = np.asarray(f_true, dtype=np.float64)
        self.support = support
        self.feats = feats                # (n, 4) v3 features before the global z-score
        self.A, self.B = estimator(run['protocol'], float(run['epsilon']))
        self.colsum_all = support.colsum()

    @property
    def n(self):
        return len(self.labels)

    def estimate(self, flagged=None):
        """Frequency estimate from the reports left after removing the flagged users."""
        if flagged is None or not flagged.any():
            return self.A * self.colsum_all / self.n - self.B
        kept = self.n - int(flagged.sum())
        if kept == 0:
            return np.full(self.support.d, np.nan)
        return self.A * (self.colsum_all - self.support.colsum(flagged)) / kept - self.B

    def metrics(self, flagged):
        flagged = np.zeros(self.n, dtype=bool) if flagged is None else np.asarray(flagged, dtype=bool)
        f = self.estimate(flagged)
        y = self.labels.astype(bool)
        tp = int((flagged & y).sum())
        fp = int((flagged & ~y).sum())
        fn = int((~flagged & y).sum())
        tn = self.n - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        beta, r = float(self.run['attacker_ratio']), int(self.run['target_set_size'])
        true_t = float(self.f_true[self.targets].sum())
        est_t = float(f[self.targets].sum())
        gain = est_t - true_t
        return {'flagged': int(flagged.sum()), 'kept': self.n - int(flagged.sum()),
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'precision': prec, 'recall': rec,
                'f1': 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                'target_true': true_t, 'target_est': est_t, 'gain': gain, 'igr': gain / (beta * r),
                'mse': float(np.mean((f - self.f_true) ** 2)),
                'mse_target': float(np.mean((f[self.targets] - self.f_true[self.targets]) ** 2))}

    # ---- cache (lets models trained later, e.g. WSP-edited ones, be scored on the same users)
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp.npz'        # per process: two launches never share a temp file
        np.savez(tmp, labels=self.labels, targets=self.targets, f_true=self.f_true,
                 feats=self.feats.astype(np.float32), run=json.dumps(self.run, default=_jsonable),
                 **self.support.save_arrays())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            run = json.loads(str(z['run']))
            return cls(run, z['labels'], z['targets'], z['f_true'], Support.load_arrays(z),
                       feats=z['feats'].astype(np.float64))


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------
# simulation (seeded like attacker_detector/baselines.py)
# ------------------------------------------------------------------------------------------

def simulate(run, processors=4):
    """Re-simulate one LDP run. Returns (X, targets, support, labels, estimate, one_list).
    The generator returns neither the true items nor the target set; both are its first seeded
    draws, so they are drawn again here with the same seed before the generator runs."""
    from generate_dataset import _resolve_protocol
    from .data.generators import generate_perturbed_data
    from .data.generators.pca_graph import get_distribution_generator

    seed, n, d = int(run['seed']), int(run['n']), int(run['domain'])
    np.random.seed(seed)
    X, _ = get_distribution_generator(run['dataset_type'])(n, d, seed=seed)
    targets = np.random.choice(d, size=min(int(run['target_set_size']), d), replace=False)

    base_protocol, olh_setting = _resolve_protocol(run['protocol'])
    random.seed(seed)                     # as baselines.py does
    support, labels, _, estimate, one_list = generate_perturbed_data(
        epsilon=run['epsilon'], domain=d, n=n, protocol=base_protocol, ratio=run['attacker_ratio'],
        target_set_size=run['target_set_size'], splits=run['splits'], dataset_type=run['dataset_type'],
        h_ao=run['h_ao'], seed=seed, processors=processors, olh_setting=olh_setting)
    return np.asarray(X), np.sort(targets), support, labels, np.asarray(estimate, dtype=np.float64), one_list


def genuine_frequency(X, labels, d):
    n_gen = int((np.asarray(labels) == 0).sum())   # genuine users come first in every generator
    return np.bincount(np.asarray(X[:n_gen], dtype=np.int64), minlength=d)[:d] / n_gen


# ------------------------------------------------------------------------------------------
# our model's input: v2 features -> v3 transform (one run) -> global z-score of the training data
# ------------------------------------------------------------------------------------------

def v3_features(v2, run, labels):
    """(n, 4) v3 features of one run before the final global z-score, through
    attacker_detector.data.dataset.apply_feature_set_v3 itself (the run is its only LDP run)."""
    from .data.dataset import NpyDataset, apply_feature_set_v3, _CONF_HOLDOUT
    from .data.generators import FEATURE_NAMES_V2
    n = len(labels)
    config = np.empty((n, _CONF_HOLDOUT + 1), dtype=object)
    config[:, 0] = str(run['target_set_size'])
    config[:, 1] = repr(float(run['attacker_ratio']))
    config[:, 2] = run['protocol']
    config[:, 3] = str(run['splits'])
    config[:, 4] = repr(float(run['epsilon']))
    config[:, 5] = run['dataset_type']
    config[:, 6] = str(n)
    config[:, 7] = '0'
    config[:, 8] = np.arange(n).astype(str)
    config[:, 9] = '1'
    k = len(FEATURE_NAMES_V2)
    ds = NpyDataset(np.asarray(v2, dtype=np.float32), np.asarray(labels, dtype=np.float32), config.astype(str),
                    list(FEATURE_NAMES_V2), {'feature_names': list(FEATURE_NAMES_V2),
                                             'mean': [0.0] * k, 'std': [1.0] * k})
    apply_feature_set_v3(ds)
    # undo the z-score apply_feature_set_v3 fitted on this one run; the training one is applied later
    return ds.features.astype(np.float64) * np.asarray(ds.norm_stats['std']) + np.asarray(ds.norm_stats['mean'])


def train_v3_norm(data_dir, path):
    """Mean / std of the v3 features over a whole training data directory (what main.py's
    apply_feature_set_v3 z-scores with, before any row filtering). Cached as JSON."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from .data import load_npy_dataset
    from .data.dataset import apply_feature_set_v3
    t0 = time.time()
    ds = load_npy_dataset(data_dir)
    apply_feature_set_v3(ds)
    out = {'mean': list(ds.norm_stats['mean']), 'std': list(ds.norm_stats['std']),
           'feature_names': ds.norm_stats['feature_names'], 'rows': len(ds),
           'runs': ds.norm_stats.get('runs'), 'data_dir': data_dir, 'seconds': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(out, f, indent=1)
    os.replace(path + '.tmp', path)
    return out


def model_input(feats, norm):
    mu, sd = np.asarray(norm['mean']), np.asarray(norm['std'])
    return ((feats - mu) / sd).astype(np.float32)


def ft_logits(model, X, device, batch=32_768):
    """The FT-Transformer's logit for every user."""
    import torch
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(X), batch):
            out[a:a + batch] = model(torch.from_numpy(X[a:a + batch]).to(device)).reshape(-1).cpu().numpy()
    return out


def ft_flags(model, X, device, batch=32_768):
    """Users the FT-Transformer flags: sigmoid(logit) > 0.5, main.py's rule."""
    return ft_logits(model, X, device, batch) > 0


def prior_cutoff(beta):
    """Logit cut-off for a model trained on 50/50 batches, moved to an attacker share beta:
    flag when p > 1 - beta, i.e. logit > log((1 - beta) / beta). beta is fixed at training time
    (the design's default share), never read from the run being scored."""
    beta = float(beta)
    if not 0.0 < beta < 1.0:
        raise ValueError(f'attacker share must be in (0, 1), got {beta}')
    return math.log((1.0 - beta) / beta)


# FT-Transformer arguments of the original runs (scripts/run_extra_eps.sh FT array, --feature-set v3)
FT_SPEC = {'d_token': 64, 'n_heads': 8, 'n_layers': 3, 'ffn_d_multiplier': 1.33,
           'attention_dropout': 0.1, 'residual_dropout': 0.0}


def build_ft(input_dim, spec=None, state_dict=None, device='cpu'):
    """FT-Transformer as main.py builds it (dropout only matters in training)."""
    from .models.ft_transformer import FTTransformer
    spec = FT_SPEC if spec is None else spec
    kw = {k: spec[k] for k in FT_SPEC if spec.get(k) is not None}
    model = FTTransformer(input_dim, dropout_rate=spec.get('dropout', 0.2), **kw)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_ft(model_path, spec, device):
    import torch
    ck = torch.load(model_path, map_location='cpu')
    return build_ft(4, spec, ck['model_state_dict'] if 'model_state_dict' in ck else ck, device)
