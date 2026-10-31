"""Detect-and-remove: per-report classifier (MLP or FT-Transformer), then standard aggregation.

Both architectures use exactly the same feature schema (features.report_features).
Training:
  * up to `reports_per_round` reports sampled per round; each row is weighted by
    1 / (rows of its round), so every round counts equally whatever its size
  * BCE on simulator labels at natural within-round prevalence (no pos_weight), so
    scores stay on a posterior-like scale. The class balance per round is whatever
    beta produced; this is documented, not rebalanced
  * standardisation fitted on training rows only; early stopping on round-weighted
    validation BCE
Threshold: picked on validation rounds by post-removal utility (UtilityScales),
from a fixed grid that includes "remove nothing", and frozen before testing.
Reaggregation uses the retained reports' own group denominators. A group with
fewer than `min_per_group` retained reports falls back to all its reports, and
that fallback is recorded.
The FT-Transformer attends across a row's features only; round context comes from
the supplied context features.
"""

import numpy as np
import torch
import torch.nn as nn

from ...models.ft_transformer import FTTransformer
from ...models.mlp import RobustAttackerDetector
from ..estimator import estimate_retained
from ..features import report_features
from ..mechanisms import get_mechanism
from ..metrics import UtilityScales
from ..transforms import Transform
from .base import Method, MethodOutput

THRESHOLDS = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2)) + (1.01,)     # 1.01 = remove nothing


def device_for(pref: str = 'auto') -> torch.device:
    if pref != 'auto':
        return torch.device(pref)
    if torch.cuda.is_available():          # first visible GPU; pick one with CUDA_VISIBLE_DEVICES
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class ClassifierRemoval(Method):
    kind = 'removal'

    def __init__(self, arch: str = 'mlp', feature_set: str = 'base', reports_per_round: int = 512,
                 epochs: int = 30, batch_size: int = 2048, lr: float = 1e-3, weight_decay: float = 1e-5,
                 patience: int = 5, min_per_group: int = 16, seed: int = 0, device: str = 'auto',
                 model_kwargs: dict = None):
        if arch not in ('mlp', 'ft'):
            raise ValueError("arch must be 'mlp' or 'ft'")
        self.arch, self.feature_set = arch, feature_set
        self.name = f"{arch}_removal" + ('' if feature_set == 'base' else f"_{feature_set}")
        self.reports_per_round, self.epochs, self.batch_size = reports_per_round, epochs, batch_size
        self.lr, self.weight_decay, self.patience = lr, weight_decay, patience
        self.min_per_group, self.seed, self.device = min_per_group, seed, device_for(device)
        self.model_kwargs = model_kwargs or {}
        self.history = []

    # ------------------------------------------------------------ data
    def _rows(self, rounds, rng, cap):
        X, y, w = [], [], []
        for r in rounds:
            F, _ = report_features(r.observation, self.feature_set)
            idx = np.arange(len(F)) if cap is None or len(F) <= cap else rng.choice(len(F), cap, replace=False)
            X.append(F[idx])
            y.append(r.is_fake[idx].astype(np.float32))
            w.append(np.full(len(idx), 1.0 / len(idx), dtype=np.float32))
        return np.concatenate(X), np.concatenate(y), np.concatenate(w)

    def _build(self, d):
        torch.manual_seed(self.seed)
        if self.arch == 'mlp':
            return RobustAttackerDetector(d, **self.model_kwargs)
        return FTTransformer(d, **{'d_token': 32, 'n_heads': 4, 'n_layers': 2, **self.model_kwargs})

    def _scores(self, F: np.ndarray) -> np.ndarray:
        self.model.eval()
        Z = torch.from_numpy(((F - self.mu_) / self.sd_).astype(np.float32))
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), 8192):
                out.append(torch.sigmoid(self.model(Z[i:i + 8192].to(self.device))).squeeze(-1).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    # ------------------------------------------------------------ fit
    def fit(self, train, val):
        rng = np.random.default_rng(self.seed)
        X, y, w = self._rows(train, rng, self.reports_per_round)
        Xv, yv, wv = self._rows(val, rng, self.reports_per_round)
        self.mu_, self.sd_ = X.mean(axis=0), X.std(axis=0)
        self.sd_[self.sd_ < 1e-8] = 1.0
        norm = lambda A: torch.from_numpy(((A - self.mu_) / self.sd_).astype(np.float32))
        Xt, yt, wt = norm(X), torch.from_numpy(y), torch.from_numpy(w)
        Xvt, yvt, wvt = norm(Xv).to(self.device), torch.from_numpy(yv).to(self.device), torch.from_numpy(wv).to(self.device)
        self.model = self._build(X.shape[1]).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss(reduction='none')
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                xb, yb, wb = Xt[b].to(self.device), yt[b].to(self.device), wt[b].to(self.device)
                loss = (bce(self.model(xb).squeeze(-1), yb) * wb).sum() / wb.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += float(loss) * len(b)
            self.model.eval()
            with torch.no_grad():       # batched: one pass over all rows overflows MPS tensor limits
                vs = sum(float((bce(self.model(Xvt[i:i + 8192]).squeeze(-1), yvt[i:i + 8192])
                                * wvt[i:i + 8192]).sum()) for i in range(0, len(Xvt), 8192))
                vl = vs / float(wvt.sum())
            self.history.append({'epoch': epoch, 'train_bce': tot / len(perm), 'val_bce': vl})
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._select_threshold(train, val)
        return self

    def _select_threshold(self, train, val):
        scales = UtilityScales.from_rounds(train)
        cached = []
        for v in val:
            F, _ = report_features(v.observation, self.feature_set)
            cached.append((v, self._scores(F)))
        res = {}
        for t in THRESHOLDS:
            losses = []
            for v, s in cached:
                out = self._aggregate(v.observation, s, t)
                losses.append(scales.loss(out.mean - v.labels['genuine_mean'], out.var - v.labels['genuine_variance']))
            res[float(t)] = float(np.mean(losses))
        self.threshold = min(res, key=res.get)
        self.val_utility_ = res
        self.scales_ = scales.state()

    # ------------------------------------------------------------ predict
    def _aggregate(self, obs, scores, threshold):
        tf = Transform(*obs['domain'], square_mode=obs['square_mode'])
        mech = get_mechanism(obs['protocol'], obs['epsilon'])
        removed = scores > threshold
        est, fb = estimate_retained(obs['reports'], obs['groups'], ~removed, mech, tf, self.min_per_group)
        e = est.normalized(tf)
        return MethodOutput(e.mean, e.variance, removed=removed, scores=scores, fallback=fb,
                            info={'threshold': threshold, 'removed_frac': float(removed.mean()),
                                  'retained_g1': int(np.sum(~removed & (obs['groups'] == 1))),
                                  'retained_g2': int(np.sum(~removed & (obs['groups'] == 2)))})

    def predict(self, obs, labels=None):
        F, _ = report_features(obs, self.feature_set)
        return self._aggregate(obs, self._scores(F), self.threshold)

    def state(self):
        return {'name': self.name, 'arch': self.arch, 'feature_set': self.feature_set,
                'threshold': self.threshold, 'val_utility': self.val_utility_, 'scales': self.scales_,
                'mu': self.mu_.tolist(), 'sd': self.sd_.tolist(), 'history': self.history,
                'reports_per_round': self.reports_per_round, 'seed': self.seed}
