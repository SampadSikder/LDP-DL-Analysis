"""Direct recovery with permutation-invariant set models (Deep Sets, Set Transformer).

Input tokens: one per report, (y / scale, is_g1); scale = s for PM, 1 for SR.
There are no position embeddings. Round context (allowed observation-only values):
log group sizes and the standard decoded mean and variance.

Outputs use a valid parameterisation on the benchmark scale:
    mean = tanh(a),  second = mean^2 + (1 - mean^2) sigmoid(b)
so -1 <= mean <= 1 and mean^2 <= second <= 1 by construction, and
variance = second - mean^2.

Hybrid loss (normalised per round):
    L_u     = ((mean_hat - mean) / scale_mean)^2 + lambda_var ((var_hat - var) / scale_var)^2
    L       = mean_attacked(L_u) + lambda_clean mean_clean(L_u)
              + lambda_aux mean_rounds(mean_tokens(BCE(score, is_fake)))
Variants: 'utility' (attacked rounds only), 'utility+clean', 'utility+clean+aux'.
Scales come from training rounds only. AG is not added as a separate term: its
MSE_before does not depend on the parameters.

Large rounds: training samples at most `token_cap` reports per round per step
(uniformly, without replacement), while the context keeps full-round counts and
moments. Inference uses every report. The approximation is recorded in state().
"""

import math

import numpy as np
import torch
import torch.nn as nn

from ..estimator import G1
from ..features import standard_moments
from ..mechanisms import get_mechanism
from ..metrics import UtilityScales
from .base import Method, MethodOutput
from .removal import device_for

LOSS_VARIANTS = ('utility', 'utility+clean', 'utility+clean+aux')


def tokens_and_context(obs):
    mech = get_mechanism(obs['protocol'], obs['epsilon'])
    scale = mech.s if obs['protocol'] == 'PM' else 1.0
    y = np.asarray(obs['reports'], np.float32) / scale
    g1 = (np.asarray(obs['groups']) == G1).astype(np.float32)
    mean_n, var_n = standard_moments(obs)
    ctx = np.array([math.log(max(obs['n_g1'], 1)), math.log(max(obs['n_g2'], 1)), mean_n, var_n], np.float32)
    return np.column_stack([y, g1]), ctx


def _heads(z):
    a, b = z[..., 0], z[..., 1]
    mean = torch.tanh(a)
    var = (1 - mean ** 2) * torch.sigmoid(b)
    return mean, var


class DeepSetsNet(nn.Module):
    def __init__(self, d_tok=2, d_ctx=4, width=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_tok, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * width + d_ctx, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 2))
        self.aux = nn.Sequential(nn.Linear(2 * width + d_ctx, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, tok, mask, ctx):
        h = self.phi(tok)                                       # (B, T, w)
        g1 = tok[..., 1:2] * mask.unsqueeze(-1)
        g2 = (1 - tok[..., 1:2]) * mask.unsqueeze(-1)
        p1 = (h * g1).sum(1) / g1.sum(1).clamp_min(1.0)
        p2 = (h * g2).sum(1) / g2.sum(1).clamp_min(1.0)
        pooled = torch.cat([p1, p2, ctx], -1)
        own = torch.where(tok[..., 1:2] > 0.5, p1.unsqueeze(1), p2.unsqueeze(1))
        logits = self.aux(torch.cat([h, own, ctx.unsqueeze(1).expand(-1, h.shape[1], -1)], -1)).squeeze(-1)
        return _heads(self.rho(pooled)), logits


class _MAB(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, q, kv, key_pad=None):
        h = self.ln1(q + self.att(q, kv, kv, key_padding_mask=key_pad, need_weights=False)[0])
        return self.ln2(h + self.ff(h))


class _ISAB(nn.Module):
    def __init__(self, d, heads, n_ind):
        super().__init__()
        self.ind = nn.Parameter(torch.randn(1, n_ind, d) / math.sqrt(d))
        self.m1, self.m2 = _MAB(d, heads), _MAB(d, heads)

    def forward(self, x, key_pad):
        h = self.m1(self.ind.expand(x.shape[0], -1, -1), x, key_pad)
        return self.m2(x, h)


class SetTransformerNet(nn.Module):
    def __init__(self, d_tok=2, d_ctx=4, d=64, heads=4, n_ind=16, n_seeds=2, n_isab=2):
        super().__init__()
        self.embed = nn.Linear(d_tok, d)
        self.isabs = nn.ModuleList([_ISAB(d, heads, n_ind) for _ in range(n_isab)])
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d) / math.sqrt(d))
        self.pma = _MAB(d, heads)
        self.head = nn.Sequential(nn.Linear(n_seeds * d + d_ctx, 128), nn.ReLU(), nn.Linear(128, 2))
        self.aux = nn.Sequential(nn.Linear(d + n_seeds * d + d_ctx, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, tok, mask, ctx):
        pad = ~mask.bool()
        h = self.embed(tok)
        for blk in self.isabs:
            h = blk(h, pad)
        pooled = self.pma(self.seeds.expand(tok.shape[0], -1, -1), h, pad).flatten(1)
        z = torch.cat([pooled, ctx], -1)
        logits = self.aux(torch.cat([h, z.unsqueeze(1).expand(-1, h.shape[1], -1)], -1)).squeeze(-1)
        return _heads(self.head(z)), logits


class SetRecovery(Method):
    kind = 'recovery'

    def __init__(self, arch: str = 'deepsets', loss: str = 'utility+clean+aux', lambda_clean: float = 1.0,
                 lambda_aux: float = 0.1, lambda_var: float = 1.0, token_cap: int = 1024, batch_rounds: int = 16,
                 epochs: int = 40, lr: float = 1e-3, weight_decay: float = 1e-5, patience: int = 6,
                 seed: int = 0, device: str = 'auto'):
        if arch not in ('deepsets', 'settransformer'):
            raise ValueError("arch must be 'deepsets' or 'settransformer'")
        if loss not in LOSS_VARIANTS:
            raise ValueError(f"loss must be one of {LOSS_VARIANTS}")
        self.arch, self.loss_variant = arch, loss
        self.name = f"{arch}_recovery" + ('' if loss == 'utility+clean+aux' else f"[{loss}]")
        self.lambda_clean = lambda_clean if 'clean' in loss else 0.0
        self.lambda_aux = lambda_aux if 'aux' in loss else 0.0
        self.lambda_var, self.token_cap, self.batch_rounds = lambda_var, token_cap, batch_rounds
        self.epochs, self.lr, self.weight_decay, self.patience = epochs, lr, weight_decay, patience
        self.seed, self.device = seed, device_for(device)
        self.history = []

    def _prep(self, rounds):
        out = []
        for r in rounds:
            tok, ctx = tokens_and_context(r.observation)
            out.append((tok, ctx, r.is_fake.astype(np.float32),
                        np.float32(r.labels['genuine_mean']), np.float32(r.labels['genuine_variance']),
                        bool(r.meta.get('n_fake', 0) == 0)))
        return out

    def _batch(self, items, rng, cap):
        T = max(min(len(it[0]), cap) if cap else len(it[0]) for it in items)
        B = len(items)
        tok = np.zeros((B, T, 2), np.float32)
        mask = np.zeros((B, T), np.float32)
        lab = np.zeros((B, T), np.float32)
        for i, (t, c, f, *_rest) in enumerate(items):
            idx = np.arange(len(t)) if not cap or len(t) <= cap else rng.choice(len(t), cap, replace=False)
            tok[i, :len(idx)], mask[i, :len(idx)], lab[i, :len(idx)] = t[idx], 1.0, f[idx]
        ctx = np.stack([it[1] for it in items])
        tgt = np.array([[it[3], it[4]] for it in items], np.float32)
        clean = np.array([it[5] for it in items])
        dev = self.device
        return (torch.from_numpy(tok).to(dev), torch.from_numpy(mask).to(dev), torch.from_numpy(ctx).to(dev),
                torch.from_numpy(lab).to(dev), torch.from_numpy(tgt).to(dev), torch.from_numpy(clean).to(dev))

    def _loss(self, batch, scales):
        tok, mask, ctx, lab, tgt, clean = batch
        (mean, var), logits = self.model(tok, mask, ctx)
        lu = ((mean - tgt[:, 0]) / scales.mean) ** 2 + self.lambda_var * ((var - tgt[:, 1]) / scales.var) ** 2
        att, cln = ~clean, clean
        loss = lu[att].mean() if att.any() else lu.sum() * 0
        if self.lambda_clean and cln.any():
            loss = loss + self.lambda_clean * lu[cln].mean()
        if self.lambda_aux:
            bce = nn.functional.binary_cross_entropy_with_logits(logits, lab, reduction='none')
            per_round = (bce * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            loss = loss + self.lambda_aux * per_round.mean()
        return loss, lu

    def fit(self, train, val):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        scales = UtilityScales.from_rounds(train, self.lambda_var)
        self.scales_ = scales
        tr, va = self._prep(train), self._prep(val)
        net = DeepSetsNet if self.arch == 'deepsets' else SetTransformerNet
        self.model = net().to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(tr))
            tot = 0.0
            for i in range(0, len(order), self.batch_rounds):
                batch = self._batch([tr[j] for j in order[i:i + self.batch_rounds]], rng, self.token_cap)
                loss, _ = self._loss(batch, scales)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(batch[0])
            vu = self._val_utility(va, scales, rng)
            self.history.append({'epoch': epoch, 'train_loss': tot / len(tr), 'val_utility': vu})
            if vu < best - 1e-9:
                best, bad = vu, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_utility_ = best
        return self

    def _val_utility(self, va, scales, rng):
        self.model.eval()
        lus = []
        with torch.no_grad():
            for i in range(0, len(va), self.batch_rounds):
                _, lu = self._loss(self._batch(va[i:i + self.batch_rounds], rng, self.token_cap), scales)
                lus.append(lu.cpu().numpy())
        return float(np.mean(np.concatenate(lus))) if lus else np.inf

    def predict(self, obs, labels=None):
        tok, ctx = tokens_and_context(obs)
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(tok).unsqueeze(0).to(self.device)
            m = torch.ones(1, len(tok), device=self.device)
            (mean, var), logits = self.model(t, m, torch.from_numpy(ctx).unsqueeze(0).to(self.device))
        return MethodOutput(float(mean), float(var), scores=torch.sigmoid(logits[0]).cpu().numpy())

    def state(self):
        return {'name': self.name, 'arch': self.arch, 'loss': self.loss_variant,
                'lambda_clean': self.lambda_clean, 'lambda_aux': self.lambda_aux, 'lambda_var': self.lambda_var,
                'scales': self.scales_.state(), 'token_cap_train': self.token_cap, 'inference': 'all reports',
                'history': self.history, 'best_val_utility': self.best_val_utility_, 'seed': self.seed}
