"""Direct recovery with permutation-invariant set models (Deep Sets, Set Transformer).

Input tokens: one per report; the first two columns are always (y / scale, is_g1), scale = s for
PM, 1 for SR. There are no position embeddings. Round context (allowed observation-only values)
starts with log group sizes and the standard decoded mean and variance.

Feature views (`feature_view`, default 'minimal'; names, widths and constants are in state()):
  minimal         tokens (y / scale, is_g1); context (log n_g1, log n_g2, standard mean, variance).
                  The original inputs, computed exactly as before.
  context         minimal tokens. Context adds, per group (g1, g2), the PM report quantiles,
                  one HIST_BINS histogram of y / s, and max / mean / std of the narrowest
                  neighbourhood density; for SR the four within-group sign shares.
  context_report  context, and each PM token adds the report's boundary distance, tie-aware
                  within-group percentile, five neighbourhood densities and three k-NN distances
                  (features.PM_REPORT_NAMES). SR has no further per-report features, so for SR
                  this is an alias of 'context' (state() records the effective view).
Every feature comes from the round's observation (features.py), on the complete round and group,
before any token sampling; training samples rows of the full-round matrix together with their labels.
The first two token columns and the first four context values are never rescaled. Added columns
are z-scored with means / stds fitted on the training rounds only (exact two-pass over every
training report for tokens, one row per training round for context; std < 1e-8 -> 1) and frozen
for validation, test and checkpoints. A model serves one protocol (checked at predict).

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
MSE_before does not depend on the parameters. Auxiliary scores are an output only: they never
remove or weight reports.

Large rounds: training samples at most `token_cap` reports per round per step
(uniformly, without replacement), while the context keeps full-round counts and
moments. Inference uses every report. The approximation is recorded in state().
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

from .. import features as F
from ..estimator import G1, G2
from ..features import standard_moments
from ..mechanisms import get_mechanism
from ..metrics import UtilityScales
from .base import Method, MethodOutput
from .removal import device_for

LOSS_VARIANTS = ('utility', 'utility+clean', 'utility+clean+aux')
FEATURE_VIEWS = ('minimal', 'context', 'context_report')
FEATURE_SCHEMA_VERSION = 1
N_BASE_TOK, N_BASE_CTX = 2, 4
BASE_TOKEN_NAMES = ['report_norm', 'is_g1']
BASE_CONTEXT_NAMES = ['log_n_g1', 'log_n_g2', 'standard_mean', 'standard_variance']
PM_TOKEN_EXTRA = list(F.PM_REPORT_NAMES[1:])                  # y / s is already token column 0
PM_CONTEXT_EXTRA = [f'{tag}_{c}' for tag in ('g1', 'g2') for c in F.PM_CONTEXT_NAMES[1:]]   # no log size
SR_CONTEXT_EXTRA = list(F.SR_NAMES[2:6])
DS_CONFIG = {'width': 64}
ST_CONFIG = {'d': 64, 'heads': 4, 'n_ind': 16, 'n_seeds': 2, 'n_isab': 2}
SCALER_POLICY = ('z-score of added columns only; fitted on training rounds (tokens: exact two-pass over every '
                 'training report; context: one row per training round); std < 1e-8 -> 1; frozen afterwards')


def effective_view(view: str, protocol) -> str:
    if view not in FEATURE_VIEWS:
        raise ValueError(f"feature_view must be one of {FEATURE_VIEWS}, got {view!r}")
    return 'context' if view == 'context_report' and protocol == 'SR' else view


def feature_names(view: str, protocol):
    """Ordered (token names, context names) of a view for one protocol."""
    v = effective_view(view, protocol)
    tok, ctx = list(BASE_TOKEN_NAMES), list(BASE_CONTEXT_NAMES)
    if v != 'minimal':
        if protocol not in ('PM', 'SR'):
            raise ValueError(f"feature view {view!r} needs a protocol (PM or SR), got {protocol!r}")
        ctx += PM_CONTEXT_EXTRA if protocol == 'PM' else SR_CONTEXT_EXTRA
    if v == 'context_report':
        tok += PM_TOKEN_EXTRA
    return tok, ctx


def feature_constants() -> dict:
    return {'QUANTILES': list(F.QUANTILES), 'HIST_BINS': F.HIST_BINS, 'NBR_WIDTHS': list(F.NBR_WIDTHS),
            'KNN': list(F.KNN)}


def tokens_and_context(obs, view: str = 'minimal'):
    """Unscaled (tokens (N, d_tok), context (d_ctx,)), float32, from the observation only."""
    v = effective_view(view, obs['protocol'])
    mech = get_mechanism(obs['protocol'], obs['epsilon'])
    scale = mech.s if obs['protocol'] == 'PM' else 1.0
    y = np.asarray(obs['reports'], np.float32) / scale
    g1 = (np.asarray(obs['groups']) == G1).astype(np.float32)
    mean_n, var_n = standard_moments(obs)
    ctx = np.array([math.log(max(obs['n_g1'], 1)), math.log(max(obs['n_g2'], 1)), mean_n, var_n], np.float32)
    tok = np.column_stack([y, g1])
    if v == 'minimal':
        return tok, ctx
    if obs['protocol'] == 'SR':
        return tok, np.r_[ctx, F.sr_sign_frequencies(obs)].astype(np.float32)
    yy, g = np.asarray(obs['reports'], np.float64), np.asarray(obs['groups'])
    extra_tok = np.zeros((len(yy), len(PM_TOKEN_EXTRA)), np.float32) if v == 'context_report' else None
    extra_ctx = []
    for grp in (G1, G2):
        idx = np.flatnonzero(g == grp)
        rf, gctx = F.pm_group_features(yy[idx], mech.s)       # complete group, before any sampling
        extra_ctx.append(gctx[1:])
        if extra_tok is not None:
            extra_tok[idx] = rf[:, 1:]
    ctx = np.r_[ctx, extra_ctx[0], extra_ctx[1]].astype(np.float32)
    return (tok if extra_tok is None else np.column_stack([tok, extra_tok])), ctx


def _heads(z):
    a, b = z[..., 0], z[..., 1]
    mean = torch.tanh(a)
    var = (1 - mean ** 2) * torch.sigmoid(b)
    return mean, var


class DeepSetsNet(nn.Module):
    """Token column 1 must be the raw 0/1 group indicator: it splits the pooling."""

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
                 feature_view: str = 'minimal', seed: int = 0, device: str = 'auto'):
        if arch not in ('deepsets', 'settransformer'):
            raise ValueError("arch must be 'deepsets' or 'settransformer'")
        if loss not in LOSS_VARIANTS:
            raise ValueError(f"loss must be one of {LOSS_VARIANTS}")
        effective_view(feature_view, None)                     # validates the name
        self.arch, self.loss_variant, self.feature_view = arch, loss, feature_view
        self.name = (f"{arch}_recovery" + ('' if feature_view == 'minimal' else f'_{feature_view}')
                     + ('' if loss == 'utility+clean+aux' else f"[{loss}]"))
        self.lambda_clean = lambda_clean if 'clean' in loss else 0.0
        self.lambda_aux = lambda_aux if 'aux' in loss else 0.0
        self.lambda_var, self.token_cap, self.batch_rounds = lambda_var, token_cap, batch_rounds
        self.epochs, self.lr, self.weight_decay, self.patience = epochs, lr, weight_decay, patience
        self.seed, self.device = seed, device_for(device)
        self.model_config = dict(DS_CONFIG if arch == 'deepsets' else ST_CONFIG)
        self.history = []
        self.protocol_ = None
        self.token_names_, self.context_names_ = feature_names('minimal', None)
        self._set_scaler([], [], [], [])
        self.scaler_rows_ = {'token_rows': 0, 'rounds': 0}
        self.train_tokens_max_ = 0

    # ------------------------------------------------------------ features
    @property
    def d_tok(self):
        return len(self.token_names_)

    @property
    def d_ctx(self):
        return len(self.context_names_)

    def _set_schema(self, protocol):
        if self.feature_view != 'minimal' and protocol not in ('PM', 'SR'):
            raise ValueError(f"{self.name}: training rounds must share one protocol, got {protocol!r}")
        self.protocol_ = protocol
        self.token_names_, self.context_names_ = feature_names(self.feature_view, protocol)

    def _set_scaler(self, tok_mu, tok_sd, ctx_mu, ctx_sd):
        self.tok_mu_, self.tok_sd_ = np.asarray(tok_mu, np.float32), np.asarray(tok_sd, np.float32)
        self.ctx_mu_, self.ctx_sd_ = np.asarray(ctx_mu, np.float32), np.asarray(ctx_sd, np.float32)

    def _fit_scaler(self, items):
        """Train-only z-score of the added columns (see SCALER_POLICY); no sampling, no extra copy."""
        dt, dc = self.d_tok - N_BASE_TOK, self.d_ctx - N_BASE_CTX
        n = sum(len(it[0]) for it in items)
        s = np.zeros(dt)
        for it in items:
            s += it[0][:, N_BASE_TOK:].sum(0, dtype=np.float64)
        tok_mu = s / max(n, 1)
        ss = np.zeros(dt)
        for it in items:
            ss += ((it[0][:, N_BASE_TOK:].astype(np.float64) - tok_mu) ** 2).sum(0)
        tok_sd = np.sqrt(ss / max(n, 1))
        C = np.stack([it[1][N_BASE_CTX:] for it in items]).astype(np.float64) if items else np.zeros((0, dc))
        ctx_mu = C.mean(0) if len(C) else np.zeros(dc)
        ctx_sd = C.std(0) if len(C) else np.ones(dc)
        safe = lambda sd: np.where(sd < 1e-8, 1.0, sd)      # constant column -> centred only  # noqa: E731
        self._set_scaler(tok_mu, safe(tok_sd), ctx_mu, safe(ctx_sd))
        self.scaler_rows_ = {'token_rows': int(n), 'rounds': len(items)}

    def _scale(self, tok, ctx):
        """In place on the added columns; the first two token and four context columns stay raw."""
        if self.d_tok > N_BASE_TOK:
            tok[:, N_BASE_TOK:] = (tok[:, N_BASE_TOK:] - self.tok_mu_) / self.tok_sd_
        if self.d_ctx > N_BASE_CTX:
            ctx[N_BASE_CTX:] = (ctx[N_BASE_CTX:] - self.ctx_mu_) / self.ctx_sd_
        return tok, ctx

    def features(self, obs):
        """Scaled (tokens, context) of one observation, as the network sees them."""
        if self.feature_view != 'minimal' and obs['protocol'] != self.protocol_:
            raise ValueError(f"{self.name} was trained on {self.protocol_} rounds, got {obs['protocol']}")
        tok, ctx = tokens_and_context(obs, self.feature_view)
        if tok.shape[1] != self.d_tok or len(ctx) != self.d_ctx:
            raise ValueError(f"feature width {tok.shape[1]}/{len(ctx)} != schema {self.d_tok}/{self.d_ctx}")
        return self._scale(tok, ctx)

    def _prep(self, rounds):
        out = []
        for r in rounds:
            tok, ctx = tokens_and_context(r.observation, self.feature_view)
            out.append((tok, ctx, r.is_fake.astype(np.float32),
                        np.float32(r.labels['genuine_mean']), np.float32(r.labels['genuine_variance']),
                        bool(r.meta.get('n_fake', 0) == 0)))
        return out

    # ------------------------------------------------------------ training
    def _net(self):
        if self.arch == 'deepsets':
            return DeepSetsNet(d_tok=self.d_tok, d_ctx=self.d_ctx, **self.model_config)
        return SetTransformerNet(d_tok=self.d_tok, d_ctx=self.d_ctx, **self.model_config)

    def _batch(self, items, rng, cap):
        T = max(min(len(it[0]), cap) if cap else len(it[0]) for it in items)
        B = len(items)
        tok = np.zeros((B, T, items[0][0].shape[1]), np.float32)
        mask = np.zeros((B, T), np.float32)
        lab = np.zeros((B, T), np.float32)
        for i, (t, c, f, *_rest) in enumerate(items):
            # rows of the full-round matrices: features are never recomputed on the sample
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
        protocols = sorted({r.observation['protocol'] for r in train})
        self._set_schema(protocols[0] if len(protocols) == 1 else None)
        tr = self._prep(train)
        self._fit_scaler(tr)                                   # training rounds only
        for it in tr:
            self._scale(it[0], it[1])
        va = self._prep(val)
        for it in va:
            self._scale(it[0], it[1])
        self.model = self._net().to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(tr))
            tot = 0.0
            for i in range(0, len(order), self.batch_rounds):
                batch = self._batch([tr[j] for j in order[i:i + self.batch_rounds]], rng, self.token_cap)
                self.train_tokens_max_ = max(self.train_tokens_max_, int(batch[1].sum(1).max()))
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

    # ------------------------------------------------------------ inference
    def predict(self, obs, labels=None):
        t0 = time.perf_counter()
        tok, ctx = self.features(obs)
        t1 = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(tok).unsqueeze(0).to(self.device)
            m = torch.ones(1, len(tok), device=self.device)
            (mean, var), logits = self.model(t, m, torch.from_numpy(ctx).unsqueeze(0).to(self.device))
            mean, var, scores = float(mean), float(var), torch.sigmoid(logits[0]).cpu().numpy()
        t2 = time.perf_counter()
        return MethodOutput(mean, var, scores=scores,
                            info={'t_features': t1 - t0, 't_network': t2 - t1, 'n_tokens': len(tok)})

    # ------------------------------------------------------------ persistence
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters())) if hasattr(self, 'model') else 0

    def state(self):
        return {'name': self.name, 'arch': self.arch, 'loss': self.loss_variant,
                'lambda_clean': self.lambda_clean, 'lambda_aux': self.lambda_aux, 'lambda_var': self.lambda_var,
                'scales': self.scales_.state(), 'token_cap_train': self.token_cap, 'inference': 'all reports',
                'history': self.history, 'best_val_utility': self.best_val_utility_, 'seed': self.seed,
                'batch_rounds': self.batch_rounds, 'epochs': self.epochs, 'lr': self.lr,
                'weight_decay': self.weight_decay, 'patience': self.patience,
                'feature_view': self.feature_view,
                'feature_view_effective': effective_view(self.feature_view, self.protocol_),
                'feature_schema_version': FEATURE_SCHEMA_VERSION, 'protocol': self.protocol_,
                'token_names': self.token_names_, 'context_names': self.context_names_,
                'd_tok': self.d_tok, 'd_ctx': self.d_ctx, 'feature_constants': feature_constants(),
                'scaler': {'policy': SCALER_POLICY, **self.scaler_rows_,
                           'tok_mean': self.tok_mu_.tolist(), 'tok_std': self.tok_sd_.tolist(),
                           'ctx_mean': self.ctx_mu_.tolist(), 'ctx_std': self.ctx_sd_.tolist()},
                'model_config': self.model_config, 'n_params': self.n_params(),
                'train_tokens_max': self.train_tokens_max_}

    def restore_features(self, st: dict):
        """Schema and scaler from a checkpoint state. States without feature metadata (written before
        feature views existed) are 'minimal'; a schema this code cannot rebuild exactly is rejected."""
        ver = st.get('feature_schema_version')
        if ver is not None and ver != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"checkpoint feature schema v{ver}; this code reads v{FEATURE_SCHEMA_VERSION}")
        self._set_schema(st.get('protocol'))
        if 'token_names' in st and (st['token_names'] != self.token_names_
                                    or st['context_names'] != self.context_names_):
            raise ValueError(f"incompatible feature schema for {self.name}: checkpoint "
                             f"{st['token_names']} / {st['context_names']}, code {self.token_names_} / "
                             f"{self.context_names_}")
        if 'feature_constants' in st and st['feature_constants'] != feature_constants():
            raise ValueError(f"feature constants changed since this checkpoint: {st['feature_constants']} "
                             f"vs {feature_constants()}")
        sc = st.get('scaler', {})
        self._set_scaler(sc.get('tok_mean', []), sc.get('tok_std', []), sc.get('ctx_mean', []),
                         sc.get('ctx_std', []))
        if len(self.tok_mu_) != self.d_tok - N_BASE_TOK or len(self.ctx_mu_) != self.d_ctx - N_BASE_CTX:
            raise ValueError(f"scaler widths {len(self.tok_mu_)}/{len(self.ctx_mu_)} do not match the schema")
        self.scaler_rows_ = {k: sc[k] for k in ('token_rows', 'rounds') if k in sc} or self.scaler_rows_
        self.model_config = st.get('model_config', self.model_config)
        self.train_tokens_max_ = st.get('train_tokens_max', 0)
