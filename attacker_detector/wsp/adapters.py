"""One adapter per target model. An adapter owns the network and the only data Stage 2 may touch:

  probe split   a fixed subset of TRAINING rounds / rows (what the agent's probes run on)
  val split     validation rounds / rows (what the gate scores)

Test data is never loaded here (frequency oracle: dropped right after the split is rebuilt).
evaluate() refuses any other split name.

Targets
  mv_ft    ClassifierRemoval(arch='ft')        mean/variance, per-report FT-Transformer + removal
  mv_ds    SetRecovery(arch='deepsets')        mean/variance, direct recovery
  mv_st    SetRecovery(arch='settransformer')  mean/variance, direct recovery
  freq_ft  FTTransformer via main.py           frequency oracle, per-user attacker detection
"""

import math
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from .cost import forward_flops
from .gate import Score

SPLITS = ('probe', 'val')


def _check_split(split):
    if split not in SPLITS:
        raise ValueError(f"split {split!r} is not available in Stage 2 (only {SPLITS})")


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class Adapter:
    target = ''

    def __init__(self, model: nn.Module, device):
        self.model = model
        self.device = torch.device(device)
        self.flops_per_eval = {}
        self.counters = {'compute_s': 0.0, 'train_steps': 0, 'train_flops': 0}

    # ------------------------------------------------------------ weights
    def weights(self):
        return {k: v.detach().cpu().numpy().astype(np.float32).copy()
                for k, v in self.model.state_dict().items() if v.dtype.is_floating_point}

    def load_weights(self, W):
        sd = self.model.state_dict()
        with torch.no_grad():
            for k, v in W.items():
                sd[k].copy_(torch.from_numpy(np.asarray(v, np.float32)).to(sd[k].device, sd[k].dtype))

    @contextmanager
    def using(self, W):
        keep = self.weights()
        self.load_weights(W)
        try:
            yield
        finally:
            self.load_weights(keep)

    def param_table(self):
        return [{'name': k, 'shape': list(v.shape), 'numel': int(v.numel())}
                for k, v in self.model.state_dict().items() if v.dtype.is_floating_point]

    def n_params(self):
        return int(sum(r['numel'] for r in self.param_table()))

    # ------------------------------------------------------------ evaluation
    def evaluate(self, split) -> Score:
        _check_split(split)
        t0 = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            s = self._evaluate(split)
        _sync(self.device)
        self.counters['compute_s'] += time.perf_counter() - t0
        return s

    def eval_flops(self, split):
        """FLOPs of one evaluate(split), counted once with FlopCounterMode and cached."""
        if split not in self.flops_per_eval:
            self.model.eval()
            params = list(self.model.parameters())
            req = [p.requires_grad for p in params]
            for p in params:                    # views of grad-requiring params trip torch's module tracker
                p.requires_grad_(False)
            try:
                with torch.no_grad():
                    self.flops_per_eval[split] = forward_flops(lambda: self._forward_all(split))
            finally:
                for p, r in zip(params, req):
                    p.requires_grad_(r)
        return self.flops_per_eval[split]

    def activations(self, split='probe', max_units=6):
        """Per Linear / LayerNorm module: output width, mean |activation|, never-positive units,
        and the most active units, over the probe split."""
        _check_split(split)
        stats = {}

        def hook(name):
            def fn(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                o = o.detach().float().reshape(-1, o.shape[-1])
                s = stats.setdefault(name, {'n': 0, 'sum': 0, 'abs': 0, 'sq': 0, 'pos': 0, 'max': None})
                s['n'] += o.shape[0]
                s['sum'] = s['sum'] + o.sum(0)
                s['abs'] = s['abs'] + o.abs().sum(0)
                s['sq'] = s['sq'] + (o ** 2).sum(0)
                s['pos'] = s['pos'] + (o > 0).sum(0)
                m = o.max(0).values
                s['max'] = m if s['max'] is None else torch.maximum(s['max'], m)
            return fn

        hs = [m.register_forward_hook(hook(n)) for n, m in self.model.named_modules()
              if isinstance(m, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention))]
        try:
            self.model.eval()
            with torch.no_grad():
                self._forward_all(split)
        finally:
            for h in hs:
                h.remove()
        out = []
        for name, s in stats.items():
            n = max(s['n'], 1)
            mean, mabs = (s['sum'] / n).cpu().numpy(), (s['abs'] / n).cpu().numpy()
            std = np.sqrt(np.maximum((s['sq'] / n).cpu().numpy() - mean ** 2, 0))
            top = np.argsort(-mabs)[:max_units]
            out.append({'module': name, 'width': int(len(mean)), 'mean_abs': round(float(mabs.mean()), 5),
                        'mean_std': round(float(std.mean()), 5),
                        'never_positive': int(((s['pos'] == 0).cpu().numpy()).sum()),
                        'near_constant': int((std < 1e-6).sum()),
                        'top_units': [[int(i), round(float(mean[i]), 4), round(float(std[i]), 4)] for i in top]})
        return out

    # ------------------------------------------------------------ to implement
    def describe(self) -> dict:
        raise NotImplementedError

    def _evaluate(self, split) -> Score:
        raise NotImplementedError

    def _forward_all(self, split):
        raise NotImplementedError

    def probe(self, kind, split='probe') -> dict:
        raise NotImplementedError

    def on_accept(self, score: Score):
        """Adapter state that follows an accepted edit (e.g. a re-picked threshold)."""

    def finetune(self, steps, lr_scale=0.1):
        raise NotImplementedError

    def epoch_steps(self) -> int:
        raise NotImplementedError

    def save(self, out_dir):
        raise NotImplementedError

    def stage1_cost(self) -> dict:
        return {}


def _round_slices(meta, values, keys=('status', 'attack', 'beta_bin')):
    """Group-wise means of `values` (dict of arrays) over round metadata."""
    rows = []
    for key in keys:
        col = np.asarray([m.get(key) for m in meta], dtype=object)
        for v in sorted(set(col), key=str):
            idx = np.flatnonzero(col == v)
            r = {'slice': f'{key}={v}', 'n': int(len(idx))}
            for name, arr in values.items():
                r[name] = round(float(np.mean(arr[idx])), 6)
            rows.append(r)
    return rows


def _beta_bin(beta):
    if beta is None or not np.isfinite(beta) or beta == 0:
        return '0'
    return '<=0.1' if beta <= 0.10 else ('<=0.4' if beta <= 0.40 else '>0.4')


# ====================================================================== mean / variance: context

class MVContext:
    """Round store, manifest and the configured round ids of the Stage-1 experiment (shared by runs)."""

    def __init__(self, config_path, results_root):
        from ..mean_variance.experiment import enumerate_specs, load_config, make_banks
        from ..mean_variance.store import RoundStore
        self.cfg = load_config(config_path)
        self.out_dir = os.path.join(results_root, self.cfg['experiment'])
        self.store = RoundStore(os.path.join(self.out_dir, 'rounds'))
        self.manifest = self.store.manifest()
        banks = make_banks(self.cfg)
        self.allowed = {s.round_id for split in ('train', 'val', 'test')
                        for s in enumerate_specs(self.cfg, split, banks)}

    def rounds(self, split, protocol, eps, sources):
        from ..mean_variance.evaluate import rounds_from_store
        return rounds_from_store(self.store, self.manifest, split, protocol, float(eps), list(sources),
                                 self.allowed)

    def checkpoint(self, key):
        return os.path.join(self.out_dir, 'checkpoints', key)

    def fit_time(self, key):
        p = os.path.join(self.out_dir, 'per_round', key + '.csv')
        if not os.path.exists(p):
            return None
        import pandas as pd
        d = pd.read_csv(p, usecols=lambda c: c == 'fit_time', nrows=1)
        return float(d['fit_time'].iloc[0]) if 'fit_time' in d and len(d) else None


def _round_meta(ctx_manifest, r):
    rec = ctx_manifest.get(r.round_id, {}) if ctx_manifest else {}
    beta = rec.get('beta', r.meta.get('realized_beta'))
    clean = r.meta.get('n_fake', 0) == 0
    return {'round_id': r.round_id, 'status': 'clean' if clean else 'attacked',
            'attack': rec.get('attack', 'none') if not clean else 'none', 'beta': beta,
            'beta_bin': _beta_bin(beta if beta is not None else 0.0), 'target': rec.get('target'),
            'n_reports': int(len(r.observation['reports'])), 'realized_beta': r.meta.get('realized_beta'),
            'clean': clean}


def _probe_subset(rounds, n, seed):
    """Fixed, stratified (clean / attacked) subset of training rounds for probes."""
    rng = np.random.default_rng(seed)
    clean = [i for i, r in enumerate(rounds) if r.meta.get('n_fake', 0) == 0]
    att = [i for i, r in enumerate(rounds) if r.meta.get('n_fake', 0) != 0]
    n_c = min(len(clean), max(1, int(round(n * len(clean) / max(len(rounds), 1))))) if clean else 0
    pick = list(rng.choice(clean, n_c, replace=False)) if n_c else []
    pick += list(rng.choice(att, min(len(att), n - len(pick)), replace=False)) if att else []
    return [rounds[i] for i in sorted(pick)]


# ====================================================================== mean / variance: set models

class SetRecoveryAdapter(Adapter):
    """Deep Sets / Set Transformer recovery. Loss per round = the method's own utility loss
    ((mean_hat - mean)/s_m)^2 + lambda_var ((var_hat - var)/s_v)^2 on every report of the round."""

    def __init__(self, method, train, val, manifest=None, probe_rounds=128, seed=0, batch_rounds=16):
        super().__init__(method.model, method.device)
        self.method = method
        self.target = 'mv_ds' if method.arch == 'deepsets' else 'mv_st'
        self.batch_rounds = batch_rounds
        self.train = train
        self.data = {}
        for split, rounds in (('probe', _probe_subset(train, probe_rounds, seed)), ('val', val)):
            self.data[split] = self._prep(rounds, manifest)
        self._ft = None

    def _prep(self, rounds, manifest):
        from ..mean_variance.methods.baselines import StandardMethod
        std = StandardMethod()
        items, meta, std_out = [], [], []
        for r in rounds:
            tok, ctx = self.method.features(r.observation)
            items.append((tok, ctx, r.is_fake.astype(np.float32), np.float32(r.labels['genuine_mean']),
                          np.float32(r.labels['genuine_variance']), bool(r.meta.get('n_fake', 0) == 0)))
            meta.append(_round_meta(manifest, r))
            o = std.predict(r.observation)
            std_out.append((o.mean, o.var))
        gm = np.array([it[3] for it in items], float)
        gv = np.array([it[4] for it in items], float)
        so = np.array(std_out, float)
        return {'items': items, 'meta': meta, 'gm': gm, 'gv': gv, 'clean': np.array([it[5] for it in items]),
                'std_mean': so[:, 0], 'std_var': so[:, 1], 'std_loss': self._lu(so[:, 0], so[:, 1], gm, gv)}

    def _lu(self, mean, var, gm, gv):
        s = self.method.scales_
        return ((mean - gm) / s.mean) ** 2 + self.method.lambda_var * ((var - gv) / s.var) ** 2

    def _forward_all(self, split):
        d = self.data[split]
        rng = np.random.default_rng(0)
        means, vars_ = [], []
        items = d['items']
        for i in range(0, len(items), self.batch_rounds):
            tok, mask, ctx, *_ = self.method._batch(items[i:i + self.batch_rounds], rng, None)
            (mean, var), _logits = self.model(tok, mask, ctx)
            means.append(mean.float().cpu().numpy())
            vars_.append(var.float().cpu().numpy())
        return np.concatenate(means), np.concatenate(vars_)

    def _evaluate(self, split):
        d = self.data[split]
        mean, var = self._forward_all(split)
        lu = self._lu(mean, var, d['gm'], d['gv'])
        cl = d['clean']
        guards, report = {}, {'loss': float(lu.mean()), 'standard_loss': float(d['std_loss'].mean()),
                              'n_rounds': int(len(lu)),
                              'bias_mean': float(np.mean(mean - d['gm'])), 'bias_var': float(np.mean(var - d['gv']))}
        for name, m in (('clean', cl), ('attacked', ~cl)):
            if m.any():
                guards[f'loss_{name}'] = float(lu[m].mean())
                report[f'loss_{name}'] = guards[f'loss_{name}']
                report[f'standard_loss_{name}'] = float(d['std_loss'][m].mean())
        return Score(float(lu.mean()), units=lu, guards=guards, report=report,
                     extra={'mean': mean, 'var': var})

    def probe(self, kind, split='probe'):
        d = self.data[split]
        if kind == 'activations':
            return {'activations': self.activations(split)}
        s = self.evaluate(split)
        mean, var, lu = s.extra['mean'], s.extra['var'], s.units
        if kind == 'metrics':
            vals = {'loss': lu, 'standard_loss': d['std_loss'], 'err_mean': mean - d['gm'],
                    'err_var': var - d['gv'], 'std_err_mean': d['std_mean'] - d['gm'],
                    'std_err_var': d['std_var'] - d['gv']}
            return {'overall': s.report, 'slices': _round_slices(d['meta'], vals)}
        if kind == 'errors':
            worst = np.argsort(-lu)[:10]
            rows = []
            for i in worst:
                m = d['meta'][i]
                rows.append({'loss': round(float(lu[i]), 4), 'status': m['status'], 'attack': m['attack'],
                             'beta': m['beta'], 'target': m['target'], 'n_reports': m['n_reports'],
                             'true_mean': round(float(d['gm'][i]), 4), 'pred_mean': round(float(mean[i]), 4),
                             'std_mean': round(float(d['std_mean'][i]), 4),
                             'true_var': round(float(d['gv'][i]), 4), 'pred_var': round(float(var[i]), 4),
                             'std_var': round(float(d['std_var'][i]), 4)})
            return {'overall': s.report, 'worst_rounds': rows}
        raise ValueError(f"unknown probe kind {kind!r}")

    def describe(self):
        m = self.method
        arch = ('Deep Sets: phi = 3-layer ReLU MLP per report token (width 64); mean-pool phi over group-1 and '
                'group-2 reports separately; rho = MLP on [pool_g1, pool_g2, context] -> 2 logits (a, b). '
                'aux = MLP on [token phi, own-group pool, context] -> per-report fake logit (auxiliary output only).'
                if m.arch == 'deepsets' else
                'Set Transformer: embed Linear(d_tok, 64); 2 ISAB blocks (16 inducing points, 4 heads, post-LN MABs '
                'with ReLU FFN); PMA with 2 seeds; head MLP on [pooled seeds, context] -> 2 logits (a, b); aux MLP '
                'per report -> fake logit (auxiliary output only).')
        return {
            'target': self.target,
            'task': ('LDP mean/variance estimation under data poisoning. Each round is a set of user reports from '
                     f"the {m.protocol_ or self.train[0].observation['protocol']} mechanism (users split into group 1 for the mean and group 2 for the "
                     'second moment). Some rounds contain fake reports from IPA/OPA attackers who push the estimate '
                     'toward a target; clean rounds have none. The network reads the whole round and outputs the '
                     'genuine mean and variance on the normalized scale [-1, 1].'),
            'architecture': arch,
            'inputs': {'token_features': m.token_names_, 'context_features': m.context_names_,
                       'note': 'token col 0 = report / scale, col 1 = group-1 indicator (splits the pooling); context '
                               'cols 2-3 = standard (no-defence) mean and variance of the round'},
            'outputs': 'mean = tanh(a); variance = (1 - mean^2) * sigmoid(b)',
            'metric': ('per-round loss ((mean_hat - mean)/%.4g)^2 + %.3g ((var_hat - var)/%.4g)^2; lower is better. '
                       'standard_loss = same loss of the no-defence estimate on the same rounds.'
                       % (m.scales_.mean, m.lambda_var, m.scales_.var)),
            'gate_guards': 'clean-round loss and attacked-round loss must each not get worse by more than the tolerance',
            'params': self.param_table(), 'n_params': self.n_params(),
            'splits': {'probe': f"{len(self.data['probe']['items'])} training rounds",
                       'val': f"{len(self.data['val']['items'])} validation rounds"},
        }

    # ------------------------------------------------------------ gradient baseline
    def _ft_state(self, lr_scale):
        if self._ft is None:
            m = self.method
            items = m._prep(self.train)
            for it in items:
                m._scale(it[0], it[1])
            opt = torch.optim.AdamW(self.model.parameters(), lr=m.lr * lr_scale, weight_decay=m.weight_decay)
            self._ft = {'items': items, 'opt': opt, 'rng': np.random.default_rng(m.seed + 1000), 'order': [],
                        'flops_step': None}
        return self._ft

    def epoch_steps(self):
        return max(1, math.ceil(len(self.train) / self.method.batch_rounds))

    def finetune(self, steps, lr_scale=0.1):
        st, m = self._ft_state(lr_scale), self.method
        t0 = time.perf_counter()
        self.model.train()
        for _ in range(steps):
            if len(st['order']) < m.batch_rounds:
                st['order'] = list(st['rng'].permutation(len(st['items'])))
            idx, st['order'] = st['order'][:m.batch_rounds], st['order'][m.batch_rounds:]
            batch = m._batch([st['items'][j] for j in idx], st['rng'], m.token_cap)

            def step():
                loss, _ = m._loss(batch, m.scales_)
                st['opt'].zero_grad()
                loss.backward()
                return loss
            if st['flops_step'] is None:
                st['flops_step'] = forward_flops(step)
            else:
                step()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            st['opt'].step()
            self.counters['train_steps'] += 1
            self.counters['train_flops'] += st['flops_step']
        self.model.eval()
        _sync(self.device)
        self.counters['compute_s'] += time.perf_counter() - t0

    def save(self, out_dir):
        from ..mean_variance.methods.checkpoint import save_checkpoint
        save_checkpoint(self.method, out_dir)

    def stage1_cost(self):
        m = self.method
        epochs = len(m.history)
        steps = epochs * self.epoch_steps()
        items = m._prep(self.train[:m.batch_rounds])
        for it in items:
            m._scale(it[0], it[1])
        batch = m._batch(items, np.random.default_rng(0), m.token_cap)
        keep = self.weights()

        def step():
            self.model.train()
            loss, _ = m._loss(batch, m.scales_)
            loss.backward()
        f = forward_flops(step)
        self.model.zero_grad(set_to_none=True)
        self.load_weights(keep)
        self.model.eval()
        return {'epochs_run': epochs, 'optimizer_steps': steps, 'train_flops': int(f * steps)}


# ====================================================================== mean / variance: FT removal

class FTRemovalAdapter(Adapter):
    """Per-report FT-Transformer, then removal at a threshold re-picked on validation rounds (the
    method's own rule) for every scored weight set. Loss per round = post-removal utility loss."""

    target = 'mv_ft'

    def __init__(self, method, train, val, manifest=None, probe_rounds=128, seed=0):
        super().__init__(method.model, method.device)
        from ..mean_variance.metrics import UtilityScales
        self.method = method
        sc = method.scales_
        self.scales = UtilityScales(sc['scale_mean'], sc['scale_var'], sc.get('lambda_var', 1.0)) \
            if isinstance(sc, dict) else sc
        self.train = train
        self.data = {s: self._prep(r, manifest) for s, r in
                     (('probe', _probe_subset(train, probe_rounds, seed)), ('val', val))}
        self._ft = None

    def _prep(self, rounds, manifest):
        from ..mean_variance.features import report_features
        from ..mean_variance.methods.baselines import StandardMethod
        std = StandardMethod()
        Z, off, meta, fake, so = [], [0], [], [], []
        for r in rounds:
            F, _ = report_features(r.observation, self.method.feature_set)
            Z.append(((F - self.method.mu_) / self.method.sd_).astype(np.float32))
            off.append(off[-1] + len(F))
            fake.append(r.is_fake.astype(np.float32))
            meta.append(_round_meta(manifest, r))
            o = std.predict(r.observation)
            so.append((o.mean, o.var))
        gm = np.array([r.labels['genuine_mean'] for r in rounds], float)
        gv = np.array([r.labels['genuine_variance'] for r in rounds], float)
        so = np.array(so, float)
        return {'rounds': rounds, 'Z': torch.from_numpy(np.concatenate(Z)), 'off': np.array(off),
                'fake': np.concatenate(fake), 'meta': meta, 'gm': gm, 'gv': gv,
                'clean': np.array([m['clean'] for m in meta]), 'std_mean': so[:, 0], 'std_var': so[:, 1],
                'std_loss': self.scales.loss(so[:, 0] - gm, so[:, 1] - gv)}

    def _forward_all(self, split):
        Z = self.data[split]['Z']
        out = []
        for i in range(0, len(Z), 8192):
            out.append(self.model(Z[i:i + 8192].to(self.device)).squeeze(-1).float().cpu())
        return torch.cat(out).numpy() if out else np.empty(0)

    def _losses_at(self, d, scores, t):
        L, M, V, R = [], [], [], []
        for i, r in enumerate(d['rounds']):
            s = scores[d['off'][i]:d['off'][i + 1]]
            o = self.method._aggregate(r.observation, s, t)
            L.append(float(self.scales.loss(o.mean - d['gm'][i], o.var - d['gv'][i])))
            M.append(o.mean)
            V.append(o.var)
            R.append(o.info['removed_frac'])
        return np.array(L), np.array(M), np.array(V), np.array(R)

    def _evaluate(self, split, threshold=None):
        from sklearn.metrics import average_precision_score
        from ..mean_variance.methods.removal import THRESHOLDS
        d = self.data[split]
        logits = self._forward_all(split)
        scores = 1 / (1 + np.exp(-logits))
        y = d['fake']
        eps = 1e-7
        bce_row = -(y * np.log(scores + eps) + (1 - y) * np.log(1 - scores + eps))
        w = np.concatenate([np.full(d['off'][i + 1] - d['off'][i], 1.0 / max(d['off'][i + 1] - d['off'][i], 1))
                            for i in range(len(d['rounds']))])
        bce = float((bce_row * w).sum() / max(w.sum(), 1e-12))
        if threshold is None:                     # the method's rule: best mean utility over the grid
            res = {float(t): self._losses_at(d, scores, t) for t in THRESHOLDS}
            threshold = min(res, key=lambda t: res[t][0].mean())
            best, grid = res[threshold], {t: float(v[0].mean()) for t, v in res.items()}
        else:
            best, grid = self._losses_at(d, scores, threshold), None
        lu, mean, var, rem = best
        cl = d['clean']
        report = {'loss': float(lu.mean()), 'standard_loss': float(d['std_loss'].mean()), 'threshold': threshold,
                  'removed_frac': float(rem.mean()), 'bce': bce, 'n_rounds': int(len(lu)),
                  'pr_auc': float(average_precision_score(y, scores)) if 0 < y.sum() < len(y) else None}
        guards = {'bce': bce}
        for name, m in (('clean', cl), ('attacked', ~cl)):
            if m.any():
                report[f'loss_{name}'] = float(lu[m].mean())
                report[f'standard_loss_{name}'] = float(d['std_loss'][m].mean())
        return Score(float(lu.mean()), units=lu, guards=guards, report=report,
                     extra={'threshold': threshold, 'grid': grid, 'scores': scores, 'mean': mean, 'var': var,
                            'removed': rem})

    def on_accept(self, score):
        self.method.threshold = score.extra['threshold']
        if score.extra.get('grid'):
            self.method.val_utility_ = score.extra['grid']

    def probe(self, kind, split='probe'):
        if kind == 'activations':
            return {'activations': self.activations(split)}
        d = self.data[split]
        s = self.evaluate(split)
        sc, lu = s.extra['scores'], s.units
        if kind == 'metrics':
            per_round_fake = np.array([sc[d['off'][i]:d['off'][i + 1]][d['fake'][d['off'][i]:d['off'][i + 1]] > 0]
                                       .mean() if d['fake'][d['off'][i]:d['off'][i + 1]].any() else np.nan
                                       for i in range(len(d['rounds']))])
            per_round_gen = np.array([sc[d['off'][i]:d['off'][i + 1]][d['fake'][d['off'][i]:d['off'][i + 1]] == 0]
                                      .mean() for i in range(len(d['rounds']))])
            vals = {'loss': lu, 'standard_loss': d['std_loss'], 'removed_frac': s.extra['removed'],
                    'score_genuine': per_round_gen, 'score_fake': np.nan_to_num(per_round_fake, nan=-1)}
            return {'overall': s.report, 'slices': _round_slices(d['meta'], vals),
                    'note': 'score_fake = -1 where a slice has no fakes'}
        if kind == 'errors':
            worst = np.argsort(-lu)[:10]
            rows = []
            for i in worst:
                m = d['meta'][i]
                f = d['fake'][d['off'][i]:d['off'][i + 1]]
                s_i = sc[d['off'][i]:d['off'][i + 1]]
                rows.append({'loss': round(float(lu[i]), 4), 'status': m['status'], 'attack': m['attack'],
                             'beta': m['beta'], 'removed_frac': round(float(s.extra['removed'][i]), 3),
                             'fake_share': round(float(f.mean()), 3),
                             'mean_score_fake': round(float(s_i[f > 0].mean()), 3) if f.any() else None,
                             'mean_score_genuine': round(float(s_i[f == 0].mean()), 3),
                             'true_mean': round(float(d['gm'][i]), 4), 'est_mean': round(float(s.extra['mean'][i]), 4),
                             'std_mean': round(float(d['std_mean'][i]), 4)})
            return {'overall': s.report, 'worst_rounds': rows}
        raise ValueError(f"unknown probe kind {kind!r}")

    def describe(self):
        from ..mean_variance.features import report_features
        r = self.data['val']['rounds'][0]
        _, names = report_features(r.observation, self.method.feature_set)
        return {
            'target': self.target,
            'task': ('LDP mean/variance estimation under data poisoning. A per-report classifier scores each report '
                     f"of a {r.observation['protocol']} round as fake; reports above a threshold are removed and the "
                     'mean/variance are re-estimated from the rest. The threshold is re-picked on validation rounds '
                     'for every weight set (grid 0.05..0.95, or 1.01 = remove nothing).'),
            'architecture': ('FT-Transformer: FeatureTokenizer (x_i * w_i + b_i per feature, d_token 32) + CLS token, '
                             '2 pre-norm blocks (4-head self-attention over feature tokens, ReGLU FFN), head = '
                             'LayerNorm -> ReLU -> Linear(32, 1) on the CLS token -> fake logit.'),
            'inputs': {'report_features (z-scored with training mean/std)': list(names)},
            'outputs': 'one logit per report (sigmoid = fake probability)',
            'metric': 'per-round post-removal utility loss (lower is better); standard_loss = no defence',
            'gate_guards': 'round-weighted validation BCE must not get worse by more than the tolerance',
            'params': self.param_table(), 'n_params': self.n_params(),
            'splits': {'probe': f"{len(self.data['probe']['rounds'])} training rounds",
                       'val': f"{len(self.data['val']['rounds'])} validation rounds"},
        }

    def _train_rows(self):
        if getattr(self, '_rows_cache', None) is None:
            m = self.method
            X, y, w = m._rows(self.train, np.random.default_rng(m.seed + 1000), m.reports_per_round)
            self._rows_cache = (torch.from_numpy(((X - m.mu_) / m.sd_).astype(np.float32)),
                                torch.from_numpy(y), torch.from_numpy(w))
        return self._rows_cache

    def _ft_state(self, lr_scale):
        if self._ft is None:
            m = self.method
            X, y, w = self._train_rows()
            opt = torch.optim.AdamW(self.model.parameters(), lr=m.lr * lr_scale, weight_decay=m.weight_decay)
            self._ft = {'X': X, 'y': y, 'w': w, 'opt': opt,
                        'g': torch.Generator().manual_seed(m.seed + 1000), 'perm': None, 'pos': 0, 'flops_step': None}
        return self._ft

    def epoch_steps(self):
        return max(1, math.ceil(len(self._train_rows()[0]) / self.method.batch_size))

    def finetune(self, steps, lr_scale=0.1):
        st, m = self._ft_state(lr_scale), self.method
        bce = nn.BCEWithLogitsLoss(reduction='none')
        t0 = time.perf_counter()
        self.model.train()
        for _ in range(steps):
            if st['perm'] is None or st['pos'] >= len(st['perm']):
                st['perm'], st['pos'] = torch.randperm(len(st['X']), generator=st['g']), 0
            b = st['perm'][st['pos']:st['pos'] + m.batch_size]
            st['pos'] += m.batch_size
            xb, yb, wb = st['X'][b].to(self.device), st['y'][b].to(self.device), st['w'][b].to(self.device)

            def step():
                loss = (bce(self.model(xb).squeeze(-1), yb) * wb).sum() / wb.sum()
                st['opt'].zero_grad()
                loss.backward()
            if st['flops_step'] is None:
                st['flops_step'] = forward_flops(step)
            else:
                step()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            st['opt'].step()
            self.counters['train_steps'] += 1
            self.counters['train_flops'] += st['flops_step']
        self.model.eval()
        _sync(self.device)
        self.counters['compute_s'] += time.perf_counter() - t0

    def save(self, out_dir):
        from ..mean_variance.methods.checkpoint import save_checkpoint
        save_checkpoint(self.method, out_dir)

    def stage1_cost(self):
        m = self.method
        epochs = len(m.history)
        steps = epochs * self.epoch_steps()
        xb = self._train_rows()[0][:m.batch_size].to(self.device)
        keep = self.weights()

        def step():
            self.model.train()
            self.model(xb).sum().backward()
        f = forward_flops(step)
        self.model.zero_grad(set_to_none=True)
        self.load_weights(keep)
        self.model.eval()
        return {'epochs_run': epochs, 'optimizer_steps': steps, 'train_flops': int(f * steps)}


# ====================================================================== frequency oracle

FT_ARG_KEYS = ('d_token', 'n_heads', 'n_layers', 'ffn_d_multiplier', 'attention_dropout', 'residual_dropout')


def build_freq_split(spec):
    """Rebuild main.py's data path for one run exactly (same functions, same order, same seed).
    Returns (ds, split)."""
    from ..data import load_npy_dataset, prepare_npy_data_oat
    from ..data.dataset import apply_feature_set_v3
    ds = load_npy_dataset(spec['data'])
    if spec.get('feature_set', 'stored') == 'v3':
        apply_feature_set_v3(ds)
    idx = np.arange(len(ds))
    if spec.get('protocol'):
        idx = idx[ds.protocols[idx] == spec['protocol']]
    if spec.get('dataset'):
        idx = idx[ds.dataset_types[idx] == spec['dataset']]
    if len(idx) < len(ds):
        ds.features, ds.labels, ds.config = ds.features[idx], ds.labels[idx], ds.config[idx]
    train_mask = test_mask = None
    if spec.get('training_method') == 'cross':
        dt = ds.dataset_types
        train_mask = np.isin(dt, spec['train_dataset'])
        test_mask = dt == spec['test_dataset']
    split = prepare_npy_data_oat(ds, train_samples=spec['train_samples'], test_samples=spec['test_samples'],
                                 val_size=spec['val_size'], random_state=spec['seed'],
                                 train_mask=train_mask, test_mask=test_mask)
    return ds, split


def freq_arrays(ds, split, cache_path=None):
    """Train / val arrays (no test rows) of a rebuilt split; optionally cached as .npz for Stage 2."""
    arrays = {'X_train': split['X_train'].astype(np.float32), 'y_train': split['y_train'].astype(np.float32),
              'X_val': split['X_val'].astype(np.float32), 'y_val': split['y_val'].astype(np.float32),
              'config_train': ds.config[split['train_indices']].astype(str),
              'config_val': ds.config[split['val_indices']].astype(str),
              'feature_names': np.array(ds.feature_names, dtype=str)}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path + '.tmp.npz', **arrays)
        os.replace(cache_path + '.tmp.npz', cache_path)
    return arrays


def build_freq_model(spec, input_dim, state_dict=None, device='cpu'):
    from ..models.ft_transformer import FTTransformer
    kw = {k: spec[k] for k in FT_ARG_KEYS if spec.get(k) is not None}
    model = FTTransformer(input_dim, dropout_rate=spec.get('dropout', 0.2), **kw)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_freq_state(model_path):
    ck = torch.load(model_path, map_location='cpu')
    return ck['model_state_dict'] if 'model_state_dict' in ck else ck


class FreqFTAdapter(Adapter):
    """Frequency-oracle FT-Transformer. Gate metric = validation F1 at 0.5 (the reported metric);
    guards = validation BCE and 1 - PR-AUC."""

    target = 'freq_ft'

    def __init__(self, model, spec, arrays, device, probe_rows=20000, seed=0):
        super().__init__(model, device)
        from ..data.dataset import (_CONF_ATTACKER_RATIO, _CONF_DATASET_TYPE, _CONF_EPSILON, _CONF_SPLITS,
                                    _CONF_TARGET_SET_SIZE)
        self.conf_cols = {'dataset': _CONF_DATASET_TYPE, 'epsilon': _CONF_EPSILON,
                          'attacker_ratio': _CONF_ATTACKER_RATIO, 'target_set_size': _CONF_TARGET_SET_SIZE,
                          'splits': _CONF_SPLITS}
        self.spec = spec
        self.feature_names = arrays.get('feature_names', [f'f{i}' for i in range(arrays['X_train'].shape[1])])
        rng = np.random.default_rng(seed)
        n = len(arrays['X_train'])
        pi = np.sort(rng.choice(n, min(probe_rows, n), replace=False))
        self.train = (arrays['X_train'], arrays['y_train'])
        self.data = {'probe': (arrays['X_train'][pi], arrays['y_train'][pi], arrays['config_train'][pi]),
                     'val': (arrays['X_val'], arrays['y_val'], arrays['config_val'])}
        self._ft = None

    @classmethod
    def from_spec(cls, spec, model_path, device, cache_path=None, probe_rows=20000, seed=0):
        """Train / val arrays only (cached as .npz); test rows are dropped here."""
        if cache_path and os.path.exists(cache_path):
            with np.load(cache_path, allow_pickle=True) as z:
                arrays = {k: z[k] for k in z.files}
            arrays['feature_names'] = list(arrays['feature_names'])
        else:
            ds, split = build_freq_split(spec)
            arrays = freq_arrays(ds, split, cache_path)
            del ds, split
            arrays['feature_names'] = list(arrays['feature_names'])
        model = build_freq_model(spec, arrays['X_train'].shape[1], load_freq_state(model_path), device)
        return cls(model, spec, arrays, device, probe_rows, seed)

    def _forward_all(self, split):
        X = self.data[split][0]
        out = []
        for i in range(0, len(X), 8192):
            out.append(self.model(torch.from_numpy(X[i:i + 8192]).to(self.device)).squeeze(-1).float().cpu())
        return torch.cat(out).numpy()

    def _evaluate(self, split):
        from sklearn.metrics import average_precision_score
        X, y, _ = self.data[split]
        logits = self._forward_all(split)
        p = 1 / (1 + np.exp(-logits))
        pred = (p > 0.5).astype(np.int8)
        yi = y.astype(np.int8)
        tp, fp = int(((pred == 1) & (yi == 1)).sum()), int(((pred == 1) & (yi == 0)).sum())
        fn = int(((pred == 0) & (yi == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        eps = 1e-7
        bce = float(np.mean(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
        prauc = float(average_precision_score(yi, p)) if 0 < yi.sum() < len(yi) else 1.0
        report = {'f1': f1, 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
                  'accuracy': float((pred == yi).mean()), 'pr_auc': prauc, 'bce': bce, 'n_rows': int(len(y))}
        return Score(1 - f1, guards={'bce': bce, 'one_minus_pr_auc': 1 - prauc}, report=report, y=yi, pred=pred,
                     extra={'p': p})

    def probe(self, kind, split='probe'):
        if kind == 'activations':
            return {'activations': self.activations(split)}
        X, y, conf = self.data[split]
        s = self.evaluate(split)
        p, pred, yi = s.extra['p'], s.pred, s.y
        if kind == 'metrics':
            rows = []
            for name, col in self.conf_cols.items():
                c = conf[:, col]
                for v in sorted(set(c)):
                    m = c == v
                    tp = int(((pred == 1) & (yi == 1) & m).sum())
                    fp = int(((pred == 1) & (yi == 0) & m).sum())
                    fn = int(((pred == 0) & (yi == 1) & m).sum())
                    rows.append({'slice': f'{name}={v}', 'n': int(m.sum()), 'f1': round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                                 'precision': round(tp / max(tp + fp, 1), 4), 'recall': round(tp / max(tp + fn, 1), 4),
                                 'mean_p_attacker': round(float(p[m & (yi == 1)].mean()), 4) if (m & (yi == 1)).any() else None,
                                 'mean_p_genuine': round(float(p[m & (yi == 0)].mean()), 4) if (m & (yi == 0)).any() else None})
            return {'overall': s.report, 'slices': rows}
        if kind == 'errors':
            groups = {'TP': (pred == 1) & (yi == 1), 'FN': (pred == 0) & (yi == 1),
                      'FP': (pred == 1) & (yi == 0), 'TN': (pred == 0) & (yi == 0)}
            feat = {g: {'n': int(m.sum()), **({n: round(float(X[m, j].mean()), 4)
                                               for j, n in enumerate(self.feature_names)} if m.any() else {})}
                    for g, m in groups.items()}
            hist = np.histogram(p[yi == 1], bins=10, range=(0, 1))[0].tolist()
            hist0 = np.histogram(p[yi == 0], bins=10, range=(0, 1))[0].tolist()
            return {'overall': s.report, 'feature_means_by_outcome': feat,
                    'p_histogram_attackers_0to1': hist, 'p_histogram_genuine_0to1': hist0}
        raise ValueError(f"unknown probe kind {kind!r}")

    def describe(self):
        sp = self.spec
        where = (f"trained on {'+'.join(sp['train_dataset'])}, evaluated later on {sp['test_dataset']}"
                 if sp.get('training_method') == 'cross' else f"dataset {sp.get('dataset')}")
        return {
            'target': self.target,
            'task': ('Detect fake users (data-poisoning attackers) in an LDP frequency-oracle protocol '
                     f"({sp.get('protocol')}, {where}). One row = one user's report summarized by "
                     f'{len(self.feature_names)} per-run-normalized features; label 1 = attacker. Validation rows are '
                     'balanced 50/50 by config; the reported metric is F1 at probability 0.5.'),
            'architecture': ('FT-Transformer: FeatureTokenizer (x_i * w_i + b_i, d_token %s) + CLS token; %s pre-norm '
                             'blocks (%s-head self-attention over the feature tokens, ReGLU FFN); head = LayerNorm -> '
                             'ReLU -> Linear(d, 1) on the CLS token -> attacker logit.'
                             % (sp.get('d_token'), sp.get('n_layers'), sp.get('n_heads'))),
            'inputs': {'features': self.feature_names},
            'outputs': 'one logit per user (sigmoid = attacker probability; > 0.5 = flagged)',
            'metric': 'loss = 1 - F1 at 0.5 on validation rows (lower is better)',
            'gate_guards': 'validation BCE and 1 - PR-AUC must not get worse by more than the tolerance',
            'params': self.param_table(), 'n_params': self.n_params(),
            'splits': {'probe': f"{len(self.data['probe'][1])} training rows",
                       'val': f"{len(self.data['val'][1])} validation rows"},
        }

    def _ft_state(self, lr_scale):
        if self._ft is None:
            from ..training.trainer import _ft_transformer_param_groups
            X, y = self.train
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            opt = torch.optim.AdamW(_ft_transformer_param_groups(self.model, 0.01),
                                    lr=self.spec.get('lr', 5e-4) * lr_scale)
            pw = torch.tensor([len(neg) / max(len(pos), 1)], dtype=torch.float32, device=self.device)
            self._ft = {'opt': opt, 'crit': nn.BCEWithLogitsLoss(pos_weight=pw), 'pos': pos, 'neg': neg,
                        'rng': np.random.default_rng(self.spec.get('seed', 0) + 1000), 'flops_step': None,
                        'bs': int(self.spec.get('batch_size', 256))}
        return self._ft

    def epoch_steps(self):
        y = self.train[1]
        half = int(self.spec.get('batch_size', 256)) // 2
        return max(1, int(min((y == 1).sum(), (y == 0).sum())) // half)

    def finetune(self, steps, lr_scale=0.1):
        st = self._ft_state(lr_scale)
        X, y = self.train
        half = st['bs'] // 2
        t0 = time.perf_counter()
        self.model.train()
        for _ in range(steps):                  # balanced batches, as main.py --balanced-batches
            b = np.concatenate([st['rng'].choice(st['pos'], half), st['rng'].choice(st['neg'], half)])
            xb = torch.from_numpy(X[b]).to(self.device)
            yb = torch.from_numpy(y[b]).to(self.device).unsqueeze(1)

            def step():
                loss = st['crit'](self.model(xb), yb)
                st['opt'].zero_grad()
                loss.backward()
            if st['flops_step'] is None:
                st['flops_step'] = forward_flops(step)
            else:
                step()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            st['opt'].step()
            self.counters['train_steps'] += 1
            self.counters['train_flops'] += st['flops_step']
        self.model.eval()
        _sync(self.device)
        self.counters['compute_s'] += time.perf_counter() - t0

    def save(self, out_dir):
        import json
        os.makedirs(out_dir, exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(out_dir, 'model.pt'))
        with open(os.path.join(out_dir, 'run_spec.json'), 'w') as f:
            json.dump(self.spec, f, indent=1, default=str)

    def stage1_cost(self):
        import pandas as pd
        hist = os.path.join(self.spec['stage1_dir'], 'training_history.csv')
        epochs = len(pd.read_csv(hist)) if os.path.exists(hist) else None
        steps = epochs * self.epoch_steps() if epochs else None
        X, y = self.train
        xb = torch.from_numpy(X[:int(self.spec.get('batch_size', 256))]).to(self.device)
        keep = self.weights()

        def step():
            self.model.train()
            self.model(xb).sum().backward()
        f = forward_flops(step)
        self.model.zero_grad(set_to_none=True)
        self.load_weights(keep)
        self.model.eval()
        return {'epochs_run': epochs, 'optimizer_steps': steps, 'train_flops': int(f * steps) if steps else None}
