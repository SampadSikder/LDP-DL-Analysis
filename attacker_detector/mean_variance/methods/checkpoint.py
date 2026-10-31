"""Rebuild a frozen method from its checkpoint directory (state.json [+ model.pt]).

Used to evaluate frozen models on new test sets (unseen epsilon, spread, attacks)
without refitting anything: thresholds, rates, preprocessing and loss scales come
from the checkpoint.
"""

import json
import os

import numpy as np
import torch

from ..metrics import UtilityScales
from .baselines import ConstantMethod, KMeansRecovery, OracleRemoval, StandardMethod
from .removal import ClassifierRemoval
from .set_models import DeepSetsNet, SetRecovery, SetTransformerNet


def save_checkpoint(method, path: str):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'state.json'), 'w') as f:
        json.dump(method.state(), f, default=float)
    if hasattr(method, 'model'):
        torch.save(method.model.state_dict(), os.path.join(path, 'model.pt'))


def load_checkpoint(path: str, device: str = 'cpu'):
    with open(os.path.join(path, 'state.json')) as f:
        st = json.load(f)
    name = st['name']
    if name == 'standard':
        return StandardMethod()
    if name == 'oracle_removal':
        return OracleRemoval()
    if name == 'constant':
        m = ConstantMethod()
        m.mean_, m.var_ = st['mean'], st['variance']
        return m
    if name == 'kmeans':
        return KMeansRecovery(rate=st['rate'], n_subsets=st['n_subsets'])
    weights = torch.load(os.path.join(path, 'model.pt'), map_location=device)
    if name.startswith(('mlp_removal', 'ft_removal')):
        m = ClassifierRemoval(st['arch'], feature_set=st['feature_set'], seed=st['seed'], device=device,
                              reports_per_round=st['reports_per_round'])
        m.mu_, m.sd_ = np.asarray(st['mu'], np.float32), np.asarray(st['sd'], np.float32)
        m.threshold, m.val_utility_, m.scales_, m.history = st['threshold'], st['val_utility'], st['scales'], st['history']
        m.model = m._build(len(m.mu_)).to(m.device)
        m.model.load_state_dict(weights)
        m.model.eval()
        return m
    if name.startswith(('deepsets_recovery', 'settransformer_recovery')):
        m = SetRecovery(st['arch'], loss=st['loss'], lambda_clean=st['lambda_clean'] or 1.0,
                        lambda_aux=st['lambda_aux'] or 0.1, lambda_var=st['lambda_var'],
                        token_cap=st['token_cap_train'], seed=st['seed'], device=device)
        sc = st['scales']
        m.scales_ = UtilityScales(sc['scale_mean'], sc['scale_var'], sc['lambda_var'])
        m.history, m.best_val_utility_ = st['history'], st['best_val_utility']
        m.model = (DeepSetsNet() if st['arch'] == 'deepsets' else SetTransformerNet()).to(m.device)
        m.model.load_state_dict(weights)
        m.model.eval()
        return m
    raise ValueError(f"no loader for method {name!r}")
