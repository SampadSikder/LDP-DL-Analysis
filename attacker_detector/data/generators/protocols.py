"""LDP Protocol implementations (OUE, OLH)."""

import math
import numpy as np
from scipy.stats import binom
from scipy.special import erfcinv


def z_bonferroni(delta_over_d: float) -> float:
    """Calculate z-score for Bonferroni correction."""
    return float((2.0 ** 0.5) * erfcinv(2.0 * delta_over_d))


def construct_omega(epsilon: float, domain: int, perturb_method: str) -> np.ndarray:
    """
    Construct the omega distribution (theoretical k-value distribution).
    
    Args:
        epsilon: Privacy parameter
        domain: Domain size
        perturb_method: 'OUE', 'OLH', 'HST_User', 'HST_Server'
    
    Returns:
        PDF array for k-values
    """
    if perturb_method == 'OUE':
        p = 1 / 2
        q = 1 / (math.exp(epsilon) + 1)
        p_bin = (1 / domain) * (p + (domain - 1) * q)
        
    elif perturb_method in ('OLH_User', 'OLH_Server', 'OLH'):
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        q = 1 / (math.exp(epsilon) + g - 1)
        p_bin = (1 / domain) * (p + (domain - 1) * q)
        if p_bin is None:
            p_bin = 0.0
            
    elif perturb_method in ('HST_Server', 'HST_User'):
        p_bin = 1 / 2
    else:
        raise ValueError(f"Unknown perturb_method: {perturb_method}")

    # Ensure p_bin is within valid range [0, 1]
    p_bin = np.clip(p_bin, 0.0, 1.0)

    k = np.arange(domain)
    pdf = binom.pmf(k, domain, p_bin)
    pdf /= pdf.sum()
    return pdf


def _oue_params(epsilon: float) -> tuple:
    """Get OUE protocol parameters."""
    p = 0.5
    q = 1 / (math.exp(epsilon) + 1)
    return p, q


def _olh_params(epsilon: float, n: int) -> tuple:
    """Get OLH protocol parameters."""
    g = int(round(math.exp(epsilon))) + 1
    denom = math.exp(epsilon) + g - 1
    p = math.exp(epsilon) / denom
    q = 1.0 / denom
    denom_pg = (p * g - 1.0)
    
    if abs(denom_pg) < 1e-12:
        a = float('inf')
        b_n = float('inf')
    else:
        a = g / denom_pg
        b_n = n / denom_pg
        
    return g, p, q, a, b_n


def build_normal_lists_from_mechanism_stochastic(
    epsilon: float,
    d: int,
    n: int,
    mechanism: str = "OUE",
    seed: int = 0
) -> tuple:
    """
    Build normal (non-attacked) support lists using LDP mechanism.
    
    Args:
        epsilon: Privacy parameter
        d: Domain size
        n: Number of users
        mechanism: 'OUE', 'OLH', 'HST_User', 'HST_Server'
        seed: Random seed
    
    Returns:
        Tuple of (support, one_list, ESTIMATE_DIST, ESTIMATE_Input)
    """
    rng = np.random.default_rng(seed)

    if mechanism == "OUE":
        omega = construct_omega(epsilon, d, 'OUE')
        p, q = _oue_params(epsilon)
        
    elif mechanism == "OLH":
        omega = construct_omega(epsilon, d, 'OLH')
        g, p_olh, q_olh, a, b_n = _olh_params(epsilon, n)
        
    elif mechanism in ("HST_User", "HST_Server"):
        omega = construct_omega(epsilon, d, mechanism)
    else:
        raise ValueError(f"mechanism must be 'OUE', 'OLH', 'HST_User', or 'HST_Server'")

    # Sample k-values from omega distribution
    k_values = np.arange(d)
    K = rng.choice(k_values, size=n, p=omega)

    # Build support matrix
    support = np.zeros((n, d), dtype=int)
    for i, k in enumerate(K):
        if k > 0:
            idx = rng.choice(d, size=min(k, d), replace=False)
            support[i, idx] = 1

    one_list = support.sum(axis=1)
    obs_counts = support.sum(axis=0).astype(float)

    # Estimate distribution
    if mechanism == "OUE":
        p, q = _oue_params(epsilon)
        normal_ESTIMATE_DIST = (obs_counts - n * q) / max(p - q, 1e-12)
    elif mechanism == "OLH":
        g, p_olh, q_olh, a, b_n = _olh_params(epsilon, n)
        normal_ESTIMATE_DIST = a * obs_counts - b_n
    else:
        normal_ESTIMATE_DIST = obs_counts

    return support, one_list, normal_ESTIMATE_DIST, None
