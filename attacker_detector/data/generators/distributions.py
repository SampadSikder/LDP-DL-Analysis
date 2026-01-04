import numpy as np

def generate_zipf_dist(n: int, domain: int, s: float = 1.5, seed: int = None) -> tuple:
    """
    Generate Zipf-distributed user data.
    
    Args:
        n: Number of users
        domain: Domain size (number of items)
        s: Zipf parameter (default 1.5)
        seed: Random seed
    
    Returns:
        Tuple of (X, REAL_DIST) where X is user data and REAL_DIST is true distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    ranks = np.arange(1, domain + 1)
    weights = 1.0 / (ranks ** s)
    REAL_DIST = weights / weights.sum()
    
    X = np.random.choice(domain, size=n, p=REAL_DIST)
    
    return X, REAL_DIST


def generate_emoji_dist(n: int, domain: int, seed: int = None) -> tuple:
    if seed is not None:
        np.random.seed(seed)
    
    ranks = np.arange(1, domain + 1)
    weights = (1.0 / ranks) * np.exp(-ranks / (domain / 5))
    REAL_DIST = weights / weights.sum()
    
    X = np.random.choice(domain, size=n, p=REAL_DIST)
    
    return X, REAL_DIST


def generate_fire_dist(n: int, domain: int, seed: int = None) -> tuple:

    if seed is not None:
        np.random.seed(seed)
    
    ranks = np.arange(1, domain + 1)
    weights = 1.0 / (ranks ** 0.8)
    
    noise = np.random.uniform(0.1, 0.5, domain)
    weights = weights * noise
    REAL_DIST = weights / weights.sum()
    
    X = np.random.choice(domain, size=n, p=REAL_DIST)
    
    return X, REAL_DIST
