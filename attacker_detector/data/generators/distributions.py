import os
import numpy as np
import pandas as pd

# Resolve paths relative to this file
base_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels to the workspace root: generators -> data -> attacker_detector -> root
project_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))

ZIPF_PATH = os.path.join(project_root, "datasets", "zipf.npy")
EMOJI_PATH = os.path.join(project_root, "datasets", "emoji.npy")
FIRE_PATH = os.path.join(project_root, "datasets", "fire.csv")

_zipf_data = None
_emoji_data = None
_fire_data = None


def get_zipf_data():
    global _zipf_data
    if _zipf_data is None:
        _zipf_data = np.load(ZIPF_PATH)
    return _zipf_data


def get_emoji_data():
    global _emoji_data
    if _emoji_data is None:
        _emoji_data = np.load(EMOJI_PATH)
    return _emoji_data


def get_fire_data():
    global _fire_data
    if _fire_data is None:
        df = pd.read_csv(FIRE_PATH)
        # Determine deterministic mapping from Unit_ID to index
        unique_units = sorted(df['Unit_ID'].dropna().unique())
        unit_to_idx = {unit: i for i, unit in enumerate(unique_units)}
        _fire_data = df['Unit_ID'].map(unit_to_idx).dropna().values.astype(np.int64)
    return _fire_data


def generate_zipf_dist(n: int, domain: int, s: float = 1.5, seed: int = None) -> tuple:
    if seed is not None:
        np.random.seed(seed)

    data = get_zipf_data()
    # Filter to requested domain
    data = data[data < domain]

    counts = np.bincount(data, minlength=domain)
    REAL_DIST = counts / counts.sum()

    replace = (n > len(data))
    X = np.random.choice(data, size=n, replace=replace)

    return X, REAL_DIST


def generate_emoji_dist(n: int, domain: int, seed: int = None) -> tuple:
    if seed is not None:
        np.random.seed(seed)

    data = get_emoji_data()
    # Filter to requested domain
    data = data[data < domain]

    counts = np.bincount(data, minlength=domain)
    REAL_DIST = counts / counts.sum()

    replace = (n > len(data))
    X = np.random.choice(data, size=n, replace=replace)

    return X, REAL_DIST


def generate_fire_dist(n: int, domain: int, seed: int = None) -> tuple:

    if seed is not None:
        np.random.seed(seed)

    data = get_fire_data()
    # Filter to requested domain
    data = data[data < domain]

    counts = np.bincount(data, minlength=domain)
    REAL_DIST = counts / counts.sum()

    replace = (n > len(data))
    X = np.random.choice(data, size=n, replace=replace)

    return X, REAL_DIST
