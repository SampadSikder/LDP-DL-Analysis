#!/usr/bin/env python
"""Go/no-go sweep: do the v2 target-block features beat v1, and which M?

For every (protocol, dataset, beta, seed) it simulates one LDP run at the
paper defaults (eps=1, r=10, r'=4) and extracts v1 and v2 features for each
candidate count M. It then reports

  * single-feature AUC of block_projection / block_overlap per M, and
  * a gradient-boosting classifier per (protocol, dataset, feature set),
    trained on seeds < the last one and scored on the last seed's runs --
    balanced accuracy (prevalence-free) and AUC.

A cheap stand-in for a full regenerate + FT-Transformer retrain.

    python scripts/sweep_feature_sets.py --out sweep_v2 --workers 16
"""

import argparse
import itertools
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_CONFIGS  # noqa: E402
from attacker_detector.data.generators import (  # noqa: E402
    generate_perturbed_data, extract_user_level_features_diffstats_style,
    FEATURE_NAMES_V2)


def one_run(task):
    protocol, dataset, beta, seed, n, ms = task
    try:
        base = "OLH" if protocol.startswith("OLH") else protocol
        domain = DATASET_CONFIGS[dataset]["domain"]
        sup, y, _, _, ones = generate_perturbed_data(
            epsilon=1.0, domain=domain, n=n, protocol=base, ratio=beta,
            target_set_size=10, splits=4, dataset_type=dataset, h_ao=1,
            seed=10_000 * seed + int(beta * 1000), processors=1,
            olh_setting="server")
        out = {"v1": extract_user_level_features_diffstats_style(
            sup, ones, 1.0, protocol, domain, n, feature_set="v1")}
        for m in ms:
            out[f"v2_M{m}"] = extract_user_level_features_diffstats_style(
                sup, ones, 1.0, protocol, domain, n, feature_set="v2",
                block_candidates=m)
        return {"ok": True, "key": (protocol, dataset, beta, seed),
                "y": y.astype(np.int8), "feats": out}
    except Exception:
        return {"ok": False, "key": (protocol, dataset, beta, seed),
                "error": traceback.format_exc()}


def balanced_subsample(y, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    k = min(len(pos), len(neg))
    return np.concatenate([rng.choice(pos, k, replace=False),
                           rng.choice(neg, k, replace=False)])


def evaluate(results, seeds, feature_sets, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    test_seed = max(seeds)
    rows = []
    groups = {}
    for r in results:
        protocol, dataset, beta, seed = r["key"]
        groups.setdefault((protocol, dataset), []).append(r)

    for (protocol, dataset), runs in sorted(groups.items()):
        train = [r for r in runs if r["key"][3] != test_seed]
        test = [r for r in runs if r["key"][3] == test_seed]
        if not train or not test:
            continue
        for fs in feature_sets:
            Xtr = np.vstack([r["feats"][fs] for r in train])
            ytr = np.concatenate([r["y"] for r in train])
            idx = balanced_subsample(ytr, rng)          # 50/50, like main.py
            clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
            clf.fit(Xtr[idx], ytr[idx])
            for r in test:
                beta = r["key"][2]
                p = clf.predict_proba(r["feats"][fs])[:, 1]
                bi = balanced_subsample(r["y"], rng)
                rows.append({
                    "protocol": protocol, "dataset": dataset, "beta": beta,
                    "feature_set": fs,
                    "bal_acc": balanced_accuracy_score(r["y"][bi], p[bi] > 0.5),
                    "auc": roc_auc_score(r["y"], p),
                })
    return pd.DataFrame(rows)


def single_feature_aucs(results):
    from sklearn.metrics import roc_auc_score
    rows = []
    ip, io = FEATURE_NAMES_V2.index("block_projection"), FEATURE_NAMES_V2.index("block_overlap")
    for r in results:
        protocol, dataset, beta, seed = r["key"]
        for fs, F in r["feats"].items():
            if fs == "v1":
                continue
            rows.append({"protocol": protocol, "dataset": dataset, "beta": beta,
                         "seed": seed, "M": int(fs.split("M")[1]),
                         "auc_projection": roc_auc_score(r["y"], F[:, ip]),
                         "auc_overlap": roc_auc_score(r["y"], F[:, io])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocols", nargs="+",
                    default=["OUE", "OLH_Server", "HST_Server", "HST_User"])
    ap.add_argument("--datasets", nargs="+", default=["zipf", "emoji", "fire"])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.01, 0.05, 0.2])
    ap.add_argument("--seeds", type=int, default=3,
                    help="the last seed is held out for the classifier test")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--candidates", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.seeds < 2:
        ap.error("--seeds must be >= 2 (one is held out)")

    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(args.seeds))
    tasks = [(p, d, b, s, args.n, args.candidates) for p, d, b, s in
             itertools.product(args.protocols, args.datasets, args.betas, seeds)]
    print(f"{len(tasks)} LDP runs at n={args.n}, M in {args.candidates}", flush=True)

    results, failed = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one_run, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["ok"]:
                results.append(r)
            else:
                failed += 1
                print(f"[FAIL] {r['key']}\n{r['error']}", flush=True)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} runs done ({failed} failed)", flush=True)

    rng = np.random.default_rng(0)
    sf = single_feature_aucs(results)
    sf.to_csv(os.path.join(args.out, "single_feature_auc.csv"), index=False)
    feature_sets = ["v1"] + [f"v2_M{m}" for m in args.candidates]
    clf = evaluate(results, seeds, feature_sets, rng)
    clf.to_csv(os.path.join(args.out, "classifier.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== single-feature AUC of block_projection, mean over seeds and beta ===")
    print(sf.pivot_table(index=["protocol", "dataset"], columns="M",
                         values="auc_projection").round(3).to_string())
    print("\n=== classifier balanced accuracy on the held-out seed, mean over beta ===")
    t = clf.pivot_table(index=["protocol", "dataset"], columns="feature_set",
                        values="bal_acc").round(3)
    print(t[feature_sets].to_string())
    print("\n=== same, by beta (v1 vs best v2) ===")
    print(clf.pivot_table(index=["protocol", "beta"], columns="feature_set",
                          values="bal_acc").round(3)[feature_sets].to_string())
    print(f"\nWrote {args.out}/single_feature_auc.csv and classifier.csv")


if __name__ == "__main__":
    main()
