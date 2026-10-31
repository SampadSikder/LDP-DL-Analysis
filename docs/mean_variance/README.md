# SR/PM robust mean and variance recovery benchmark

Code: `attacker_detector/mean_variance/`. Tests: `tests/mean_variance/`. Configs: `configs/mean_variance/`.
Background: Li et al., USENIX Security 2023. See `deviations.md` for equation corrections and
dataset substitutes, `data_dictionary.md` for every record, and `implementation_map.md` for
what is reused.

## Files in this folder

| File | What it contains |
|---|---|
| `README.md` | This file: the README index, then commands for checks, smoke, breadth, budget experiments |
| `implementation_map.md` | Which existing repo components are reused (FT-Transformer, MLP) or not (row-wise trainer, categorical pipeline), corrected components, new modules, missing data and dependencies |
| `deviations.md` | Equation corrections (Sec. 4.2.2, Eq. 3, Eqs. 10-13), the budget stage, underspecified choices (g2 convention, IPA solver, PM spread, rounding, infeasibility), attack validation vs Figs. 5-11 with the open Table 4 discrepancy, the k-means (Fig. 12) non-reproduction and why, dataset provenance and the Retirement substitute, proposed extensions |
| `data_dictionary.md` | Source files and where to get them; every manifest field (budget, theory, construction, realized), the observation shard arrays, per-round result columns, aggregate and figure outputs |
| `completion_report.md` | Status against the brief: what is implemented and verified, which experiments ran (with numbers), pilot timing and full-grid estimate, audit findings, blocked or remaining work with resume commands, supported conclusions |

## Index of folder READMEs

| README | Covers |
|---|---|
| `attacker_detector/mean_variance/README.md` | Every module of the benchmark package and how they depend on each other |
| `attacker_detector/mean_variance/methods/README.md` | The method interface and each method (standard, constant, oracle, k-means, MLP / FT removal, Deep Sets, Set Transformer, CatBoost, checkpoints) |
| `scripts/mean_variance/README.md` | The `run.py` CLI and every command |
| `tests/mean_variance/README.md` | Each test file, its test count, and what it verifies |
| `configs/mean_variance/README.md` | Each config: command, purpose, paper-faithful or proposed; common keys; git-ignored one-off configs |
| `docs/mean_variance/README.md` | This folder (above) and the run commands (below) |

All commands use the project environment:
```
PY=~/miniconda3/envs/ldp-attacker-detector/bin/python
```

## Checks
```
$PY -m unittest discover -s tests/mean_variance -t .        # protocol, attack, generation, feature, method tests
$PY scripts/mean_variance/run.py audit-data                  # source statistics vs Table 3, provenance JSON
```

## Smoke (every component, small; about 5 minutes)
```
$PY scripts/mean_variance/run.py smoke --config configs/mean_variance/smoke.yaml
```
Output: `results/mean_variance/smoke/` (per-round predictions, aggregate.csv, figures).

## Breadth benchmark
```
$PY scripts/mean_variance/run.py generate --config configs/mean_variance/breadth.yaml
$PY scripts/mean_variance/run.py evaluate --config configs/mean_variance/breadth.yaml
$PY scripts/mean_variance/run.py report   --config configs/mean_variance/breadth.yaml
```
Each step resumes: completed rounds and completed (training set, method, seed) units are
skipped. `evaluate` trains one model per (protocol, epsilon, training-source set) and evaluates
the frozen checkpoint on every destination. Single-source sets give the within-source result
and all six transfer directions; the two-source sets give the three multi-source transfers.
Run a subset with `--methods kmeans deepsets_recovery`.

## Minimum-budget stage and budget experiments
```
$PY scripts/mean_variance/run.py budget-curves --epsilon 1.0                   # beta_min vs target (attacker vs oracle)
$PY scripts/mean_variance/run.py smoke --config configs/mean_variance/budget_sweep.yaml        # m = ceil(f m_min)
$PY scripts/mean_variance/run.py generate --config configs/mean_variance/below_budget.yaml --splits test
$PY scripts/mean_variance/run.py evaluate-frozen --config configs/mean_variance/below_budget.yaml \
    --checkpoints results/mean_variance/budget_sweep/checkpoints                  # separately labelled
```

## Not yet implemented (see the completion report)
paper-scale reproduction configs, one-factor sweep configs (the generator supports
`size_mode: fixed_genuine` and `sweep:` blocks), epsilon-conditioned and unseen-shift tests,
the adaptive attack stage, and variation-across-fits uncertainty features.
