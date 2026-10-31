# Implementation map: robust SR/PM utility recovery

The code is in `attacker_detector/mean_variance/`. It is separate from the
frequency-oracle pipeline (`attacker_detector/data`, `generate_dataset.py`,
`main.py`), and that pipeline is left untouched.

## Existing components reused
| Component | Where | Use |
|---|---|---|
| FT-Transformer | `attacker_detector/models/ft_transformer.py` (`FTTransformer`) | Per-report classifier: one token per feature plus a CLS token. Round context comes from supplied features |
| MLP | `attacker_detector/models/mlp.py` (`RobustAttackerDetector`) | Per-report classifier, same feature schema as the FT-Transformer |
| Plot style | `scripts/plots/style.py` | Figure styling |

## Existing components not reused (and why)
| Component | Reason |
|---|---|
| `attacker_detector/training/trainer.py` | Trains row by row with a fixed 0.5 threshold. It has no round weighting, no utility-based threshold and no group-aware reaggregation, so the new `train_classifier.py` reuses only the models |
| `attacker_detector/data/*`, `generate_dataset.py`, `analysis/sensitivity.py` | Categorical frequency oracles (OUE/OLH/HST). Not applicable to scalar reports |

## Components corrected (step 1 of this package, written earlier this session)
| Module | Correction |
|---|---|
| `transforms.py` | Adds the g2 query convention as a versioned option. `raw` squares x and then normalises it, as in paper §2.2.1. `normalized` normalises x and then squares it, as in paper §5 and Table 3's S(2). The x² range is derived correctly for domains that cross zero |
| `estimator.py` | Group-generic decoding for both conventions; aggregation of a retained subset, with a fallback, for detect-and-remove |

## New modules
| Module | Content |
|---|---|
| `knowledge.py` | Attacker knowledge regimes: compromised sample (paper), exact (named variant), error-injected |
| `feasibility.py` | Sufficient conditions (Eqs 8–13, corrected), beta_min |
| `attacks.py` | IPA (closed form and paper-style projected Adam), OPA-SR (integer counts), OPA-PM (sum-preserving spread sampler). Every attack records a status |
| `sources.py` | Loaders for Taxi, Income and Retirement with declared domains, checksums and provenance |
| `populations.py` | Record partitioning; parent-population recipes (subpopulation selection, exponential and quadratic tilting) |
| `scenarios.py` | Generation of one reporting round: manifest, observation view, label view, pairing fields |
| `store.py` | Resumable manifests (JSONL), round regeneration from seeds with checksum verification, cache |
| `features.py` | SR and PM feature views (per report and per round), with caching |
| `metrics.py` | Utility MSE, accuracy gain (AG), shift ratio, classification diagnostics |
| `methods/` | standard, constant, oracle, k-means, MLP/FT removal, CatBoost, Deep Sets, Set Transformer |
| `evaluate.py`, `plots.py` | Shared test-round evaluation, aggregate tables, figures |
| `scripts/mean_variance/run.py` | CLI: `audit-data`, `generate`, `train`, `evaluate`, `plot`, `smoke` |

## Missing data and dependencies
| Item | Status |
|---|---|
| Taxi (`green_tripdata_2020-12.parquet`) | Present; pyarrow is installed |
| Income (2019 ACS PUMS `psam_pusa/b.csv`) | Present; exact Table 3 match |
| Retirement (SF Employee Compensation, Kaggle Jan 2021) | Present, but no subset matches Table 3. A substitute is used; see `deviations.md` |
| CatBoost | **Not installed.** The CatBoost method reports `unavailable` until `pip install catboost` is run in `ldp-attacker-detector` |
