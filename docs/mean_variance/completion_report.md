# Completion report: SR/PM robust utility-recovery benchmark (2026-10-29)

This report separates what is implemented, what is verified, what was run, and what is
blocked or remaining. **No benchmark conclusions are claimed**: the breadth grid has not been
run, and the smoke and pilot runs are too small to compare methods.

## 1. Implemented
| Brief section | Status | Where |
|---|---|---|
| 2 Repository inspection, implementation map | done | `implementation_map.md` |
| 3 Scope: one scalar plus group per user; SR, PM, IPA, OPA, clean rounds; genuine empirical moments (ddof 0) | done | `scenarios.py`, `estimator.py` |
| 4 Allowed vs forbidden model inputs | done; enforced by an observation-only interface and tests | `Round.observation()`, `features.py`, `test_features_methods.py` |
| 5 `paper_settings.yaml`, `deviations.md`, dataset provenance (checksums, columns, filters, domains) | done | `configs/mean_variance/paper_settings.yaml`, `docs/mean_variance/deviations.md`, `audit-data` |
| 6.1 beta = m/(n+m); requested vs realized beta; group-specific counts | done | `scenarios.py` (meta) |
| 6.2 g2 convention audit; zero-crossing domains; consistent decoders | done; convention versioned (`square_mode`) | `transforms.py` |
| 6.3 Equation corrections, verified numerically | done | `attacks.py`, `test_attacks.py` |
| 6.4 IPA through the actual randomizer; feasibility, convergence and residual recorded separately | done | `attacks.ipa_inputs`, `scenarios.py` |
| 6.5 OPA-SR integer counts, rounding residual | done | `attacks.opa` |
| 6.6 OPA-PM sum-preserving spread in (-s, s), no clipping | done | `attacks.pm_spread` |
| 6.7 Status taxonomy (feasible / target error / infeasible / solver failure), no silent substitution | done | `AttackStatus` |
| 6.8 PM spread ratios [0 ... 1] × plateau width; nominal, achievable and realized widths | done (sampler and recording); spread sweep config runs only as attack validation | `pm_spread` |
| 7 Breadth grid config, run-size profiles, one-factor sweep support | config done; sweep support in the generator (`sweep:` and `size_mode: fixed_genuine`) | `breadth.yaml`, `experiment.py` |
| 8.1-8.2 Six single-source transfer directions, three multi-source transfers; validation from training sources; one frozen checkpoint for all destinations | done (`evaluate`) | `evaluate.run_unit`, `run.py training_sets` |
| 8.4 Population variants (subpopulation, rank-based tilting) and constant-moment baseline | done | `populations.py`, `ConstantMethod` |
| 8.6 Records partitioned before populations; variants stay in their split; declared domains | done, tested | `record_partition`, `test_generation.py` |
| 9 Manifest, observation, label and pairing views; output records | done | `scenarios.py`, `store.py`, `evaluate.py`, `data_dictionary.md` |
| 9 PM / SR features, mechanism-aware ablation (NNLS residuals, interval excess) | done; variation across admissible fits **unavailable** | `features.py` |
| 10 Methods: standard, constant, oracle removal, k-means, MLP / FT removal, CatBoost, Deep Sets, Set Transformer | done; CatBoost **unavailable** (not installed) | `methods/` |
| 11 Valid parameterisation, projection, hybrid loss and its three variants | done | `methods/set_models.py`, `metrics.project_moments` |
| 12 Utility MSE, AG (mean / variance), median, p90, fraction worse, runtime, peak memory, fallback, classifier diagnostics, population-clustered bootstrap CIs | done | `evaluate.aggregate`, `metrics.py` |
| 13 Figures | done as functions, wired into `report`: beta sweeps, eps×beta heatmaps (AG on a diverging, zero-centred scale), within / cross / multi-source transfer matrices, AG bars, runtime and memory, feasibility maps; SVG + PNG + manifest | `plots.py`, `validation.py` |
| 14 Adaptive attack stage | **not implemented** | — |
| 15 Resumable generation and evaluation; `--shard i/n` parallel evaluation | done | `store.py`, `run.py` |

## 2. Verified
`python -m unittest discover -s tests/mean_variance -t .`: all tests pass. Covered:
- Mechanisms: SR / PM probabilities and densities normalise; reports stay in their support; the e^eps ratio holds; E[unbias(y) | x] = x; the PM variance closed form equals the density integral; a chi-square fit holds.
- Transforms: both conventions, zero-crossing domains, sum conversions, and error injection on benchmark-scale sums.
- Estimation: clean estimates are unbiased (Monte Carlo, ±5 SE), in both conventions.
- Attacks:
  - With exact knowledge, OPA-SR, OPA-PM and IPA (through the randomizer) hit the target in expectation; the second moment is checked against Eq. 5's slack target.
  - OPA-PM preserves sums and support, never emits ±s, and handles truncated spread.
  - OPA-SR uses integer, attainable counts.
  - Infeasible targets and solver failures never produce a round.
  - Thm. 1 is maximal, and the analytic B_max equals the constructed one.
  - The paper typos demonstrably miss the target.
- Feasibility: beta_min is minimal; the sufficient conditions agree with the generators; OPA needs fewer fakes than IPA, and PM fewer than SR.
- Generation: the partition is fixed; populations stay in their split and vary in moments; rounds are deterministic with checksums; attack variants share genuine reports; the clean counterpart equals the genuine subset.
- Store: resume and crash-after-shard recovery leave no duplicates.
- Features and models:
  - Shuffling permutes per-report features and scores and leaves aggregate predictions unchanged (classifiers and both set models).
  - Identical SR cells get identical scores.
  - Feature signatures and names contain no hidden fields.
  - Removal fallback is deterministic.
  - Clean-round metric conventions hold.

Brief §16 checks **not yet automated**: "every model receives the same test rounds" (true by construction, since every method in a unit reads the same `rounds_from_store` list; the smoke run shows equal round counts) and "dataset identity cannot leak through directory names" (shard names contain the source, but models only ever receive observation dicts).

## 3. Experiments run (all real; none fabricated)
| Run | Scope | Outcome |
|---|---|---|
| Source audit | 3 sources vs Table 3 | Taxi and Income reproduce Table 3 (Income exactly). Retirement: no subset matches, so a substitute is used |
| Smoke | SR+PM, ε=1, 3 sources, β {0, 0.1, 0.4}, 12/4/6 rounds per cell, 4 epochs | All methods produce saved predictions, aggregate tables and figures. **Not a comparison**: models are deliberately undertrained (set models' mean MSE 0.17-0.35 vs standard 0.004) |
| Attack validation, Taxi, paper scale (n = 83,130, 100 reps) | Table 4 targets, ε=1, β=0.1; S_e(1), S_e(2), n_e and β sweeps | Paper-level MSE to target (~1e-4) and the paper's qualitative trends. Variance MSE matches only under the normalized g2 convention; Table 4 fixed knowledge leaves an open variance discrepancy (deviations.md) |
| Feasibility maps | 3 sources × {IPA, OPA-SR, OPA-PM}, ε=1, exact knowledge | IPA can reach about 172 of 323 targets, OPA all 323; OPA-PM needs the fewest fakes |
| Pilot | PM, ε=1, Taxi, full beta/attack/target grid, 20/10/20 rounds | completed; timing and directional utility in section 4 |
| Frozen shift (smoke) | eps 1 checkpoints → eps 2 test rounds | completed (pipeline check only) |
| k-means defense verification (paper Sec. 8.1 / Fig. 12) | Taxi at full size, eps 1, Table 4 knowledge, 100 repetitions; r sweep 0.1-0.9, then beta, target-mean and target-variance sweeps at r = 0.6 | **Paper gains not reproduced.** PM AG ~0; SR's small gain (<= 14% of damage) is a deterministic hypergeometric-skew shift toward the domain midpoint, which also raises MSE on clean rounds. See deviations.md and `results/mean_variance/verify_kmeans_figures/` |

## 4. Pilot timing (PM, ε=1, Taxi; 3,260 train / 1,630 val / 3,260 test rounds; N cycled over 2,048 / 4,096 / 8,192)
Measured on this machine (Apple silicon, 14 cores, MPS), one seed. The pilot config used `square_mode: raw` because it predates the switch to `normalized`; runtime does not depend on the convention, but its utility numbers do.

| Method | Fit (s) | Inference (s/round) | Peak RSS | Unit total (s) |
|---|---|---|---|---|
| standard / oracle removal | 0 | ~1e-4 | 1.2 GB | <1 |
| k-means (5,000 subsets; rate on 300 val rounds; 12-process pool) | 160 | 0.30 CPU (≈0.03 wall pooled) | 1.2 GB | 242 |
| MLP removal | 154 | 0.0085 | 2.4 GB | 182 |
| FT-Transformer removal | 377 | 0.036 | 1.2 GB | 490 |
| Deep Sets recovery | 51 | 0.0029 | 1.8 GB | 62 |
| Set Transformer recovery | 167 | 0.0091 | 2.2 GB | 197 |

**Breadth estimate**, scaling the measured fit time with training rounds and inference with test rounds:
- Each single-source unit evaluates 48,900 test rounds (3 destinations × 100 replicates × 163 cells); each multi-source unit evaluates 16,300.
- About 16.5 h per (protocol, epsilon) for all methods, with 3 seeds for the learned methods and excluding CatBoost. That is about **165 h ≈ 7 days sequential** for the 10 (protocol, epsilon) pairs.
- FT-Transformer removal is about 53% of it.
- With 3-4 `--shard` processes, expect roughly 2-3 days of wall time. MPS and the k-means pool are shared, so the speedup is less than linear.
- Storage: about 17 GB of stored rounds (26.5 KB per round measured) plus about 13 GB of per-round result tables.
- These numbers come from one PM cell; SR features are cheaper.

**Pilot utility (directional only; one seed, raw convention, 20 training replicates per cell, Taxi):**
- MLP and FT-Transformer removal both picked "remove nothing" on validation (threshold 1.01), so they equal the standard estimate. Their per-report validation BCE (0.46-0.47) is barely below the constant-prevalence level, consistent with the 1-D per-user detection ceiling analysed earlier.
- Deep Sets and Set Transformer lowered variance MSE (0.0027-0.0029 vs 0.0162 for standard) but raised mean MSE (0.0070-0.0083 vs 0.0045). They made 40% of rounds worse.
- Oracle removal was worse than standard (fewer reports at high beta), which illustrates the brief's note that it is not a lower bound.
- No constant baseline was in the pilot, so it is not yet known how much of the variance gain is shrinkage toward typical training moments.
- **None of this is a benchmark result.**

**Frozen-checkpoint shift check (smoke):** 12 smoke checkpoints per method (trained at eps = 1) were evaluated with no refitting on eps = 2 test rounds (60 evaluations; `results/mean_variance/smoke_shift_eps2/`). The path works; smoke models are undertrained, so there is no interpretation.

## 5. Audit findings (changes made because of them)
1. **g2 convention.** §2.2.1 says "square, then normalise", but Table 3 / §5 and the paper's variance-MSE level match "normalise, then square". The default is now `normalized`; `raw` is kept as a named variant.
2. **Paper equation typos** (§4.2.2, Eq. 3, Eqs. 10-13) are corrected and each is tested.
3. **Projected Adam fixed point.** A plain projected Adam on Eq. 3 stalls once all values share a sign; the tangent-space gradient fixes it.
4. **Knowledge-error coordinates.** Errors must be applied to the benchmark-scale sums at the attacker's n_e; shifting raw sums and n_e separately moves S~1 by dn·c.
5. **Taxi column.** The paper's "mileage" is actually `DOLocationID` (n, mean, variance, S1, S2 all reproduced). Table 3's printed mean -0.022 disagrees with its own S1 (-0.026).
6. **Infrastructure bugs found by the pilot.**
   - NpzFile re-reads an array on every key access (20 GB OOM, fixed).
   - Spec-schema changes silently doubled stored rounds (evaluation is now restricted to the configured round ids).
   - A single-pass FT validation loss overflowed an MPS tensor limit and aborted the process (now batched).
   - Sequential k-means rate selection took over an hour per unit (now a process pool plus a fixed 300-round validation subsample).

## 6. Blocked or remaining, with exact commands
| Item | Status | To resume |
|---|---|---|
| CatBoost recovery | blocked: package missing | `~/miniconda3/envs/ldp-attacker-detector/bin/pip install catboost`, then rerun `evaluate` (finished units are skipped; CatBoost units marked `unavailable` must have their `units/*catboost*.done.json` deleted first) |
| Breadth benchmark | configured, not run (compute estimate in section 4) | `run.py generate --config configs/mean_variance/breadth.yaml`, then `run.py evaluate --config ... --shard i/4` in 4 processes, then `run.py report` |
| Within-source and cross-population results | produced by the breadth `evaluate` | as above |
| Income paper-scale attack validation | not run (~1 h) | `run.py attack-validate --config configs/mean_variance/paper_attack_default.yaml` |
| Frozen-checkpoint evaluation on unseen ε / spread / attacks (§8.3) | loader and command done (`methods/checkpoint.py`, round-trip tested); shift configs not yet run | `run.py generate --config <shift config> --splits test`, then `run.py evaluate-frozen --config <shift config> --checkpoints results/mean_variance/breadth/checkpoints` |
| Epsilon-conditioned models, per-beta specialists | not implemented | — |
| Adaptive attack stage (§14) | not implemented | — |
| Learning curves vs training rounds | not implemented (training histories are saved per unit) | — |
| Uncertainty / variation-across-fits features | unavailable | — |
| Remaining figures (eps×beta heatmaps, transfer panels, runtime / memory, loss and feature ablations) | not implemented | — |
| Retirement substitute | proposed; needs user confirmation | `deviations.md` |

## 7. Supported conclusions
Only implementation-level statements are supported:
- The attacks reproduce the paper's Taxi attack-target error levels and trends under the normalized convention.
- OPA has a much larger feasible target region than IPA.
- The paper's k-means defense, implemented as described, does not detect fakes in these settings (Taxi, eps 1). Its SR behaviour is a hypergeometric-skew artifact.
- All benchmark components run end to end and pass the acceptance tests listed above.

No claim is made about which defense recovers utility best.
