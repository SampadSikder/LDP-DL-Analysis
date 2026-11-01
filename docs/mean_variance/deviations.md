# Deviations from Li et al. (USENIX Security 2023) and corrections

Every item names the paper location, what the code does, and how it was verified.

## Equation corrections (verified in `tests/mean_variance/test_attacks.py`)
| Paper | Printed | Implemented | Check |
|---|---|---|---|
| Sec. 4.2.2 (OPA-PM, g2) | `(sigma_t^2 + mu_t)` | `sigma_t^2 + mu_t^2` (Eq. 5) | `test_paper_typos_miss_the_target` |
| Sec. 4.2.2 | uses `n` | uses `n_e`, as in Eq. 6. The attacker knows only n_e | code |
| Eq. 3 (IPA objective) | `(Sum y^2 - (n_e+m)(sigma^2+mu^2) - S_e^(2))^2` | `(Sum y^2 - [(n_e+m)(sigma^2+mu^2) - S_e^(2)])^2`, consistent with Eq. 2 | `test_paper_typos_miss_the_target`, `test_closed_form_solves_eqs_1_2` |
| Eqs. 10-13 (Sec. 5) | treat g2 values as if on [-1, 1] and print S_e without the 1/2 | conditions derived from Eqs. 6-7, keeping g2's rescaling (k2, lo2) and S_e/2 | `test_sufficient_matches_generator` |
| Eqs. 6-7 | OPA attack equations | implemented as printed (`sum_rule='paper'`), plus a `group_aware` variant that uses the attacker's known m_g | `test_expected_estimate_hits_target`, `test_group_aware_rule` |

## Budget stage (mandatory, before any attack is generated)
- `feasibility.py` checks theoretical feasibility per attack (IPA joint, OPA-SR integer counts, OPA-PM support and spread room) at a planned allocation of m_1 = ceil(m/2), m_2 = floor(m/2), using the attacker's estimates. m_min is the smallest feasible integer budget; a local recheck below the bisection result guards against non-monotonicity from integer rounding.
- The generator has its own status, kept separate: optimizer failure is `solver_failure`, never infeasibility. A plan that is feasible but defeated by the server's realized group split is `allocation_infeasible`. In both cases no other attack is substituted.
- Fixed-beta sweeps keep the configured budget: below m_min the round is `infeasible_target` (theory) and not generated. The budget is never raised and the target never changed.
- The budget ladder (`budget_sweep.yaml`) and the below-budget evaluation (`below_budget.yaml`, saturated best effort, status `below_budget`) are proposed extensions, kept apart from the paper's settings.
- The feasibility heatmaps use exact genuine sums and are labelled **oracle diagnostics**. `budget-curves` draws the attacker-estimate and oracle beta_min side by side.

## Underspecified items: choices made
| Item | Paper | Here |
|---|---|---|
| g2 query convention | Sec. 2.2.1 squares raw x and then normalises. Sec. 5 and Table 3 square the normalised x | **Default `normalized`** in all configs. Paper-scale Taxi attack validation gives a target-variance MSE of ~9e-5, as in Figs. 5-6, only under `normalized`; `raw` gives ~1e-3 (~10x), consistent with its ~4x larger standard deviation in the decoded variance. `raw` stays available as the Sec. 2.2.1 variant. Both are tested to be unbiased |
| Attacker knowledge noise | Table 4 lists one fixed S*_e (a single 1,000-user sample) | The default draws a fresh compromised sample per round, which adds knowledge bias (PM mean MSE ~3e-4 vs the paper's ~1e-4 on Taxi). Table 4's fixed knowledge is reproducible as `knowledge_regime: exact` plus the errors listed in paper_settings.yaml |
| x^2 range when a < 0 < b | `[a^2, b^2]` | `[0, max(a^2, b^2)]` |
| IPA solver | Adam (lr 1e-3, 10k iterations); constraint handling not stated | Default `closed_form`, exact. `adam` is projected-gradient Adam that returns the best iterate. A plain projected Adam stalls at a fixed point once all values share a sign (found while testing) |
| OPA-PM spread sampler | "randomly perturbs ... maintaining the sum ... in [-s, s]" | `attacks.pm_spread`: uniform window, centred, one common scale factor; no clipping. Nominal, achievable and realized widths are recorded |
| OPA-SR rounding | not discussed | `#(+1) = round((m_g + T)/2)`. The residual is recorded and the status is `generated_with_target_error` |
| Infeasible targets | "no MSE recorded" | status `infeasible_target`; the round is not generated and no other attack replaces it |
| Variance estimate | unclipped | unclipped; raw estimates are kept, and the feasibility projection is applied only where a method declares it |

## Attack validation at paper scale (Taxi, eps 1, beta 0.1, 100 repetitions; `attack-validate`)
| Knowledge | MSE to target mean | MSE to target variance | Paper (Figs. 5-6, Taxi) |
|---|---|---|---|
| compromised sample per round (default) | 3.4e-4 to 4.1e-4 | 7.5e-5 to 1.1e-4 | mean about 1e-4 to 3e-4, variance about 6e-5 to 2e-4 |
| Table 4 fixed knowledge (`paper_attack_table4_taxi.yaml`) | 8.3e-5 to 1.1e-4 | 6.1e-4 to 7.6e-4 | same |

Under the default, the mean error is dominated by knowledge noise from a fresh 1,000-user sample per round. Under Table 4's knowledge (S*_e(2) 4.3% high, n_e 3,130 low), the second-moment target is missed by about 0.025, whose square (6e-4) matches what we observe. The paper reports a lower variance error with the same knowledge, so its implementation must have used S_e(2) / n_e differently; how is not stated (Sec. 4.2.2's use of n instead of n_e may be related). This is an **open reproduction discrepancy**, not a claimed match. Error injection is defined on the benchmark-scale sums at the attacker's n_e (`scenarios.inject_knowledge_error`).

## k-means defense verification (Sec. 8.1, Fig. 12; `verify-kmeans`)
Taxi at full size (n = 83,130), eps 1, beta 0.1, targets mu_t1 / sigma_t1^2, Table 4 fixed knowledge, 100 repetitions, 5,000 subsets per group. AG = MSE_before - MSE_after against the genuine moments, paired per round, using raw estimates as the paper does.

| Rate r | OPA-SR AG mean | OPA-SR AG var | OPA-PM AG mean | OPA-PM AG var |
|---|---|---|---|---|
| 0.1 | -4.5e-3 | -5.4e-4 | -3.5e-4 | -2.7e-4 |
| 0.3 | -2.2e-3 | -3.7e-4 | -2.6e-4 | -7.8e-5 |
| 0.6 (best) | +1.0e-3 | +1.4e-4 | -5.2e-5 | +1.5e-5 |
| 0.9 | +4.4e-4 | +2.4e-5 | -4.0e-5 | +4.4e-6 |

MSE_before is 7.4e-3 (SR) and 7.5e-3 (PM) for the mean, and 1.2-1.3e-3 for the variance. IPA is infeasible at this target and beta on all 100 rounds, so it has no k-means result here.

**Further sweeps at r = 0.6** (the rate-sweep optimum, following the paper's "one optimal r"):
- beta 0.05-0.75: OPA-SR AG_mean 1.05e-3 falling to 5.6e-4; OPA-PM between -1.6e-4 and +4e-5; IPA (feasible only at beta >= 0.25) SR +9.5e-4 to +5.6e-4, PM ~0.
- Target mean -0.06 to +0.06: OPA-SR -1.2e-4 to +1.0e-3; OPA-PM within +-1e-4.
- Target variance 0.33-0.40: OPA-SR AG_mean 1.05e-3 throughout (the mean target is fixed); OPA-PM ~0.
- MSE_before for the mean is about 7.5e-3 throughout. Figure: `results/figures/mean_variance/originals/verify_kmeans_figures/fig12_kmeans_taxi.{svg,png}`.

**Not reproduced.** The paper shows positive gains peaking around r ~ 0.3. Here k-means removes at most about 14% of the mean damage (SR, r >= 0.6) and nothing measurable for PM.

**Why (diagnosed):**
- *PM.* Every subset carries about a beta share of the fakes, so all subset means share the attack's shift. Their distribution is symmetric (skew ~ 0), the clusters split about 50/50, and the chosen centre moves about 0.8 subset-sd to a random side (k-means shift mean ~0, about 50% downward). AG is therefore about 0 or slightly negative.
- *SR.* Subset means are hypergeometric counts of +1s. Hypergeometric skewness has sign (N - 2K)(N - 2n), so the majority cluster sits on a side fixed by sign(p - 1/2) * sign(r - 1/2). The k-means shift is deterministic (sd ~ 0): for r > 0.5 it pulls the estimate toward the domain midpoint, for r < 0.5 away from it, whatever the fakes are. On **clean rounds with no fakes**, SR k-means increases MSE: Taxi 1.27e-4 -> 3.38e-4 (r 0.3) / 1.42e-4 (r 0.6); Income 3.7e-6 -> 4.4e-6 / 6.8e-6. The positive AG on attacked Taxi at r >= 0.6 arises only because the attack pushes the estimate above the midpoint, near which Taxi's true mean (-0.026) happens to lie. It is not fake detection.
- The implementation matches scikit-learn's KMeans (k-means++, n_init 10) up to convergence tolerance and near-50/50 ties.

The paper's gains may depend on unstated details (fake spread, subset construction, cluster choice) and could not be recovered from the description.

## Datasets
| Source | Paper | Here | Status |
|---|---|---|---|
| Taxi | "mileage of taxi in a day", Dec 2020 | green taxi Dec 2020, column `DOLocationID`, domain [1, 265] | Table 3 reproduced through the NYC Open Data API (n 83,128 vs 83,130; mean 129.51; var 5,933). Table 3's mean_n = -0.022 disagrees with its own S1 = -2,194 (S1/n = -0.026) |
| Income | IPUMS 2019 ACS | Census 2019 ACS PUMS, `PINCP > 0`, domain [1, 1,629,000] | exact: n 2,390,203, S1_n -2,239,154, S2_n 2,115,290 (paper 2,115,289) |
| Retirement | SF employee compensation, n 97,220, mean 46,249 | **substitute**: `Retirement` column, Fiscal 2019-2020, n 86,904, declared domain [-2,000, 125,000] | No subset of the Jan 2021 Kaggle file reproduces Table 3. Searched: year spans × columns × filters, categories, per-employee aggregates, deduplication. Best relative-error score 0.64. Paper-target reproduction on Retirement is therefore not claimed; the `offset` target mode keeps the attacks meaningful |

## Proposed extensions (not in the paper)
- The breadth profile uses round sizes 2,048 / 4,096 / 8,192 with the total fixed per round (n = N - m), so the report count cannot reveal beta.
- Population variants: record-partitioned pools, subpopulation selection, and rank-based exponential and quadratic tilting.
- OPA-PM spread ratios of [0, 0.05, 0.10, 0.25, 0.50, 1.0] × plateau width.
- The breadth target mode is `offset` (see paper_settings.yaml).
- Recovery models (CatBoost, Deep Sets, Set Transformer), the hybrid loss and the classifiers.
