# Data dictionary: SR/PM utility-recovery benchmark

## Source files (`datasets/mean_variance/`, git-ignored)
| Source | Expected file(s) | Column / filter | Declared domain | Where to get it |
|---|---|---|---|---|
| taxi | `taxi/green_tripdata_2020-12.parquet` (or a `.csv` export with the same columns) | `DOLocationID`, no filter | [1, 265] | NYC TLC trip records page, green taxi, 2020-12 |
| income | `income/psam_pusa.csv`, `income/psam_pusb.csv` | `PINCP > 0`, raw (no ADJINC) | [1, 1,629,000] | Census 2019 ACS 1-year PUMS, `csv_pus.zip` |
| retirement | `retirement/employee-compensation.csv` | `Retirement`; Year Type = Fiscal, Year in {2019, 2020} | [-2,000, 125,000] | Kaggle `san-francisco/sf-employee-compensation` (Jan 2021 snapshot) |

`python scripts/mean_variance/run.py audit-data` prints each source's statistics next to Table 3 and writes `results/mean_variance/provenance_<source>.json` (file SHA-256, column, filter, domain, record count). Covariates (taxi weekday, hour bin, payment type, vendor; income state, age bin, sex; retirement organisation group, year) are used only to build subpopulations and never reach a model.

## Generated rounds (`results/mean_variance/<experiment>/rounds/`)
### Scenario manifest (`manifest.jsonl`, one JSON object per spec; orchestration only)
| Field | Meaning |
|---|---|
| `round_id`, `base_round_id`, `clean_counterpart_id` | Pairing ids. Variants that differ only in attack or target share `base_round_id` and identical genuine reports. `clean_counterpart_id` is the genuine-only view of the same round |
| `experiment`, `split`, `source`, `population_id`, `rep` | Provenance. `population_id` fixes the parent population (recipe in `PopulationRecipe`) |
| `protocol`, `epsilon`, `domain`, `square_mode` | Mechanism and query convention |
| `total_size`, `size_mode`, `beta` | Requested size and beta. `fixed_total`: N fixed, m = round(beta N). `fixed_genuine`: n fixed, m = round(beta n / (1 - beta)) |
| `attack`, `target`, `target_mode`, `target_mean_value`, `target_var_value` | Attack family; target label (`m{i}v{j}`, `clean`, or `explicit`) |
| `knowledge_regime`, `knowledge_h`, `n_e_rule`, `dS1_n`, `dS2_n`, `dn` | Attacker knowledge and injected estimation errors (benchmark-scale sums) |
| `spread_ratio`, `ipa_solver`, `sum_rule` | Attack parameters |
| `status` | Round outcome: `feasible`, `generated_with_target_error`, `infeasible_target` (theory says no; not generated), `allocation_infeasible` (planned feasible, realized group split not), `solver_failure`, `below_budget` (separate below-budget evaluation), `no_fakes_needed`, `no_budget_exists` |
| `budget_mode`, `budget_factor`, `below_budget` | `beta` (m from beta) or `mmin_multiple` (m = ceil(factor * m_min)); `skip` or `saturate` |
| `meta.n_genuine`, `meta.n_fake`, `meta.realized_beta`, `meta.m_g1`, `meta.m_g2`, `meta.n_g1_genuine`, `meta.n_g2_genuine` | Realized counts |
| `meta.target_mean`, `meta.target_variance`, `meta.target_second_moment`, `meta.target_valid_population` | Resolved targets, q_t = mu_t^2 + sigma_t^2, and whether the targets describe a valid population (separate from attack feasibility) |
| `meta.budget` | Budget stage, computed before generation: `m` (actual budget), `m_min` / `beta_min` / `plan_status` / `n_planning` (attacker's estimates), `oracle_m_min` / `oracle_beta_min` (exact genuine information, **oracle diagnostic**), `budget_ratio` = m / m_min, `oracle_budget_ratio`, `mode`, `factor` |
| `meta.theory` | Theoretical check at m with the planned allocation: `status` (feasible / infeasible / no_fakes_needed / no_budget_exists), per-group required sums, caps, margins, SR integer counts, PM spread room, IPA A / B / B_min / B_max, `approx_error_mean` / `approx_error_variance` (SR rounding) |
| `meta.construction` | Generator outcome, separate from theory: `status` (constructed, constructed_with_rounding_error, allocation_infeasible, solver_failure, below_budget_saturated, not_attempted), `support_ok`, per-group unbiased-sum residuals (OPA) or Sum y / Sum y^2 residuals and convergence (IPA) |
| `meta.realized` | `target_error_mean`, `target_error_variance`: the final standard estimate minus the target, after genuine noise and knowledge error (never used to discard a round) |
| `meta.knowledge` | n_e, h, S1_e, S2_e actually used |
| `meta.attack_info` | Requirements (R_g, U_g, A, B), residuals, spread (nominal, achievable, realized, truncated), solver convergence |
| `meta.checksum` | SHA-256 prefix of (reports, groups, is_fake); regeneration must match it |
| `labels.*` | Label view: `genuine_mean`, `genuine_second_moment`, `genuine_variance` (benchmark scale, ddof = 0) and their `_raw` versions; `clean_mean` / `clean_variance` (oracle removal); `standard_mean` / `standard_variance` (the server's attacked estimate) |

### Observation shards (`<split>/<protocol>_eps<e>_<source>.npz`)
| Array | Content |
|---|---|
| `reports` (float32) | Shuffled privatized values of every generated round, concatenated |
| `groups` (int8) | 1 = g1 (reports x), 2 = g2 (reports the second-moment query) |
| `is_fake` (bool) | **Label**. Loaded separately and never passed to `predict` |
| `offsets`, `round_ids` | Round boundaries |

A model receives `Round.observation()` / `TrainingRound.observation`: `reports`, `groups`, `n_g1`, `n_g2`, `protocol`, `epsilon`, `domain`, `square_mode`. Nothing else.

## Results (`results/mean_variance/<experiment>/`)
| File | Content |
|---|---|
| `per_round/<unit>.csv` | One row per (method, seed, test round): predicted moments raw and projected, squared errors against the genuine moments, the standard estimate's errors (for AG), target, runtime, fallback groups, retained group sizes, classifier diagnostics, fit time, peak RSS. Budget metadata (evaluation only, never a model input): `m_min`, `beta_min`, `oracle_m_min`, `oracle_beta_min`, `budget_ratio`, `oracle_budget_ratio`, `n_fake`, `theory_status`, `construction_status`, `realized_target_error_mean`, `realized_target_error_variance`, `budget_mode`, `budget_factor` |
| `units/<unit>.done.json` | Unit status (`completed`, `unavailable`, `failed`), so finished units are skipped on resume |
| `checkpoints/<unit>/state.json`, `model.pt` | Frozen thresholds and rates, preprocessing (feature mean/std), loss scales, training history, weights |
| `aggregate.csv` | Per cell: MSE (mean, variance) with population-clustered bootstrap 95% CIs, AG_mean, AG_var, median and p90 absolute error, fraction of rounds made worse, fallback rate, runtime, classifier diagnostics |
| `coverage.csv` | Every spec with its generation status (infeasible cells stay visible) |
| `figures/*.svg`, `*.png`, `*.manifest.json` | Figures and the exact inputs and filters behind each |
