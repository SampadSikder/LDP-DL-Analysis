# `configs/mean_variance/`: experiment configurations

Each YAML file is one experiment, run with `scripts/mean_variance/run.py <command> --config <file>`.
Its outputs go to `results/mean_variance/<experiment>/`. Paper-faithful settings and proposed
extensions live in separate files. `paper_settings.yaml` classifies every value as paper-stated,
inferred-from-figure or proposed-extension.

| File | Command | What it is |
|---|---|---|
| `paper_settings.yaml` | (reference) | Every protocol, attacker, dataset, sweep, target and budget-stage setting, with its classification and its section, table or figure in the paper |
| `smoke.yaml` | `smoke` | Small run that touches every component end to end (SR+PM, eps 1, 3 sources, 3 betas, all methods, 4 epochs). Not a result |
| `breadth.yaml` | `generate` / `evaluate` / `report` | The full breadth grid from the brief: 2 protocols × 3 sources × 5 eps × 10 betas × 2 attacks × 9 targets, 20/10/100 rounds, sizes 2,048 / 4,096 / 8,192, 3 seeds. Configured but not run (estimated at about 7 days sequential) |
| `breadth_mech.yaml` | `evaluate` (after breadth `generate`) | **Ablation.** `extends: breadth.yaml`, PM only, methods `mlp_removal_mech` / `ft_removal_mech` (base features + residuals against a sum-to-one NNLS fit of PM output densities). Same experiment name, so it reuses breadth's rounds; compare with `scripts/mean_variance/compare_mech.py`. Run on the server with `run_server.sh mech` |
| `breadth_setfeat.yaml` | `evaluate` (after breadth `generate`) | **Ablation.** `extends: breadth.yaml`, PM and SR, methods `deepsets_recovery_context` / `settransformer_recovery_context`: the original tokens plus a richer observation-only round context (PM per-group quantiles, histogram, narrowest-density stats; SR sign shares). Same rounds, seeds, neural settings and loss as breadth; the `minimal` baseline is breadth's own `deepsets_recovery` / `settransformer_recovery`. Compare with `scripts/mean_variance/compare_setfeat.py`; run with `run_server.sh setfeat` |
| `breadth_setfeat_report.yaml` | `evaluate` (after breadth `generate`) | **Ablation.** PM only, methods `*_recovery_context_report`: the `context` inputs plus per-report PM features on each token. Not run for SR, where the view is an alias of `context` |
| `budget_sweep.yaml` | `smoke` | **Proposed extension.** Budget ladder m = ceil(f * m_min), f in {1, 1.1, 1.25, 1.5, 2}, with a fixed genuine population (n = 4,096) |
| `budget_sweep_setfeat.yaml` | `evaluate` (after budget_sweep) | Adds FT-Transformer removal and Deep Sets with group context (`deepsets_recovery_context`) to the budget ladder. `extends: budget_sweep.yaml`, same experiment, rounds and settings. Run with `run_server.sh budget_extra` (also scores their frozen checkpoints on below-budget rounds, then reruns `asr`) |
| `below_budget.yaml` | `generate --splits test`, then `evaluate-frozen` | **Separately labelled** below-budget run: f in {0.5, 0.75, 0.9}, saturated best effort, scored with frozen `budget_sweep` checkpoints |
| `paper_attack_default.yaml` | `attack-validate` | Paper defaults at full scale (Taxi, Income): eps 1, beta 0.1, Table 4 targets, 100 repetitions |
| `paper_attack_eps.yaml` | `attack-validate` | Fig. 10 analogue: eps in {0.1, 1, 2, 4, 6} |
| `paper_attack_table4_taxi.yaml` | `attack-validate` | Taxi with the paper's fixed Table 4 attacker knowledge (as errors against the truth) |
| `paper_attack_S1_taxi.yaml` | `attack-validate` | Fig. 7 analogue: attacker error in S_e(1) |
| `paper_attack_S2_taxi.yaml` | `attack-validate` | Fig. 8 analogue: attacker error in S_e(2) |
| `paper_attack_ne_taxi.yaml` | `attack-validate` | Fig. 9 analogue: attacker error in n_e |
| `paper_attack_beta_taxi.yaml` | `attack-validate` | Fig. 11 analogue: beta in {0.05, 0.25, 0.5, 0.75} |
| `verify_kmeans_rate_taxi.yaml` | `verify-kmeans` | Fig. 12 (left): k-means sampling rate 0.1-0.9 |
| `verify_kmeans_beta_taxi.yaml` | `verify-kmeans` | Fig. 12 (middle): beta sweep at r = 0.6 |
| `verify_kmeans_mean_taxi.yaml` | `verify-kmeans` | Fig. 12 (right, top): target-mean sweep at r = 0.6 |
| `verify_kmeans_var_taxi.yaml` | `verify-kmeans` | Fig. 12 (right, bottom): target-variance sweep at r = 0.6 |

Also in this folder but git-ignored (one-off): `pilot.yaml` (runtime pilot: PM, eps 1, Taxi),
`smoke_shift_eps2.yaml` (frozen-checkpoint check at an unseen eps), and `paper_attack_default_taxi.yaml`
(a Taxi-only copy of `paper_attack_default.yaml`).

Common keys:
- `extends: other.yaml` (relative to this file): load that config first, then override its top-level keys
- `experiment`, `protocols`, `sources`, `epsilon`, `beta`, `attacks`
- `targets`: `m{i}v{j}` from the printed triplets
- `rounds` (per split), `sizes`, `size_mode` (`fixed_total` | `fixed_genuine`)
- `target_mode` (`offset` | `absolute`), `square_mode`
- `knowledge_regime`, `knowledge_h`, `n_e_rule`, `dS1_n` / `dS2_n` / `dn`
- `pm_spread_ratio(s)`, `ipa_solver`, `sum_rule`
- `sweep:` (one factor), `budget_sweep:` (ladder)
- `methods`, `seeds`, `multi_source`, `kmeans_subsets` / `kmeans_rates`, `classifier:`, `neural:`, `n_jobs`
