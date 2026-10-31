# `tests/mean_variance/`: unit and acceptance tests (104 tests)

Run them with `~/miniconda3/envs/ldp-attacker-detector/bin/python -m unittest discover -s tests/mean_variance -t .`
(standard-library `unittest`; pytest is not needed). Statistical checks use fixed seeds and ±4-5
standard-error tolerances. The tests that need data run on the Taxi file and are skipped if it is
absent.

| File | Tests | What it covers |
|---|---|---|
| `__init__.py` | – | Package marker |
| `test_transforms.py` | 11 | Round trips for g1 and g2 in both square modes; endpoints map to ±1; domains crossing zero; benchmark-scale moments, sums, query means; `raw_targets`; rejecting bad ranges and modes |
| `test_mechanisms.py` | 17 | SR: P(+1 \| x), the e^eps ratio, unbiasedness. PM: s, plateau width, mass inside [l, r], density ratio, chi-square fit, mean and variance (equals the Lemma 1 closed form). Phi decoding; domain checks; the factory |
| `test_estimator.py` | 10 | Exact halves; group routing; unbiased mean and E[x^2] without attack (both modes); variance identity; agreement with the paper's 2/(n+m) form; retained-subset estimate and fallback; `exact_moments` |
| `test_knowledge.py` | 7 | Table 4 n_e rounding; exact and compromised knowledge (unbiased; sampled users stay genuine); history; `with_error`; benchmark-scale error injection; `denorm_sums` |
| `test_feasibility.py` | 4 | Closed-form mean condition; `beta_min` is minimal; OPA needs fewer fakes than IPA, and PM fewer than SR; the checker agrees with the generators |
| `test_attacks.py` | 14 | Paper typos miss the target; OPA-SR and OPA-PM hit the target in expectation (both modes, both sum rules); SR integer counts; PM sum, range, spread and truncation; infeasible attacks are not generated; Thm. 1 is maximal (analytic equals constructed); IPA closed form and Adam; solver failure is flagged; IPA reaches the target through the randomizer |
| `test_budget.py` | 15 | Planned allocation; SR integer counts with approximation error; PM range and spread room; IPA joint (not separate) constraints; oracle labels; m_min is minimal and every rounded rung is rechecked; ladder rounding; no-fakes-needed; attacker estimate vs oracle; the budget is recorded before generation; fixed-beta below budget is neither generated nor raised; budget-ladder rounds; below-budget saturation is labelled separately; solver failure is not infeasibility; default fields keep round ids |
| `test_generation.py` | 12 | Fixed record partition; populations stay in their split; recipes vary moments; observation view keys; counts and labels; determinism and checksums; attack variants share genuine reports; clean counterpart; infeasible rounds; clean rounds; offset targets; store resume and crash recovery without duplicates |
| `test_features_methods.py` | 14 | PM features permute with the reports; identical SR cells give identical rows; no hidden inputs in feature signatures or names; standard method and projection; constant uses training data only; deterministic k-means; removal fallback; classifier scores permute and are equal within SR cells; set models are permutation-invariant with valid outputs; frozen checkpoint round trip; CSV append keeps one column layout; clean-round metric conventions; shift ratio |
