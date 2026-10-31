# `attacker_detector/mean_variance/methods/`: utility-recovery methods

Every method follows `base.Method`. `fit(train, val)` receives `TrainingRound` objects (observation
plus labels). `predict(observation)` receives the observation only, except for `needs_labels`
references (oracle removal). Each method returns a `MethodOutput` holding raw moments, the
feasibility-projected moments, and optionally removed reports, scores and fallback groups.

| File | What it contains |
|---|---|
| `__init__.py` | Registry. `make_method(name, seed, cfg)` raises `MethodUnavailable` instead of substituting another method. `ALL_METHODS`; loss-variant names such as `deepsets_recovery[utility]` |
| `base.py` | `TrainingRound`, `MethodOutput` (projection applied on creation), `Method` (`fit`, `predict`, `timed_predict`, `predict_many` (batched; per-round runtime measured where each prediction runs), `state` (JSON of frozen choices)) |
| `baselines.py` | `StandardMethod`: ordinary decoded moments. `ConstantMethod`: average genuine moments of the training rounds. `OracleRemoval`: standard aggregation of the true genuine reports (an evaluation reference, not a lower bound). `KMeansRecovery`: paper Sec. 8.1. Per group it takes 5,000 subsets, decodes their means, runs 2-cluster 1-D k-means and keeps the majority centre. The sampling rate is chosen on at most 300 validation rounds; predictions use a process pool; SR uses an exact hypergeometric shortcut |
| `removal.py` | `ClassifierRemoval(arch='mlp'|'ft', feature_set='base'|'mech')`: detect, then remove, then re-estimate. Reuses the repo's `RobustAttackerDetector` and `FTTransformer` with the same feature schema. Training uses round-weighted BCE at natural prevalence. The removal threshold is chosen on validation utility (a grid that includes "remove nothing") and frozen. Re-estimation uses the retained group sizes with a fallback. Validation is batched (MPS-safe). Also `device_for` |
| `set_models.py` | Direct recovery with permutation-invariant set models. `DeepSetsNet` uses group-wise mean pooling; `SetTransformerNet` uses ISAB + PMA, with no position embeddings. Outputs are always valid (mean = tanh, second moment between mean^2 and 1). `SetRecovery` trains with the hybrid loss (utility; plus clean rounds; plus auxiliary per-report BCE), samples at most `token_cap` reports per round in training, and uses all reports at inference. Also `tokens_and_context` |
| `catboost_recovery.py` | `CatBoostRecovery`: two regressors (mean, second moment) on `features.round_summary`. Reports itself unavailable when `catboost` isn't installed |
| `checkpoint.py` | `save_checkpoint` writes `state.json` and `model.pt`. `load_checkpoint` rebuilds a frozen method (thresholds, rates, preprocessing, loss scales, weights) for `evaluate-frozen`, with no refitting |
