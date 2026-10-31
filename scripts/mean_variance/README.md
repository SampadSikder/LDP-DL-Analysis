# `scripts/mean_variance/`: command line

| File | What it contains |
|---|---|
| `run_server.sh` | Runs every experiment on the lab server (Linux, CUDA): preflight, smoke, paper attack validation, k-means verification, budget ladder, breadth grid with `evaluate` sharded over GPUs. Resumable. `nohup bash scripts/mean_variance/run_server.sh all > run_mv_all.log 2>&1 &`; see the header for stages and env overrides |
| `run.py` | The single CLI for the benchmark. Run it with `~/miniconda3/envs/ldp-attacker-detector/bin/python scripts/mean_variance/run.py <command> [--config ...]`. Outputs go to `results/mean_variance/<experiment>/` (git-ignored) |

| Command | What it does |
|---|---|
| `audit-data` | Compares source statistics with the paper's Table 3; writes `results/mean_variance/provenance_<source>.json` |
| `generate --config C [--splits ...]` | Generates the config's rounds (resumable); writes `coverage.csv` |
| `evaluate --config C [--methods ...] [--shard i/n]` | Trains once per (protocol, eps, training-source set) and evaluates every destination. Finished units are skipped; `--shard` splits units across processes |
| `report --config C` | `aggregate.csv`, `summary_by_method.csv`, and figures: beta sweeps, eps×beta heatmaps, transfer matrices, runtime, and utility vs m/m_min with per-rung outcomes for budget configs |
| `smoke --config C` | `generate`, then `evaluate`, then `report` |
| `evaluate-frozen --config C --checkpoints DIR` | Scores saved checkpoints on another config's test rounds (unseen eps, spread or attacks), with no refitting |
| `attack-validate --config C` | Paper-scale attack check: rounds are generated in memory and scored by MSE to the attacker's target |
| `feasibility-maps [--epsilon E]` | Oracle beta_min heatmaps (exact knowledge) |
| `budget-curves [--epsilon E]` | beta_min vs target mean and variance, attacker estimate vs oracle |
| `verify-kmeans --config C [--rates ...]` | Paper Sec. 8.1 k-means verification: AG per sampling rate, paired per round |
