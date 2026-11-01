# `scripts/mean_variance/`: command line

| File | What it contains |
|---|---|
| `run_server.sh` | Runs every experiment on the lab server (Linux, CUDA): preflight, smoke, paper attack validation, k-means verification, budget ladder, breadth grid with `evaluate` sharded over GPUs. Resumable. Every output, including its own console log `run_server.log`, goes to `results/mean_variance_server/` (`RESULTS=` overrides). `nohup bash scripts/mean_variance/run_server.sh all > /dev/null 2>&1 &`; see the header for stages and env overrides |
| `check_cuda.py` | CUDA preflight: GPUs visible to torch, kernels run on each GPU, `device_for('auto')` and every neural method in the config pick cuda, MLP / FT / Deep Sets / Set Transformer each train one epoch on GPU, and each `CUDA_VISIBLE_DEVICES=i` shard sees its own GPU. Exit 0 = pass. Also run by `run_server.sh preflight` / `run_server.sh cuda` |
| `eta.py` | Progress and finish-time estimate for a sharded `evaluate` run (default: breadth in `results/mean_variance_server`). Calibrates the Mac pilot's per-unit costs against finished units' logged times, per method and protocol; lists failed units (they are marked done and skipped on rerun). Read-only |
| `compare_mech.py` | Base vs `mech` features for MLP / FT removal, paired on the same PM test rounds: change in mean / variance MSE (% of base, population-bootstrap 95% CI) by view (within / cross / multi-source) and attack, clean-round damage against the standard method, per eps, and per transfer pair (e.g. taxi -> income). Writes `<experiment>/mech_comparison.csv`. Run by `run_server.sh mech` |
| `run.py` | The single CLI for the benchmark. Run it with `~/miniconda3/envs/ldp-attacker-detector/bin/python scripts/mean_variance/run.py <command> [--config ...]`. Outputs go to `results/mean_variance/<experiment>/` (git-ignored); `--results-root DIR` moves every output of a command under DIR |

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
