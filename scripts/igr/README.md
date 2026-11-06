# IGR comparison (frequency oracles)

Item gain ratio (IGR) measures utility recovery when a detector is used as a defence: remove the users it
flags, re-estimate the frequencies from the remaining reports, and measure how much of the attack's
gain on the target items is left. The definition is from Li et al., *Mitigating Data Poisoning Attacks
to Local Differential Privacy* (CCS 2025, arXiv 2506.02156), computed the way their code does it
(`MDPA_LDP/Recovery.py`):

```
IGR = (sum_t f_hat_t - sum_t f_t) / (beta * r)
```

- `f_t` is the true frequency of target `t` among the genuine users.
- `f_hat` is the protocol's unbiased estimate from the kept reports, divided by the number of kept reports.
- `beta` is the share of fake users and `r` is the size of the target set.

How to read it:

| IGR | Meaning |
| --- | --- |
| about 1/r | the level of the baseline attack (fake users who just report a target honestly) |
| 0 | the attack's gain on the targets is gone |
| below 0 | genuine supporters of the targets were removed too |

The paper applies IGR to post-processing of the attacked estimate. Here it is applied to removal, for:

- our v3 FT-Transformer, in-distribution and cross-dataset;
- FIAD;
- Diffstats;
- two reference levels: no defence, and perfect removal (LDP noise only).

All of them are scored on the same simulated users. Each row also carries MSE over all items and the
detector's TP/FP/FN/TN on that run.

The datasets keep only features, not reports. So every held-out LDP run of a dataset is simulated again,
seeded exactly like `main.py --baseline`: the same true items and the same target set, with fresh LDP
noise. Our model reads features built from that simulated run through `main.py`'s own path: the v2
extractor, then `apply_feature_set_v3`, then the training directory's global z-score.

`collect` writes `igr_f1_check.csv`, which compares our F1 on the re-simulated runs with the stored
held-out predictions. It should agree up to LDP noise.

## On the server

```bash
nohup bash scripts/igr/run_igr_server.sh all > /dev/null 2>&1 &   # tests, norms, every run, collect, figures, pack
tail -f results/igr_logs/run_igr_server.log
grep -c "sim " results/igr_logs/run_shard*.log                     # finished runs per shard
```

Without FIAD (it takes most of the run time), keeping Diffstats:

```bash
BASELINES=diffstats SHARDS=16 nohup bash scripts/igr/run_igr_server.sh all > /dev/null 2>&1 &
```

This scores the saved `model.pt` checkpoints (no training) and Diffstats on the re-simulated held-out
runs, with the `no_defense` and `oracle` (true-label removal) rows beside them. `BASELINES=` (empty)
scores our models only. A later `run` that asks for FIAD simulates those runs again, so that every
detector in a run is scored on the same users.

- `SHARDS` (default 8) sets how many processes run in parallel. `PROC` (default 4) sets threads per shard.
- FIAD dominates the run time. Its earlier baseline pass took about 26 CPU-hours (4 threads per run) over the four MGA-A roots.
- Peak memory is roughly 15 GB per shard (emoji: 500k × 1496 reports).
- Restarting `run` skips the runs already finished.

### After Codex

IGR does not need the WSP add-on; only this `wsp` step does.

Run this once Codex has finished, `run_wsp_server.sh import_codex` has been run, and the WSP `baselines` stage (fine-tune and random) is done:

```bash
bash scripts/igr/run_igr_server.sh wsp collect figures pack
```

- `wsp` scores the WSP Stage-1 checkpoints (seeds 42–44) and the edited, fine-tune and random checkpoints.
- It uses the cached users of the cells the WSP runs test on: OUE on zipf, emoji and fire, and OLH-Server on fire, all on `oat_b015_mga-a_n500k_v3f`. So no new simulation or FIAD run is needed.

## On your machine

```bash
scp <server>:/home/boby/lab-work/LDP-DL-Analysis/results/igr_results.tar.gz . && tar xzf igr_results.tar.gz
python scripts/igr/igr_figures.py      # redraw from results/*/igr/igr_runs.csv
```

Figures go to `results/figures/frequency_oracle/igr/`:

- `igr_<root>_<PROTOCOL>.png`: one-at-a-time sweeps.
- `igr_<root>_default.png`: the default point, IGR and MSE.
- `igr_wsp.png`: the WSP runs.
- `tables/`: the same numbers as CSV.

## Files

- `attacker_detector/igr.py`: estimator, IGR and MSE, re-simulation, feature path, per-run cache.
- `scripts/igr/run_igr.py`: the `norms`, `run`, `wsp`, `collect` and `plan` commands.
- `scripts/igr/igr_figures.py`: figures and tables.
- `tests/igr/test_igr.py`: unit tests. Run with `python -m unittest discover -s tests/igr -t .`.

Outputs per root are in `results/<root>/igr/`:

- `runs/<Dataset>_<PROTOCOL>/*.csv`
- `cache/`: the WSP cells only, about 2 GB.
- `v3_norm_*.json`, `design_*.json`
- `igr_runs.csv`, `igr_f1_check.csv`
