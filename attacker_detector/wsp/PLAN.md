# Weight-space programming (WSP) add-on — plan

LLM agent edits trained weights of four poisoning-defence models directly. Idea from
"Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab; MIT CSAIL, 2026-11-02,
https://rikiyat.github.io/blog/weight-space-programming/). Blog has no code or prompts; logs only
(github.com/RikiyaT/lm_understand-logs). Harness below is ours.

Status: built 2026-11-02 (run instructions: `scripts/wsp/README.md`). Differences from the plan text below:
- Frequency-oracle Stage 1, seed 42 = the existing runs (they already used `--epochs 50 --patience 10`);
  only seeds 43 / 44 are trained. `stage1_freq.py verify` checks every rebuilt split reproduces the saved
  test F1 before Stage 2.
- Mean/variance Stage 1 trains ε {1, 4, 6} only (the Stage-2 cells), all sources (for transfer).
- Pure mode is enforced by construction: the sandbox has numpy only (no torch).
- Resume uses the run's own state files (`state.json`, `latest.npz`, `edits.jsonl`), not a LangGraph
  SQLite checkpointer; a resumed agent starts a new conversation with the edit log.
- Fine-tune baseline skips the sparsity caps (gradient steps change every weight); same gate otherwise.
- Tests use `unittest` (as tests/mean_variance), 42 tests, no API calls.
- **Codex mode (2026-11-03), used because there are no API credits.** Codex (ChatGPT plan) is the agent, on
  the user's machine.
  - The server exports per run the Stage-1 weights plus exactly the probe subset and the validation data
    (`codex.py export_bundle`; no other training data, no test data).
  - `scripts/wsp/wspctl` exposes the same tools, gate, caps and budgets, one shell command at a time.
  - Tokens are attributed per run from Codex session logs, along with the weekly-limit %. `$` is the API
    list-price equivalent.
  - The integrity rules (no direct data access, no torch) are instructions plus a post-run audit of
    Codex's commands, not a hard sandbox. Report that in the paper.
  - Finished runs go back to the server for `eval` / `report`.

---

## 1. Rules

- **Add-on only.** Zero edits to existing files (main.py, run_server.sh, methods/, evaluate.py, figure
  scripts, configs). New code imports existing modules read-only. New outputs go to new folders. Delete
  the add-on folders → repo back to today.
- **Two stages.** Stage 1 = gradient training, max 50 epochs, existing training code. Stage 2 = LLM edits
  from Stage-1 weights, no gradients. Gradient does bulk work cheaply; tokens only buy refinement.
- **Test data never reachable in Stage 2.** Agent sees train rows/rounds via probes. Gate uses val.
  Test and unseen sources used once, by harness, at the end.
- **Shapes fixed.** Edits change values only. Edited checkpoints load with existing loaders.
- **Every token counted.** Cost logged per LLM call from API usage fields, not estimated.

## 2. Targets

| Target | Code | Weights | Prior result (figure) | Room |
|---|---|---|---|---|
| FT removal, mean/var | `methods/removal.py` `arch='ft'` | ~17.6k | ≈1.0× standard everywhere (01_overview) | Small: PM eps ≥ 1 only. SR = control (chance PR-AUC) |
| Deep Sets recovery | `methods/set_models.py` `arch='deepsets'` | ~42.5k | Clean PM eps 4/6 mean 2.8× / 7.4×; transfer 30–200× | Main target |
| Set Transformer recovery | `methods/set_models.py` `arch='settransformer'` | ~200k | Clean PM eps 4/6 mean 5.3× / 8.4×; transfer 30–200× | Main target |
| FT-Transformer, frequency oracle | `models/ft_transformer.py` via `main.py` | ~101k | OUE in-dist F1 ≈ 0.95–1.0; OUE cross→Fire gap −0.50 (fig3_gap, b015 mga-a v3f) | Cross OUE→Fire. In-dist = ceiling check. OLH-Server = control (at Bayes ceiling) |

Set models start from the `minimal` feature view (breadth baseline). `context` view = optional later.
HST-User excluded: sign-count leak still in generator.

## 3. Stage 1 — gradient training (≤ 50 epochs)

Existing code, new config and output folder only.

**Mean/variance** — `configs/wsp/stage1_mv.yaml`:
```yaml
extends: ../mean_variance/breadth.yaml
methods: [ft_removal, deepsets_recovery, settransformer_recovery]
classifier: {epochs: 50, reports_per_round: 512, patience: 5}
neural: {epochs: 50, token_cap: 1024, patience: 6}
```
Run with existing `scripts/mean_variance/run.py evaluate --config configs/wsp/stage1_mv.yaml
--results-root results/mean_variance_wsp/stage1`. Breadth's generated rounds reused read-only: symlink
`results/mean_variance_wsp/stage1/breadth/rounds` → `results/mean_variance_server/breadth/rounds`
(round IDs checked equal), so same train/val/test rounds as existing figures. `extends` replaces whole
top-level keys, so `classifier` / `neural` repeat every breadth value. Full breadth grid (2 protocols × 5 eps × 6 source sets × 3 seeds = 180
units per model) → within/cross/multi results for free. Output: checkpoints + per_round with `fit_time`.

**Frequency oracle** — `scripts/wsp/stage1_freq.py` calls `main.py` as subprocess with `--epochs 50`,
same other args as original runs, `--output-dir results/oat_b015_mga-a_n500k_v3f_wsp/stage1/...`,
seeds 0/1/2. Launcher writes args to `run_args.yaml` (main.py saves none). FT shape read back from
`model.pt` tensor shapes; head count from args.

Runs: OUE in-dist {Zipf, Emoji, Fire}, OUE cross {Zipf, Emoji, Fire}_test, OLH-Server cross Fire_test
(control).

Stage-1 cost logged: wall time, GPU-seconds, epochs run, optimizer steps, training FLOPs
(`torch.utils.flop_counter.FlopCounterMode` on one step × steps), peak memory.

## 4. Stage 2 — LLM weight editing

### Loop
```
Stage-1 checkpoint ─► agent (GPT-6 Astra, LangGraph)
   tools: model_card · tensor_stats · read_tensor · python (sandbox) · probe (budgeted)
                │ stages edit in sandbox
                ▼
   submit_edit(rationale) ─► gate (plain code): caps + shapes → score on val
                │ accept: new current weights │ reject: revert
                ▼
   stop: budget spent / N rejects in a row / finish() ─► save checkpoint ─► harness test eval
```

### Tools
| Tool | Returns | Cost |
|---|---|---|
| `model_card()` | arch, param table, feature names, task text, val metrics of current weights + standard estimator on same val rounds, budget left | free |
| `tensor_stats(names)` | mean/std/min/max, zero share, L2, top-5 singular values, dead units | free |
| `read_tensor(name, index)` | raw values, ≤ 2,048 numbers per call | free |
| `python(code)` | persistent sandbox; `W` = editable numpy copies of weights; numpy/torch CPU; no data, files, network, subprocess; 60 s timeout; output capped | free |
| `probe(kind, slice)` | forward pass on train-round/row subset with staged or current weights. `metrics` by slice (clean/attacked, eps, attack, β; freq: by config), `errors` (worst-k summary), `activations` (per-layer unit stats, pooled vectors) | 1 probe |
| `submit_edit(rationale)` | gate verdict, metric deltas, weights changed, relative L2 | 1 gate eval |
| `history()` | compact edit log | free |
| `finish(summary)` | ends run | — |

### Gate (accept only if all hold)
- Shapes unchanged; ≤ 5% of weights changed and relative L2 ≤ 0.1 per edit (configurable).
- Val score better, paired bootstrap over val rounds/rows, 90% lower bound > 0.
- Guards, none worse beyond tolerance:
  - DS / ST: val utility loss (same formula as `SetRecovery._val_utility`), clean and attacked rounds
    scored separately.
  - FT removal: threshold re-picked from existing grid after edit (same rule as `_select_threshold`);
    val post-removal utility + val BCE.
  - Freq FT: val F1 @ 0.5 (reported metric) + PR-AUC.
- Gate evals capped per run (default 10) to limit val overfitting.

### Pure mode (default)
Sandbox runs with grad disabled. Code with `backward`, `optim`, `.grad`, `autograd`, `open`, `socket`,
`subprocess` rejected before running. Every code cell logged.

### Token savers
1. Stage 1 does bulk optimization (user direction).
2. No whole-weight dumps: stats and slices on request; tool output capped (~2k tokens).
3. History compaction: last K turns kept verbatim; older turns replaced by harness-written edit log
   (deterministic, no LLM call).
4. Fixed system prompt + model card at start → provider prompt caching where supported.
5. Early stop: N rejected edits in a row, or no probe improvement in M probes.
6. Hard caps per run: tokens, wall clock. Global token cap across sweep (section 6).
7. Reasoning effort `xhigh` (user choice; Astra accepts low / medium / high / xhigh / max, no `none`).
   Reasoning tokens bill as output ($50/M) and count toward caps → biggest cost line; fewer LLM calls
   fit in each per-run cap. Reasoning items passed back between tool calls (Responses API) so the
   model doesn't redo its reasoning each step; they count as input, mostly cached (check in Phase 0).
8. Every request kept < 200K input tokens (hard limit in compaction): above 272K input, whole request
   reprices to $20 / $75 per M.

### Agent
- LangGraph tool-calling agent, `langchain-openai` chat model, OpenAI Responses API, model
  `gpt-6-astra`. Key and settings from `.env` at repo root (already in `.gitignore`, line 60):
  `OPENAI_API_KEY`, `WSP_MODEL=gpt-6-astra`, optional `WSP_BASE_URL`, `WSP_REASONING_EFFORT`.
  Loaded with python-dotenv; never printed or logged. `run_wsp_server.sh` refuses to start unless
  `git check-ignore .env` passes. Library versions pinned in Phase 0 after checking current API.
- System prompt: task (LDP protocol, attack types, what model outputs, loss/metric), architecture, rules
  (no gradients, edit caps, budget), tool list. No test results, no figure numbers. Agent finds failure
  modes itself from val/probe data.
- One agent first. Analyst + editor split only if Phase 3 logs show need.
- LangSmith tracing off (sends data outside).
- SQLite checkpointer → crashed run resumes.

## 5. Baselines (same Stage-1 start, same gate, same budget)

1. **Stage 1 alone** — "before" row.
2. **Gradient fine-tune** — extra epochs from Stage 1, same gate and forward-pass budget.
3. **Random edit search** — random perturbations (same edit caps), same gate and budget, $0 LLM.
4. **Controls** — SR FT removal; freq OLH-Server. Gain there = gate overfitting or leak → investigate.
5. **Reference** — existing breadth / OAT results (30–40 epochs, early stop) shown in figures, not
   used for paired claims.

## 6. Cost accounting

Per LLM call (JSONL): timestamp, model, input tokens, cached input tokens, output tokens, reasoning
tokens, latency, tool calls. From API usage metadata.

Per run `cost.json`:

| Group | Fields |
|---|---|
| LLM | calls, input (uncached / cached), output, reasoning tokens, $ (from `configs/wsp/pricing.yaml`, below), LLM wait time |
| Harness | probes, gate evals, rows/rounds evaluated, forward FLOPs, sandbox CPU-s, GPU-s, wall time |
| Edits | submitted, accepted, weights changed, relative L2 |
| Derived | $ per run, $ per accepted edit, tokens per 1% val-loss drop, $ per 1% test-loss drop |

Same table for Stage 1, fine-tune, random search (tokens = 0). LLM FLOPs unknown (closed model) →
cost unit for Stage 2 = tokens and $, for gradient = GPU-s and FLOPs; both reported side by side, not
converted.

### Prices — `configs/wsp/pricing.yaml` (checked 2026-11-02)

| Item | $ / 1M tokens | Source |
|---|---|---|
| Input | 10 | OpenAI GPT-6 Astra announcement |
| Output (incl. reasoning) | 50 | OpenAI GPT-6 Astra announcement |
| Cached input | 1 | aggregators only — confirm on OpenAI pricing page |
| Long context (> 272K input, whole request) | 20 in / 75 out | one aggregator — avoided by design (< 200K) |
| Fast mode | 2× | OpenAI announcement — not used |
| Batch / Flex | 0.5× | aggregators — batch unsuited to agent loop; check Flex availability |

Context window 1.05M, max output 128K (third-party gateway listing). Rate limits: from org dashboard.

### Token budget (user's weekly quota)

- 138.8M tokens = 9% used → weekly quota ≈ **1.54B tokens**.
- Cap = 50% of weekly quota ≈ **771M tokens/week**. In the current week, 9% already used → WSP limit
  **633M** so the week stays ≤ 50%. Default config value: 633M (`WSP_TOKEN_CAP`).
- Counted as all tokens (input + cached + output + reasoning), the conservative reading. If the quota
  weighs cached tokens less, real usage is lower.
- Calibration: note dashboard % before and after pilot → tokens per 1% measured, cap corrected.
- Harness stops launching runs at 90% of cap; running runs finish under their own caps.

### Per-run caps and sweep total

| Probe budget | Per-run token cap | Runs | Total |
|---|---|---|---|
| 10 | 3M | 9 (budget curve) | 27M |
| 25 | 6M | 54 Tier A + 15 freq | 414M |
| 50 | 10M | 9 (budget curve) | 90M |
| pilot | 6M | 2 | 12M |
| **Sum** | | 89 | **543M** (≤ 633M; ~90M left for reruns) |

$ equivalent at list prices (paper reporting only): with `xhigh`, output/reasoning share is high →
guess ~$40–100 per 6M-token run → roughly $3.5k–9k for the sweep. Pilot replaces this guess.

If pilot runs hit their token cap before using their probe budget, caps are not raised. Grid shrinks
instead, in this order: (1) budget-50 runs, (2) frequency in-dist ceiling check, (3) controls (SR FT
removal, OLH-Server) down to 1 seed. Total stays ≤ 633M.

Run hitting its cap stops, keeps best accepted weights, records `stopped: budget`. Tier B (+108 runs)
only in a later week, after pilot calibration.

## 7. Experiment grid

**Pilot:** DS, PM eps 4, taxi, seed 0, budget 10 probes. Measures tokens/$ per run.

**Tier A (main):** taxi within-source; ft_removal, deepsets_recovery, settransformer_recovery × PM/SR ×
eps {1, 4, 6} × seeds {0, 1, 2} = 54 runs, budget 25 probes.

**Frequency oracle:** b015 mga-a v3f; OUE cross ×3, OUE in-dist Fire ×1, OLH-Server cross Fire ×1
(control) × seeds {0, 1, 2} = 15 runs.

**Budget curve:** PM eps 4 taxi, 3 models × budgets {10, 50} × 3 seeds = 18 extra runs (25 from Tier A).

**Tier B (if budget allows):** income + retirement within-source = +108 runs. APA regime for freq.

Every edited model evaluated on all test sources (within, cross, multi) through existing
`load_checkpoint` + `rounds_from_store` + `_row`, renamed `*_wsp` so rows never mix with baselines.
Agent never sees other sources → transfer = honest held-out test.

## 8. Comparison figures

Output: `results/figures/wsp/` (PNG) + `tables/` (CSV). Script `scripts/wsp/wsp_figures.py` imports
style and helpers from `summary_figures.py` / `selected_figures.py` (`Out`, palette, `asr_rates`,
`load_asr`); method lists extended in memory at runtime, files untouched. Freq figures load
add-on output with existing `scripts/plots/loader.load_oat(root)` (same folder layout).

| WSP figure | Based on existing | Shows |
|---|---|---|
| `wsp_01_overview` | `mean_variance_server/01_overview_ratio_vs_standard.png` | MSE ÷ standard heatmap; rows: Stage 1, +fine-tune, +random, +WSP per model; eps {1, 4, 6} |
| `wsp_04_transfer` | `mean_variance_server/04_transfer_within_cross_multi.png` | do edits learned on taxi help or hurt other sources |
| `wsp_mse_vs_eps` | `mean_variance_selected/mse_vs_eps.png` | dashed = Stage 1, solid = WSP, per model |
| `wsp_asr_vs_eps` | `mean_variance_server/12_attack_success_vs_eps.png`, `selected/attack_success_vs_eps.png` | attack success before/after |
| `wsp_detection` | `mean_variance_server/05_detection.png` | FT removal PR-AUC, precision, recall, removal rate |
| `wsp_overall_table` | `mean_variance_selected/overall_table.png` | one-table summary |
| `wsp_freq_cross` | `frequency_oracle/originals/oat_b015_mga-a_n500k_v3f/fig6_cross_f1_score.png` | OUE cross F1 vs ε/β/r', Fire panel main |
| `wsp_freq_gap` | same folder, `fig3_gap_oat_b015_mga-a_n500k_v3f.png` | generalization gap, Stage 1 vs WSP |
| `wsp_freq_indist` | same folder, `fig1_OUE_f1_score.png` | ceiling check; FIAD/Diffstats from `baselines_used.csv` |
| `wsp_freq_control` | same folder, `fig1_OLH_Server_f1_score.png` | control vs Bayes ceiling |
| `wsp_cost` | new (style of `11_runtime.png`) | (a) Stage-1 GPU-s vs Stage-2 wall time; (b) tokens per run by type; (c) test gain vs $ per run, baselines at $0 |
| `wsp_budget_curve` | new | test gain vs probe budget {10, 25, 50} and vs tokens |
| `wsp_edit_anatomy` | new | share of weights edited per layer, relative L2, accept rate |

Tables: same columns as `tables/ratio_vs_standard_within.csv`, `transfer_within_cross_multi.csv`,
`detection.csv`, `selected/tables/overall.csv`, plus `wsp_cost.csv`.

## 9. Files (all new)

```
attacker_detector/wsp/
  PLAN.md          this file
  adapters.py      FTRemovalAdapter, SetRecoveryAdapter (DS + ST), FreqFTAdapter: load, params, probe, gate score, save
  sandbox.py       subprocess kernel, blocked calls, timeout, output cap
  tools.py         LangChain tools (section 4)
  gate.py          accept rule, bootstrap, caps
  budget.py        probes, gate evals, token/$/time caps
  cost.py          usage callback, pricing, cost.json, FLOP counting
  agent.py         LangGraph graph, prompts, history compaction
  baselines.py     gradient fine-tune, random search
  frozen_eval.py   test eval of edited checkpoints, *_wsp naming
  log.py           JSONL transcript (prompts, replies, code, outputs, diffs; key redacted)
scripts/wsp/
  check_api.py     one tiny call: model id, usage fields present
  stage1_freq.py   main.py launcher, run_args.yaml
  run_wsp.py       one run: --target {mv_ft,mv_ds,mv_st,freq_ft} --ckpt ... --budget 25 --mode pure
  run_wsp_server.sh  stages: preflight, stage1, baselines, pilot, sweep, eval, figures, cost (resumable)
  compare_wsp.py   paired Stage 1 vs fine-tune vs random vs WSP
  wsp_figures.py   section 8
  cost_report.py   wsp_cost.csv + figure
configs/wsp/       stage1_mv.yaml, stage2.yaml (budgets, caps, gate), pricing.yaml, freq_runs.yaml
tests/wsp/         section 10
requirements-wsp.txt
```

## 10. Tests (fake LLM, no API cost)

- Round-trip per adapter: load → save → load → identical predictions.
- No-op edit → gate metrics equal Stage-1 val metrics.
- Sandbox: blocked calls rejected; timeout; output cap.
- Leak: Stage-2 adapters hold no test handle; probe/gate refuse test split.
- Budget: probe/gate counts; token cap aborts run, best weights kept.
- Gate: rejects worse edit, shape change, over-cap edit.
- Cost: fake chat model with known usage → totals and $ exact.
- End-to-end dry run with scripted fake agent on tiny rounds (like `check_cuda.tiny_rounds`).

## 11. Server

```bash
conda create --clone attacker-detector -n attacker-detector-wsp     # existing env untouched
/home/boby/miniconda3/envs/attacker-detector-wsp/bin/pip install -r requirements-wsp.txt
# key: create .env at repo root on the server (gitignored; never rsync/commit it), chmod 600:
#   OPENAI_API_KEY=...   WSP_MODEL=gpt-6-astra   WSP_REASONING_EFFORT=xhigh   WSP_TOKEN_CAP=633000000
python scripts/wsp/check_api.py                                    # outbound HTTPS + usage fields
nohup bash scripts/wsp/run_wsp_server.sh all > /dev/null 2>&1 &
tail -f results/mean_variance_wsp/run_wsp_server.log
```
LLM calls are I/O bound: runs in parallel up to `WSP_CONCURRENCY` (API rate limit), forward passes on
CPU/GPU. Outputs: `results/mean_variance_wsp/`, `results/oat_b015_mga-a_n500k_v3f_wsp/`. Copy
`cost.json`, per_round, figure tables back like before.

## 12. Phases

| Phase | Work | Exit check |
|---|---|---|
| 0 Setup | env clone, deps pinned, key file, `check_api.py`, pricing filled | one call returns usage |
| 1 Stage 1 + harness | stage-1 runs (MV + freq), adapters, gate, sandbox, fine-tune + random baselines, Stage-1 cost | tests pass; Stage-1 numbers near breadth/OAT figures |
| 2 Agent + pilot | tools, agent, cost logging; fake-LLM dry run; real pilot (DS PM eps 4) | tokens/$ per run measured → **user approves sweep size** |
| 3 Sweep | Tier A + freq (+ budget curve) | all runs done or capped |
| 4 Eval + report | frozen test eval, `compare_wsp.py`, figures, cost report | figures in `results/figures/wsp/` |
| 5 Optional | Tier B, APA regime, `context` view start, analyst/editor split, zero-init mode (blog Exp 5/6) | — |

## 13. Expected results and risks

- Blog: edits remove obstacles, can't add missing ability. No gain expected where data lack signal:
  SR detection (chance), server-side freq protocols (Bayes ceiling), OUE in-dist (≈1.0).
- Best chances: DS/ST clean-round bias at eps 4/6 (standard mean/variance already a context input —
  a near-identity path to it is a plausible edit); FT-removal score calibration at PM eps ≥ 1.
- Transfer gains unlikely: gate sees one source only. Reported honestly either way.
- Gate overfits val with many accept/reject steps → bootstrap bound, gate-eval cap, test used once.
- Cost: agent resends history each call → input tokens grow with steps; compaction + caps bound it.
  Real numbers come from pilot, not guesses.
- LLM output varies run to run → 3 seeds per cell, report spread.

## 14. Decisions

Settled:
- Model `gpt-6-astra` via OpenAI Responses API, reasoning effort `xhigh`; key in gitignored `.env`;
  prices in section 6.
- Token cap: 50% of weekly quota (≈ 771M/week; 633M this week). Per-run caps in section 6.
- Tier A + freq + budget curve this week (89 runs, ~543M); Tier B later.
- Stage 1 up to 50 epochs before LLM editing.

Open (defaults used unless changed):
1. Stage 1: existing early stopping, best-val weights kept (default), or force all 50 (`patience: 50`).
2. Pure mode (default) or hybrid (agent may request gradients) as later ablation.
3. Freq regime: b015 mga-a v3f only (default) or + APA.
4. Rate limit (RPM/TPM) for your org → sets `WSP_CONCURRENCY`.
