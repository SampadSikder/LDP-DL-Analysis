# Weight-space programming (WSP) add-on

An LLM agent (GPT-6 Astra, LangGraph) edits the trained weights of four models directly, with no
gradients. It is compared against gated gradient fine-tuning and gated random edits, all starting from
the same Stage-1 weights. Design and reasoning: `attacker_detector/wsp/PLAN.md`.

This is an add-on. No existing file is changed. Existing results are only read. To remove it, delete
`attacker_detector/wsp/`, `scripts/wsp/`, `configs/wsp/`, `tests/wsp/`, `requirements-wsp.txt` and the
`results/*wsp*` folders.

## On the server

```bash
git pull                                              # code only; results/ and datasets/ stay as they are
bash scripts/wsp/run_wsp_server.sh setup              # once: clone conda env -> attacker-detector-wsp, pip install
nano .env && chmod 600 .env                           # see below; git-ignored
nohup bash scripts/wsp/run_wsp_server.sh all > /dev/null 2>&1 &
tail -f results/wsp/run_wsp_server.log
```

`all` runs preflight, Stage 1 (mean/variance and frequency oracle), both baselines and the 2 pilot LLM
runs, then **stops**. Read the pilot projection at the end of the log, compare it with your OpenAI
dashboard %, then:

```bash
nohup bash scripts/wsp/run_wsp_server.sh wsp eval report > /dev/null 2>&1 &
```

`.env` (repo root, never committed, never printed):

```
OPENAI_API_KEY=sk-...
WSP_MODEL=gpt-6-astra
WSP_REASONING_EFFORT=xhigh
WSP_TOKEN_CAP=633000000
# optional: WSP_CONCURRENCY=4 (parallel LLM runs), WSP_LEDGER_SINCE=2026-11-09 (count tokens from the
# weekly reset), WSP_REASONING_SUMMARY=auto (log reasoning summaries), WSP_SANDBOX_PREFIX="firejail --net=none --quiet"
```

## Stages

| Stage | What | Tokens |
|---|---|---|
| `setup` | conda env clone + `requirements-wsp.txt` | 0 |
| `preflight` | `.env` present and git-ignored, packages, `tests/wsp`, one tiny API call | a few thousand |
| `stage1_mv` | existing `run.py evaluate` with `configs/wsp/stage1_mv.yaml` (up to 50 epochs) on breadth's rounds | 0 |
| `stage1_freq` | `main.py` (unchanged) for seeds 43 / 44; seed 42 = the existing runs; checks every run reproduces its saved test F1 | 0 |
| `baselines` | gated fine-tune + gated random search, 87 units | 0 |
| `pilot` | 2 LLM runs, then a projected total for the sweep against the cap | ~9M max |
| `wsp` | 87 LLM runs (54 mean/variance tier A, 18 budget curve, 15 frequency oracle); stops launching at 90% of the cap | ≤ 543M planned |
| `eval` | test evaluation of every variant; existing `attack_success.py` on the result | 0 |
| `report` | `cost_report.py`, `compare_wsp.py`, `wsp_figures.py` | 0 |
| `tier_b` | later week(s): income / retirement units, 108 LLM runs | up to 648M at the per-run caps; spans weeks |

Every stage can be resumed. A crashed or killed LLM run continues from its last accepted weights.
Tokens it already spent stay in the ledger. A run that reaches the weekly cap pauses (`paused.json`)
instead of finishing. After the weekly reset, set `WSP_LEDGER_SINCE` to the reset date in `.env` and
re-run the same stage.

## Where things go

| Path | Content |
|---|---|
| `results/mean_variance_wsp/stage1/breadth/` | Stage-1 checkpoints, per_round, aggregate (rounds linked from breadth) |
| `results/mean_variance_wsp/stage2/<mode>_b<budget>/<unit>/` | one run: `checkpoint/`, `wsp.json`, `cost.json`, `edits.jsonl`, `transcript.jsonl`, `llm_calls.jsonl` |
| `results/mean_variance_wsp/eval/` | per_round test rows of every variant + `attack_success.csv.gz` |
| `results/oat_b015_mga-a_n500k_v3f_wsp/` | frequency oracle: `stage1_s43/`, `stage1_s44/`, `stage2/`, `eval/<variant>_s<seed>/`, `cache/` |
| `results/wsp/` | `ledger.jsonl` (every LLM call), `cost_summary.csv`, `logs/` |
| `results/figures/wsp/` | figures (PNG) + `tables/` |

Copy back for plotting locally: `results/figures/wsp/`, `results/wsp/cost_summary.csv`, and the
`transcript.jsonl` files if you want the agent logs.

## Single runs and checks

```bash
PY=/home/boby/miniconda3/envs/attacker-detector-wsp/bin/python
$PY scripts/wsp/run_wsp.py list --set tier_a --mode wsp          # todo / done
$PY scripts/wsp/run_wsp.py job --id mv/wsp_b10/single-source__PM_eps4__taxi__deepsets_recovery__s0
$PY scripts/wsp/sweep.py wsp --dry-run                           # what would run
$PY scripts/wsp/cost_report.py --pilot                           # tokens so far, projection vs cap
$PY -m unittest discover -s tests/wsp -t .                        # 35 tests, no API calls
```

## Cost accounting

- **LLM:** every call records input, cached, output and reasoning tokens from the API's usage fields, plus
  latency, in `llm_calls.jsonl` and the shared `results/wsp/ledger.jsonl`. Dollars use
  `configs/wsp/pricing.yaml` ($10 / $1 cached / $50 output per 1M; reasoning is billed as output).
- **Caps:** 3M / 6M / 10M tokens per run (probe budget 10 / 25 / 50). The global cap (`WSP_TOKEN_CAP`) is
  checked before every call and every launch.
- **Compute:** forward passes and FLOPs (torch `FlopCounterMode`) for probes and gate evaluations,
  fine-tune steps and FLOPs, compute seconds, and Stage-1 epochs / optimizer steps / training FLOPs /
  wall time. `cost_report.py` gathers all of it into `results/wsp/cost_summary.csv`.

## Rules the harness enforces

- The agent sees training data only through budgeted probes. The gate uses validation data. Test data is
  used once, by `eval`.
- Edits change values only (shapes fixed): at most 5% of weights and relative L2 ≤ 0.1 per edit. An edit
  is kept only if the 90% paired-bootstrap bound shows improvement and no guard metric worsens by
  more than 2%.
- The sandbox gives the agent numpy only: no torch (so no gradients), no files, no network, and no
  API key in its environment.
- Fine-tune skips the sparsity caps, because gradient steps change every weight. It still has to pass
  the same validation gate.
