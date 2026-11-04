# WSP task for Codex: improve trained detectors by editing their weights

You are a **weight-space programmer**. Each run hands you one neural network that was already trained with
gradients (Stage 1, up to 50 epochs, on the lab server). The networks defend against data poisoning in local
differential privacy (LDP):

- **mean/variance estimation** (`mv/...` runs): an FT-Transformer that scores reports for removal, or a Deep Sets /
  Set Transformer that reads a whole round of reports and outputs the mean and variance directly;
- **frequency oracles** (`freq/...` runs): an FT-Transformer that flags fake users.

Your job is to make the network better **by reading and editing its weights directly, with numpy**. There is no
optimizer and no gradient. You understand the network from its weights and from a few forward-pass probes, then
write targeted edits. A harness (plain code, not you) decides whether an edit is kept.

Each run's model card states its task, architecture, inputs, outputs and metric. Read it first.

## Hard rules

1. **All access to models and data goes through `scripts/wsp/wspctl`.** Do not open, list, copy or load anything
   under `results/wsp_codex/bundle/`, the run folders (`results/*_wsp/stage2/...`), datasets or checkpoints.
   Every shell command you run is recorded in the Codex session log and audited after the run.
2. **No gradients, no training, no torch.** Edits are numpy code run through `wspctl py`. Do not run Python on the
   model or the data outside `wspctl`.
3. **Do not change repository files.** Write your scratch scripts only under `results/wsp_codex/scratch/`. Reading
   the source code under `attacker_detector/` to understand an architecture is fine.
4. **Run one `wspctl` command at a time** and wait for it to finish. `start`, `probe`, `submit` and `finish` can
   take a few minutes on this machine's CPU.
5. **Never claim a result you have not probed or gated.**
6. If `wspctl` crashes with a Python traceback, or prints something you do not understand, **stop and tell the
   user**. Do not patch the harness.

## Commands

All commands act on the active run. `W` means your staged weights. "Committed" means the weights currently
accepted by the gate.

| Command | What it does | Cost |
|---|---|---|
| `scripts/wsp/wspctl queue` | runs and their status | free |
| `scripts/wsp/wspctl next` | start (or resume) the next run; prints its model card | free |
| `scripts/wsp/wspctl card` | model card, committed validation metrics, budget left, gate rules | free |
| `scripts/wsp/wspctl stats [NAME ...] [--staged]` | per-tensor statistics (2-D: top singular values, stable rank, row norms) | free |
| `scripts/wsp/wspctl read NAME ['0:4, :'] [--staged]` | raw values of a slice (max 2048 values) | free |
| `scripts/wsp/wspctl py FILE` (or `py -c 'CODE'`) | run numpy code in the sandbox (see below) | free |
| `scripts/wsp/wspctl probe [metrics\|errors\|activations] [--committed]` | forward pass on a fixed subset of **training** data. Without `--committed` it probes your staged `W` and shows the change against committed | **1 probe** |
| `scripts/wsp/wspctl submit "RATIONALE"` | gate the staged `W` on **validation** data | **1 gate evaluation** |
| `scripts/wsp/wspctl history` | your edits with verdicts, recent probes, budget | free |
| `scripts/wsp/wspctl status` | budget left and stop reason, if any | free |
| `scripts/wsp/wspctl finish "SUMMARY"` | end the run. The committed weights are final | free |

Probe kinds:
- `metrics`: loss or F1 per slice (clean / attacked, attack type, attack-share bin; for the frequency oracle,
  per dataset and epsilon).
- `errors`: the worst cases.
- `activations`: per layer, mean |activation|, units that never fire, and the most active units.

### Sandbox (`wspctl py`)

- Names available:
  - `W`: dict of tensor name to float32 array. Edit it in place and keep the shapes.
  - `W0`: committed weights, read-only.
  - `WS`: Stage-1 weights, read-only.
  - `np`, `math`.
  - `names()`: list of `(name, shape)`.
  - `diff()`: staged vs committed, i.e. changed entries and relative L2.
  - `revert(name=None)`: reset one tensor, or all, to committed.
- Allowed imports: `numpy`, `math`, `itertools`, `functools`, `statistics`, `collections`, `heapq`.
- No files, no network, 60 s per call.
- **Only `W` persists between calls.** Any other variable is gone next time, so put helper code in each script.
- Write each script as a file, for example `results/wsp_codex/scratch/<run>/edit3.py`, and run it with
  `scripts/wsp/wspctl py results/wsp_codex/scratch/<run>/edit3.py`. Keep printed output small.

## How an edit is judged

- **Caps per edit:** at most 5% of all weights changed, and relative L2 change ≤ 0.10 (against committed). Names
  and shapes must not change. The card's `gate` field has the exact values.
- **Validation:** the gate scores the edit on **validation** data. It is kept only if the 90% one-sided paired
  bootstrap bound shows the loss improves (for the frequency oracle, F1@0.5 improves) and no guard metric gets
  worse by more than 2%.
- **After a verdict:**
  - Accepted: the committed weights become your edit, and `W` resets to them.
  - Rejected: nothing is committed and your edit stays staged. Fix it, or `revert()`.
- Probes run on **training** data: they are cheap evidence, and the gate decides.

## Budget and stopping

Every run has a number of probes, a number of gate evaluations and a token cap. The card and `status` show
what's left. Tokens are counted from the Codex session log and include your reasoning.

The run is over when any of these happens:
- the gate evaluations are spent;
- 6 edits in a row are rejected;
- the token cap is reached;
- you decide further gains are unlikely.

`wspctl` then tells you to finish. Always end with `wspctl finish "SUMMARY"`. In the summary, say what failure
you found, what you changed, what was accepted, and why you stopped.

## Per-run loop

1. `wspctl next`. Read the model card: task, architecture, inputs, metric, tensor names, budget.
2. Find where the committed model fails:
   - spend one probe on `probe metrics --committed`, maybe also `errors` or `activations`;
   - read the slices: which slice is worst, and which direction is the error?
3. Inspect the weights tied to that failure: `stats`, `read`, and numpy analysis in `py`. Examples:
   - output-layer rows and biases;
   - units that never fire, outlier rows, low-rank structure;
   - how the input features reach the output.
4. Form one concrete hypothesis and stage a small, targeted edit that tests it. Examples:
   - a bias shift;
   - rescaling one row or column;
   - a rank-1 update;
   - reviving or silencing a unit.

   Stay well inside the caps.
5. Spend one probe on `probe metrics` to compare staged with committed (the `d_loss` / `d_f1` per slice).
6. `submit` only when the probe supports the edit. After a rejection, read the reasons (cap, bootstrap bound or
   guard), then adjust or try another idea.
7. Repeat until the run is over, then `finish`.

Then `wspctl next` for the next run. Stop when it says **queue empty**, when it refuses because of the weekly
limit, or when the user asked for a fixed number of runs. Finish by telling the user:
- the runs done, each one's Stage-1 → final validation line (`finish` prints it);
- the runs left.
