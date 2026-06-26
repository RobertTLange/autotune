# autotune

Automatic hyperparameter optimization CLI for scripts that print an `autotune_metric=<value>` line.

`autotune` analyzes a script with `headless`, proposes an Optuna search space, writes a generated Python runner into `.autotune/`, and executes trials without modifying the original script.

## Requirements

- Node.js 22+
- `python3` 3.9+
- Python package: `optuna`
- `headless` CLI, or `npx -y @roberttlange/headless`
- The runtime for the script being tuned, such as `python3`, `bash`, `julia`, `Rscript`, or `ruby`

## Install

```bash
npm install -g @roberttlange/autotune
python3 -m pip install optuna
```

From this repository:

```bash
npm install
npm run build
node dist/cli.js --help
```

## Metric Contract

Your script must print the target metric to stdout:

```text
autotune_metric=0.9432
```

If multiple metric lines are printed, the last one wins. If a trial exits non-zero or does not print a metric, that trial is pruned and the study continues.

## Usage

```bash
autotune run train.py --trials 50 --direction maximize
```

Common flags:

```bash
autotune run train.py \
  --trials 50 \
  --agent codex \
  --sampler tpe \
  --pruner none \
  --n-jobs 1 \
  --work-dir .autotune \
  --output .autotune/results.json
```

Use a custom runtime command without invoking a shell:

```bash
autotune run train.jl --trials 30 --command "julia +nightly"
```

Skip Phase 1 analysis with a known search space:

```bash
autotune run train.py --trials 20 --config search_space.yaml --yes
```

## Search Space Format

```yaml
parameters:
  - name: lr
    cli_flag: --lr
    type: float
    low: 0.00001
    high: 0.1
    log: true
  - name: batch_size
    cli_flag: --batch-size
    type: int
    low: 16
    high: 128
  - name: optimizer
    cli_flag: --optimizer
    type: categorical
    choices: [adam, sgd]
has_arg_parsing: true
needs_wrapper: false
direction: maximize
reasoning: accuracy-style metric
```

## Commands

```bash
autotune analyze <script> [--json] [--output search_space.yaml]
autotune doctor [script] [--agent codex]
autotune run <script> --trials N [flags]
autotune results [.autotune] [--top 10] [--json]
autotune resume --storage sqlite:///study.db --trials N [--work-dir .autotune]
```

Use `doctor` to verify prerequisites before a run:

```bash
PATH=$PWD/.venv/bin:$PATH autotune doctor examples/quadratic.py --agent codex
```

## Example

```bash
python3 -m pip install optuna
npm run build
node dist/cli.js run examples/quadratic.py \
  --trials 8 \
  --agent codex \
  --yes \
  --json \
  --work-dir /tmp/autotune-example
```

Expected behavior: `headless` proposes an `x` search space, the generated Optuna runner calls `examples/quadratic.py --x <value>`, and the best trial approaches `x = 0.7`.

## Generated Files

By default, `.autotune/` contains:

- `analyze_prompt.md`
- `generate_prompt.md`
- `search_space.yaml`
- `<script>_optuna.py`
- `results.json`
- `study.db`, when SQLite storage is configured

## Development

```bash
npm install
npm run lint
npm test
npm run build
```

The test suite uses fake `python3` and `headless` binaries for fast deterministic workflow coverage. Real end-to-end verification was run with `headless codex`, Optuna 4.9.0, and `examples/quadratic.py`.
