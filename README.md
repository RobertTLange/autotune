# autotune

Automatic hyperparameter optimization CLI for scripts that print an `autotune_metric=<value>` line.

`autotune` analyzes a script with `headless`, proposes an Optuna search space, asks for confirmation or feedback, then runs an Optuna study without modifying the original script. If a script is missing CLI parsing or metric output, autotune can ask the agent to generate a compatible copy for the run.

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
  --output results.json
```

When `--direction`, `--sampler`, or `--pruner` are omitted, autotune uses the agent-proposed settings from the confirmed search space. Explicit CLI flags override agent proposals.

By default, `run` pauses after analysis and asks whether to run, revise, edit, or abort:

```text
Run search with this space? [Y/feedback/edit/n]
```

- `Y` or Enter: run the search
- `feedback`: enter free-form feedback; the configured headless agent revises the search space and asks again
- any other non-empty text: treated directly as feedback
- `edit`: edit the generated `search_space.yaml` manually
- `n`: abort

Use `--yes` only when you want to accept the first proposed search space without review.

If the analyzed script does not accept the proposed CLI flags or does not print `autotune_metric=<value>`, autotune can ask `headless` to generate a modified copy. The original script is left untouched, and the Optuna runner invokes the modified copy.

Autotune prompts the agent to keep objective measurement fixed. Proposed parameters should change the candidate behavior being evaluated, not the metric formula, evaluation input set, aggregation, thresholds, baselines, reporting, or measurement-only random seeds.

Use a custom runtime command without invoking a shell:

```bash
autotune run train.jl --trials 30 --command "julia +nightly"
```

Run a build step once before analysis and trials:

```bash
autotune run model.cpp \
  --build-command "g++ -std=c++17 -O2 {script} -o {work-dir}/model" \
  --command "{work-dir}/model" \
  --trials 30
```

`--build-command` and `--command` support `{script}` and `{work-dir}` placeholders.

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
has_metric_output: true
direction: maximize
optuna:
  sampler: tpe
  pruner: none
  reasoning: TPE is a good default for mixed continuous/categorical spaces.
reasoning: accuracy-style metric
```

## Commands

```bash
autotune analyze <script> [--json] [--output search_space.yaml]
autotune doctor [script] [--agent codex]
autotune run <script> --trials N [flags]
autotune results [results.json] [--top 10] [--json]
autotune resume --storage sqlite:///study.db --trials N
```

Use `doctor` to verify prerequisites before a run:

```bash
PATH=$PWD/.venv/bin:$PATH autotune doctor examples/mnist_cnn_no_cli.py --agent codex
```

## MNIST CNN Example

`examples/mnist_cnn_no_cli.py` trains a small PyTorch CNN on MNIST with hardcoded hyperparameters. It intentionally has no CLI parsing and no `autotune_metric` print, so autotune asks the agent to create a compatible copy for the run.

Install runtime packages first if needed:

```bash
python3 -m pip install optuna
uv pip install --python .venv/bin/python torch torchvision
```

Run:

```bash
PATH=$PWD/.venv/bin:$PATH autotune run examples/mnist_cnn_no_cli.py \
  --trials 8 \
  --agent codex \
  --json
```

Expected behavior: the agent proposes a search space, autotune asks for confirmation or feedback, then generates a compatible copy that accepts hyperparameter flags and prints validation accuracy. The first run downloads MNIST into `/tmp/autotune-mnist-data`.

## C++ Example

`examples/pid_controller.cpp` is a single-file C++ simulation with manual flag parsing and built-in metric output. It tunes PID controller gains against a fixed tracking scenario with a disturbance.

```bash
autotune run examples/pid_controller.cpp \
  --build-command "g++ -std=c++17 -O2 {script} -o {work-dir}/pid_controller" \
  --command "{work-dir}/pid_controller" \
  --trials 12 \
  --agent codex \
  --json
```

Expected behavior: the agent proposes `--kp`, `--ki`, and `--kd`, the runner calls the compiled binary with trial values, and the best trial improves the fixed PID tracking score.

## Development

```bash
npm install
npm run lint
npm test
npm run build
```

The test suite uses fake `python3` and `headless` binaries for fast deterministic workflow coverage.
