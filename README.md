<p align="center">
  <img src="docs/logo.png" alt="Autotune hyperparameter optimization flow" width="200" style="border-radius: 24px;" />
</p>

<h1 align="center">Autotune</h1>

<p align="center">
  Agent-assisted hyperparameter optimization for scripts that report one metric line.
</p>

<p align="center">
  <img alt="Node.js 22+" src="https://img.shields.io/badge/Node.js-22%2B-339933?logo=node.js&logoColor=white" />
  <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-ESM-3178C6?logo=typescript&logoColor=white" />
  <img alt="Optuna" src="https://img.shields.io/badge/Optuna-powered-214478" />
</p>

Autotune combines `headless` agents with Optuna. It analyzes your script, proposes a search space, asks for confirmation, generates a safe trial runner, and executes trials without modifying the original script. If your script is missing CLI parsing or metric output, Autotune can ask the agent to generate a compatible copy for the run.

## How It Works

Autotune operates in clear phases:

1. **Prerequisites**: checks `python3`, Optuna, the selected `headless` agent, and the target runtime.
2. **Analysis**: asks the agent to inspect your script and propose tunable parameters, direction, sampler, and pruner.
3. **Confirmation**: shows the proposed space and accepts `Y`, `feedback`, `edit`, or `n`.
4. **Compatibility**: when needed, generates a modified copy that adds CLI flags or `autotune_metric`.
5. **Optuna Run**: writes a Python runner, launches trials, reports progress, and stores results.
6. **Refinement**: optionally runs extra agentic rounds that revise the search space from completed trial evidence.

The original script is left untouched.

## Install

```bash
python3 -m pip install optuna
npm install
npm run build
node dist/cli.js --help
```

## Core Usage

```bash
autotune run train.py --trials 50 --agent codex
```

Common flags:

```bash
autotune run train.py \
  --trials 50 \
  --agent codex \
  --model gpt-5.5 \
  --reasoning-effort high \
  --sampler tpe \
  --pruner none \
  --n-jobs 1 \
  --output results.json
```

Use `--effort low|medium|high|xhigh` as a shorter alias for `--reasoning-effort`.

When `--direction`, `--sampler`, or `--pruner` are omitted, Autotune uses the agent-proposed settings from the confirmed search space. Explicit CLI flags override agent proposals.

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

## Commands

```bash
autotune analyze <script> [--agent codex] [--model MODEL] [--reasoning-effort high]
autotune doctor [script] [--agent codex]
autotune run <script> --trials N [--agent codex] [--model MODEL] [--reasoning-effort high]
autotune results [results.json] [--top 10] [--json]
autotune resume --storage sqlite:///study.db --trials N
```

Use `doctor` to verify prerequisites before a run:

```bash
PATH=$PWD/.venv/bin:$PATH autotune doctor examples/mnist_cnn_no_cli.py --agent codex
```

Use built-in help for full flag details:

```bash
autotune --help
autotune run --help
autotune results --help
```

## Runtime Commands

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

Skip analysis with a known search space:

```bash
autotune run train.py --trials 20 --config search_space.yaml --yes
```

## Agentic Refinement

Run multiple refinement rounds:

```bash
autotune run train.py \
  --trials 20 \
  --refine-rounds 2 \
  --refine-trials 10 \
  --refine-mode ask
```

After each round, Autotune summarizes completed trials and asks the agent to revise the search space. The agent may narrow promising ranges, broaden ranges when best values sit near bounds, or add/remove variables when justified by the script and trial evidence. `--refine-mode ask` asks for approval before each revised space; `--refine-mode auto` accepts revised spaces automatically.

Each round starts a new Optuna study and writes `search_space.round_N.yaml` and `results.round_N.json`. The latest round is also written to `search_space.yaml` and `results.json`.

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

## Examples

### MNIST CNN

`examples/mnist_cnn_no_cli.py` trains a small PyTorch CNN on MNIST with hardcoded hyperparameters. It intentionally has no CLI parsing and no `autotune_metric` print, so Autotune asks the agent to create a compatible copy for the run.

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

Expected behavior: the agent proposes a search space, Autotune asks for confirmation or feedback, then generates a compatible copy that accepts hyperparameter flags and prints validation accuracy. The first run downloads MNIST into `/tmp/autotune-mnist-data`.

### C++

`examples/pid_controller.cpp` is a single-file C++ simulation with manual flag parsing and built-in metric output. It tunes PID controller gains against a fixed tracking scenario with a disturbance.

```bash
autotune run examples/pid_controller.cpp \
  --build-command "g++ -std=c++17 -O2 {script} -o {work-dir}/pid_controller" \
  --command "{work-dir}/pid_controller" \
  --trials 12 \
  --refine-rounds 2 \
  --refine-trials 8 \
  --refine-mode auto \
  --agent codex \
  --json
```

Expected behavior: the agent proposes `--kp`, `--ki`, and `--kd`, the runner calls the compiled binary with trial values, and later rounds refine the fixed PID tracking search space from previous trial results.

## Development

```bash
npm install
npm run lint
npm test
npm run build
```

The test suite uses fake `python3` and `headless` binaries for fast deterministic workflow coverage.
