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
  --agent-guidance "prefer optimizer and regularization parameters" \
  --sampler tpe \
  --pruner none \
  --n-jobs 1 \
  --time-budget-seconds 86400 \
  --output results.json
```

Use `--effort low|medium|high|xhigh` as a shorter alias for `--reasoning-effort`.

When `--direction`, `--sampler`, or `--pruner` are omitted, Autotune uses the agent-proposed settings from the confirmed search space. Explicit CLI flags override agent proposals.

Use `--agent-guidance <text>` or `--agent-guidance-file <file>` to add advisory instructions for search-space generation and refinement, such as parameters to prefer or avoid. If both are provided, file guidance is applied first and inline guidance is appended. Guidance does not apply to modified-script generation and cannot override schema, metric comparability, or objective-measurement constraints. Guidance is sent to the agent and stored in prompt artifacts; guidance files must be regular files no larger than 65536 bytes.

By default, each run gets its own timestamped artifact directory next to the target script:

```text
autotune/
  latest.json
  <script-name>/
    latest.json
    runs/
      2026-06-27T181650000Z-<run-id>/
        analyze_prompt.md
        search_space.yaml
        train_optuna.py
        rounds.json
        results.json
```

`latest.json` points at the newest run, so `autotune results` can read the latest run from an `autotune/` directory. If the script lives in a subdirectory, run `autotune results <script-dir>/autotune` or run the command from that script directory. The script name keeps its extension, such as `train.py`, so sibling scripts do not share artifact roots. Pass `--work-dir <dir>` when you want an exact artifact directory instead of the timestamped default. Pass `--output <file>` when you want to copy the final JSON results to a specific file.

During trials, Autotune prints a compact colored progress table to stderr. The generated runner refreshes its run-local `results.json` after every completed trial, so `autotune results <run-dir>` can inspect long runs before they finish.

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
autotune results [autotune|run-dir|results.json] [--top 10] [--json]
autotune plot-progress <ablation-run-dir> --output progress.svg
autotune resume --storage sqlite:///study.db --trials N
```

Use `doctor` to verify prerequisites before a run:

```bash
PATH=$PWD/.venv/bin:$PATH node dist/cli.js doctor examples/mnist_cnn.py --agent codex
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

By default, refinement transfers useful context into the next round. If a parameter is removed from the active search space, Autotune fixes it at the previous best value and passes that fixed CLI flag to every later trial. It also seeds the next Optuna study with previous completed trials whose full effective parameter configuration is still valid for the refined active and fixed space. Disable these behaviors with `--no-refine-transfer-fixed-params` or `--no-refine-transfer-trials`.

Each round starts a new Optuna study and writes `search_space.round_N.yaml`, `results.round_N.json`, and `<script>_optuna.round_N.py` inside the run directory. `rounds.json` records the round paths, study name, storage URI, seed count, and transfer settings. The latest round is also written to `search_space.yaml`, `<script>_optuna.py`, and `results.json`.

## Progress Plots

Use `plot-progress` to visualize ablation runs that contain variant subdirectories such as `01_base_optuna`, `02_resets_no_trial_transfer`, and `03_resets_trial_transfer`. The plot counts real evaluated trials on the x-axis, skips transferred seed trials, ignores failed/timeout trials when updating the best-so-far score, and marks search-space reset boundaries.

```bash
node dist/cli.js plot-progress examples/autotune/cifar10_speedrun_ablations/62004 \
  --output docs/cifar10_speedrun_progress.svg \
  --title "CIFAR-10 speedrun ablations" \
  --max-trials 100

node dist/cli.js plot-progress examples/autotune/nanobench_ablation/20260703T214035Z \
  --output docs/nanobench_progress.svg \
  --title "Nanochat nanobench ablations" \
  --max-trials 100
```

<table>
  <tr>
    <td align="center"><strong>CIFAR-10 speedrun</strong></td>
    <td align="center"><strong>Nanochat nanobench</strong></td>
  </tr>
  <tr>
    <td><img src="docs/cifar10_speedrun_progress.svg" alt="CIFAR-10 speedrun ablation progress plot" /></td>
    <td><img src="docs/nanobench_progress.svg" alt="Nanochat nanobench ablation progress plot" /></td>
  </tr>
</table>

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
fixed_parameters:
  - name: weight_decay
    cli_flag: --weight-decay
    value: 0.0005
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

See `examples/README.md` for benchmark setup details, dataset/cache controls, and smoke-vs-full run guidance.

### MNIST CNN

`examples/mnist_cnn.py` trains a small PyTorch CNN on MNIST with hardcoded hyperparameters. It intentionally has no CLI parsing and no `autotune_metric` print, so Autotune asks the agent to create a compatible copy for the run.

Install runtime packages first if needed:

```bash
uv venv .venv
uv pip install --python .venv/bin/python optuna torch torchvision
```

Run:

```bash
PATH=$PWD/.venv/bin:$PATH node dist/cli.js run examples/mnist_cnn.py \
  --trials 8 \
  --agent codex \
  --json
```

Expected behavior: the agent proposes a search space, Autotune asks for confirmation or feedback, then generates a compatible copy that accepts hyperparameter flags and prints validation accuracy. The first run downloads MNIST into `/tmp/autotune-mnist-data`.

### CIFAR-10 ResNet

`examples/cifar10_resnet.py` trains a CIFAR-style ResNet-18 on the full CIFAR-10 training set with hardcoded hyperparameters. It intentionally has no CLI parsing and no `autotune_metric` print, so Autotune asks the agent to create a compatible copy for each trial.

Install runtime packages first if needed:

```bash
uv venv .venv
uv pip install --python .venv/bin/python optuna torch torchvision
```

Run:

```bash
PATH=$PWD/.venv/bin:$PATH node dist/cli.js run examples/cifar10_resnet.py \
  --trials 4 \
  --timeout-seconds 1800 \
  --agent codex \
  --json
```

Expected behavior: the agent proposes a search space for values such as learning rate, momentum, weight decay, batch size, dropout, and epochs, then generates a compatible copy that accepts trial flags and prints validation accuracy. Each trial is a real full-data training run and may take about 15 minutes on a GPU depending on hardware, so the example raises the per-trial timeout to 30 minutes. The first run downloads CIFAR-10 into `/tmp/autotune-cifar10-data`.

For a sequential transfer ablation, run:

```bash
./examples/run_cifar10_transfer_ablation.sh
```

The script analyzes CIFAR-10 once, reuses the same initial search space, then runs five 40-trial variants: one single-shot baseline and four refinement runs covering no transfer, full transfer, fixed-parameter transfer only, and trial-seeding transfer only.

### CIFAR-10 Speedrun

`examples/cifar10_speedrun.py` is an Autotune-native version of the Agentic Scientist CIFAR-10 speedrun baseline. It accepts explicit hyperparameter flags and prints the combined speedrun score as `autotune_metric`. See `examples/README.md` for data cache and full scoring controls.

```bash
uv venv .venv
uv pip install --python .venv/bin/python optuna torch torchvision numpy scipy
CIFAR10_SPEEDRUN_NUM_RUNS=5 \
PATH=$PWD/.venv/bin:$PATH node dist/cli.js run examples/cifar10_speedrun.py \
  --trials 20 \
  --timeout-seconds 1800 \
  --direction maximize \
  --agent codex \
  --agent-guidance "Tune only training hyperparameters and preserve the CIFAR-10 speedrun scoring protocol." \
  --json
```

### Nanochat Benchmark

`examples/nanochat_benchmark.py` wraps a local `karpathy/nanochat` checkout and uses the paper-inspired 14-hyperparameter search space in `examples/nanochat_search_space.yaml`. See `examples/README.md` for nanochat setup, data prep, and smoke-run controls.

```bash
export NANOCHAT_DIR=~/projects/nanochat
./examples/run_nanochat_benchmark.sh
```

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
