# Autotune Examples

This directory contains small compatibility examples and larger benchmark-style workloads. Start with a small trial count, inspect the proposed search space, then scale up once the target script and metric look correct.

## Quick Examples

- `mnist_cnn.py`: agent-compatible PyTorch MNIST example. It intentionally lacks CLI flags and `autotune_metric`, so Autotune asks the agent to create a compatible copy.
- `cifar10_resnet.py`: full CIFAR-10 ResNet training example with the same agent-compatible flow. It downloads CIFAR-10 into `/tmp/autotune-cifar10-data`.
- `pid_controller.cpp`: C++ PID simulation with built-in flags and metric output.

## CIFAR-10 Speedrun

`cifar10_speedrun.py` is an Autotune-native version of the Agentic Scientist CIFAR-10 speedrun baseline. It accepts explicit training-hyperparameter flags and prints a combined score as `autotune_metric`.

Install runtime packages:

```bash
uv venv .venv
uv pip install --python .venv/bin/python optuna torch torchvision numpy scipy
```

Data is downloaded by `torchvision` on first use. By default, the cache path is `~/data/cifar10`; override it with:

```bash
export CIFAR10_SPEEDRUN_DATA_DIR=/path/to/cifar10-cache
```

Run a smoke-scoring search:

```bash
CIFAR10_SPEEDRUN_NUM_RUNS=5 \
PATH=$PWD/.venv/bin:$PATH node dist/cli.js run examples/cifar10_speedrun.py \
  --trials 20 \
  --timeout-seconds 1800 \
  --direction maximize \
  --agent codex \
  --agent-guidance "Tune only training hyperparameters and preserve the CIFAR-10 speedrun scoring protocol." \
  --json
```

For full Agentic Scientist-style scoring, set `CIFAR10_SPEEDRUN_NUM_RUNS=100`. Scoring controls are environment-only and should not enter the search space:

- `CIFAR10_SPEEDRUN_NUM_RUNS`: timed-run count; defaults to `5`.
- `CIFAR10_SPEEDRUN_MAX_TIME_PER_RUN`: early termination threshold; defaults to `3.0`.
- `CIFAR10_SPEEDRUN_DATA_DIR`: dataset cache directory.
- `CIFAR10_SPEEDRUN_RESULTS_JSON`: optional per-trial metrics JSON path.

## Nanochat Benchmark

`nanochat_benchmark.py` wraps a local `karpathy/nanochat` checkout and exposes the paper-inspired 14-hyperparameter search space in `nanochat_search_space.yaml`. It minimizes validation bits-per-byte and prints `autotune_metric=<val_bpb>`. CUDA out-of-memory and incompatible batch-geometry trials are reported as `100.0`, matching the finite penalty used in the paper setup.

Prepare nanochat separately:

```bash
git clone https://github.com/karpathy/nanochat ~/projects/nanochat
cd ~/projects/nanochat
uv sync --extra gpu
```

Set the checkout path before running Autotune:

```bash
export NANOCHAT_DIR=~/projects/nanochat
```

Nanochat downloads and prepares data through its own scripts. For a fresh checkout, follow the current nanochat README. The CPU demo path runs:

```bash
cd "$NANOCHAT_DIR"
source .venv/bin/activate
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval
```

Run the paper-style TPE workload with a 24-hour cumulative trial-time budget. The wrapper targets 300 seconds of training per trial by default, using a calibrated tokens/sec estimate to choose nanochat `--num-iterations` before launch:

```bash
export NANOCHAT_DIR=~/projects/nanochat
export NANOCHAT_BENCHMARK_NPROC_PER_NODE=8
export NANOCHAT_BENCHMARK_TOKENS_PER_SECOND=10000000
./examples/run_nanochat_benchmark.sh
```

Run a short smoke benchmark:

```bash
export NANOCHAT_DIR=~/projects/nanochat
NANOCHAT_BENCHMARK_NUM_ITERATIONS=20 \
NANOCHAT_BENCHMARK_EVAL_TOKENS=524288 \
TRIALS=3 \
TIME_BUDGET_SECONDS=1800 \
./examples/run_nanochat_benchmark.sh
```

Measurement controls are environment-only and should not be tuned:

- `NANOCHAT_DIR`: required local nanochat checkout.
- `NANOCHAT_PYTHON`: optional Python executable; defaults to `$NANOCHAT_DIR/.venv/bin/python`.
- `NANOCHAT_BENCHMARK_NUM_ITERATIONS`: explicit training steps per trial; overrides every target mode.
- `NANOCHAT_BENCHMARK_TARGET_TOKENS`: approximate training tokens per trial when `NANOCHAT_BENCHMARK_NUM_ITERATIONS` is unset; overrides calibrated time mode.
- `NANOCHAT_BENCHMARK_TARGET_SECONDS`: approximate training seconds per trial; `run_nanochat_benchmark.sh` defaults to `300`.
- `NANOCHAT_BENCHMARK_TOKENS_PER_SECOND`: calibrated training throughput used with `NANOCHAT_BENCHMARK_TARGET_SECONDS`; `run_nanochat_benchmark.sh` defaults to `1666667`, matching roughly 500M tokens in 5 minutes on the paper's H200 setup. Set this from an observed `tok/sec` line for other hardware.
- `NANOCHAT_BENCHMARK_MAX_SEQ_LEN`: context length; defaults to `2048`.
- `NANOCHAT_BENCHMARK_EVAL_EVERY`: validation cadence; defaults to the trial's final step.
- `NANOCHAT_BENCHMARK_EVAL_TOKENS`: validation token count; defaults to `524288`.
- `NANOCHAT_BENCHMARK_DEVICE_TYPE`: optional nanochat `--device-type` override.
- `NANOCHAT_BENCHMARK_NPROC_PER_NODE`: optional `torchrun` process count; defaults to `1`. Set to `8` on an 8-GPU H100 node.
- `NANOCHAT_BENCHMARK_RUN`: wandb run name; defaults to `dummy`.
- `NANOCHAT_BENCHMARK_MODEL_TAG`: checkpoint tag; defaults to a process-specific Autotune tag.
- `NANOCHAT_TRIAL_TIMEOUT_SECONDS`: outer process timeout; defaults to `NANOCHAT_BENCHMARK_TARGET_SECONDS + 600`, matching the paper runner's training budget plus startup/compile grace.
- `STORAGE`: Optuna storage URI; defaults to `sqlite:///$WORK_DIR/study.db` so interrupted runs can be resumed from the same work directory.
- `STUDY_NAME`: Optuna study name; defaults to `nanochat_benchmark_autotune`.

## Packaging Notes

The npm package includes example Python, C++, shell, YAML, and this README file. Generated datasets, checkpoints, and `examples/autotune/` run artifacts are intentionally not packaged.
