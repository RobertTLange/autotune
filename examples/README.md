# Autotune Examples

Each task has its own subdirectory. Start with BBOB for a fast local smoke test, then scale to the training examples once the target script and metric look correct.

## Quick Examples

- `bbob/benchmark.py`: dependency-free Sphere, Ellipsoid, Rosenbrock, and Rastrigin objectives.
- `mnist/mnist_cnn.py`: agent-compatible PyTorch MNIST example.
- `pid_controller/pid_controller.cpp`: C++ PID simulation with a Centaur search space.
- `cifar10_speedrun/cifar10_speedrun.py`: Autotune-native CIFAR-10 speedrun benchmark.
- `nanochat/nanochat_benchmark.py`: wrapper around a local nanochat checkout.

## BBOB

The BBOB-style objectives are deterministic, five-dimensional, and require only Python when evaluated directly. Run Rosenbrock:

```bash
python3 examples/bbob/benchmark.py --function rosenbrock \
  --x1 -1.2 --x2 1.0 --x3 1.0 --x4 1.0 --x5 1.0
```

Run a fast optimization with the provided search space:

```bash
autotune run examples/bbob/benchmark.py \
  --config examples/bbob/rosenbrock_search_space.yaml \
  --trials 30 \
  --yes
```

Swap in `sphere_search_space.yaml`, `ellipsoid_search_space.yaml`, or `rastrigin_search_space.yaml` to change the objective. These are compact, unshifted, and unrotated BBOB-style functions rather than the full COCO benchmark suite.

Run the equal-budget comparison across all four objectives:

```bash
./examples/bbob/run_experiments.sh
```

The launcher additionally requires built Autotune, Optuna with `cmaes`, and a configured Headless agent. For each objective, it runs base CMA-ES, resets without transfer, resets with completed-trial transfer, and Centaur. Defaults are 10 seeded repetitions and 100 new evaluations per method (`50 + 2 × 25` for reset methods), for 160 experiments and 16,000 objective evaluations. The four objectives run in parallel while seeds and methods run sequentially, limiting concurrency to four jobs. Outputs and logs go under `examples/bbob/autotune/experiments/`, grouped by objective, method, and seed.

## MNIST CNN

`mnist/mnist_cnn.py` trains a small PyTorch CNN with hardcoded hyperparameters. It intentionally has no CLI parsing or `autotune_metric` output, exercising Autotune's compatible-copy generation.

```bash
uv venv .venv
uv pip install --python .venv/bin/python optuna torch torchvision
autotune run examples/mnist/mnist_cnn.py \
  --trials 8 \
  --agent codex \
  --json
```

The first run downloads MNIST into `/tmp/autotune-mnist-data`.

## PID Controller

Centaur requires Optuna 4.8 or newer but below 5, `cmaes` 0.12 or newer, at least two numeric parameters, one trial worker, and no agentic refinement rounds. Persistent runs currently require file-backed SQLite storage.

```bash
python3 -m pip install 'optuna>=4.8,<5' 'cmaes>=0.12'
npm install -g '@roberttlange/headless@0.4.0'
autotune run examples/pid_controller/pid_controller.cpp \
  --build-command "g++ -std=c++17 -O2 {script} -o {work-dir}/pid_controller" \
  --command "{work-dir}/pid_controller" \
  --config examples/pid_controller/pid_centaur_search_space.yaml \
  --trials 20 \
  --n-jobs 1 \
  --refine-rounds 0 \
  --agent codex \
  --yes \
  --json
```

The config explicitly opts into Centaur with the default LLM probability (`0.3`), CMA-ES-only warmup (`10` trials), and scheduler seed (`0`). Override them with `--centaur-llm-probability`, `--centaur-warmup-trials`, and `--centaur-seed`; if the config does not already select Centaur, also pass `--sampler centaur`. Resume only from trusted persistent Optuna storage because Optuna stores serialized CMA-ES optimizer state there.

## CIFAR-10 Speedrun

`cifar10_speedrun/cifar10_speedrun.py` is an Autotune-native version of the Agentic Scientist CIFAR-10 speedrun baseline. It accepts explicit training-hyperparameter flags and prints a combined score as `autotune_metric`.

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
CIFAR10_SPEEDRUN_RESULTS_DIR=examples/cifar10_speedrun/autotune/trial_metrics \
autotune run examples/cifar10_speedrun/cifar10_speedrun.py \
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
- `CIFAR10_SPEEDRUN_RESULTS_JSON`: optional metrics JSON path for a single direct run.
- `CIFAR10_SPEEDRUN_RESULTS_DIR`: optional directory for per-trial metrics JSON files during Autotune sweeps. Each file includes combined score, accuracy/time statistics, the effective config, and a config fingerprint.

Submit the equal-budget speedrun ablation on Slurm with `sbatch examples/cifar10_speedrun/run_cifar10_speedrun_ablations.sbatch`.

## Nanochat Benchmark

`nanochat/nanochat_benchmark.py` wraps a local `karpathy/nanochat` checkout and exposes the paper-inspired 14-hyperparameter search space in `nanochat/nanochat_search_space.yaml`. It minimizes validation bits-per-byte and prints `autotune_metric=<val_bpb>`. CUDA out-of-memory and incompatible batch-geometry trials are reported as `100.0`, matching the finite penalty used in the paper setup.

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
./examples/nanochat/run_nanochat_benchmark.sh
```

Run a short smoke benchmark:

```bash
export NANOCHAT_DIR=~/projects/nanochat
NANOCHAT_BENCHMARK_NUM_ITERATIONS=20 \
NANOCHAT_BENCHMARK_EVAL_TOKENS=524288 \
TRIALS=3 \
TIME_BUDGET_SECONDS=1800 \
./examples/nanochat/run_nanochat_benchmark.sh
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

Submit the equal-budget nanobench ablation on Slurm with `sbatch examples/nanochat/run_nanobench_ablation.sbatch`.

## Packaging Notes

The npm package includes example Python, C++, shell, Slurm, YAML, and this README file. Generated datasets, checkpoints, and task-local `autotune/` run artifacts are intentionally not packaged.
