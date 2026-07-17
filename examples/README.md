# Autotune Examples

Each task has its own subdirectory. Start with BBOB for a fast local smoke test, then scale to the training examples once the target script and metric look correct.

## Quick Examples

- `bbob/benchmark.py`: dependency-free Sphere, Ellipsoid, Rosenbrock, and Rastrigin objectives.
- `mnist/mnist_cnn.py`: agent-compatible PyTorch MNIST example.
- `pid_controller/pid_controller.cpp`: C++ PID simulation with a Centaur search space.
- `cifar10_speedrun/cifar10_speedrun.py`: Autotune-native CIFAR-10 speedrun benchmark.
- `nanochat/nanochat_benchmark.py`: HPO wrapper around the pinned autoresearch evaluator.

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

`nanochat/nanochat_benchmark.py` keeps Autotune's adaptive HPO workflow while using the evaluator from `karpathy/autoresearch` commit `228791fb499afffb54b46200aca536f79142f117`. Autotune changes only the 14 declared training constants and never edits the wider codebase.

Prepare the pinned checkout and its default clean cache:

```bash
git clone https://github.com/karpathy/autoresearch ~/projects/autoresearch
cd ~/projects/autoresearch
git checkout 228791fb499afffb54b46200aca536f79142f117
uv sync --frozen
uv run --frozen prepare.py
export AUTORESEARCH_DIR=~/projects/autoresearch
cd ~/autotune
python3 examples/nanochat/prepare_nanochat_cache.py create
```

Use a fresh `~/.cache/autoresearch` containing exactly the default ten training shards plus pinned validation shard `06542`. Manifest creation is an explicit trust step: it hashes every shard and the tokenizer after pinned preparation. Each benchmark launcher rehashes the cache once before starting trials, then passes the verified content identity to every trial.

Run 100 seeded TPE trials:

```bash
export AUTORESEARCH_DIR=~/projects/autoresearch
./examples/nanochat/run_nanochat_benchmark.sh
```

Every score uses one GPU, a 2048-token context, 8192-token vocabulary, 300 measured training seconds excluding compilation/startup, and 20,971,520 validation tokens. `NANOCHAT_BENCHMARK_SEED` controls training randomness but is not tunable; discovery defaults to seed 42. The wrapper syncs the pinned lockfile through `uv --frozen`, emits source/data/tokenizer/runtime provenance, and assigns `100.0` to infeasible or OOM trials. Hardware still affects how much training fits into five minutes, so compare runs on the same accelerator type.

Key launcher controls: `TRIALS`, `SAMPLER`, `SAMPLER_SEED`, `TIME_BUDGET_SECONDS`, `WORK_DIR`, `STORAGE`, `STUDY_NAME`, and `NANOCHAT_TRIAL_TIMEOUT_SECONDS` (default 1200 seconds). `NANOCHAT_BENCHMARK_RESULTS_DIR` stores per-trial provenance JSON.

Submit the equal-budget nanobench ablation on Slurm with `sbatch examples/nanochat/run_nanobench_ablation.sbatch`.

## Packaging Notes

The npm package includes example Python, C++, shell, Slurm, YAML, and this README file. Generated datasets, checkpoints, and task-local `autotune/` run artifacts are intentionally not packaged.
