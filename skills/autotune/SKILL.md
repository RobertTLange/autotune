---
name: autotune
description: Use when an agent needs to run or explain the autotune CLI for hyperparameter optimization, tune a local script, inspect autotune results, or use the packaged examples.
---

# Autotune CLI

Use `autotune` to analyze a local script, confirm or revise a search space, then run Optuna trials. The target must print a metric line:

```text
autotune_metric=<value>
```

If the script lacks CLI flags or metric output, autotune may ask the agent to generate a modified copy; the original script is left untouched.

## Prerequisites

Check that the CLI and runtime are available before starting:

```bash
autotune --version
autotune doctor --agent codex
```

If the CLI is missing, ask before installing the npm package globally:

```bash
npm install -g @roberttlange/autotune
```

## When to Use

Use this skill when asked to tune script hyperparameters, run an autotune example, explain autotune CLI usage, inspect autotune results, or prepare a command for Python/C++/compiled training scripts.

## More CLI Help

Use built-in help when exact flags or subcommands matter:

```bash
autotune --help
autotune run --help
autotune results --help
autotune doctor --help
```

## Basic Flow

Check prerequisites first:

```bash
autotune doctor train.py --agent codex
```

Run a search:

```bash
autotune run train.py --trials 20 --agent codex
```

During confirmation, prefer review over blind acceptance:

- `Y`: run with the proposed space
- `feedback`: ask the agent to revise the space
- `edit`: manually edit `search_space.yaml`
- `n`: abort

Use `--yes` only when accepting the first proposed space is intended.

## Useful Flags

- Omit `--direction`, `--sampler`, and `--pruner` to use agent-proposed Optuna settings shown at confirmation.
- Override only when needed: `--direction maximize`, `--sampler tpe`, `--pruner none`.
- Configure the Headless backend with `--agent codex`, `--model gpt-5.5`, and `--reasoning-effort high`; `--effort high` is a shorter alias.
- Use `--output results.json` for an explicit results path.
- Use refinement rounds when useful: `--refine-rounds 2 --refine-trials 10 --refine-mode ask`.
- Inspect prior output with `autotune results autotune` or `autotune results results.json`.

## Compiled Scripts

Use `--build-command` once before trials, and `--command` for the runtime. Both support `{script}` and `{work-dir}` placeholders.

```bash
autotune run examples/pid_controller/pid_controller.cpp \
  --build-command "g++ -std=c++17 -O2 {script} -o {work-dir}/pid_controller" \
  --command "{work-dir}/pid_controller" \
  --trials 12 \
  --agent codex
```

## Guardrails

- Do not tune metric computation, evaluation input sets, thresholds, baselines, reporting-only values, or measurement-only random seeds.
- Tune candidate behavior only: model/config/hyperparameters that affect what is being evaluated.
- For expensive scripts, start with a small trial count and increase after the search space looks correct.
