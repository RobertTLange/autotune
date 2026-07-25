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

Run the CLI directly through npx; this requires Node.js with npm/npx, Python 3.9+, and first-run registry access. Python must provide `venv` with pip unless `uv` is available. No global Autotune, Headless, Optuna, or `cmaes` installation is required:

```bash
npx -y @roberttlange/autotune --version
npx -y @roberttlange/autotune doctor --agent codex
```

Omitting `@latest` fetches npm's `latest` dist-tag when no matching local package exists. Inside a checkout or project that already provides Autotune, npx may reuse that local version; use `@latest` to force registry resolution. Autotune provisions controller-only Python dependencies without changing the target training environment.

## When to Use

Use this skill when asked to tune script hyperparameters, run an autotune example, explain autotune CLI usage, inspect autotune results, or prepare a command for Python/C++/compiled training scripts.

## More CLI Help

Use built-in help when exact flags or subcommands matter:

```bash
npx -y @roberttlange/autotune --help
npx -y @roberttlange/autotune run --help
npx -y @roberttlange/autotune results --help
npx -y @roberttlange/autotune doctor --help
```

## Basic Flow

Check prerequisites first:

```bash
npx -y @roberttlange/autotune doctor train.py --agent codex
```

Run a search:

```bash
npx -y @roberttlange/autotune run train.py --trials 20 --agent codex
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
- Inspect prior output with `npx -y @roberttlange/autotune results autotune` or `npx -y @roberttlange/autotune results results.json`.

## Compiled Scripts

Use `--build-command` once before trials, and `--command` for the runtime. Both support `{script}` and `{work-dir}` placeholders.

```bash
npx -y @roberttlange/autotune run examples/pid_controller/pid_controller.cpp \
  --build-command "g++ -std=c++17 -O2 {script} -o {work-dir}/pid_controller" \
  --command "{work-dir}/pid_controller" \
  --trials 12 \
  --agent codex
```

## Guardrails

- Do not tune metric computation, evaluation input sets, thresholds, baselines, reporting-only values, or measurement-only random seeds.
- Tune candidate behavior only: model/config/hyperparameters that affect what is being evaluated.
- For expensive scripts, start with a small trial count and increase after the search space looks correct.
