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

## How It Works

Autotune operates in clear phases:

1. **Prerequisites**: checks `python3`, Optuna, the selected `headless` agent, and the target runtime.
2. **Analysis**: asks the agent to inspect your script and propose tunable parameters, direction, sampler, and pruner.
3. **Confirmation**: shows the proposed space and accepts `Y`, `feedback`, `edit`, or `n`.
4. **Compatibility**: when needed, generates a modified copy that adds CLI flags or `autotune_metric`.
5. **Optuna Run**: writes a Python runner, launches trials, reports progress, and stores results.
6. **Refinement**: optionally runs extra agentic rounds that revise the search space from completed trial evidence.

The original script is left untouched.

## Requirements

- Node.js 22 or newer.
- Python 3 with Optuna 4.x.
- [Headless](https://www.npmjs.com/package/@roberttlange/headless) configured for the agent you want to use.

## Install

Install the CLI and its runtime prerequisites:

```bash
npm install -g @roberttlange/autotune
npm install -g @roberttlange/headless@0.4.0
python3 -m pip install 'optuna>=4.8,<5'
autotune doctor --agent codex
```

Run without a global Autotune installation:

```bash
npx @roberttlange/autotune --help
```

### Agent Skill

The optional Autotune agent skill is maintained in this repository rather than bundled in the npm package. Install it globally for Codex from GitHub:

```bash
npx skills add RobertTLange/autotune --skill autotune --agent codex --global
```

## Data and Privacy

Headless runs with the target script directory as its working directory, so the configured agent can inspect the target source and adjacent repository files. Source code, prompts, guidance, and trial summaries may be sent to the configured model provider. Prompts, search spaces, generated runners, and results are stored locally in the run artifact directory.

Use Autotune only with repositories and model providers you trust. Remove credentials, private datasets, and other sensitive material from the target directory and guidance before starting a run.

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
autotune doctor examples/mnist/mnist_cnn.py --agent codex
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

## Centaur Sampler

[Centaur](https://arxiv.org/abs/2603.24647) combines CMA-ES with occasional LLM-proposed trials.

Install its dependencies and select it explicitly:

```bash
python3 -m pip install 'optuna>=4.8,<5' 'cmaes>=0.12'
npm install -g '@roberttlange/headless@0.4.0'
autotune run train.py --trials 50 --sampler centaur --agent codex
```

Centaur requires at least two numeric parameters, `--n-jobs 1`, and `--refine-rounds 0`. Persistent runs require trusted, file-backed SQLite storage. Proposal agents run through Headless in read-only mode with an allowlisted environment; only use trusted repositories and configurations. See the [PID controller example](examples/README.md#pid-controller) for a complete setup.

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

Useful parameters and completed trials transfer between rounds by default. Each round gets its own search space, results, runner, and Optuna study in the run directory.

## Search Space Format

```yaml
parameters:
  - name: lr
    cli_flag: --lr
    type: float
    low: 0.00001
    high: 0.1
    log: true
has_arg_parsing: true
needs_wrapper: false
has_metric_output: true
direction: maximize
optuna:
  sampler: tpe
  pruner: none
```

Parameters may be `float`, `int`, or `categorical`. Use `fixed_parameters` for CLI values that should remain constant.

## Examples

See [`examples/README.md`](examples/README.md) for runnable examples, benchmark setup, dataset and cache controls, and smoke-versus-full run guidance.

## Development

```bash
npm ci
npm run lint
npm test
npm run build
```

The test suite uses fake `python3` and `headless` binaries for deterministic workflow coverage. Slurm launcher tests run only on Linux; portable manifest, cache, and local-launcher tests run on every supported platform.

Maintainers: see [RELEASING.md](RELEASING.md) for the version, tag, npm provenance, and trusted-publisher workflow.

## Support

Report bugs and request features through [GitHub Issues](https://github.com/RobertTLange/autotune/issues). Include `autotune --version`, the failing command, and relevant terminal output. Do not include credentials or private training data.

## License

Autotune is available under the [MIT License](LICENSE).
