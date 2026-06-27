import { chmod, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type { Invocation, SearchSpace } from "./types.js";

export function renderOptunaRunner(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  outputPath: string;
  resultsPath: string;
  timeoutSeconds?: number;
}): string {
  const timeout = input.timeoutSeconds ?? 900;
  const payload = {
    command: input.invocation.command,
    script: input.invocation.script,
    script_arg_mode: input.invocation.scriptArgument ?? inferScriptArgument(input.invocation),
    parameters: input.searchSpace.parameters,
    results_path: input.resultsPath,
    timeout
  };

  return `#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import optuna
from optuna.trial import TrialState

CONFIG = json.loads(${JSON.stringify(JSON.stringify(payload))})
CURRENT_TRIAL_TARGET = 0
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_value(trial, parameter):
    name = parameter["name"]
    if parameter["type"] == "float":
        return trial.suggest_float(name, parameter["low"], parameter["high"], log=parameter.get("log", False))
    if parameter["type"] == "int":
        return trial.suggest_int(name, int(parameter["low"]), int(parameter["high"]))
    if parameter["type"] == "categorical":
        return trial.suggest_categorical(name, parameter["choices"])
    raise ValueError(f"Unsupported parameter type: {parameter['type']}")


def build_argv(params):
    command = list(CONFIG["command"])
    script = CONFIG["script"]
    script_arg_mode = CONFIG.get("script_arg_mode", "append")
    if script_arg_mode in ("included", "none"):
        argv = command[:]
    else:
        argv = command + [script]
    for parameter in CONFIG["parameters"]:
        argv.extend([parameter["cli_flag"], str(params[parameter["name"]])])
    return argv


def parse_metric(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("autotune_metric="):
            return float(line.split("=", 1)[1].strip())
    raise ValueError("No autotune_metric= line found in trial stdout")


def objective(trial):
    params = {parameter["name"]: suggest_value(trial, parameter) for parameter in CONFIG["parameters"]}
    argv = build_argv(params)
    result = subprocess.run(argv, capture_output=True, text=True, timeout=CONFIG["timeout"])
    if result.returncode != 0:
        raise optuna.TrialPruned(f"Trial command exited {result.returncode}: {result.stderr[-1000:]}")
    try:
        return parse_metric(result.stdout)
    except Exception as exc:
        raise optuna.TrialPruned(str(exc)) from exc


def make_sampler(name):
    if name == "tpe":
        return optuna.samplers.TPESampler()
    if name == "random":
        return optuna.samplers.RandomSampler()
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler()
    if name == "grid":
        search_space = {}
        for parameter in CONFIG["parameters"]:
            if parameter["type"] != "categorical":
                raise ValueError("grid sampler requires categorical parameters only")
            search_space[parameter["name"]] = parameter["choices"]
        return optuna.samplers.GridSampler(search_space)
    raise ValueError(f"Unsupported sampler: {name}")


def make_pruner(name):
    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"Unsupported pruner: {name}")


def serialize_trial(trial):
    return {
        "number": trial.number,
        "value": trial.value,
        "params": trial.params,
        "state": trial.state.name,
    }


def serialize_study(study, direction):
    complete = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    best = study.best_trial if complete else None
    return {
        "study_name": study.study_name,
        "direction": direction,
        "n_trials": len(study.trials),
        "best_trial": serialize_trial(best) if best else None,
        "all_trials": [serialize_trial(trial) for trial in study.trials],
    }


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def color_enabled():
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") not in (None, "", "0"):
        return True
    return sys.stderr.isatty()


def style(text, code):
    if not color_enabled():
        return text
    return f"\\033[{code}m{text}\\033[0m"


def style_state(state):
    if state == "COMPLETE":
        return style(state, "32")
    if state == "PRUNED":
        return style(state, "33")
    if state == "FAIL":
        return style(state, "31")
    return state


def report_progress(study, trial):
    complete = [item for item in study.trials if item.state == TrialState.COMPLETE]
    finished = len([item for item in study.trials if item.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)])
    value = f" value={trial.value}" if trial.value is not None else ""
    best = f" best={study.best_value}" if complete else ""
    print(f"[{timestamp()}] Trial {finished}/{CURRENT_TRIAL_TARGET} {style_state(trial.state.name)}{value}{best}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--direction", choices=["maximize", "minimize"], required=True)
    parser.add_argument("--sampler", default="tpe")
    parser.add_argument("--pruner", default="none")
    parser.add_argument("--storage")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", default=CONFIG["results_path"])
    args = parser.parse_args()

    global CURRENT_TRIAL_TARGET
    CURRENT_TRIAL_TARGET = args.trials
    study = optuna.create_study(
        direction=args.direction,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=make_sampler(args.sampler),
        pruner=make_pruner(args.pruner),
    )
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs, callbacks=[report_progress])
    result = serialize_study(study, args.direction)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    if result["best_trial"] is None:
        print(f"[{timestamp()}] all trials failed", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
`;
}

function inferScriptArgument(invocation: Invocation): "append" | "included" {
  return invocation.command.length === 1 && path.resolve(invocation.command[0] ?? "") === path.resolve(invocation.script)
    ? "included"
    : "append";
}

export async function writeOptunaRunner(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  outputPath: string;
  resultsPath: string;
  timeoutSeconds?: number;
}): Promise<void> {
  await mkdir(path.dirname(input.outputPath), { recursive: true });
  await writeFile(input.outputPath, renderOptunaRunner(input), "utf8");
  await chmod(input.outputPath, 0o755);
}
