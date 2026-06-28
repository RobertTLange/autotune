import { chmod, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type { Invocation, SearchSpace } from "./types.js";

export interface SeedTrial {
  value: number;
  params: Record<string, string | number | boolean>;
}

export function renderOptunaRunner(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  outputPath: string;
  resultsPath: string;
  studyName?: string;
  timeoutSeconds?: number;
  seedTrials?: SeedTrial[];
}): string {
  const timeout = input.timeoutSeconds ?? 900;
  const payload = {
    command: input.invocation.command,
    script: input.invocation.script,
    script_arg_mode: input.invocation.scriptArgument ?? inferScriptArgument(input.invocation),
    study_name: input.studyName,
    parameters: input.searchSpace.parameters,
    fixed_parameters: input.searchSpace.fixed_parameters ?? [],
    seed_trials: input.seedTrials ?? [],
    results_path: input.resultsPath,
    max_output_bytes: 65536,
    timeout
  };

  return `#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import optuna
from optuna.trial import TrialState

CONFIG = json.loads(${JSON.stringify(JSON.stringify(payload))})
CURRENT_TRIAL_TARGET = 0
BASELINE_FINISHED_COUNT = 0
SEED_TRIAL_COUNT = 0
STARTED_TRIAL_COUNT = 0
PROGRESS_HEADER_PRINTED = False
PROGRESS_LOCK = threading.Lock()
ANSI_PATTERN = re.compile(r"\\033\\[[0-9;]*m")
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
    for parameter in CONFIG.get("fixed_parameters", []):
        argv.extend([parameter["cli_flag"], str(parameter["value"])])
    return argv


def effective_params(params):
    effective = dict(params)
    for parameter in CONFIG.get("fixed_parameters", []):
        effective[parameter["name"]] = parameter["value"]
    return effective


def parse_metric(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("autotune_metric="):
            return float(line.split("=", 1)[1].strip())
    raise ValueError("No autotune_metric= line found in trial stdout")


class OutputCapture:
    def __init__(self, max_chars):
        self.max_chars = max_chars
        self.tail = ""
        self.pending = ""
        self.metric = None
        self.metric_error = None

    def feed(self, text, parse_metrics=False):
        self.tail = (self.tail + text)[-self.max_chars:]
        if parse_metrics:
            self.pending += text
            self.pending = self.pending[-self.max_chars:]
            while "\\n" in self.pending:
                line, self.pending = self.pending.split("\\n", 1)
                self.parse_metric_line(line)

    def finish(self, parse_metrics=False):
        if parse_metrics and self.pending:
            self.parse_metric_line(self.pending)
            self.pending = ""

    def parse_metric_line(self, line):
        if line.startswith("autotune_metric="):
            try:
                self.metric = float(line.split("=", 1)[1].strip())
                self.metric_error = None
            except Exception as exc:
                self.metric_error = exc


def drain_stream(stream, capture, parse_metrics=False, mirror=False):
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            capture.feed(chunk, parse_metrics=parse_metrics)
            if mirror:
                sys.stderr.write(chunk)
                sys.stderr.flush()
    finally:
        capture.finish(parse_metrics=parse_metrics)


def kill_process_tree(process):
    if hasattr(os, "killpg"):
        os.killpg(process.pid, 9)
    else:
        process.kill()


def run_trial_command(argv):
    max_output_bytes = CONFIG.get("max_output_bytes", 65536)
    stdout_capture = OutputCapture(max_output_bytes)
    stderr_capture = OutputCapture(max_output_bytes)
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=hasattr(os, "setsid"),
    )
    stdout_thread = threading.Thread(target=drain_stream, args=(process.stdout, stdout_capture, True, True))
    stderr_thread = threading.Thread(target=drain_stream, args=(process.stderr, stderr_capture, False, True))
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=CONFIG["timeout"])
    except subprocess.TimeoutExpired as exc:
        kill_process_tree(process)
        returncode = process.wait()
        raise RuntimeError(f"Trial command timed out after {CONFIG['timeout']}s") from exc
    finally:
        stdout_thread.join()
        stderr_thread.join()
    return {
        "returncode": returncode,
        "stdout": stdout_capture.tail,
        "stderr": stderr_capture.tail,
        "metric": stdout_capture.metric,
        "metric_error": stdout_capture.metric_error,
    }


def objective(trial):
    global STARTED_TRIAL_COUNT
    params = {parameter["name"]: suggest_value(trial, parameter) for parameter in CONFIG["parameters"]}
    with PROGRESS_LOCK:
        STARTED_TRIAL_COUNT += 1
        started = STARTED_TRIAL_COUNT
    print_progress_row(f"{started}/{CURRENT_TRIAL_TARGET}", "RUNNING", None, None, effective_params(params))
    argv = build_argv(params)
    result = run_trial_command(argv)
    if result["returncode"] != 0:
        raise RuntimeError(f"Trial command exited {result['returncode']}: {result['stderr'][-1000:]}")
    if result["metric_error"] is not None:
        raise RuntimeError(str(result["metric_error"])) from result["metric_error"]
    try:
        return result["metric"] if result["metric"] is not None else parse_metric(result["stdout"])
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


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


def distribution_for_parameter(parameter):
    if parameter["type"] == "float":
        return optuna.distributions.FloatDistribution(parameter["low"], parameter["high"], log=parameter.get("log", False))
    if parameter["type"] == "int":
        return optuna.distributions.IntDistribution(int(parameter["low"]), int(parameter["high"]))
    if parameter["type"] == "categorical":
        return optuna.distributions.CategoricalDistribution(parameter["choices"])
    raise ValueError(f"Unsupported parameter type: {parameter['type']}")


def add_seed_trials(study):
    parameters = CONFIG["parameters"]
    if not parameters or study.trials:
        return 0
    distributions = {parameter["name"]: distribution_for_parameter(parameter) for parameter in parameters}
    active_names = set(distributions)
    count = 0
    for seed in CONFIG.get("seed_trials", []):
        params = {name: value for name, value in seed["params"].items() if name in active_names}
        if set(params) != active_names:
            print(f"[{timestamp()}] skipped transferred seed trial with incomplete active params", file=sys.stderr, flush=True)
            continue
        try:
            study.add_trial(optuna.trial.create_trial(params=params, distributions=distributions, value=seed["value"]))
            count += 1
        except Exception as exc:
            print(f"[{timestamp()}] skipped transferred seed trial: {exc}", file=sys.stderr, flush=True)
    return count


def serialize_trial(trial):
    return {
        "number": trial.number,
        "value": trial.value,
        "params": effective_params(trial.params),
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


def write_results(study, direction, output_path):
    result = serialize_study(study, direction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp_path, output_path)
    return result


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
    if state == "RUNNING":
        return style(state, "36")
    if state == "COMPLETE":
        return style(state, "32")
    if state == "PRUNED":
        return style(state, "33")
    if state == "FAIL":
        return style(state, "31")
    return state


def visible_width(text):
    return len(ANSI_PATTERN.sub("", str(text)))


def pad_cell(value, width, align="left"):
    value = str(value)
    padding = " " * max(0, width - visible_width(value))
    return f"{padding}{value}" if align == "right" else f"{value}{padding}"


def format_number(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_params(params, max_chars=88):
    text = " ".join(f"{name}={format_number(value)}" for name, value in sorted(params.items()))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def progress_columns():
    return [
        ("Trial", 8, "right"),
        ("State", 10, "left"),
        ("Value", 12, "right"),
        ("Best", 12, "right"),
        ("Params", 88, "left"),
    ]


def progress_prefix():
    return style(f"[{timestamp()}]", "2")


def progress_row(values):
    return "  ".join(
        pad_cell(value, width, align)
        for value, (_, width, align) in zip(values, progress_columns())
    )


def print_progress_header():
    global PROGRESS_HEADER_PRINTED
    if PROGRESS_HEADER_PRINTED:
        return
    columns = progress_columns()
    print(f"{progress_prefix()} {style(progress_row([name for name, _, _ in columns]), '1')}", file=sys.stderr, flush=True)
    print(f"{progress_prefix()} {style(progress_row(['-' * width for _, width, _ in columns]), '2')}", file=sys.stderr, flush=True)
    PROGRESS_HEADER_PRINTED = True


def print_progress_row(trial_label, state, value, best, params):
    with PROGRESS_LOCK:
        print_progress_header()
        print(
            f"{progress_prefix()} {progress_row([trial_label, style_state(state), format_number(value), format_number(best), format_params(params)])}",
            file=sys.stderr,
            flush=True,
        )


def current_best_value(study):
    complete = [item for item in study.trials if item.state == TrialState.COMPLETE]
    return study.best_value if complete else None


def report_progress(study, trial):
    finished = len([item for item in study.trials if item.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)])
    finished_new = max(0, finished - BASELINE_FINISHED_COUNT - SEED_TRIAL_COUNT)
    print_progress_row(f"{finished_new}/{CURRENT_TRIAL_TARGET}", trial.state.name, trial.value, current_best_value(study), effective_params(trial.params))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--direction", choices=["maximize", "minimize"], required=True)
    parser.add_argument("--sampler", default="tpe")
    parser.add_argument("--pruner", default="none")
    parser.add_argument("--storage")
    parser.add_argument("--study-name", default=CONFIG.get("study_name"))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", default=CONFIG["results_path"])
    args = parser.parse_args()

    global CURRENT_TRIAL_TARGET
    global BASELINE_FINISHED_COUNT
    global SEED_TRIAL_COUNT
    CURRENT_TRIAL_TARGET = args.trials
    study = optuna.create_study(
        direction=args.direction,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=make_sampler(args.sampler),
        pruner=make_pruner(args.pruner),
    )
    BASELINE_FINISHED_COUNT = len([item for item in study.trials if item.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)])
    SEED_TRIAL_COUNT = add_seed_trials(study)
    output_path = Path(args.output)

    def on_trial_complete(study, trial):
        report_progress(study, trial)
        write_results(study, args.direction, output_path)

    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs, callbacks=[on_trial_complete])
    result = write_results(study, args.direction, output_path)
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
  studyName?: string;
  timeoutSeconds?: number;
  seedTrials?: SeedTrial[];
}): Promise<void> {
  await mkdir(path.dirname(input.outputPath), { recursive: true });
  await writeFile(input.outputPath, renderOptunaRunner(input), "utf8");
  await chmod(input.outputPath, 0o755);
}
