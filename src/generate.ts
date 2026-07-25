import { chmod, mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { FALLBACK_HEADLESS_PACKAGE } from "./headless.js";
import type { HeadlessOptions, Invocation, SearchSpace } from "./types.js";

export interface SeedTrial {
  value: number;
  params: Record<string, string | number | boolean>;
  source_round?: number;
  source_trial_number?: number;
}

const CENTAUR_RUNTIME_URL = new URL("../templates/autotune_centaur_runtime.py", import.meta.url);
const CENTAUR_SUPPORT_URL = new URL("../templates/autotune_centaur_support.py", import.meta.url);

export function renderOptunaRunner(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  outputPath: string;
  resultsPath: string;
  studyName?: string;
  timeoutSeconds?: number;
  timeBudgetSeconds?: number;
  seedTrials?: SeedTrial[];
  headless?: HeadlessOptions;
}): string {
  if (input.searchSpace.optuna?.sampler === "centaur" && !input.headless?.agent.trim()) {
    throw new Error("Centaur requires proposal-agent options");
  }
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
    direction: input.searchSpace.direction,
    sampler_seed: input.searchSpace.optuna?.seed,
    failure_value: input.searchSpace.failure_value,
    max_output_bytes: 65536,
    timeout,
    time_budget_seconds: input.timeBudgetSeconds,
    objective_context: input.searchSpace.reasoning,
    headless_fallback_package: FALLBACK_HEADLESS_PACKAGE,
    centaur: input.searchSpace.optuna?.sampler === "centaur"
      ? {
          ...input.searchSpace.optuna.centaur,
          agent: input.headless?.agent,
          model: input.headless?.model,
          reasoning_effort: input.headless?.reasoningEffort
        }
      : undefined
  };

  return `#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import optuna
from optuna.trial import TrialState
${input.searchSpace.optuna?.sampler === "centaur" ? `
def _load_centaur_module(name):
    module_path = Path(__file__).resolve().with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_load_centaur_module("autotune_centaur_support")
_load_centaur_module("autotune_centaur_runtime")
from autotune_centaur_runtime import CentaurSampler
` : ""}

CONFIG = json.loads(${JSON.stringify(JSON.stringify(payload))})
TARGET_PYTHON_ENV = json.loads(os.environ.pop("AUTOTUNE_TARGET_PYTHON_ENV", "{}"))
CURRENT_TRIAL_TARGET = 0
BASELINE_FINISHED_COUNT = 0
SEED_TRIAL_COUNT = 0
STARTED_TRIAL_COUNT = 0
PROGRESS_HEADER_PRINTED = False
PROGRESS_LOCK = threading.Lock()
ACTIVE_PROCESSES = set()
ACTIVE_PROCESS_LOCK = threading.Lock()
INTERRUPTED = threading.Event()
SPAWNING_PROCESS_COUNT = 0
OPTIMIZATION_DIRECTION = CONFIG.get("direction", "maximize")
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


def parameter_type_by_name():
    mapping = {parameter["name"]: parameter["type"] for parameter in CONFIG["parameters"]}
    for parameter in CONFIG.get("fixed_parameters", []):
        mapping.setdefault(parameter["name"], type(parameter["value"]).__name__)
    return mapping


def canonical_param_value(name, value):
    parameter_type = parameter_type_by_name().get(name)
    if parameter_type == "float":
        return float(value)
    if parameter_type == "int":
        return int(value)
    return value


def canonical_params(params):
    return {name: canonical_param_value(name, value) for name, value in sorted(params.items())}


def effective_param_hash(params):
    encoded = json.dumps(canonical_params(params), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def existing_trial_hashes(study):
    hashes = set()
    for trial in study.trials:
        try:
            hashes.add(effective_param_hash(effective_params(trial.params)))
        except Exception:
            continue
    return hashes


def autotune_user_attrs(trial):
    return {
        name: value
        for name, value in trial.user_attrs.items()
        if name.startswith("autotune_")
    }


def parse_metric(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("autotune_metric="):
            return float(line.split("=", 1)[1].strip())
    raise ValueError("No autotune_metric= line found in trial stdout")


def failure_value():
    configured = CONFIG.get("failure_value")
    if configured is not None:
        return float(configured)
    return 100.0 if OPTIMIZATION_DIRECTION == "minimize" else -100.0


def penalize_trial(trial, reason, detail=None):
    value = failure_value()
    trial.set_user_attr("autotune_failure_reason", reason)
    if detail:
        trial.set_user_attr("autotune_failure_detail", str(detail)[-1000:])
    print(f"[{timestamp()}] trial penalized with {value}: {reason}", file=sys.stderr, flush=True)
    return value


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
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, 9)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def request_interrupt(signum, frame):
    INTERRUPTED.set()
    for process in tuple(ACTIVE_PROCESSES):
        kill_process_tree(process)
    if SPAWNING_PROCESS_COUNT == 0:
        raise KeyboardInterrupt


def terminate_active_trials():
    with ACTIVE_PROCESS_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for process in processes:
        kill_process_tree(process)


def run_trial_command(argv):
    global SPAWNING_PROCESS_COUNT
    max_output_bytes = CONFIG.get("max_output_bytes", 65536)
    stdout_capture = OutputCapture(max_output_bytes)
    stderr_capture = OutputCapture(max_output_bytes)
    with ACTIVE_PROCESS_LOCK:
        SPAWNING_PROCESS_COUNT += 1
    try:
        process = subprocess.Popen(
            argv,
            env={**os.environ, **TARGET_PYTHON_ENV},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=hasattr(os, "setsid"),
        )
        with ACTIVE_PROCESS_LOCK:
            ACTIVE_PROCESSES.add(process)
    finally:
        with ACTIVE_PROCESS_LOCK:
            SPAWNING_PROCESS_COUNT -= 1
    if INTERRUPTED.is_set():
        kill_process_tree(process)
    stdout_thread = threading.Thread(target=drain_stream, args=(process.stdout, stdout_capture, True, True))
    stderr_thread = threading.Thread(target=drain_stream, args=(process.stderr, stderr_capture, False, True))
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    error = None
    try:
        returncode = process.wait(timeout=CONFIG["timeout"])
        if INTERRUPTED.is_set():
            raise KeyboardInterrupt
    except subprocess.TimeoutExpired:
        timed_out = True
        error = f"Trial command timed out after {CONFIG['timeout']}s"
        kill_process_tree(process)
        returncode = process.wait()
    except BaseException:
        kill_process_tree(process)
        process.wait()
        raise
    finally:
        stdout_thread.join()
        stderr_thread.join()
        with ACTIVE_PROCESS_LOCK:
            ACTIVE_PROCESSES.discard(process)
    return {
        "returncode": returncode,
        "stdout": stdout_capture.tail,
        "stderr": stderr_capture.tail,
        "metric": stdout_capture.metric,
        "metric_error": stdout_capture.metric_error,
        "timed_out": timed_out,
        "error": error,
    }


def objective(trial):
    global STARTED_TRIAL_COUNT
    timer = time.monotonic()
    params = {parameter["name"]: suggest_value(trial, parameter) for parameter in CONFIG["parameters"]}
    started_at = time.monotonic() if CONFIG.get("centaur") else timer
    trial.set_user_attr("autotune_effective_param_hash", effective_param_hash(effective_params(params)))
    try:
        with PROGRESS_LOCK:
            STARTED_TRIAL_COUNT += 1
            started = STARTED_TRIAL_COUNT
        print_progress_row(f"{started}/{CURRENT_TRIAL_TARGET}", "RUNNING", None, None, effective_params(params))
        argv = build_argv(params)
        result = run_trial_command(argv)
        if result["metric_error"] is not None:
            return penalize_trial(trial, "invalid_metric", result["metric_error"])
        if result["metric"] is not None:
            if result["timed_out"]:
                trial.set_user_attr("autotune_failure_reason", "timeout_after_metric")
                trial.set_user_attr("autotune_failure_detail", result["error"])
            elif result["returncode"] != 0:
                trial.set_user_attr("autotune_failure_reason", "nonzero_exit_after_metric")
                trial.set_user_attr("autotune_failure_detail", result["stderr"][-1000:])
            return result["metric"]
        if result["timed_out"]:
            return penalize_trial(trial, "timeout", result["error"])
        if result["returncode"] != 0:
            return penalize_trial(trial, "nonzero_exit", f"Trial command exited {result['returncode']}: {result['stderr'][-1000:]}")
        try:
            return parse_metric(result["stdout"])
        except Exception as exc:
            return penalize_trial(trial, "missing_metric", exc)
    finally:
        trial.set_user_attr("autotune_duration_seconds", time.monotonic() - started_at)


def make_sampler(name, study_name=None, direction=None, storage=None):
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=CONFIG.get("sampler_seed"))
    if name == "random":
        return optuna.samplers.RandomSampler(seed=CONFIG.get("sampler_seed"))
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=CONFIG.get("sampler_seed"))
    if name == "centaur":
        config = CONFIG["centaur"]
        return CentaurSampler(
            parameters=CONFIG["parameters"],
            fixed_parameters=CONFIG.get("fixed_parameters", []),
            direction=direction or CONFIG.get("direction", "maximize"),
            study_name=study_name or CONFIG.get("study_name"),
            storage=storage,
            work_dir=Path(__file__).resolve().parent,
            objective_context=CONFIG.get("objective_context"),
            llm_probability=config["llm_probability"],
            warmup_trials=config["warmup_trials"],
            seed=config["seed"],
            agent=config["agent"],
            model=config.get("model"),
            reasoning_effort=config.get("reasoning_effort"),
            headless_fallback_package=CONFIG["headless_fallback_package"],
        )
    if name == "grid":
        search_space = {}
        for parameter in CONFIG["parameters"]:
            if parameter["type"] != "categorical":
                raise ValueError("grid sampler requires categorical parameters only")
            search_space[parameter["name"]] = parameter["choices"]
        return optuna.samplers.GridSampler(search_space, seed=CONFIG.get("sampler_seed"))
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
    if not parameters:
        return 0
    distributions = {parameter["name"]: distribution_for_parameter(parameter) for parameter in parameters}
    active_names = set(distributions)
    existing_hashes = existing_trial_hashes(study)
    count = 0
    for seed in CONFIG.get("seed_trials", []):
        params = {name: value for name, value in seed["params"].items() if name in active_names}
        if set(params) != active_names:
            print(f"[{timestamp()}] skipped transferred seed trial with incomplete active params", file=sys.stderr, flush=True)
            continue
        seed_hash = effective_param_hash(effective_params(params))
        if seed_hash in existing_hashes:
            print(f"[{timestamp()}] skipped duplicate transferred seed trial", file=sys.stderr, flush=True)
            continue
        seed_attrs = {
            "autotune_transfer": True,
            "autotune_effective_param_hash": seed_hash,
        }
        if "source_round" in seed:
            seed_attrs["autotune_source_round"] = seed["source_round"]
        if "source_trial_number" in seed:
            seed_attrs["autotune_source_trial_number"] = seed["source_trial_number"]
        try:
            study.add_trial(optuna.trial.create_trial(params=params, distributions=distributions, value=seed["value"], user_attrs=seed_attrs))
            existing_hashes.add(seed_hash)
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
        "user_attrs": autotune_user_attrs(trial),
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


def format_params(params, max_chars):
    text = " ".join(f"{name}={format_number(value)}" for name, value in sorted(params.items()))
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return text[: max_chars - 3] + "..."


def progress_columns():
    terminal_width = shutil.get_terminal_size((120, 24)).columns
    prefix_width = 11
    separator_width = 8
    fixed_columns = [
        ("Trial", 7, "right"),
        ("State", 8, "left"),
        ("Value", 10, "right"),
        ("Best", 10, "right"),
    ]
    fixed_width = sum(width for _, width, _ in fixed_columns)
    params_width = max(8, terminal_width - prefix_width - separator_width - fixed_width)
    return [
        *fixed_columns,
        ("Params", params_width, "left"),
    ]


def progress_prefix():
    return style(f"[{timestamp()}]", "2")


def progress_row(values, columns):
    return "  ".join(
        pad_cell(value, width, align)
        for value, (_, width, align) in zip(values, columns)
    )


def print_progress_header():
    global PROGRESS_HEADER_PRINTED
    if PROGRESS_HEADER_PRINTED:
        return
    columns = progress_columns()
    print(f"{progress_prefix()} {style(progress_row([name for name, _, _ in columns], columns), '1')}", file=sys.stderr, flush=True)
    print(f"{progress_prefix()} {style(progress_row(['-' * width for _, width, _ in columns], columns), '2')}", file=sys.stderr, flush=True)
    PROGRESS_HEADER_PRINTED = True


def print_progress_row(trial_label, state, value, best, params):
    with PROGRESS_LOCK:
        print_progress_header()
        columns = progress_columns()
        params_width = columns[-1][1]
        print(
            f"{progress_prefix()} {progress_row([trial_label, style_state(state), format_number(value), format_number(best), format_params(params, params_width)], columns)}",
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


def cumulative_trial_seconds(study):
    total = 0.0
    for trial in study.trials:
        value = trial.user_attrs.get("autotune_duration_seconds")
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def time_budget_exhausted(study):
    time_budget_seconds = CONFIG.get("time_budget_seconds")
    if not time_budget_seconds:
        return False
    used_seconds = cumulative_trial_seconds(study)
    if used_seconds < time_budget_seconds:
        return False
    print(f"[{timestamp()}] time budget reached: {used_seconds:.1f}s / {time_budget_seconds}s", file=sys.stderr, flush=True)
    return True


def main():
    signal.signal(signal.SIGINT, request_interrupt)
    signal.signal(signal.SIGTERM, request_interrupt)
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
    if args.sampler == "centaur" and args.n_jobs != 1:
        parser.error("Centaur requires --n-jobs 1")

    global CURRENT_TRIAL_TARGET
    global BASELINE_FINISHED_COUNT
    global SEED_TRIAL_COUNT
    global OPTIMIZATION_DIRECTION
    CURRENT_TRIAL_TARGET = args.trials
    OPTIMIZATION_DIRECTION = args.direction
    study = optuna.create_study(
        direction=args.direction,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=make_sampler(args.sampler, args.study_name, args.direction, args.storage),
        pruner=make_pruner(args.pruner),
    )
    BASELINE_FINISHED_COUNT = len([item for item in study.trials if item.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)])
    SEED_TRIAL_COUNT = add_seed_trials(study)
    output_path = Path(args.output)

    def on_trial_complete(study, trial):
        report_progress(study, trial)
        write_results(study, args.direction, output_path)
        if time_budget_exhausted(study):
            study.stop()

    try:
        if not time_budget_exhausted(study):
            study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs, callbacks=[on_trial_complete])
    finally:
        terminate_active_trials()
    if INTERRUPTED.is_set():
        raise KeyboardInterrupt
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
  timeBudgetSeconds?: number;
  seedTrials?: SeedTrial[];
  headless?: HeadlessOptions;
}): Promise<void> {
  await mkdir(path.dirname(input.outputPath), { recursive: true });
  await writeAtomicFile(input.outputPath, renderOptunaRunner(input), 0o755);
  if (input.searchSpace.optuna?.sampler === "centaur") {
    const outputDirectory = path.dirname(input.outputPath);
    const runtimePath = path.join(outputDirectory, "autotune_centaur_runtime.py");
    const supportPath = path.join(outputDirectory, "autotune_centaur_support.py");
    await writeAtomicFile(runtimePath, await readFile(CENTAUR_RUNTIME_URL), 0o600);
    await writeAtomicFile(supportPath, await readFile(CENTAUR_SUPPORT_URL), 0o600);
  }
}

async function writeAtomicFile(filePath: string, content: string | Uint8Array, mode: number): Promise<void> {
  const temporaryPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, content, { flag: "wx", mode });
    await chmod(temporaryPath, mode);
    await rename(temporaryPath, filePath);
  } finally {
    await unlink(temporaryPath).catch((error: NodeJS.ErrnoException) => {
      if (error.code !== "ENOENT") {
        throw error;
      }
    });
  }
}
