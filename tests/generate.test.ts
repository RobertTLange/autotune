import { access, mkdir, mkdtemp, readFile, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { spawn } from "node:child_process";
import path from "node:path";
import { renderOptunaRunner, writeOptunaRunner } from "../src/generate.js";
import { FALLBACK_HEADLESS_PACKAGE } from "../src/headless.js";

const searchSpace = {
  parameters: [{ name: "x", cli_flag: "--x", type: "float", low: -5, high: 5 }],
  has_arg_parsing: true,
  needs_wrapper: false,
  direction: "minimize"
} as const;

const centaurSearchSpace = {
  ...searchSpace,
  parameters: [
    { name: "x", cli_flag: "--x", type: "float", low: -5, high: 5 },
    { name: "y", cli_flag: "--y", type: "float", low: 0, high: 1 },
    { name: "optimizer", cli_flag: "--optimizer", type: "categorical", choices: ["adam", "sgd"] }
  ],
  optuna: {
    sampler: "centaur",
    pruner: "none",
    centaur: { llm_probability: 0.3, warmup_trials: 10, seed: 7 }
  },
  reasoning: "minimize validation loss"
} as const;

describe("renderOptunaRunner", () => {
  it("renders safe subprocess argv and metric parsing", () => {
    const code = renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json"
    });
    expect(code).toContain("subprocess.Popen(");
    expect(code).toContain('os.environ.pop("AUTOTUNE_NODE_EXECUTABLE", CONFIG.get("node_executable"))');
    expect(code).toContain("env={**os.environ, **TARGET_PYTHON_ENV}");
    expect(code).toContain("except BaseException:");
    expect(code).toContain("signal.signal(signal.SIGTERM, request_interrupt)");
    expect(code).toContain("class OutputCapture");
    expect(code).toContain("self.pending = self.pending[-self.max_chars:]");
    expect(code).toContain("metric_error");
    expect(code).toContain("threading.Thread");
    expect(code).toContain("sys.stderr.write(chunk)");
    expect(code).toContain("CONFIG = json.loads");
    expect(code).toContain("shutil.get_terminal_size");
    expect(code).toContain("autotune_metric=");
    expect(code).toContain("def report_progress");
    expect(code).toContain("def print_progress_header");
    expect(code).toContain("def print_progress_row");
    expect(code).toContain("def format_params");
    expect(code).toContain("def write_results");
    expect(code).toContain("threading.get_ident()");
    expect(code).toContain("os.replace(tmp_path, output_path)");
    expect(code).toContain("def on_trial_complete(study, trial):");
    expect(code).toContain("write_results(study, args.direction, output_path)");
    expect(code).toContain("callbacks=[on_trial_complete]");
    expect(code).toContain("file=sys.stderr");
    expect(code).toContain('"RUNNING"');
    expect(code).toContain('"Params"');
    expect(code).not.toContain("params {json.dumps(effective_params(params), sort_keys=True)}");
    expect(code).toContain("def timestamp");
    expect(code).toContain('"Trial"');
    expect(code).toContain('"Value"');
    expect(code).toContain('"Best"');
    expect(code).toContain("trial.suggest_float");
    expect(code).not.toContain("shell=True");
  });

  it("renders bounded cross-platform trial cleanup", () => {
    const code = renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json"
    });

    expect(code).toContain('taskkill = system_root / "System32" / "taskkill.exe"');
    expect(code).toContain('[str(taskkill), "/PID", str(process.pid), "/T", "/F"]');
    expect(code).toContain("def join_output_threads(process, threads, pipe_identities):");
    expect(code).toContain("deadline = time.monotonic() + 1");
    expect(code).toContain("if stream and not thread.is_alive():");
    expect(code).toContain("terminate_output_holders(pipe_identities, process.pid)");
    const centaurCode = renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace: centaurSearchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json",
      headless: { agent: "codex" }
    });
    expect(centaurCode.indexOf('_load_centaur_module("autotune_process_io")'))
      .toBeLessThan(centaurCode.indexOf('_load_centaur_module("autotune_centaur_support")'));
  });

  it("prioritizes recent PIDs in the shipped Python helper", async () => {
    const script = [
      "import sys, time",
      `sys.path.insert(0, ${JSON.stringify(path.resolve("templates"))})`,
      "from autotune_process_io import _recent_process_pids",
      "Entry = type('Entry', (), {})",
      "entries = []",
      "for pid in range(1, 5001):",
      "    entry = Entry()",
      "    entry.name = str(pid)",
      "    entries.append(entry)",
      "recent = _recent_process_pids(entries, time.monotonic() + 5, last_pid=5000, pid_max=32768)",
      "assert len(recent) == 4096",
      "assert recent[0] == 5000",
      "assert recent[-1] == 905",
      "wrapped = _recent_process_pids(entries[-3:] + entries[:3], time.monotonic() + 5, last_pid=3, pid_max=5000)",
      "assert wrapped == [3, 2, 1, 5000, 4999, 4998]",
      "fallback = _recent_process_pids(entries, time.monotonic() + 5, last_pid=0, pid_max=0)",
      "assert fallback[0] == 5000",
      "assert fallback[-1] == 905"
    ].join("\n");

    await expect(runPython(["-c", script])).resolves.toBe("");
  });

  it("seeds stochastic Optuna samplers", () => {
    const code = renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace: { ...searchSpace, optuna: { sampler: "cmaes", seed: 7 } },
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json"
    });

    expect(code).toContain('\\"sampler_seed\\":7');
    expect(code).toContain('TPESampler(seed=CONFIG.get("sampler_seed"))');
    expect(code).toContain('RandomSampler(seed=CONFIG.get("sampler_seed"))');
    expect(code).toContain('CmaEsSampler(seed=CONFIG.get("sampler_seed"))');
    expect(code).toContain('GridSampler(search_space, seed=CONFIG.get("sampler_seed"))');
  });

  it("does not append source scripts for standalone runtime commands", () => {
    const code = renderOptunaRunner({
      invocation: {
        language: "cpp",
        command: ["/work/.autotune/model"],
        script: "/work/model.cpp",
        scriptArgument: "none"
      },
      searchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json"
    });

    expect(code).toContain('\\"script_arg_mode\\":\\"none\\"');
    expect(code).toContain('if script_arg_mode in ("included", "none"):');
  });

  it("preserves legacy direct executable invocations without metadata", () => {
    const script = "/work/train";
    const code = renderOptunaRunner({
      invocation: {
        language: "executable",
        command: [script],
        script
      },
      searchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json"
    });

    expect(code).toContain('\\"script_arg_mode\\":\\"included\\"');
  });

  it("penalizes infrastructure trial errors instead of pruning them", () => {
    const code = renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json"
    });

    expect(code).toContain("def penalize_trial");
    expect(code).toContain("return 100.0 if OPTIMIZATION_DIRECTION == \"minimize\" else -100.0");
    expect(code).toContain("return penalize_trial(trial, \"timeout\", result[\"error\"])");
    expect(code).toContain("return penalize_trial(trial, \"nonzero_exit\"");
    expect(code).toContain("return penalize_trial(trial, \"missing_metric\", exc)");
    expect(code).not.toContain("raise RuntimeError(f\"Trial command timed out");
    expect(code).not.toContain("raise RuntimeError(f\"Trial command exited");
    expect(code).not.toContain("raise optuna.TrialPruned");
  });

  it("renders fixed parameters and transferred seed trials", () => {
    const code = renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace: {
        ...searchSpace,
        fixed_parameters: [{ name: "y", cli_flag: "--y", value: 0.5 }]
      },
      seedTrials: [{ value: 1, params: { x: 0.25, y: 0.5 } }],
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json"
    });

    expect(code).toContain('\\"fixed_parameters\\":[{\\"name\\":\\"y\\",\\"cli_flag\\":\\"--y\\",\\"value\\":0.5}]');
    expect(code).toContain('\\"seed_trials\\":[{\\"value\\":1,\\"params\\":{\\"x\\":0.25,\\"y\\":0.5}}]');
    expect(code).toContain("def effective_params");
    expect(code).toContain("def add_seed_trials");
    expect(code).toContain("def effective_param_hash");
    expect(code).toContain("existing_hashes");
    expect(code).toContain("optuna.trial.create_trial");
    expect(code).toContain("user_attrs=seed_attrs");
    expect(code).toContain('"autotune_transfer": True');
    expect(code).toContain('"autotune_effective_param_hash": seed_hash');
    expect(code).toContain("def autotune_user_attrs");
    expect(code).toContain('if name.startswith("autotune_")');
    expect(code).toContain('"user_attrs": autotune_user_attrs(trial)');
    expect(code).toContain("SEED_TRIAL_COUNT");
    expect(code).toContain("BASELINE_FINISHED_COUNT");
    expect(code).toContain("if not parameters:");
    expect(code).not.toContain("if not parameters or study.trials:");
    expect(code).toContain("skipped transferred seed trial");
    expect(code).toContain("skipped duplicate transferred seed trial");
    expect(code).toContain("argv.extend([parameter[\"cli_flag\"], str(parameter[\"value\"])])");
  });

  it("records trial duration and stops when the run time budget is exhausted", () => {
    const code = renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json",
      timeBudgetSeconds: 86400
    });

    expect(code).toContain('\\"time_budget_seconds\\":86400');
    expect(code).toContain('trial.set_user_attr("autotune_duration_seconds"');
    expect(code).toContain("def cumulative_trial_seconds(study):");
    expect(code).toContain("if not time_budget_exhausted(study):");
    expect(code).toContain("study.stop()");
  });

  it("renders the Centaur sampler and excludes proposal latency from trial duration", () => {
    const code = renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace: centaurSearchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json",
      studyName: "centaur_test",
      headless: { agent: "codex", model: "gpt-5.5", reasoningEffort: "high" }
    });

    expect(code).toContain("from autotune_centaur_runtime import CentaurSampler");
    expect(code).toContain("importlib.util.spec_from_file_location(name, module_path)");
    expect(code).toContain('if name == "centaur":');
    expect(code).toContain("return CentaurSampler(");
    expect(code).toContain('\\"node_executable\\":');
    expect(code).toContain('NODE_EXECUTABLE = os.environ.pop("AUTOTUNE_NODE_EXECUTABLE", CONFIG.get("node_executable"))');
    expect(code).toContain("node_executable=NODE_EXECUTABLE");
    expect(code).toContain(`\\"headless_fallback_package\\":\\"${FALLBACK_HEADLESS_PACKAGE}\\"`);
    expect(code).toContain('headless_fallback_package=CONFIG["headless_fallback_package"]');
    expect(code).toContain('work_dir=Path(__file__).resolve().parent');
    expect(code).toContain('objective_context=CONFIG.get("objective_context")');
    expect(code.indexOf("params = {parameter")).toBeLessThan(code.indexOf("started_at = time.monotonic()"));
    expect(code).toContain('if args.sampler == "centaur" and args.n_jobs != 1:');
  });

  it("requires proposal-agent options for a Centaur runner", () => {
    expect(() => renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace: centaurSearchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json"
    })).toThrow(/Centaur requires proposal-agent options/i);
  });

  it("copies the Centaur runtime companion only for Centaur runners", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-centaur-companion-"));
    const centaurRunner = path.join(dir, "centaur", "train_optuna.py");
    const regularRunner = path.join(dir, "regular", "train_optuna.py");

    await writeOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace: centaurSearchSpace,
      outputPath: centaurRunner,
      resultsPath: path.join(dir, "centaur", "results.json"),
      headless: { agent: "codex" }
    });
    await writeOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace,
      outputPath: regularRunner,
      resultsPath: path.join(dir, "regular", "results.json")
    });

    const companion = path.join(dir, "centaur", "autotune_centaur_runtime.py");
    const support = path.join(dir, "centaur", "autotune_centaur_support.py");
    const headlessLock = path.join(dir, "centaur", "autotune_headless_runtime.lock.json");
    await expect(access(companion)).resolves.toBeUndefined();
    await expect(access(support)).resolves.toBeUndefined();
    await expect(access(headlessLock)).resolves.toBeUndefined();
    await expect(runPython(["-m", "py_compile", companion, support, centaurRunner])).resolves.toBe("");
    const runtime = await readFile(companion, "utf8");
    expect(runtime).toContain('"--allow"');
    expect(runtime).toContain('"read-only"');
    expect(runtime).toContain('"ci",');
    expect(runtime).toContain('"--ignore-scripts",');
    expect(runtime).toContain("_resolve_npm_command(node)");
    expect(runtime).not.toContain("shell=True");
    await expect(access(path.join(dir, "regular", "autotune_centaur_runtime.py"))).rejects.toThrow();
    await expect(access(path.join(dir, "regular", "autotune_centaur_support.py"))).rejects.toThrow();
    await expect(access(path.join(dir, "regular", "autotune_headless_runtime.lock.json"))).rejects.toThrow();
  });

  it("replaces output symlinks without overwriting their targets", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-atomic-runner-"));
    const outputDirectory = path.join(dir, "output");
    const target = path.join(dir, "target.txt");
    const runner = path.join(outputDirectory, "train_optuna.py");
    await mkdir(outputDirectory);
    await writeFile(target, "preserve me", "utf8");
    await symlink(target, runner);

    await writeOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace,
      outputPath: runner,
      resultsPath: path.join(dir, "results.json")
    });

    expect(await readFile(target, "utf8")).toBe("preserve me");
    expect(await readFile(runner, "utf8")).toContain("def objective");
  });

  it("does not start another trial when a resumed study exhausted its time budget", async () => {
    if (!(await pythonHasOptuna())) {
      return;
    }
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-resume-budget-runner-"));
    const train = path.join(dir, "train.py");
    const runner = path.join(dir, "train_optuna.py");
    const marker = path.join(dir, "trial-count.txt");
    const storage = `sqlite:///${path.join(dir, "study.db")}`;
    const runnerArgs = [
      "--trials", "1",
      "--direction", "minimize",
      "--sampler", "random",
      "--pruner", "none",
      "--n-jobs", "1",
      "--storage", storage
    ];
    await writeFile(
      train,
      `from pathlib import Path\nmarker = Path(${JSON.stringify(marker)})\nmarker.write_text(marker.read_text() + "x" if marker.exists() else "x")\nprint("autotune_metric=0.5")\n`,
      "utf8"
    );
    await writeOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: train },
      searchSpace,
      outputPath: runner,
      resultsPath: path.join(dir, "first-results.json"),
      studyName: "resume_budget_test"
    });
    await runPython([runner, ...runnerArgs]);

    await writeOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: train },
      searchSpace,
      outputPath: runner,
      resultsPath: path.join(dir, "second-results.json"),
      studyName: "resume_budget_test",
      timeBudgetSeconds: 0.000001
    });
    await runPython([runner, ...runnerArgs]);

    expect(await readFile(marker, "utf8")).toBe("x");
  });

  it("keeps the latest streamed metric when a trial times out", async () => {
    if (!(await pythonHasOptuna())) {
      return;
    }
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-timeout-runner-"));
    const train = path.join(dir, "train.py");
    const runner = path.join(dir, "train_optuna.py");
    const results = path.join(dir, "results.json");
    await writeFile(
      train,
      "import time\nprint('autotune_metric=0.75', flush=True)\ntime.sleep(2)\n",
      "utf8"
    );
    await writeOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: train },
      searchSpace,
      outputPath: runner,
      resultsPath: results,
      timeoutSeconds: 1
    });

    await runPython([runner, "--trials", "1", "--direction", "minimize", "--sampler", "random", "--pruner", "none", "--n-jobs", "1"]);

    const result = JSON.parse(await readFile(results, "utf8"));
    expect(result.all_trials).toHaveLength(1);
    expect(result.all_trials[0].state).toBe("COMPLETE");
    expect(result.all_trials[0].value).toBe(0.75);
    expect(result.all_trials[0].user_attrs).toMatchObject({
      autotune_failure_reason: "timeout_after_metric"
    });
  });

  it.skipIf(process.platform === "win32")("terminates active trials when the runner is stopped", async () => {
    if (!(await pythonHasOptuna())) {
      return;
    }
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-signal-runner-"));
    const train = path.join(dir, "train.py");
    const runner = path.join(dir, "train_optuna.py");
    const marker = path.join(dir, "trial.pid");
    await writeFile(
      train,
      `import os,time\nfrom pathlib import Path\nPath(${JSON.stringify(marker)}).write_text(str(os.getpid()))\ntime.sleep(60)\nprint('autotune_metric=1')\n`,
      "utf8"
    );
    await writeOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: train },
      searchSpace,
      outputPath: runner,
      resultsPath: path.join(dir, "results.json")
    });
    const controller = spawn("python3", [
      "-I", runner, "--trials", "1", "--direction", "minimize", "--sampler", "random",
      "--pruner", "none", "--n-jobs", "1"
    ], { stdio: "ignore" });
    let trialPid: number | undefined;
    try {
      trialPid = Number(await waitForFileText(marker));
      controller.kill("SIGTERM");
      await new Promise<void>((resolve) => controller.once("close", () => resolve()));
      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(processIsAlive(trialPid)).toBe(false);
    } finally {
      controller.kill("SIGKILL");
      if (trialPid && processIsAlive(trialPid)) {
        process.kill(-trialPid, "SIGKILL");
      }
    }
  }, 15_000);

  it("writes an executable runner file", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-generate-"));
    const runner = path.join(dir, "train_optuna.py");
    await writeOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: path.join(dir, "train.py") },
      searchSpace,
      outputPath: runner,
      resultsPath: path.join(dir, "results.json")
    });
    await writeFile(path.join(dir, "train.py"), "print('autotune_metric=1')\n");
    expect(await readFile(runner, "utf8")).toContain("def objective");
  });

  it("imports transferred seeds idempotently with provenance", async () => {
    if (!(await pythonHasOptuna())) {
      return;
    }
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-seed-runner-"));
    const train = path.join(dir, "train.py");
    const runner = path.join(dir, "train_optuna.py");
    const results = path.join(dir, "results.json");
    const storage = `sqlite:///${path.join(dir, "study.db")}`;
    await writeFile(train, "print('autotune_metric=0.5')\n", "utf8");
    await writeOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: train },
      searchSpace,
      outputPath: runner,
      resultsPath: results,
      studyName: "seed_transfer_test",
      seedTrials: [
        { value: 1, params: { x: 1 }, source_round: 0, source_trial_number: 7 },
        { value: 1, params: { x: 1.0 }, source_round: 0, source_trial_number: 7 }
      ]
    });

    await runPython([runner, "--trials", "0", "--direction", "minimize", "--sampler", "random", "--pruner", "none", "--n-jobs", "1", "--storage", storage]);
    await runPython([runner, "--trials", "0", "--direction", "minimize", "--sampler", "random", "--pruner", "none", "--n-jobs", "1", "--storage", storage]);

    const result = JSON.parse(await readFile(results, "utf8"));
    expect(result.all_trials).toHaveLength(1);
    expect(result.all_trials[0].params).toEqual({ x: 1 });
    expect(result.all_trials[0].user_attrs).toMatchObject({
      autotune_transfer: true,
      autotune_source_round: 0,
      autotune_source_trial_number: 7
    });
    expect(Object.keys(result.all_trials[0].user_attrs).every((key) => key.startsWith("autotune_"))).toBe(true);
  });
});

async function pythonHasOptuna(): Promise<boolean> {
  try {
    await runPython(["-c", "import optuna"]);
    return true;
  } catch {
    return false;
  }
}

function runPython(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn("python3", args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve(stdout);
        return;
      }
      reject(new Error(`python3 ${args.join(" ")} failed with ${code}: ${stderr || stdout}`));
    });
  });
}

async function waitForFileText(filePath: string): Promise<string> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      return await readFile(filePath, "utf8");
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
  throw new Error(`timed out waiting for ${filePath}`);
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
