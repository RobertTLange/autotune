import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { spawn } from "node:child_process";
import path from "node:path";
import { renderOptunaRunner, writeOptunaRunner } from "../src/generate.js";

const searchSpace = {
  parameters: [{ name: "x", cli_flag: "--x", type: "float", low: -5, high: 5 }],
  has_arg_parsing: true,
  needs_wrapper: false,
  direction: "minimize"
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
