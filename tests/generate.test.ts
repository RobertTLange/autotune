import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
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

  it("fails infrastructure trial errors instead of pruning them", () => {
    const code = renderOptunaRunner({
      invocation: { language: "python", command: ["python3"], script: "/tmp/train.py" },
      searchSpace,
      outputPath: "/tmp/runner.py",
      resultsPath: "/tmp/results.json"
    });

    expect(code).toContain("raise RuntimeError(f\"Trial command exited");
    expect(code).toContain("raise RuntimeError(str(exc)) from exc");
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
    expect(code).toContain("optuna.trial.create_trial");
    expect(code).toContain("SEED_TRIAL_COUNT");
    expect(code).toContain("BASELINE_FINISHED_COUNT");
    expect(code).toContain("if not parameters or study.trials:");
    expect(code).toContain("skipped transferred seed trial");
    expect(code).toContain("argv.extend([parameter[\"cli_flag\"], str(parameter[\"value\"])])");
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
});
