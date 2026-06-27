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
    expect(code).toContain("autotune_metric=");
    expect(code).toContain("def report_progress");
    expect(code).toContain("file=sys.stderr");
    expect(code).toContain("def timestamp");
    expect(code).toContain("[{timestamp()}] Trial");
    expect(code).toContain("value=");
    expect(code).toContain("best=");
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
