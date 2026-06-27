import { chmod, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { analyzeOnly, doctorAutotune, resumeStudy, runAutotune, showResults } from "../src/workflow.js";
import { writeSearchSpace } from "../src/search-space.js";

describe("runAutotune", () => {
  const originalPath = process.env.PATH;
  const originalHeadless = process.env.AUTOTUNE_HEADLESS_BIN;
  const originalHeadlessArgLog = process.env.AUTOTUNE_HEADLESS_ARG_LOG;

  afterEach(() => {
    process.env.PATH = originalPath;
    if (originalHeadless === undefined) {
      delete process.env.AUTOTUNE_HEADLESS_BIN;
    } else {
      process.env.AUTOTUNE_HEADLESS_BIN = originalHeadless;
    }
    if (originalHeadlessArgLog === undefined) {
      delete process.env.AUTOTUNE_HEADLESS_ARG_LOG;
    } else {
      process.env.AUTOTUNE_HEADLESS_ARG_LOG = originalHeadlessArgLog;
    }
  });

  it("runs analyze, confirmation, generation, and runner phases with argv-safe fakes", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-workflow-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    const progress = await captureStderr(async () => {
      await runAutotune(script, {
        trials: 2,
        direction: "maximize",
        sampler: "tpe",
        pruner: "none",
        nJobs: 1,
        workDir,
        agent: "claude",
        json: true,
        yes: true
      });
    });

    const searchSpace = await readFile(path.join(workDir, "search_space.yaml"), "utf8");
    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    const result = JSON.parse(await readFile(path.join(workDir, "results.json"), "utf8"));
    expect(searchSpace).toContain("cli_flag: --x");
    expect(runner).toContain("subprocess.run(argv");
    expect(result.best_trial.params).toEqual({ x: 0.5 });
    expect(progress).toEqual(
      expect.arrayContaining([
        expect.stringContaining("Phase 1: analyzing"),
        expect.stringContaining("Phase 2: generating"),
        expect.stringContaining("Writing Optuna runner"),
        expect.stringContaining("Running 2 Optuna trials"),
        expect.stringContaining("Trials complete")
      ])
    );
  });

  it("rejects invalid trial counts before spawning tools", async () => {
    await expect(
      runAutotune("train.py", {
        trials: 0,
        direction: "maximize",
        sampler: "tpe",
        pruner: "none",
        nJobs: 1,
        workDir: ".autotune",
        agent: "claude",
        json: false,
        yes: true
      })
    ).rejects.toThrow(/trials/);
  });

  it("runs analyze-only and writes search-space output", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-analyze-"));
    const binDir = path.join(dir, "bin");
    const script = path.join(dir, "train.py");
    const output = path.join(dir, "space.yaml");
    const argLog = path.join(dir, "headless-args.jsonl");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");
    process.env.AUTOTUNE_HEADLESS_ARG_LOG = argLog;

    await analyzeOnly(script, {
      agent: "claude",
      model: "claude-opus-4-6",
      reasoningEffort: "xhigh",
      json: false,
      output,
      workDir: path.join(dir, ".autotune")
    });

    expect(await readFile(output, "utf8")).toContain("reasoning: test metric");
    expect(await readFile(argLog, "utf8")).toContain('"--model","claude-opus-4-6","--reasoning-effort","xhigh"');
  });

  it("renders previous results and resumes from an existing runner", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-resume-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    await mkdir(workDir, { recursive: true });
    await writeFakePython(path.join(binDir, "python3"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    await writeFile(path.join(workDir, "train_optuna.py"), "# fake runner\n", "utf8");
    await writeSearchSpace(path.join(workDir, "search_space.yaml"), {
      parameters: [{ name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 }],
      has_arg_parsing: true,
      needs_wrapper: false,
      direction: "maximize"
    });

    await resumeStudy({
      workDir,
      storage: "sqlite:///study.db",
      trials: 1,
      nJobs: 1,
      direction: "maximize"
    });
    await showResults({ dir: workDir, json: false, top: 1 });

    expect(await readFile(path.join(workDir, "results.json"), "utf8")).toContain("best_trial");
  });

  it("passes stable study names for stored runs and resume", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-study-name-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 2,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      storage: "sqlite:///study.db",
      json: true,
      yes: true
    });
    const runArgv = JSON.parse(await readFile(path.join(workDir, "results.json.argv.json"), "utf8")) as string[];
    expect(runArgv).toEqual(expect.arrayContaining(["--study-name", "train_autotune"]));

    await resumeStudy({
      workDir,
      storage: "sqlite:///study.db",
      trials: 1,
      nJobs: 1,
      direction: "maximize"
    });
    const resumeArgv = JSON.parse(await readFile(path.join(workDir, "results.json.argv.json"), "utf8")) as string[];
    expect(resumeArgv).not.toContain("--study-name");
  });

  it("passes round-specific study names for stored refinement rounds", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-study-rounds-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 2,
      refineRounds: 1,
      refineTrials: 2,
      refineMode: "auto",
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      storage: "sqlite:///study.db",
      json: true,
      yes: true
    });

    const round0Argv = JSON.parse(await readFile(path.join(workDir, "results.round_0.json.argv.json"), "utf8")) as string[];
    const round1Argv = JSON.parse(await readFile(path.join(workDir, "results.round_1.json.argv.json"), "utf8")) as string[];
    expect(round0Argv).toEqual(expect.arrayContaining(["--study-name", "train_autotune_round_0"]));
    expect(round1Argv).toEqual(expect.arrayContaining(["--study-name", "train_autotune_round_1"]));
  });

  it("lets resume override the runner study name explicitly", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-resume-study-name-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    await mkdir(workDir, { recursive: true });
    await writeFakePython(path.join(binDir, "python3"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    await writeFile(path.join(workDir, "train_optuna.py"), "# fake runner\n", "utf8");
    await writeSearchSpace(path.join(workDir, "search_space.yaml"), {
      parameters: [{ name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 }],
      has_arg_parsing: true,
      needs_wrapper: false,
      direction: "maximize"
    });

    await resumeStudy({
      workDir,
      storage: "sqlite:///study.db",
      studyName: "custom_study",
      trials: 1,
      nJobs: 1,
      direction: "maximize"
    });
    const argv = JSON.parse(await readFile(path.join(workDir, "results.json.argv.json"), "utf8")) as string[];
    expect(argv).toEqual(expect.arrayContaining(["--study-name", "custom_study"]));
  });

  it("prints doctor checks for base tools and script runtime", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-doctor-"));
    const binDir = path.join(dir, "bin");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    const lines = await captureStdout(async () => {
      await doctorAutotune({ script, agent: "claude" });
    });

    expect(lines).toEqual(
      expect.arrayContaining([
        expect.stringContaining("python3"),
        expect.stringContaining("optuna"),
        expect.stringContaining("headless"),
        expect.stringContaining("runtime")
      ])
    );
  });

  it("passes feedback to headless and runs with the revised search space", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-feedback-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");
    const answers = ["feedback", "make x wider", "Y"];

    await runAutotune(script, {
      trials: 2,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: false,
      ask: async () => answers.shift() ?? "Y"
    });

    const searchSpace = await readFile(path.join(workDir, "search_space.yaml"), "utf8");
    expect(searchSpace).toContain("low: -1");
    expect(searchSpace).toContain("high: 2");
  });

  it("generates a modified script copy when the search space needs a wrapper", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-wrapper-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "x = 0\nprint('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 2,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: true,
      config: await writeWrapperSearchSpace(dir)
    });

    const modified = path.join(workDir, "train_modified.py");
    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    expect(await readFile(modified, "utf8")).toContain("argparse.ArgumentParser");
    expect(runner).toContain(modified);
    expect(runner).not.toContain(script);
  });

  it("rewrites explicit script slots to generated modified copies", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-wrapper-command-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "x = 0\nprint('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 2,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      command: `python3 -u ${script}`,
      json: true,
      yes: true,
      config: await writeWrapperSearchSpace(dir)
    });

    const modified = path.join(workDir, "train_modified.py");
    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    expect(runner).toContain(modified);
    expect(runner).not.toContain(`-u\\",\\"${script}`);
  });

  it("generates a modified script copy when metric output is missing", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-metric-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(
      script,
      "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--x', type=float, default=0.0)\nargs = parser.parse_args()\nscore = args.x\n",
      "utf8"
    );
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    const progress = await captureStderr(async () => {
      await runAutotune(script, {
        trials: 2,
        direction: "maximize",
        sampler: "tpe",
        pruner: "none",
        nJobs: 1,
        workDir,
        agent: "claude",
        json: true,
        yes: true,
        config: await writeMissingMetricSearchSpace(dir)
      });
    });

    const modified = path.join(workDir, "train_modified.py");
    const searchSpace = await readFile(path.join(workDir, "search_space.yaml"), "utf8");
    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    expect(searchSpace).toContain("has_metric_output: false");
    expect(await readFile(modified, "utf8")).toContain("autotune_metric=");
    expect(runner).toContain(modified);
    expect(progress).toEqual(expect.arrayContaining([expect.stringContaining("adding metric output")]));
  });

  it("uses agent-proposed Optuna settings when CLI settings are absent", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-agent-optuna-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 2,
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: true,
      config: await writeOptunaSearchSpace(dir)
    });

    const argv = JSON.parse(await readFile(path.join(workDir, "results.json.argv.json"), "utf8")) as string[];
    expect(argv).toEqual(expect.arrayContaining(["--direction", "minimize", "--sampler", "random", "--pruner", "hyperband"]));
  });

  it("lets explicit CLI Optuna settings override agent proposals", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-cli-optuna-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 2,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: true,
      config: await writeOptunaSearchSpace(dir)
    });

    const argv = JSON.parse(await readFile(path.join(workDir, "results.json.argv.json"), "utf8")) as string[];
    expect(argv).toEqual(expect.arrayContaining(["--direction", "maximize", "--sampler", "tpe", "--pruner", "none"]));
  });

  it("runs a build command once before checking and using the runtime command", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune build command-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.cpp");
    const runtime = path.join(workDir, "train-bin");
    const buildLog = path.join(dir, "build.log");
    await writeFile(script, "int main() { /* autotune_metric */ return 0; }\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    await writeFakeBuilder(path.join(binDir, "fake-build"), buildLog);
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 2,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      command: "{work-dir}/train-bin",
      buildCommand: "fake-build {script} {work-dir}/train-bin",
      json: true,
      yes: true
    });

    expect(await readFile(buildLog, "utf8")).toBe(`${script}\n${runtime}\n`);
    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    expect(runner).toContain(runtime);
  });

  it("runs automatic agent refinement rounds with separate round artifacts", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-refine-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 2,
      refineRounds: 1,
      refineTrials: 3,
      refineMode: "auto",
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: true
    });

    expect(await readFile(path.join(workDir, "search_space.round_0.yaml"), "utf8")).toContain("high: 1");
    expect(await readFile(path.join(workDir, "search_space.round_1.yaml"), "utf8")).toContain("low: 0.25");
    expect(await readFile(path.join(workDir, "results.round_0.json"), "utf8")).toContain("best_trial");
    expect(await readFile(path.join(workDir, "results.round_1.json"), "utf8")).toContain("best_trial");
    expect(await readFile(path.join(workDir, "refine_prompt.round_1.md"), "utf8")).toContain("Trial result summary");
    expect(await readFile(path.join(workDir, "results.json"), "utf8")).toBe(
      await readFile(path.join(workDir, "results.round_1.json"), "utf8")
    );
  });

  it("passes model and reasoning effort to each headless phase", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-headless-options-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    const argLog = path.join(dir, "headless-args.jsonl");
    await writeFile(script, "x = 0\nprint('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");
    process.env.AUTOTUNE_HEADLESS_ARG_LOG = argLog;

    await runAutotune(script, {
      trials: 2,
      refineRounds: 1,
      refineTrials: 2,
      refineMode: "auto",
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "codex",
      model: "gpt-5.5",
      reasoningEffort: "high",
      json: true,
      yes: true,
      config: await writeWrapperSearchSpace(dir)
    });

    const calls = (await readFile(argLog, "utf8"))
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as string[]);
    const agentCalls = calls.filter((args) => !args.includes("--check"));
    expect(agentCalls.length).toBeGreaterThanOrEqual(3);
    for (const args of agentCalls) {
      expect(args).toEqual(expect.arrayContaining(["codex", "--model", "gpt-5.5", "--reasoning-effort", "high"]));
    }
  });
});

async function captureStderr(action: () => Promise<void>): Promise<string[]> {
  const original = console.error;
  const lines: string[] = [];
  console.error = (value?: unknown) => {
    lines.push(String(value ?? ""));
  };
  try {
    await action();
  } finally {
    console.error = original;
  }
  return lines;
}

async function captureStdout(action: () => Promise<void>): Promise<string[]> {
  const original = console.log;
  const lines: string[] = [];
  console.log = (value?: unknown) => {
    lines.push(String(value ?? ""));
  };
  try {
    await action();
  } finally {
    console.log = original;
  }
  return lines;
}

async function writeFakePython(filePath: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(
    filePath,
    `#!/usr/bin/env node
const fs = require('node:fs');
const args = process.argv.slice(2);
if (args[0] === '--version') {
  console.log('Python 3.12.0');
  process.exit(0);
}
if (args[0] === '-c') {
  console.log('3.6.1');
  process.exit(0);
}
const outputIndex = args.indexOf('--output');
const output = outputIndex >= 0 ? args[outputIndex + 1] : '.autotune/results.json';
const result = {
  study_name: 'train_autotune',
  direction: 'maximize',
  n_trials: 2,
  best_trial: { number: 0, value: 1, params: { x: 0.5 }, state: 'COMPLETE' },
  all_trials: [{ number: 0, value: 1, params: { x: 0.5 }, state: 'COMPLETE' }]
};
fs.mkdirSync(require('node:path').dirname(output), { recursive: true });
fs.writeFileSync(output, JSON.stringify(result));
fs.writeFileSync(output + '.argv.json', JSON.stringify(args));
console.log(JSON.stringify(result));
`,
    "utf8"
  );
  await chmod(filePath, 0o755);
}

async function writeFakeHeadless(filePath: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(
    filePath,
    `#!/usr/bin/env node
if (process.argv.includes('--check')) {
  console.log('claude ok');
  process.exit(0);
}
if (process.env.AUTOTUNE_HEADLESS_ARG_LOG) {
  require('node:fs').appendFileSync(process.env.AUTOTUNE_HEADLESS_ARG_LOG, JSON.stringify(process.argv.slice(2)) + '\\n');
}
if (process.argv.join(' ').includes('modified_prompt')) {
  console.log(JSON.stringify({
    code: "import argparse\\nparser = argparse.ArgumentParser()\\nparser.add_argument('--x', type=float, default=0.0)\\nargs = parser.parse_args()\\nprint(f'autotune_metric={args.x}')\\n"
  }));
  process.exit(0);
}
if (process.argv.join(' ').includes('refine_prompt')) {
  console.log(JSON.stringify({
    parameters: [{ name: 'x', cli_flag: '--x', type: 'float', low: 0.25, high: 0.75 }],
    has_arg_parsing: true,
    needs_wrapper: false,
    direction: 'maximize',
    reasoning: 'narrow x around completed best trials'
  }));
  process.exit(0);
}
console.log(JSON.stringify({
  parameters: [{ name: 'x', cli_flag: '--x', type: 'float', low: process.argv.join(' ').includes('revise_prompt') ? -1 : 0, high: process.argv.join(' ').includes('revise_prompt') ? 2 : 1 }],
  has_arg_parsing: true,
  needs_wrapper: false,
  direction: 'maximize',
  reasoning: 'test metric'
}));
`,
    "utf8"
  );
  await chmod(filePath, 0o755);
}

async function writeFakeBuilder(filePath: string, logPath: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(
    filePath,
    `#!/usr/bin/env node
const fs = require('node:fs');
const [script, output] = process.argv.slice(2);
fs.writeFileSync(${JSON.stringify(logPath)}, script + '\\n' + output + '\\n');
fs.writeFileSync(output, '#!/usr/bin/env node\\nconsole.log("autotune_metric=1")\\n');
fs.chmodSync(output, 0o755);
`,
    "utf8"
  );
  await chmod(filePath, 0o755);
}

async function writeWrapperSearchSpace(dir: string): Promise<string> {
  const filePath = path.join(dir, "wrapper-space.yaml");
  await writeSearchSpace(filePath, {
    parameters: [{ name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 }],
    has_arg_parsing: false,
    needs_wrapper: true,
    direction: "maximize",
    reasoning: "script has hardcoded x"
  });
  return filePath;
}

async function writeMissingMetricSearchSpace(dir: string): Promise<string> {
  const filePath = path.join(dir, "missing-metric-space.yaml");
  await writeSearchSpace(filePath, {
    parameters: [{ name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 }],
    has_arg_parsing: true,
    needs_wrapper: false,
    direction: "maximize",
    reasoning: "script computes score but does not print autotune_metric"
  });
  return filePath;
}

async function writeOptunaSearchSpace(dir: string): Promise<string> {
  const filePath = path.join(dir, "optuna-space.yaml");
  await writeSearchSpace(filePath, {
    parameters: [{ name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 }],
    has_arg_parsing: true,
    needs_wrapper: false,
    direction: "minimize",
    optuna: {
      sampler: "random",
      pruner: "hyperband",
      reasoning: "broad exploratory search"
    }
  });
  return filePath;
}
