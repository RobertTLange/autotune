import { chmod, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { analyzeOnly, doctorAutotune, resumeStudy, runAutotune, showResults } from "../src/workflow.js";
import { writeSearchSpace } from "../src/search-space.js";

describe("runAutotune", () => {
  const originalPath = process.env.PATH;
  const originalHeadless = process.env.AUTOTUNE_HEADLESS_BIN;

  afterEach(() => {
    process.env.PATH = originalPath;
    if (originalHeadless === undefined) {
      delete process.env.AUTOTUNE_HEADLESS_BIN;
    } else {
      process.env.AUTOTUNE_HEADLESS_BIN = originalHeadless;
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
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await analyzeOnly(script, {
      agent: "claude",
      json: false,
      output,
      workDir: path.join(dir, ".autotune")
    });

    expect(await readFile(output, "utf8")).toContain("reasoning: test metric");
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
