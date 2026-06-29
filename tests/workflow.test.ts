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
        timeoutSeconds: 1800,
        workDir,
        agent: "claude",
        json: true,
        yes: true
      });
    });

    const searchSpace = await readFile(path.join(workDir, "search_space.yaml"), "utf8");
    const analyzePrompt = await readFile(path.join(workDir, "analyze_prompt.md"), "utf8");
    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    const result = JSON.parse(await readFile(path.join(workDir, "results.json"), "utf8"));
    expect(searchSpace).toContain("cli_flag: --x");
    expect(analyzePrompt).toContain("initial_trials: 2");
    expect(analyzePrompt).toContain("per_trial_timeout_seconds: 1800");
    expect(runner).toContain("subprocess.Popen(");
    expect(runner).toContain('\\"timeout\\":1800');
    expect(result.best_trial.params).toEqual({ x: 0.5 });
    expect(progress).toEqual(
      expect.arrayContaining([
        expect.stringContaining("Phase 1: analyzing"),
        expect.stringContaining("Writing Optuna runner"),
        expect.stringContaining("Running 2 Optuna trials"),
        expect.stringContaining("Trials complete")
      ])
    );
  });

  it("uses a timestamped default run directory next to the script", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-default-workdir-"));
    const binDir = path.join(dir, "bin");
    const script = path.join(dir, "train.py");
    const output = path.join(dir, "chosen-results.json");
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
      agent: "claude",
      json: true,
      output,
      yes: true
    });

    const latestPath = path.join(dir, "autotune", "train.py", "latest.json");
    const latest = JSON.parse(await readFile(latestPath, "utf8")) as { run_dir: string; work_dir: string };
    const rootLatest = JSON.parse(await readFile(path.join(dir, "autotune", "latest.json"), "utf8")) as { run_dir: string };
    const runDir = path.join(dir, "autotune", "train.py", latest.run_dir);
    expect(latest.run_dir).toMatch(/^runs\/\d{4}-\d{2}-\d{2}T\d{6}\d{3}Z-[0-9a-f-]{36}$/);
    expect(rootLatest.run_dir).toBe(path.join("train.py", latest.run_dir));
    expect(latest.work_dir).toBe(runDir);
    expect(await readFile(path.join(runDir, "results.json"), "utf8")).toContain("best_trial");
    expect(await readFile(output, "utf8")).toContain("best_trial");
    expect(await readFile(path.join(runDir, "analyze_prompt.md"), "utf8")).toContain("initial_trials: 2");
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

  it("skips headless prerequisites for accepted compatible configs", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-config-no-headless-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePython(path.join(binDir, "python3"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    delete process.env.AUTOTUNE_HEADLESS_BIN;

    await runAutotune(script, {
      trials: 2,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "definitely-missing-agent",
      json: true,
      yes: true,
      config: await writeOptunaSearchSpace(dir)
    });

    expect(await readFile(path.join(workDir, "results.json"), "utf8")).toContain("best_trial");
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
      timeoutSeconds: 1800,
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
    const revisePrompt = await readFile(path.join(workDir, "revise_prompt.md"), "utf8");
    expect(searchSpace).toContain("low: -1");
    expect(searchSpace).toContain("high: 2");
    expect(revisePrompt).toContain("initial_trials: 2");
    expect(revisePrompt).toContain("total_planned_trials: 2");
    expect(revisePrompt).toContain("per_trial_timeout_seconds: 1800");
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
    expect(await readFile(path.join(workDir, "train_optuna.round_0.py"), "utf8")).toContain("study_name");
    expect(await readFile(path.join(workDir, "train_optuna.round_1.py"), "utf8")).toContain("study_name");
    const manifest = JSON.parse(await readFile(path.join(workDir, "rounds.json"), "utf8"));
    expect(manifest.rounds).toHaveLength(2);
    expect(manifest.rounds[0]).toMatchObject({
      round: 0,
      trials: 2,
      seed_count: 0,
      study_name: "train_autotune_round_0"
    });
    expect(manifest.rounds[0].runner_path).toContain("train_optuna.round_0.py");
    expect(manifest.rounds[1]).toMatchObject({
      round: 1,
      trials: 3,
      seed_count: 1,
      study_name: "train_autotune_round_1"
    });
    expect(manifest.rounds[1].runner_path).toContain("train_optuna.round_1.py");
    const refinePrompt = await readFile(path.join(workDir, "refine_prompt.round_1.md"), "utf8");
    expect(refinePrompt).toContain("Trial result summary");
    expect(refinePrompt).toContain("current_refinement_round: 1");
    expect(refinePrompt).toContain("current_round_trials: 3");
    expect(refinePrompt).toContain('"state_counts"');
    expect(refinePrompt).toContain('"transfer_counts"');
    expect(refinePrompt).toContain('"boundary_hits"');
    expect(refinePrompt).toContain('"value_samples"');
    expect(refinePrompt).toContain('"performance_samples"');
    expect(await readFile(path.join(workDir, "results.json"), "utf8")).toBe(
      await readFile(path.join(workDir, "results.round_1.json"), "utf8")
    );
  });

  it("fixes dropped parameters and seeds valid previous trials during refinement", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-refine-transfer-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePythonWithResult(path.join(binDir, "python3"), {
      study_name: "train_autotune",
      direction: "maximize",
      n_trials: 3,
      best_trial: { number: 0, value: 1, params: { x: 0.5, y: 0.5, z: "keep" }, state: "COMPLETE" },
      all_trials: [
        { number: 0, value: 1, params: { x: 0.5, y: 0.5, z: "keep" }, state: "COMPLETE" },
        { number: 1, value: 0.8, params: { x: 0.6, y: 0.2, z: "keep" }, state: "COMPLETE" },
        { number: 2, value: 0.7, params: { x: 0.9, y: 0.5, z: "keep" }, state: "COMPLETE" }
      ]
    });
    await writeFakeHeadless(path.join(binDir, "headless"), { refinedFixed: true });
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 3,
      refineRounds: 1,
      refineTrials: 2,
      refineMode: "auto",
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: true,
      config: await writeTwoParameterSearchSpace(dir)
    });

    const searchSpace = await readFile(path.join(workDir, "search_space.round_1.yaml"), "utf8");
    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    expect(searchSpace).toContain("fixed_parameters");
    expect(searchSpace).toContain("name: y");
    expect(searchSpace).toContain("value: 0.5");
    expect(searchSpace).toContain("name: z");
    expect(searchSpace).toContain("value: keep");
    expect(runner).toContain('\\"fixed_parameters\\":[{\\"name\\":\\"z\\",\\"cli_flag\\":\\"--z\\",\\"value\\":\\"keep\\"},{\\"name\\":\\"y\\",\\"cli_flag\\":\\"--y\\",\\"value\\":0.5}]');
    expect(runner).toContain('\\"seed_trials\\":[{\\"value\\":1,\\"params\\":{\\"x\\":0.5,\\"y\\":0.5,\\"z\\":\\"keep\\"},\\"source_round\\":0,\\"source_trial_number\\":0}]');
    expect(runner).not.toContain('\\"x\\":0.9');
  });

  it("rejects transferred trials with stale params outside the refined effective space", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-refine-stale-seed-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePythonWithResult(path.join(binDir, "python3"), {
      study_name: "train_autotune",
      direction: "maximize",
      n_trials: 1,
      best_trial: { number: 0, value: 1, params: { x: 0.5, y: 0.5 }, state: "COMPLETE" },
      all_trials: [{ number: 0, value: 1, params: { x: 0.5, y: 0.5 }, state: "COMPLETE" }]
    });
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 1,
      refineRounds: 1,
      refineTrials: 1,
      refineMode: "auto",
      refineTransferFixedParams: false,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: true,
      config: await writeTwoParameterSearchSpace(dir)
    });

    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    expect(runner).toContain('\\"fixed_parameters\\":[]');
    expect(runner).toContain('\\"seed_trials\\":[]');
  });

  it("fills added active parameters from current_value when transferring trials", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-refine-added-param-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePythonWithResult(path.join(binDir, "python3"), {
      study_name: "train_autotune",
      direction: "maximize",
      n_trials: 1,
      best_trial: { number: 0, value: 1, params: { x: 0.5 }, state: "COMPLETE" },
      all_trials: [{ number: 0, value: 1, params: { x: 0.5 }, state: "COMPLETE" }]
    });
    await writeFakeHeadless(path.join(binDir, "headless"), { addedParamCurrentValue: true });
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 1,
      refineRounds: 1,
      refineTrials: 1,
      refineMode: "auto",
      refineTransferFixedParams: false,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: true,
      config: await writeSingleParameterSearchSpace(dir)
    });

    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    expect(runner).toContain('\\"seed_trials\\":[{\\"value\\":1,\\"params\\":{\\"x\\":0.5,\\"y\\":0.25},\\"source_round\\":0,\\"source_trial_number\\":0}]');
  });

  it("keeps dropped parameters fixed after feedback revises a refined space", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-refine-feedback-transfer-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePythonWithResult(path.join(binDir, "python3"), {
      study_name: "train_autotune",
      direction: "maximize",
      n_trials: 1,
      best_trial: { number: 0, value: 1, params: { x: 0.5, y: 0.5 }, state: "COMPLETE" },
      all_trials: [{ number: 0, value: 1, params: { x: 0.5, y: 0.5 }, state: "COMPLETE" }]
    });
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");
    const answers = ["Y", "feedback", "make x wider", "Y"];

    await runAutotune(script, {
      trials: 1,
      refineRounds: 1,
      refineTrials: 1,
      refineMode: "ask",
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: false,
      ask: async () => answers.shift() ?? "Y",
      config: await writeTwoParameterSearchSpace(dir)
    });

    const searchSpace = await readFile(path.join(workDir, "search_space.round_1.yaml"), "utf8");
    expect(searchSpace).toContain("high: 2");
    expect(searchSpace).toContain("fixed_parameters");
    expect(searchSpace).toContain("name: y");
    expect(searchSpace).toContain("value: 0.5");
  });

  it("can disable refinement transfer behavior", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-refine-no-transfer-"));
    const binDir = path.join(dir, "bin");
    const workDir = path.join(dir, ".autotune");
    const script = path.join(dir, "train.py");
    await writeFile(script, "print('autotune_metric=1')\n", "utf8");
    await writeFakePythonWithResult(path.join(binDir, "python3"), {
      study_name: "train_autotune",
      direction: "maximize",
      n_trials: 1,
      best_trial: { number: 0, value: 1, params: { x: 0.5, y: 0.5 }, state: "COMPLETE" },
      all_trials: [{ number: 0, value: 1, params: { x: 0.5, y: 0.5 }, state: "COMPLETE" }]
    });
    await writeFakeHeadless(path.join(binDir, "headless"));
    process.env.PATH = `${binDir}${path.delimiter}${originalPath ?? ""}`;
    process.env.AUTOTUNE_HEADLESS_BIN = path.join(binDir, "headless");

    await runAutotune(script, {
      trials: 1,
      refineRounds: 1,
      refineTrials: 1,
      refineMode: "auto",
      refineTransferFixedParams: false,
      refineTransferTrials: false,
      direction: "maximize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir,
      agent: "claude",
      json: true,
      yes: true,
      config: await writeTwoParameterSearchSpace(dir)
    });

    const runner = await readFile(path.join(workDir, "train_optuna.py"), "utf8");
    expect(runner).toContain('\\"fixed_parameters\\":[]');
    expect(runner).toContain('\\"seed_trials\\":[]');
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
    expect(agentCalls.length).toBeGreaterThanOrEqual(2);
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

async function writeFakePythonWithResult(filePath: string, result: unknown): Promise<void> {
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
const result = ${JSON.stringify(result)};
fs.mkdirSync(require('node:path').dirname(output), { recursive: true });
fs.writeFileSync(output, JSON.stringify(result));
fs.writeFileSync(output + '.argv.json', JSON.stringify(args));
console.log(JSON.stringify(result));
`,
    "utf8"
  );
  await chmod(filePath, 0o755);
}

async function writeFakeHeadless(filePath: string, options: { refinedFixed?: boolean; addedParamCurrentValue?: boolean } = {}): Promise<void> {
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
    parameters: ${options.addedParamCurrentValue
      ? "[{ name: 'x', cli_flag: '--x', type: 'float', low: 0.25, high: 0.75 }, { name: 'y', cli_flag: '--y', type: 'float', low: 0, high: 1, current_value: 0.25 }]"
      : "[{ name: 'x', cli_flag: '--x', type: 'float', low: 0.25, high: 0.75 }]"},
    ${options.refinedFixed ? "fixed_parameters: [{ name: 'z', cli_flag: '--z', value: 'keep' }]," : ""}
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

async function writeTwoParameterSearchSpace(dir: string): Promise<string> {
  const filePath = path.join(dir, "two-parameter-space.yaml");
  await writeSearchSpace(filePath, {
    parameters: [
      { name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 },
      { name: "y", cli_flag: "--y", type: "float", low: 0, high: 1 }
    ],
    has_arg_parsing: true,
    needs_wrapper: false,
    direction: "maximize",
    reasoning: "two parameter test space"
  });
  return filePath;
}

async function writeSingleParameterSearchSpace(dir: string): Promise<string> {
  const filePath = path.join(dir, "single-parameter-space.yaml");
  await writeSearchSpace(filePath, {
    parameters: [{ name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 }],
    has_arg_parsing: true,
    needs_wrapper: false,
    direction: "maximize",
    reasoning: "single parameter test space"
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
