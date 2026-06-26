import { access, mkdir, readFile } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { analyzeScript, generateModifiedScript, requestWrapperGeneration, reviseSearchSpace } from "./analyze.js";
import { checkDoctorPrerequisites, checkPrerequisites, type DoctorCheck } from "./check.js";
import { confirmSearchSpace } from "./confirm.js";
import { detectInvocation } from "./detect.js";
import { writeOptunaRunner } from "./generate.js";
import { readResults, renderResults } from "./results.js";
import { runPythonRunner } from "./runner.js";
import { readSearchSpace, writeSearchSpace } from "./search-space.js";
import type { RunOptions, SearchSpace } from "./types.js";

export async function runAutotune(script: string, options: RunOptions): Promise<void> {
  if (!Number.isInteger(options.trials) || options.trials < 1) {
    throw new Error("--trials must be a positive integer");
  }

  const scriptPath = path.resolve(script);
  await access(scriptPath, constants.R_OK);
  const workDir = path.resolve(options.workDir);
  await mkdir(workDir, { recursive: true });
  const invocation = detectInvocation(scriptPath, options.command);

  console.error("Checking prerequisites...");
  const prerequisites = await checkPrerequisites({ invocation, agent: options.agent });
  console.error(`  python3 ${prerequisites.python}`);
  console.error(`  optuna ${prerequisites.optuna}`);
  console.error(`  headless ${prerequisites.headless}`);
  console.error(`  runtime ${prerequisites.runtime}`);

  const searchSpacePath = path.join(workDir, "search_space.yaml");
  const proposed = options.config
    ? await loadConfiguredSearchSpace(options.config)
    : await runAnalysisPhase({ invocation, workDir, agent: options.agent });
  const searchSpace = await prepareSearchSpaceForScript(proposed, options.direction, scriptPath);
  console.error(`Saving confirmed search space: ${searchSpacePath}`);
  const confirmed = await confirmSearchSpace({
    searchSpace,
    filePath: searchSpacePath,
    yes: options.yes,
    ask: options.ask,
    revise: async (current, feedback) => {
      console.error(`Phase 1b: revising search space with headless ${options.agent}...`);
      const revised = await reviseSearchSpace({
        invocation,
        searchSpace: current,
        feedback,
        workDir,
        agent: options.agent
      });
      console.error(`Revision complete: ${revised.parameters.length} parameter(s) proposed.`);
      return prepareSearchSpaceForScript(revised, options.direction, scriptPath);
    }
  });

  const executionInvocation = needsModifiedCopy(confirmed)
    ? await prepareModifiedInvocation({ invocation, searchSpace: confirmed, workDir, agent: options.agent })
    : invocation;
  const runnerPath = path.join(workDir, `${path.basename(scriptPath, path.extname(scriptPath))}_optuna.py`);
  const resultsPath = options.output ? path.resolve(options.output) : path.join(workDir, "results.json");
  console.error(`Phase 2: generating Optuna wrapper with headless ${options.agent}...`);
  console.error("  This can take a minute on first run.");
  await requestWrapperGeneration({
    invocation: executionInvocation,
    searchSpace: confirmed,
    workDir,
    agent: options.agent,
    outputPath: runnerPath
  });
  console.error(`Writing Optuna runner: ${runnerPath}`);
  await writeOptunaRunner({
    invocation: executionInvocation,
    searchSpace: confirmed,
    outputPath: runnerPath,
    resultsPath
  });

  console.error(`Running ${options.trials} Optuna trials...`);
  await runPythonRunner({
    runnerPath,
    trials: options.trials,
    direction: confirmed.direction,
    sampler: options.sampler,
    pruner: options.pruner,
    nJobs: options.nJobs,
    storage: options.storage,
    output: resultsPath
  });
  console.error("Trials complete.");

  const result = await readResults(resultsPath);
  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(renderResults(result));
    console.error(`\nResults saved to ${resultsPath}`);
  }
}

export async function analyzeOnly(script: string, options: {
  agent: string;
  json: boolean;
  output?: string;
  workDir: string;
  command?: string;
}): Promise<void> {
  const scriptPath = path.resolve(script);
  await access(scriptPath, constants.R_OK);
  const workDir = path.resolve(options.workDir);
  const invocation = detectInvocation(scriptPath, options.command);
  const searchSpace = await analyzeScript({ invocation, workDir, agent: options.agent });
  if (options.output) {
    await writeSearchSpace(path.resolve(options.output), searchSpace);
  }
  console.log(options.json ? JSON.stringify(searchSpace, null, 2) : renderSearchSpaceSummary(searchSpace));
}

export async function doctorAutotune(options: {
  script?: string;
  agent: string;
  command?: string;
}): Promise<void> {
  const invocation = options.script
    ? detectInvocation(await resolveReadableScript(options.script), options.command)
    : undefined;
  const checks = await checkDoctorPrerequisites({ invocation, agent: options.agent });
  console.log("autotune doctor");
  for (const check of checks) {
    console.log(formatDoctorCheck(check));
  }
  const failures = checks.filter((check) => check.status === "fail");
  if (failures.length > 0) {
    throw new Error(`${failures.length} prerequisite check failed`);
  }
}

export async function showResults(options: { dir: string; json: boolean; top: number }): Promise<void> {
  const result = await readResults(path.resolve(options.dir));
  console.log(options.json ? JSON.stringify(result, null, 2) : renderResults(result, options.top));
}

export async function resumeStudy(options: {
  workDir: string;
  storage: string;
  trials: number;
  nJobs: number;
  direction: "maximize" | "minimize";
}): Promise<void> {
  const workDir = path.resolve(options.workDir);
  const searchSpace = await readSearchSpace(path.join(workDir, "search_space.yaml"));
  const runnerPath = await findRunner(workDir);
  const resultsPath = path.join(workDir, "results.json");
  await runPythonRunner({
    runnerPath,
    trials: options.trials,
    direction: searchSpace.direction ?? options.direction,
    sampler: "tpe",
    pruner: "none",
    nJobs: options.nJobs,
    storage: options.storage,
    output: resultsPath
  });
  console.log(renderResults(await readResults(resultsPath)));
}

function normalizeDirection(searchSpace: SearchSpace, direction: "maximize" | "minimize"): SearchSpace {
  return { ...searchSpace, direction: direction ?? searchSpace.direction };
}

async function prepareSearchSpaceForScript(
  searchSpace: SearchSpace,
  direction: "maximize" | "minimize",
  scriptPath: string
): Promise<SearchSpace> {
  const normalized = normalizeDirection(searchSpace, direction);
  if (await scriptContainsMetricOutput(scriptPath)) {
    return normalized;
  }
  return { ...normalized, has_metric_output: false };
}

async function scriptContainsMetricOutput(scriptPath: string): Promise<boolean> {
  return (await readFile(scriptPath, "utf8")).includes("autotune_metric");
}

function needsModifiedCopy(searchSpace: SearchSpace): boolean {
  return searchSpace.needs_wrapper || searchSpace.has_metric_output === false;
}

async function prepareModifiedInvocation(input: {
  invocation: ReturnType<typeof detectInvocation>;
  searchSpace: SearchSpace;
  workDir: string;
  agent: string;
}): Promise<ReturnType<typeof detectInvocation>> {
  const extension = path.extname(input.invocation.script);
  const baseName = path.basename(input.invocation.script, extension);
  const modifiedPath = path.join(input.workDir, `${baseName}_modified${extension}`);
  console.error(`Script needs compatibility changes (${modifiedCopyReason(input.searchSpace)}); generating modified copy: ${modifiedPath}`);
  await generateModifiedScript({
    invocation: input.invocation,
    searchSpace: input.searchSpace,
    workDir: input.workDir,
    agent: input.agent,
    outputPath: modifiedPath
  });
  return {
    ...input.invocation,
    script: modifiedPath,
    command: commandForModifiedScript(input.invocation, modifiedPath)
  };
}

function modifiedCopyReason(searchSpace: SearchSpace): string {
  const reasons = [];
  if (searchSpace.needs_wrapper) {
    reasons.push("adding CLI parsing");
  }
  if (searchSpace.has_metric_output === false) {
    reasons.push("adding metric output");
  }
  return reasons.join(", ");
}

function commandForModifiedScript(invocation: ReturnType<typeof detectInvocation>, modifiedPath: string): string[] {
  if (invocation.command.length === 1 && path.resolve(invocation.command[0] ?? "") === path.resolve(invocation.script)) {
    return [modifiedPath];
  }
  return invocation.command;
}

async function loadConfiguredSearchSpace(configPath: string): Promise<SearchSpace> {
  const resolved = path.resolve(configPath);
  console.error(`Loading search space config: ${resolved}`);
  return readSearchSpace(resolved);
}

async function runAnalysisPhase(input: {
  invocation: ReturnType<typeof detectInvocation>;
  workDir: string;
  agent: string;
}): Promise<SearchSpace> {
  console.error(`Phase 1: analyzing ${input.invocation.script} with headless ${input.agent}...`);
  console.error("  This can take a minute on first run.");
  const searchSpace = await analyzeScript(input);
  console.error(`Analysis complete: ${searchSpace.parameters.length} parameter(s) proposed.`);
  return searchSpace;
}

async function resolveReadableScript(script: string): Promise<string> {
  const scriptPath = path.resolve(script);
  await access(scriptPath, constants.R_OK);
  return scriptPath;
}

function formatDoctorCheck(check: DoctorCheck): string {
  const label = check.status === "ok" ? "[ok]" : check.status === "skip" ? "[skip]" : "[fail]";
  return `${label} ${check.name}: ${check.detail}`;
}

function renderSearchSpaceSummary(searchSpace: SearchSpace): string {
  return [
    `Direction: ${searchSpace.direction}`,
    `Arg parsing: ${searchSpace.has_arg_parsing ? "yes" : "no"}`,
    `Metric output: ${searchSpace.has_metric_output === false ? "no" : "yes"}`,
    `Needs wrapper: ${searchSpace.needs_wrapper ? "yes" : "no"}`,
    "Parameters:",
    ...searchSpace.parameters.map((parameter) => `  ${parameter.name} (${parameter.type}) ${parameter.cli_flag}`)
  ].join("\n");
}

async function findRunner(workDir: string): Promise<string> {
  const { readdir } = await import("node:fs/promises");
  const files = await readdir(workDir);
  const runner = files.find((file) => file.endsWith("_optuna.py"));
  if (!runner) {
    throw new Error(`no *_optuna.py runner found in ${workDir}`);
  }
  return path.join(workDir, runner);
}
