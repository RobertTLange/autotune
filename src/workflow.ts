import { access, mkdir } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { analyzeScript, requestWrapperGeneration } from "./analyze.js";
import { checkPrerequisites } from "./check.js";
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
    ? await readSearchSpace(path.resolve(options.config))
    : await analyzeScript({ invocation, workDir, agent: options.agent });
  const searchSpace = normalizeDirection(proposed, options.direction);
  const confirmed = await confirmSearchSpace({ searchSpace, filePath: searchSpacePath, yes: options.yes });

  const runnerPath = path.join(workDir, `${path.basename(scriptPath, path.extname(scriptPath))}_optuna.py`);
  const resultsPath = options.output ? path.resolve(options.output) : path.join(workDir, "results.json");
  await requestWrapperGeneration({
    invocation,
    searchSpace: confirmed,
    workDir,
    agent: options.agent,
    outputPath: runnerPath
  });
  await writeOptunaRunner({
    invocation,
    searchSpace: confirmed,
    outputPath: runnerPath,
    resultsPath
  });

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

function renderSearchSpaceSummary(searchSpace: SearchSpace): string {
  return [
    `Direction: ${searchSpace.direction}`,
    `Arg parsing: ${searchSpace.has_arg_parsing ? "yes" : "no"}`,
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
