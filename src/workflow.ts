import { access, copyFile, mkdir, readFile } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import {
  analyzeScript,
  generateModifiedScript,
  refineSearchSpaceFromTrials,
  requestWrapperGeneration,
  reviseSearchSpace
} from "./analyze.js";
import { checkDoctorPrerequisites, checkPrerequisites, type DoctorCheck } from "./check.js";
import { confirmSearchSpace } from "./confirm.js";
import { detectInvocation, splitCommand } from "./detect.js";
import { writeOptunaRunner } from "./generate.js";
import { runCommand } from "./process.js";
import { readResults, renderResults, type StudyResult, type TrialResult } from "./results.js";
import { runPythonRunner } from "./runner.js";
import { readSearchSpace, writeSearchSpace } from "./search-space.js";
import { styles, writeStatus } from "./terminal.js";
import type { Direction, HeadlessOptions, Pruner, RunOptions, Sampler, SearchSpace } from "./types.js";

const DEFAULT_DIRECTION: Direction = "maximize";
const DEFAULT_SAMPLER: Sampler = "tpe";
const DEFAULT_PRUNER: Pruner = "none";

export async function runAutotune(script: string, options: RunOptions): Promise<void> {
  if (!Number.isInteger(options.trials) || options.trials < 1) {
    throw new Error("--trials must be a positive integer");
  }
  validateRefinementOptions(options);

  const scriptPath = path.resolve(script);
  await access(scriptPath, constants.R_OK);
  const workDir = path.resolve(options.workDir);
  await mkdir(workDir, { recursive: true });
  const commandContext = { scriptPath, workDir };
  if (options.buildCommand) {
    await runBuildCommand(options.buildCommand, commandContext);
  }
  const commandOverride = options.command ? expandCommandTemplateArgs(splitCommand(options.command), commandContext) : undefined;
  const invocation = detectInvocation(scriptPath, commandOverride);
  const headless = pickHeadlessOptions(options);

  writeStatus("Checking prerequisites...");
  const prerequisites = await checkPrerequisites({ invocation, agent: options.agent });
  writeStatus(`python3 ${prerequisites.python}`, "success");
  writeStatus(`optuna ${prerequisites.optuna}`, "success");
  writeStatus(`headless ${prerequisites.headless}`, "success");
  writeStatus(`runtime ${prerequisites.runtime}`, "success");

  const refineRounds = options.refineRounds ?? 0;
  const searchSpacePath = path.join(workDir, "search_space.yaml");
  const finalResultsPath = options.output ? path.resolve(options.output) : path.join(workDir, "results.json");
  const studyName = options.studyName ?? defaultStudyName(scriptPath);
  const proposed = options.config
    ? await loadConfiguredSearchSpace(options.config)
    : await runAnalysisPhase({ invocation, workDir, ...headless });
  const searchSpace = await prepareSearchSpaceForRun(proposed, options, scriptPath);
  const initialSearchSpacePath = searchSpacePathForRound(workDir, 0, refineRounds);
  writeStatus(`Saving confirmed search space: ${styles.dim(initialSearchSpacePath)}`);
  let confirmed = await confirmSearchSpace({
    searchSpace,
    filePath: initialSearchSpacePath,
    yes: options.yes,
    ask: options.ask,
    revise: async (current, feedback) => {
      writeStatus(`Phase 1b: revising search space with ${formatHeadlessLabel(headless)}...`);
      const revised = await reviseSearchSpace({
        invocation,
        searchSpace: current,
        feedback,
        workDir,
        ...headless
      });
      writeStatus(`Revision complete: ${revised.parameters.length} parameter(s) proposed.`, "success");
      return prepareSearchSpaceForRun(revised, options, scriptPath);
    }
  });
  await writeSearchSpace(searchSpacePath, confirmed);

  let result: StudyResult | undefined;
  for (let round = 0; round <= refineRounds; round += 1) {
    if (round > 0) {
      if (!result) {
        throw new Error("cannot refine before a completed trial round");
      }
      confirmed = await refineSearchSpaceForRound({
        invocation,
        current: confirmed,
        previousResult: result,
        round,
        workDir,
        headless,
        options,
        scriptPath,
        searchSpacePath,
        roundSearchSpacePath: searchSpacePathForRound(workDir, round, refineRounds)
      });
    }

    const roundResultsPath = resultsPathForRound(workDir, finalResultsPath, round, refineRounds);
    result = await runSearchRound({
      invocation,
      searchSpace: confirmed,
      workDir,
      headless,
      scriptPath,
      trials: round === 0 ? options.trials : options.refineTrials ?? options.trials,
      options,
      resultsPath: roundResultsPath,
      studyName: studyNameForRound(studyName, round, refineRounds),
      round,
      totalRounds: refineRounds
    });
    if (refineRounds > 0) {
      await mkdir(path.dirname(finalResultsPath), { recursive: true });
      if (path.resolve(roundResultsPath) !== path.resolve(finalResultsPath)) {
        await copyFile(roundResultsPath, finalResultsPath);
      }
    }
  }

  if (!result) {
    throw new Error("no trial results produced");
  }
  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(renderResults(result));
    writeStatus(`Results saved to ${styles.dim(finalResultsPath)}`, "success");
  }
}

export async function analyzeOnly(script: string, options: {
  agent: string;
  model?: string;
  reasoningEffort?: RunOptions["reasoningEffort"];
  json: boolean;
  output?: string;
  workDir: string;
  command?: string;
}): Promise<void> {
  const scriptPath = path.resolve(script);
  await access(scriptPath, constants.R_OK);
  const workDir = path.resolve(options.workDir);
  const invocation = detectInvocation(scriptPath, options.command);
  const searchSpace = withEffectiveOptunaSettings(await analyzeScript({ invocation, workDir, ...pickHeadlessOptions(options) }), {});
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
  studyName?: string;
}): Promise<void> {
  const workDir = path.resolve(options.workDir);
  const searchSpace = await readSearchSpace(path.join(workDir, "search_space.yaml"));
  const runnerPath = await findRunner(workDir);
  const resultsPath = path.join(workDir, "results.json");
  await runPythonRunner({
    runnerPath,
    trials: options.trials,
    direction: searchSpace.direction ?? options.direction,
    sampler: searchSpace.optuna?.sampler ?? DEFAULT_SAMPLER,
      pruner: searchSpace.optuna?.pruner ?? DEFAULT_PRUNER,
      nJobs: options.nJobs,
      storage: options.storage,
      studyName: options.studyName,
      output: resultsPath
  });
  console.log(renderResults(await readResults(resultsPath)));
}

async function prepareSearchSpaceForRun(
  searchSpace: SearchSpace,
  options: Pick<RunOptions, "direction" | "sampler" | "pruner">,
  scriptPath: string
): Promise<SearchSpace> {
  const normalized = withEffectiveOptunaSettings(searchSpace, options);
  if (await scriptContainsMetricOutput(scriptPath)) {
    return normalized;
  }
  return { ...normalized, has_metric_output: false };
}

async function runSearchRound(input: {
  invocation: ReturnType<typeof detectInvocation>;
  searchSpace: SearchSpace;
  workDir: string;
  headless: HeadlessOptions;
  scriptPath: string;
  trials: number;
  options: RunOptions;
  resultsPath: string;
  studyName: string;
  round: number;
  totalRounds: number;
}): Promise<StudyResult> {
  const executionInvocation = needsModifiedCopy(input.searchSpace)
    ? await prepareModifiedInvocation({
        invocation: input.invocation,
        searchSpace: input.searchSpace,
        workDir: input.workDir,
        ...input.headless
      })
    : input.invocation;
  const runnerPath = path.join(input.workDir, `${path.basename(input.scriptPath, path.extname(input.scriptPath))}_optuna.py`);
  writeStatus(`Phase 2: generating Optuna wrapper with ${formatHeadlessLabel(input.headless)}...`);
  writeStatus("This can take a minute on first run.");
  await requestWrapperGeneration({
    invocation: executionInvocation,
    searchSpace: input.searchSpace,
    workDir: input.workDir,
    ...input.headless,
    outputPath: runnerPath
  });
  writeStatus(`Writing Optuna runner: ${styles.dim(runnerPath)}`);
  await writeOptunaRunner({
    invocation: executionInvocation,
    searchSpace: input.searchSpace,
    outputPath: runnerPath,
    resultsPath: input.resultsPath,
    studyName: input.studyName
  });

  const label = input.totalRounds > 0 ? `Round ${input.round + 1}/${input.totalRounds + 1}: ` : "";
  writeStatus(`${label}Running ${input.trials} Optuna trials...`);
  await runPythonRunner({
    runnerPath,
    trials: input.trials,
    direction: input.searchSpace.direction,
    sampler: input.searchSpace.optuna?.sampler ?? DEFAULT_SAMPLER,
    pruner: input.searchSpace.optuna?.pruner ?? DEFAULT_PRUNER,
    nJobs: input.options.nJobs,
    storage: input.options.storage,
    studyName: input.studyName,
    output: input.resultsPath
  });
  writeStatus(`${label}Trials complete.`, "success");
  return readResults(input.resultsPath);
}

async function refineSearchSpaceForRound(input: {
  invocation: ReturnType<typeof detectInvocation>;
  current: SearchSpace;
  previousResult: StudyResult;
  round: number;
  workDir: string;
  headless: HeadlessOptions;
  options: RunOptions;
  scriptPath: string;
  searchSpacePath: string;
  roundSearchSpacePath: string;
}): Promise<SearchSpace> {
  writeStatus(`Phase 3: refining search space for round ${input.round + 1} with ${formatHeadlessLabel(input.headless)}...`);
  const refined = await refineSearchSpaceFromTrials({
    invocation: input.invocation,
    searchSpace: input.current,
    trialSummary: summarizeTrialResults(input.previousResult, input.current),
    round: input.round,
    workDir: input.workDir,
    ...input.headless
  });
  writeStatus(`Refinement complete: ${refined.parameters.length} parameter(s) proposed.`, "success");
  const prepared = await prepareSearchSpaceForRun(refined, input.options, input.scriptPath);
  const confirmed = await confirmSearchSpace({
    searchSpace: prepared,
    filePath: input.roundSearchSpacePath,
    yes: input.options.yes || input.options.refineMode === "auto",
    ask: input.options.ask,
    revise: async (current, feedback) => {
      writeStatus(`Phase 3b: revising refined space with ${formatHeadlessLabel(input.headless)}...`);
      const revised = await reviseSearchSpace({
        invocation: input.invocation,
        searchSpace: current,
        feedback,
        workDir: input.workDir,
        ...input.headless
      });
      writeStatus(`Revision complete: ${revised.parameters.length} parameter(s) proposed.`, "success");
      return prepareSearchSpaceForRun(revised, input.options, input.scriptPath);
    }
  });
  await writeSearchSpace(input.searchSpacePath, confirmed);
  return confirmed;
}

function summarizeTrialResults(result: StudyResult, searchSpace: SearchSpace) {
  const topTrials = completedTrials(result).slice(0, 5);
  return {
    direction: result.direction,
    n_trials: result.n_trials,
    best_trial: result.best_trial,
    top_trials: topTrials,
    parameter_ranges: searchSpace.parameters.map((parameter) => ({
      name: parameter.name,
      type: parameter.type,
      cli_flag: parameter.cli_flag,
      low: parameter.low,
      high: parameter.high,
      choices: parameter.choices,
      best_value: result.best_trial?.params[parameter.name],
      sampled_values: uniqueSampledValues(result.all_trials, parameter.name).slice(0, 10)
    }))
  };
}

function completedTrials(result: StudyResult): TrialResult[] {
  return result.all_trials
    .filter((trial) => typeof trial.value === "number")
    .sort((left, right) =>
      result.direction === "maximize"
        ? Number(right.value) - Number(left.value)
        : Number(left.value) - Number(right.value)
    );
}

function uniqueSampledValues(trials: TrialResult[], parameterName: string): unknown[] {
  const seen = new Set<string>();
  const values = [];
  for (const trial of trials) {
    if (!(parameterName in trial.params)) {
      continue;
    }
    const value = trial.params[parameterName];
    const key = JSON.stringify(value);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    values.push(value);
  }
  return values;
}

function validateRefinementOptions(options: RunOptions): void {
  const refineRounds = options.refineRounds ?? 0;
  if (!Number.isInteger(refineRounds) || refineRounds < 0) {
    throw new Error("--refine-rounds must be a non-negative integer");
  }
  if (options.refineTrials !== undefined && (!Number.isInteger(options.refineTrials) || options.refineTrials < 1)) {
    throw new Error("--refine-trials must be a positive integer");
  }
}

function searchSpacePathForRound(workDir: string, round: number, refineRounds: number): string {
  return refineRounds > 0 ? path.join(workDir, `search_space.round_${round}.yaml`) : path.join(workDir, "search_space.yaml");
}

function resultsPathForRound(workDir: string, finalResultsPath: string, round: number, refineRounds: number): string {
  return refineRounds > 0 ? path.join(workDir, `results.round_${round}.json`) : finalResultsPath;
}

async function runBuildCommand(template: string, context: CommandTemplateContext): Promise<void> {
  const command = expandCommandTemplateArgs(splitCommand(template), context);
  const [executable, ...args] = command;
  if (!executable) {
    throw new Error("build command cannot be empty");
  }
  writeStatus(`Building runtime: ${styles.dim(command.join(" "))}`);
  await runCommand(executable, args);
  writeStatus("Build complete.", "success");
}

interface CommandTemplateContext {
  scriptPath: string;
  workDir: string;
}

function expandCommandTemplateArgs(args: string[], context: CommandTemplateContext): string[] {
  return args.map((arg) => arg.replaceAll("{script}", context.scriptPath).replaceAll("{work-dir}", context.workDir));
}

function withEffectiveOptunaSettings(
  searchSpace: SearchSpace,
  options: Pick<RunOptions, "direction" | "sampler" | "pruner">
): SearchSpace {
  return {
    ...searchSpace,
    direction: options.direction ?? searchSpace.direction ?? DEFAULT_DIRECTION,
    optuna: {
      ...searchSpace.optuna,
      sampler: options.sampler ?? searchSpace.optuna?.sampler ?? DEFAULT_SAMPLER,
      pruner: options.pruner ?? searchSpace.optuna?.pruner ?? DEFAULT_PRUNER
    }
  };
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
} & HeadlessOptions): Promise<ReturnType<typeof detectInvocation>> {
  const extension = path.extname(input.invocation.script);
  const baseName = path.basename(input.invocation.script, extension);
  const modifiedPath = path.join(input.workDir, `${baseName}_modified${extension}`);
  writeStatus(
    `Script needs compatibility changes (${modifiedCopyReason(input.searchSpace)}); generating modified copy: ${styles.dim(modifiedPath)}`,
    "warning"
  );
  await generateModifiedScript({
    invocation: input.invocation,
    searchSpace: input.searchSpace,
    workDir: input.workDir,
    agent: input.agent,
    model: input.model,
    reasoningEffort: input.reasoningEffort,
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
  const replaced = invocation.command.map((arg) =>
    path.resolve(arg) === path.resolve(invocation.script) ? modifiedPath : arg
  );
  if (replaced.some((arg, index) => arg !== invocation.command[index])) {
    return replaced;
  }
  if (invocation.command.length === 1 && path.resolve(invocation.command[0] ?? "") === path.resolve(invocation.script)) {
    return [modifiedPath];
  }
  return invocation.command;
}

async function loadConfiguredSearchSpace(configPath: string): Promise<SearchSpace> {
  const resolved = path.resolve(configPath);
  writeStatus(`Loading search space config: ${styles.dim(resolved)}`);
  return readSearchSpace(resolved);
}

async function runAnalysisPhase(input: {
  invocation: ReturnType<typeof detectInvocation>;
  workDir: string;
} & HeadlessOptions): Promise<SearchSpace> {
  writeStatus(`Phase 1: analyzing ${styles.dim(input.invocation.script)} with ${formatHeadlessLabel(input)}...`);
  writeStatus("This can take a minute on first run.");
  const searchSpace = await analyzeScript(input);
  writeStatus(`Analysis complete: ${searchSpace.parameters.length} parameter(s) proposed.`, "success");
  return searchSpace;
}

function pickHeadlessOptions(options: HeadlessOptions): HeadlessOptions {
  return {
    agent: options.agent,
    model: options.model,
    reasoningEffort: options.reasoningEffort
  };
}

function formatHeadlessLabel(options: HeadlessOptions): string {
  const details = [options.model, options.reasoningEffort ? `effort=${options.reasoningEffort}` : undefined]
    .filter(Boolean)
    .join(", ");
  return details ? `headless ${options.agent} (${details})` : `headless ${options.agent}`;
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
    `Sampler: ${searchSpace.optuna?.sampler ?? DEFAULT_SAMPLER}`,
    `Pruner: ${searchSpace.optuna?.pruner ?? DEFAULT_PRUNER}`,
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

function defaultStudyName(scriptPath: string): string {
  return `${path.basename(scriptPath, path.extname(scriptPath))}_autotune`;
}

function studyNameForRound(studyName: string, round: number, refineRounds: number): string {
  return refineRounds > 0 ? `${studyName}_round_${round}` : studyName;
}
