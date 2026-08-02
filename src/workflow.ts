import { access, copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import {
  analyzeScript,
  generateModifiedScript,
  refineSearchSpaceFromTrials,
  reviseSearchSpace
} from "./analyze.js";
import { resolveRunArtifactLayout, resolveRunDirectory, writeLatestRun } from "./artifacts.js";
import { checkDoctorPrerequisites, checkPrerequisites, type DoctorCheck } from "./check.js";
import { confirmSearchSpace } from "./confirm.js";
import { detectInvocation, splitCommand } from "./detect.js";
import { writeOptunaRunner, type SeedTrial } from "./generate.js";
import { runCommand } from "./process.js";
import { readResults, renderResults, type StudyResult, type TrialResult } from "./results.js";
import { runPythonRunner } from "./runner.js";
import {
  correctSearchSpaceParameterLimit,
  readSearchSpace,
  validateSearchSpaceParameterLimit,
  writeSearchSpace
} from "./search-space.js";
import { styles, writeStatus } from "./terminal.js";
import type { CentaurConfig, Direction, FixedParameter, HeadlessOptions, Pruner, RunOptions, Sampler, SearchBudget, SearchParameter, SearchSpace } from "./types.js";

const DEFAULT_DIRECTION: Direction = "maximize";
const DEFAULT_SAMPLER: Sampler = "tpe";
const DEFAULT_PRUNER: Pruner = "none";
const DEFAULT_CENTAUR_CONFIG: CentaurConfig = { llm_probability: 0.3, warmup_trials: 10, seed: 0 };

export async function runAutotune(script: string, options: RunOptions): Promise<StudyResult> {
  if (!Number.isInteger(options.trials) || options.trials < 1) {
    throw new Error("--trials must be a positive integer");
  }
  validateMaxParametersOption(options.maxParameters);
  validateRefinementOptions(options);
  validateSamplerSeedOption(options);

  const scriptPath = path.resolve(script);
  await access(scriptPath, constants.R_OK);
  const configuredSearchSpace = options.config
    ? await loadConfiguredSearchSpace(options.config)
    : undefined;
  if (configuredSearchSpace) {
    validateSearchSpaceParameterLimit(configuredSearchSpace, options.maxParameters);
  }
  const effectiveSampler = options.sampler ?? configuredSearchSpace?.optuna?.sampler;
  if (effectiveSampler === "centaur" && options.maxParameters !== undefined && options.maxParameters < 2) {
    throw new Error("--max-parameters must be at least 2 with Centaur");
  }

  const artifactLayout = resolveRunArtifactLayout(scriptPath, options.workDir);
  const workDir = artifactLayout.workDir;
  await mkdir(workDir, { recursive: true });
  const commandContext = { scriptPath, workDir };
  if (options.buildCommand) {
    await runBuildCommand(options.buildCommand, commandContext, options.silent);
  }
  const commandOverride = options.command ? expandCommandTemplateArgs(splitCommand(options.command), commandContext) : undefined;
  const invocation = detectInvocation(scriptPath, commandOverride);
  const headless = pickHeadlessOptions(options);
  const refineRounds = options.refineRounds ?? 0;
  const initialBudget = searchBudgetForOptions(options, refineRounds);
  const searchSpacePath = path.join(workDir, "search_space.yaml");
  const finalResultsPath = path.join(workDir, "results.json");
  const outputResultsPath = options.output ? path.resolve(options.output) : undefined;
  const studyName = options.studyName ?? defaultStudyName(scriptPath);
  const roundManifests: RoundManifest[] = [];
  const preparedConfiguredSearchSpace = configuredSearchSpace
    ? await prepareSearchSpaceForRun(configuredSearchSpace, options, scriptPath)
    : undefined;
  if (effectiveSampler === "centaur" && options.samplerSeed !== undefined) {
    throw new Error("Centaur uses --centaur-seed instead of --sampler-seed");
  }
  const centaurAuthorized = effectiveSampler === "centaur";
  validateCentaurOptions(options, effectiveSampler);

  writeStatus("Checking prerequisites...");
  const prerequisites = await checkPrerequisites({
    invocation,
    agent: options.agent,
    model: options.model,
    centaur: effectiveSampler === "centaur",
    skipHeadless: shouldSkipHeadlessPrerequisite({ searchSpace: preparedConfiguredSearchSpace, options, refineRounds })
  });
  writeStatus(
    `python ${prerequisites.python}${prerequisites.managedPython ? " (managed control environment)" : ""}`,
    "success"
  );
  writeStatus(`optuna ${prerequisites.optuna}`, "success");
  if (prerequisites.cmaes) {
    writeStatus(`cmaes ${prerequisites.cmaes}`, "success");
  }
  writeStatus(`headless ${prerequisites.headless}`, "success");
  writeStatus(`runtime ${prerequisites.runtime}`, "success");

  const searchSpace = preparedConfiguredSearchSpace ?? await prepareSearchSpaceForRun(
    await correctAgentParameterLimit({
      candidate: await runAnalysisPhase({
        invocation,
        workDir,
        budget: initialBudget,
        agentGuidance: options.agentGuidance,
        maxParameters: options.maxParameters,
        ...headless
      }),
      invocation,
      workDir,
      budget: initialBudget,
      agentGuidance: options.agentGuidance,
      maxParameters: options.maxParameters,
      ...headless
    }),
    options,
    scriptPath
  );
  const initialSearchSpacePath = searchSpacePathForRound(workDir, 0, refineRounds);
  writeStatus(`Saving confirmed search space: ${styles.dim(initialSearchSpacePath)}`);
  let confirmed = await confirmSearchSpace({
    searchSpace,
    filePath: initialSearchSpacePath,
    yes: options.yes,
    ask: options.ask,
    validate: (candidate) => validateSearchSpaceParameterLimit(candidate, options.maxParameters),
    revise: async (current, feedback) => {
      writeStatus(`Phase 1b: revising search space with ${formatHeadlessLabel(headless)}...`);
      const revised = await reviseSearchSpace({
        invocation,
        searchSpace: current,
        feedback,
        budget: initialBudget,
        agentGuidance: options.agentGuidance,
        maxParameters: options.maxParameters,
        workDir,
        ...headless
      });
      writeStatus(`Revision complete: ${revised.parameters.length} parameter(s) proposed.`, "success");
      const corrected = await correctAgentParameterLimit({
        candidate: revised,
        invocation,
        workDir,
        budget: initialBudget,
        agentGuidance: options.agentGuidance,
        maxParameters: options.maxParameters,
        ...headless
      });
      return prepareSearchSpaceForRun(corrected, options, scriptPath);
    }
  });
  if (confirmed.optuna?.sampler === "centaur" && !centaurAuthorized) {
    throw new Error("Centaur requires explicit --sampler centaur or a Centaur config");
  }
  validateCentaurSearchSpace(confirmed, options.maxParameters);
  validateSamplerSeededSearchSpace(confirmed, options.nJobs);
  await writeSearchSpace(searchSpacePath, confirmed);

  let result: StudyResult | undefined;
  let remainingTrialTimeBudgetSeconds = options.timeBudgetSeconds;
  for (let round = 0; round <= refineRounds; round += 1) {
    if (remainingTrialTimeBudgetSeconds !== undefined && remainingTrialTimeBudgetSeconds <= 0) {
      writeStatus("Trial time budget exhausted; skipping remaining refinement rounds.");
      break;
    }
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
    const seedTrials = round > 0 && result && options.refineTransferTrials !== false
      ? seedTrialsForSearchSpace(result, confirmed, round - 1)
      : [];
    result = await runSearchRound({
      invocation,
      searchSpace: confirmed,
      workDir,
      headless,
      scriptPath,
      trials: round === 0 ? options.trials : options.refineTrials ?? options.trials,
      options,
      timeBudgetSeconds: remainingTrialTimeBudgetSeconds,
      resultsPath: roundResultsPath,
      studyName: studyNameForRound(studyName, round, refineRounds),
      seedTrials,
      round,
      totalRounds: refineRounds,
      controllerPython: prerequisites.pythonExecutable
    });
    remainingTrialTimeBudgetSeconds = remainingTrialTimeBudget(remainingTrialTimeBudgetSeconds, result);
    const manifest = buildRoundManifest({
      workDir,
      round,
      totalRounds: refineRounds,
      trials: round === 0 ? options.trials : options.refineTrials ?? options.trials,
      searchSpacePath: searchSpacePathForRound(workDir, round, refineRounds),
      resultsPath: roundResultsPath,
      runnerPath: runnerPathForRound(workDir, scriptPath, round, refineRounds),
      studyName: studyNameForRound(studyName, round, refineRounds),
      seedCount: seedTrials.length,
      searchSpace: confirmed,
      headless,
      options
    });
    roundManifests.push(manifest);
    await writeRoundManifest(workDir, roundManifests);
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
  if (outputResultsPath && path.resolve(outputResultsPath) !== path.resolve(finalResultsPath)) {
    await mkdir(path.dirname(outputResultsPath), { recursive: true });
    await copyFile(finalResultsPath, outputResultsPath);
  }
  if (!options.silent) {
    if (options.json) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log(renderResults(result));
      writeStatus(`Results saved to ${styles.dim(outputResultsPath ?? finalResultsPath)}`, "success");
    }
  }
  await writeLatestRun(artifactLayout);
  return result;
}

export async function analyzeOnly(script: string, options: {
  agent: string;
  model?: string;
  reasoningEffort?: RunOptions["reasoningEffort"];
  json: boolean;
  output?: string;
  workDir: string;
  command?: string;
  agentGuidance?: string;
  maxParameters?: number;
  silent?: boolean;
}): Promise<SearchSpace> {
  validateMaxParametersOption(options.maxParameters);
  const scriptPath = path.resolve(script);
  await access(scriptPath, constants.R_OK);
  const workDir = path.resolve(options.workDir);
  const invocation = detectInvocation(scriptPath, options.command);
  const headless = pickHeadlessOptions(options);
  const analyzed = await analyzeScript({
    invocation,
    workDir,
    agentGuidance: options.agentGuidance,
    maxParameters: options.maxParameters,
    ...headless
  });
  const searchSpace = withEffectiveOptunaSettings(
    await correctAgentParameterLimit({
      candidate: analyzed,
      invocation,
      workDir,
      agentGuidance: options.agentGuidance,
      maxParameters: options.maxParameters,
      ...headless
    }),
    {}
  );
  validateSearchSpaceParameterLimit(searchSpace, options.maxParameters);
  validateCentaurSearchSpace(searchSpace, options.maxParameters);
  if (options.output) {
    await writeSearchSpace(path.resolve(options.output), searchSpace);
  }
  if (!options.silent) {
    console.log(options.json ? JSON.stringify(searchSpace, null, 2) : renderSearchSpaceSummary(searchSpace));
  }
  return searchSpace;
}

export async function doctorAutotune(options: {
  script?: string;
  agent: string;
  model?: string;
  command?: string;
  silent?: boolean;
}): Promise<DoctorCheck[]> {
  const invocation = options.script
    ? detectInvocation(await resolveReadableScript(options.script), options.command)
    : undefined;
  const checks = await checkDoctorPrerequisites({ invocation, agent: options.agent, model: options.model });
  if (!options.silent) {
    console.log("autotune doctor");
    for (const check of checks) {
      console.log(formatDoctorCheck(check));
    }
  }
  const failures = checks.filter((check) => check.status === "fail");
  if (!options.silent && failures.length > 0) {
    throw new Error(`${failures.length} prerequisite check failed`);
  }
  return checks;
}

export async function showResults(options: { dir: string; json: boolean; top: number; silent?: boolean }): Promise<StudyResult> {
  const result = await readResults(path.resolve(options.dir));
  if (!options.silent) {
    console.log(options.json ? JSON.stringify(result, null, 2) : renderResults(result, options.top));
  }
  return result;
}

export async function resumeStudy(options: {
  workDir: string;
  storage: string;
  trials: number;
  nJobs: number;
  direction: "maximize" | "minimize";
  studyName?: string;
  silent?: boolean;
}): Promise<StudyResult> {
  const workDir = await resolveRunDirectory(options.workDir);
  const manifest = await readLatestRoundManifest(workDir);
  const searchSpacePath = manifest?.search_space_path
    ? path.join(workDir, manifest.search_space_path)
    : path.join(workDir, "search_space.yaml");
  const searchSpace = await readSearchSpace(searchSpacePath);
  if (searchSpace.optuna?.sampler === "centaur" && options.nJobs !== 1) {
    throw new Error("Centaur requires --n-jobs 1");
  }
  validateSamplerSeededSearchSpace(searchSpace, options.nJobs);
  const runnerPath = manifest?.runner_path
    ? path.join(workDir, manifest.runner_path)
    : await findRunner(workDir);
  const resultsPath = path.join(workDir, "results.json");
  await runPythonRunner({
    runnerPath,
    trials: options.trials,
    direction: searchSpace.direction ?? options.direction,
    sampler: searchSpace.optuna?.sampler ?? DEFAULT_SAMPLER,
      pruner: searchSpace.optuna?.pruner ?? DEFAULT_PRUNER,
      nJobs: options.nJobs,
      storage: options.storage,
      studyName: options.studyName ?? manifest?.study_name,
      output: resultsPath
  });
  const result = await readResults(resultsPath);
  if (!options.silent) {
    console.log(renderResults(result));
  }
  return result;
}

async function prepareSearchSpaceForRun(
  searchSpace: SearchSpace,
  options: Pick<RunOptions, "direction" | "sampler" | "samplerSeed" | "pruner" | "nJobs" | "maxParameters">,
  scriptPath: string
): Promise<SearchSpace> {
  validateSearchSpaceParameterLimit(searchSpace, options.maxParameters);
  const normalized = withEffectiveOptunaSettings(searchSpace, options);
  validateSamplerSeededSearchSpace(normalized, options.nJobs);
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
  timeBudgetSeconds?: number;
  resultsPath: string;
  studyName: string;
  seedTrials: SeedTrial[];
  round: number;
  totalRounds: number;
  controllerPython: string;
}): Promise<StudyResult> {
  const executionInvocation = needsModifiedCopy(input.searchSpace)
    ? await prepareModifiedInvocation({
        invocation: input.invocation,
        searchSpace: input.searchSpace,
        workDir: input.workDir,
        round: input.round,
        totalRounds: input.totalRounds,
        ...input.headless
      })
    : input.invocation;
  const runnerPath = runnerPathForRound(input.workDir, input.scriptPath, input.round, input.totalRounds);
  writeStatus(`Writing Optuna runner: ${styles.dim(runnerPath)}`);
  await writeOptunaRunner({
    invocation: executionInvocation,
    searchSpace: input.searchSpace,
    outputPath: runnerPath,
    resultsPath: input.resultsPath,
    studyName: input.studyName,
    timeoutSeconds: input.options.timeoutSeconds,
    timeBudgetSeconds: input.timeBudgetSeconds,
    seedTrials: input.seedTrials,
    headless: input.headless
  });
  await copyLatestRunnerAlias(runnerPath, input.workDir, input.scriptPath, input.totalRounds);

  const label = input.totalRounds > 0 ? `Round ${input.round + 1}/${input.totalRounds + 1}: ` : "";
  writeStatus(`${label}Running ${input.trials} Optuna trials...`);
  await runPythonRunner({
    python: input.controllerPython,
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
    budget: searchBudgetForOptions(input.options, input.options.refineRounds ?? 0, input.round),
    agentGuidance: input.options.agentGuidance,
    maxParameters: input.options.maxParameters,
    workDir: input.workDir,
    ...input.headless
  });
  writeStatus(`Refinement complete: ${refined.parameters.length} parameter(s) proposed.`, "success");
  const correctedRefined = await correctAgentParameterLimit({
    candidate: refined,
    invocation: input.invocation,
    workDir: input.workDir,
    budget: searchBudgetForOptions(input.options, input.options.refineRounds ?? 0, input.round),
    agentGuidance: input.options.agentGuidance,
    maxParameters: input.options.maxParameters,
    ...input.headless
  });
  const prepared = await prepareRefinedSearchSpaceForRun({
    candidate: correctedRefined,
    previous: input.current,
    result: input.previousResult,
    options: input.options,
    scriptPath: input.scriptPath
  });
  const confirmed = await confirmSearchSpace({
    searchSpace: prepared,
    filePath: input.roundSearchSpacePath,
    yes: input.options.yes || input.options.refineMode === "auto",
    ask: input.options.ask,
    validate: (candidate) => validateSearchSpaceParameterLimit(candidate, input.options.maxParameters),
    revise: async (current, feedback) => {
      writeStatus(`Phase 3b: revising refined space with ${formatHeadlessLabel(input.headless)}...`);
      const revised = await reviseSearchSpace({
        invocation: input.invocation,
        searchSpace: current,
        feedback,
        budget: searchBudgetForOptions(input.options, input.options.refineRounds ?? 0, input.round),
        agentGuidance: input.options.agentGuidance,
        maxParameters: input.options.maxParameters,
        workDir: input.workDir,
        ...input.headless
      });
      writeStatus(`Revision complete: ${revised.parameters.length} parameter(s) proposed.`, "success");
      const corrected = await correctAgentParameterLimit({
        candidate: revised,
        invocation: input.invocation,
        workDir: input.workDir,
        budget: searchBudgetForOptions(input.options, input.options.refineRounds ?? 0, input.round),
        agentGuidance: input.options.agentGuidance,
        maxParameters: input.options.maxParameters,
        ...input.headless
      });
      return prepareRefinedSearchSpaceForRun({
        candidate: corrected,
        previous: input.current,
        result: input.previousResult,
        options: input.options,
        scriptPath: input.scriptPath
      });
    }
  });
  validateSamplerSeededSearchSpace(confirmed, input.options.nJobs);
  validateCentaurSearchSpace(confirmed, input.options.maxParameters);
  if (confirmed.optuna?.sampler === "centaur") {
    throw new Error("Centaur requires explicit --sampler centaur or a Centaur config");
  }
  await writeSearchSpace(input.searchSpacePath, confirmed);
  return confirmed;
}

async function prepareRefinedSearchSpaceForRun(input: {
  candidate: SearchSpace;
  previous: SearchSpace;
  result: StudyResult;
  options: RunOptions;
  scriptPath: string;
}): Promise<SearchSpace> {
  const prepared = await prepareSearchSpaceForRun(input.candidate, input.options, input.scriptPath);
  const seeded = preserveConfiguredSamplerSeed(prepared, input.previous);
  if (input.options.refineTransferFixedParams === false) {
    return seeded;
  }
  return transferDroppedParameters({
    previous: input.previous,
    refined: seeded,
    result: input.result
  });
}

function summarizeTrialResults(result: StudyResult, searchSpace: SearchSpace) {
  const topTrials = completedTrials(result).slice(0, 5);
  const completed = completedTrials(result);
  return {
    direction: result.direction,
    n_trials: result.n_trials,
    state_counts: countTrialStates(result.all_trials),
    transfer_counts: countTransferredTrials(result.all_trials),
    best_trial: result.best_trial,
    top_trials: topTrials,
    bottom_trials: completed.slice(-5).reverse(),
    parameter_ranges: searchSpace.parameters.map((parameter) => ({
      name: parameter.name,
      type: parameter.type,
      cli_flag: parameter.cli_flag,
      low: parameter.low,
      high: parameter.high,
      choices: parameter.choices,
      best_value: result.best_trial?.params[parameter.name],
      sampled_values: uniqueSampledValues(result.all_trials, parameter.name).slice(0, 10),
      value_samples: sampledParameterValues(result.all_trials, parameter.name).slice(0, 20),
      performance_samples: performanceSamples(result, parameter.name).slice(0, 20),
      boundary_hits: boundaryHits(result.all_trials, parameter)
    }))
  };
}

function countTrialStates(trials: TrialResult[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const trial of trials) {
    const state = trial.state ?? "UNKNOWN";
    counts[state] = (counts[state] ?? 0) + 1;
  }
  return counts;
}

function countTransferredTrials(trials: TrialResult[]): { transferred: number; real: number } {
  let transferred = 0;
  let real = 0;
  for (const trial of trials) {
    if (trial.user_attrs?.autotune_transfer === true) {
      transferred += 1;
    } else {
      real += 1;
    }
  }
  return { transferred, real };
}

function remainingTrialTimeBudget(budgetSeconds: number | undefined, result: StudyResult): number | undefined {
  if (budgetSeconds === undefined) {
    return undefined;
  }
  const usedSeconds = result.all_trials.reduce((total, trial) => total + trialDurationSeconds(trial), 0);
  return Math.max(0, budgetSeconds - usedSeconds);
}

function trialDurationSeconds(trial: TrialResult): number {
  const duration = trial.user_attrs?.autotune_duration_seconds;
  return typeof duration === "number" && Number.isFinite(duration) && duration > 0 ? duration : 0;
}

function sampledParameterValues(trials: TrialResult[], parameterName: string): Array<{ trial: number; state?: string; value: unknown }> {
  return trials
    .filter((trial) => parameterName in trial.params)
    .map((trial) => ({ trial: trial.number, state: trial.state, value: trial.params[parameterName] }));
}

function performanceSamples(result: StudyResult, parameterName: string): Array<{ trial: number; objective: number; value: unknown; transferred: boolean }> {
  return result.all_trials
    .filter((trial) => typeof trial.value === "number" && parameterName in trial.params)
    .sort((left, right) =>
      result.direction === "maximize"
        ? Number(right.value) - Number(left.value)
        : Number(left.value) - Number(right.value)
    )
    .map((trial) => ({
      trial: trial.number,
      objective: Number(trial.value),
      value: trial.params[parameterName],
      transferred: trial.user_attrs?.autotune_transfer === true
    }));
}

function boundaryHits(trials: TrialResult[], parameter: SearchParameter): { low: number; high: number } | undefined {
  if (typeof parameter.low !== "number" || typeof parameter.high !== "number") {
    return undefined;
  }
  let low = 0;
  let high = 0;
  for (const trial of trials) {
    const value = trial.params[parameter.name];
    if (typeof value !== "number") {
      continue;
    }
    if (value === parameter.low) {
      low += 1;
    }
    if (value === parameter.high) {
      high += 1;
    }
  }
  return { low, high };
}

function transferDroppedParameters(input: {
  previous: SearchSpace;
  refined: SearchSpace;
  result: StudyResult;
}): SearchSpace {
  const bestParams = input.result.best_trial?.params;
  if (!bestParams) {
    return input.refined;
  }
  const activeNames = new Set(input.refined.parameters.map((parameter) => parameter.name));
  const fixed = new Map<string, FixedParameter>(
    (input.refined.fixed_parameters ?? []).map((parameter) => [parameter.name, parameter])
  );
  for (const parameter of input.previous.fixed_parameters ?? []) {
    if (!activeNames.has(parameter.name)) {
      fixed.set(parameter.name, parameter);
    }
  }
  for (const parameter of input.previous.parameters) {
    if (activeNames.has(parameter.name)) {
      continue;
    }
    const value = bestParams[parameter.name];
    if (isPrimitive(value)) {
      fixed.set(parameter.name, { name: parameter.name, cli_flag: parameter.cli_flag, value });
    }
  }
  return { ...input.refined, fixed_parameters: [...fixed.values()] };
}

function seedTrialsForSearchSpace(result: StudyResult, searchSpace: SearchSpace, sourceRound: number): SeedTrial[] {
  return completedTrials(result)
    .map((trial) => seedTrialForSearchSpace(trial, searchSpace, sourceRound))
    .filter((trial): trial is SeedTrial => trial !== undefined);
}

function seedTrialForSearchSpace(trial: TrialResult, searchSpace: SearchSpace, sourceRound: number): SeedTrial | undefined {
  if (trial.value === null) {
    return undefined;
  }
  const activeNames = new Set(searchSpace.parameters.map((parameter) => parameter.name));
  const fixedNames = new Set((searchSpace.fixed_parameters ?? []).map((parameter) => parameter.name));
  for (const name of Object.keys(trial.params)) {
    if (!activeNames.has(name) && !fixedNames.has(name)) {
      return undefined;
    }
  }
  for (const fixed of searchSpace.fixed_parameters ?? []) {
    if (trial.params[fixed.name] !== fixed.value) {
      return undefined;
    }
  }
  const params = primitiveParams(trial.params);
  for (const parameter of searchSpace.parameters) {
    if (parameter.name in params) {
      if (!parameterValueIsValid(parameter, params[parameter.name])) {
        return undefined;
      }
      continue;
    }
    if (!isPrimitive(parameter.current_value) || !parameterValueIsValid(parameter, parameter.current_value)) {
      return undefined;
    }
    params[parameter.name] = parameter.current_value;
  }
  return {
    value: Number(trial.value),
    params,
    source_round: sourceRound,
    source_trial_number: trial.number
  };
}

function parameterValueIsValid(parameter: SearchParameter, value: unknown): boolean {
  if (parameter.type === "categorical") {
    return (parameter.choices ?? []).some((choice) => choice === value);
  }
  if (typeof value !== "number" || typeof parameter.low !== "number" || typeof parameter.high !== "number") {
    return false;
  }
  if (value < parameter.low || value > parameter.high) {
    return false;
  }
  return parameter.type !== "int" || Number.isInteger(value);
}

function primitiveParams(params: Record<string, unknown>): Record<string, string | number | boolean> {
  return Object.fromEntries(Object.entries(params).filter((entry): entry is [string, string | number | boolean] => isPrimitive(entry[1])));
}

function isPrimitive(value: unknown): value is string | number | boolean {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean";
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

function validateMaxParametersOption(maxParameters: number | undefined): void {
  if (maxParameters !== undefined && (!Number.isInteger(maxParameters) || maxParameters < 1)) {
    throw new Error("--max-parameters must be a positive integer");
  }
}

function validateSamplerSeedOption(options: RunOptions): void {
  if (options.samplerSeed === undefined) {
    return;
  }
  if (!Number.isInteger(options.samplerSeed) || options.samplerSeed < 0 || options.samplerSeed > 0xffffffff) {
    throw new Error("--sampler-seed must be an integer between 0 and 4294967295");
  }
  if (options.nJobs !== 1) {
    throw new Error("--sampler-seed requires --n-jobs 1 for reproducible ordering");
  }
}

function validateSamplerSeededSearchSpace(searchSpace: SearchSpace, nJobs: number): void {
  if (searchSpace.optuna?.seed !== undefined && nJobs !== 1) {
    throw new Error("A sampler seed requires --n-jobs 1 for reproducible ordering");
  }
}

function validateCentaurOptions(options: RunOptions, sampler: Sampler | undefined): void {
  if (options.centaur !== undefined && sampler !== "centaur") {
    throw new Error("Centaur options require the centaur sampler");
  }
  if (sampler !== "centaur") {
    return;
  }
  if (options.nJobs !== 1) {
    throw new Error("Centaur requires --n-jobs 1");
  }
  if ((options.refineRounds ?? 0) !== 0) {
    throw new Error("Centaur requires --refine-rounds 0 because CMA-ES uses a fixed search space");
  }
}

function validateCentaurSearchSpace(searchSpace: SearchSpace, maxParameters?: number): void {
  if (searchSpace.optuna?.sampler !== "centaur") {
    return;
  }
  if (maxParameters !== undefined && maxParameters < 2) {
    throw new Error("--max-parameters must be at least 2 with Centaur");
  }
  const numericParameterCount = searchSpace.parameters.filter(
    (parameter) => parameter.type === "float" || parameter.type === "int"
  ).length;
  if (numericParameterCount < 2) {
    throw new Error("Centaur requires at least 2 numeric parameters for CMA-ES");
  }
}

function searchSpacePathForRound(workDir: string, round: number, refineRounds: number): string {
  return refineRounds > 0 ? path.join(workDir, `search_space.round_${round}.yaml`) : path.join(workDir, "search_space.yaml");
}

function resultsPathForRound(workDir: string, finalResultsPath: string, round: number, refineRounds: number): string {
  return refineRounds > 0 ? path.join(workDir, `results.round_${round}.json`) : finalResultsPath;
}

function runnerPathForRound(workDir: string, scriptPath: string, round: number, refineRounds: number): string {
  const base = `${path.basename(scriptPath, path.extname(scriptPath))}_optuna`;
  return refineRounds > 0 ? path.join(workDir, `${base}.round_${round}.py`) : path.join(workDir, `${base}.py`);
}

async function copyLatestRunnerAlias(runnerPath: string, workDir: string, scriptPath: string, refineRounds: number): Promise<void> {
  if (refineRounds === 0) {
    return;
  }
  const latestPath = path.join(workDir, `${path.basename(scriptPath, path.extname(scriptPath))}_optuna.py`);
  await copyFile(runnerPath, latestPath);
}

async function runBuildCommand(
  template: string,
  context: CommandTemplateContext,
  silent: boolean | undefined
): Promise<void> {
  const command = expandCommandTemplateArgs(splitCommand(template), context);
  const [executable, ...args] = command;
  if (!executable) {
    throw new Error("build command cannot be empty");
  }
  if (!silent) {
    writeStatus(`Building runtime: ${styles.dim(command.join(" "))}`);
  }
  await runCommand(executable, args);
  if (!silent) {
    writeStatus("Build complete.", "success");
  }
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
  options: Pick<RunOptions, "direction" | "sampler" | "samplerSeed" | "pruner" | "centaur">
): SearchSpace {
  const sampler = options.sampler ?? searchSpace.optuna?.sampler ?? DEFAULT_SAMPLER;
  const { centaur: configuredCentaur, seed: configuredSeed, ...optuna } = searchSpace.optuna ?? {};
  const samplerSeed = sampler === "centaur" ? undefined : options.samplerSeed ?? configuredSeed;
  return {
    ...searchSpace,
    direction: options.direction ?? searchSpace.direction ?? DEFAULT_DIRECTION,
    optuna: {
      ...optuna,
      sampler,
      ...(samplerSeed === undefined ? {} : { seed: samplerSeed }),
      pruner: options.pruner ?? searchSpace.optuna?.pruner ?? DEFAULT_PRUNER,
      ...(sampler === "centaur"
        ? { centaur: { ...DEFAULT_CENTAUR_CONFIG, ...configuredCentaur, ...options.centaur } }
        : {})
    }
  };
}

function preserveConfiguredSamplerSeed(candidate: SearchSpace, previous: SearchSpace): SearchSpace {
  if (candidate.optuna?.sampler === "centaur") {
    const { seed: _seed, ...optuna } = candidate.optuna;
    return { ...candidate, optuna };
  }
  const seed = previous.optuna?.seed ?? candidate.optuna?.seed;
  if (seed === undefined) {
    return candidate;
  }
  return { ...candidate, optuna: { ...candidate.optuna, seed } };
}

async function scriptContainsMetricOutput(scriptPath: string): Promise<boolean> {
  return (await readFile(scriptPath, "utf8")).includes("autotune_metric");
}

function needsModifiedCopy(searchSpace: SearchSpace): boolean {
  return searchSpace.needs_wrapper || searchSpace.has_metric_output === false;
}

function shouldSkipHeadlessPrerequisite(input: {
  searchSpace?: SearchSpace;
  options: Pick<RunOptions, "yes">;
  refineRounds: number;
}): boolean {
  return Boolean(
    input.searchSpace &&
      input.searchSpace.optuna?.sampler !== "centaur" &&
      input.options.yes &&
      input.refineRounds === 0 &&
      !needsModifiedCopy(input.searchSpace)
  );
}

async function prepareModifiedInvocation(input: {
  invocation: ReturnType<typeof detectInvocation>;
  searchSpace: SearchSpace;
  workDir: string;
  round: number;
  totalRounds: number;
} & HeadlessOptions): Promise<ReturnType<typeof detectInvocation>> {
  const extension = path.extname(input.invocation.script);
  const baseName = path.basename(input.invocation.script, extension);
  const modifiedPath = input.totalRounds > 0
    ? path.join(input.workDir, `${baseName}_modified.round_${input.round}${extension}`)
    : path.join(input.workDir, `${baseName}_modified${extension}`);
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
  await copyLatestModifiedAlias(modifiedPath, input.workDir, input.invocation.script, input.totalRounds);
  return {
    ...input.invocation,
    script: modifiedPath,
    command: commandForModifiedScript(input.invocation, modifiedPath)
  };
}

async function copyLatestModifiedAlias(modifiedPath: string, workDir: string, scriptPath: string, refineRounds: number): Promise<void> {
  if (refineRounds === 0) {
    return;
  }
  const extension = path.extname(scriptPath);
  const baseName = path.basename(scriptPath, extension);
  await copyFile(modifiedPath, path.join(workDir, `${baseName}_modified${extension}`));
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
  budget?: SearchBudget;
  agentGuidance?: string;
  maxParameters?: number;
} & HeadlessOptions): Promise<SearchSpace> {
  writeStatus(`Phase 1: analyzing ${styles.dim(input.invocation.script)} with ${formatHeadlessLabel(input)}...`);
  writeStatus("This can take a minute on first run.");
  const searchSpace = await analyzeScript(input);
  writeStatus(`Analysis complete: ${searchSpace.parameters.length} parameter(s) proposed.`, "success");
  return searchSpace;
}

async function correctAgentParameterLimit(input: {
  candidate: SearchSpace;
  invocation: ReturnType<typeof detectInvocation>;
  workDir: string;
  budget?: SearchBudget;
  agentGuidance?: string;
  maxParameters?: number;
} & HeadlessOptions): Promise<SearchSpace> {
  return correctSearchSpaceParameterLimit(
    input.candidate,
    input.maxParameters,
    async (current, feedback) => {
      writeStatus(
        `Search space exceeds --max-parameters; requesting one correction with ${formatHeadlessLabel(input)}...`,
        "warning"
      );
      return reviseSearchSpace({
        invocation: input.invocation,
        searchSpace: current,
        feedback,
        workDir: input.workDir,
        budget: input.budget,
        agentGuidance: input.agentGuidance,
        maxParameters: input.maxParameters,
        agent: input.agent,
        model: input.model,
        reasoningEffort: input.reasoningEffort
      });
    }
  );
}

function searchBudgetForOptions(options: RunOptions, refineRounds: number, currentRefinementRound?: number): SearchBudget {
  const refineTrials = options.refineTrials ?? options.trials;
  return {
    trials: options.trials,
    timeoutSeconds: options.timeoutSeconds,
    refineRounds,
    refineTrials,
    refineMode: options.refineMode,
    currentRefinementRound,
    currentRoundTrials: currentRefinementRound === undefined ? options.trials : refineTrials
  };
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

async function readLatestRoundManifest(workDir: string): Promise<RoundManifest | undefined> {
  try {
    const text = await readFile(path.join(workDir, "rounds.json"), "utf8");
    const parsed = JSON.parse(text) as { rounds?: RoundManifest[] };
    const rounds = parsed.rounds ?? [];
    return rounds.reduce<RoundManifest | undefined>(
      (latest, round) => latest === undefined || round.round > latest.round ? round : latest,
      undefined
    );
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      return undefined;
    }
    throw error;
  }
}

function defaultStudyName(scriptPath: string): string {
  return `${path.basename(scriptPath, path.extname(scriptPath))}_autotune`;
}

function studyNameForRound(studyName: string, round: number, refineRounds: number): string {
  return refineRounds > 0 ? `${studyName}_round_${round}` : studyName;
}

interface RoundManifest {
  round: number;
  trials: number;
  search_space_path: string;
  results_path: string;
  runner_path: string;
  study_name: string;
  seed_count: number;
  storage?: string;
  sampler?: Sampler;
  sampler_seed?: number;
  centaur?: CentaurConfig & {
    agent: string;
    model?: string;
    reasoning_effort?: HeadlessOptions["reasoningEffort"];
  };
  refine_transfer_fixed_params: boolean;
  refine_transfer_trials: boolean;
}

function buildRoundManifest(input: {
  workDir: string;
  round: number;
  totalRounds: number;
  trials: number;
  searchSpacePath: string;
  resultsPath: string;
  runnerPath: string;
  studyName: string;
  seedCount: number;
  searchSpace: SearchSpace;
  headless: HeadlessOptions;
  options: RunOptions;
}): RoundManifest {
  const sampler = input.searchSpace.optuna?.sampler;
  const centaur = input.searchSpace.optuna?.centaur;
  return {
    round: input.round,
    trials: input.trials,
    search_space_path: path.relative(input.workDir, input.searchSpacePath),
    results_path: path.relative(input.workDir, input.resultsPath),
    runner_path: path.relative(input.workDir, input.runnerPath),
    study_name: input.studyName,
    seed_count: input.seedCount,
    storage: input.options.storage,
    sampler,
    sampler_seed: input.searchSpace.optuna?.seed,
    ...(sampler === "centaur" && centaur
      ? {
          centaur: {
            ...centaur,
            agent: input.headless.agent,
            model: input.headless.model,
            reasoning_effort: input.headless.reasoningEffort
          }
        }
      : {}),
    refine_transfer_fixed_params: input.options.refineTransferFixedParams !== false,
    refine_transfer_trials: input.options.refineTransferTrials !== false
  };
}

async function writeRoundManifest(workDir: string, rounds: RoundManifest[]): Promise<void> {
  await writeFile(path.join(workDir, "rounds.json"), `${JSON.stringify({ rounds }, null, 2)}\n`, "utf8");
}
