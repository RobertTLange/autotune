#!/usr/bin/env node
import { Command, Option } from "commander";
import { closeSync, constants, fstatSync, openSync, readFileSync, readSync, realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { plotProgress } from "./progress-plot.js";
import type { ProgressXAxis } from "./progress-plot.js";
import { SDK_PROTOCOL_VERSION, redactSdkErrorMessage, renderSdkError, renderSdkResult } from "./sdk.js";
import { analyzeOnly, doctorAutotune, resumeStudy, runAutotune, showResults } from "./workflow.js";
import type { Direction, Pruner, ReasoningEffort, RefineMode, RunOptions, Sampler } from "./types.js";

const MAX_AGENT_GUIDANCE_FILE_BYTES = 65536;
export const PACKAGE_VERSION = (JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
  version: string;
}).version;

export function createProgram(): Command {
  const program = new Command();

  program
    .name("autotune")
    .description("Automatic hyperparameter optimization CLI powered by headless and Optuna.")
    .version(PACKAGE_VERSION);

  program
  .command("run")
  .argument("<script>", "script or executable to optimize")
  .requiredOption("--trials <n>", "number of Optuna trials", parsePositiveInt)
  .addOption(new Option("--direction <direction>", "optimization direction").choices(["maximize", "minimize"]))
  .addOption(new Option("--sampler <sampler>", "Optuna sampler").choices(["tpe", "random", "cmaes", "grid", "centaur"]))
  .option("--sampler-seed <n>", "reproducible seed for TPE, random, CMA-ES, or grid", parseSamplerSeed)
  .option("--centaur-llm-probability <probability>", "probability that Centaur requests an LLM proposal", parseProbability)
  .option("--centaur-warmup-trials <n>", "CMA-ES-only trials before Centaur requests LLM proposals", parseNonNegativeInt)
  .option("--centaur-seed <n>", "Centaur proposal scheduler seed", parseNonNegativeInt)
  .addOption(new Option("--pruner <pruner>", "Optuna pruner").choices(["none", "median", "hyperband"]))
  .option("--storage <uri>", "Optuna storage URI, such as sqlite:///study.db")
  .option("--study-name <name>", "Optuna study name for persistent storage")
  .option("--n-jobs <n>", "parallel trial workers", parsePositiveInt, 1)
  .option("--agent <name>", "headless agent", "claude")
  .option("--model <name>", "headless model override")
  .option("--agent-guidance <text>", "extra guidance for search-space generation and refinement")
  .option("--agent-guidance-file <file>", "file with extra guidance for search-space generation and refinement")
  .option("--max-parameters <n>", "maximum active parameters in the search space", parsePositiveInt)
  .addOption(new Option("--reasoning-effort <level>", "headless reasoning effort").choices(["low", "medium", "high", "xhigh"]))
  .addOption(new Option("--effort <level>", "alias for --reasoning-effort").choices(["low", "medium", "high", "xhigh"]))
  .option("--command <command>", "override script invocation command")
  .option("--build-command <command>", "command to run once before analysis/trials; supports {script} and {work-dir}")
  .option("--timeout-seconds <n>", "per-trial timeout in seconds", parsePositiveInt)
  .option("--time-budget-seconds <n>", "stop after cumulative completed trial runtime reaches this many seconds", parsePositiveInt)
  .option("--refine-rounds <n>", "agentic search-space refinement rounds after the initial trials", parseNonNegativeInt, 0)
  .option("--refine-trials <n>", "trials per refinement round; defaults to --trials", parsePositiveInt)
  .addOption(new Option("--refine-mode <mode>", "refinement approval mode").choices(["ask", "auto"]).default("ask"))
  .option("--no-refine-transfer-fixed-params", "do not fix dropped refinement parameters at previous-best values")
  .option("--no-refine-transfer-trials", "do not seed refinement rounds with compatible previous trials")
  .option("--json", "print JSON results", false)
  .option("--output <file>", "write JSON results to file")
  .option("--work-dir <dir>", "artifact directory")
  .option("--yes", "skip confirmation prompts", false)
  .option("--config <file>", "pre-defined search space YAML/JSON")
  .addOption(sdkFormatOption())
  .action(async (script: string, raw: Record<string, unknown>, command: Command) => {
    const sdk = isSdkRequest(raw);
    rejectConflictingSdkOptions(raw, sdk);
    requireSdkRunConfirmation(raw, sdk);
    const result = await runAutotune(script, { ...normalizeRunOptions(raw, command), silent: sdk });
    if (sdk) printSdkResult("run", result);
  });

  program
  .command("analyze")
  .argument("<script>", "script or executable to analyze")
  .option("--agent <name>", "headless agent", "claude")
  .option("--model <name>", "headless model override")
  .option("--agent-guidance <text>", "extra guidance for search-space generation")
  .option("--agent-guidance-file <file>", "file with extra guidance for search-space generation")
  .option("--max-parameters <n>", "maximum active parameters in the search space", parsePositiveInt)
  .addOption(new Option("--reasoning-effort <level>", "headless reasoning effort").choices(["low", "medium", "high", "xhigh"]))
  .addOption(new Option("--effort <level>", "alias for --reasoning-effort").choices(["low", "medium", "high", "xhigh"]))
  .option("--json", "print JSON search space", false)
  .option("--output <file>", "write search space YAML to file")
  .option("--work-dir <dir>", "artifact directory", "autotune")
  .option("--command <command>", "override script invocation command")
  .addOption(sdkFormatOption())
  .action(async (script: string, raw: {
    agent: string;
    model?: string;
    reasoningEffort?: ReasoningEffort;
    effort?: ReasoningEffort;
    json: boolean;
    output?: string;
    workDir: string;
    command?: string;
    agentGuidance?: string;
    agentGuidanceFile?: string;
    maxParameters?: number;
    sdkFormat?: "json";
  }) => {
    const sdk = isSdkRequest(raw);
    rejectConflictingSdkOptions(raw, sdk);
    const result = await analyzeOnly(script, {
      ...raw,
      reasoningEffort: normalizeReasoningEffort(raw),
      agentGuidance: normalizeAgentGuidance(raw),
      silent: sdk
    });
    if (sdk) printSdkResult("analyze", result);
  });

  program
  .command("doctor")
  .argument("[script]", "optional script to check runtime detection")
  .option("--agent <name>", "headless agent", "claude")
  .option("--model <name>", "headless model override")
  .option("--command <command>", "override script invocation command")
  .addOption(sdkFormatOption())
  .action(async (script: string | undefined, options: { agent: string; model?: string; command?: string; sdkFormat?: "json" }) => {
    const sdk = isSdkRequest(options);
    const checks = await doctorAutotune({ script, agent: options.agent, model: options.model, command: options.command, silent: sdk });
    if (sdk) printSdkResult("doctor", checks);
  });

  program
  .command("results")
  .argument("[dir]", "artifact directory or results JSON file", "autotune")
  .option("--json", "print JSON", false)
  .option("--top <n>", "number of top trials", parsePositiveInt, 10)
  .addOption(sdkFormatOption())
  .action(async (dir: string, options: { json: boolean; top: number; sdkFormat?: "json" }) => {
    const sdk = isSdkRequest(options);
    rejectConflictingSdkOptions(options, sdk);
    const result = await showResults({ dir, json: options.json, top: options.top, silent: sdk });
    if (sdk) printSdkResult("results", result);
  });

  program
  .command("plot-progress")
  .argument("<run-dir>", "ablation run directory containing variant result subdirectories")
  .requiredOption("--output <file>", "write SVG plot to file")
  .option("--title <title>", "plot title")
  .option("--max-trials <n>", "maximum evaluated trials to include", parsePositiveInt, 100)
  .addOption(new Option("--x-axis <axis>", "x-axis scale").choices(["trials", "runtime"]).default("trials"))
  .option("--max-runtime-hours <n>", "maximum cumulative runtime on a runtime x-axis", parseFiniteNumber)
  .option("--width <px>", "SVG width", parsePositiveInt, 1100)
  .option("--height <px>", "SVG height", parsePositiveInt, 650)
  .option("--y-min <n>", "minimum y-axis value", parseFiniteNumber)
  .option("--y-max <n>", "maximum y-axis value", parseFiniteNumber)
  .option("--include-failed", "include failed/timeout trial values when updating best-so-far", false)
  .addOption(sdkFormatOption())
  .action(async (runDir: string, options: {
    output: string;
    title?: string;
    maxTrials: number;
    xAxis: ProgressXAxis;
    maxRuntimeHours?: number;
    width: number;
    height: number;
    yMin?: number;
    yMax?: number;
    includeFailed: boolean;
    sdkFormat?: "json";
  }) => {
    await plotProgress(runDir, options);
    if (isSdkRequest(options)) printSdkResult("plot-progress", { output: options.output });
    else console.log(`Wrote ${options.output}`);
  });

  program
  .command("resume")
  .requiredOption("--storage <uri>", "Optuna storage URI")
  .option("--study-name <name>", "Optuna study name")
  .requiredOption("--trials <n>", "additional trials", parsePositiveInt)
  .option("--n-jobs <n>", "parallel trial workers", parsePositiveInt, 1)
  .option("--work-dir <dir>", "artifact directory", "autotune")
  .addOption(new Option("--direction <direction>", "fallback direction").choices(["maximize", "minimize"]).default("maximize"))
  .addOption(sdkFormatOption())
  .action(async (options: { storage: string; studyName?: string; trials: number; nJobs: number; workDir: string; direction: Direction; sdkFormat?: "json" }) => {
    const sdk = isSdkRequest(options);
    const result = await resumeStudy({ ...options, silent: sdk });
    if (sdk) printSdkResult("resume", result);
  });

  program
    .command("capabilities")
    .addOption(sdkFormatOption())
    .action((options: { sdkFormat?: "json" }) => {
      const capabilities = {
        protocolVersion: SDK_PROTOCOL_VERSION,
        commands: ["analyze", "doctor", "plot-progress", "results", "resume", "run"]
      };
      if (isSdkRequest(options)) printSdkResult("capabilities", capabilities);
      else console.log(JSON.stringify(capabilities, null, 2));
    });

  return program;
}

if (isMainModule()) {
  const program = createProgram();
  configureExecutableProgram(program, usesSdkFormat(process.argv));
  program.parseAsync(process.argv).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    const exitCode = existingOrErrorExitCode(error);
    if (usesSdkFormat(process.argv)) {
      console.log(renderSdkError(
        redactSdkErrorMessage(message, process.argv),
        exitCode,
        sdkCommandFromArgv(process.argv)
      ));
    } else {
      console.error(message);
    }
    process.exitCode = exitCode;
  });
}

function sdkFormatOption(): Option {
  return new Option("--sdk-format <format>", "versioned SDK output").choices(["json"]);
}

function isSdkRequest(options: { sdkFormat?: unknown }): boolean {
  return options.sdkFormat === "json";
}

function rejectConflictingSdkOptions(options: { json?: unknown }, sdk: boolean): void {
  if (sdk && options.json === true) {
    throw new Error("--json cannot be used with --sdk-format");
  }
}

function requireSdkRunConfirmation(options: { yes?: unknown }, sdk: boolean): void {
  if (sdk && options.yes !== true) {
    throw new Error("--sdk-format run requires --yes");
  }
}

function printSdkResult(command: string, data: unknown): void {
  console.log(renderSdkResult(command, data));
}

export function usesSdkFormat(argv: string[]): boolean {
  return argv.some((value) => value === "--sdk-format" || value.startsWith("--sdk-format="));
}

export function sdkCommandFromArgv(argv: string[]): string {
  const command = argv[2];
  return command !== undefined &&
    ["analyze", "capabilities", "doctor", "plot-progress", "results", "resume", "run"].includes(command)
    ? command
    : "cli";
}

function errorExitCode(error: unknown): number {
  if (typeof error === "object" && error !== null && "exitCode" in error) {
    const exitCode = (error as { exitCode?: unknown }).exitCode;
    if (typeof exitCode === "number" && Number.isInteger(exitCode) && exitCode > 0) {
      return exitCode;
    }
  }
  return 1;
}

function existingOrErrorExitCode(error: unknown): number {
  const existing = process.exitCode;
  return typeof existing === "number" && Number.isInteger(existing) && existing > 0
    ? existing
    : errorExitCode(error);
}

export function configureExecutableProgram(program: Command, sdk: boolean): void {
  if (!sdk) {
    return;
  }
  for (const command of [program, ...program.commands]) {
    command.exitOverride();
    command.configureOutput({ writeErr: () => undefined });
  }
}

export function normalizeRunOptions(raw: Record<string, unknown>, command: Command): RunOptions {
  const sampler = optionValue(raw, command, "sampler") as Sampler | undefined;
  const samplerSeed = typeof raw.samplerSeed === "number" ? raw.samplerSeed : undefined;
  const centaur = normalizeCentaurOverrides(raw);
  if (centaur !== undefined && sampler !== undefined && sampler !== "centaur") {
    throw new Error("Centaur options require --sampler centaur");
  }
  if (samplerSeed !== undefined && sampler === "centaur") {
    throw new Error("Centaur uses --centaur-seed instead of --sampler-seed");
  }
  return {
    trials: Number(raw.trials),
    direction: optionValue(raw, command, "direction") as Direction | undefined,
    sampler,
    samplerSeed,
    pruner: optionValue(raw, command, "pruner") as Pruner | undefined,
    centaur,
    nJobs: Number(raw.nJobs),
    workDir: typeof raw.workDir === "string" ? raw.workDir : undefined,
    agent: String(raw.agent),
    model: typeof raw.model === "string" ? raw.model : undefined,
    agentGuidance: normalizeAgentGuidance(raw),
    maxParameters: typeof raw.maxParameters === "number" ? raw.maxParameters : undefined,
    reasoningEffort: normalizeReasoningEffort(raw),
    command: typeof raw.command === "string" ? raw.command : undefined,
    buildCommand: typeof raw.buildCommand === "string" ? raw.buildCommand : undefined,
    timeoutSeconds: typeof raw.timeoutSeconds === "number" ? raw.timeoutSeconds : undefined,
    timeBudgetSeconds: typeof raw.timeBudgetSeconds === "number" ? raw.timeBudgetSeconds : undefined,
    refineRounds: Number(raw.refineRounds),
    refineTrials: typeof raw.refineTrials === "number" ? raw.refineTrials : undefined,
    refineMode: raw.refineMode as RefineMode,
    refineTransferFixedParams: raw.refineTransferFixedParams !== false,
    refineTransferTrials: raw.refineTransferTrials !== false,
    json: Boolean(raw.json),
    output: typeof raw.output === "string" ? raw.output : undefined,
    storage: typeof raw.storage === "string" ? raw.storage : undefined,
    studyName: typeof raw.studyName === "string" ? raw.studyName : undefined,
    yes: Boolean(raw.yes),
    config: typeof raw.config === "string" ? raw.config : undefined
  };
}

function normalizeCentaurOverrides(raw: Record<string, unknown>): RunOptions["centaur"] {
  const overrides: NonNullable<RunOptions["centaur"]> = {};
  if (typeof raw.centaurLlmProbability === "number") {
    overrides.llm_probability = raw.centaurLlmProbability;
  }
  if (typeof raw.centaurWarmupTrials === "number") {
    overrides.warmup_trials = raw.centaurWarmupTrials;
  }
  if (typeof raw.centaurSeed === "number") {
    overrides.seed = raw.centaurSeed;
  }
  return Object.keys(overrides).length > 0 ? overrides : undefined;
}

export function normalizeAgentGuidance(raw: Record<string, unknown>): string | undefined {
  const parts = [];
  if (typeof raw.agentGuidanceFile === "string") {
    parts.push(readGuidance(raw.agentGuidanceFile, "--agent-guidance-file"));
  }
  if (typeof raw.agentGuidance === "string") {
    parts.push(trimGuidance(raw.agentGuidance, "--agent-guidance"));
  }
  return parts.length > 0 ? parts.join("\n\n") : undefined;
}

function readGuidance(filePath: string, label: string): string {
  let fd: number | undefined;
  try {
    fd = openSync(filePath, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    const stats = fstatSync(fd);
    if (!stats.isFile()) {
      throw new Error(`${label} must point to a regular file: ${filePath}`);
    }
    if (stats.size > MAX_AGENT_GUIDANCE_FILE_BYTES) {
      throw new Error(`${label} must be ${MAX_AGENT_GUIDANCE_FILE_BYTES} bytes or smaller: ${filePath}`);
    }
    return trimGuidance(readBoundedGuidance(fd, label, filePath), label);
  } catch (error) {
    const code = errorCode(error);
    if (code === "ENOENT") {
      throw new Error(`${label} file not found: ${filePath}`);
    }
    if (code === "ELOOP" || code === "EISDIR") {
      throw new Error(`${label} must point to a regular file: ${filePath}`);
    }
    throw error;
  } finally {
    if (fd !== undefined) {
      closeSync(fd);
    }
  }
}

function readBoundedGuidance(fd: number, label: string, filePath: string): string {
  const chunks: Buffer[] = [];
  let total = 0;
  while (true) {
    const remaining = MAX_AGENT_GUIDANCE_FILE_BYTES + 1 - total;
    const buffer = Buffer.allocUnsafe(Math.min(8192, remaining));
    const bytesRead = readSync(fd, buffer, 0, buffer.length, null);
    if (bytesRead === 0) {
      return Buffer.concat(chunks, total).toString("utf8");
    }
    total += bytesRead;
    if (total > MAX_AGENT_GUIDANCE_FILE_BYTES) {
      throw new Error(`${label} must be ${MAX_AGENT_GUIDANCE_FILE_BYTES} bytes or smaller: ${filePath}`);
    }
    chunks.push(Buffer.from(buffer.subarray(0, bytesRead)));
  }
}

function errorCode(error: unknown): string | undefined {
  if (typeof error !== "object" || error === null || !("code" in error)) {
    return undefined;
  }
  const code = (error as { code?: unknown }).code;
  return typeof code === "string" ? code : undefined;
}

function trimGuidance(value: string, label: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error(`${label} must contain non-empty agent guidance`);
  }
  return trimmed;
}

export function normalizeReasoningEffort(raw: Record<string, unknown>): ReasoningEffort | undefined {
  return (
    typeof raw.reasoningEffort === "string"
      ? raw.reasoningEffort
      : typeof raw.effort === "string"
        ? raw.effort
        : undefined
  ) as ReasoningEffort | undefined;
}

function optionValue(raw: Record<string, unknown>, command: Command, name: string): string | undefined {
  const value = raw[name];
  if (typeof value !== "string") {
    return undefined;
  }
  return command.getOptionValueSource(name) === undefined ? undefined : value;
}

export function parsePositiveInt(value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`expected positive integer, got ${value}`);
  }
  return parsed;
}

export function parseNonNegativeInt(value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`expected non-negative integer, got ${value}`);
  }
  return parsed;
}

export function parseSamplerSeed(value: string): number {
  const parsed = parseNonNegativeInt(value);
  if (parsed > 0xffffffff) {
    throw new Error(`expected sampler seed at most ${0xffffffff}, got ${value}`);
  }
  return parsed;
}

export function parseFiniteNumber(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`expected finite number, got ${value}`);
  }
  return parsed;
}

export function parseProbability(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`expected number between 0 and 1, got ${value}`);
  }
  return parsed;
}

export function isMainModule(moduleUrl = import.meta.url, argvPath = process.argv[1]): boolean {
  if (argvPath === undefined) {
    return false;
  }
  return realpathSync(fileURLToPath(moduleUrl)) === realpathSync(argvPath);
}
