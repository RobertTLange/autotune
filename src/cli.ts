#!/usr/bin/env node
import { Command, Option } from "commander";
import { closeSync, constants, fstatSync, openSync, readSync, realpathSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { plotProgress } from "./progress-plot.js";
import { analyzeOnly, doctorAutotune, resumeStudy, runAutotune, showResults } from "./workflow.js";
import type { Direction, Pruner, ReasoningEffort, RefineMode, RunOptions, Sampler } from "./types.js";

const MAX_AGENT_GUIDANCE_FILE_BYTES = 65536;

export function createProgram(): Command {
  const program = new Command();

  program
    .name("autotune")
    .description("Automatic hyperparameter optimization CLI powered by headless and Optuna.")
    .version("0.1.0");

  program
  .command("run")
  .argument("<script>", "script or executable to optimize")
  .requiredOption("--trials <n>", "number of Optuna trials", parsePositiveInt)
  .addOption(new Option("--direction <direction>", "optimization direction").choices(["maximize", "minimize"]))
  .addOption(new Option("--sampler <sampler>", "Optuna sampler").choices(["tpe", "random", "cmaes", "grid"]))
  .addOption(new Option("--pruner <pruner>", "Optuna pruner").choices(["none", "median", "hyperband"]))
  .option("--storage <uri>", "Optuna storage URI, such as sqlite:///study.db")
  .option("--study-name <name>", "Optuna study name for persistent storage")
  .option("--n-jobs <n>", "parallel trial workers", parsePositiveInt, 1)
  .option("--agent <name>", "headless agent", "claude")
  .option("--model <name>", "headless model override")
  .option("--agent-guidance <text>", "extra guidance for search-space generation and refinement")
  .option("--agent-guidance-file <file>", "file with extra guidance for search-space generation and refinement")
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
  .action(async (script: string, raw: Record<string, unknown>, command: Command) => {
    await runAutotune(script, normalizeRunOptions(raw, command));
  });

  program
  .command("analyze")
  .argument("<script>", "script or executable to analyze")
  .option("--agent <name>", "headless agent", "claude")
  .option("--model <name>", "headless model override")
  .option("--agent-guidance <text>", "extra guidance for search-space generation")
  .option("--agent-guidance-file <file>", "file with extra guidance for search-space generation")
  .addOption(new Option("--reasoning-effort <level>", "headless reasoning effort").choices(["low", "medium", "high", "xhigh"]))
  .addOption(new Option("--effort <level>", "alias for --reasoning-effort").choices(["low", "medium", "high", "xhigh"]))
  .option("--json", "print JSON search space", false)
  .option("--output <file>", "write search space YAML to file")
  .option("--work-dir <dir>", "artifact directory", "autotune")
  .option("--command <command>", "override script invocation command")
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
  }) => {
    await analyzeOnly(script, { ...raw, reasoningEffort: normalizeReasoningEffort(raw), agentGuidance: normalizeAgentGuidance(raw) });
  });

  program
  .command("doctor")
  .argument("[script]", "optional script to check runtime detection")
  .option("--agent <name>", "headless agent", "claude")
  .option("--command <command>", "override script invocation command")
  .action(async (script: string | undefined, options: { agent: string; command?: string }) => {
    await doctorAutotune({ script, agent: options.agent, command: options.command });
  });

  program
  .command("results")
  .argument("[dir]", "artifact directory or results JSON file", "autotune")
  .option("--json", "print JSON", false)
  .option("--top <n>", "number of top trials", parsePositiveInt, 10)
  .action(async (dir: string, options: { json: boolean; top: number }) => {
    await showResults({ dir, json: options.json, top: options.top });
  });

  program
  .command("plot-progress")
  .argument("<run-dir>", "ablation run directory containing variant result subdirectories")
  .requiredOption("--output <file>", "write SVG plot to file")
  .option("--title <title>", "plot title")
  .option("--max-trials <n>", "maximum evaluated trials on the x-axis", parsePositiveInt, 100)
  .option("--width <px>", "SVG width", parsePositiveInt, 1100)
  .option("--height <px>", "SVG height", parsePositiveInt, 650)
  .option("--include-failed", "include failed/timeout trial values when updating best-so-far", false)
  .action(async (runDir: string, options: {
    output: string;
    title?: string;
    maxTrials: number;
    width: number;
    height: number;
    includeFailed: boolean;
  }) => {
    await plotProgress(runDir, options);
    console.log(`Wrote ${options.output}`);
  });

  program
  .command("resume")
  .requiredOption("--storage <uri>", "Optuna storage URI")
  .option("--study-name <name>", "Optuna study name")
  .requiredOption("--trials <n>", "additional trials", parsePositiveInt)
  .option("--n-jobs <n>", "parallel trial workers", parsePositiveInt, 1)
  .option("--work-dir <dir>", "artifact directory", "autotune")
  .addOption(new Option("--direction <direction>", "fallback direction").choices(["maximize", "minimize"]).default("maximize"))
  .action(async (options: { storage: string; studyName?: string; trials: number; nJobs: number; workDir: string; direction: Direction }) => {
    await resumeStudy(options);
  });

  return program;
}

if (isMainModule()) {
  createProgram().parseAsync(process.argv).catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}

export function normalizeRunOptions(raw: Record<string, unknown>, command: Command): RunOptions {
  return {
    trials: Number(raw.trials),
    direction: optionValue(raw, command, "direction") as Direction | undefined,
    sampler: optionValue(raw, command, "sampler") as Sampler | undefined,
    pruner: optionValue(raw, command, "pruner") as Pruner | undefined,
    nJobs: Number(raw.nJobs),
    workDir: typeof raw.workDir === "string" ? raw.workDir : undefined,
    agent: String(raw.agent),
    model: typeof raw.model === "string" ? raw.model : undefined,
    agentGuidance: normalizeAgentGuidance(raw),
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

export function isMainModule(moduleUrl = import.meta.url, argvPath = process.argv[1]): boolean {
  if (argvPath === undefined) {
    return false;
  }
  return moduleUrl === pathToFileURL(realpathSync(argvPath)).href;
}
