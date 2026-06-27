#!/usr/bin/env node
import { Command, Option } from "commander";
import { pathToFileURL } from "node:url";
import { analyzeOnly, doctorAutotune, resumeStudy, runAutotune, showResults } from "./workflow.js";
import type { Direction, Pruner, ReasoningEffort, RefineMode, RunOptions, Sampler } from "./types.js";

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
  .addOption(new Option("--reasoning-effort <level>", "headless reasoning effort").choices(["low", "medium", "high", "xhigh"]))
  .addOption(new Option("--effort <level>", "alias for --reasoning-effort").choices(["low", "medium", "high", "xhigh"]))
  .option("--command <command>", "override script invocation command")
  .option("--build-command <command>", "command to run once before analysis/trials; supports {script} and {work-dir}")
  .option("--refine-rounds <n>", "agentic search-space refinement rounds after the initial trials", parseNonNegativeInt, 0)
  .option("--refine-trials <n>", "trials per refinement round; defaults to --trials", parsePositiveInt)
  .addOption(new Option("--refine-mode <mode>", "refinement approval mode").choices(["ask", "auto"]).default("ask"))
  .option("--json", "print JSON results", false)
  .option("--output <file>", "write JSON results to file")
  .option("--work-dir <dir>", "artifact directory", ".autotune")
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
  .addOption(new Option("--reasoning-effort <level>", "headless reasoning effort").choices(["low", "medium", "high", "xhigh"]))
  .addOption(new Option("--effort <level>", "alias for --reasoning-effort").choices(["low", "medium", "high", "xhigh"]))
  .option("--json", "print JSON search space", false)
  .option("--output <file>", "write search space YAML to file")
  .option("--work-dir <dir>", "artifact directory", ".autotune")
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
  }) => {
    await analyzeOnly(script, { ...raw, reasoningEffort: normalizeReasoningEffort(raw) });
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
  .argument("[dir]", "artifact directory or results JSON file", ".autotune")
  .option("--json", "print JSON", false)
  .option("--top <n>", "number of top trials", parsePositiveInt, 10)
  .action(async (dir: string, options: { json: boolean; top: number }) => {
    await showResults({ dir, json: options.json, top: options.top });
  });

  program
  .command("resume")
  .requiredOption("--storage <uri>", "Optuna storage URI")
  .option("--study-name <name>", "Optuna study name")
  .requiredOption("--trials <n>", "additional trials", parsePositiveInt)
  .option("--n-jobs <n>", "parallel trial workers", parsePositiveInt, 1)
  .option("--work-dir <dir>", "artifact directory", ".autotune")
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
    workDir: String(raw.workDir),
    agent: String(raw.agent),
    model: typeof raw.model === "string" ? raw.model : undefined,
    reasoningEffort: normalizeReasoningEffort(raw),
    command: typeof raw.command === "string" ? raw.command : undefined,
    buildCommand: typeof raw.buildCommand === "string" ? raw.buildCommand : undefined,
    refineRounds: Number(raw.refineRounds),
    refineTrials: typeof raw.refineTrials === "number" ? raw.refineTrials : undefined,
    refineMode: raw.refineMode as RefineMode,
    json: Boolean(raw.json),
    output: typeof raw.output === "string" ? raw.output : undefined,
    storage: typeof raw.storage === "string" ? raw.storage : undefined,
    studyName: typeof raw.studyName === "string" ? raw.studyName : undefined,
    yes: Boolean(raw.yes),
    config: typeof raw.config === "string" ? raw.config : undefined
  };
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

function isMainModule(): boolean {
  return process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
}
