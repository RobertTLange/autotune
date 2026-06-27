import { chmod, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { extractHeadlessJson, extractHeadlessObject, runHeadless } from "./headless.js";
import {
  renderAnalyzePrompt,
  renderModifiedScriptPrompt,
  renderRefineSearchSpacePrompt,
  renderReviseSearchSpacePrompt,
  type TrialResultSummary
} from "./prompts.js";
import type { HeadlessOptions, Invocation, SearchSpace } from "./types.js";

export async function analyzeScript(input: {
  invocation: Invocation;
  workDir: string;
} & HeadlessOptions): Promise<SearchSpace> {
  await mkdir(input.workDir, { recursive: true });
  const promptPath = path.join(input.workDir, "analyze_prompt.md");
  await writeFile(promptPath, renderAnalyzePrompt({ invocation: input.invocation }), "utf8");
  const output = await retryHeadless(buildHeadlessArgs(input, promptPath));
  return extractHeadlessJson(output);
}

export async function reviseSearchSpace(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  feedback: string;
  workDir: string;
} & HeadlessOptions): Promise<SearchSpace> {
  await mkdir(input.workDir, { recursive: true });
  const promptPath = path.join(input.workDir, "revise_prompt.md");
  await writeFile(
    promptPath,
    renderReviseSearchSpacePrompt({
      invocation: input.invocation,
      searchSpace: input.searchSpace,
      feedback: input.feedback
    }),
    "utf8"
  );
  const output = await retryHeadless(buildHeadlessArgs(input, promptPath));
  return extractHeadlessJson(output);
}

export async function refineSearchSpaceFromTrials(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  trialSummary: TrialResultSummary;
  round: number;
  workDir: string;
} & HeadlessOptions): Promise<SearchSpace> {
  await mkdir(input.workDir, { recursive: true });
  const promptPath = path.join(input.workDir, `refine_prompt.round_${input.round}.md`);
  await writeFile(
    promptPath,
    renderRefineSearchSpacePrompt({
      invocation: input.invocation,
      searchSpace: input.searchSpace,
      trialSummary: input.trialSummary,
      round: input.round
    }),
    "utf8"
  );
  const output = await retryHeadless(buildHeadlessArgs(input, promptPath));
  return extractHeadlessJson(output);
}

export async function generateModifiedScript(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  workDir: string;
  outputPath: string;
} & HeadlessOptions): Promise<string> {
  await mkdir(input.workDir, { recursive: true });
  const promptPath = path.join(input.workDir, "modified_prompt.md");
  await writeFile(
    promptPath,
    renderModifiedScriptPrompt({
      invocation: input.invocation,
      searchSpace: input.searchSpace,
      outputPath: input.outputPath
    }),
    "utf8"
  );
  const output = await retryHeadless(buildHeadlessArgs(input, promptPath));
  const parsed = extractHeadlessObject(output);
  if (typeof parsed.code !== "string" || parsed.code.trim().length === 0) {
    throw new Error("headless modified script response must contain a non-empty code string");
  }
  await writeFile(input.outputPath, parsed.code, "utf8");
  await chmod(input.outputPath, 0o755);
  return input.outputPath;
}

function buildHeadlessArgs(input: { invocation: Invocation } & HeadlessOptions, promptPath: string): string[] {
  const args = [input.agent];
  if (input.model) {
    args.push("--model", input.model);
  }
  if (input.reasoningEffort) {
    args.push("--reasoning-effort", input.reasoningEffort);
  }
  args.push("--prompt-file", promptPath, "--work-dir", path.dirname(input.invocation.script), "--json");
  return args;
}

async function retryHeadless(args: string[]): Promise<string> {
  try {
    return await runHeadless(args, { cwd: process.cwd() });
  } catch (firstError) {
    try {
      return await runHeadless(args, { cwd: process.cwd() });
    } catch (secondError) {
      throw new Error(`headless failed after retry: ${String(secondError)}; first error: ${String(firstError)}`);
    }
  }
}
