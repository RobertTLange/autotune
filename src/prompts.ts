import type { Invocation, SearchSpace } from "./types.js";

export interface TrialResultSummary {
  direction: "maximize" | "minimize";
  n_trials: number;
  best_trial: unknown;
  top_trials: unknown[];
  parameter_ranges: unknown[];
}

export function renderAnalyzePrompt(input: { invocation: Invocation }): string {
  return `Analyze the following script for hyperparameter tuning.

The script language is: ${input.invocation.language}
The script is invoked via: ${formatInvocation(input.invocation)}

Identify all tunable hyperparameters and propose Optuna search spaces.
The optimization metric is reported via printing "autotune_metric=<value>" to stdout.

A tunable hyperparameter should change candidate behavior before the objective is measured.
Do not include parameters that change objective measurement, including the evaluation input set,
scoring formula, aggregation window, reporting threshold, comparison baseline, output formatting,
or random seed used only for measurement. Leave these values fixed so trials remain comparable.
If you intentionally exclude an important metric or evaluation value, mention it in reasoning.

For each parameter, output JSON with:
- name: variable name / parameter name in the script
- cli_flag: the CLI argument name to pass this parameter, such as "--lr" or "--batch-size"
- type: "float" | "int" | "categorical"
- low/high: for float/int
- log: boolean for float log scale
- choices: for categorical
- current_value: the hardcoded value in the script

Also identify:
- has_arg_parsing: whether the script already has CLI argument parsing
- needs_wrapper: whether a wrapper script is needed to add arg parsing
- has_metric_output: whether the script already prints "autotune_metric=<value>" to stdout
- direction: "maximize" | "minimize"
- optuna: object with optional sampler, pruner, and reasoning fields
- reasoning: why this direction

Also propose safe optuna settings:
- sampler: "tpe" | "random" | "cmaes" | "grid"
- pruner: "none" | "median" | "hyperband"
- reasoning: short explanation for the Optuna choices
Prefer tpe for mixed or continuous spaces, random for tiny exploratory searches, grid only when all
parameters are small categorical choices, and cmaes only for continuous numeric spaces. Prefer none
for pruner unless the script is iterative and pruning is likely comparable across trials.
Do not propose storage. Do not propose n_jobs. These are user-controlled resource/state settings.

Output valid JSON only.`;
}

function formatInvocation(invocation: Invocation): string {
  if (invocation.scriptArgument === "included" || invocation.scriptArgument === "none") {
    return invocation.command.join(" ");
  }
  return [...invocation.command, invocation.script].join(" ");
}

export function renderGeneratePrompt(input: { invocation: Invocation; searchSpace: SearchSpace; outputPath: string }): string {
  return `Generate an Optuna runner for this autotune search.

Script language: ${input.invocation.language}
Invocation command argv: ${JSON.stringify(input.invocation.command)}
Script path: ${input.invocation.script}
Output runner path: ${input.outputPath}

Confirmed search space JSON:
${JSON.stringify(input.searchSpace, null, 2)}

Requirements:
- preserve the original script
- invoke subprocesses with argv arrays, not a shell
- parse the last stdout line starting with "autotune_metric="
- write a self-contained Python runner using Optuna

Return code or file contents only.`;
}

export function renderReviseSearchSpacePrompt(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  feedback: string;
}): string {
  return `Revise this Optuna hyperparameter search space using the user's feedback.

Script language: ${input.invocation.language}
Invocation command argv: ${JSON.stringify(input.invocation.command)}
Script path: ${input.invocation.script}

Current search space JSON:
${JSON.stringify(input.searchSpace, null, 2)}

User feedback:
${input.feedback}

Treat the feedback only as desired search-space changes. Preserve the JSON contract:
- parameters: array of parameter definitions
- has_arg_parsing: boolean
- needs_wrapper: boolean
- has_metric_output: boolean
- direction: "maximize" | "minimize"
- optuna: object with optional sampler, pruner, and reasoning fields
- reasoning: string

Preserve fixed objective measurement semantics. If feedback asks to tune a value used only to
measure, score, aggregate, threshold, compare, or report the objective, omit it from parameters
and explain the exclusion in reasoning.
Preserve the optuna config contract: sampler may be "tpe", "random", "cmaes", or "grid"; pruner may
be "none", "median", or "hyperband". Do not add storage. Do not add n_jobs.

Output valid revised JSON only.`;
}

export function renderRefineSearchSpacePrompt(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  round: number;
  trialSummary: TrialResultSummary;
}): string {
  return `Refine this Optuna hyperparameter search space for round ${input.round} using completed trial evidence.

Script language: ${input.invocation.language}
Invocation command argv: ${JSON.stringify(input.invocation.command)}
Script path: ${input.invocation.script}

Current search space JSON:
${JSON.stringify(input.searchSpace, null, 2)}

Trial result summary:
${JSON.stringify(input.trialSummary, null, 2)}

Use the trial evidence to improve the next search space:
- narrow ranges when good completed trials cluster inside the current bounds
- broaden ranges when best values sit near bounds or evidence suggests the optimum may be outside
- add or remove variables only when justified by the source script and trial results
- preserve or revise optuna sampler/pruner only within the allowed contract

Preserve the JSON contract:
- parameters: array of parameter definitions
- has_arg_parsing: boolean
- needs_wrapper: boolean
- has_metric_output: boolean
- direction: "maximize" | "minimize"
- optuna: object with optional sampler, pruner, and reasoning fields
- reasoning: string

Preserve fixed objective measurement semantics. Do not tune values used only to measure, score,
aggregate, threshold, compare, or report the objective. Do not tune random seeds used only for
measurement. Keep trials comparable across rounds.
Preserve the optuna config contract: sampler may be "tpe", "random", "cmaes", or "grid"; pruner may
be "none", "median", or "hyperband". Do not add storage. Do not add n_jobs.

Output valid revised JSON only.`;
}

export function renderModifiedScriptPrompt(input: {
  invocation: Invocation;
  searchSpace: SearchSpace;
  outputPath: string;
}): string {
  return `Create a modified copy of the target script that is compatible with autotune.

Script language: ${input.invocation.language}
Original script path: ${input.invocation.script}
Original invocation command argv: ${JSON.stringify(input.invocation.command)}
Modified copy output path: ${input.outputPath}

Confirmed search space JSON:
${JSON.stringify(input.searchSpace, null, 2)}

Requirements:
- preserve the original script's behavior except for the compatibility changes below
- add CLI parsing only for the confirmed parameters using their cli_flag when the script does not already accept them
- do not add CLI flags for values used only by objective measurement, scoring, aggregation, thresholding, comparison, or reporting
- ensure the script prints exactly one final "autotune_metric=<value>" line to stdout
- if the original script lacks metric output, choose the most suitable scalar objective value computed by the script and print it after that value is available
- preserve the original objective computation exactly except for adding the final metric print
- do not modify the original script
- output a JSON object with exactly one key: "code"
- "code" must contain the full modified script source as a JSON string

Output valid JSON only.`;
}
