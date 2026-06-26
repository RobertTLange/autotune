import type { Invocation, SearchSpace } from "./types.js";

export function renderAnalyzePrompt(input: { invocation: Invocation }): string {
  return `Analyze the following script for hyperparameter tuning.

The script language is: ${input.invocation.language}
The script is invoked via: ${[...input.invocation.command, input.invocation.script].join(" ")}

Identify all tunable hyperparameters and propose Optuna search spaces.
The optimization metric is reported via printing "autotune_metric=<value>" to stdout.

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
- reasoning: why this direction

Output valid JSON only.`;
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
- reasoning: string

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
- add CLI parsing for every parameter using its cli_flag when the script does not already accept it
- ensure the script prints exactly one final "autotune_metric=<value>" line to stdout
- if the original script lacks metric output, choose the most suitable scalar score/loss/accuracy value computed by the script and print it after that value is available
- do not modify the original script
- output a JSON object with exactly one key: "code"
- "code" must contain the full modified script source as a JSON string

Output valid JSON only.`;
}
