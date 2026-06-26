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
- direction: "maximize" | "minimize"
- reasoning: string

Output valid revised JSON only.`;
}
