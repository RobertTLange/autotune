Generate an Optuna runner for this autotune search.

Script language: {language}
Invocation command argv: {command_argv}
Script path: {script}
Output runner path: {output_path}

Confirmed search space JSON:
{search_space_json}

Requirements:
- preserve the original script
- invoke subprocesses with argv arrays, not a shell
- parse the last stdout line starting with "autotune_metric="
- write a self-contained Python runner using Optuna
