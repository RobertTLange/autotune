Create a modified copy of the target script that accepts the confirmed hyperparameters as CLI arguments.

Script language: {language}
Original script path: {script}
Original invocation command argv: {command_argv}
Modified copy output path: {output_path}

Confirmed search space JSON:
{search_space_json}

Requirements:
- preserve the original script's behavior except for reading the listed hyperparameters from CLI flags
- add CLI parsing for every parameter using its cli_flag
- keep printing "autotune_metric=<value>" to stdout
- do not modify the original script
- output a JSON object with exactly one key: "code"
- "code" must contain the full modified script source as a JSON string

Output valid JSON only.
