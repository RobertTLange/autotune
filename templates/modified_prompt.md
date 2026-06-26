Create a modified copy of the target script that is compatible with autotune.

Script language: {language}
Original script path: {script}
Original invocation command argv: {command_argv}
Modified copy output path: {output_path}

Confirmed search space JSON:
{search_space_json}

Requirements:
- preserve the original script's behavior except for the compatibility changes below
- add CLI parsing for every parameter using its cli_flag when the script does not already accept it
- ensure the script prints exactly one final "autotune_metric=<value>" line to stdout
- if the original script lacks metric output, choose the most suitable scalar score/loss/accuracy value computed by the script and print it after that value is available
- do not modify the original script
- output a JSON object with exactly one key: "code"
- "code" must contain the full modified script source as a JSON string

Output valid JSON only.
