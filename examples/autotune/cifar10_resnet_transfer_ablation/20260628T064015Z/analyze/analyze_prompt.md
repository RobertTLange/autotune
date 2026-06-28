Analyze the following script for hyperparameter tuning.

The script language is: python
The script is invoked via: python3 /home/katya/projects/autotune/examples/cifar10_resnet.py


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
- fixed_parameters: optional array of parameters to pass as fixed CLI values, each with name, cli_flag, and value
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

Output valid JSON only.