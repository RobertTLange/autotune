Analyze the following script for hyperparameter tuning.

The script language is: {language}
The script is invoked via: {command} {script}

Identify all tunable hyperparameters and propose Optuna search spaces.
The optimization metric is reported via printing "autotune_metric=<value>" to stdout.
Report whether the script already prints that metric line as has_metric_output.

A tunable hyperparameter should change candidate behavior before the objective is measured.
Do not include parameters that change objective measurement, including the evaluation input set,
scoring formula, aggregation window, reporting threshold, comparison baseline, output formatting,
or random seed used only for measurement. Leave these values fixed so trials remain comparable.
If you intentionally exclude an important metric or evaluation value, mention it in reasoning.

Also propose safe optuna settings:
- sampler: "tpe" | "random" | "cmaes" | "grid"
- pruner: "none" | "median" | "hyperband"
- reasoning: short explanation for the Optuna choices
Prefer tpe for mixed or continuous spaces, random for tiny exploratory searches, grid only when all
parameters are small categorical choices, and cmaes only for continuous numeric spaces. Prefer none
for pruner unless the script is iterative and pruning is likely comparable across trials.
Do not propose storage. Do not propose n_jobs. These are user-controlled resource/state settings.

Output valid JSON only.
