Analyze the following script for hyperparameter tuning.

The script language is: {language}
The script is invoked via: {command} {script}
{agent_guidance_block}

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
Use random for 10 or fewer total planned trials because TPESampler defaults to 10 startup trials.
Use tpe for mixed or continuous spaces when the budget exceeds the startup trials and grid is not exhaustive.
Use grid only when all active parameters are small categorical choices and the full combination count fits the trial budget.
Use cmaes only for all-numeric fixed-dimensional spaces with enough trials and no categorical parameters.
Use pruner none unless intermediate metrics can be reported to Optuna with comparable steps across trials.
A final-only autotune_metric does not make median or hyperband pruning useful.
Do not propose storage. Do not propose n_jobs. These are user-controlled resource/state settings.

Output valid JSON only.
