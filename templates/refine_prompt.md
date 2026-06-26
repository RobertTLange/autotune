Refine this Optuna hyperparameter search space for round {round} using completed trial evidence.

Script language: {language}
Invocation command argv: {command_argv}
Script path: {script}

Current search space JSON:
{search_space_json}

Trial result summary:
{trial_summary_json}

Use the trial evidence to improve the next search space:
- narrow ranges when good completed trials cluster inside the current bounds
- broaden ranges when best values sit near bounds or evidence suggests the optimum may be outside
- add or remove variables only when justified by the source script and trial results
- preserve or revise optuna sampler/pruner only within the allowed contract

Preserve the search-space JSON contract. Preserve fixed objective measurement semantics. Do not tune
values used only to measure, score, aggregate, threshold, compare, or report the objective. Do not tune
random seeds used only for measurement. Do not add storage. Do not add n_jobs.
Output valid revised JSON only.
