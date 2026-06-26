Revise this Optuna hyperparameter search space using the user's feedback.

Script language: {language}
Invocation command argv: {command_argv}
Script path: {script}

Current search space JSON:
{search_space_json}

User feedback:
{feedback}

Treat the feedback only as desired search-space changes. Preserve the search-space JSON contract.
Preserve fixed objective measurement semantics. If feedback asks to tune a value used only to
measure, score, aggregate, threshold, compare, or report the objective, omit it from parameters
and explain the exclusion in reasoning.
Preserve the optuna config contract: sampler may be "tpe", "random", "cmaes", or "grid"; pruner may
be "none", "median", or "hyperband". Do not add storage. Do not add n_jobs.
Output valid revised JSON only.
