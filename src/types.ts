export type Direction = "maximize" | "minimize";
export type Sampler = "tpe" | "random" | "cmaes" | "grid" | "centaur";
export type Pruner = "none" | "median" | "hyperband";
export type RefineMode = "ask" | "auto";
export type ReasoningEffort = "low" | "medium" | "high" | "xhigh";

export type ParameterType = "float" | "int" | "categorical";

export interface SearchParameter {
  name: string;
  cli_flag: string;
  type: ParameterType;
  low?: number;
  high?: number;
  log?: boolean;
  choices?: ReadonlyArray<string | number | boolean>;
  current_value?: unknown;
}

export interface FixedParameter {
  name: string;
  cli_flag: string;
  value: string | number | boolean;
}

export interface SearchSpace {
  parameters: ReadonlyArray<SearchParameter>;
  fixed_parameters?: ReadonlyArray<FixedParameter>;
  has_arg_parsing: boolean;
  needs_wrapper: boolean;
  has_metric_output?: boolean;
  direction: Direction;
  failure_value?: number;
  optuna?: OptunaConfig;
  reasoning?: string;
}

export interface OptunaConfig {
  sampler?: Sampler;
  pruner?: Pruner;
  centaur?: CentaurConfig;
  reasoning?: string;
}

export interface CentaurConfig {
  llm_probability: number;
  warmup_trials: number;
  seed: number;
}

export interface Invocation {
  language: string;
  command: string[];
  script: string;
  scriptArgument?: "append" | "included" | "none";
}

export interface HeadlessOptions {
  agent: string;
  model?: string;
  reasoningEffort?: ReasoningEffort;
}

export interface RunOptions {
  trials: number;
  direction?: Direction;
  sampler?: Sampler;
  pruner?: Pruner;
  centaur?: Partial<CentaurConfig>;
  nJobs: number;
  workDir?: string;
  agent: string;
  model?: string;
  agentGuidance?: string;
  reasoningEffort?: ReasoningEffort;
  command?: string;
  buildCommand?: string;
  timeoutSeconds?: number;
  timeBudgetSeconds?: number;
  refineRounds?: number;
  refineTrials?: number;
  refineMode?: RefineMode;
  refineTransferFixedParams?: boolean;
  refineTransferTrials?: boolean;
  json: boolean;
  output?: string;
  storage?: string;
  studyName?: string;
  yes: boolean;
  config?: string;
  ask?: (prompt: string) => Promise<string>;
}

export interface SearchBudget {
  trials: number;
  timeoutSeconds?: number;
  refineRounds?: number;
  refineTrials?: number;
  refineMode?: RefineMode;
  currentRefinementRound?: number;
  currentRoundTrials?: number;
}
