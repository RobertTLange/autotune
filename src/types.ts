export type Direction = "maximize" | "minimize";
export type Sampler = "tpe" | "random" | "cmaes" | "grid";
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

export interface SearchSpace {
  parameters: ReadonlyArray<SearchParameter>;
  has_arg_parsing: boolean;
  needs_wrapper: boolean;
  has_metric_output?: boolean;
  direction: Direction;
  optuna?: OptunaConfig;
  reasoning?: string;
}

export interface OptunaConfig {
  sampler?: Sampler;
  pruner?: Pruner;
  reasoning?: string;
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
  nJobs: number;
  workDir: string;
  agent: string;
  model?: string;
  reasoningEffort?: ReasoningEffort;
  command?: string;
  buildCommand?: string;
  refineRounds?: number;
  refineTrials?: number;
  refineMode?: RefineMode;
  json: boolean;
  output?: string;
  storage?: string;
  studyName?: string;
  yes: boolean;
  config?: string;
  ask?: (prompt: string) => Promise<string>;
}
