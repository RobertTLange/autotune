export type Direction = "maximize" | "minimize";

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
  direction: Direction;
  reasoning?: string;
}

export interface Invocation {
  language: string;
  command: string[];
  script: string;
}

export interface RunOptions {
  trials: number;
  direction: Direction;
  sampler: string;
  pruner: string;
  nJobs: number;
  workDir: string;
  agent: string;
  command?: string;
  json: boolean;
  output?: string;
  storage?: string;
  yes: boolean;
  config?: string;
}
