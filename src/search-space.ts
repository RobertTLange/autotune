import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import YAML from "yaml";
import { z } from "zod";
import type { SearchSpace } from "./types.js";

const directionSchema = z.enum(["maximize", "minimize"]);
const samplerSchema = z.enum(["tpe", "random", "cmaes", "grid", "centaur"]);
const prunerSchema = z.enum(["none", "median", "hyperband"]);
const primitiveSchema = z.union([z.string(), z.number(), z.boolean()]);

const centaurConfigSchema = z
  .object({
    llm_probability: z.number().min(0).max(1).default(0.3),
    warmup_trials: z.number().int().nonnegative().default(10),
    seed: z.number().int().nonnegative().default(0)
  })
  .strict();

const optunaConfigSchema = z
  .object({
    sampler: samplerSchema.optional(),
    pruner: prunerSchema.optional(),
    seed: z.number().int().min(0).max(0xffffffff).optional(),
    centaur: centaurConfigSchema.optional(),
    reasoning: z.string().optional()
  })
  .strict()
  .superRefine((optuna, ctx) => {
    if (optuna.centaur !== undefined && optuna.sampler !== "centaur") {
      ctx.addIssue({
        code: "custom",
        path: ["centaur"],
        message: "centaur settings require the centaur sampler"
      });
    }
    if (optuna.seed !== undefined && optuna.sampler === "centaur") {
      ctx.addIssue({
        code: "custom",
        path: ["seed"],
        message: "root sampler seed is not used by Centaur; configure centaur.seed instead"
      });
    }
  })
  .transform((optuna) =>
    optuna.sampler === "centaur"
      ? { ...optuna, centaur: optuna.centaur ?? centaurConfigSchema.parse({}) }
      : optuna
  );

const parameterSchema = z
  .object({
    name: z.string().min(1),
    cli_flag: z.string().regex(/^--[A-Za-z0-9][A-Za-z0-9-]*$/),
    type: z.enum(["float", "int", "categorical"]),
    low: z.number().optional(),
    high: z.number().optional(),
    log: z.boolean().optional(),
    choices: z.array(primitiveSchema).optional(),
    current_value: z.unknown().optional()
  })
  .superRefine((parameter, ctx) => {
    if (parameter.type === "float" || parameter.type === "int") {
      if (typeof parameter.low !== "number") {
        ctx.addIssue({ code: "custom", message: `${parameter.name} missing low` });
      }
      if (typeof parameter.high !== "number") {
        ctx.addIssue({ code: "custom", message: `${parameter.name} missing high` });
      }
      if (
        typeof parameter.low === "number" &&
        typeof parameter.high === "number" &&
        parameter.low >= parameter.high
      ) {
        ctx.addIssue({ code: "custom", message: `${parameter.name} low must be less than high` });
      }
      if (parameter.type === "int") {
        if (typeof parameter.low === "number" && !Number.isInteger(parameter.low)) {
          ctx.addIssue({ code: "custom", message: `${parameter.name} low must be an integer` });
        }
        if (typeof parameter.high === "number" && !Number.isInteger(parameter.high)) {
          ctx.addIssue({ code: "custom", message: `${parameter.name} high must be an integer` });
        }
      }
      if (parameter.type === "float" && parameter.log === true && typeof parameter.low === "number" && parameter.low <= 0) {
        ctx.addIssue({ code: "custom", message: `${parameter.name} log-scale low must be positive` });
      }
    }
    if (parameter.type === "categorical" && (!parameter.choices || parameter.choices.length === 0)) {
      ctx.addIssue({ code: "custom", message: `${parameter.name} missing choices` });
    }
  });

const fixedParameterSchema = z.object({
  name: z.string().min(1),
  cli_flag: z.string().regex(/^--[A-Za-z0-9][A-Za-z0-9-]*$/),
  value: primitiveSchema
});

const searchSpaceSchema = z
  .object({
    parameters: z.array(parameterSchema),
    fixed_parameters: z.array(fixedParameterSchema).optional(),
    has_arg_parsing: z.boolean(),
    needs_wrapper: z.boolean(),
    has_metric_output: z.boolean().default(true),
    direction: directionSchema,
    failure_value: z.number().optional(),
    optuna: optunaConfigSchema.optional(),
    reasoning: z.string().optional()
  })
  .superRefine((searchSpace, ctx) => {
    const allParameters = [...searchSpace.parameters, ...(searchSpace.fixed_parameters ?? [])];
    addDuplicateIssue(allParameters, "name", "parameter name", ctx);
    addDuplicateIssue(allParameters, "cli_flag", "cli_flag", ctx);
  });

function addDuplicateIssue(
  parameters: Array<{ name: string; cli_flag: string }>,
  key: "name" | "cli_flag",
  label: string,
  ctx: z.RefinementCtx
): void {
  const seen = new Map<string, number>();
  parameters.forEach((parameter, index) => {
    const value = parameter[key];
    const firstIndex = seen.get(value);
    if (firstIndex === undefined) {
      seen.set(value, index);
      return;
    }
    ctx.addIssue({
      code: "custom",
      path: ["parameters", index, key],
      message: `duplicate ${label}: ${value}`
    });
  });
}

export function parseSearchSpaceText(text: string): SearchSpace {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("search space is empty");
  }

  const parsed = trimmed.startsWith("{") || trimmed.startsWith("[") ? JSON.parse(trimmed) : YAML.parse(trimmed);
  return searchSpaceSchema.parse(parsed);
}

export async function readSearchSpace(filePath: string): Promise<SearchSpace> {
  return parseSearchSpaceText(await readFile(filePath, "utf8"));
}

export async function writeSearchSpace(filePath: string, searchSpace: SearchSpace): Promise<void> {
  const validated = searchSpaceSchema.parse(searchSpace);
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, YAML.stringify(validated), "utf8");
}

export function validateSearchSpaceParameterLimit(
  searchSpace: SearchSpace,
  maxParameters: number | undefined
): void {
  if (maxParameters === undefined || searchSpace.parameters.length <= maxParameters) {
    return;
  }
  throw new Error(
    `Search space has ${searchSpace.parameters.length} active parameters, exceeding --max-parameters ${maxParameters}`
  );
}

export async function correctSearchSpaceParameterLimit(
  searchSpace: SearchSpace,
  maxParameters: number | undefined,
  revise: (searchSpace: SearchSpace, feedback: string) => Promise<SearchSpace>
): Promise<SearchSpace> {
  if (maxParameters === undefined || searchSpace.parameters.length <= maxParameters) {
    return searchSpace;
  }
  const corrected = await revise(
    searchSpace,
    `The proposed search space has ${searchSpace.parameters.length} active parameters. ` +
      `Reduce the parameters array to at most ${maxParameters} highest-impact active parameters. ` +
      "The fixed_parameters array does not count toward this limit."
  );
  validateSearchSpaceParameterLimit(corrected, maxParameters);
  return corrected;
}
