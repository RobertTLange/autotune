import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import YAML from "yaml";
import { z } from "zod";
import type { SearchSpace } from "./types.js";

const directionSchema = z.enum(["maximize", "minimize"]);
const samplerSchema = z.enum(["tpe", "random", "cmaes", "grid"]);
const prunerSchema = z.enum(["none", "median", "hyperband"]);
const primitiveSchema = z.union([z.string(), z.number(), z.boolean()]);

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
    }
    if (parameter.type === "categorical" && (!parameter.choices || parameter.choices.length === 0)) {
      ctx.addIssue({ code: "custom", message: `${parameter.name} missing choices` });
    }
  });

const searchSpaceSchema = z.object({
  parameters: z.array(parameterSchema),
  has_arg_parsing: z.boolean(),
  needs_wrapper: z.boolean(),
  has_metric_output: z.boolean().default(true),
  direction: directionSchema,
  optuna: z
    .object({
      sampler: samplerSchema.optional(),
      pruner: prunerSchema.optional(),
      reasoning: z.string().optional()
    })
    .strict()
    .optional(),
  reasoning: z.string().optional()
});

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
