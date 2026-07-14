import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { parseSearchSpaceText, readSearchSpace, writeSearchSpace } from "../src/search-space.js";

const searchSpace = {
  parameters: [
    { name: "lr", cli_flag: "--lr", type: "float", low: 0.0001, high: 0.1, log: true },
    { name: "batch_size", cli_flag: "--batch-size", type: "int", low: 16, high: 128 },
    { name: "optimizer", cli_flag: "--optimizer", type: "categorical", choices: ["adam", "sgd"] }
  ],
  has_arg_parsing: true,
  needs_wrapper: false,
  has_metric_output: true,
  direction: "maximize",
  reasoning: "accuracy"
} as const;

describe("parseSearchSpaceText", () => {
  it("parses JSON search spaces", () => {
    expect(parseSearchSpaceText(JSON.stringify(searchSpace))).toEqual(searchSpace);
  });

  it("parses YAML search spaces", () => {
    expect(
      parseSearchSpaceText(`
parameters:
  - name: lr
    cli_flag: --lr
    type: float
    low: 0.0001
    high: 0.1
    log: true
has_arg_parsing: true
needs_wrapper: false
has_metric_output: true
direction: maximize
`)
    ).toMatchObject({ direction: "maximize", parameters: [{ name: "lr" }] });
  });

  it("defaults missing metric-output metadata to true for old configs", () => {
    expect(
      parseSearchSpaceText(`
parameters: []
has_arg_parsing: true
needs_wrapper: false
direction: maximize
`)
    ).toMatchObject({ has_metric_output: true });
  });

  it("parses agent-proposed Optuna settings", () => {
    expect(
      parseSearchSpaceText(`
parameters: []
has_arg_parsing: true
needs_wrapper: false
direction: minimize
optuna:
  sampler: random
  pruner: hyperband
  reasoning: broad exploratory search
`)
    ).toMatchObject({
      direction: "minimize",
      optuna: { sampler: "random", pruner: "hyperband", reasoning: "broad exploratory search" }
    });
  });

  it("parses Centaur settings and applies the paper defaults", () => {
    expect(
      parseSearchSpaceText(`
parameters: []
has_arg_parsing: true
needs_wrapper: false
direction: minimize
optuna:
  sampler: centaur
`)
    ).toMatchObject({
      optuna: {
        sampler: "centaur",
        centaur: { llm_probability: 0.3, warmup_trials: 10, seed: 0 }
      }
    });

    expect(
      parseSearchSpaceText(`
parameters: []
has_arg_parsing: true
needs_wrapper: false
direction: minimize
optuna:
  sampler: centaur
  centaur:
    llm_probability: 0.75
    warmup_trials: 4
    seed: 17
`)
    ).toMatchObject({
      optuna: {
        sampler: "centaur",
        centaur: { llm_probability: 0.75, warmup_trials: 4, seed: 17 }
      }
    });
  });

  it.each([
    ["llm_probability: -0.01", /llm_probability|probability/i],
    ["llm_probability: 1.01", /llm_probability|probability/i],
    ["warmup_trials: -1", /warmup_trials|non-negative/i],
    ["warmup_trials: 1.5", /warmup_trials|integer/i],
    ["seed: -1", /seed|non-negative/i],
    ["seed: 1.5", /seed|integer/i],
    ["unknown: true", /unknown|unrecognized/i]
  ])("rejects invalid Centaur setting %s", (setting, expected) => {
    expect(() =>
      parseSearchSpaceText(`
parameters: []
has_arg_parsing: true
needs_wrapper: false
direction: maximize
optuna:
  sampler: centaur
  centaur:
    ${setting}
`)
    ).toThrow(expected);
  });

  it("rejects Centaur settings for another sampler", () => {
    expect(() =>
      parseSearchSpaceText(`
parameters: []
has_arg_parsing: true
needs_wrapper: false
direction: maximize
optuna:
  sampler: tpe
  centaur:
    llm_probability: 0.3
`)
    ).toThrow(/centaur.*sampler|sampler.*centaur/i);
  });

  it("parses fixed parameters", () => {
    expect(
      parseSearchSpaceText(`
parameters:
  - name: lr
    cli_flag: --lr
    type: float
    low: 0.001
    high: 0.1
fixed_parameters:
  - name: batch_size
    cli_flag: --batch-size
    value: 128
has_arg_parsing: true
needs_wrapper: false
direction: maximize
`)
    ).toMatchObject({
      parameters: [{ name: "lr" }],
      fixed_parameters: [{ name: "batch_size", cli_flag: "--batch-size", value: 128 }]
    });
  });

  it("rejects duplicate active and fixed parameter names", () => {
    expect(() =>
      parseSearchSpaceText(`
parameters:
  - name: lr
    cli_flag: --lr
    type: float
    low: 0.001
    high: 0.1
fixed_parameters:
  - name: lr
    cli_flag: --fixed-lr
    value: 0.01
has_arg_parsing: true
needs_wrapper: false
direction: maximize
`)
    ).toThrow(/duplicate parameter name/i);
  });

  it("rejects unsupported Optuna settings", () => {
    expect(() =>
      parseSearchSpaceText(`
parameters: []
has_arg_parsing: true
needs_wrapper: false
direction: maximize
optuna:
  sampler: bayes
  pruner: none
`)
    ).toThrow(/sampler|Invalid/i);
  });

  it("validates parameter bounds", () => {
    expect(() =>
      parseSearchSpaceText(
        JSON.stringify({
          ...searchSpace,
          parameters: [{ name: "bad", cli_flag: "--bad", type: "float", low: 1 }]
        })
      )
    ).toThrow(/high/i);
  });

  it("rejects log-scale float ranges with non-positive lower bounds", () => {
    expect(() =>
      parseSearchSpaceText(
        JSON.stringify({
          ...searchSpace,
          parameters: [{ name: "lr", cli_flag: "--lr", type: "float", low: 0, high: 1, log: true }]
        })
      )
    ).toThrow(/positive/i);
  });

  it("rejects fractional integer bounds", () => {
    expect(() =>
      parseSearchSpaceText(
        JSON.stringify({
          ...searchSpace,
          parameters: [{ name: "layers", cli_flag: "--layers", type: "int", low: 1.5, high: 4 }]
        })
      )
    ).toThrow(/integer/i);
  });

  it("rejects duplicate parameter names and CLI flags", () => {
    expect(() =>
      parseSearchSpaceText(
        JSON.stringify({
          ...searchSpace,
          parameters: [
            { name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 },
            { name: "x", cli_flag: "--x2", type: "float", low: 0, high: 1 }
          ]
        })
      )
    ).toThrow(/duplicate parameter name/i);

    expect(() =>
      parseSearchSpaceText(
        JSON.stringify({
          ...searchSpace,
          parameters: [
            { name: "x", cli_flag: "--x", type: "float", low: 0, high: 1 },
            { name: "y", cli_flag: "--x", type: "float", low: 0, high: 1 }
          ]
        })
      )
    ).toThrow(/duplicate cli_flag/i);
  });
});

describe("search space files", () => {
  it("round-trips YAML files", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-space-"));
    const file = path.join(dir, "search_space.yaml");
    await writeSearchSpace(file, searchSpace);
    expect(await readSearchSpace(file)).toEqual(searchSpace);
    expect(await readFile(file, "utf8")).toContain("parameters:");
  });
});
