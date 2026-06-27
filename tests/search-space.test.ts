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
