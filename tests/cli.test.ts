import { Command, Option } from "commander";
import {
  normalizeReasoningEffort,
  normalizeRunOptions,
  parseNonNegativeInt,
  parsePositiveInt
} from "../src/cli.js";

describe("CLI option normalization", () => {
  it("uses --effort as an alias for --reasoning-effort", () => {
    expect(normalizeReasoningEffort({ effort: "high" })).toBe("high");
    expect(normalizeReasoningEffort({ reasoningEffort: "xhigh", effort: "low" })).toBe("xhigh");
  });

  it("only preserves explicit Optuna overrides and study names", () => {
    const command = new Command()
      .option("--direction <direction>")
      .option("--sampler <sampler>")
      .option("--pruner <pruner>");
    command.parse(["--direction", "minimize"], { from: "user" });
    const raw = {
      trials: 3,
      direction: "minimize",
      sampler: "tpe",
      pruner: "none",
      nJobs: 1,
      workDir: ".autotune",
      agent: "claude",
      timeoutSeconds: 1800,
      refineRounds: 0,
      refineMode: "ask",
      json: false,
      storage: "sqlite:///study.db",
      studyName: "custom",
      yes: true
    };

    expect(normalizeRunOptions(raw, command)).toMatchObject({
      direction: "minimize",
      sampler: undefined,
      pruner: undefined,
      storage: "sqlite:///study.db",
      studyName: "custom",
      timeoutSeconds: 1800
    });
  });

  it("validates integer options", () => {
    expect(parsePositiveInt("3")).toBe(3);
    expect(parseNonNegativeInt("0")).toBe(0);
    expect(() => parsePositiveInt("0")).toThrow(/positive integer/);
    expect(() => parseNonNegativeInt("-1")).toThrow(/non-negative integer/);
  });
});
