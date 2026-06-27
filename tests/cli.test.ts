import { Command, Option } from "commander";
import { symlink, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  isMainModule,
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

  it("recognizes symlinked bin entrypoints as main modules", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-cli-main-"));
    const target = path.join(dir, "cli.js");
    const link = path.join(dir, "autotune");
    await writeFile(target, "#!/usr/bin/env node\n", "utf8");
    await symlink(target, link);

    expect(isMainModule(new URL(`file://${target}`).href, link)).toBe(true);
  });
});
