import { Command, Option } from "commander";
import { mkdir, mkdtemp, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  createProgram,
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
      refineTransferFixedParams: true,
      refineTransferTrials: true,
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
      timeoutSeconds: 1800,
      refineTransferFixedParams: true,
      refineTransferTrials: true
    });
  });

  it("preserves disabled refinement transfer options", () => {
    const command = new Command()
      .option("--no-refine-transfer-fixed-params")
      .option("--no-refine-transfer-trials");
    command.parse(["--no-refine-transfer-fixed-params", "--no-refine-transfer-trials"], { from: "user" });
    const raw = {
      trials: 3,
      nJobs: 1,
      agent: "claude",
      timeoutSeconds: 1800,
      refineRounds: 1,
      refineMode: "auto",
      ...command.opts(),
      json: false,
      yes: true
    };

    expect(normalizeRunOptions(raw, command)).toMatchObject({
      refineTransferFixedParams: false,
      refineTransferTrials: false
    });
  });

  it("exposes refinement transfer opt-out flags on run", () => {
    const runCommand = createProgram().commands.find((command) => command.name() === "run");
    expect(runCommand?.options.map((option) => option.long)).toEqual(
      expect.arrayContaining(["--no-refine-transfer-fixed-params", "--no-refine-transfer-trials"])
    );
  });

  it("exposes agent guidance options on run and analyze", () => {
    const program = createProgram();
    const runCommand = program.commands.find((command) => command.name() === "run");
    const analyzeCommand = program.commands.find((command) => command.name() === "analyze");

    expect(runCommand?.options.map((option) => option.long)).toEqual(
      expect.arrayContaining(["--agent-guidance", "--agent-guidance-file"])
    );
    expect(analyzeCommand?.options.map((option) => option.long)).toEqual(
      expect.arrayContaining(["--agent-guidance", "--agent-guidance-file"])
    );
  });

  it("combines guidance file and inline guidance for run options", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-guidance-cli-"));
    const guidanceFile = path.join(dir, "guidance.txt");
    await writeFile(guidanceFile, "prefer compact spaces\n", "utf8");
    const command = new Command()
      .option("--agent-guidance <text>")
      .option("--agent-guidance-file <file>");
    command.parse(["--agent-guidance-file", guidanceFile, "--agent-guidance", "avoid batch size"], { from: "user" });
    const raw = {
      trials: 3,
      nJobs: 1,
      agent: "claude",
      refineRounds: 0,
      refineMode: "ask",
      json: false,
      yes: true,
      ...command.opts()
    };

    expect(normalizeRunOptions(raw, command)).toMatchObject({
      agentGuidance: "prefer compact spaces\n\navoid batch size"
    });
  });

  it("rejects empty agent guidance", async () => {
    const command = new Command().option("--agent-guidance <text>");
    command.parse(["--agent-guidance", "   "], { from: "user" });
    const raw = {
      trials: 3,
      nJobs: 1,
      agent: "claude",
      refineRounds: 0,
      refineMode: "ask",
      json: false,
      yes: true,
      ...command.opts()
    };

    expect(() => normalizeRunOptions(raw, command)).toThrow(/agent guidance/i);
  });

  it("rejects missing and empty guidance files", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-guidance-missing-"));
    const emptyFile = path.join(dir, "empty.txt");
    await writeFile(emptyFile, "  \n", "utf8");
    const baseRaw = {
      trials: 3,
      nJobs: 1,
      agent: "claude",
      refineRounds: 0,
      refineMode: "ask",
      json: false,
      yes: true
    };
    const missingCommand = new Command().option("--agent-guidance-file <file>");
    missingCommand.parse(["--agent-guidance-file", path.join(dir, "missing.txt")], { from: "user" });
    const emptyCommand = new Command().option("--agent-guidance-file <file>");
    emptyCommand.parse(["--agent-guidance-file", emptyFile], { from: "user" });

    expect(() => normalizeRunOptions({ ...baseRaw, ...missingCommand.opts() }, missingCommand)).toThrow(/file not found/i);
    expect(() => normalizeRunOptions({ ...baseRaw, ...emptyCommand.opts() }, emptyCommand)).toThrow(/non-empty agent guidance/i);
  });

  it("rejects non-regular, symlinked, and oversized guidance files", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-guidance-invalid-"));
    const guidanceDir = path.join(dir, "guidance-dir");
    const targetFile = path.join(dir, "target.txt");
    const guidanceLink = path.join(dir, "guidance-link.txt");
    const largeFile = path.join(dir, "large.txt");
    await mkdir(guidanceDir);
    await writeFile(targetFile, "prefer compact spaces\n", "utf8");
    await symlink(targetFile, guidanceLink);
    await writeFile(largeFile, "x".repeat(65537), "utf8");
    const baseRaw = {
      trials: 3,
      nJobs: 1,
      agent: "claude",
      refineRounds: 0,
      refineMode: "ask",
      json: false,
      yes: true
    };
    const directoryCommand = new Command().option("--agent-guidance-file <file>");
    directoryCommand.parse(["--agent-guidance-file", guidanceDir], { from: "user" });
    const symlinkCommand = new Command().option("--agent-guidance-file <file>");
    symlinkCommand.parse(["--agent-guidance-file", guidanceLink], { from: "user" });
    const largeCommand = new Command().option("--agent-guidance-file <file>");
    largeCommand.parse(["--agent-guidance-file", largeFile], { from: "user" });

    expect(() => normalizeRunOptions({ ...baseRaw, ...directoryCommand.opts() }, directoryCommand)).toThrow(/regular file/i);
    expect(() => normalizeRunOptions({ ...baseRaw, ...symlinkCommand.opts() }, symlinkCommand)).toThrow(/regular file/i);
    expect(() => normalizeRunOptions({ ...baseRaw, ...largeCommand.opts() }, largeCommand)).toThrow(/65536 bytes/i);
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
