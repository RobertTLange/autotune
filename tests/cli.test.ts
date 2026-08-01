import { Command, Option } from "commander";
import { mkdir, mkdtemp, readFile, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  createProgram,
  isMainModule,
  normalizeReasoningEffort,
  normalizeRunOptions,
  PACKAGE_VERSION,
  parseProbability,
  parseNonNegativeInt,
  parsePositiveInt,
  parseSamplerSeed,
  configureExecutableProgram,
  usesSdkFormat
} from "../src/cli.js";
import { SDK_PROTOCOL_VERSION } from "../src/sdk.js";

vi.mock("../src/check.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/check.js")>();
  return {
    ...actual,
    checkDoctorPrerequisites: vi.fn(async () => [
      { name: "headless", status: "fail", detail: "configured headless executable must not be empty" }
    ])
  };
});

describe("CLI option normalization", () => {
  it("exposes versioned SDK capabilities", async () => {
    const output = vi.spyOn(console, "log").mockImplementation(() => undefined);

    await createProgram().parseAsync(["node", "autotune", "capabilities", "--sdk-format", "json"]);

    expect(JSON.parse(String(output.mock.calls[0]?.[0]))).toEqual({
      protocolVersion: SDK_PROTOCOL_VERSION,
      type: "result",
      command: "capabilities",
      exitCode: 0,
      data: {
        protocolVersion: SDK_PROTOCOL_VERSION,
        commands: ["analyze", "doctor", "plot-progress", "results", "resume", "run"]
      }
    });
    output.mockRestore();
  });

  it("exposes the SDK format on supported commands", () => {
    const sdkCommands = ["analyze", "doctor", "plot-progress", "results", "resume", "run"];
    expect(createProgram().commands.filter((command) => sdkCommands.includes(command.name())).every((command) =>
      command.options.some((option) => option.long === "--sdk-format")
    )).toBe(true);
  });

  it("returns typed doctor failures to SDK callers", async () => {
    const output = vi.spyOn(console, "log").mockImplementation(() => undefined);

    try {
      await createProgram().parseAsync(["node", "autotune", "doctor", "--sdk-format", "json"]);
      const response = JSON.parse(String(output.mock.calls.at(-1)?.[0])) as {
        type: string;
        exitCode: number;
        data: Array<{ name: string; status: string; detail: string }>;
      };
      expect(response.type).toBe("result");
      expect(response.exitCode).toBe(0);
      expect(response.data).toContainEqual(expect.objectContaining({ name: "headless", status: "fail" }));
    } finally {
      output.mockRestore();
    }
  });

  it("keeps doctor failures nonzero for human callers", async () => {
    const output = vi.spyOn(console, "log").mockImplementation(() => undefined);

    try {
      await expect(createProgram().parseAsync(["node", "autotune", "doctor"]))
        .rejects.toThrow("prerequisite check failed");
    } finally {
      output.mockRestore();
    }
  });

  it("requires explicit run confirmation for SDK calls", async () => {
    await expect(
      createProgram().parseAsync(["node", "autotune", "run", "train.py", "--trials", "1", "--sdk-format", "json"])
    ).rejects.toThrow("--sdk-format run requires --yes");
  });

  it("recognizes both SDK flag forms in parser errors", () => {
    expect(usesSdkFormat(["node", "autotune", "run", "--sdk-format", "json"])).toBe(true);
    expect(usesSdkFormat(["node", "autotune", "run", "--sdk-format=json"])).toBe(true);
    expect(usesSdkFormat(["node", "autotune", "run"])).toBe(false);
  });

  it("routes SDK subcommand parser failures to the caller", async () => {
    const program = createProgram();
    configureExecutableProgram(program, true);

    await expect(
      program.parseAsync(["node", "autotune", "run", "--sdk-format=json"])
    ).rejects.toThrow("required option '--trials <n>' not specified");
  });

  it("preserves successful version exits for the human CLI", async () => {
    const exit = vi.spyOn(process, "exit").mockImplementation((code) => {
      throw new Error(`process exited with ${String(code)}`);
    });
    const output = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    const program = createProgram();
    configureExecutableProgram(program, false);

    await expect(program.parseAsync(["node", "autotune", "--version"]))
      .rejects.toThrow("process exited with 0");

    expect(exit).toHaveBeenCalledOnce();
    expect(exit).toHaveBeenCalledWith(0);
    expect(output).toHaveBeenCalledWith(`${PACKAGE_VERSION}\n`);
  });

  it("uses package.json as the CLI version source of truth", async () => {
    const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")) as {
      version: string;
    };

    expect(PACKAGE_VERSION).toBe(packageJson.version);
    expect(createProgram().version()).toBe(packageJson.version);
  });

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
      timeBudgetSeconds: 86400,
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
      timeBudgetSeconds: 86400,
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

  it("exposes Centaur sampler and override flags on run", () => {
    const runCommand = createProgram().commands.find((command) => command.name() === "run");
    const sampler = runCommand?.options.find((option) => option.long === "--sampler") as Option | undefined;

    expect(sampler?.argChoices).toContain("centaur");
    expect(runCommand?.options.map((option) => option.long)).toEqual(
      expect.arrayContaining([
        "--centaur-llm-probability",
        "--centaur-warmup-trials",
        "--centaur-seed",
        "--sampler-seed"
      ])
    );
  });

  it("normalizes an explicit sampler seed", () => {
    const command = new Command().option("--sampler-seed <n>", "sampler seed", parseNonNegativeInt);
    command.parse(["--sampler-seed", "7"], { from: "user" });

    expect(normalizeRunOptions({
      trials: 3,
      nJobs: 1,
      agent: "claude",
      refineRounds: 0,
      refineMode: "ask",
      json: false,
      yes: true,
      ...command.opts()
    }, command)).toMatchObject({ samplerSeed: 7 });
  });

  it("bounds sampler seeds to the uint32 range", () => {
    expect(parseSamplerSeed("4294967295")).toBe(4294967295);
    expect(() => parseSamplerSeed("4294967296")).toThrow(/sampler seed/i);
  });

  it("normalizes explicit Centaur CLI overrides without injecting defaults", () => {
    const command = new Command()
      .addOption(new Option("--sampler <sampler>").choices(["tpe", "centaur"]))
      .option("--centaur-llm-probability <probability>", "LLM probability", parseProbability)
      .option("--centaur-warmup-trials <n>", "warmup trials", parseNonNegativeInt)
      .option("--centaur-seed <n>", "scheduler seed", parseNonNegativeInt);
    command.parse([
      "--sampler", "centaur",
      "--centaur-llm-probability", "0.65",
      "--centaur-seed", "42"
    ], { from: "user" });

    const options = normalizeRunOptions({
      trials: 3,
      nJobs: 1,
      agent: "claude",
      refineRounds: 0,
      refineMode: "ask",
      json: false,
      yes: true,
      ...command.opts()
    }, command);

    expect(options).toMatchObject({
      sampler: "centaur",
      centaur: { llm_probability: 0.65, seed: 42 }
    });
    expect(options.centaur).not.toHaveProperty("warmup_trials");
  });

  it("rejects Centaur overrides with an explicitly different sampler", () => {
    const command = new Command()
      .addOption(new Option("--sampler <sampler>").choices(["tpe", "centaur"]))
      .option("--centaur-warmup-trials <n>", "warmup trials", parseNonNegativeInt);
    command.parse(["--sampler", "tpe", "--centaur-warmup-trials", "2"], { from: "user" });

    expect(() => normalizeRunOptions({
      trials: 3,
      nJobs: 1,
      agent: "claude",
      refineRounds: 0,
      refineMode: "ask",
      json: false,
      yes: true,
      ...command.opts()
    }, command)).toThrow(/centaur.*sampler|sampler.*centaur/i);
  });

  it("exposes agent guidance options on run and analyze", () => {
    const program = createProgram();
    const runCommand = program.commands.find((command) => command.name() === "run");
    const analyzeCommand = program.commands.find((command) => command.name() === "analyze");

    expect(runCommand?.options.map((option) => option.long)).toEqual(
      expect.arrayContaining(["--agent-guidance", "--agent-guidance-file", "--max-parameters"])
    );
    expect(analyzeCommand?.options.map((option) => option.long)).toEqual(
      expect.arrayContaining(["--agent-guidance", "--agent-guidance-file", "--max-parameters"])
    );
  });

  it("normalizes an explicit maximum parameter count", () => {
    const command = new Command().option("--max-parameters <n>", "maximum active parameters", parsePositiveInt);
    command.parse(["--max-parameters", "3"], { from: "user" });

    expect(normalizeRunOptions({
      trials: 3,
      nJobs: 1,
      agent: "claude",
      refineRounds: 0,
      refineMode: "ask",
      json: false,
      yes: true,
      ...command.opts()
    }, command)).toMatchObject({ maxParameters: 3 });
  });

  it("exposes model selection on doctor", () => {
    const doctorCommand = createProgram().commands.find((command) => command.name() === "doctor");

    expect(doctorCommand?.options.map((option) => option.long)).toContain("--model");
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

  it("validates probability options including boundaries", () => {
    expect(parseProbability("0")).toBe(0);
    expect(parseProbability("0.3")).toBe(0.3);
    expect(parseProbability("1")).toBe(1);
    expect(() => parseProbability("-0.01")).toThrow(/between 0 and 1/i);
    expect(() => parseProbability("1.01")).toThrow(/between 0 and 1/i);
    expect(() => parseProbability("NaN")).toThrow(/between 0 and 1/i);
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
