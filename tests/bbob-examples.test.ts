import { chmod, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { spawn, spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";

const BBOB_DIR = path.join("examples", "bbob");
const BENCHMARK = path.join(BBOB_DIR, "benchmark.py");
const EXPERIMENTS = path.join(BBOB_DIR, "run_experiments.sh");
const PROCESS_SUPERVISOR = path.join(BBOB_DIR, "supervise_process.py");
const OBJECTIVES = ["sphere", "ellipsoid", "rosenbrock", "rastrigin"] as const;

describe("BBOB examples", () => {
  it.each([
    ["sphere", ["1", "2", "3", "4", "5"], 55],
    ["ellipsoid", ["1", "0", "0", "0", "1"], 1_000_001],
    ["ellipsoid", ["0", "1", "0", "0", "0"], Math.sqrt(1_000)],
    ["rosenbrock", ["1", "1", "1", "1", "1"], 0],
    ["rosenbrock", ["0", "0", "0", "0", "0"], 4],
    ["rastrigin", ["0", "0", "0", "0", "0"], 0]
  ])("evaluates the %s objective", async (objective, coordinates, expected) => {
    const result = await runCommand("python3", [
      BENCHMARK,
      "--function", objective,
      ...coordinates.flatMap((coordinate, index) => [`--x${index + 1}`, coordinate])
    ], {});

    expect(result.code).toBe(0);
    expect(readMetric(result.stdout)).toBeCloseTo(expected, 10);
  });

  it.each(OBJECTIVES)("ships a search space for %s", async (objective) => {
    const { parseSearchSpaceText } = await import("../src/search-space.js");
    const text = await readFile(path.join(BBOB_DIR, `${objective}_search_space.yaml`), "utf8");
    const searchSpace = parseSearchSpaceText(text);

    expect(searchSpace.parameters.map((parameter) => parameter.name)).toEqual(["x1", "x2", "x3", "x4", "x5"]);
    expect(searchSpace.fixed_parameters).toEqual([
      { name: "function", cli_flag: "--function", value: objective }
    ]);
    expect(searchSpace.direction).toBe("minimize");
  });

  it.each(["nan", "inf", "5.0001"])("rejects invalid coordinate %s", async (coordinate) => {
    const result = await runCommand("python3", [BENCHMARK, "--x5", coordinate], {});

    expect(result.code).not.toBe(0);
    expect(result.stdout).not.toContain("autotune_metric=");
  });

  it("runs four variants across all objectives for each seed", async () => {
    const fixture = await createLauncherFixture();
    const result = await runCommand("bash", [EXPERIMENTS], fixture.env, 25_000);

    expect(result.code).toBe(0);
    for (const seed of [0, 1]) {
      for (const objective of OBJECTIVES) {
        const baseline = await readExperimentArgv(fixture.outputDir, objective, "01_base_cmaes", seed);
        const noTransfer = await readExperimentArgv(fixture.outputDir, objective, "02_reset_no_transfer", seed);
        const withTransfer = await readExperimentArgv(fixture.outputDir, objective, "03_reset_with_transfer", seed);
        const centaur = await readExperimentArgv(fixture.outputDir, objective, "04_centaur", seed);
        const config = path.resolve(BBOB_DIR, `${objective}_search_space.yaml`);

        expectFlagValues(baseline, {
          "--config": config, "--sampler": "cmaes", "--sampler-seed": String(seed),
          "--trials": "4", "--refine-rounds": "0", "--n-jobs": "1"
        });
        expectFlagValues(noTransfer, {
          "--sampler": "cmaes", "--sampler-seed": String(seed),
          "--trials": "2", "--refine-rounds": "1", "--refine-trials": "2"
        });
        expect(noTransfer).toEqual(expect.arrayContaining([
          "--no-refine-transfer-fixed-params", "--no-refine-transfer-trials"
        ]));
        expectFlagValues(withTransfer, {
          "--sampler": "cmaes", "--sampler-seed": String(seed),
          "--trials": "2", "--refine-rounds": "1", "--refine-trials": "2"
        });
        expect(withTransfer).toContain("--no-refine-transfer-fixed-params");
        expect(withTransfer).not.toContain("--no-refine-transfer-trials");
        expectFlagValues(centaur, {
          "--sampler": "centaur",
          "--trials": "4", "--n-jobs": "1", "--refine-rounds": "0",
          "--centaur-llm-probability": "0.3", "--centaur-warmup-trials": "1", "--centaur-seed": String(seed)
        });
        expect(centaur).not.toContain("--sampler-seed");
        expectFlagValues(centaur, {
          "--study-name": `bbob_${objective}_04_centaur_seed_${seed}`
        });
      }
    }
    expect(result.stdout).toContain("budget: 2 + 1 x 2 = 4");
    expect(result.stdout).toContain("seed 1/2");
    expect(result.stdout).toContain("seed 2/2");
  }, 30_000);

  it("defaults to ten seeds and 100 evaluations per experiment", async () => {
    const launcher = await readFile(EXPERIMENTS, "utf8");

    expect(launcher).toContain('SEED_COUNT="${SEED_COUNT:-10}"');
    expect(launcher).toContain('TOTAL_TRIALS="${TOTAL_TRIALS:-100}"');
    expect(launcher).toContain('REFINE_INITIAL_TRIALS="${REFINE_INITIAL_TRIALS:-50}"');
    expect(launcher).toContain('REFINE_ROUNDS="${REFINE_ROUNDS:-2}"');
    expect(launcher).toContain('REFINE_TRIALS="${REFINE_TRIALS:-25}"');
  });

  it("fails when any experiment fails", async () => {
    const fixture = await createLauncherFixture({ failVariant: "04_centaur" });
    const result = await runCommand("bash", [EXPERIMENTS], fixture.env);

    expect(result.code).not.toBe(0);
  }, 15_000);

  it("reaps process-group descendants after the command exits", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-bbob-supervisor-"));
    const descendantPidFile = path.join(dir, "descendant.pid");
    const command = [
      "bash -c 'trap \"\" TERM; exec sleep 60' &",
      "child=$!",
      "printf '%s\\n' \"$child\" > \"$DESCENDANT_PID_FILE\""
    ].join("\n");

    const result = await runCommand("python3", [
      PROCESS_SUPERVISOR,
      "--grace-seconds", "1",
      "--",
      "bash", "-c", command
    ], { DESCENDANT_PID_FILE: descendantPidFile }, 8_000);

    expect(result.code, result.stderr).toBe(0);
    const descendantPid = Number((await readFile(descendantPidFile, "utf8")).trim());
    await expectProcessStopped(descendantPid);
  }, 10_000);

  it("force-kills a TERM-resistant process-group leader", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-bbob-supervisor-term-"));
    const processPidFile = path.join(dir, "processes.pid");
    const command = [
      "trap '' TERM",
      "bash -c 'trap \"\" TERM; exec sleep 60' &",
      "descendant=$!",
      "printf '%s %s\\n' \"$$\" \"$descendant\" > \"$PROCESS_PID_FILE\"",
      "while true; do sleep 1; done"
    ].join("\n");
    const supervisor = spawn("python3", [
      PROCESS_SUPERVISOR,
      "--grace-seconds", "1",
      "--",
      "bash", "-c", command
    ], {
      env: { ...process.env, PROCESS_PID_FILE: processPidFile },
      stdio: ["ignore", "pipe", "pipe"]
    });
    const completion = collectChild(supervisor);
    await waitForFile(processPidFile);
    const [leader, descendant] = (await readFile(processPidFile, "utf8")).trim().split(" ").map(Number);

    supervisor.kill("SIGTERM");
    const result = await completion;

    expect(result.code, result.stderr).toBe(143);
    await expectProcessStopped(leader);
    await expectProcessStopped(descendant);
  }, 10_000);

  it.skipIf(process.platform !== "linux")("reaps descendants that escape into a new session", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-bbob-supervisor-setsid-"));
    const descendantPidFile = path.join(dir, "descendant.pid");
    const command = [
      "setsid bash -c 'trap \"\" TERM; exec sleep 60' &",
      "descendant=$!",
      "printf '%s\\n' \"$descendant\" > \"$DESCENDANT_PID_FILE\""
    ].join("\n");

    const result = await runCommand("python3", [
      PROCESS_SUPERVISOR,
      "--grace-seconds", "1",
      "--",
      "bash", "-c", command
    ], { DESCENDANT_PID_FILE: descendantPidFile }, 8_000);

    expect(result.code, result.stderr).toBe(0);
    const descendantPid = Number((await readFile(descendantPidFile, "utf8")).trim());
    await expectProcessStopped(descendantPid);
  }, 10_000);

  it("rejects hostile trial counts before shell arithmetic", async () => {
    const fixture = await createLauncherFixture();
    const marker = path.join(path.dirname(fixture.outputDir), "injected");
    const result = await runCommand("bash", [EXPERIMENTS], {
      ...fixture.env,
      TOTAL_TRIALS: `1;touch ${marker}`
    });

    expect(result.code).toBe(2);
    await expect(readFile(marker, "utf8")).rejects.toThrow();
  });

  it("rejects non-canonical seed counts", async () => {
    const fixture = await createLauncherFixture();
    const result = await runCommand("bash", [EXPERIMENTS], { ...fixture.env, SEED_COUNT: "02" });

    expect(result.code).toBe(2);
    expect(result.stderr).toContain("SEED_COUNT must be a canonical non-negative integer");
  });

  it("terminates active experiment descendants on SIGTERM", async () => {
    const fixture = await createLauncherFixture({ longRunningName: "sphere/01_base_cmaes/seed_0" });
    const child = spawn("bash", [EXPERIMENTS], {
      env: { ...process.env, ...fixture.env },
      stdio: ["ignore", "pipe", "pipe"]
    });
    const completion = collectChild(child);
    await waitForFile(fixture.childPidFile);
    const [pid, leaderGroup, descendantGroup] = (await readFile(fixture.childPidFile, "utf8")).trim().split(" ");
    const descendantPid = Number(pid);

    if (fixture.hasSetsid) {
      expect(descendantGroup).not.toBe(leaderGroup);
    } else {
      expect(descendantGroup).toBe(leaderGroup);
    }

    child.kill("SIGTERM");
    const result = await completion;

    expect(result.code).toBe(143);
    await expectProcessStopped(descendantPid);
  }, 10_000);
});

function runCommand(
  command: string,
  args: string[],
  env: Record<string, string>,
  timeoutMs = 12_000
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { env: { ...process.env, ...env }, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let forceKill: ReturnType<typeof setTimeout> | undefined;
    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      forceKill = setTimeout(() => child.kill("SIGKILL"), 5_000);
    }, timeoutMs);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", (error) => {
      clearTimeout(timeout);
      if (forceKill !== undefined) clearTimeout(forceKill);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timeout);
      if (forceKill !== undefined) clearTimeout(forceKill);
      if (timedOut) {
        reject(new Error(`${command} timed out after ${timeoutMs}ms`));
      } else {
        resolve({ code, stdout, stderr });
      }
    });
  });
}

async function createLauncherFixture(options: {
  failVariant?: string;
  longRunningName?: string;
} = {}): Promise<{
  childPidFile: string;
  env: Record<string, string>;
  hasSetsid: boolean;
  outputDir: string;
}> {
  const dir = await mkdtemp(path.join(tmpdir(), "autotune-bbob-launcher-"));
  const binDir = path.join(dir, "bin");
  const outputDir = path.join(dir, "results");
  const syncDir = path.join(dir, "sync");
  const childPidFile = path.join(dir, "child.pid");
  await mkdir(binDir, { recursive: true });
  await mkdir(syncDir, { recursive: true });
  const fakeNode = path.join(binDir, "node");
  const fakeCli = path.join(dir, "cli.js");
  await writeFile(fakeNode, fakeNodeScript(), "utf8");
  await writeFile(fakeCli, "// fake\n", "utf8");
  await chmod(fakeNode, 0o755);
  return {
    childPidFile,
    hasSetsid: spawnSync("bash", ["-c", "command -v setsid"], { encoding: "utf8" }).status === 0,
    outputDir,
    env: {
      PATH: `${binDir}:${process.env.PATH ?? ""}`,
      OUT_ROOT: outputDir,
      RUN_GROUP: "test-run",
      TOTAL_TRIALS: "4",
      REFINE_INITIAL_TRIALS: "2",
      REFINE_ROUNDS: "1",
      REFINE_TRIALS: "2",
      CENTAUR_WARMUP_TRIALS: "1",
      BBOB_SYNC_DIR: syncDir,
      AUTOTUNE_CLI: fakeCli,
      BUILD: "0",
      TERMINATION_GRACE_SECONDS: "1",
      SEED_COUNT: "2",
      ...(options.failVariant ? { FAIL_VARIANT: options.failVariant } : {}),
      ...(options.longRunningName ? {
        LONG_RUNNING_NAME: options.longRunningName,
        CHILD_PID_FILE: childPidFile
      } : {})
    }
  };
}

function fakeNodeScript(): string {
  return [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "original=(\"$@\")",
    "work_dir=",
    "while (($#)); do",
    "  if [[ \"$1\" == \"--work-dir\" ]]; then work_dir=\"$2\"; shift 2; else shift; fi",
    "done",
    "seed=$(basename \"$work_dir\")",
    "variant=$(basename \"$(dirname \"$work_dir\")\")",
    "objective=$(basename \"$(dirname \"$(dirname \"$work_dir\")\")\")",
    "sync_prefix=$BBOB_SYNC_DIR/$variant-$seed",
    "touch \"$sync_prefix-$objective\"",
    "for _ in {1..200}; do",
    "  count=$(find \"$BBOB_SYNC_DIR\" -type f -name \"$variant-$seed-*\" | wc -l)",
    "  if [[ \"$count\" -eq 4 ]]; then break; fi",
    "  sleep 0.01",
    "done",
    "[[ \"$count\" -eq 4 ]] || exit 97",
    "printf '%s\\n' \"${original[@]}\" > \"$work_dir/argv.txt\"",
    "if [[ \"${LONG_RUNNING_NAME:-}\" == \"$objective/$variant/$seed\" ]]; then",
    "  if command -v setsid >/dev/null 2>&1; then",
    "    setsid bash -c 'trap \"\" TERM; exec sleep 60' &",
    "  else",
    "    bash -c 'trap \"\" TERM; exec sleep 60' &",
    "  fi",
    "  child=$!",
    "  leader_group=$(ps -o pgid= -p $$ | tr -d ' ')",
    "  child_group=",
    "  for _ in {1..200}; do",
    "    child_group=$(ps -o pgid= -p \"$child\" | tr -d ' ')",
    "    if [[ -n \"$child_group\" ]]; then break; fi",
    "    sleep 0.01",
    "  done",
    "  [[ -n \"$child_group\" ]] || exit 98",
    "  printf '%s %s %s\\n' \"$child\" \"$leader_group\" \"$child_group\" > \"$CHILD_PID_FILE\"",
    "  trap 'exit 0' TERM",
    "  wait \"$child\"",
    "fi",
    "[[ \"${FAIL_VARIANT:-}\" != \"$variant\" ]] || exit 19"
  ].join("\n");
}

async function readExperimentArgv(outputDir: string, objective: string, variant: string, seed: number): Promise<string[]> {
  const text = await readFile(path.join(outputDir, objective, variant, `seed_${seed}`, "argv.txt"), "utf8");
  return text.trim().split("\n");
}

function expectFlagValues(argv: string[], expected: Record<string, string>): void {
  for (const [flag, value] of Object.entries(expected)) {
    const index = argv.indexOf(flag);
    expect(index).toBeGreaterThanOrEqual(0);
    expect(argv[index + 1]).toBe(value);
  }
}

function readMetric(output: string): number {
  const match = output.match(/^autotune_metric=(.+)$/m);
  if (!match) {
    throw new Error(`Missing autotune metric in output: ${output}`);
  }
  return Number(match[1]);
}

function collectChild(child: ReturnType<typeof spawn>): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    child.stdout?.setEncoding("utf8");
    child.stderr?.setEncoding("utf8");
    child.stdout?.on("data", (chunk) => { stdout += chunk; });
    child.stderr?.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => { resolve({ code, stdout, stderr }); });
  });
}

async function waitForFile(filePath: string): Promise<void> {
  for (let attempt = 0; attempt < 500; attempt += 1) {
    try {
      await readFile(filePath, "utf8");
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }
  throw new Error(`Timed out waiting for ${filePath}`);
}

async function expectProcessStopped(pid: number): Promise<void> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const result = spawnSync("ps", ["-o", "stat=", "-p", String(pid)], { encoding: "utf8" });
    if (result.status !== 0 || result.stdout.trim().startsWith("Z")) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error(`Process ${pid} remained active after launcher shutdown`);
}
