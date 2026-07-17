import { chmod, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";

const ROOT_DIR = process.cwd();
const LAUNCHER = path.join(ROOT_DIR, "examples", "nanochat", "run_nanobench_ablation.sbatch");
const LOCAL_LAUNCHER = path.join(ROOT_DIR, "examples", "nanochat", "run_nanochat_benchmark.sh");
const MANIFEST_TOOL = path.join(ROOT_DIR, "examples", "nanochat", "manage_nanochat_run.py");

describe("nanochat Slurm ablation launcher", () => {
  it("omits the unsupported sampler seed for local Centaur runs", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-local-launcher-"));
    const binDir = path.join(dir, "bin");
    const nodeLog = path.join(dir, "node.log");
    await mkdir(binDir);
    await writeExecutable(binDir, "python3", ["#!/usr/bin/env bash", "printf '%064d\\n' 0"]);
    await writeExecutable(binDir, "node", ["#!/usr/bin/env bash", `printf '%s\\n' "$*" > ${JSON.stringify(nodeLog)}`]);

    const launcherEnv = {
      ...process.env,
      AUTORESEARCH_DIR: path.join(dir, "autoresearch"),
      HOME: path.join(dir, "home"),
      PATH: `${binDir}:${process.env.PATH}`,
      SAMPLER: "centaur",
      VALIDATE_FINALISTS: "0",
      WORK_DIR: path.join(dir, "work")
    };
    const result = await runProcess("bash", [LOCAL_LAUNCHER], launcherEnv);

    expect(result.code, `${result.stderr}\n${result.stdout}`).toBe(0);
    const command = (await readFile(nodeLog, "utf8")).trim().split(" ");
    expect(flagValue(command, "--sampler")).toBe("centaur");
    expect(command).not.toContain("--sampler-seed");

    const seeded = await runProcess("bash", [LOCAL_LAUNCHER], {
      ...launcherEnv,
      SAMPLER: "tpe",
      SAMPLER_SEED: "17"
    });
    expect(seeded.code, `${seeded.stderr}\n${seeded.stdout}`).toBe(0);
    const seededCommand = (await readFile(nodeLog, "utf8")).trim().split(" ");
    expect(flagValue(seededCommand, "--sampler")).toBe("tpe");
    expect(seededCommand.filter((argument) => argument === "--sampler-seed")).toHaveLength(1);
    expect(flagValue(seededCommand, "--sampler-seed")).toBe("17");
  });

  it("runs four one-GPU discovery arms concurrently and validates every method", async () => {
    const fixture = await createLauncherFixture();
    const result = await runLauncher(fixture.env);

    expect(result.code, `${result.stderr}\n${result.stdout}`).toBe(0);
    expect(await readFile(LAUNCHER, "utf8")).toContain("#SBATCH --time=2-00:00:00");
    const commands = (await readFile(fixture.nodeLog, "utf8")).trim().split("\n").map((line) => line.split(" "));
    expect(commands).toHaveLength(4);

    for (const name of [
      "01_optuna_baseline",
      "02_autotune_resets_no_trial_transfer",
      "03_autotune_resets_trial_transfer"
    ]) {
      const command = commandFor(commands, name);
      expect(flagValue(command, "--sampler")).toBe("tpe");
      expect(flagValue(command, "--sampler-seed")).toBe("42");
      expect(flagValue(command, "--time-budget-seconds")).toBe("108000");
    }

    expect(flagValue(commandFor(commands, "01_optuna_baseline"), "--trials")).toBe("240");
    for (const name of ["02_autotune_resets_no_trial_transfer", "03_autotune_resets_trial_transfer"]) {
      const command = commandFor(commands, name);
      expect(flagValue(command, "--trials")).toBe("60");
      expect(flagValue(command, "--refine-rounds")).toBe("3");
      expect(flagValue(command, "--refine-trials")).toBe("60");
    }

    const centaur = commandFor(commands, "04_centaur");
    expect(flagValue(centaur, "--sampler")).toBe("centaur");
    expect(centaur).not.toContain("--sampler-seed");
    expect(flagValue(centaur, "--trials")).toBe("240");
    expect(flagValue(centaur, "--refine-rounds")).toBe("0");
    expect(flagValue(centaur, "--centaur-llm-probability")).toBe("0.3");
    expect(flagValue(centaur, "--centaur-warmup-trials")).toBe("10");
    expect(flagValue(centaur, "--centaur-seed")).toBe("42");

    const validation = await readFile(fixture.validationLog, "utf8");
    for (const label of ["optuna_baseline", "resets_no_transfer", "resets_trial_transfer", "centaur"]) {
      expect(validation).toContain(`--result ${label}=`);
    }
    for (const round of [0, 1, 2, 3]) {
      expect(validation).toContain(`results.round_${round}.json`);
    }
  });

  it("rejects unequal Centaur trials unless explicitly allowed", async () => {
    const fixture = await createLauncherFixture();
    const rejected = await runLauncher({ ...fixture.env, CENTAUR_TRIALS: "99" });

    expect(rejected.code).toBe(2);
    expect(rejected.stderr).toContain("baseline=240 refined=240 centaur=99");

    const allowed = await runLauncher({
      ...fixture.env,
      ALLOW_UNEQUAL_TRIALS: "1",
      CENTAUR_TRIALS: "99"
    });
    expect(allowed.code, `${allowed.stderr}\n${allowed.stdout}`).toBe(0);
    const commands = (await readFile(fixture.nodeLog, "utf8")).trim().split("\n").map((line) => line.split(" "));
    expect(flagValue(commandFor(commands, "04_centaur"), "--trials")).toBe("99");
  });

  it("rejects arithmetic expressions in numeric environment overrides", async () => {
    const fixture = await createLauncherFixture();
    const marker = path.join(path.dirname(fixture.outRoot), "arithmetic-injection");
    const result = await runLauncher({
      ...fixture.env,
      REFINE_ROUNDS: `rounds[$(touch ${marker})]`
    });

    expect(result.code).toBe(2);
    expect(result.stderr).toContain("REFINE_ROUNDS must be a decimal integer");
    await expect(readFile(marker, "utf8")).rejects.toThrow();
  });

  it("rejects unsafe cancellation grace values before launching jobs", async () => {
    const fixture = await createLauncherFixture();
    const leadingZero = await runLauncher({ ...fixture.env, CANCEL_GRACE_SECONDS: "09" });

    expect(leadingZero.code).toBe(2);
    expect(leadingZero.stderr).toContain("CANCEL_GRACE_SECONDS must be a decimal integer");

    const overflow = await runLauncher({
      ...fixture.env,
      CANCEL_GRACE_SECONDS: "18446744073709551616"
    });
    expect(overflow.code).toBe(2);
    expect(overflow.stderr).toContain("CANCEL_GRACE_SECONDS must be a decimal integer");
    await expect(readFile(fixture.nodeLog, "utf8")).rejects.toThrow();
  });

  it("rejects a duplicate Centaur sampler arm and existing discovery storage", async () => {
    const samplerFixture = await createLauncherFixture();
    const duplicateCentaur = await runLauncher({ ...samplerFixture.env, SAMPLER: "centaur" });
    expect(duplicateCentaur.code).toBe(2);
    expect(duplicateCentaur.stderr).toContain("Centaur has a dedicated fourth arm");

    const storageFixture = await createLauncherFixture();
    const existingArm = path.join(storageFixture.outRoot, "02_autotune_resets_no_trial_transfer");
    await mkdir(existingArm, { recursive: true });
    await writeFile(path.join(existingArm, "study.db"), "", "utf8");
    const existingStorage = await runLauncher(storageFixture.env);
    expect(existingStorage.code).toBe(2);
    expect(existingStorage.stderr).toContain("Run root already exists");
  });

  it("resumes finalist validation without relaunching discovery", async () => {
    const fixture = await createLauncherFixture();
    await mkdir(fixture.outRoot, { recursive: true });
    await writeFile(path.join(fixture.outRoot, "discovery_complete.json"), "{}", "utf8");
    const result = await runLauncher({
      ...fixture.env,
      VALIDATION_ONLY: "1",
      NANOCHAT_TEST_REFINE_ROUNDS: "3"
    });

    expect(result.code, `${result.stderr}\n${result.stdout}`).toBe(0);
    await expect(readFile(fixture.nodeLog, "utf8")).rejects.toThrow();
    expect(await readFile(fixture.validationLog, "utf8")).toContain("validate_nanochat.py");
    expect(await readFile(fixture.validationLog, "utf8")).toContain("results.round_3.json");
  });

  it("fails fast and cancels sibling discovery steps", async () => {
    const fixture = await createLauncherFixture();
    const startedAt = Date.now();
    const result = await runLauncher({
      ...fixture.env,
      NANOCHAT_FAIL_VARIANT: "04_centaur",
      NANOCHAT_BLOCK_SIBLINGS: "1"
    });

    expect(result.code).toBe(1);
    expect(Date.now() - startedAt, `${result.stderr}\n${result.stdout}`).toBeLessThan(1500);
    expect(result.stderr).toContain("04_centaur");
    expect((await readFile(fixture.cancellationLog, "utf8")).trim().split("\n")).toHaveLength(3);
    await expect(readFile(fixture.validationLog, "utf8")).rejects.toThrow();
  });

  it("fails fast when a discovery arm stops before its expected trial count", async () => {
    const fixture = await createLauncherFixture();
    const result = await runLauncher({
      ...fixture.env,
      NANOCHAT_INCOMPLETE_VARIANT: "04_centaur"
    });

    expect(result.code).toBe(1);
    expect(result.stderr).toContain("04_centaur");
    expect((await readFile(fixture.cancellationLog, "utf8")).trim().split("\n")).toHaveLength(3);
    await expect(readFile(fixture.validationLog, "utf8")).rejects.toThrow();
  });

  it("attests complete discovery counts and detects result drift", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-manifest-"));
    const manifest = path.join(dir, "discovery_complete.json");
    const results = ["baseline", "centaur", "reset_0", "reset_1"];
    for (const name of results) {
      await writeFile(
        path.join(dir, `${name}.json`),
        JSON.stringify({ all_trials: [{ number: 0, user_attrs: {} }, { number: 1, user_attrs: { autotune_transfer: true } }] }),
        "utf8"
      );
    }
    const resultArgs = results.flatMap((name) => ["--result", `${name}=1:${path.join(dir, `${name}.json`)}`]);
    const checked = await runProcess("python3", [
      MANIFEST_TOOL, "check", "--root", dir, "--result", `baseline=1:${path.join(dir, "baseline.json")}`
    ]);
    expect(checked.code, checked.stderr).toBe(0);
    const incomplete = await runProcess("python3", [
      MANIFEST_TOOL, "check", "--root", dir, "--result", `baseline=2:${path.join(dir, "baseline.json")}`
    ]);
    expect(incomplete.code).not.toBe(0);
    expect(incomplete.stderr).toContain("attempted 1 new trials; expected 2");

    const created = await runProcess("python3", [
      MANIFEST_TOOL, "create", "--manifest", manifest, "--refine-rounds", "1", ...resultArgs
    ]);
    expect(created.code, created.stderr).toBe(0);
    expect((await runProcess("python3", [MANIFEST_TOOL, "verify", "--manifest", manifest])).stdout.trim()).toBe("1");

    await writeFile(path.join(dir, "centaur.json"), JSON.stringify({ all_trials: [] }), "utf8");
    const drifted = await runProcess("python3", [MANIFEST_TOOL, "verify", "--manifest", manifest]);
    expect(drifted.code).not.toBe(0);
    expect(drifted.stderr).toContain("changed after discovery completed");
  });
});

async function createLauncherFixture() {
  const dir = await mkdtemp(path.join(tmpdir(), "autotune-nanochat-launcher-"));
  const binDir = path.join(dir, "bin");
  const home = path.join(dir, "home");
  const autoresearch = path.join(dir, "autoresearch");
  const nodeLog = path.join(dir, "node.log");
  const validationLog = path.join(dir, "validation.log");
  const cancellationLog = path.join(dir, "cancellation.log");
  const syncDir = path.join(dir, "sync");
  const outRoot = path.join(dir, "out");
  await mkdir(binDir, { recursive: true });
  await mkdir(autoresearch, { recursive: true });
  await mkdir(syncDir, { recursive: true });
  await mkdir(path.join(home, ".cache", "autoresearch", "data"), { recursive: true });
  await mkdir(path.join(home, ".cache", "autoresearch", "tokenizer"), { recursive: true });
  await writeFile(path.join(autoresearch, "train.py"), "", "utf8");
  await writeFile(path.join(autoresearch, "prepare.py"), "", "utf8");
  await writeFile(path.join(home, ".cache", "autoresearch", "data", "shard_06542.parquet"), "", "utf8");
  await writeFile(path.join(home, ".cache", "autoresearch", "tokenizer", "tokenizer.pkl"), "", "utf8");
  await writeFile(path.join(home, ".cache", "autoresearch", "tokenizer", "token_bytes.pt"), "", "utf8");

  await writeExecutable(binDir, "npm", ["#!/usr/bin/env bash", "exit 0"]);
  await writeExecutable(binDir, "srun", [
    "#!/usr/bin/env bash",
    "if [[ \" $* \" == *validate_nanochat.py* ]]; then",
    "  printf '%s\\n' \"$*\" > \"$NANOCHAT_VALIDATION_LOG\"",
    "  exit 0",
    "fi",
    "while [[ \"$1\" == --* ]]; do shift; done",
    "exec \"$@\""
  ]);
  await writeExecutable(binDir, "node", [
    "#!/usr/bin/env bash",
    "if [[ -n \"${NANOCHAT_INCOMPLETE_VARIANT:-}\" ]]; then",
    "  if [[ \" $* \" == *\"/$NANOCHAT_INCOMPLETE_VARIANT \"* ]]; then printf '%s\\n' \"$*\" >> \"$NANOCHAT_NODE_LOG\"; exit 0; fi",
    "  trap 'printf \"%s\\n\" \"$$\" >> \"$NANOCHAT_CANCELLATION_LOG\"; exit 143' TERM",
    "  for _ in {1..30}; do sleep 0.1; done",
    "  exit 90",
    "fi",
    "if [[ -n \"${NANOCHAT_FAIL_VARIANT:-}\" && \" $* \" == *\"/$NANOCHAT_FAIL_VARIANT \"* ]]; then exit 17; fi",
    "if [[ \"${NANOCHAT_BLOCK_SIBLINGS:-0}\" == 1 ]]; then",
    "  trap 'printf \"%s\\n\" \"$$\" >> \"$NANOCHAT_CANCELLATION_LOG\"; exit 143' TERM",
    "  while :; do sleep 0.1; done",
    "fi",
    "mkdir \"$NANOCHAT_SYNC_DIR/$$\"",
    "for _ in {1..100}; do",
    "  count=$(find \"$NANOCHAT_SYNC_DIR\" -mindepth 1 -maxdepth 1 -type d | wc -l)",
    "  [[ $count -ge 4 ]] && break",
    "  sleep 0.01",
    "done",
    "[[ $count -ge 4 ]] || exit 90",
    "printf '%s\\n' \"$*\" >> \"$NANOCHAT_NODE_LOG\""
  ]);
  await writeExecutable(binDir, "python3", [
    "#!/usr/bin/env bash",
    "if [[ \"$1\" == *prepare_nanochat_cache.py ]]; then",
    "  printf '%064d\\n' 0",
    "elif [[ \"$1\" == *manage_nanochat_run.py && \"$2\" == check ]]; then",
    "  if [[ -n \"${NANOCHAT_INCOMPLETE_VARIANT:-}\" && \" $* \" == *\"/$NANOCHAT_INCOMPLETE_VARIANT/\"* ]]; then exit 19; fi",
    "elif [[ \"$1\" == *manage_nanochat_run.py && \"$2\" == verify ]]; then",
    "  printf '%s\\n' \"${NANOCHAT_TEST_REFINE_ROUNDS:-2}\"",
    "fi"
  ]);

  return {
    nodeLog,
    outRoot,
    validationLog,
    cancellationLog,
    env: {
      ...process.env,
      ROOT_DIR,
      AUTORESEARCH_DIR: autoresearch,
      HOME: home,
      OUT_ROOT: outRoot,
      RUN_GROUP: "test",
      PATH: `${binDir}:${process.env.PATH}`,
      NANOCHAT_NODE_LOG: nodeLog,
      NANOCHAT_VALIDATION_LOG: validationLog,
      NANOCHAT_CANCELLATION_LOG: cancellationLog,
      NANOCHAT_SYNC_DIR: syncDir
    }
  };
}

async function writeExecutable(directory: string, name: string, lines: string[]) {
  const file = path.join(directory, name);
  await writeFile(file, lines.join("\n"), "utf8");
  await chmod(file, 0o755);
}

function runLauncher(env: NodeJS.ProcessEnv): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return runProcess("bash", [LAUNCHER], env);
}

function runProcess(
  command: string,
  args: string[],
  env: NodeJS.ProcessEnv = process.env
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { env, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => { resolve({ code, stdout, stderr }); });
  });
}

function commandFor(commands: string[][], name: string): string[] {
  const command = commands.find((items) => flagValue(items, "--work-dir")?.endsWith(`/${name}`));
  if (!command) throw new Error(`missing command for ${name}`);
  return command;
}

function flagValue(command: string[], flag: string): string | undefined {
  const index = command.indexOf(flag);
  return index === -1 ? undefined : command[index + 1];
}
