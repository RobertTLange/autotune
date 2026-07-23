import { chmod, copyFile, mkdir, mkdtemp, readFile, stat, symlink, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";

const ROOT_DIR = process.cwd();
const EXAMPLE_DIR = path.join(ROOT_DIR, "examples", "cifar10_speedrun");
const LAUNCHER = path.join(EXAMPLE_DIR, "run_cifar10_speedrun_ablations.sbatch");
const MANIFEST_TOOL = path.join(EXAMPLE_DIR, "manage_cifar10_run.py");
const PREPARE_TOOL = path.join(EXAMPLE_DIR, "prepare_cifar10_data.py");
describe.skipIf(process.platform !== "linux")("CIFAR-10 speedrun Slurm launcher", () => {
  it("runs four isolated one-GPU arms concurrently with the two-day preset", async () => {
    const fixture = await createLauncherFixture();
    const result = await runLauncher(fixture.env);
    expect(result.code, `${result.stderr}\n${result.stdout}`).toBe(0);
    const launcher = await readFile(LAUNCHER, "utf8");
    expect(launcher).toContain("#SBATCH --nodes=1");
    expect(launcher).toContain("#SBATCH --ntasks=4");
    expect(launcher).toContain("#SBATCH --gres=gpu:4");
    expect(launcher).toContain("#SBATCH --cpus-per-task=8");
    expect(launcher).toContain("#SBATCH --mem=256G");
    expect(launcher).toContain("#SBATCH --time=2-00:00:00");
    expect((await readFile(fixture.prepareLog, "utf8")).trim().split("\n")).toHaveLength(1);
    expect((await readFile(fixture.analyzeLog, "utf8")).trim().split("\n")).toHaveLength(1);
    const srunCommands = (await readFile(fixture.srunLog, "utf8")).trim().split("\n");
    expect(srunCommands).toHaveLength(4);
    for (const command of srunCommands) {
      expect(command).toContain("--exclusive --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G /usr/bin/env");
    }
    const commands = await readCommands(fixture.nodeLog);
    expect(commands).toHaveLength(4);
    for (const name of [
      "01_optuna_baseline",
      "02_autotune_resets_no_trial_transfer",
      "03_autotune_resets_trial_transfer"
    ]) {
      const command = commandFor(commands, name);
      expect(flagValue(command, "--sampler")).toBe("tpe");
      expect(flagValue(command, "--sampler-seed")).toBe("42");
      expect(flagValue(command, "--time-budget-seconds")).toBe("162000");
      expect(flagValue(command, "--timeout-seconds")).toBe("1800");
      expect(flagValue(command, "--n-jobs")).toBe("1");
      expect(flagValue(command, "--pruner")).toBe("none");
      expect(flagValue(command, "--storage")).toContain(`${name}/study.db`);
      expect(flagValue(command, "--study-name")).toContain(name);
      expect(flagValue(command, "--config")).toBe(path.join(fixture.outRoot, "search_space.initial.yaml"));
    }

    expect(flagValue(commandFor(commands, "01_optuna_baseline"), "--trials")).toBe("500");
    for (const name of ["02_autotune_resets_no_trial_transfer", "03_autotune_resets_trial_transfer"]) {
      const command = commandFor(commands, name);
      expect(flagValue(command, "--trials")).toBe("125");
      expect(flagValue(command, "--refine-rounds")).toBe("3");
      expect(flagValue(command, "--refine-trials")).toBe("125");
    }
    const resetWithoutTransfer = commandFor(commands, "02_autotune_resets_no_trial_transfer");
    expect(resetWithoutTransfer).toContain("--no-refine-transfer-fixed-params");
    expect(resetWithoutTransfer).toContain("--no-refine-transfer-trials");
    const resetWithTransfer = commandFor(commands, "03_autotune_resets_trial_transfer");
    expect(resetWithTransfer).toContain("--no-refine-transfer-fixed-params");
    expect(resetWithTransfer).not.toContain("--no-refine-transfer-trials");

    const centaur = commandFor(commands, "04_centaur");
    expect(flagValue(centaur, "--sampler")).toBe("centaur");
    expect(centaur).not.toContain("--sampler-seed");
    expect(flagValue(centaur, "--trials")).toBe("500");
    expect(flagValue(centaur, "--refine-rounds")).toBe("0");
    expect(flagValue(centaur, "--centaur-llm-probability")).toBe("0.3");
    expect(flagValue(centaur, "--centaur-warmup-trials")).toBe("10");
    expect(flagValue(centaur, "--centaur-seed")).toBe("42");

    const environments = (await readFile(fixture.environmentLog, "utf8")).trim().split("\n");
    expect(environments).toHaveLength(4);
    for (const name of ["01_optuna_baseline", "02_autotune_resets_no_trial_transfer", "03_autotune_resets_trial_transfer", "04_centaur"]) {
      const environment = environments.find((line) => line.startsWith(`${name}|`));
      expect(environment).toContain(`/runtime_cache/${name}/tmp|`);
      expect(environment).toContain(`/runtime_cache/${name}/torchinductor|`);
      expect(environment).toContain(`/runtime_cache/${name}/triton|`);
      expect(environment).toContain(`/runtime_cache/${name}/cuda|`);
      expect(environment).toContain(`/runtime_cache/${name}/torch_extensions|`);
      expect(environment).toContain(`/runtime_cache/${name}/xdg|`);
      expect(environment).toContain(`/${name}/trial_metrics`);
      expect(environment).toContain(`|${fixture.pythonDir}:`);
      expect(environment).toMatch(/\|100$/);
    }
    expect((await stat(fixture.outRoot)).mode & 0o077).toBe(0);
    const manifest = JSON.parse(await readFile(path.join(fixture.outRoot, "discovery_complete.json"), "utf8"));
    expect(manifest).toMatchObject({
      schema_version: 1,
      status: "complete",
      refine_rounds: 3
    });
    expect(manifest.results).toHaveLength(10);
  });

  it("preserves SINGLE_SHOT_TRIALS as the baseline fallback", async () => {
    const fixture = await createLauncherFixture();
    const result = await runLauncher({
      ...fixture.env,
      SINGLE_SHOT_TRIALS: "20",
      REFINE_INITIAL_TRIALS: "5",
      REFINE_ROUNDS: "3",
      REFINE_TRIALS: "5",
      CENTAUR_TRIALS: "20"
    });

    expect(result.code, `${result.stderr}\n${result.stdout}`).toBe(0);
    const commands = await readCommands(fixture.nodeLog);
    expect(flagValue(commandFor(commands, "01_optuna_baseline"), "--trials")).toBe("20");
  });

  it("defaults results under the CIFAR-10 example directory", async () => {
    const fixture = await createLauncherFixture();
    const rootDir = path.join(path.dirname(fixture.outRoot), "repo");
    const exampleDir = path.join(rootDir, "examples", "cifar10_speedrun");
    await mkdir(exampleDir, { recursive: true });
    await chmod(rootDir, 0o775);
    await chmod(path.join(rootDir, "examples"), 0o775);
    await chmod(exampleDir, 0o775);
    await writeFile(path.join(rootDir, "package.json"), "{}", "utf8");
    await copyFile(MANIFEST_TOOL, path.join(exampleDir, "manage_cifar10_run.py"));
    const env: NodeJS.ProcessEnv = { ...fixture.env, ROOT_DIR: rootDir };
    delete env.OUT_ROOT;

    const result = await runLauncher(env);

    expect(result.code, `${result.stderr}\n${result.stdout}`).toBe(0);
    const defaultRoot = path.join(exampleDir, "autotune", "cifar10_speedrun_ablations", "test");
    expect((await stat(path.join(exampleDir, "autotune"))).mode & 0o077).toBe(0);
    expect((await stat(defaultRoot)).mode & 0o077).toBe(0);
    const commands = await readCommands(fixture.nodeLog);
    expect(flagValue(commandFor(commands, "01_optuna_baseline"), "--work-dir"))
      .toBe(path.join(defaultRoot, "01_optuna_baseline"));

    const explicitRoot = path.join(exampleDir, "autotune", "cifar10_speedrun_ablations", "explicit");
    const explicit = await runLauncher({ ...env, OUT_ROOT: explicitRoot });
    expect(explicit.code).toBe(2);
    expect(explicit.stderr).toContain("unsafe writable parent");

    const explicitCache = await runLauncher({
      ...env,
      OUT_ROOT: fixture.outRoot,
      AUTOTUNE_RUNTIME_CACHE_DIR: path.join(exampleDir, "explicit-cache")
    });
    expect(explicitCache.code).toBe(2);
    expect(explicitCache.stderr).toContain("unsafe writable parent");
  });

  it("rejects unequal budgets unless explicitly allowed", async () => {
    const rejectedFixture = await createLauncherFixture();
    const rejected = await runLauncher({ ...rejectedFixture.env, CENTAUR_TRIALS: "499" });
    expect(rejected.code).toBe(2);
    expect(rejected.stderr).toContain("baseline=500 refined=500 centaur=499");

    const allowedFixture = await createLauncherFixture();
    const allowed = await runLauncher({
      ...allowedFixture.env,
      ALLOW_UNEQUAL_TRIALS: "1",
      CENTAUR_TRIALS: "499"
    });
    expect(allowed.code, `${allowed.stderr}\n${allowed.stdout}`).toBe(0);
  });

  it("rejects unsafe numeric overrides and duplicate run roots before launch", async () => {
    const fixture = await createLauncherFixture();
    const marker = path.join(path.dirname(fixture.outRoot), "injected");
    const injection = await runLauncher({
      ...fixture.env,
      REFINE_ROUNDS: `rounds[$(touch ${marker})]`
    });
    expect(injection.code).toBe(2);
    expect(injection.stderr).toContain("REFINE_ROUNDS must be a decimal integer");
    await expect(readFile(marker, "utf8")).rejects.toThrow();

    const leadingZeroFixture = await createLauncherFixture();
    const leadingZero = await runLauncher({ ...leadingZeroFixture.env, CANCEL_GRACE_SECONDS: "09" });
    expect(leadingZero.code).toBe(2);
    expect(leadingZero.stderr).toContain("CANCEL_GRACE_SECONDS must be a decimal integer");

    const overflowFixture = await createLauncherFixture();
    const overflow = await runLauncher({
      ...overflowFixture.env,
      CANCEL_GRACE_SECONDS: "18446744073709551616"
    });
    expect(overflow.code).toBe(2);
    expect(overflow.stderr).toContain("CANCEL_GRACE_SECONDS must be a decimal integer");

    const existingFixture = await createLauncherFixture();
    await mkdir(existingFixture.outRoot, { recursive: true });
    const existing = await runLauncher(existingFixture.env);
    expect(existing.code).toBe(2);
    expect(existing.stderr).toContain("Run root already exists");

    const reservedPathFixture = await createLauncherFixture();
    const reservedPath = await runLauncher({
      ...reservedPathFixture.env,
      OUT_ROOT: `${reservedPathFixture.outRoot}?query`
    });
    expect(reservedPath.code).toBe(2);
    expect(reservedPath.stderr).toContain("SQLite URL metacharacters");

    const writableParentFixture = await createLauncherFixture();
    const writableParent = path.join(path.dirname(writableParentFixture.outRoot), "shared");
    const stickyChild = path.join(writableParent, "sticky");
    await mkdir(writableParent, { mode: 0o770 });
    await chmod(writableParent, 0o770);
    await mkdir(stickyChild, { mode: 0o1777 });
    await chmod(stickyChild, 0o1777);
    const unsafeParent = await runLauncher({
      ...writableParentFixture.env,
      OUT_ROOT: path.join(stickyChild, "out")
    });
    expect(unsafeParent.code).toBe(2);
    expect(unsafeParent.stderr).toContain("unsafe writable parent");
  });

  it("fails fast and cancels the other discovery arms", async () => {
    const fixture = await createLauncherFixture();
    const startedAt = Date.now();
    const result = await runLauncher({
      ...fixture.env,
      CIFAR10_FAIL_VARIANT: "04_centaur",
      CIFAR10_BLOCK_SIBLINGS: "1"
    });

    expect(result.code).toBe(1);
    expect(Date.now() - startedAt, `${result.stderr}\n${result.stdout}`).toBeLessThan(2000);
    expect(result.stderr).toContain("04_centaur");
    expect((await readFile(fixture.cancellationLog, "utf8")).trim().split("\n")).toHaveLength(3);
    await expect(readFile(path.join(fixture.outRoot, "discovery_complete.json"), "utf8")).rejects.toThrow();
  });

  it("rejects an arm that exits successfully with incomplete results", async () => {
    const fixture = await createLauncherFixture();
    const result = await runLauncher({
      ...fixture.env,
      CIFAR10_INCOMPLETE_VARIANT: "04_centaur"
    });

    expect(result.code).toBe(1);
    expect(result.stderr).toContain("04_centaur");
    expect(result.stderr).toContain("attempted 499 new trials; expected 500");
    expect((await readFile(fixture.cancellationLog, "utf8")).trim().split("\n")).toHaveLength(3);
  });
});

describe("CIFAR-10 discovery completion manifest", () => {
  it("counts only new trials and detects post-completion result drift", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-cifar10-manifest-"));
    const resultPath = path.join(dir, "results.json");
    const manifest = path.join(dir, "discovery_complete.json");
    const payload = JSON.stringify({
      all_trials: [
        { number: 0, user_attrs: {} },
        { number: 1, user_attrs: { autotune_transfer: true } }
      ]
    });
    const resultPaths = {
      optuna_baseline: resultPath,
      centaur: path.join(dir, "centaur.json"),
      resets_no_transfer_round_0: path.join(dir, "reset-no-transfer.json"),
      resets_trial_transfer_round_0: path.join(dir, "reset-transfer.json")
    };
    for (const pathValue of Object.values(resultPaths)) {
      await writeFile(pathValue, payload, { encoding: "utf8", mode: 0o600 });
      await chmod(pathValue, 0o600);
    }

    const checked = await runProcess("python3", [
      MANIFEST_TOOL, "check", "--root", dir, "--result", `baseline=1:${resultPath}`
    ]);
    expect(checked.code, checked.stderr).toBe(0);
    const incomplete = await runProcess("python3", [
      MANIFEST_TOOL, "check", "--root", dir, "--result", `baseline=2:${resultPath}`
    ]);
    expect(incomplete.code).not.toBe(0);
    expect(incomplete.stderr).toContain("attempted 1 new trials; expected 2");
    const symlinkPath = path.join(dir, "symlink.json");
    await symlink(resultPath, symlinkPath);
    const symlinked = await runProcess("python3", [
      MANIFEST_TOOL, "check", "--root", dir, "--result", `baseline=1:${symlinkPath}`
    ]);
    expect(symlinked.code).not.toBe(0);
    expect(symlinked.stderr).toContain("must not be a symlink");

    const reducedTopology = await runProcess("python3", [
      MANIFEST_TOOL, "create", "--manifest", manifest, "--refine-rounds", "0",
      "--result", `baseline=1:${resultPath}`
    ]);
    expect(reducedTopology.code).not.toBe(0);
    expect(reducedTopology.stderr).toContain("expected four-arm topology");
    const labels = ["optuna_baseline", "centaur", "resets_no_transfer_round_0", "resets_trial_transfer_round_0"];
    const aliased = await runProcess("python3", [
      MANIFEST_TOOL, "create", "--manifest", manifest, "--refine-rounds", "0",
      ...labels.flatMap((label) => ["--result", `${label}=1:${resultPath}`])
    ]);
    expect(aliased.code).not.toBe(0);
    expect(aliased.stderr).toContain("result files must be unique");
    const resultArguments = Object.entries(resultPaths).flatMap(([label, pathValue]) => [
      "--result", `${label}=1:${pathValue}`
    ]);
    const created = await runProcess("python3", [
      MANIFEST_TOOL, "create", "--manifest", manifest, "--refine-rounds", "0", ...resultArguments
    ]);
    expect(created.code, created.stderr).toBe(0);
    const duplicate = await runProcess("python3", [
      MANIFEST_TOOL, "create", "--manifest", manifest, "--refine-rounds", "0", ...resultArguments
    ]);
    expect(duplicate.code).not.toBe(0);
    expect(duplicate.stderr).toContain("manifest already exists");
    await writeFile(resultPath, JSON.stringify({ all_trials: [] }), "utf8");
    const drifted = await runProcess("python3", [MANIFEST_TOOL, "verify", "--manifest", manifest]);
    expect(drifted.code).not.toBe(0);
    expect(drifted.stderr).toContain("changed after discovery completed");
  });
});

describe("CIFAR-10 data preparation", () => {
  it("rebuilds and validates both canonical tensor splits", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-cifar10-data-"));
    const modules = path.join(dir, "modules");
    const dataDir = path.join(dir, "data");
    const downloadLog = path.join(dir, "downloads.log");
    await mkdir(path.join(modules, "torchvision"), { recursive: true });
    await writeFile(path.join(modules, "torch.py"), [
      "import json",
      "uint8 = 'uint8'",
      "int64 = 'int64'",
      "class Tensor:",
      "    def __init__(self, shape, dtype): self.shape, self.dtype = tuple(shape), dtype",
      "def tensor(value): return Tensor(value['shape'], value['dtype'])",
      "def save(payload, output):",
      "    encoded = {key: ({'shape': list(value.shape), 'dtype': value.dtype} if isinstance(value, Tensor) else value) for key, value in payload.items()}",
      "    with open(output, 'w') as handle: json.dump(encoded, handle)",
      "def load(path, map_location, weights_only):",
      "    with open(path) as handle: payload = json.load(handle)",
      "    payload['images'] = Tensor(payload['images']['shape'], payload['images']['dtype'])",
      "    payload['labels'] = Tensor(payload['labels']['shape'], payload['labels']['dtype'])",
      "    return payload"
    ].join("\n"), "utf8");
    await writeFile(path.join(modules, "torchvision", "__init__.py"), "", "utf8");
    await writeFile(path.join(modules, "torchvision", "datasets.py"), [
      "import os",
      "class CIFAR10:",
      "    def __init__(self, root, download, train):",
      "        with open(os.environ['CIFAR10_DOWNLOAD_LOG'], 'a') as handle: handle.write(f'{train}\\n')",
      "        count = 50000 if train else 10000",
      "        self.data = {'shape': [count, 32, 32, 3], 'dtype': 'uint8'}",
      "        self.targets = {'shape': [count], 'dtype': 'int64'}",
      "        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']"
    ].join("\n"), "utf8");
    const env = {
      ...process.env,
      PYTHONPATH: modules,
      CIFAR10_DOWNLOAD_LOG: downloadLog
    };

    const first = await runProcess("python3", [PREPARE_TOOL, "--data-dir", dataDir], env);
    expect(first.code, first.stderr).toBe(0);
    await writeFile(path.join(dataDir, "test.pt"), "{}", "utf8");
    const second = await runProcess("python3", [PREPARE_TOOL, "--data-dir", dataDir], env);
    expect(second.code, second.stderr).toBe(0);
    expect((await readFile(downloadLog, "utf8")).trim().split("\n")).toEqual(["True", "False", "True", "False"]);
    expect(JSON.parse(await readFile(path.join(dataDir, "train.pt"), "utf8")).classes[0]).toBe("airplane");
    expect(JSON.parse(await readFile(path.join(dataDir, "test.pt"), "utf8")).classes[0]).toBe("airplane");
  });

  it("refuses a symlinked tensor cache", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-cifar10-symlink-"));
    const dataDir = path.join(dir, "data");
    const target = path.join(dir, "outside.pt");
    await mkdir(dataDir);
    await writeFile(target, "outside", "utf8");
    await symlink(target, path.join(dataDir, "train.pt"));

    const result = await runProcess("python3", [PREPARE_TOOL, "--data-dir", dataDir]);
    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("refusing symlinked CIFAR-10 tensor cache");
  });

  it("refuses a group-writable cache parent", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "autotune-cifar10-parent-"));
    const parent = path.join(dir, "shared");
    const stickyChild = path.join(parent, "sticky");
    await mkdir(parent, { mode: 0o770 });
    await chmod(parent, 0o770);
    await mkdir(stickyChild, { mode: 0o1777 });
    await chmod(stickyChild, 0o1777);
    const result = await runProcess("python3", [PREPARE_TOOL, "--data-dir", path.join(stickyChild, "data")]);
    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("data path has an unsafe writable parent");
  });
});

async function createLauncherFixture() {
  const dir = await mkdtemp(path.join(tmpdir(), "autotune-cifar10-launcher-"));
  const binDir = path.join(dir, "bin");
  const pythonDir = path.join(dir, "python-bin");
  const home = path.join(dir, "home");
  const outRoot = path.join(dir, "out");
  const nodeLog = path.join(dir, "node.log");
  const srunLog = path.join(dir, "srun.log");
  const analyzeLog = path.join(dir, "analyze.log");
  const prepareLog = path.join(dir, "prepare.log");
  const environmentLog = path.join(dir, "environment.log");
  const cancellationLog = path.join(dir, "cancellation.log");
  const syncDir = path.join(dir, "sync");
  await mkdir(binDir, { recursive: true });
  await mkdir(pythonDir, { recursive: true });
  await mkdir(syncDir, { recursive: true });
  await writeExecutable(binDir, "npm", ["#!/usr/bin/env bash", "exit 0"]);
  await writeExecutable(binDir, "srun", [
    "#!/usr/bin/env bash",
    "printf '%s\\n' \"$*\" >> \"$CIFAR10_SRUN_LOG\"",
    "while [[ \"$1\" == --* ]]; do shift; done",
    "exec \"$@\""
  ]);
  await writeExecutable(pythonDir, "python3", [
    "#!/usr/bin/env bash",
    "if [[ \"${1:-}\" == -c && \"${2:-}\" == 'import torch, torchvision' ]]; then exit 0; fi",
    "if [[ \"$1\" == *prepare_cifar10_data.py ]]; then",
    "  printf '%s\\n' \"$*\" >> \"$CIFAR10_PREPARE_LOG\"",
    "  mkdir -p \"$CIFAR10_SPEEDRUN_DATA_DIR\"",
    "  touch \"$CIFAR10_SPEEDRUN_DATA_DIR/train.pt\" \"$CIFAR10_SPEEDRUN_DATA_DIR/test.pt\"",
    "  exit 0",
    "fi",
    "exec /usr/bin/python3 \"$@\""
  ]);
  await writeExecutable(binDir, "node", mockNodeScript());

  return {
    outRoot,
    pythonDir,
    nodeLog,
    srunLog,
    analyzeLog,
    prepareLog,
    environmentLog,
    cancellationLog,
    env: {
      ...process.env,
      ROOT_DIR,
      HOME: home,
      OUT_ROOT: outRoot,
      RUN_GROUP: "test",
      PATH: `${binDir}:${process.env.PATH}`,
      CIFAR10_NODE_LOG: nodeLog,
      CIFAR10_SRUN_LOG: srunLog,
      CIFAR10_ANALYZE_LOG: analyzeLog,
      CIFAR10_PREPARE_LOG: prepareLog,
      CIFAR10_ENVIRONMENT_LOG: environmentLog,
      CIFAR10_CANCELLATION_LOG: cancellationLog,
      CIFAR10_SYNC_DIR: syncDir,
      AUTOTUNE_PYTHON: path.join(pythonDir, "python3"),
      CANCEL_GRACE_SECONDS: "1"
    }
  };
}

function mockNodeScript(): string[] {
  return [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "if [[ \" $* \" == *\" analyze \"* ]]; then",
    "  printf '%s\\n' \"$*\" >> \"$CIFAR10_ANALYZE_LOG\"",
    "  while (($#)); do [[ \"$1\" == --output ]] && { shift; mkdir -p \"$(dirname \"$1\")\"; touch \"$1\"; exit 0; }; shift; done",
    "fi",
    "args=(\"$@\")",
    "work_dir='' trials='' rounds=0 refine_trials=''",
    "while (($#)); do",
    "  case \"$1\" in",
    "    --work-dir) shift; work_dir=\"$1\" ;;",
    "    --trials) shift; trials=\"$1\" ;;",
    "    --refine-rounds) shift; rounds=\"$1\" ;;",
    "    --refine-trials) shift; refine_trials=\"$1\" ;;",
    "  esac",
    "  shift",
    "done",
    "name=$(basename \"$work_dir\")",
    "printf '%s\\n' \"${args[*]}\" >> \"$CIFAR10_NODE_LOG\"",
    "printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' \"$name\" \"$TMPDIR\" \"$TORCHINDUCTOR_CACHE_DIR\" \"$TRITON_CACHE_DIR\" \"$CUDA_CACHE_PATH\" \"$TORCH_EXTENSIONS_DIR\" \"$XDG_CACHE_HOME\" \"$CIFAR10_SPEEDRUN_RESULTS_DIR\" \"$PATH\" \"$CIFAR10_SPEEDRUN_NUM_RUNS\" >> \"$CIFAR10_ENVIRONMENT_LOG\"",
    "cancelled() { printf '%s\\n' \"$name\" >> \"$CIFAR10_CANCELLATION_LOG\"; exit 143; }",
    "trap cancelled TERM",
    "if [[ -n \"${CIFAR10_FAIL_VARIANT:-}\" && \"$name\" == \"$CIFAR10_FAIL_VARIANT\" ]]; then exit 17; fi",
    "if [[ \"${CIFAR10_BLOCK_SIBLINGS:-0}\" == 1 ]]; then while :; do sleep 0.05; done; fi",
    "mkdir \"$CIFAR10_SYNC_DIR/$name\"",
    "for _ in {1..100}; do count=$(find \"$CIFAR10_SYNC_DIR\" -mindepth 1 -maxdepth 1 -type d | wc -l); [[ $count -ge 4 ]] && break; sleep 0.01; done",
    "[[ $count -ge 4 ]] || exit 90",
    "if [[ -n \"${CIFAR10_INCOMPLETE_VARIANT:-}\" && \"$name\" != \"$CIFAR10_INCOMPLETE_VARIANT\" ]]; then while :; do sleep 0.05; done; fi",
    "make_result() {",
    "  local output=\"$1\" count=\"$2\"",
    "  mkdir -p \"$(dirname \"$output\")\"",
    "  /usr/bin/python3 -c 'import json,sys; json.dump({\"all_trials\": [{\"number\": i, \"user_attrs\": {}} for i in range(int(sys.argv[2]))]}, open(sys.argv[1], \"w\"))' \"$output\" \"$count\"",
    "}",
    "if ((rounds > 0)); then",
    "  for ((round=0; round<=rounds; round++)); do expected=\"$refine_trials\"; ((round == 0)) && expected=\"$trials\"; make_result \"$work_dir/results.round_$round.json\" \"$expected\"; done",
    "else",
    "  expected=\"$trials\"; [[ \"$name\" == \"${CIFAR10_INCOMPLETE_VARIANT:-}\" ]] && expected=$((trials - 1)); make_result \"$work_dir/results.json\" \"$expected\"",
    "fi"
  ];
}

async function readCommands(log: string): Promise<string[][]> {
  return (await readFile(log, "utf8")).trim().split("\n").map((line) => line.split(" "));
}

async function writeExecutable(directory: string, name: string, lines: string[]) {
  const file = path.join(directory, name);
  await writeFile(file, lines.join("\n"), "utf8");
  await chmod(file, 0o755);
}

function runLauncher(env: NodeJS.ProcessEnv) {
  return runProcess("bash", [LAUNCHER], env);
}

function runProcess(command: string, args: string[], env: NodeJS.ProcessEnv = process.env) {
  return new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve, reject) => {
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
