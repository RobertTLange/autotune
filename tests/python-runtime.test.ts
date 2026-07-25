import { chmod, mkdir, mkdtemp, readFile, realpath, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { ensurePythonRuntime } from "../src/python-runtime.js";

describe("ensurePythonRuntime", () => {
  it("reuses a compatible bootstrap interpreter without provisioning", async () => {
    const fixture = await createFixture({ withUv: true });

    const runtime = await ensurePythonRuntime({
      includeCmaes: false,
      bootstrapPython: fixture.python,
      cacheDir: fixture.cache,
      env: { ...process.env, PATH: fixture.path, FAKE_SYSTEM_OPTUNA: "1", FAKE_RUNTIME_LOG: fixture.log }
    });

    expect(runtime).toMatchObject({
      pythonVersion: "3.12.1",
      optunaVersion: "4.8.0",
      managed: false
    });
    expect(await realpath(runtime.python)).toBe(await realpath(fixture.python));
    expect(await readLog(fixture.log)).not.toContainEqual(expect.arrayContaining(["venv"]));
  });

  it("provisions and reuses a private uv environment when packages are missing", async () => {
    const fixture = await createFixture({ withUv: true });
    const options = {
      includeCmaes: true,
      bootstrapPython: fixture.python,
      cacheDir: fixture.cache,
      env: { ...process.env, PATH: fixture.path, FAKE_RUNTIME_LOG: fixture.log }
    };

    const first = await ensurePythonRuntime(options);
    const firstLog = await readLog(fixture.log);
    const second = await ensurePythonRuntime(options);

    expect(first).toMatchObject({
      pythonVersion: "3.12.1",
      optunaVersion: "4.8.0",
      cmaesVersion: "0.12.0",
      managed: true
    });
    expect(first.python).toContain(fixture.cache);
    expect(second).toEqual(first);
    expect(firstLog.filter((args) => args[0] === "venv")).toHaveLength(1);
    expect(firstLog.find((args) => args[0] === "pip")).toEqual(
      expect.arrayContaining(["--require-hashes", "--requirements"])
    );
    const secondLog = await readLog(fixture.log);
    expect(secondLog.filter((args) => args[0] === "venv")).toHaveLength(1);
    expect(secondLog.filter((args) => args[0] === "pip")).toHaveLength(1);
  });

  it("falls back to stdlib venv and pip when uv is unavailable", async () => {
    const fixture = await createFixture({ withUv: false });

    const runtime = await ensurePythonRuntime({
      includeCmaes: false,
      bootstrapPython: fixture.python,
      cacheDir: fixture.cache,
      env: { ...process.env, PATH: fixture.path, FAKE_RUNTIME_LOG: fixture.log }
    });

    const log = await readLog(fixture.log);
    expect(runtime).toMatchObject({ optunaVersion: "4.8.0", managed: true });
    expect(log).toContainEqual(expect.arrayContaining(["-m", "venv"]));
    expect(log).toContainEqual(expect.arrayContaining(["-m", "pip", "install", "--require-hashes"]));
  });

  it("serializes concurrent provisioning for the same runtime", async () => {
    const fixture = await createFixture({ withUv: true });
    const options = {
      includeCmaes: false,
      bootstrapPython: fixture.python,
      cacheDir: fixture.cache,
      env: { ...process.env, PATH: fixture.path, FAKE_RUNTIME_LOG: fixture.log }
    };

    const runtimes = await Promise.all([
      ensurePythonRuntime(options),
      ensurePythonRuntime(options),
      ensurePythonRuntime(options)
    ]);

    expect(new Set(runtimes.map((runtime) => runtime.python))).toHaveLength(1);
    expect((await readLog(fixture.log)).filter((args) => args[0] === "venv")).toHaveLength(1);
  });

  it("recovers from corrupt published metadata", async () => {
    const fixture = await createFixture({ withUv: true });
    const options = {
      includeCmaes: false,
      bootstrapPython: fixture.python,
      cacheDir: fixture.cache,
      env: { ...process.env, PATH: fixture.path, FAKE_RUNTIME_LOG: fixture.log }
    };
    const first = await ensurePythonRuntime(options);
    const runtimeDirectory = path.dirname(path.dirname(path.dirname(first.python)));
    await writeFile(path.join(runtimeDirectory, "runtime.json"), "{truncated", "utf8");

    const recovered = await ensurePythonRuntime(options);

    expect(recovered).toMatchObject({ optunaVersion: "4.8.0", managed: true });
    expect((await readLog(fixture.log)).filter((args) => args[0] === "venv")).toHaveLength(2);
  });

  it.skipIf(process.platform === "win32")("rejects a cache writable by other users", async () => {
    const fixture = await createFixture({ withUv: true });
    await mkdir(fixture.cache, { mode: 0o777 });
    await chmod(fixture.cache, 0o777);

    await expect(ensurePythonRuntime({
      includeCmaes: false,
      bootstrapPython: fixture.python,
      cacheDir: fixture.cache,
      env: { ...process.env, PATH: fixture.path, FAKE_RUNTIME_LOG: fixture.log }
    })).rejects.toThrow(/writable by group or others/i);
  });

  it("isolates Python probes from caller import settings", async () => {
    const fixture = await createFixture({ withUv: true });

    await ensurePythonRuntime({
      includeCmaes: false,
      bootstrapPython: fixture.python,
      cacheDir: fixture.cache,
      env: {
        ...process.env,
        PATH: fixture.path,
        PYTHONPATH: "/tmp/untrusted-python-path",
        FAKE_SYSTEM_OPTUNA: "1",
        FAKE_RUNTIME_LOG: fixture.log
      }
    });

    const probes = (await readLog(fixture.log)).filter((args) => args.includes("-c"));
    expect(probes).not.toHaveLength(0);
    expect(probes.every((args) => args.includes("-I"))).toBe(true);
  });

  it("falls back to another Python command when the default is unsupported", async () => {
    const fixture = await createFixture({ withUv: true });
    const fallbackPython = path.join(fixture.bin, "python");
    await writeExecutable(fixture.python, fakeUnsupportedPythonSource());
    await writeExecutable(fallbackPython, fakePythonSource());

    const runtime = await ensurePythonRuntime({
      includeCmaes: false,
      cacheDir: fixture.cache,
      env: {
        ...process.env,
        PATH: fixture.path,
        FAKE_SYSTEM_OPTUNA: "1",
        FAKE_RUNTIME_LOG: fixture.log
      }
    });

    expect(await realpath(runtime.python)).toBe(await realpath(fallbackPython));
  });

  it("reuses a compatible fallback before provisioning from the default", async () => {
    const fixture = await createFixture({ withUv: true });
    const fallbackPython = path.join(fixture.bin, "python");
    await writeExecutable(
      fixture.python,
      fakePythonSource().replace('process.env.FAKE_SYSTEM_OPTUNA === "1"', "false")
    );
    await writeExecutable(fallbackPython, fakePythonSource());

    const runtime = await ensurePythonRuntime({
      includeCmaes: false,
      cacheDir: fixture.cache,
      env: {
        ...process.env,
        PATH: fixture.path,
        FAKE_SYSTEM_OPTUNA: "1",
        FAKE_RUNTIME_LOG: fixture.log
      }
    });

    expect(await realpath(runtime.python)).toBe(await realpath(fallbackPython));
    expect((await readLog(fixture.log)).filter((args) => args[0] === "venv")).toHaveLength(0);
  });
});

async function createFixture(options: { withUv: boolean }): Promise<{
  bin: string;
  cache: string;
  log: string;
  path: string;
  python: string;
}> {
  const root = await mkdtemp(path.join(tmpdir(), "autotune-python-runtime-"));
  const bin = path.join(root, "bin");
  const cache = path.join(root, "cache");
  const log = path.join(root, "commands.jsonl");
  await mkdir(bin);
  const python = path.join(bin, "python3");
  await writeExecutable(python, fakePythonSource());
  if (options.withUv) {
    await writeExecutable(path.join(bin, "uv"), fakeUvSource());
  }
  const inheritedPath = options.withUv ? process.env.PATH ?? "" : "/usr/bin:/bin";
  return { bin, cache, log, path: `${bin}${path.delimiter}${inheritedPath}`, python };
}

async function readLog(filePath: string): Promise<string[][]> {
  const source = await readFile(filePath, "utf8").catch(() => "");
  return source.trim() ? source.trim().split("\n").map((line) => JSON.parse(line) as string[]) : [];
}

async function writeExecutable(filePath: string, source: string): Promise<void> {
  await writeFile(filePath, source, "utf8");
  await chmod(filePath, 0o755);
}

function fakePythonSource(): string {
  return `#!${process.execPath}
import { appendFileSync, chmodSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const executable = fileURLToPath(import.meta.url);
const args = process.argv.slice(2);
if (process.env.FAKE_RUNTIME_LOG) appendFileSync(process.env.FAKE_RUNTIME_LOG, JSON.stringify(args) + "\\n");
const moduleIndex = args.indexOf("-m");
if (moduleIndex >= 0 && args[moduleIndex + 1] === "venv") {
  const target = args[moduleIndex + 2];
  const managed = path.join(target, process.platform === "win32" ? "Scripts" : "bin", process.platform === "win32" ? "python.exe" : "python");
  mkdirSync(path.dirname(managed), { recursive: true });
  writeFileSync(managed, ${JSON.stringify(fakeManagedPythonSource())});
  chmodSync(managed, 0o755);
  process.exit(0);
}
const codeIndex = args.indexOf("-c");
const code = codeIndex >= 0 ? args[codeIndex + 1] || "" : "";
if (codeIndex >= 0 && code.includes("platform.machine")) {
  console.log(JSON.stringify({ executable, version: "3.12.1", implementation: "cpython", platform: process.platform, arch: process.arch, macVersion: "14.0" }));
  process.exit(0);
}

if (codeIndex >= 0 && code.includes("optuna") && process.env.FAKE_SYSTEM_OPTUNA === "1") {
  console.log(JSON.stringify({ optuna: "4.8.0" }));
  process.exit(0);
}
process.exit(1);
`;
}

function fakeUnsupportedPythonSource(): string {
  return `#!${process.execPath}
import { fileURLToPath } from "node:url";
const executable = fileURLToPath(import.meta.url);
if (process.argv.includes("-c")) {
  console.log(JSON.stringify({ executable, version: "3.8.0", implementation: "cpython", platform: process.platform, arch: process.arch, macVersion: "14.0" }));
  process.exit(0);
}
process.exit(1);
`;
}

function fakeManagedPythonSource(): string {
  return `#!${process.execPath}
const { appendFileSync } = require("node:fs");
const args = process.argv.slice(2);
if (process.env.FAKE_RUNTIME_LOG) appendFileSync(process.env.FAKE_RUNTIME_LOG, JSON.stringify(args) + "\\n");
const moduleIndex = args.indexOf("-m");
if (moduleIndex >= 0 && args[moduleIndex + 1] === "pip") process.exit(0);
if (args.includes("-c")) {
  console.log(JSON.stringify({ optuna: "4.8.0", cmaes: "0.12.0" }));
  process.exit(0);
}
process.exit(1);
`;
}

function fakeUvSource(): string {
  return `#!${process.execPath}
import { appendFileSync, chmodSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
const args = process.argv.slice(2);
if (process.env.FAKE_RUNTIME_LOG) appendFileSync(process.env.FAKE_RUNTIME_LOG, JSON.stringify(args) + "\\n");
if (args[0] === "--version") {
  console.log("uv 1.0.0");
  process.exit(0);
}
if (args[0] === "venv") {
  const target = args[args.length - 1];
  const managed = path.join(target, process.platform === "win32" ? "Scripts" : "bin", process.platform === "win32" ? "python.exe" : "python");
  mkdirSync(path.dirname(managed), { recursive: true });
  writeFileSync(managed, ${JSON.stringify(fakeManagedPythonSource())});
  chmodSync(managed, 0o755);
  process.exit(0);
}
if (args[0] === "pip") process.exit(0);
process.exit(1);
`;
}
