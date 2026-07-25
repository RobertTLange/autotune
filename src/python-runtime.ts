import { createHash } from "node:crypto";
import { readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { isCommandInterruptedError, runCommand } from "./process.js";
import {
  claimOwnsQueue,
  cleanupAbandonedRuntimeEntries,
  createProvisioningClaim,
  ensureOwnedPrivateDirectory,
  releaseProvisioningClaim,
  validateWindowsCacheLocation,
  writePrivateJsonAtomic
} from "./python-runtime-cache.js";

export interface PythonRuntime {
  python: string;
  pythonVersion: string;
  optunaVersion: string;
  cmaesVersion?: string;
  managed: boolean;
}

export interface PythonRuntimeOptions {
  includeCmaes: boolean;
  bootstrapPython?: string;
  cacheDir?: string;
  env?: NodeJS.ProcessEnv;
}

export interface PythonInterpreter {
  python: string;
  pythonVersion: string;
}

interface PythonIdentity {
  executable: string;
  version: string;
  implementation: string;
  platform: string;
  arch: string;
  macVersion?: string;
}

interface PackageVersions {
  optuna: string;
  cmaes?: string;
}

const OPTUNA_VERSION = "4.8.0";
const CMAES_VERSION = "0.12.0";
const CACHE_SCHEMA = 2;
const WAIT_INTERVAL_MS = 200;
const PROBE_TIMEOUT_MS = 30_000;
const COMMAND_TIMEOUT_MS = 10 * 60 * 1000;
const READY_FILE = "runtime.json";
const PYPI_INDEX = "https://pypi.org/simple";

const IDENTITY_SCRIPT = [
  "import json, platform, sys",
  "print(json.dumps({",
  "  'executable': sys.executable,",
  "  'version': platform.python_version(),",
  "  'implementation': platform.python_implementation().lower(),",
  "  'platform': sys.platform,",
  "  'arch': platform.machine(),",
  "  'macVersion': platform.mac_ver()[0],",
  "}))"
].join("\n");

export async function ensurePythonRuntime(options: PythonRuntimeOptions): Promise<PythonRuntime> {
  const callerEnv = options.env ?? process.env;
  const toolEnv = isolatedToolEnvironment(callerEnv);
  const { identity, existing } = await resolvePythonRuntimeCandidate(
    options.bootstrapPython,
    callerEnv,
    toolEnv,
    options.includeCmaes
  );
  if (existing) {
    return toRuntime(identity.executable, identity.version, existing, false);
  }

  const requirementsFile = runtimeRequirementsFile();
  const requirementsDigest = createHash("sha256")
    .update(await readFile(requirementsFile))
    .digest("hex");
  const cacheRoot = await ensureOwnedPrivateDirectory(
    path.resolve(options.cacheDir ?? defaultCacheRoot(callerEnv))
  );
  await validateWindowsCacheLocation(cacheRoot, callerEnv);
  const runtimeDir = path.join(cacheRoot, runtimeCacheKey(identity, requirementsDigest));
  await ensureOwnedPrivateDirectory(runtimeDir);
  const claimsDir = `${runtimeDir}.claims`;
  await ensureOwnedPrivateDirectory(claimsDir);
  const cached = await readCachedRuntime(runtimeDir, identity.version, options.includeCmaes, toolEnv);
  if (cached) {
    await cleanupAbandonedRuntimeEntries({
      runtimeDir,
      claimsDir,
      publishedPython: cached.python
    });
    return cached;
  }
  let claim = await createProvisioningClaim(claimsDir);

  try {
    for (;;) {
      const published = await readCachedRuntime(runtimeDir, identity.version, options.includeCmaes, toolEnv);
      if (published) {
        return published;
      }
      if (!await claimOwnsQueue(claimsDir, claim)) {
        if (claim.lost) {
          await releaseProvisioningClaim(claim);
          claim = await createProvisioningClaim(claimsDir);
          continue;
        }
        await delay();
        continue;
      }

      const winner = await readCachedRuntime(
        runtimeDir,
        identity.version,
        options.includeCmaes,
        toolEnv
      );
      if (winner) {
        await cleanupAbandonedRuntimeEntries({
          runtimeDir,
          claimsDir,
          activeToken: claim.token,
          publishedPython: winner.python
        });
        return winner;
      }

      await removeInvalidReadyFile(runtimeDir);
      await cleanupAbandonedRuntimeEntries({
        runtimeDir,
        claimsDir,
        activeToken: claim.token
      });
      const environmentDir = path.join(runtimeDir, `env-${claim.token}`);
      const python = managedPythonPath(environmentDir);
      try {
        await provisionRuntime({
          environmentDir,
          python,
          bootstrapPython: identity.executable,
          requirementsFile,
          env: toolEnv,
          cwd: cacheRoot
        });
        await ensureOwnedPrivateDirectory(environmentDir);
        const versions = await readPackageVersions(python, options.includeCmaes, toolEnv);
        if (!versions) {
          throw new Error("managed Python runtime did not contain supported Optuna packages");
        }
        if (!await claimOwnsQueue(claimsDir, claim)) {
          await rm(environmentDir, { recursive: true, force: true });
          await delay();
          continue;
        }
        const runtime = toRuntime(python, identity.version, versions, true);
        await writePrivateJsonAtomic(path.join(runtimeDir, READY_FILE), runtime, claim.token);
        await cleanupAbandonedRuntimeEntries({
          runtimeDir,
          claimsDir,
          activeToken: claim.token,
          publishedPython: runtime.python
        });
        return runtime;
      } catch (error) {
        await rm(environmentDir, { recursive: true, force: true });
        throw error;
      }
    }
  } finally {
    await releaseProvisioningClaim(claim);
  }
}

export async function inspectPythonInterpreter(options: {
  bootstrapPython?: string;
  env?: NodeJS.ProcessEnv;
} = {}): Promise<PythonInterpreter> {
  const callerEnv = options.env ?? process.env;
  const toolEnv = isolatedToolEnvironment(callerEnv);
  const candidates = pythonCandidates(options.bootstrapPython, callerEnv);
  const errors: Error[] = [];
  for (const candidate of candidates) {
    try {
      const identity = await readPythonIdentity(candidate, toolEnv);
      return { python: identity.executable, pythonVersion: identity.version };
    } catch (error) {
      if (isCommandInterruptedError(error)) throw error;
      errors.push(labeledPythonError(candidate, error));
    }
  }
  throw pythonCandidateError(candidates, errors);
}

function runtimeRequirementsFile(): string {
  return path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
    "resources",
    "python-runtime.lock"
  );
}

async function readPythonIdentity(python: string, env: NodeJS.ProcessEnv): Promise<PythonIdentity> {
  let stdout: string;
  try {
    ({ stdout } = await runCommand(python, ["-I", "-c", IDENTITY_SCRIPT], {
      cwd: os.tmpdir(),
      env,
      timeoutMs: PROBE_TIMEOUT_MS
    }));
  } catch (error) {
    if (isCommandInterruptedError(error)) throw error;
    throw new Error(`Python 3.9 or newer is required: could not execute ${python}`, { cause: error });
  }
  const identity = parseJson<PythonIdentity>(stdout, "Python identity");
  const [major = 0, minor = 0] = identity.version.split(".").map(Number);
  if (
    !identity.executable
    || major < 3
    || (major === 3 && minor < 9)
    || (identity.platform === "darwin" && !identity.macVersion)
  ) {
    throw new Error(`Python 3.9 or newer is required, found ${identity.version || "unknown"}`);
  }
  return identity;
}

async function resolvePythonRuntimeCandidate(
  configuredPython: string | undefined,
  callerEnv: NodeJS.ProcessEnv,
  toolEnv: NodeJS.ProcessEnv,
  includeCmaes: boolean
): Promise<{ identity: PythonIdentity; existing?: PackageVersions }> {
  const candidates = pythonCandidates(configuredPython, callerEnv);
  const errors: Error[] = [];
  let firstSupported: PythonIdentity | undefined;
  for (const candidate of candidates) {
    try {
      const identity = await readPythonIdentity(candidate, toolEnv);
      firstSupported ??= identity;
      const existing = await readPackageVersions(identity.executable, includeCmaes, toolEnv);
      if (existing) {
        return { identity, existing };
      }
    } catch (error) {
      if (isCommandInterruptedError(error)) throw error;
      errors.push(labeledPythonError(candidate, error));
    }
  }
  if (firstSupported) {
    return { identity: firstSupported };
  }
  throw pythonCandidateError(candidates, errors);
}

function pythonCandidates(configuredPython: string | undefined, env: NodeJS.ProcessEnv): string[] {
  const explicit = configuredPython ?? env.AUTOTUNE_PYTHON;
  return explicit
    ? [explicit]
    : process.platform === "win32"
      ? ["python", "python3"]
      : ["python3", "python"];
}

function labeledPythonError(candidate: string, error: unknown): Error {
  return new Error(`${candidate}: ${error instanceof Error ? error.message : String(error)}`, {
    cause: error
  });
}

function pythonCandidateError(candidates: string[], errors: Error[]): AggregateError {
  const details = errors.map((error) => error.message).join("; ");
  return new AggregateError(
    errors,
    `Python 3.9 or newer is required; tried ${candidates.join(", ")}${details ? ` (${details})` : ""}`
  );
}

async function readPackageVersions(
  python: string,
  includeCmaes: boolean,
  env: NodeJS.ProcessEnv
): Promise<PackageVersions | undefined> {
  const script = includeCmaes
    ? "import cmaes, json, optuna; print(json.dumps({'optuna': optuna.__version__, 'cmaes': cmaes.__version__}))"
    : "import json, optuna; print(json.dumps({'optuna': optuna.__version__}))";
  try {
    const { stdout } = await runCommand(python, ["-I", "-c", script], {
      cwd: os.tmpdir(),
      env,
      timeoutMs: PROBE_TIMEOUT_MS
    });
    const versions = parseJson<PackageVersions>(stdout, "Python package versions");
    return supportedVersions(versions, includeCmaes) ? versions : undefined;
  } catch (error) {
    if (isCommandInterruptedError(error)) throw error;
    return undefined;
  }
}

function supportedVersions(versions: PackageVersions, includeCmaes: boolean): boolean {
  if (versions.optuna !== OPTUNA_VERSION) {
    return false;
  }
  return !includeCmaes || versions.cmaes === CMAES_VERSION;
}

function toRuntime(
  python: string,
  pythonVersion: string,
  versions: PackageVersions,
  managed: boolean
): PythonRuntime {
  return {
    python,
    pythonVersion,
    optunaVersion: versions.optuna,
    ...(versions.cmaes ? { cmaesVersion: versions.cmaes } : {}),
    managed
  };
}

function defaultCacheRoot(env: NodeJS.ProcessEnv): string {
  const base = process.platform === "win32"
    ? env.LOCALAPPDATA?.trim() || path.join(os.homedir(), "AppData", "Local")
    : env.XDG_CACHE_HOME?.trim() || path.join(os.homedir(), ".cache");
  return path.join(base, "autotune", "python");
}

function runtimeCacheKey(identity: PythonIdentity, requirementsDigest: string): string {
  const digest = createHash("sha256")
    .update(JSON.stringify({ schema: CACHE_SCHEMA, identity, requirementsDigest }))
    .digest("hex")
    .slice(0, 16);
  return `${identity.implementation}-${identity.version}-${identity.platform}-${identity.arch}-${digest}`
    .replace(/[^A-Za-z0-9._-]/g, "-");
}

async function readCachedRuntime(
  runtimeDir: string,
  pythonVersion: string,
  includeCmaes: boolean,
  env: NodeJS.ProcessEnv
): Promise<PythonRuntime | undefined> {
  let recorded: PythonRuntime;
  try {
    recorded = parseJson<PythonRuntime>(
      await readFile(path.join(runtimeDir, READY_FILE), "utf8"),
      "runtime cache"
    );
  } catch {
    return undefined;
  }
  if (!recorded.managed || !isPublishedPythonPath(runtimeDir, recorded.python)) {
    return undefined;
  }
  try {
    await ensureOwnedPrivateDirectory(path.dirname(path.dirname(recorded.python)));
  } catch {
    return undefined;
  }
  const versions = await readPackageVersions(recorded.python, includeCmaes, env);
  return versions ? toRuntime(recorded.python, recorded.pythonVersion || pythonVersion, versions, true) : undefined;
}

function isPublishedPythonPath(runtimeDir: string, python: unknown): python is string {
  if (typeof python !== "string" || !path.isAbsolute(python)) {
    return false;
  }
  const relative = path.relative(runtimeDir, python);
  const [environmentName] = relative.split(path.sep);
  const expectedSuffix = process.platform === "win32"
    ? path.join("Scripts", "python.exe")
    : path.join("bin", "python");
  return /^env-[a-f0-9]{32}$/.test(environmentName ?? "")
    && relative === path.join(environmentName, expectedSuffix);
}

async function provisionRuntime(input: {
  environmentDir: string;
  python: string;
  bootstrapPython: string;
  requirementsFile: string;
  env: NodeJS.ProcessEnv;
  cwd: string;
}): Promise<void> {
  if (await uvAvailable(input.env, input.cwd)) {
    try {
      await runCommand("uv", [
        "venv", "--no-project", "--no-config", "--no-python-downloads",
        "--python", input.bootstrapPython, input.environmentDir
      ], { cwd: input.cwd, env: input.env, timeoutMs: COMMAND_TIMEOUT_MS });
      await runCommand("uv", [
        "pip", "install", "--no-config", "--only-binary", ":all:",
        "--default-index", PYPI_INDEX, "--require-hashes", "--python", input.python,
        "--requirements", input.requirementsFile
      ], { cwd: input.cwd, env: input.env, timeoutMs: COMMAND_TIMEOUT_MS });
      return;
    } catch (uvError) {
      if (isCommandInterruptedError(uvError)) throw uvError;
      await rm(input.environmentDir, { recursive: true, force: true });
      try {
        await provisionWithVenv(input);
        return;
      } catch (venvError) {
        if (isCommandInterruptedError(venvError)) throw venvError;
        throw new AggregateError(
          [uvError, venvError],
          "failed to provision the managed Python runtime with uv or python -m venv"
        );
      }
    }
  }
  await provisionWithVenv(input);
}

async function provisionWithVenv(input: {
  environmentDir: string;
  python: string;
  bootstrapPython: string;
  requirementsFile: string;
  env: NodeJS.ProcessEnv;
  cwd: string;
}): Promise<void> {
  try {
    await runCommand(input.bootstrapPython, ["-I", "-m", "venv", input.environmentDir], {
      cwd: input.cwd,
      env: input.env,
      timeoutMs: COMMAND_TIMEOUT_MS
    });
    await runCommand(input.python, [
      "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check", "--no-input",
      "--only-binary", ":all:", "--index-url", PYPI_INDEX, "--require-hashes",
      "--requirement", input.requirementsFile
    ], { cwd: input.cwd, env: input.env, timeoutMs: COMMAND_TIMEOUT_MS });
  } catch (error) {
    if (isCommandInterruptedError(error)) throw error;
    throw new Error("failed to provision the managed Python runtime with python -m venv and pip", {
      cause: error
    });
  }
}

async function uvAvailable(env: NodeJS.ProcessEnv, cwd: string): Promise<boolean> {
  try {
    await runCommand("uv", ["--version"], { cwd, env, timeoutMs: PROBE_TIMEOUT_MS });
    return true;
  } catch (error) {
    if (isCommandInterruptedError(error)) throw error;
    return false;
  }
}

function isolatedToolEnvironment(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const isolated = { ...env };
  for (const name of Object.keys(isolated)) {
    if (/^(PYTHON|PIP_|UV_)/i.test(name)) {
      delete isolated[name];
    }
  }
  isolated.PYTHONNOUSERSITE = "1";
  isolated.PYTHONSAFEPATH = "1";
  return isolated;
}

function managedPythonPath(environmentDir: string): string {
  return process.platform === "win32"
    ? path.join(environmentDir, "Scripts", "python.exe")
    : path.join(environmentDir, "bin", "python");
}

async function removeInvalidReadyFile(runtimeDir: string): Promise<void> {
  await rm(path.join(runtimeDir, READY_FILE), { force: true });
}

function parseJson<T>(source: string, label: string): T {
  try {
    return JSON.parse(source.trim()) as T;
  } catch {
    throw new Error(`${label} was not valid JSON`);
  }
}

function delay(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, WAIT_INTERVAL_MS));
}
