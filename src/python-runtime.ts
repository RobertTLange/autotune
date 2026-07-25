import { createHash, randomBytes } from "node:crypto";
import {
  chmod,
  lstat,
  mkdir,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  utimes,
  writeFile
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { runCommand } from "./process.js";

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

interface PythonIdentity {
  executable: string;
  version: string;
  implementation: string;
  platform: string;
  arch: string;
}

interface PackageVersions {
  optuna: string;
  cmaes?: string;
}

interface ProvisioningMarker {
  pid: number;
  hostname: string;
  startedAt: number;
  token: string;
}

interface ProvisioningClaim extends ProvisioningMarker {
  directory: string;
  heartbeat: NodeJS.Timeout;
  lost: boolean;
}

const OPTUNA_VERSION = "4.8.0";
const CMAES_VERSION = "0.12.0";
const CACHE_SCHEMA = 2;
const CLAIM_HEARTBEAT_MS = 2_000;
const CLAIM_EXPIRY_MS = 15_000;
const WAIT_INTERVAL_MS = 200;
const PROBE_TIMEOUT_MS = 30_000;
const COMMAND_TIMEOUT_MS = 10 * 60 * 1000;
const READY_FILE = "runtime.json";
const LOCK_FILE = "owner.json";
const PYPI_INDEX = "https://pypi.org/simple";

const IDENTITY_SCRIPT = [
  "import json, platform, sys",
  "print(json.dumps({",
  "  'executable': sys.executable,",
  "  'version': platform.python_version(),",
  "  'implementation': platform.python_implementation().lower(),",
  "  'platform': sys.platform,",
  "  'arch': platform.machine(),",
  "}))"
].join("\n");

export async function ensurePythonRuntime(options: PythonRuntimeOptions): Promise<PythonRuntime> {
  const callerEnv = options.env ?? process.env;
  const toolEnv = isolatedToolEnvironment(callerEnv);
  const bootstrapPython = options.bootstrapPython
    ?? callerEnv.AUTOTUNE_PYTHON
    ?? (process.platform === "win32" ? "python" : "python3");
  const identity = await readPythonIdentity(bootstrapPython, toolEnv);
  const existing = await readPackageVersions(identity.executable, options.includeCmaes, toolEnv);
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
  const cached = await readCachedRuntime(runtimeDir, identity.version, options.includeCmaes, toolEnv);
  if (cached) {
    return cached;
  }
  const claimsDir = `${runtimeDir}.claims`;
  await ensureOwnedPrivateDirectory(claimsDir);
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
        return winner;
      }

      const environmentDir = path.join(runtimeDir, `env-${claim.token}`);
      await removeInvalidReadyFile(runtimeDir);
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
    throw new Error(`Python 3.9 or newer is required: could not execute ${python}`, { cause: error });
  }
  const identity = parseJson<PythonIdentity>(stdout, "Python identity");
  const [major = 0, minor = 0] = identity.version.split(".").map(Number);
  if (!identity.executable || major < 3 || (major === 3 && minor < 9)) {
    throw new Error(`Python 3.9 or newer is required, found ${identity.version || "unknown"}`);
  }
  return identity;
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
  } catch {
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

async function ensureOwnedPrivateDirectory(directory: string): Promise<string> {
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const canonical = await realpath(directory);
  const metadata = await lstat(canonical);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    throw new Error(`managed Python cache must be a real directory: ${directory}`);
  }
  if (process.platform !== "win32") {
    if ((metadata.mode & 0o022) !== 0) {
      throw new Error(`managed Python cache must not be writable by group or others: ${directory}`);
    }
    if (process.getuid && metadata.uid !== process.getuid()) {
      throw new Error(`managed Python cache must be owned by the current user: ${directory}`);
    }
    await chmod(canonical, 0o700);
    await validateSecureAncestors(canonical);
  }
  return canonical;
}

async function validateSecureAncestors(directory: string): Promise<void> {
  const currentUser = process.getuid?.();
  let ancestor = path.dirname(directory);
  for (;;) {
    const metadata = await lstat(ancestor);
    const trustedOwner = metadata.uid === currentUser || metadata.uid === 0;
    const writableByOthers = (metadata.mode & 0o022) !== 0;
    const sticky = (metadata.mode & 0o1000) !== 0;
    if (!metadata.isDirectory() || metadata.isSymbolicLink() || !trustedOwner || (writableByOthers && !sticky)) {
      throw new Error(`managed Python cache has an unsafe parent directory: ${ancestor}`);
    }
    const parent = path.dirname(ancestor);
    if (parent === ancestor) {
      return;
    }
    ancestor = parent;
  }
}

async function validateWindowsCacheLocation(
  cacheRoot: string,
  env: NodeJS.ProcessEnv
): Promise<void> {
  if (process.platform !== "win32") {
    return;
  }
  const profileCandidates = [env.LOCALAPPDATA, env.USERPROFILE, os.homedir()].filter(
    (candidate): candidate is string => Boolean(candidate?.trim())
  );
  for (const candidate of profileCandidates) {
    const profile = await realpath(candidate).catch(() => undefined);
    if (profile && isPathInside(profile, cacheRoot)) {
      return;
    }
  }
  throw new Error("managed Python cache must be inside the current Windows user profile");
}

function isPathInside(parent: string, candidate: string): boolean {
  const relative = path.relative(parent, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
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

async function createProvisioningClaim(claimsDir: string): Promise<ProvisioningClaim> {
  const token = randomBytes(16).toString("hex");
  const marker: ProvisioningMarker = {
    pid: process.pid,
    hostname: os.hostname(),
    startedAt: Date.now(),
    token
  };
  const staging = path.join(claimsDir, `.${token}.staging`);
  const directory = path.join(claimsDir, token);
  await mkdir(staging, { mode: 0o700 });
  try {
    await writePrivateJsonAtomic(path.join(staging, LOCK_FILE), marker, token);
    await rename(staging, directory);
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
  const claim = { ...marker, directory, lost: false } as ProvisioningClaim;
  claim.heartbeat = setInterval(() => void refreshProvisioningClaim(claim), CLAIM_HEARTBEAT_MS);
  claim.heartbeat.unref();
  return claim;
}

async function refreshProvisioningClaim(claim: ProvisioningClaim): Promise<void> {
  try {
    const metadata = await lstat(claim.directory);
    if (Date.now() - metadata.mtimeMs > CLAIM_EXPIRY_MS) {
      claim.lost = true;
      clearInterval(claim.heartbeat);
      return;
    }
    const now = new Date();
    await utimes(claim.directory, now, now);
  } catch {
    claim.lost = true;
    clearInterval(claim.heartbeat);
  }
}

async function claimOwnsQueue(claimsDir: string, claim: ProvisioningClaim): Promise<boolean> {
  if (claim.lost) {
    return false;
  }
  const active: ProvisioningMarker[] = [];
  for (const entry of await readdir(claimsDir, { withFileTypes: true })) {
    if (!entry.isDirectory() || !/^[a-f0-9]{32}$/.test(entry.name)) {
      continue;
    }
    const directory = path.join(claimsDir, entry.name);
    try {
      const [metadata, marker] = await Promise.all([
        lstat(directory),
        readFile(path.join(directory, LOCK_FILE), "utf8").then((source) =>
          parseJson<ProvisioningMarker>(source, "runtime provisioning marker")
        )
      ]);
      const fresh = Date.now() - metadata.mtimeMs <= CLAIM_EXPIRY_MS;
      const locallyAlive = marker.hostname !== os.hostname() || processIsAlive(marker.pid);
      if (fresh && locallyAlive && marker.token === entry.name) {
        active.push(marker);
      }
    } catch {
      // Incomplete and expired claims never become queue owners.
    }
  }
  active.sort((left, right) => left.startedAt - right.startedAt || left.token.localeCompare(right.token));
  return active[0]?.token === claim.token;
}

async function releaseProvisioningClaim(claim: ProvisioningClaim): Promise<void> {
  clearInterval(claim.heartbeat);
  await rm(claim.directory, { recursive: true, force: true }).catch(() => undefined);
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return typeof error === "object" && error !== null && "code" in error && error.code === "EPERM";
  }
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
      await rm(input.environmentDir, { recursive: true, force: true });
      try {
        await provisionWithVenv(input);
        return;
      } catch (venvError) {
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
    throw new Error("failed to provision the managed Python runtime with python -m venv and pip", {
      cause: error
    });
  }
}

async function uvAvailable(env: NodeJS.ProcessEnv, cwd: string): Promise<boolean> {
  try {
    await runCommand("uv", ["--version"], { cwd, env, timeoutMs: PROBE_TIMEOUT_MS });
    return true;
  } catch {
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

async function writePrivateJsonAtomic(filePath: string, value: unknown, token: string): Promise<void> {
  const temporary = path.join(path.dirname(filePath), `.${path.basename(filePath)}.${token}.tmp`);
  await writeFile(temporary, `${JSON.stringify(value)}\n`, { encoding: "utf8", mode: 0o600 });
  if (process.platform !== "win32") {
    await chmod(temporary, 0o600);
  }
  await rename(temporary, filePath);
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
