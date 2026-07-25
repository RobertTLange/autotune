import { randomBytes } from "node:crypto";
import {
  chmod,
  lstat,
  mkdir,
  opendir,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  utimes,
  writeFile
} from "node:fs/promises";
import type { Dirent } from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";

interface ProvisioningMarker {
  pid: number;
  hostname: string;
  startedAt: number;
  token: string;
}

export interface ProvisioningClaim extends ProvisioningMarker {
  directory: string;
  heartbeat: NodeJS.Timeout;
  lost: boolean;
}

const CLAIM_HEARTBEAT_MS = 2_000;
const CLAIM_EXPIRY_MS = 15_000;
const ORPHAN_ENVIRONMENT_RETENTION_MS = 24 * 60 * 60 * 1000;
const MAX_CLEANUP_ENTRIES_PER_PASS = 16;
const OWNER_FILE = "owner.json";
const CLAIM_NAME = /^[a-f0-9]{32}$/;
const STAGING_NAME = /^\.([a-f0-9]{32})\.staging$/;
const ENVIRONMENT_NAME = /^env-([a-f0-9]{32})$/;

export async function ensureOwnedPrivateDirectory(directory: string): Promise<string> {
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

export async function validateWindowsCacheLocation(
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

export async function createProvisioningClaim(claimsDir: string): Promise<ProvisioningClaim> {
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
    await writePrivateJsonAtomic(path.join(staging, OWNER_FILE), marker, token);
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

export async function claimOwnsQueue(
  claimsDir: string,
  claim: ProvisioningClaim
): Promise<boolean> {
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
        readFile(path.join(directory, OWNER_FILE), "utf8").then(parseProvisioningMarker)
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

export async function releaseProvisioningClaim(claim: ProvisioningClaim): Promise<void> {
  clearInterval(claim.heartbeat);
  await rm(claim.directory, { recursive: true, force: true }).catch(() => undefined);
}

export async function cleanupAbandonedRuntimeEntries(input: {
  runtimeDir: string;
  claimsDir: string;
  activeToken?: string;
  publishedPython?: string;
  now?: number;
}): Promise<void> {
  const now = input.now ?? Date.now();
  const activeToken = input.activeToken && CLAIM_NAME.test(input.activeToken) ? input.activeToken : undefined;
  let remaining = MAX_CLEANUP_ENTRIES_PER_PASS;

  const claims = streamCandidateEntries(
    input.claimsDir,
    (entry) => CLAIM_NAME.test(entry.name) || STAGING_NAME.test(entry.name)
  );
  for await (const entry of claims) {
    if (remaining === 0) break;
    const claimToken = CLAIM_NAME.test(entry.name) ? entry.name : undefined;
    const stagingToken = STAGING_NAME.exec(entry.name)?.[1];
    if (!entry.isDirectory() || (!claimToken && !stagingToken)) continue;
    if (claimToken === activeToken) continue;
    const directory = path.join(input.claimsDir, entry.name);
    try {
      const metadata = await lstat(directory);
      if (!metadata.isDirectory() || metadata.isSymbolicLink()) continue;
      const expired = now - metadata.mtimeMs > CLAIM_EXPIRY_MS;
      if (claimToken) {
        const marker = await readFile(path.join(directory, OWNER_FILE), "utf8")
          .then(parseProvisioningMarker)
          .catch(() => undefined);
        const alive = marker
          && marker.token === claimToken
          && marker.hostname === os.hostname()
          && processIsAlive(marker.pid);
        const remotelyActive = marker
          && marker.token === claimToken
          && marker.hostname !== os.hostname();
        if (!expired && (alive || remotelyActive)) {
          continue;
        }
      }
      if (expired && await removeRealDirectory(directory)) remaining -= 1;
    } catch {
      // Cleanup is best-effort; provisioning remains authoritative.
    }
  }

  const publishedEnvironment = publishedEnvironmentName(input.runtimeDir, input.publishedPython);
  const runtimeEntries = streamCandidateEntries(
    input.runtimeDir,
    (entry) => ENVIRONMENT_NAME.test(entry.name)
  );
  remaining = MAX_CLEANUP_ENTRIES_PER_PASS;
  for await (const entry of runtimeEntries) {
    if (remaining === 0) break;
    const token = ENVIRONMENT_NAME.exec(entry.name)?.[1];
    if (
      !token
      || !entry.isDirectory()
      || token === activeToken
      || entry.name === publishedEnvironment
      || await claimIsActive(input.claimsDir, token, now)
    ) continue;
    const directory = path.join(input.runtimeDir, entry.name);
    try {
      const metadata = await lstat(directory);
      if (
        metadata.isDirectory()
        && !metadata.isSymbolicLink()
        && now - metadata.mtimeMs > ORPHAN_ENVIRONMENT_RETENTION_MS
        && await removeRealDirectory(directory)
      ) {
        remaining -= 1;
      }
    } catch {
      // Cleanup is best-effort; provisioning remains authoritative.
    }
  }
}

export async function writePrivateJsonAtomic(
  filePath: string,
  value: unknown,
  token: string
): Promise<void> {
  const temporary = path.join(path.dirname(filePath), `.${path.basename(filePath)}.${token}.tmp`);
  await writeFile(temporary, `${JSON.stringify(value)}\n`, { encoding: "utf8", mode: 0o600 });
  if (process.platform !== "win32") {
    await chmod(temporary, 0o600);
  }
  await rename(temporary, filePath);
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

function parseProvisioningMarker(source: string): ProvisioningMarker {
  const value = JSON.parse(source.trim()) as Partial<ProvisioningMarker>;
  if (
    !Number.isInteger(value.pid)
    || (value.pid ?? 0) < 1
    || typeof value.hostname !== "string"
    || !value.hostname
    || typeof value.startedAt !== "number"
    || !Number.isFinite(value.startedAt)
    || typeof value.token !== "string"
    || !CLAIM_NAME.test(value.token)
  ) {
    throw new Error("invalid provisioning claim marker");
  }
  return value as ProvisioningMarker;
}

function publishedEnvironmentName(runtimeDir: string, python: string | undefined): string | undefined {
  if (!python || !path.isAbsolute(python)) return undefined;
  const relative = path.relative(runtimeDir, python);
  const [environmentName] = relative.split(path.sep);
  const expectedSuffix = process.platform === "win32"
    ? path.join("Scripts", "python.exe")
    : path.join("bin", "python");
  return ENVIRONMENT_NAME.test(environmentName ?? "")
    && relative === path.join(environmentName as string, expectedSuffix)
    ? environmentName
    : undefined;
}

async function removeRealDirectory(directory: string): Promise<boolean> {
  try {
    const metadata = await lstat(directory);
    if (!metadata.isDirectory() || metadata.isSymbolicLink()) return false;
    await rm(directory, { recursive: true, force: true });
    return true;
  } catch {
    return false;
  }
}

async function claimIsActive(claimsDir: string, token: string, now: number): Promise<boolean> {
  const directory = path.join(claimsDir, token);
  try {
    const [metadata, marker] = await Promise.all([
      lstat(directory),
      readFile(path.join(directory, OWNER_FILE), "utf8").then(parseProvisioningMarker)
    ]);
    if (!metadata.isDirectory() || metadata.isSymbolicLink() || now - metadata.mtimeMs > CLAIM_EXPIRY_MS) {
      return false;
    }
    return marker.token === token
      && (marker.hostname !== os.hostname() || processIsAlive(marker.pid));
  } catch {
    return false;
  }
}

async function* streamCandidateEntries(
  directory: string,
  isCandidate: (entry: Dirent) => boolean
): AsyncGenerator<Dirent> {
  try {
    const handle = await opendir(directory);
    for await (const entry of handle) {
      if (isCandidate(entry)) yield entry;
    }
  } catch {
    return;
  }
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return typeof error === "object" && error !== null && "code" in error && error.code === "EPERM";
  }
}

function isPathInside(parent: string, candidate: string): boolean {
  const relative = path.relative(parent, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}
