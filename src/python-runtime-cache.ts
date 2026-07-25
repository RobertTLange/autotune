import { randomBytes } from "node:crypto";
import {
  chmod,
  link,
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

interface ProvisioningLease {
  marker: ProvisioningMarker;
  tokens: string[];
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
const MAX_LEASE_GENERATIONS = 1_024;
const OWNER_FILE = "owner.json";
const ACTIVE_CLAIM_NAME = ".active";
const CLAIM_NAME = /^[a-f0-9]{32}$/;
const STAGING_NAME = /^\.([a-f0-9]{32})\.staging$/;
const ACTIVE_PREPARE_NAME = /^\.active\.prepare\.[a-f0-9]{32}$/;
const ACTIVE_RETIRED_NAME = /^\.active\.retired\.[a-f0-9]{32}$/;
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
  const now = Date.now();
  const lease = await readProvisioningLease(claimsDir);
  if (lease && await claimIsActive(claimsDir, lease.marker.token, now)) {
    return lease.marker.token === claim.token;
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
  if (active[0]?.token !== claim.token) {
    return false;
  }
  return acquireProvisioningLease(claimsDir, claim, lease?.marker.token);
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
  const publishedEnvironment = publishedEnvironmentName(input.runtimeDir, input.publishedPython);
  let remaining = MAX_CLEANUP_ENTRIES_PER_PASS;

  const claims = streamCandidateEntries(
    input.claimsDir,
    (entry) => CLAIM_NAME.test(entry.name)
      || STAGING_NAME.test(entry.name)
      || ACTIVE_PREPARE_NAME.test(entry.name)
      || ACTIVE_RETIRED_NAME.test(entry.name)
  );
  for await (const entry of claims) {
    if (remaining === 0) break;
    const claimToken = CLAIM_NAME.test(entry.name) ? entry.name : undefined;
    const stagingToken = STAGING_NAME.exec(entry.name)?.[1];
    const staleLease = ACTIVE_PREPARE_NAME.test(entry.name) || ACTIVE_RETIRED_NAME.test(entry.name);
    if (!entry.isDirectory() || (!claimToken && !stagingToken && !staleLease)) continue;
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
  if (publishedEnvironment) {
    await retireCompletedProvisioningLease(input.claimsDir);
  }

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

async function acquireProvisioningLease(
  claimsDir: string,
  claim: ProvisioningClaim,
  observedToken: string | undefined
): Promise<boolean> {
  const directory = path.join(claimsDir, ACTIVE_CLAIM_NAME);
  const prepared = path.join(claimsDir, `.active.prepare.${claim.token}`);
  try {
    await mkdir(prepared, { mode: 0o700 });
    await writePrivateJsonAtomic(path.join(prepared, OWNER_FILE), {
      pid: claim.pid,
      hostname: claim.hostname,
      startedAt: claim.startedAt,
      token: claim.token
    }, claim.token);
    if (!observedToken) {
      try {
        await rename(prepared, directory);
        return true;
      } catch (error) {
        if (await lstat(directory).then(() => true).catch(() => false)) return false;
        throw error;
      }
    }
    const current = await readProvisioningLease(claimsDir);
    if (!current || current.marker.token !== observedToken) return current?.marker.token === claim.token;
    if (await claimIsActive(claimsDir, current.marker.token, Date.now())) {
      return current.marker.token === claim.token;
    }
    if (current.tokens.includes(claim.token)) {
      claim.lost = true;
      clearInterval(claim.heartbeat);
      return false;
    }
    try {
      await link(
        path.join(prepared, OWNER_FILE),
        path.join(directory, successorFileName(current.marker.token))
      );
      return (await readProvisioningLease(claimsDir))?.marker.token === claim.token;
    } catch (error) {
      if (errorCode(error) === "EEXIST" || errorCode(error) === "ENOENT") return false;
      throw error;
    }
  } catch (error) {
    if (errorCode(error) === "EEXIST") return false;
    throw error;
  } finally {
    await rm(prepared, { recursive: true, force: true }).catch(() => undefined);
  }
}

async function readProvisioningLease(
  claimsDir: string
): Promise<ProvisioningLease | undefined> {
  const directory = path.join(claimsDir, ACTIVE_CLAIM_NAME);
  try {
    return await readProvisioningLeaseDirectory(directory);
  } catch (error) {
    if (errorCode(error) === "ENOENT") return undefined;
    throw error;
  }
}

async function readProvisioningLeaseDirectory(directory: string): Promise<ProvisioningLease> {
  const metadata = await lstat(directory);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    throw new Error("managed Python cache contains an invalid active provisioning lease");
  }
  let marker = await readLeaseMarker(path.join(directory, OWNER_FILE));
  const seen = new Set<string>();
  const tokens: string[] = [];
  for (let generation = 0; generation < MAX_LEASE_GENERATIONS; generation += 1) {
    if (seen.has(marker.token)) {
      throw new Error("managed Python cache contains a cyclic provisioning lease");
    }
    seen.add(marker.token);
    tokens.push(marker.token);
    const successor = path.join(directory, successorFileName(marker.token));
    try {
      marker = await readLeaseMarker(successor);
    } catch (error) {
      if (errorCode(error) === "ENOENT") return { marker, tokens };
      throw error;
    }
  }
  throw new Error("managed Python cache provisioning lease exceeded its generation limit");
}

async function retireCompletedProvisioningLease(claimsDir: string): Promise<void> {
  const lease = await readProvisioningLease(claimsDir);
  if (!lease || await anyClaimDirectoryExists(claimsDir, lease.tokens)) return;
  const active = path.join(claimsDir, ACTIVE_CLAIM_NAME);
  const retired = path.join(claimsDir, `.active.retired.${randomBytes(16).toString("hex")}`);
  try {
    await rename(active, retired);
  } catch {
    return;
  }
  const moved = await readProvisioningLeaseDirectory(retired).catch(() => undefined);
  if (moved?.marker.token !== lease.marker.token || moved.tokens.length !== lease.tokens.length) {
    await rename(retired, active).catch(() => undefined);
    return;
  }
  await rm(retired, { recursive: true, force: true }).catch(() => undefined);
}

async function anyClaimDirectoryExists(claimsDir: string, tokens: string[]): Promise<boolean> {
  for (const token of tokens) {
    const exists = await lstat(path.join(claimsDir, token))
      .then((metadata) => metadata.isDirectory() && !metadata.isSymbolicLink())
      .catch(() => false);
    if (exists) return true;
  }
  return false;
}

async function readLeaseMarker(filePath: string): Promise<ProvisioningMarker> {
  const metadata = await lstat(filePath);
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    throw new Error("managed Python cache contains an invalid provisioning lease marker");
  }
  return readFile(filePath, "utf8").then(parseProvisioningMarker);
}

function successorFileName(token: string): string {
  return `successor-${token}.json`;
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

export async function writePrivateJsonExclusive(
  filePath: string,
  value: unknown,
  token: string
): Promise<boolean> {
  const temporary = path.join(path.dirname(filePath), `.${path.basename(filePath)}.${token}.exclusive`);
  try {
    await writeFile(temporary, `${JSON.stringify(value)}\n`, { encoding: "utf8", mode: 0o600 });
    if (process.platform !== "win32") await chmod(temporary, 0o600);
    try {
      await link(temporary, filePath);
      return true;
    } catch (error) {
      if (errorCode(error) === "EEXIST") return false;
      throw error;
    }
  } finally {
    await rm(temporary, { force: true }).catch(() => undefined);
  }
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

function errorCode(error: unknown): string | undefined {
  return typeof error === "object" && error !== null && "code" in error && typeof error.code === "string"
    ? error.code
    : undefined;
}
