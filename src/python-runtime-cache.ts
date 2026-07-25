import { randomBytes } from "node:crypto";
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
const OWNER_FILE = "owner.json";

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
  return JSON.parse(source.trim()) as ProvisioningMarker;
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
