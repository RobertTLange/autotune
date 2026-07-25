import { access, mkdir, mkdtemp, symlink, utimes, writeFile } from "node:fs/promises";
import os, { tmpdir } from "node:os";
import path from "node:path";
import {
  claimOwnsQueue,
  cleanupAbandonedRuntimeEntries,
  type ProvisioningClaim
} from "../src/python-runtime-cache.js";

describe("cleanupAbandonedRuntimeEntries", () => {
  it("does not let a later-published earlier claim preempt the active provisioner", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-runtime-lease-"));
    const claimsDir = path.join(root, "runtime.claims");
    const activeToken = "b".repeat(32);
    const delayedEarlierToken = "a".repeat(32);
    await mkdir(claimsDir);
    await writeClaim(claimsDir, activeToken, process.pid, 2);
    const active = claim(claimsDir, activeToken, 2);
    const delayedEarlier = claim(claimsDir, delayedEarlierToken, 1);

    try {
      await expect(claimOwnsQueue(claimsDir, active)).resolves.toBe(true);
      await writeClaim(claimsDir, delayedEarlierToken, process.pid, 1);
      await expect(claimOwnsQueue(claimsDir, delayedEarlier)).resolves.toBe(false);
      await expect(claimOwnsQueue(claimsDir, active)).resolves.toBe(true);
    } finally {
      clearInterval(active.heartbeat);
      clearInterval(delayedEarlier.heartbeat);
    }
  });

  it("recovers an active lease whose claim disappeared", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-runtime-stale-lease-"));
    const claimsDir = path.join(root, "runtime.claims");
    const activeLease = path.join(claimsDir, ".active");
    const staleToken = "a".repeat(32);
    const waitingToken = "b".repeat(32);
    await mkdir(claimsDir);
    await mkdir(activeLease);
    await writeFile(path.join(activeLease, "owner.json"), JSON.stringify({
      pid: 999_999_999,
      hostname: os.hostname(),
      startedAt: 1,
      token: staleToken
    }));
    await writeClaim(claimsDir, waitingToken, process.pid, 2);
    const waiting = claim(claimsDir, waitingToken, 2);

    try {
      await expect(claimOwnsQueue(claimsDir, waiting)).resolves.toBe(false);
      await expect(claimOwnsQueue(claimsDir, waiting)).resolves.toBe(true);
    } finally {
      clearInterval(waiting.heartbeat);
    }
  });

  it("removes stale claims and aged orphan environments while preserving live entries", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-runtime-gc-"));
    const runtimeDir = path.join(root, "runtime");
    const claimsDir = path.join(root, "runtime.claims");
    const activeToken = "a".repeat(32);
    const publishedToken = "b".repeat(32);
    const orphanToken = "c".repeat(32);
    const freshOrphanToken = "d".repeat(32);
    const staleClaimToken = "e".repeat(32);
    const liveClaimToken = "f".repeat(32);
    const stagingToken = "1".repeat(32);
    const preparedLeaseToken = "3".repeat(32);
    const now = Date.now();
    const old = new Date(now - 2 * 24 * 60 * 60 * 1000);

    await mkdir(runtimeDir);
    await mkdir(claimsDir);
    for (const token of [activeToken, publishedToken, orphanToken, freshOrphanToken]) {
      await mkdir(path.join(runtimeDir, `env-${token}`));
    }
    await writeFile(path.join(runtimeDir, "runtime.json"), JSON.stringify({
      managed: true,
      python: path.join(runtimeDir, `env-${publishedToken}`, process.platform === "win32" ? "Scripts/python.exe" : "bin/python")
    }));
    await utimes(path.join(runtimeDir, `env-${activeToken}`), old, old);
    await utimes(path.join(runtimeDir, `env-${publishedToken}`), old, old);
    await utimes(path.join(runtimeDir, `env-${orphanToken}`), old, old);

    await writeClaim(claimsDir, staleClaimToken, 999_999_999);
    await utimes(path.join(claimsDir, staleClaimToken), old, old);
    await writeClaim(claimsDir, activeToken, process.pid);
    await utimes(path.join(claimsDir, activeToken), old, old);
    await writeClaim(claimsDir, liveClaimToken, process.pid);
    await mkdir(path.join(claimsDir, `.${stagingToken}.staging`));
    await utimes(path.join(claimsDir, `.${stagingToken}.staging`), old, old);
    await mkdir(path.join(claimsDir, `.active.prepare.${preparedLeaseToken}`));
    await utimes(path.join(claimsDir, `.active.prepare.${preparedLeaseToken}`), old, old);
    await symlink(root, path.join(runtimeDir, `env-${"2".repeat(32)}`));

    await cleanupAbandonedRuntimeEntries({
      runtimeDir,
      claimsDir,
      activeToken,
      publishedPython: path.join(
        runtimeDir,
        `env-${publishedToken}`,
        process.platform === "win32" ? "Scripts/python.exe" : "bin/python"
      ),
      now
    });

    await expect(access(path.join(claimsDir, staleClaimToken))).rejects.toThrow();
    await expect(access(path.join(claimsDir, `.${stagingToken}.staging`))).rejects.toThrow();
    await expect(access(path.join(claimsDir, `.active.prepare.${preparedLeaseToken}`))).rejects.toThrow();
    await expect(access(path.join(runtimeDir, `env-${orphanToken}`))).rejects.toThrow();
    await expect(access(path.join(claimsDir, liveClaimToken))).resolves.toBeUndefined();
    await expect(access(path.join(claimsDir, activeToken))).resolves.toBeUndefined();
    await expect(access(path.join(runtimeDir, `env-${activeToken}`))).resolves.toBeUndefined();
    await expect(access(path.join(runtimeDir, `env-${publishedToken}`))).resolves.toBeUndefined();
    await expect(access(path.join(runtimeDir, `env-${freshOrphanToken}`))).resolves.toBeUndefined();
    await expect(access(path.join(runtimeDir, `env-${"2".repeat(32)}`))).resolves.toBeUndefined();
  });

  it("does not let a large live-claim set starve orphan cleanup", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-runtime-gc-decoys-"));
    const runtimeDir = path.join(root, "runtime");
    const claimsDir = path.join(root, "runtime.claims");
    const orphanToken = "0".repeat(32);
    const staleClaimToken = "f".repeat(32);
    const now = Date.now();
    const old = new Date(now - 2 * 24 * 60 * 60 * 1000);
    await mkdir(runtimeDir);
    await mkdir(claimsDir);
    const orphan = path.join(runtimeDir, `env-${orphanToken}`);
    await mkdir(orphan);
    await utimes(orphan, old, old);
    for (let index = 1; index <= 129; index += 1) {
      await writeClaim(claimsDir, index.toString(16).padStart(32, "0"), process.pid);
    }
    await writeClaim(claimsDir, staleClaimToken, 999_999_999);
    await utimes(path.join(claimsDir, staleClaimToken), old, old);

    await cleanupAbandonedRuntimeEntries({ runtimeDir, claimsDir, now });

    await expect(access(orphan)).rejects.toThrow();
    await expect(access(path.join(claimsDir, staleClaimToken))).rejects.toThrow();
  });
});

async function writeClaim(
  claimsDir: string,
  token: string,
  pid: number,
  startedAt = Date.now()
): Promise<void> {
  const directory = path.join(claimsDir, token);
  await mkdir(directory);
  await writeFile(path.join(directory, "owner.json"), JSON.stringify({
    pid,
    hostname: os.hostname(),
    startedAt,
    token
  }));
}

function claim(claimsDir: string, token: string, startedAt: number): ProvisioningClaim {
  const heartbeat = setInterval(() => undefined, 60_000);
  heartbeat.unref();
  return {
    pid: process.pid,
    hostname: os.hostname(),
    startedAt,
    token,
    directory: path.join(claimsDir, token),
    heartbeat,
    lost: false
  };
}
