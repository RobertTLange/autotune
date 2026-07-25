import { access, mkdir, mkdtemp, readFile, rm, symlink, utimes, writeFile } from "node:fs/promises";
import os, { tmpdir } from "node:os";
import path from "node:path";
import {
  claimOwnsQueue,
  cleanupAbandonedRuntimeEntries,
  type ProvisioningClaim,
  writePrivateJsonExclusive
} from "../src/python-runtime-cache.js";

describe("cleanupAbandonedRuntimeEntries", () => {
  it("publishes exactly one complete winner without replacement", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-runtime-exclusive-publish-"));
    const ready = path.join(root, "runtime.json");
    const candidates = [{ winner: "a" }, { winner: "b" }];

    const results = await Promise.all(candidates.map((candidate, index) => writePrivateJsonExclusive(
      ready,
      candidate,
      (index + 1).toString(16).padStart(32, "0")
    )));

    expect(results.filter(Boolean)).toHaveLength(1);
    expect(candidates).toContainEqual(JSON.parse(await readFile(ready, "utf8")));
  });

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

  it("atomically takes over an active lease whose claim disappeared", async () => {
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
      await expect(claimOwnsQueue(claimsDir, waiting)).resolves.toBe(true);
    } finally {
      clearInterval(waiting.heartbeat);
    }
  });

  it("keeps a stale owner fenced if its heartbeat later revives", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-runtime-revived-lease-"));
    const runtimeDir = path.join(root, "runtime");
    const claimsDir = path.join(root, "runtime.claims");
    const activeLease = path.join(claimsDir, ".active");
    const staleToken = "a".repeat(32);
    const successorToken = "b".repeat(32);
    const old = new Date(Date.now() - 60_000);
    await mkdir(runtimeDir);
    await mkdir(claimsDir);
    await mkdir(activeLease);
    await writeFile(path.join(activeLease, "owner.json"), JSON.stringify({
      pid: process.pid,
      hostname: os.hostname(),
      startedAt: 1,
      token: staleToken
    }));
    await writeClaim(claimsDir, staleToken, process.pid, 1);
    await utimes(path.join(claimsDir, staleToken), old, old);
    await writeClaim(claimsDir, successorToken, process.pid, 2);
    const stale = claim(claimsDir, staleToken, 1);
    const successor = claim(claimsDir, successorToken, 2);

    try {
      await expect(claimOwnsQueue(claimsDir, successor)).resolves.toBe(true);
      const now = new Date();
      await utimes(path.join(claimsDir, staleToken), now, now);
      await expect(claimOwnsQueue(claimsDir, stale)).resolves.toBe(false);
      await expect(claimOwnsQueue(claimsDir, successor)).resolves.toBe(true);
      await cleanupAbandonedRuntimeEntries({
        runtimeDir,
        claimsDir,
        publishedPython: managedPythonForTest(runtimeDir, successorToken)
      });
      await expect(access(activeLease)).resolves.toBeUndefined();
    } finally {
      clearInterval(stale.heartbeat);
      clearInterval(successor.heartbeat);
    }
  });

  it("recovers when a successor claim disappears", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-runtime-successor-recovery-"));
    const claimsDir = path.join(root, "runtime.claims");
    const activeLease = path.join(claimsDir, ".active");
    const staleToken = "a".repeat(32);
    const abandonedToken = "b".repeat(32);
    const recoveryToken = "c".repeat(32);
    await mkdir(claimsDir);
    await mkdir(activeLease);
    await writeFile(path.join(activeLease, "owner.json"), JSON.stringify({
      pid: process.pid,
      hostname: os.hostname(),
      startedAt: 1,
      token: staleToken
    }));
    await writeClaim(claimsDir, staleToken, process.pid, 1);
    const old = new Date(Date.now() - 60_000);
    await utimes(path.join(claimsDir, staleToken), old, old);
    await writeClaim(claimsDir, abandonedToken, process.pid, 2);
    const abandoned = claim(claimsDir, abandonedToken, 2);
    await expect(claimOwnsQueue(claimsDir, abandoned)).resolves.toBe(true);
    clearInterval(abandoned.heartbeat);
    await rm(path.join(claimsDir, abandonedToken), { recursive: true });
    const revived = claim(claimsDir, staleToken, 1);
    const now = new Date();
    await utimes(path.join(claimsDir, staleToken), now, now);
    await expect(claimOwnsQueue(claimsDir, revived)).resolves.toBe(false);
    expect(revived.lost).toBe(true);
    await rm(path.join(claimsDir, staleToken), { recursive: true });
    await writeClaim(claimsDir, recoveryToken, process.pid, 3);
    const recovery = claim(claimsDir, recoveryToken, 3);

    try {
      await expect(claimOwnsQueue(claimsDir, recovery)).resolves.toBe(true);
    } finally {
      clearInterval(recovery.heartbeat);
    }
  });

  it.skipIf(process.platform === "win32")("rejects an active lease symlink without following it", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-runtime-symlink-lease-"));
    const claimsDir = path.join(root, "runtime.claims");
    const outside = path.join(root, "outside");
    const waitingToken = "b".repeat(32);
    await mkdir(claimsDir);
    await mkdir(outside);
    await writeFile(path.join(outside, "owner.json"), JSON.stringify({
      pid: process.pid,
      hostname: os.hostname(),
      startedAt: 1,
      token: "a".repeat(32)
    }));
    await symlink(outside, path.join(claimsDir, ".active"));
    await writeClaim(claimsDir, waitingToken, process.pid, 2);
    const waiting = claim(claimsDir, waitingToken, 2);

    try {
      await expect(claimOwnsQueue(claimsDir, waiting)).rejects.toThrow(/invalid active provisioning lease/i);
      await expect(access(path.join(outside, "owner.json"))).resolves.toBeUndefined();
    } finally {
      clearInterval(waiting.heartbeat);
    }
  });

  it("retires the lease generation chain after publication", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autotune-runtime-retired-lease-"));
    const runtimeDir = path.join(root, "runtime");
    const claimsDir = path.join(root, "runtime.claims");
    const activeLease = path.join(claimsDir, ".active");
    const staleToken = "a".repeat(32);
    const publisherToken = "b".repeat(32);
    await mkdir(runtimeDir);
    await mkdir(claimsDir);
    await mkdir(activeLease);
    await writeFile(path.join(activeLease, "owner.json"), JSON.stringify({
      pid: 999_999_999,
      hostname: os.hostname(),
      startedAt: 1,
      token: staleToken
    }));
    await writeClaim(claimsDir, publisherToken, process.pid, 2);
    const publisher = claim(claimsDir, publisherToken, 2);

    try {
      await expect(claimOwnsQueue(claimsDir, publisher)).resolves.toBe(true);
      clearInterval(publisher.heartbeat);
      await rm(path.join(claimsDir, publisherToken), { recursive: true });
      await cleanupAbandonedRuntimeEntries({
        runtimeDir,
        claimsDir,
        publishedPython: managedPythonForTest(runtimeDir, publisherToken)
      });
      await expect(access(activeLease)).rejects.toThrow();
    } finally {
      clearInterval(publisher.heartbeat);
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

function managedPythonForTest(runtimeDir: string, token: string): string {
  const suffix = process.platform === "win32" ? ["Scripts", "python.exe"] : ["bin", "python"];
  return path.join(runtimeDir, `env-${token}`, ...suffix);
}
