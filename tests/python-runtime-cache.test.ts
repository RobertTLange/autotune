import { access, mkdir, mkdtemp, symlink, utimes, writeFile } from "node:fs/promises";
import os, { tmpdir } from "node:os";
import path from "node:path";
import { cleanupAbandonedRuntimeEntries } from "../src/python-runtime-cache.js";

describe("cleanupAbandonedRuntimeEntries", () => {
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

async function writeClaim(claimsDir: string, token: string, pid: number): Promise<void> {
  const directory = path.join(claimsDir, token);
  await mkdir(directory);
  await writeFile(path.join(directory, "owner.json"), JSON.stringify({
    pid,
    hostname: os.hostname(),
    startedAt: Date.now(),
    token
  }));
}
