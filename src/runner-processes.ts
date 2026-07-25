import { spawnSync } from "node:child_process";
import { accessSync, constants, opendirSync, readFileSync } from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import process from "node:process";
import { killWindowsProcessTree } from "./process.js";
import {
  readLinuxProcessSnapshotEntry,
  readProcessIdentity,
  sameProcessIdentity,
  type ProcessIdentity,
  type ProcessSnapshotEntry
} from "./runner-output.js";

const PROCESS_SNAPSHOT_TIMEOUT_MS = 500;
const PROCESS_SNAPSHOT_MAX_BYTES = 1024 * 1024;
const MAX_DESCENDANT_PROCESSES = 4_096;

export function captureRunnerIdentity(pid: number | undefined): ProcessIdentity | undefined {
  if (!pid || process.platform === "win32") return undefined;
  return readProcessIdentity(pid, performance.now() + PROCESS_SNAPSHOT_TIMEOUT_MS);
}

export function signalDetachedDescendants(
  parentPid: number | undefined,
  rootIdentity: ProcessIdentity | undefined
): void {
  if (!parentPid) return;
  if (signalWindowsRunnerTree(parentPid)) return;
  if (!rootIdentity) return;
  const deadline = performance.now() + PROCESS_SNAPSHOT_TIMEOUT_MS;
  for (const descendant of descendantProcesses(parentPid, rootIdentity, deadline)) {
    if (performance.now() >= deadline) break;
    const currentUid = process.getuid?.();
    if (currentUid === undefined || descendant.identity.uid !== currentUid) continue;
    const revalidated = readProcessIdentity(descendant.pid, deadline);
    if (!sameProcessIdentity(descendant.identity, revalidated)) continue;
    try {
      if (descendant.identity.pgid === descendant.pid) {
        process.kill(-descendant.pid, "SIGKILL");
      }
      process.kill(descendant.pid, "SIGKILL");
    } catch {
      // Revalidated descendant may exit during signal delivery.
    }
  }
}

export function signalWindowsRunnerTree(
  parentPid: number,
  platform: NodeJS.Platform = process.platform,
  terminate: typeof killWindowsProcessTree = killWindowsProcessTree
): boolean {
  if (platform !== "win32") return false;
  terminate(parentPid, "SIGKILL");
  return true;
}

function descendantProcesses(
  parentPid: number,
  rootIdentity: ProcessIdentity,
  deadline: number
): ProcessSnapshotEntry[] {
  if (process.platform === "linux") {
    return linuxDescendantProcesses(parentPid, rootIdentity, deadline);
  }
  const snapshot = posixProcessSnapshot(deadline);
  return descendantsFromPosixSnapshot(parentPid, rootIdentity, snapshot);
}

export function descendantsFromPosixSnapshot(
  parentPid: number,
  rootIdentity: ProcessIdentity,
  snapshot: ProcessSnapshotEntry[]
): ProcessSnapshotEntry[] {
  const root = snapshot.find((entry) => entry.pid === parentPid);
  if (
    !root
    || root.parentPid !== process.pid
    || !sameProcessIdentity(rootIdentity, root.identity)
  ) return [];
  const childrenByParent = new Map<number, ProcessSnapshotEntry[]>();
  for (const entry of snapshot) {
    const children = childrenByParent.get(entry.parentPid) ?? [];
    children.push(entry);
    childrenByParent.set(entry.parentPid, children);
  }

  const descendants: ProcessSnapshotEntry[] = [];
  const pending = [...(childrenByParent.get(parentPid) ?? [])];
  const seen = new Set<number>();
  while (pending.length > 0 && descendants.length < MAX_DESCENDANT_PROCESSES) {
    const entry = pending.pop() as ProcessSnapshotEntry;
    if (seen.has(entry.pid)) continue;
    seen.add(entry.pid);
    descendants.push(entry);
    pending.push(...(childrenByParent.get(entry.pid) ?? []));
  }
  return descendants.reverse();
}

function posixProcessSnapshot(deadline: number): ProcessSnapshotEntry[] {
  const command = resolvePosixPsCommand();
  if (!command) return [];
  const snapshot = readPosixProcessSnapshot(command, deadline);
  return snapshot ? parsePosixProcessSnapshot(snapshot) : [];
}

function linuxDescendantProcesses(
  parentPid: number,
  rootIdentity: ProcessIdentity,
  deadline: number
): ProcessSnapshotEntry[] {
  const root = readLinuxProcessSnapshotEntry(parentPid, deadline);
  if (
    !root
    || root.parentPid !== process.pid
    || !sameProcessIdentity(rootIdentity, root.identity)
  ) return [];
  const descendants: ProcessSnapshotEntry[] = [];
  const pending = [root];
  const seen = new Set([parentPid]);
  while (
    pending.length > 0
    && descendants.length < MAX_DESCENDANT_PROCESSES
    && performance.now() < deadline
  ) {
    const parent = pending.pop() as ProcessSnapshotEntry;
    for (const childPid of linuxChildProcessIds(parent.pid, deadline)) {
      if (seen.has(childPid)) continue;
      seen.add(childPid);
      const child = readLinuxProcessSnapshotEntry(childPid, deadline);
      if (!child || child.parentPid !== parent.pid) continue;
      descendants.push(child);
      pending.push(child);
      if (descendants.length >= MAX_DESCENDANT_PROCESSES) break;
    }
  }
  return descendants.reverse();
}

function linuxChildProcessIds(parentPid: number, deadline: number): number[] {
  const children = new Set<number>();
  let tasks: ReturnType<typeof opendirSync>;
  try {
    tasks = opendirSync(`/proc/${parentPid}/task`);
  } catch {
    return [];
  }
  try {
    let entry;
    while (
      children.size < MAX_DESCENDANT_PROCESSES
      && performance.now() < deadline
      && (entry = tasks.readSync()) !== null
    ) {
      if (!/^\d+$/.test(entry.name)) continue;
      try {
        const source = readFileSync(
          `/proc/${parentPid}/task/${entry.name}/children`,
          "utf8"
        );
        for (const child of source.trim().split(/\s+/)) {
          const childPid = Number(child);
          if (Number.isInteger(childPid) && childPid > 0) children.add(childPid);
          if (children.size >= MAX_DESCENDANT_PROCESSES) break;
        }
      } catch {
        // Threads and child lists may disappear during the traversal.
      }
    }
  } finally {
    tasks.closeSync();
  }
  return [...children];
}

export function parsePosixProcessSnapshot(source: string): ProcessSnapshotEntry[] {
  const snapshot: ProcessSnapshotEntry[] = [];
  for (const line of source.split("\n")) {
    const fields = line.trim().split(/\s+/);
    const [pidText, parentText, uidText, pgidText] = fields;
    const pid = Number(pidText);
    const parentPid = Number(parentText);
    const uid = Number(uidText);
    const pgid = Number(pgidText);
    if (
      fields.length < 9
      || !Number.isInteger(pid)
      || pid < 1
      || !Number.isInteger(parentPid)
      || parentPid < 0
      || !Number.isInteger(uid)
      || !Number.isInteger(pgid)
    ) continue;
    snapshot.push({
      pid,
      parentPid,
      identity: {
        uid,
        pgid,
        generation: [pidText, uidText, pgidText, ...fields.slice(4, 9)].join(" ")
      }
    });
  }
  return snapshot;
}

function readPosixProcessSnapshot(command: string, deadline: number): string | undefined {
  const argumentVariants = [
    ["-axo", "pid=,ppid=,uid=,pgid=,lstart="],
    ["-Ao", "pid,ppid,uid,pgid,lstart"]
  ];
  for (const args of argumentVariants) {
    const result = spawnSync(command, args, {
      encoding: "utf8",
      env: { PATH: path.dirname(command), LC_ALL: "C" },
      killSignal: "SIGKILL",
      maxBuffer: PROCESS_SNAPSHOT_MAX_BYTES,
      timeout: Math.max(1, Math.ceil(deadline - performance.now())),
      windowsHide: true
    });
    if (!result.error && result.status === 0) return result.stdout;
  }
  return undefined;
}

function resolvePosixPsCommand(): string | undefined {
  const candidates = ["/bin/ps", "/usr/bin/ps", "/run/current-system/sw/bin/ps"];
  for (const candidate of candidates) {
    try {
      accessSync(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Keep scanning fixed operating-system locations.
    }
  }
  return undefined;
}

export function signalChildTree(
  pid: number | undefined,
  signal: NodeJS.Signals,
  expectedIdentity: ProcessIdentity | undefined
): void {
  if (!pid) return;
  if (process.platform !== "win32") {
    const deadline = performance.now() + PROCESS_SNAPSHOT_TIMEOUT_MS;
    if (!expectedIdentity || !sameProcessIdentity(
      expectedIdentity,
      readProcessIdentity(pid, deadline)
    )) return;
  }
  try {
    process.kill(process.platform === "win32" ? pid : -pid, signal);
  } catch {
    // The controller may have exited between signal delivery and forwarding.
  }
}
