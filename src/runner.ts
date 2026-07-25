import { spawn, spawnSync } from "node:child_process";
import { accessSync, constants, opendirSync, readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import process from "node:process";
import { ensurePythonRuntime } from "./python-runtime.js";
import { killWindowsProcessTree } from "./process.js";
import {
  cleanupOutputCapture,
  closeOutputWriters,
  createOutputCapture,
  drainCapturedOutput,
  MAX_CAPTURE_BYTES,
  readLinuxProcessSnapshotEntry,
  readProcessIdentity,
  sameProcessIdentity,
  signalOutputHolders,
  type OutputCapture,
  type ProcessIdentity,
  type ProcessSnapshotEntry
} from "./runner-output.js";

export const TARGET_PYTHON_ENV = "AUTOTUNE_TARGET_PYTHON_ENV";
export const NODE_EXECUTABLE_ENV = "AUTOTUNE_NODE_EXECUTABLE";

const RUNNER_KILL_GRACE_MS = 2_000;
const RUNNER_HARD_STOP_GRACE_MS = 1_000;
const PROCESS_SNAPSHOT_TIMEOUT_MS = 500;
const PROCESS_SNAPSHOT_MAX_BYTES = 1024 * 1024;
const MAX_DESCENDANT_PROCESSES = 4_096;

export async function runPythonRunner(input: {
  runnerPath: string;
  trials: number;
  direction: "maximize" | "minimize";
  sampler: string;
  pruner: string;
  nJobs: number;
  storage?: string;
  studyName?: string;
  output?: string;
  python?: string;
  env?: NodeJS.ProcessEnv;
}): Promise<string> {
  const callerEnv = input.env ?? process.env;
  const python = input.python ?? (await ensurePythonRuntime({
    includeCmaes: true,
    env: callerEnv
  })).python;
  const supportsTargetEnvironment = await runnerSupportsTargetEnvironment(input.runnerPath);
  const args = [
    supportsTargetEnvironment ? "-I" : "-E",
    input.runnerPath,
    "--trials",
    String(input.trials),
    "--direction",
    input.direction,
    "--sampler",
    input.sampler,
    "--pruner",
    input.pruner,
    "--n-jobs",
    String(input.nJobs)
  ];

  if (input.storage) {
    args.push("--storage", input.storage);
  }
  if (input.studyName) {
    args.push("--study-name", input.studyName);
  }
  if (input.output) {
    args.push("--output", input.output);
  }

  return new Promise((resolve, reject) => {
    let capture: OutputCapture | undefined;
    let child: ReturnType<typeof spawn>;
    try {
      capture = process.platform === "win32" ? undefined : createOutputCapture();
      child = spawn(python, args, {
        detached: process.platform !== "win32",
        env: supportsTargetEnvironment ? controllerEnvironment(callerEnv) : callerEnv,
        stdio: capture
          ? ["ignore", capture.stdoutWriter, capture.stderrWriter]
          : ["ignore", "pipe", "pipe"]
      });
    } catch (error) {
      if (capture) cleanupOutputCapture(capture);
      reject(error);
      return;
    } finally {
      if (capture) closeOutputWriters(capture);
    }
    const runnerIdentity = child.pid && process.platform !== "win32"
      ? readProcessIdentity(child.pid, performance.now() + PROCESS_SNAPSHOT_TIMEOUT_MS)
      : undefined;
    let settled = false;
    let forwardedSignal: NodeJS.Signals | undefined;
    let escalationTimer: NodeJS.Timeout | undefined;
    let hardStopTimer: NodeJS.Timeout | undefined;
    let postCloseTimer: NodeJS.Timeout | undefined;
    let progressTimer: NodeJS.Timeout | undefined;
    let childClosed = false;
    let childCode: number | null = null;
    let stdoutEnded = false;
    let stderrEnded = false;
    let stdoutBytes = 0;
    let stderrBytes = 0;
    let stdout = "";
    let stderr = "";
    const finish = (error: Error | undefined, detachChild = false) => {
      if (settled) return;
      settled = true;
      cleanupSignals();
      if (detachChild) {
        child.stdout?.destroy();
        child.stderr?.destroy();
        child.unref();
      }
      if (capture) cleanupOutputCapture(capture);
      if (error) reject(error);
      else resolve(stdout);
    };
    const interruptedError = () => {
      const signal = forwardedSignal as NodeJS.Signals;
      process.exitCode = 128 + (signal === "SIGINT" ? 2 : 15);
      return new Error(`python runner interrupted by ${signal}`);
    };
    const forceStop = () => {
      if (!hardStopTimer) {
        hardStopTimer = setTimeout(
          () => finish(interruptedError(), true),
          RUNNER_HARD_STOP_GRACE_MS
        );
        hardStopTimer.unref();
      }
      signalChildTree(child.pid, "SIGKILL", runnerIdentity);
      signalOutputHolders(capture, child.pid);
    };
    const forwardSignal = (signal: NodeJS.Signals) => {
      if (forwardedSignal) {
        forceStop();
        return;
      }
      forwardedSignal = signal;
      escalationTimer = setTimeout(forceStop, RUNNER_KILL_GRACE_MS);
      escalationTimer.unref();
      signalDetachedDescendants(child.pid, runnerIdentity);
      signalChildTree(child.pid, signal, runnerIdentity);
      signalOutputHolders(capture, child.pid);
    };
    const onSigint = () => forwardSignal("SIGINT");
    const onSigterm = () => forwardSignal("SIGTERM");
    process.on("SIGINT", onSigint);
    process.on("SIGTERM", onSigterm);
    const cleanupSignals = () => {
      process.off("SIGINT", onSigint);
      process.off("SIGTERM", onSigterm);
      if (escalationTimer) {
        clearTimeout(escalationTimer);
      }
      if (hardStopTimer) {
        clearTimeout(hardStopTimer);
      }
      if (postCloseTimer) {
        clearTimeout(postCloseTimer);
      }
      if (progressTimer) {
        clearTimeout(progressTimer);
      }
    };
    const resultError = () => {
      if (forwardedSignal) return interruptedError();
      if (childCode === 0) return undefined;
      return new Error(`python runner exited with ${childCode}: ${(stderr || stdout).trim()}`);
    };
    const completeWhenDrained = () => {
      if (childClosed && stdoutEnded && stderrEnded) finish(resultError());
    };
    const abortRunner = (error: Error) => {
      if (settled) return;
      signalDetachedDescendants(child.pid, runnerIdentity);
      signalChildTree(child.pid, "SIGKILL", runnerIdentity);
      signalOutputHolders(capture, child.pid);
      finish(error, true);
    };
    const failForOutput = (stream: "stdout" | "stderr") => {
      abortRunner(new Error(`python runner ${stream} exceeded ${MAX_CAPTURE_BYTES} bytes`));
    };
    const consumeStdout = (chunk: string, bytes = Buffer.byteLength(chunk)) => {
      if (settled) return;
      stdoutBytes += bytes;
      if (stdoutBytes > MAX_CAPTURE_BYTES) failForOutput("stdout");
      else stdout += chunk;
    };
    const consumeStderr = (chunk: string, bytes = Buffer.byteLength(chunk)) => {
      if (settled) return;
      stderrBytes += bytes;
      if (stderrBytes > MAX_CAPTURE_BYTES) {
        failForOutput("stderr");
        return;
      }
      stderr += chunk;
      process.stderr.write(chunk);
    };
    const drainCapture = (): number => {
      if (!capture || settled) return 0;
      const nextStdout = drainCapturedOutput(capture, "stdout");
      consumeStdout(nextStdout.text, nextStdout.bytes);
      stdoutEnded = nextStdout.ended;
      if (settled) return nextStdout.bytes;
      const nextStderr = drainCapturedOutput(capture, "stderr");
      consumeStderr(nextStderr.text, nextStderr.bytes);
      stderrEnded = nextStderr.ended;
      completeWhenDrained();
      return nextStdout.bytes + nextStderr.bytes;
    };
    if (capture) {
      let activeUntil = 0;
      const pollCapture = () => {
        if (settled) return;
        try {
          const bytes = drainCapture();
          if (bytes > 0) activeUntil = Date.now() + 100;
          progressTimer = setTimeout(pollCapture, Date.now() < activeUntil ? 0 : 10);
          progressTimer.unref();
        } catch (error) {
          abortRunner(toError(error));
        }
      };
      progressTimer = setTimeout(pollCapture, 0);
      progressTimer.unref();
    } else {
      child.stdout?.setEncoding("utf8");
      child.stderr?.setEncoding("utf8");
      child.stdout?.on("data", consumeStdout);
      child.stderr?.on("data", consumeStderr);
      child.stdout?.on("end", () => {
        stdoutEnded = true;
        completeWhenDrained();
      });
      child.stderr?.on("end", () => {
        stderrEnded = true;
        completeWhenDrained();
      });
      child.stdout?.on("error", (error: Error) => abortRunner(error));
      child.stderr?.on("error", (error: Error) => abortRunner(error));
    }
    child.on("error", (error) => {
      finish(forwardedSignal ? interruptedError() : error);
    });
    child.on("close", (code) => {
      if (settled) return;
      childClosed = true;
      childCode = code;
      if (capture && !forwardedSignal) signalOutputHolders(capture, child.pid);
      try {
        drainCapture();
      } catch (error) {
        abortRunner(toError(error));
        return;
      }
      if (settled) return;
      if (!postCloseTimer) {
        postCloseTimer = setTimeout(
          () => finish(resultError(), true),
          RUNNER_HARD_STOP_GRACE_MS
        );
      }
      completeWhenDrained();
    });
  });
}

function signalDetachedDescendants(
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

function signalChildTree(
  pid: number | undefined,
  signal: NodeJS.Signals,
  expectedIdentity: ProcessIdentity | undefined
): void {
  if (!pid) {
    return;
  }
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

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

async function runnerSupportsTargetEnvironment(runnerPath: string): Promise<boolean> {
  try {
    return (await readFile(runnerPath, "utf8")).includes(TARGET_PYTHON_ENV);
  } catch {
    return false;
  }
}

function controllerEnvironment(callerEnv: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const controllerEnv = { ...callerEnv };
  const targetPythonEnvironment: Record<string, string> = {};
  for (const [name, value] of Object.entries(callerEnv)) {
    if ((/^PYTHON/i.test(name) || name === TARGET_PYTHON_ENV) && value !== undefined) {
      targetPythonEnvironment[name] = value;
      delete controllerEnv[name];
    }
  }
  controllerEnv[TARGET_PYTHON_ENV] = JSON.stringify(targetPythonEnvironment);
  controllerEnv[NODE_EXECUTABLE_ENV] = process.execPath;
  return controllerEnv;
}
