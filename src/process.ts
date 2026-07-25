import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";

export interface CommandResult {
  stdout: string;
  stderr: string;
}

export interface CommandOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  timeoutMs?: number;
  maxOutputBytes?: number;
}

const DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024;
const DEFAULT_KILL_GRACE_MS = 1000;
const WINDOWS_TREE_KILL_TIMEOUT_MS = 5_000;

export class CommandInterruptedError extends Error {
  readonly code = "ERR_COMMAND_INTERRUPTED";

  constructor(command: string, readonly signal: NodeJS.Signals) {
    super(`${command} interrupted by ${signal}`);
    this.name = "CommandInterruptedError";
  }
}

export function isCommandInterruptedError(error: unknown): error is CommandInterruptedError {
  return error instanceof CommandInterruptedError;
}

export async function runCommand(
  command: string,
  args: string[],
  options: CommandOptions = {}
): Promise<CommandResult> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timedOut = false;
    let forwardedSignal: NodeJS.Signals | undefined;
    let timeout: NodeJS.Timeout | undefined;
    let killTimer: NodeJS.Timeout | undefined;
    let hardStopTimer: NodeJS.Timeout | undefined;
    let childClosed = false;
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32"
    });
    const finishTermination = (detachChild = false) => {
      if (settled) return;
      settled = true;
      if (timeout) clearTimeout(timeout);
      if (killTimer) clearTimeout(killTimer);
      if (hardStopTimer) clearTimeout(hardStopTimer);
      cleanupSignals();
      if (detachChild) {
        child.stdout.destroy();
        child.stderr.destroy();
        child.unref();
      }
      reject(terminationError(command, options.timeoutMs, forwardedSignal));
    };
    const terminate = (signal: NodeJS.Signals) => {
      killChildTree(child.pid, signal);
      killTimer = setTimeout(() => {
        killTimer = undefined;
        if (childClosed) {
          finishTermination();
          return;
        }
        killChildTree(child.pid, "SIGKILL");
        hardStopTimer = setTimeout(() => finishTermination(true), DEFAULT_KILL_GRACE_MS);
      }, DEFAULT_KILL_GRACE_MS);
    };
    const cleanupSignals = installCleanupHandlers((signal) => {
      if (settled) return;
      if (forwardedSignal) {
        killChildTree(child.pid, "SIGKILL");
        return;
      }
      forwardedSignal = signal;
      if (timeout) {
        clearTimeout(timeout);
      }
      process.exitCode = 128 + signalNumber(signal);
      if (timedOut) {
        killChildTree(child.pid, "SIGKILL");
        return;
      }
      terminate(signal);
    });
    let stdout = "";
    let stderr = "";
    const maxOutputBytes = options.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES;
    timeout = options.timeoutMs === undefined
      ? undefined
      : setTimeout(() => {
          timedOut = true;
          terminate("SIGTERM");
        }, options.timeoutMs);

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout = appendBounded(stdout, chunk, maxOutputBytes);
    });
    child.stderr.on("data", (chunk) => {
      stderr = appendBounded(stderr, chunk, maxOutputBytes);
    });
    child.on("error", (error) => {
      if (settled) {
        return;
      }
      if (timedOut || forwardedSignal) {
        return;
      }
      settled = true;
      if (timeout) {
        clearTimeout(timeout);
      }
      cleanupSignals();
      reject(error);
    });
    child.on("close", (code) => {
      if (settled) {
        return;
      }
      childClosed = true;
      if (forwardedSignal || timedOut) {
        if (killTimer) {
          clearTimeout(killTimer);
          killTimer = undefined;
        }
        finishTermination();
        return;
      }
      settled = true;
      if (timeout) {
        clearTimeout(timeout);
      }
      if (killTimer) {
        clearTimeout(killTimer);
      }
      if (hardStopTimer) {
        clearTimeout(hardStopTimer);
      }
      cleanupSignals();
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`${command} exited with ${code}: ${(stderr || stdout).trim()}`));
      }
    });
  });
}

function installCleanupHandlers(handleSignal: (signal: NodeJS.Signals) => void): () => void {
  const onSigint = () => handleSignal("SIGINT");
  const onSigterm = () => handleSignal("SIGTERM");
  process.on("SIGINT", onSigint);
  process.on("SIGTERM", onSigterm);
  return () => {
    process.off("SIGINT", onSigint);
    process.off("SIGTERM", onSigterm);
  };
}

function terminationError(
  command: string,
  timeoutMs: number | undefined,
  signal?: NodeJS.Signals
): Error {
  return signal
    ? new CommandInterruptedError(command, signal)
    : new Error(`${command} timed out after ${timeoutMs}ms`);
}

function signalNumber(signal: NodeJS.Signals): number {
  return signal === "SIGINT" ? 2 : 15;
}

function killChildTree(pid: number | undefined, signal: NodeJS.Signals): void {
  if (!pid) {
    return;
  }
  try {
    if (process.platform === "win32") {
      killWindowsProcessTree(pid, signal);
    } else {
      process.kill(-pid, signal);
    }
  } catch {
    // Process may have exited between timeout and kill.
  }
}

export function killWindowsProcessTree(pid: number, signal: NodeJS.Signals): void {
  let treeKilled = false;
  try {
    const invocation = windowsTaskkillInvocation(pid);
    treeKilled = spawnSync(invocation.command, invocation.args, invocation.options).status === 0;
  } catch {
    // Fall back to signaling the controller process directly.
  }
  if (!treeKilled) {
    try {
      process.kill(pid, signal);
    } catch {
      // The process may already have exited.
    }
  }
}

export function windowsTaskkillInvocation(
  pid: number,
  fileExists: (filePath: string) => boolean = existsSync,
  configuredRoot: string | undefined = process.env.SystemRoot
) {
  const systemRoot = resolveWindowsSystemRoot(fileExists, configuredRoot);
  return {
    command: path.win32.join(systemRoot, "System32", "taskkill.exe"),
    args: ["/PID", String(pid), "/T", "/F"],
    options: {
      env: { SystemRoot: systemRoot, windir: systemRoot },
      stdio: "ignore" as const,
      timeout: WINDOWS_TREE_KILL_TIMEOUT_MS,
      windowsHide: true
    }
  };
}

export function resolveWindowsSystemRoot(
  fileExists: (filePath: string) => boolean = existsSync,
  configured: string | undefined = process.env.SystemRoot
): string {
  const standard = "C:\\Windows";
  if (fileExists(path.win32.join(standard, "System32", "taskkill.exe"))) {
    return standard;
  }
  if (configured && path.win32.isAbsolute(configured)) {
    const normalized = path.win32.resolve(configured);
    const parsed = path.win32.parse(normalized);
    const taskkill = path.win32.join(normalized, "System32", "taskkill.exe");
    if (
      /^[A-Za-z]:\\$/.test(parsed.root) &&
      path.win32.dirname(normalized).toLowerCase() === parsed.root.toLowerCase() &&
      ["windows", "winnt"].includes(path.win32.basename(normalized).toLowerCase()) &&
      fileExists(taskkill)
    ) {
      return normalized;
    }
  }
  return standard;
}

function appendBounded(current: string, chunk: string, maxBytes: number): string {
  const combined = current + chunk;
  if (Buffer.byteLength(combined, "utf8") <= maxBytes) {
    return combined;
  }
  return Buffer.from(combined, "utf8").subarray(-maxBytes).toString("utf8");
}
