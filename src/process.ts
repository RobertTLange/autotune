import { spawn } from "node:child_process";
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

export async function runCommand(
  command: string,
  args: string[],
  options: CommandOptions = {}
): Promise<CommandResult> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let timedOut = false;
    let killTimer: NodeJS.Timeout | undefined;
    let hardStopTimer: NodeJS.Timeout | undefined;
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32"
    });
    const cleanupSignals = installCleanupHandlers(child.pid);
    let stdout = "";
    let stderr = "";
    const maxOutputBytes = options.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES;
    const timeout =
      options.timeoutMs === undefined
        ? undefined
        : setTimeout(() => {
            timedOut = true;
            killChildTree(child.pid, "SIGTERM");
            killTimer = setTimeout(() => {
              killChildTree(child.pid, "SIGKILL");
              hardStopTimer = setTimeout(() => {
                if (settled) return;
                settled = true;
                cleanupSignals();
                child.stdout.destroy();
                child.stderr.destroy();
                child.unref();
                reject(new Error(`${command} timed out after ${options.timeoutMs}ms`));
              }, DEFAULT_KILL_GRACE_MS);
            }, DEFAULT_KILL_GRACE_MS);
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
      if (timedOut) {
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
      if (timedOut) {
        reject(new Error(`${command} timed out after ${options.timeoutMs}ms`));
        return;
      }
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`${command} exited with ${code}: ${(stderr || stdout).trim()}`));
      }
    });
  });
}

function installCleanupHandlers(pid: number | undefined): () => void {
  if (!pid || process.platform === "win32") {
    return () => {};
  }
  const handleSignal = (signal: NodeJS.Signals) => {
    killChildTree(pid, signal);
    process.exit(128 + signalNumber(signal));
  };
  const onSigint = () => handleSignal("SIGINT");
  const onSigterm = () => handleSignal("SIGTERM");
  process.once("SIGINT", onSigint);
  process.once("SIGTERM", onSigterm);
  return () => {
    process.off("SIGINT", onSigint);
    process.off("SIGTERM", onSigterm);
  };
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
      const systemRoot = resolveWindowsSystemRoot();
      const taskkill = path.join(systemRoot, "System32", "taskkill.exe");
      const killer = spawn(taskkill, ["/PID", String(pid), "/T", "/F"], {
        env: { SystemRoot: systemRoot, windir: systemRoot },
        stdio: "ignore",
        windowsHide: true
      });
      killer.on("error", () => {
        try {
          process.kill(pid, signal);
        } catch {
          // The process may already have exited.
        }
      });
      killer.on("close", (code) => {
        if (code === 0) return;
        try {
          process.kill(pid, signal);
        } catch {
          // The process may already have exited.
        }
      });
      killer.unref();
    } else {
      process.kill(-pid, signal);
    }
  } catch {
    // Process may have exited between timeout and kill.
  }
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
