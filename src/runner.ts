import { spawn, spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import process from "node:process";
import { ensurePythonRuntime } from "./python-runtime.js";

export const TARGET_PYTHON_ENV = "AUTOTUNE_TARGET_PYTHON_ENV";
export const NODE_EXECUTABLE_ENV = "AUTOTUNE_NODE_EXECUTABLE";

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
    const child = spawn(python, args, {
      detached: process.platform !== "win32",
      env: supportsTargetEnvironment ? controllerEnvironment(callerEnv) : callerEnv,
      stdio: ["ignore", "pipe", "pipe"]
    });
    let forwardedSignal: NodeJS.Signals | undefined;
    let escalationTimer: NodeJS.Timeout | undefined;
    const forwardSignal = (signal: NodeJS.Signals) => {
      if (forwardedSignal) {
        signalChildTree(child.pid, "SIGKILL");
        return;
      }
      forwardedSignal = signal;
      signalDetachedDescendants(child.pid);
      signalChildTree(child.pid, signal);
      escalationTimer = setTimeout(() => signalChildTree(child.pid, "SIGKILL"), 2_000);
      escalationTimer.unref();
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
    };
    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      process.stderr.write(chunk);
    });
    child.on("error", (error) => {
      cleanupSignals();
      reject(error);
    });
    child.on("close", (code) => {
      cleanupSignals();
      if (forwardedSignal) {
        process.exitCode = 128 + (forwardedSignal === "SIGINT" ? 2 : 15);
        reject(new Error(`python runner interrupted by ${forwardedSignal}`));
        return;
      }
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`python runner exited with ${code}: ${(stderr || stdout).trim()}`));
      }
    });
  });
}

function signalDetachedDescendants(parentPid: number | undefined): void {
  if (!parentPid) {
    return;
  }
  if (process.platform === "win32") {
    spawnSync("C:\\Windows\\System32\\taskkill.exe", ["/PID", String(parentPid), "/T", "/F"], {
      stdio: "ignore"
    });
    return;
  }
  const result = spawnSync("/usr/bin/pgrep", ["-P", String(parentPid)], { encoding: "utf8" });
  for (const value of result.stdout?.split(/\s+/) ?? []) {
    const childPid = Number(value);
    if (!Number.isInteger(childPid) || childPid < 1) {
      continue;
    }
    signalDetachedDescendants(childPid);
    signalChildTree(childPid, "SIGKILL");
    try {
      process.kill(childPid, "SIGKILL");
    } catch {
      // Descendant may already have exited with its process group.
    }
  }
}

function signalChildTree(pid: number | undefined, signal: NodeJS.Signals): void {
  if (!pid) {
    return;
  }
  try {
    process.kill(process.platform === "win32" ? pid : -pid, signal);
  } catch {
    // The controller may have exited between signal delivery and forwarding.
  }
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
