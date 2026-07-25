import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import process from "node:process";
import { ensurePythonRuntime } from "./python-runtime.js";
import {
  captureRunnerIdentity,
  signalChildTree,
  signalDetachedDescendants
} from "./runner-processes.js";
import {
  cleanupOutputCapture,
  closeOutputWriters,
  createOutputCapture,
  drainCapturedOutput,
  MAX_CAPTURE_BYTES,
  signalOutputHolders,
  type OutputCapture
} from "./runner-output.js";

export const TARGET_PYTHON_ENV = "AUTOTUNE_TARGET_PYTHON_ENV";
export const NODE_EXECUTABLE_ENV = "AUTOTUNE_NODE_EXECUTABLE";

const RUNNER_KILL_GRACE_MS = 2_000;
const RUNNER_HARD_STOP_GRACE_MS = 1_000;

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
    const runnerIdentity = captureRunnerIdentity(child.pid);
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
