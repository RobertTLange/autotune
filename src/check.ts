import { access } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { isCommandInterruptedError, runCommand } from "./process.js";
import { FALLBACK_HEADLESS_PACKAGE } from "./headless.js";
import { findExecutableOnPath, isWindowsBatchShim, resolveNpxCommand } from "./npx.js";
import { ensurePythonRuntime, inspectPythonInterpreter } from "./python-runtime.js";
import type { Invocation } from "./types.js";

export interface PrerequisiteReport {
  python: string;
  optuna: string;
  cmaes?: string;
  headless: string;
  runtime: string;
  managedPython: boolean;
  pythonExecutable: string;
}

export interface DoctorCheck {
  name: string;
  status: "ok" | "fail" | "skip";
  detail: string;
}

export async function checkPrerequisites(input: {
  invocation: Invocation;
  agent: string;
  centaur?: boolean;
  skipHeadless?: boolean;
}): Promise<PrerequisiteReport> {
  const pythonRuntime = await ensurePythonRuntime({ includeCmaes: true });
  const headless = input.skipHeadless
    ? "skipped"
    : input.centaur
      ? await checkCentaurHeadless(input.agent)
      : await checkHeadless(input.agent);
  const runtime = await checkRuntime(input.invocation);
  return {
    python: pythonRuntime.pythonVersion,
    optuna: pythonRuntime.optunaVersion,
    ...(pythonRuntime.cmaesVersion ? { cmaes: pythonRuntime.cmaesVersion } : {}),
    headless,
    runtime,
    managedPython: pythonRuntime.managed,
    pythonExecutable: pythonRuntime.python
  };
}

export async function checkDoctorPrerequisites(input: {
  invocation?: Invocation;
  agent: string;
}): Promise<DoctorCheck[]> {
  const checks: DoctorCheck[] = [];
  checks.push(await runDoctorCheck("python3", checkPython));
  checks.push(await runDoctorCheck("optuna", checkOptuna));
  checks.push(await runDoctorCheck("headless", () => checkHeadless(input.agent)));
  checks.push(
    input.invocation
      ? await runDoctorCheck("runtime", () => checkRuntime(input.invocation as Invocation))
      : {
          name: "runtime",
          status: "skip",
          detail: "pass a script to check its runtime"
        }
  );
  return checks;
}

async function runDoctorCheck(name: string, check: () => Promise<string>): Promise<DoctorCheck> {
  try {
    return { name, status: "ok", detail: await check() };
  } catch (error) {
    if (isCommandInterruptedError(error)) throw error;
    return { name, status: "fail", detail: error instanceof Error ? error.message : String(error) };
  }
}

export async function checkPython(): Promise<string> {
  return (await inspectPythonInterpreter()).pythonVersion;
}

export async function checkOptuna(): Promise<string> {
  return (await ensurePythonRuntime({ includeCmaes: true })).optunaVersion;
}

export async function checkHeadless(agent: string): Promise<string> {
  const { label, output } = await runHeadlessCheck();
  if (!headlessListsAgent(output, agent)) {
    return `${label} (agent ${agent} not listed by --check)`;
  }
  return `${label} (${agent})`;
}

async function checkCentaurHeadless(agent: string): Promise<string> {
  const { label, output } = await runHeadlessCheck();
  if (!headlessAgentAvailable(output, agent)) {
    throw new Error(`Centaur proposal agent ${agent} is not available according to headless --check`);
  }
  return `${label} (${agent})`;
}

async function runHeadlessCheck(): Promise<{ label: string; output: string }> {
  const configured = process.env.AUTOTUNE_HEADLESS_BIN;
  if (configured !== undefined && !configured.trim()) {
    throw new Error("configured headless executable must not be empty");
  }
  if (configured !== undefined) {
    const bin = configured;
    const { stdout, stderr } = await runCommand(bin, ["--check"], HEADLESS_CHECK_OPTIONS);
    return { label: bin, output: `${stdout}\n${stderr}` };
  }
  const installed = await findExecutableOnPath("headless");
  if (installed && !isWindowsBatchShim(installed)) {
    const { stdout, stderr } = await runCommand(installed, ["--check"], HEADLESS_CHECK_OPTIONS);
    return { label: installed, output: `${stdout}\n${stderr}` };
  }
  const npx = await resolveNpxCommand();
  const { stdout, stderr } = await runCommand(
    npx.command,
    [...npx.args, "-y", FALLBACK_HEADLESS_PACKAGE, "--check"],
    HEADLESS_CHECK_OPTIONS
  );
  return {
    label: `npx -y ${FALLBACK_HEADLESS_PACKAGE}`,
    output: `${stdout}\n${stderr}`
  };
}

const HEADLESS_CHECK_OPTIONS = {
  timeoutMs: 2 * 60 * 1000,
  maxOutputBytes: 1024 * 1024
};

function headlessAgentAvailable(output: string, agent: string): boolean {
  const normalizedAgent = agent.trim().toLowerCase();
  return output.split(/\r?\n/).some((line) => {
    const columns = line.split("|").slice(1, -1).map((column) => column.trim());
    return columns[0]?.toLowerCase() === normalizedAgent && columns[1] === "✓";
  });
}

function headlessListsAgent(output: string, agent: string): boolean {
  const available = output.toLowerCase().split(/[^a-z0-9_-]+/);
  return available.includes(agent.trim().toLowerCase());
}

export async function checkRuntime(invocation: Invocation): Promise<string> {
  const executable = invocation.command[0];
  if (!executable) {
    throw new Error("empty invocation command");
  }
  if (executable.includes("/") || executable.startsWith(".")) {
    await access(executable, constants.X_OK);
    return executable;
  }
  const resolved = await findOnPath(executable);
  if (!resolved) {
    throw new Error(`runtime not found on PATH: ${executable}`);
  }
  return resolved;
}

async function findOnPath(executable: string): Promise<string | undefined> {
  for (const directory of (process.env.PATH ?? "").split(path.delimiter)) {
    const candidate = path.join(directory, executable);
    try {
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Keep scanning PATH.
    }
  }
  return undefined;
}
